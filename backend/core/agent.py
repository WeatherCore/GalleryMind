# agent.py(Agent 编排核心):项目的"自然语言交互外壳",把 LangGraph Agent 包装成 SSE 流式服务。
# 解决三个核心问题:
#  1. 工具调度:LLM 自主决定调 search_images(找图)还是 describe_image(看图说话),循环推理直到给出最终答案
#  2. 流式推送:Agent 推理过程拆解为 6 类 SSE 事件(thinking/tool_call/process/results/summary/complete)实时推前端
#  3. 单例管理:AgentManager 全局唯一,避免重复创建 ChatOpenAI 连接池与 LangGraph 实例
# 调用方:routers/agent.py 的 chat_sse 接口;启动期 main.py 的 lifespan 调 initialize()

import json
import asyncio
import base64
from pathlib import Path
from typing import AsyncGenerator
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from backend.config import (
    AGENT_LLM_MODEL, AGENT_LLM_API_KEY, AGENT_LLM_BASE_URL,
    VISION_LLM_MODEL, VISION_LLM_API_KEY, VISION_LLM_BASE_URL
)
from backend.core.retrieval import retrieval_engine

# ============================================================
# 全局LLM实例（避免重复创建）
# ============================================================
# 设计意图:ChatOpenAI 构造含连接池开销,且多模态模型调用按 token 计费;
# 全局化复用同一实例,避免每次请求重建

# Agent 决策大模型，延迟实例化
# 为什么延迟：项目启动时不加载，第一次聊天请求到来时才初始化，节省启动开销；mock 模式下甚至完全不用创建
_agent_llm = None

# 视觉LLM：用来做图片描述，文件加载时就直接实例化
_vision_llm = ChatOpenAI(
    model=VISION_LLM_MODEL,  
    temperature=0,                # temperature=0:看图描述要稳定可重现,不要发挥
    api_key=VISION_LLM_API_KEY,
    base_url=VISION_LLM_BASE_URL
)

# ============================================================
# Tools 定义 (需要在此重新定义以避免循环导入问题)
# ============================================================
# [补充] 循环导入说明:core/retrieval.py 依赖 retrieval_engine 单例,
# 而 retrieval_engine 又依赖 retrieval.py 模块。工具在 agent.py 重新定义避免互相 import 卡死

# search_images(图像检索工具)：Agent 调用的多模态图像搜索工具
# 根据 image_path 是否存在决定走哪条检索路径:
#   有图 → image_to_image_search(图搜图或图文混合搜)
#   无图 → text_to_image_search(文搜图)
# 返回 JSON 字符串(给 LLM 看的,不是给前端的;前端通过 SSE results 事件收结果)
@tool
def search_images(query: str, image_path: str = None) -> str:
    """
    多模态图像搜索工具。
    Args:
        query: 文本查询描述
        image_path: 图片路径 (本地绝对路径)
    """

    try:
        # 如果传入图片路径 → 走【图搜图】，调用image_to_image_search；可以同时带上文本 query 做混合检索
        if image_path:
             results = retrieval_engine.image_to_image_search(image_path, query if query else None)
        # 没有图片，只有文字 → 走【文搜图】，调用text_to_image_search(query)
        else:
             results = retrieval_engine.text_to_image_search(query)

        # 检索结果为空时，告诉 LLM 没找到，让它决定下一步(改写 query 重试 / 直接告诉用户)
        if not results:
            return json.dumps({"success": False, "message": "未找到相关图片", "results": []})

        # 检索成功，返回json字符串供 Agent Manager 使用
        # [补充] success/results 结构是 AgentManager._run_agent_async 解析 SSE results 事件的契约,
        # 改这里必须同步改那边的 json.loads 解析逻辑
        return json.dumps({
            "success": True,                                # bool 标记检索成功与否
            "message": f"找到 {len(results)} 张相关图片",   # 一句话描述结果，给 LLM 阅读
            "results": results                  # 检索出来的图片列表（图片 url、score 等，传给 Agent 后续整理回答
        }, ensure_ascii=False)
        # ensure_ascii=False:保留中文不转义成\uXXXX，方便大模型阅读中文内容

    # 只要检索代码抛任何异常（Milvus 连接失败、模型没初始化、图片路径错误等），捕获异常，返回带错误信息的 json 字符串
    except Exception as e:
        return json.dumps({"success": False, "message": f"检索出错: {str(e)}", "results": []})

# describe_image(图片描述工具)：Agent 调用的"看图说话"动作，调用视觉模型对图片内容生成自然语言描述。
# 与 search_images 的区别:search 用 Qwen3-VL embedding 向量匹配找相似图，describe 用多模态视觉模型生成自然语言描述
# 作用：Agent 拿到服务器本地磁盘路径，读取图片，base64 编码，调用 GPT‑4o，返回图片文字描述
@tool
def describe_image(image_path: str) -> str:
    """使用视觉模型描述图片内容（模型由 .env 中 VISION_LLM_MODEL 指定）"""

    # 先判断磁盘上这个图片文件是否存在；如果上层传过来错误路径，直接返回提示字符串
    if not Path(image_path).exists():
        return "❌ 图片不存在"

    try:
        # base64.b64encode()：二进制图片 → base64 字节，方便塞进消息传给多模态LLM
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # HumanMessage：langchain_core 的人类消息结构体，支持多模态混合内容（文字+图片）
        msg = HumanMessage(content=[
            {"type": "text", "text": "请详细描述这张图片的内容，包括主要物体、场景和显著特征。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
        ])

        # 调用视觉大模型，传入构造好的消息，拿到图片描述结果
        response = _vision_llm.invoke([msg])

        # 拼接结果返回给Agent，带上图片文件名 + 模型输出的描述文本
        return f"📸 图片描述 ({Path(image_path).name}):\n{response.content}"

    except Exception as e:
        return f"❌ 描述失败: {str(e)}"


# ============================================================
# AgentManager：管理 Agent 实例、初始化、对外暴露 chat_stream() SSE 流式接口
# 单例实现:__new__ 拦截类创建,首次创建存 _instance,后续直接返回;
# initialized 标志位双重保险,防止 agent 已创建但未完成初始化时被重复调
# 整体链路：router/agent.py → agent_manager.chat_stream() → 执行 Agent，产出 SSE 事件字符串 → StreamingResponse 推送给前端
# ============================================================
class AgentManager:

    # 单例标记：保存唯一实例，静态变量，属于类本身，不是实例
    _instance = None

    # __new__：Python 对象实例化魔术方法，控制类如何创建对象
    # __new__ 在 __init__ 之前执行，负责造出对象。Python 创建对象是两步：先 __new__ 分配内存返回一个空壳，再 __init__ 往里填属性。单例的关键就是在 __new__ 阶段拦截——后续再调用AgentManager()，如果已经有实例了，就不再分配新内存，直接返回已经创建好的_instance，不会新建对象
    def __new__(cls):

        # 判断：还没有创建过实例
        if cls._instance is None:
            # 调用父类object的__new__，真正创建空对象
            cls._instance = super(AgentManager, cls).__new__(cls)
            # 给这个唯一实例初始化成员变量
            cls._instance.agent = None
            # LangGraph 配置，thread_id会话 ID，InMemorySaver靠这个 id 保存对话记忆，默认为 session_default，所有用户共用同一个会话记忆！多用户同时对话会互相串上下文。demo 可以，生产记得每个前端会话分配独立 thread_id
            cls._instance.config = {"configurable": {"thread_id": "session_default"}}

        # 无论多少次 new，永远返回同一个实例对象
        return cls._instance

    # initialize(初始化 Agent):创建 LLM 实例 + 注册 tools + system_prompt + create_agent。
    # 仅在首次调用时执行(用 self.agent 是否为 None 做幂等),后续调用直接返回
    def initialize(self):

        # 声明：使用全局变量 _agent_llm，不是函数内部局部变量
        global _agent_llm

        # 幂等保护，已经初始化过就直接返回，避免重复创建 Agent
        if self.agent:
            return

        # 如果全局_agent_llm为 None，实例化 Agent 所用的大模型
        if _agent_llm is None:
            _agent_llm = ChatOpenAI(
                model=AGENT_LLM_MODEL,
                temperature=0,
                api_key=AGENT_LLM_API_KEY,
                base_url=AGENT_LLM_BASE_URL
            )

        # 把上面两个 @tool 工具注册给Agent
        tools = [search_images, describe_image]

        # system_prompt(系统提示词):给 LLM 立"人设"+工具选择策略。
        # 关键约束:"不要直接返回图片路径列表,而是用自然语言描述"——避免 LLM 吐一堆 JSON 路径给用户
        system_prompt = """你是一个多模态检索助手。
        工具策略:
        1. 找图 -> search_images
        2. 看图 -> describe_image

        请用中文回答。
        如果找到了图片，请在最终回复中简要总结图片内容。
        不要直接返回图片路径列表，而是用自然语言描述。
        """

        # 使用 create_agent (创建 LangGraph Agent)
        self.agent = create_agent(
            model=_agent_llm,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=InMemorySaver()
        )
        print("🤖 Agent initialized.")

    # chat_stream(SSE 流式入口):把 Agent 推理过程包装成 SSE 6 类事件流。
    # 事件顺序:thinking → tool_call → process(50) → results → process(100) → summary → complete
    # session_id:LangGraph thread_id,决定多轮记忆挂在哪个会话;不传落回默认会话。
    # 前端"新对话"= 换新 sessionId,旧记忆自然隔离(InMemorySaver 按 thread_id 隔离)
    # 调用方:routers/agent.py 的 chat_sse,通过 StreamingResponse 转发
    async def chat_stream(self, query: str, image_path: str = None, session_id: str = None) -> AsyncGenerator[str, None]:

        # 如果agent还没初始化，执行初始化（懒加载）
        if not self.agent:
            self.initialize()

        # 组装会话配置：thread_id 就是会话ID，用来隔离不同用户的聊天记忆
        # 如果没传session_id，默认使用session_default
        thread_config = {"configurable": {"thread_id": session_id or "session_default"}}

        # 如果用户上传了图片，把图片路径以文本方式拼进 content,LLM 看到后调 describe_image 工具读取该路径
        content = query
        if image_path:
            content += f"\n[参考图片路径: {image_path}]"

        # 关于 yield（普通生成器）
        # yield 的作用：暂停函数，把值 “吐出去”，保存函数当前状态，下次调用从暂停处继续往下执行
        # 普通return是直接结束函数；yield只是暂停，函数没有结束

        # def count():
        #     yield 1             # 吐出 1，暂停在这里
        #     yield 2             # 上次暂停后继续，吐出 2，再暂停
        #     yield 3             # 继续，吐出 3，暂停
        #
        # g = count()
        # print(next(g))          # 1，函数执行到第一个yield暂停
        # print(next(g))          # 2，从上次暂停继续跑
        # print(next(g))          # 3
        # 
        # for n in count():
        #     print(n)            # 1（此时函数暂停在第一个 yield）
        #                         # 2（此时函数暂停在第二个 yield）
        #                         # 3（此时函数暂停在第三个 yield）

        # 为了对接前端 SSE，我们需要发送特定格式的 Event:
        # thinking, tool_call, process, tool_result, summary, complete

        try:

            # yield：SSE推送第一条事件，告诉前端：正在思考
            yield self._format_sse("thinking", {"content": "正在分析您的需求...", "timestamp": _ts()})
            
            # 循环读取Agent内部产生的事件（工具调用、结果、AI回答），逐个推送给前端
            async for event in self._run_agent_async(content, thread_config):
                yield event

            # 全部处理完毕，推送complete事件，前端知道对话结束
            yield self._format_sse("complete", {})

        # 捕获异常，向前端推送错误信息，再发送complete标记结束流
        except Exception as e:
            yield self._format_sse("thinking", {"content": f"发生错误: {e}", "timestamp": _ts()})
            yield self._format_sse("complete", {})

    # reset_session(重置会话记忆):清除某一个session_id对应的对话记忆
    # InMemorySaver 提供 delete_thread;若当前 langgraph 版本没有该方法,
    # 优雅降级返回 supported=False——前端本就会换新 sessionId,旧记忆读不到,不影响使用
    def reset_session(self, session_id: str = "session_default") -> dict:

        # Agent都没初始化，自然没有会话历史，直接返回
        if not self.agent:
            return {"status": "ok", "supported": True, "message": "Agent 尚未初始化,无历史可清"}

        # 取出agent的记忆存储对象 checkpointer
        checkpointer = getattr(self.agent, "checkpointer", None)

        # 获取删除会话的方法；getattr安全取值，不存在不会直接报错
        delete_thread = getattr(checkpointer, "delete_thread", None)

        # 如果当前checkpointer不支持删除会话（InMemorySaver早期版本就会出现这个情况）
        if delete_thread is None:
            return {"status": "ok", "supported": False, "message": "当前 checkpointer 不支持删除,已通过新会话 ID 隔离"}

        try:
            # 调用方法，删除该thread_id对应的历史对话
            delete_thread(session_id)
            return {"status": "ok", "supported": True, "message": f"会话 {session_id} 已重置"}
        except Exception as e:
            # 删除失败时返回提示：最简单方案——前端直接换全新session_id，间接隔离历史
            return {"status": "ok", "supported": False, "message": f"重置异常({e}),建议直接使用新会话 ID"}

    # _run_agent_async(遍历 Agent 事件流):把 LangGraph 的同步 stream 事件转成  SSE 事件
    # stream_mode="values" 每步返回完整 state(含 messages 列表),取最后一条消息判断类型分发
    async def _run_agent_async(self, content, thread_config: dict = None):
        """Helper to run synchronous agent stream in async way if needed,
           or just iterate if library supports it."""

        # 这里简化处理：假设 agent.stream 是同步的，我们直接遍历
        # 实际生产中应放入 thread pool
        # [补充] 同步 generator 在 async 函数中遍历会阻塞事件循环;
        # 此处简化处理——实际生产应用 asyncio.to_thread 包裹

        # 调用LangGraph agent.stream，开启agent的事件流，stream_mode="values"表示拿到每一步状态
        events = self.agent.stream(
            {"messages": [{"role": "user", "content": content}]},
            config=thread_config or self.config,
            stream_mode="values"    # stream_mode="values"：LangGraph 输出模式，每一步返回完整状态，包含全部 messages
        )

        # 遍历agent产生的每一步事件（这是同步迭代器）
        for event in events:

            # 每个 event 是"状态快照"，里面有个 messages 键，值是到目前为止的所有消息列表，
            # 但不是每一个 event 里面都一定有 "messages" 这个 key。
            # # ✅有 messages，我们要处理
            # {
            #     "messages": [HumanMessage, AIMessage(有tool_calls)],
            #     "thread_id": "xxx"
            # }
            # 
            # # ❌没有 messages，仅仅是图状态更新、中间回调、元数据事件
            # {
            #     "metadata": {"step": 1},
            #     "queue": []
            # }

            # 事件里包含messages消息列表才处理
            if "messages" in event:

                # 取 messages 列表最新一条消息(LangGraph 每步追加新 message,最后一条是本步产物)
                msg = event["messages"][-1]

                # | msg.type | 谁产生的 | 含义 |
                # |   ---    |    ---   |  --- |
                # | "ai"     | LLM      | LLM 的输出：要么是决定调工具，要么是最终答案 |
                # | "tool"   | 工具函数 | 工具执行完毕的返回结果 |
                if msg.type == "ai":
                    # AI 消息分两种:有 tool_calls(决定调工具) / 无 tool_calls(最终答案)
                    # 如果AI消息里面携带tool_calls，代表Agent决定调用工具
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:

                        # LLM 决定调工具,推送 tool_call SSE事件让前端渲染工具气泡，即将调用某个工具
                        tool_call = msg.tool_calls[0]
                        yield self._format_sse("tool_call", {
                            "toolName": tool_call['name'],
                            "timestamp": _ts()
                        })

                        # SSE事件：前端展示进度，进度条推进到 50% (工具还没执行，刚决定要调)
                        yield self._format_sse("process", {
                             "stepId": str(tool_call['id'] or '1'),
                             "progress": 50,
                             "status": "processing"
                        })

                    # AI不调用工具，直接输出最终回答summary
                    else:
                        # 无 tool_calls 说明 LLM 给出最终答案,推送 summary 事件触发前端流式渲染总结文本
                        yield self._format_sse("summary", {
                            "content": msg.content,
                            "done": True
                        })

                # 消息类型是tool：工具执行完成，拿到工具返回结果
                elif msg.type == "tool":
                     
                     # 工具执行完了，msg.content 是工具返回的字符串。这里分两种情况：
                     # - search_images 返回的：是 JSON 字符串，含 success + results（图片列表），解析后推 results 事件给前端渲染图片卡片
                     # - describe_image 返回的：是自然语言描述（"这张图片展示了..."），不是 JSON，json.loads 会抛异常，走 except 静默跳过，不推 results 事件
                     # 但是无论哪种情况，最后都推 process 进度 100%，前端进度条走完
                     try:
                         # 尝试 json.loads(msg.content) 解析我们工具返回的 JSON 字符串（search_images返回的就是json）
                         tool_result = json.loads(msg.content)
                         # 如果检索成功，并且有图片结果，单独推results事件给前端渲染图片
                         if tool_result.get("success") and tool_result.get("results"):
                             yield self._format_sse("results", {
                                 "results": tool_result["results"]
                             })

                    # 容错：describe_image返回普通文本，不是json，会解析失败，直接忽略
                     except (json.JSONDecodeError, KeyError):
                         # 降级处理：非结构化工具结果
                         # describe_image 返回的是自然语言描述(非 JSON),走 except 静默跳过
                         pass

                     # 推送事件：当前工具步骤执行完毕，进度100
                     yield self._format_sse("process", {
                             "stepId": msg.tool_call_id,
                             "progress": 100,
                             "status": "completed"
                     })

            # 非阻塞地睡 10 毫秒。跟 Java 的 Thread.sleep(10) 很像，但区别是：睡的时候不占着 CPU，把控制权还给事件循环，让服务器能同时处理其他请求
            # 为什么睡？让出节奏——每发完一个事件稍微歇一下，前端 UI 能正常刷新渲染，不会因为事件刷太快而卡顿
            await asyncio.sleep(0.01)

    # _format_sse(SSE 格式化)：把事件拼装标准 SSE 事件字符串
    def _format_sse(self, event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        # event: thinking
        # data: {"content": "正在分析您的需求...", "timestamp": "1234567890.123"}
        # 浏览器通过 EventSource 或 fetch 读这个流时，每收到一个这样的块就触发一次回调，前端就能逐个渲染

# 时间戳工具(模块末尾 import time 是反模式,但保持原样不重构)
import time

# 简单工具函数，返回 Unix 时间戳字符串，给 SSE 事件带上时间
def _ts(): 
    return str(time.time())

# 模块级单例:routers/agent.py 与 main.py 都引用这个单例,不重复创建
agent_manager = AgentManager()
