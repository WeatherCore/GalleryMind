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

# 用于Agent 推理使用的大模型实例
# 延迟创建:仅在下面的 AgentManager类 的 initialize() 时实例化,避免模块导入时连 OpenAI
_agent_llm = None

# 用于describe_image工具的多模特大模型实例（全局，避免每次调用都创建）
# 模块导入时即创建:describe_image 频繁调用(每次看图都触发),提前建好避免首次调用延迟
_vision_llm = ChatOpenAI(
    model=VISION_LLM_MODEL,  
    temperature=0,              # temperature=0:看图描述要稳定可重现,不要发挥
    api_key=VISION_LLM_API_KEY,
    base_url=VISION_LLM_BASE_URL
)

# ============================================================
# Tools 定义 (需要在此重新定义以避免循环导入问题)
# ============================================================
# [补充] 循环导入说明:core/retrieval.py 依赖 retrieval_engine 单例,
# 而 retrieval_engine 又依赖 retrieval.py 模块。工具在 agent.py 重新定义避免互相 import 卡死

# search_images(图像检索工具)：Agent 调用的"找图"动作。
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
        if image_path:
             # 代表图文混合/图搜图，调用image_to_image_search；可以同时带上文本 query 做混合检索
             results = retrieval_engine.image_to_image_search(image_path, query if query else None)
        else:
             # 没有图片路径：纯文本检索，调用text_to_image_search(query)
             results = retrieval_engine.text_to_image_search(query)
        # results 是我们定义的 SearchResult 对象

        # 检索结果为空时，告诉 LLM 没找到,让它决定下一步(改写 query 重试 / 直接告诉用户)
        if not results:
            return json.dumps({"success": False, "message": "未找到相关图片", "results": []})

        # 返回结构化数据供 Agent Manager 使用
        # [补充] success/results 结构是 AgentManager._run_agent_async 解析 SSE results 事件的契约,
        # 改这里必须同步改那边的 json.loads 解析逻辑
        return json.dumps({
            "success": True,
            "message": f"找到 {len(results)} 张相关图片",
            "results": results  # 完整的 SearchResult 数据
        }, ensure_ascii=False)  # ensure_ascii=False:保留中文不转义成\uXXXX，方便大模型阅读中文内容
    
    except Exception as e:
        # 工具异常返回结构化错误(不抛异常,让 LLM 看到失败原因决定下一步)
        return json.dumps({"success": False, "message": f"检索出错: {str(e)}", "results": []})

# describe_image(图片描述工具)：Agent 调用的"看图说话"动作,调用视觉模型对图片内容生成自然语言描述。
# 与 search_images 的区别:search 用 Qwen3-VL embedding 向量匹配找相似图,describe 用多模态视觉模型生成自然语言描述
# 作用：Agent 拿到服务器本地磁盘路径，读取图片，base64 编码，调用 GPT‑4o，返回图片文字描述
@tool
def describe_image(image_path: str) -> str:
    """使用视觉模型描述图片内容（模型由 .env 中 VISION_LLM_MODEL 指定）"""

    # 先判断磁盘上这个图片文件是否存在；如果上层传过来错误路径，直接返回提示字符串
    if not Path(image_path).exists():
        return "❌ 图片不存在"

    try:
        # base64.b64encode()：二进制图片 → base64 字节
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # HumanMessage 多模态消息格式，一条消息里面混合文本 + 图片
        msg = HumanMessage(content=[
            {"type": "text", "text": "请详细描述这张图片的内容，包括主要物体、场景和显著特征。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
        ])

        # 调用视觉模型
        response = _vision_llm.invoke([msg])
        # 把模型返回的图片描述包装成字符串，返回给 Agent
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

    # 类变量：存唯一的那个实例，初始为空
    _instance = None

    # __new__：Python 对象实例化魔术方法，控制类如何创建对象
    # __new__ 在 __init__ 之前执行，负责造出对象。Python 创建对象是两步：先 __new__ 分配内存返回一个空壳，再 __init__ 往里填属性。单例的关键就是在 __new__ 阶段拦截——后续再调用AgentManager()，如果已经有实例了，就不再分配新内存，直接返回已经创建好的_instance，不会新建对象
    def __new__(cls):
        if cls._instance is None:
            # 如果类属性_instance为None → 第一次实例化，真正 new 创建对象
            cls._instance = super(AgentManager, cls).__new__(cls)
            # LangChain Agent 执行器实例，延迟初始化，因为需要等模型加载完成
            cls._instance.agent = None
            # LangGraph 配置，thread_id会话 ID，InMemorySaver靠这个 id 保存对话记忆，默认为 session_default，所有用户共用同一个会话记忆！多用户同时对话会互相串上下文。demo 可以，生产记得每个前端会话分配独立 thread_id
            cls._instance.config = {"configurable": {"thread_id": "session_default"}}
        return cls._instance

    # initialize(初始化 Agent):创建 LLM 实例 + 注册 tools + system_prompt + create_agent。
    # 仅在首次调用时执行(用 self.agent 是否为 None 做幂等),后续调用直接返回
    def initialize(self):

        # 声明使用模块全局变量_agent_llm，避免重复创建
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

        # 两个工具:search_images 找图、describe_image 看图说话。
        # LLM 通过 tool description 决定何时调哪个,所以工具的 docstring 写清楚很关键
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

        # 使用 create_agent (封装 LangGraph)
        self.agent = create_agent(
            model=_agent_llm,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=InMemorySaver()
        )
        print("🤖 Agent initialized.")

    # chat_stream(SSE 流式入口):把 Agent 推理过程包装成 SSE 6 类事件流。
    # 事件顺序:thinking → tool_call → process(50) → results → process(100) → summary → complete
    # 调用方:routers/agent.py 的 chat_sse,通过 StreamingResponse 转发
    async def chat_stream(self, query: str, image_path: str = None) -> AsyncGenerator[str, None]:

        # 如果 Agent 未初始化，执行initialize()
        if not self.agent:
            self.initialize()

        # 把用户问题和图片路径拼成一个 prompt 文本
        content = query
        if image_path:
            # 把图片路径以文本方式拼进 content,LLM 看到后调 describe_image 工具读取该路径
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
            # 先发一个 thinking 事件让前端立刻有反馈(避免用户等待数十秒无响应)
            yield self._format_sse("thinking", {"content": "正在分析您的需求...", "timestamp": _ts()})
            

            # _run_agent_async 是真正跑 Agent 的方法，就在下面，它本身也是一个异步生成器——Agent 每推进一步，它就 yield 一个事件。这里用 async for 逐个接收，然后原样 yield 出去。相当于：Agent 内部产出一个事件 → 这里接住 → 立刻转手推给前端
            async for event in self._run_agent_async(content):
                yield event

            # 收尾事件:告诉前端流结束,关闭加载状态
            yield self._format_sse("complete", {})

        except Exception as e:
            # 异常也走 SSE 事件流(不抛 HTTP 异常),前端体验一致
            yield self._format_sse("thinking", {"content": f"发生错误: {e}", "timestamp": _ts()})
            yield self._format_sse("complete", {})

    # _run_agent_async(遍历 Agent 事件流):把 LangGraph 的同步 stream 事件转成  SSE 事件
    # stream_mode="values" 每步返回完整 state(含 messages 列表),取最后一条消息判断类型分发
    async def _run_agent_async(self, content):
        """Helper to run synchronous agent stream in async way if needed,
           or just iterate if library supports it."""

        # 这里简化处理：假设 agent.stream 是同步的，我们直接遍历
        # 实际生产中应放入 thread pool
        # [补充] 同步 generator 在 async 函数中遍历会阻塞事件循环;
        # 此处简化处理——实际生产应用 asyncio.to_thread 包裹

        # 调用 LangGraph 的 stream 接口，获取事件流
        events = self.agent.stream(
            {"messages": [{"role": "user", "content": content}]},
            config=self.config,
            stream_mode="values"    # stream_mode="values"：LangGraph 输出模式，每一步返回完整状态，包含全部 messages
        )


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

            if "messages" in event:

                # 取 messages 列表最后一条(LangGraph 每步追加新 message,最后一条是本步产物)
                msg = event["messages"][-1]

                # | msg.type | 谁产生的 | 含义 |
                # | --- | --- | --- |
                # | "ai"   | LLM      | LLM 的输出：要么是决定调工具，要么是最终答案 |
                # | "tool" | 工具函数 | 工具执行完毕的返回结果 |
                if msg.type == "ai":
                    # AI 消息分两种:有 tool_calls(决定调工具) / 无 tool_calls(最终答案)
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:

                        # LLM 决定调工具,推送 tool_call 事件让前端渲染工具气泡，取第一个工具调用
                        tool_call = msg.tool_calls[0]
                        yield self._format_sse("tool_call", {
                            "toolName": tool_call['name'],
                            "timestamp": _ts()
                        })

                        # 进度条推进到 50% (工具还没执行，刚决定要调)
                        yield self._format_sse("process", {
                             "stepId": str(tool_call['id'] or '1'),
                             "progress": 50,
                             "status": "processing"
                        })
                    else:
                        # Final Answer (Summary)
                        # 无 tool_calls 说明 LLM 给出最终答案,推送 summary 事件触发前端流式渲染总结文本
                        yield self._format_sse("summary", {
                            "content": msg.content,
                            "done": True
                        })

                elif msg.type == "tool":
                     
                     # 工具执行完了，msg.content 是工具返回的字符串。这里分两种情况：
                     # - search_images 返回的：是 JSON 字符串，含 success + results（图片列表），解析后推 results 事件给前端渲染图片卡片
                     # - describe_image 返回的：是自然语言描述（"这张图片展示了..."），不是 JSON，json.loads 会抛异常，走 except 静默跳过，不推 results 事件
                     # 但是无论哪种情况，最后都推 process 进度 100%，前端进度条走完
                     try:
                         # 尝试json.loads(msg.content)解析我们工具返回的 JSON 字符串
                         tool_result = json.loads(msg.content)
                         # 如果检索成功并且有 results，推送自定义results事件，把SearchResult列表直接发给前端！前端可以直接拿到图片列表渲染图片
                         if tool_result.get("success") and tool_result.get("results"):
                             yield self._format_sse("results", {
                                 "results": tool_result["results"]
                             })
                    
                     except (json.JSONDecodeError, KeyError):
                         # 降级处理：非结构化工具结果
                         # describe_image 返回的是自然语言描述(非 JSON),走 except 静默跳过
                         pass

                     # 工具完成,进度推进到 100%
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
