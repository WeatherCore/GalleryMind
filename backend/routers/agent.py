# agent.py(Agent 聊天 SSE 接口):前端 Agent 模式的 HTTP 入口。
# 解决两个核心问题:
#  1. 接收消息+图片:从 AgentChatRequest 取出 message 与 base64 image,图片落地为本地文件供 Agent 工具调用
#  2. SSE 流式封装:把 agent_manager.chat_stream 异步生成器包成 StreamingResponse 推给前端
# 调用方:前端 AgentChat.tsx 的 fetch POST /api/agent/chat

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.core.agent import agent_manager
from backend.core.utils import save_base64_image
from backend.config import UPLOAD_DIR

from backend.schemas import AgentChatRequest, SessionResetRequest

# 创建独立路由实例，相当于一个子路由容器
# prefix="/api/agent"：这个 router 下面所有接口自动拼接前缀。所以下面@router.post("/chat")最终完整路径：POST /api/agent/chat
router = APIRouter(prefix="/api/agent", tags=["Agent Chat"])

# chat_sse(Agent 流式聊天 /api/agent/chat)：
# 接收用户消息与可选图片,启动 SSE 流推 6 类事件(thinking/tool_call/process/results/summary/complete)。
# 用 POST 而非 GET：图片 base64 payload 较大，GET 受 URL 长度限制(2KB 左右),POST 无限制
@router.post("/chat")
async def chat_sse(request: AgentChatRequest):
    """
    SSE Endpoint for Agent Chat (POST to support large image payload)
    """

    # 提取请求里三个核心参数：
    # - message：用户文本提问
    # - image_data：前端传的图片的 base64 字符串，可以为空（纯文本提问）
    # - session_id：会话 ID！用来区分不同用户 / 不同对话窗口，LangGraph 用这个 id 隔离对话记忆
    message = request.message
    image_data = request.image
    session_id = request.sessionId 
    
    # `image_path` 初始化 None，代表没有上传图片
    image_path = None

    # 如果前端传了图片 base64，尝试保存图片
    if image_data:
        # try/except 静默降级:
        # 兜底：图片解析失败、base64 损坏、磁盘满等异常，直接忽略，image_path保持 None，不会直接中断整个对话
        # 效果：图片上传失败，Agent 依旧可以处理文本提问，只是没有图片输入
        try:
            # 调用`save_base64_image`，把 base64 图片保存到 UPLOAD_DIR，得到本地 Path
            path = save_base64_image(image_data, UPLOAD_DIR)
            # 把 Path 对象转字符串，得到服务器本地磁盘绝对路径，传给 agent_manager，给大模型多模态接口读取本地图片
            image_path = str(path)
        except Exception:
            pass

    # `StreamingResponse`接收这个异步生成器，把每一段 yield 出来的数据包装成 SSE 事件，持续推送到前端
    return StreamingResponse(
        # 异步生成器 async generator（带 yield） 就是我们前面聊yield的那段代码！这个函数内部循环 yield 一段一段的文本 / 事件
        agent_manager.chat_stream(query=message or "你好", image_path=image_path, session_id=session_id),
        media_type="text/event-stream"  # 告诉浏览器，这是 SSE 流式事件流，长连接，服务器持续推送数据
    )


# reset_session(重置会话 /api/agent/session/reset):清空指定 sessionId 的 Agent 对话记忆。
# 配合前端"新对话"按钮:换 sessionId 已能隔离旧上下文,此接口负责真正释放旧记忆
@router.post("/session/reset")
def reset_session(req: SessionResetRequest):
    # 调用 agent_manager 的 reset_session 方法，清空这个会话对应的记忆（LangGraph 的 checkpoint）
    return agent_manager.reset_session(req.sessionId)
