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

from backend.schemas import AgentChatRequest

# prefix="/api/agent"：这个 router 下面所有接口自动拼接前缀。所以下面@router.post("/chat")最终完整路径：POST /api/agent/chat
router = APIRouter(prefix="/api/agent", tags=["Agent Chat"])

# chat_sse(Agent 流式聊天):接收用户消息与可选图片,启动 SSE 流推 6 类事件(thinking/tool_call/process/results/summary/complete)。
# 用 POST 而非 GET:图片 base64 payload 较大,GET 受 URL 长度限制(2KB 左右),POST 无限制
@router.post("/chat")
async def chat_sse(request: AgentChatRequest):
    """
    SSE Endpoint for Agent Chat (POST to support large image payload)
    """

    # 获取参数
    message = request.message
    image_data = request.image
    # 保存临时图片供 Agent 使用
    image_path = None

    # 处理图片
    # 若有图片则落地为本地 PNG 文件,Agent 的 describe_image 工具会读取本地路径(不直接吃 base64)
    if image_data:
        # try/except 静默降级:图片保存失败不阻塞对话,Agent 仍可纯文本交互(用户体验优先于严格性)
        try:
            # 解析 base64，把图片写入磁盘UPLOAD_DIR，返回 Path 对象
            path = save_base64_image(image_data, UPLOAD_DIR)
            # 把 Path 对象转字符串，得到服务器本地磁盘绝对路径，传给 agent_manager，给大模型多模态接口读取本地图片
            image_path = str(path)
        except Exception:
            pass

    # 调用 agent_manager.chat_stream，返回 StreamingResponse，把 SSE 流式事件流转发给前端
    return StreamingResponse(
        agent_manager.chat_stream(query=message or "你好", image_path=image_path),
        media_type="text/event-stream"  # 告诉浏览器，这是 SSE 流式事件流，长连接，服务器持续推送数据
    )
