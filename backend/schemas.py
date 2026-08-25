# schemas.py(前后端契约层):所有 HTTP 请求/响应的 Pydantic 模型定义。
# 解决两个核心问题:
#  1. 类型校验:FastAPI 自动校验请求字段类型与必填项,错误请求 422 早退,不进业务逻辑
#  2. 文档生成:OpenAPI schema 自动生成,前端可直接基于此对接
# 调用方:routers/search.py(SearchRequest/SearchResponse)、routers/agent.py(AgentChatRequest)

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# --- Search Related ---

# SearchRequest:/api/search 的检索请求体：文本搜图/图搜图入参
# searchMode 用中文字符串枚举('文搜图'/'图搜图'/'混合搜索'),后端 retrieval_engine.search() 直接 if/elif 分发,
# 简单直接但扩展性一般(若加新模式需改后端 if 链)
class SearchRequest(BaseModel):
    textQuery: str = ""                                  # 文本查询(文搜图/混合搜时必填)
    uploadedImage: Optional[str] = None # Base64 string # 图搜图/混合搜时必填,前端上传的 base64 图片
    recallTopK: int = 20                                 # 召回阶段数量(快速捞一大批候选)
    rerankTopK: int = 5                                  # 精排阶段数量(慢但准挑少数)
    threshold: float = 0.0                               # 相似度阈值,低于此分数的结果被过滤(0-1)
    searchMode: str # '文搜图' | '图搜图' | '混合搜索'  # 检索模式,决定走哪条索引路径

# ProcessStep:检索过程处理步骤(给前端 ProcessVisualization 渲染进度条用)。
# 注意:后端目前 processSteps 始终返回空数组,真实进度由前端 processSimulator.ts 模拟
class ProcessStep(BaseModel):
    id: str                                              # 步骤 ID('1'/'2'/'3')
    name: str                                            # 步骤显示名，例如"向量召回"、"重排序"
    status: str # 'pending' | 'processing' | 'completed' # 状态机三态
    progress: int                                        # 进度百分比 0-100
    duration: Optional[str] = None                       # 步骤耗时(完成后填充,如"0.8s")

# SearchResult:单条检索结果对象。score 与 relevanceScore 表达同一信息但单位不同,
# 是历史决策——score 0-100 给前端展示直观,relevanceScore 0-1 给程序使用
class SearchResult(BaseModel):
    id: str                                              # 结果序号(从 1 开始)
    imageUrl: str                                        # 图片访问 URL(形如 /static/images/xxx.png)
    title: str                                           # 标题(默认用文件名)
    description: Optional[str] = None                    # 图片描述(若有 caption 缓存则填充)
    score: float # 0-100                                 # 展示分数(relevanceScore*100,前端人类可读)
    relevanceScore: float # 0-1                         # 原始相似度(Milvus/Reranker 返回值,程序用)
    rerankScore: Optional[float] = None                  # Reranker 重排分数(可选,目前未单独填充)
    metadata: Optional[Dict[str, Any]] = None           # 元数据(file_path/file_name 等)

# SearchResponse:/api/search 的检索接口返回响应对象
class SearchResponse(BaseModel):
    results: List[SearchResult]                          # 检索结果列表
    processSteps: Optional[List[ProcessStep]] = None     # 过程步骤(目前始终为 [])
    totalTime: float                                     # 总耗时(秒,用于前端展示)

# --- Agent Related ---

# Milvus初始化请求，Agent 接口实际使用 SSE 流式输出，不直接返回 JSON
class InitRequest(BaseModel):
    milvus_uri: str = "http://localhost:19530"           # 预留:动态初始化 Milvus URI(目前未启用)

# 初始化接口返回对象
class InitResponse(BaseModel):
    status: str                                          # 初始化状态("ok"/"error")
    message: str                                         # 提示信息

# AgentChatRequest:/api/agent/chat 的请求体，Agent对话接口入参
class AgentChatRequest(BaseModel):
    message: str                                         # 用户消息文本
    image: Optional[str] = None # Base64 string          # 可选图片(base64),Agent 调 describe_image 时用
