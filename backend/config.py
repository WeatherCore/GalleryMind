# config.py(配置中枢):项目所有可调参数的中枢。它解决三个核心问题:
#  1. 凭证安全:OpenAI Key、Milvus URI 等敏感信息从 .env 加载,不硬编码进源码
#  2. 路径布局:统一管理图片库、上传目录、模型缓存、caption 缓存的路径
#  3. 启动可用性:启动时自动创建必要目录,避免运行时 FileNotFoundError
# 调用方:main.py 启动时导入、core/retrieval.py 与 core/agent.py 运行时读取

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env(override=False 表示已存在的环境变量优先, .env 只补缺:
# 这样 USE_MOCK_DATA=true python -m backend.main 这类命令行临时开关才能生效,
# 否则 .env 里写死的 USE_MOCK_DATA=false 会反过来覆盖命令行传的 true)
load_dotenv(override=False)

# 基础路径(BASE_DIR = backend/，所有相对路径都从这里派生，避免 working directory 依赖)
# 1. `Path(__file__)`：拿到当前`config.py`文件的路径对象
# 2. `.resolve()`：解析成绝对路径，自动处理相对路径、软链接，消除`../`这类符号，得到完整真实路径
# 3. `.parent`：取【文件所在的文件夹】
BASE_DIR = Path(__file__).resolve().parent 

# 再往上跳一级，得到项目根目录 GalleryMind/
PROJECT_ROOT = BASE_DIR.parent

# backend/data，这个文件夹是项目所有本地数据的总仓库：上传图片、缓存、向量库文件都放这里
DATA_DIR = BASE_DIR / "data"  
# backend/data/uploads，`save_base64_image`工具函数保存用户上传图片的目录
UPLOAD_DIR = DATA_DIR / "uploads"
# 模型缓存目录，Qwen3-VL-Embedding、Reranker 权重下载到这里（优先读 .env 的 MODELSCOPE_CACHE，未设置则用默认目录）
MODEL_CACHE_DIR = Path(os.getenv("MODELSCOPE_CACHE", str(DATA_DIR / "models")))

# 图片描述文本缓存目录
# 多模态 RAG 在预处理图库时，会调用 VL 模型生成图片 caption，把结果缓存到这里，避免重复调用大模型，节省算力
CAPTION_CACHE_DIR = DATA_DIR / "caption_cache"

# 创建必要目录(parents+exist_ok 保证幂等:目录已存在不报错,父目录不存在自动建)
# 业务场景：程序启动时读取 config 配置的时候，自动把这三个业务目录全部准备好，不然第一次跑项目，文件夹还没建立。后面代码往UPLOAD_DIR保存用户上传图片，就会报文件夹不存在的错误
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CAPTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 这是项目原始图库目录，也就是提前入库、做向量检索的图片库，区分于用户实时上传的 uploads 临时图片
# 默认指向项目内 backend/data/images，但是也可以通过环境变量覆盖，指向其他目录
DEFAULT_IMAGE_DIR = Path(os.getenv("IMAGE_DIR", str(DATA_DIR / "images")))
if not DEFAULT_IMAGE_DIR.exists():
    DEFAULT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# 从.env读取密钥、接口地址等配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")

# Agent 推理模型配置（AgentManager 初始化时创建，负责工具调度与最终回答）
# 未设置时回退到 DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL
AGENT_LLM_MODEL = os.getenv("AGENT_LLM_MODEL", "qwen3.5-omni-flash")
AGENT_LLM_API_KEY = os.getenv("AGENT_LLM_API_KEY", DASHSCOPE_API_KEY)
AGENT_LLM_BASE_URL = os.getenv("AGENT_LLM_BASE_URL", DASHSCOPE_BASE_URL)

# Vision 视觉模型配置（describe_image 工具用，需要支持多模态图片输入）
# 未设置时回退到 DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL
VISION_LLM_MODEL = os.getenv("VISION_LLM_MODEL", "qwen3.5-omni-plus")
VISION_LLM_API_KEY = os.getenv("VISION_LLM_API_KEY", DASHSCOPE_API_KEY)
VISION_LLM_BASE_URL = os.getenv("VISION_LLM_BASE_URL", DASHSCOPE_BASE_URL)

# Milvus URI:Docker Compose 默认暴露 19530 端口
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")

# Model IDs(ModelScope 上的 Qwen3-VL 模型 ID;首次启动会下载到 MODEL_CACHE_DIR)
# EMBEDDING:把文本/图片编码为 512 维向量,Milvus 索引的"指纹"
# RERANKER:对召回的候选图重新打分,挑出真正最像的几张(精排阶段)
EMBEDDING_MODEL_MS = "qwen/Qwen3-VL-Embedding-2B"
RERANKER_MODEL_MS = "qwen/Qwen3-VL-Reranker-2B"

# FastAPI 后端服务配置：监听地址、端口、跨域允许所有来源访问
HOST = "0.0.0.0"           # 允许容器内访问
PORT = 3001                #  与前端 vite proxy 对齐
ALLOWED_ORIGINS = ["*"]    # ALLOWED_ORIGINS=["*"] 开发友好,生产应收紧到具体域名
