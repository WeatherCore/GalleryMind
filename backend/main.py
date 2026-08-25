# main.py(FastAPI 入口):整个 FastAPI 服务启动、资源初始化、路由注册、静态文件、跨域全部写在这里
# 解决三个核心问题:
#  1. 启动编排:通过 lifespan 在应用启动时预热检索引擎与 Agent,避免首请求延迟数十秒
#  2. 静态服务:挂载 /static 让前端能 HTTP 访问 backend/data 下的图片(检索结果图片 URL 指向这里)
#  3. 路由聚合:注册 health/search/agent 三个路由模块,统一加 CORS 中间件
# 调用方:python -m backend.main 或 python main.py 启动;uvicorn 部署时引用 app 对象
import os
import sys
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 添加项目根目录到 sys.path，使得可以在 backend 目录下直接运行，这样既支持 python main.py 也支持 python -m backend.main
# 这段只在 __main__ 块内执行,避免被 import 时副作用
if __name__ == "__main__":
    # main.py 所在的文件夹，也就是 backend 文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # backend 的上一级，整个项目的根文件夹，也就是 GalleryMind/
    parent_dir = os.path.dirname(current_dir)
    # 手动把「项目根目录」塞进 Python 的模块搜索路径列表第 0 位，优先级最高，优先在这里找包
    # 让32行和33行的from backend.config import ... 这种导入能找到 backend 包，这个他要求 sys.path 含项目根目录(GalleryMind/),否则 Python 找不到 backend 包
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
# 启动方式	                            sys.path 里有没有根目录	              结果
# python main.py（在 backend 目录下）	  ❌ 只有 backend/	            需要这段代码补路
# python -m backend.main（在根目录下）	  ✅ 天然包含根目录	          这段代码只是检查后跳过

# Java 的导入靠的是 classpath（类路径），跟 Python 的 sys.path 是同一个东西，不过Maven/Gradle 会自动把依赖和项目路径配进 classpath
# 如果强制只允许在根目录下执行 python -m backend.main 来启动，那段 sys.path 代码确实可以删。python -m 会把当前工作目录（而不是脚本所在目录）自动加进 sys.path——只要用户在项目根目录 GalleryMind/ 下执行，from backend.config import ... 天然能找到，那段 if 检查会发现根目录已在列表里、直接跳过

# 导入配置
from backend.config import DATA_DIR, UPLOAD_DIR, HOST, PORT, ALLOWED_ORIGINS
from backend.routers import search, agent, health


# lifespan（应用生命周期【重中之重】）是 FastAPI 的新写法，替代了旧版教程里的 @app.on_event("startup") / @app.on_event("shutdown")
# 启动时:自动预热模型(Qwen3-VL 模型加载耗时数十秒,必须启动期完成,不能等首请求)
# 关闭时:自动清理资源，目前仅打印日志，实际可在此释放模型显存、关闭连接池
# 调用方:FastAPI 框架在 app 启动/关闭时自动调用
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ──────── yield 之前：应用【启动】时执行 ────────
    print(f"🚀 Server starting on {HOST}:{PORT}")
    print(f"📂 Static files mounted at: {DATA_DIR} -> /static")

    # 这里用局部 import 而非顶部 import,是为了延迟加载(只在真正启动时才导入 torch/transformers 等重依赖)
    print("⏳ 初始化检索引擎...")
    from .core.retrieval import retrieval_engine
    from .core.agent import agent_manager
    from .config import DEFAULT_IMAGE_DIR
    # 检索引擎初始化，加载 embedding 模型、连接 Milvus 向量库、加载图片素材
    retrieval_engine.initialize(DEFAULT_IMAGE_DIR)
    # Agent 管理器初始化，初始化大模型客户端
    agent_manager.initialize()
    print("✅ 检索引擎和Agent就绪")
    # 如果初始化这里抛出异常，整个 FastAPI 服务直接启动失败，不会对外提供服务，符合业务预期：向量库连不上，就不要跑接口

    yield   # ← 分界线：到这里服务开始"营业"，yield 之前的代码仅在服务启动时执行一次，之后 = 关闭时跑
    # 不用学上下文管理器的细节，只要记住"yield 把启动逻辑和关闭逻辑拆在上下两段"就够了

    # ──────── yield 之后：应用【关闭】时执行 ────────
    print("👋 Server shutting down")

# 旧版长这样，见过就懂了：
#  @app.on_event("startup")
#  async def startup():          # 启动时执行
#      ...
#
#  @app.on_event("shutdown")
#  async def shutdown():         # 关闭时执行
#      ...
# 两个函数被合并成一个 lifespan，用 yield 分界——这就是你"没见过"的原因，写法变了而已


# 初始化 App，绑定上面写好的生命周期函数，把启动关闭逻辑交给 app
app = FastAPI(title="Multimodal RAG Agent", version="1.0.0", lifespan=lifespan)

# CORS 中间件:允许跨域请求。ALLOWED_ORIGINS=["*"] 开发友好,生产环境应收紧到具体域名
# 调用方:前端 Vite dev server (:3000) 访问后端 (:3001) 时需要 CORS 放行
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注意: URL是 /static/images/xxx.png, 所以 DATA_DIR 需要包含 images 子目录
# 关键耦合:core/utils.py 的 get_image_url() 假设 /static -> DATA_DIR,改这里必须同步改那边
# Path 对象转成字符串给 StaticFiles 用，因为 StaticFiles 不支持 Path 对象
static_path = str(DATA_DIR)
print(f"📂 Mounting static files from: {static_path}")

# 挂载静态文件 (用于访问图片)，app.mount(路由前缀, StaticFiles(本地文件夹), name="别名")
# 为什么这个项目需要它：这是个多模态检索问答系统，检索结果会返回图片。前端要显示图片，但浏览器访问不了后端磁盘文件，所以后端开了 /static 这个"门"，把图片文件夹公开成网址
if DATA_DIR.exists():
    # 规则：URL 里的 /static 前缀被剥掉，剩下的路径直接拼到 DATA_DIR 后面
    # 前端请求的 URL：http://localhost:3001/static/images/1.png，实际读的磁盘文件：backend/data/images/1.png
    app.mount("/static", StaticFiles(directory=static_path), name="static")
else:
    print(f"⚠️ Warning: Static directory does not exist: {static_path}")
    
# 学过的 @app.get("/")，本质是"注册一个处理函数"——请求来了，代码跑一遍，返回 JSON。而 mount 本质是"把文件夹变成网址"——请求来了，框架连代码都不跑，直接去磁盘上找文件返回给你


# 注册路由(health 优先注册:OpenAPI 文档中靠前显示,且探活请求不会因业务路由异常而失败)
app.include_router(health.router)  # 健康检查路由 (优先)
app.include_router(search.router)
app.include_router(agent.router)

# 根路由:简单探活端点,无需认证,用于反向代理/CDN 健康检查
@app.get("/")
def root():
    return {"status": "ok", "message": "Multimodal RAG Agent Backend Running"}

# 当直接运行 python backend/main.py 的时候，启动 uvicorn 服务器
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)
