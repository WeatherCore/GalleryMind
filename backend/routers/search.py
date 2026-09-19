# search.py(多模态图像检索业务接口):前端检索模式的 HTTP 入口，支持文本搜图、以图搜图、图文混合检索；同时内置 Mock 假数据分支，和我们前面`lifespan`的 mock 开关配套
# 解决两个核心问题:
#  1. 同步检索:接收 SearchRequest,调 retrieval_engine.search,返回 SearchResponse JSON
#  2. Mock 开关:USE_MOCK_DATA=true 时不加载任何模型,直接返回预设假数据,用于纯前端开发与 CI 联调
# 调用方:前端 App.tsx 的 handleSearch() 通过 fetch POST /api/search 调用
import time
import os
from fastapi import APIRouter, HTTPException
from pathlib import Path

from backend.schemas import SearchRequest, SearchResponse, SearchResult
from backend.core.utils import save_base64_image
from backend.config import UPLOAD_DIR
from backend.routers.history import record_search

# 创建独立路由实例，相当于一个子路由容器
router = APIRouter(prefix="/api/search", tags=["Multimodal Search"])

# 读取环境变量 Mock 模式开关（用于快速测试前后端联动，无需加载模型）
# 默认关闭 Mock 模式，启用真实检索
# [补充] 设计意图:让纯前端开发者无需 GPU/模型也能跑通前后端联调;
# 启动命令 USE_MOCK_DATA=true python -m backend.main 即可激活
USE_MOCK_DATA = os.environ.get("USE_MOCK_DATA", "false").lower() == "true"

# Mock 模式的硬编码假结果列表 - 使用实际存在的图片文件
# 这 6 条假数据模拟真实检索结果结构(含 imageUrl/score/relevanceScore/metadata/tags),
# 结构完全对齐 SearchResult 的字段，这样后面可以直接 SearchResult(**res) 解包生成模型实例，
# 前端无需任何修改即可渲染
MOCK_RESULTS = [
    {
        "id": "1",
        "imageUrl": "/static/images/20251216110324_89_50.png",
        "title": "架构图示例1",
        "score": 95.5,
        "relevanceScore": 0.955,
        "metadata": {"file_name": "20251216110324_89_50.png", "file_path": "/data/images/20251216110324_89_50.png"},
        "dimensions": "800x600",
        "format": "PNG",
        "tags": ["Architecture", "Floor Plan", "Modern"]
    },
    {
        "id": "2",
        "imageUrl": "/static/images/20251216110347_90_50.png",
        "title": "系统流程图",
        "score": 88.2,
        "relevanceScore": 0.882,
        "metadata": {"file_name": "20251216110347_90_50.png", "file_path": "/data/images/20251216110347_90_50.png"},
        "dimensions": "1024x768",
        "format": "PNG",
        "tags": ["System Design", "Flowchart"]
    },
    {
        "id": "3",
        "imageUrl": "/static/images/20251216110427_91_50.png",
        "title": "数据流向图",
        "score": 75.3,
        "relevanceScore": 0.753,
        "metadata": {"file_name": "20251216110427_91_50.png", "file_path": "/data/images/20251216110427_91_50.png"},
        "dimensions": "1200x900",
        "format": "JPG",
        "tags": ["Data Flow", "Diagram"]
    },
    {
        "id": "4",
        "imageUrl": "/static/images/20251216110526_92_50.png",
        "title": "技术架构图",
        "score": 68.7,
        "relevanceScore": 0.687,
        "metadata": {"file_name": "20251216110526_92_50.png", "file_path": "/data/images/20251216110526_92_50.png"},
        "dimensions": "1920x1080",
        "format": "PNG",
        "tags": ["Tech Stack", "Architecture"]
    },
    {
        "id": "5",
        "imageUrl": "/static/images/Image_7k9my37k9my37k9m.png",
        "title": "系统设计图",
        "score": 62.4,
        "relevanceScore": 0.624,
        "metadata": {"file_name": "Image_7k9my37k9my37k9m.png", "file_path": "/data/images/Image_7k9my37k9my37k9m.png"},
        "dimensions": "800x600",
        "format": "JPG",
        "tags": ["Design", "System"]
    },
    {
        "id": "6",
        "imageUrl": "/static/images/Image_aw9my8aw9my8aw9m.png",
        "title": "模块架构图",
        "score": 55.1,
        "relevanceScore": 0.551,
        "metadata": {"file_name": "Image_aw9my8aw9my8aw9m.png", "file_path": "/data/images/Image_aw9my8aw9my8aw9m.png"},
        "dimensions": "1024x768",
        "format": "PNG",
        "tags": ["Module", "Architecture"]
    },
]

# search(同步检索 /api/search):检索模式统一入口
# response_model=SearchResponse 让 FastAPI 自动校验+序列化响应
# Mock 模式:直接返回 MOCK_RESULTS[:rerankTopK],模拟 0.5s 延迟(让前端进度条有时间渲染);
# 真实模式:调 retrieval_engine.search,异常时 traceback 打印 + HTTP 500 返回
@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest):

    # 获取当前时间戳（单位：秒），记录请求进入接口的时刻。后面用「当前时间 - start_time」算出整个检索耗时，返回给前端展示
    start_time = time.time()

    # Mock 模式分支：环境变量 USE_MOCK_DATA=true 才进这里，直接返回假数据用于测试前后端联动
    if USE_MOCK_DATA:

        # 打印日志，方便开发看前端发过来是什么检索模式、文本查询词
        print(f"🧪 [MOCK MODE] 收到检索请求: mode={request.searchMode}, query={request.textQuery}")
        # 局部延迟导入。只有 mock 模式才加载 asyncio，正常业务分支不执行这行，节省资源
        import asyncio
        # 模拟处理延迟(异步休眠 0.5 秒，让前端 ProcessVisualization 进度条有可视化时间)
        await asyncio.sleep(0.5)

        # 计算耗时：当前时间减去请求起始时间
        duration = time.time() - start_time

        # - MOCK_RESULTS[:request.rerankTopK]：mock 假数据数组，切片，只取前rerankTopK条，和真实逻辑保持一致（由前端指定返回多少条结果）
        # - SearchResult(**res) **res字典解包：mock 里的字典 key，直接作为参数传给 SearchResult Pydantic 模型，自动构造对象，自动校验字段
        results = [SearchResult(**res) for res in MOCK_RESULTS[:request.rerankTopK]]

        # 调用历史记录函数，保存本次检索记录。`mock=True`标记这条是假数据，方便区分测试记录和真实检索记录
        record_search(request.textQuery, request.searchMode, len(results), duration, mock=True)

        # 返回`SearchResponse`实例：
        # - results：包装好的检索结果列表
        # - processSteps：空列表，预留字段，本 demo 没有实现检索分步进度推送（Agent 那边是 SSE 做步骤推送，检索这里简化了）
        # - totalTime：round 保留 3 位小数，把耗时传给前端页面展示
        return SearchResponse(
            results=results,
            processSteps=[],
            totalTime=round(duration, 3)
        )

    # 真实检索模式分支：使用 Qwen3-VL 引擎
    # try/except 捕获 retrieval_engine 异常,traceback 打印到服务端日志 + HTTP 500 给前端
    try:
        # 延迟导入 retrieval_engine 检索引擎单例，
        # `..`代表上一级目录，当前文件`routers/search.py`，`..`就是 backend 文件夹
        from ..core.retrieval import retrieval_engine
        from ..config import DEFAULT_IMAGE_DIR

        # 懒初始化兜底：若引擎未就绪则现场初始化(正常情况下，main.py 的 lifespan 在服务启动阶段就执行)
        if not retrieval_engine.initialized:
            retrieval_engine.initialize(image_dir=DEFAULT_IMAGE_DIR)
        # 缺陷：
        # 并发不安全 如果两个检索请求同时到达，此时initialized=False，两个请求会同时调用 initialize () → 并行加载模型，多次实例化 Qwen3-VL 模型，直接爆显存、程序崩溃
        # 解决方案：初始化加锁`threading.Lock()`，保证 initialize 只能被执行一次

        # image_path 传给检索引擎，用来做图像 Embedding，实现以图搜图；
        # 如果没有上传图片，image_path=None，走纯文本检索
        image_path = None

        # 如果前端上传图片（以图搜图场景，uploadedImage 不为空），则保存到本地
        if request.uploadedImage:
            # 调用save_base64_image（我们前面拆解的 utils 工具函数），base64 图片解码保存到 UPLOAD_DIR，返回本地图片路径
            image_path = save_base64_image(request.uploadedImage, UPLOAD_DIR)

        # 整条检索链路的入口调用：把前端所有参数一次性传给 retrieval_engine 的 search 方法
        raw_results = retrieval_engine.search(
            mode=request.searchMode,
            query=request.textQuery,
            image_path=image_path,
            recall_top_k=request.recallTopK,
            rerank_top_k=request.rerankTopK,
            score_threshold=request.threshold
        )
        # 内部完整流程回顾：
        # 1. 根据 searchMode 判断检索类型（文本检索 / 图像检索 / 图文混合）
        # 2. 文本 / 图片送入 Qwen3-VL Embedding 生成向量
        # 3. Milvus 向量召回 + BM25 关键词召回，多路召回融合
        # 4. Qwen3-VL Reranker 做精排
        # 5. 根据 threshold 过滤低分结果
        # 6. 返回 raw_results：字典列表，里面包含图片 id、本地路径、分数、metadata

        # raw_results 是 List[Dict] 检索引擎返回的原始字典
        # 循环用`SearchResult(**res)`转成 Pydantic 模型，统一校验数据格式
        results = [SearchResult(**res) for res in raw_results]

        # 计算耗时：当前时间减去请求起始时间
        duration = time.time() - start_time

        # 调用历史记录函数，保存本次检索记录
        record_search(request.textQuery, request.searchMode, len(results), duration, mock=False)

        # 组装响应对象返回前端。
        # processSteps 为空数组，预留扩展位，后面可以改成 SSE 流式返回检索每一步状态（编码向量 → 向量召回 → 重排完成）
        return SearchResponse(
            results=results,
            processSteps=[],  # 始终空数组,真实进度由前端 processSimulator.ts 模拟
            totalTime=round(duration, 3)
        )

    # 全局异常捕获块：任何报错：图片解析失败、Milvus 断连、模型推理报错、文件读取异常，都会进到 except
    except Exception as e:

        import traceback
        # `traceback.print_exc()`：后端控制台打印完整堆栈，开发阶段定位 bug 非常方便
        traceback.print_exc()
        # `raise HTTPException(500)` 返回服务内部错误，把异常信息传给前端
        raise HTTPException(status_code=500, detail=str(e))
    
    # 缺陷：开发模式没问题；生产环境不推荐。直接把底层异常暴露给前端，容易泄露服务器路径、内部代码信息，存在安全风险。生产需要捕获异常，返回统一的友好提示，日志单独记录详细错误

# 遗留问题清单（复刻项目优化清单）
# 1. retrieval_engine.initialize 并发问题，缺少锁
# 2. uploadedImage 没有校验图片大小，恶意超大 base64 会占用大量内存
# 3. UPLOAD_DIR 图片不会自动清理，长期运行磁盘爆满
# 4. processSteps 没有实现分步状态推送，无法在前端展示检索进度
# 5. 异常直接返回底层错误信息，生产不安全
# 6. 没有请求限流，高频请求会重复触发模型推理，GPU 资源耗尽