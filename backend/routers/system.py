# system.py(系统信息接口):前端"图库浏览页"与"系统状态页"的数据源,兼图片库管理。
# 解决四个核心问题:
#  1. 图片列表:GET /api/images 返回图片库全部图片(文件名/URL/大小/时间/caption),
#     解决 /static 只能按文件名访问、无法"列出有哪些图"的问题
#  2. 图片管理:POST /api/images/upload 上传入库、DELETE /api/images/{name} 删图,
#     配合 POST /api/images/reindex 实现不重启服务的热重建索引
#  3. 系统状态:GET /api/system/status 汇总后端版本/Milvus 连通性/引擎初始化/模型配置/图片数量,
#     全部只读探测,绝不触发引擎初始化(状态页不应该把服务拖去加载模型)
# 调用方:前端 GalleryPage.tsx 与 StatusPage.tsx

import os
import re
import time
import socket
import threading
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, UploadFile

from backend.config import (
    CAPTION_CACHE_DIR, DEFAULT_IMAGE_DIR, MILVUS_URI,
    EMBEDDING_MODEL_MS, RERANKER_MODEL_MS,
    AGENT_LLM_MODEL, VISION_LLM_MODEL,
)

# 创建独立路由实例，相当于一个子路由容器
router = APIRouter(prefix="/api", tags=["System Info"])

# 集合，存放允许的图片后缀。扫描文件时，只识别这 4 种格式，其他文件直接跳过
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif"}

# 合法文件名白名单：仅允许字母数字下划线连字符点号，杜绝路径穿越(../)与非法字符，
# 后面`_validate_filename`会使用这个正则
SAFE_FILENAME = re.compile(r"^[\w\-. ]+\.(png|jpg|jpeg|gif)$", re.IGNORECASE)

# 热重建的并发保护线程锁：同一时刻只允许一个重建任务；状态供接口拒绝重复触发
_reindex_lock = threading.Lock()

# _load_captions(读取 caption 缓存)：把 caption_cache/*.txt 读为 {图片stem: 描述}。
# 与 retrieval.py 的 load_caption_cache 逻辑一致,但这里本地重实现,
# 避免 import retrieval 连带拉起 torch(列表接口要轻、要快)
def _load_captions() -> dict[str, str]:

    # 初始化空字典。最终返回的结构：`key`是文件名主干，`value`是图片描述文本
    captions = {}
    # 判断 caption 缓存目录是否存在。如果文件夹根本不存在，直接跳过循环，返回空字典
    if CAPTION_CACHE_DIR.exists():
        # 遍历这个目录下所有后缀是 txt 的文件
        for f in CAPTION_CACHE_DIR.glob("*.txt"):
            try:
                # f.stem：Path 内置属性，拿到文件名去掉后缀
                # f.read_text(encoding="utf-8")：直接读取 txt 全部内容，用 utf8 编码，拿到图片 caption
                # 存入字典：key 是图片主干名，value 是描述文字
                captions[f.stem] = f.read_text(encoding="utf-8")
            except OSError:
                # 捕获文件系统异常：文件权限不足、文件损坏、读取时文件被删除
                continue
    # 示例：`{"img001": "海边落日，一张风景照片"}`
    return captions


# _scan_images(扫描图片库)：扫描图库目录`DEFAULT_IMAGE_DIR`，筛选合法图片文件，读取图片元信息，并匹配对应的 caption 描述，组装成图片信息列表给前端
# 与检索索引同源(同一个 DEFAULT_IMAGE_DIR),保证图库页看到的 = 引擎能搜到的
def _scan_images() -> list[dict]:

    # 先一次性加载全部 caption 到内存字典
    captions = _load_captions()

    # 用来存放每张图片的信息字典，最后返回给前端
    items = []

    # 判断图库文件夹是否存在，不存在直接跳过遍历
    if DEFAULT_IMAGE_DIR.exists():

        # iterdir()遍历图库目录下面所有子项（文件 + 文件夹）。path 是 Path 对象
        for path in DEFAULT_IMAGE_DIR.iterdir():

            # - 只处理文件，跳过子文件夹，
            # - 拿到后缀转小写，判断是否在允许图片后缀集合IMAGE_EXTS={".png",".jpg",".jpeg",".gif"}
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:

                try:
                    # path.stat()调用操作系统接口，拿到文件元数据：
                    # - stat.st_size：文件大小，单位字节
                    # - stat.st_mtime：文件最后修改时间，返回 Unix 时间戳（浮点数）
                    stat = path.stat()

                    items.append({
                        "filename": path.name,
                        "url": f"/static/images/{path.name}",
                        "sizeBytes": stat.st_size,
                        "modifiedAt": stat.st_mtime,
                        "hasCaption": path.stem in captions,
                        "caption": captions.get(path.stem),
                    })
                    # 组装单张图片的信息：
                    # - `filename`：图片全名，例如`img001.png`
                    # - `url`：前端静态资源访问地址，FastAPI 挂载 static 目录，浏览器直接访问展示图片（前面我们讲过 StaticFiles）
                    # - `sizeBytes`：图片字节大小
                    # - `modifiedAt`：修改时间戳，前端可以转成可读日期
                    # - `hasCaption`：布尔值。判断这张图片有没有对应的 txt 描述文件
                    # `path.stem in captions`：图片主干名是否存在 caption 字典的 key 中
                    # - `caption`：如果有，就是 txt 里的图片描述；没有就是`None`。`dict.get`安全取值，key 不存在返回 None，不会抛 KeyError

                except OSError:

                    # 个别文件读不到 stat(被占用/权限),跳过不中断整个列表
                    continue
    
    # 列表排序：
    # - key=lambda x: x["modifiedAt"]：拿每张图片的修改时间作为排序依据
    # - reverse=True：倒序，新修改 / 新上传的图片排在列表最顶部。符合前端用户习惯，最新图片放最前面
    items.sort(key=lambda x: x["modifiedAt"], reverse=True)

    # 返回图片信息数组，交给`GET /api/images`接口，返回 JSON 给前端图库页面渲染
    return items


# _probe_milvus(Milvus 探活)：对 MILVUS_URI 做一次轻量 TCP 连接测试，检测 Milvus 向量数据库服务端口能不能连通。仅做 TCP 端口探测，不是完整 Milvus 客户端连接，用来做服务健康检查
# 为什么不用 pymilvus 客户端:客户端 SDK 会做握手/版本协商,且可能触发重连逻辑;
# 一个短超时 TCP connect 足以回答"通没通",状态页要快、要无副作用
# - uri: str：Milvus 连接 uri，示例milvus://127.0.0.1:19530
# - timeout: float = 0.6：可选，TCP 连接超时时间，单位秒，默认 0.6s
def _probe_milvus(uri: str, timeout: float = 0.6) -> bool:
    try:
        # urlparse是 urllib 内置工具，专门解析 url 字符串，把 milvus://127.0.0.1:19530 拆成多个部分
        # parsed 会得到一个 ParseResult 对象：
        # - parsed.scheme → `milvus`
        # - parsed.hostname → `127.0.0.1`
        # - parsed.port → `19530`
        parsed = urlparse(uri)

        # 拿到解析出来的主机 IP / 域名
        host = parsed.hostname or "localhost"
        port = parsed.port or 19530

        #socket.create_connection：底层 TCP 建立连接，向 (host,port) 发起握手。
        # - (host, port)：目标地址和端口
        # - timeout=timeout：连接最长等待时间，超过就抛超时异常
        # - with语法：连接成功之后，代码块结束自动关闭 socket 套接字，不用手动 close，避免文件句柄泄漏
        with socket.create_connection((host, port), timeout=timeout):
            # TCP 握手成功，说明网络层面端口可达，返回 True
            return True

    # `OSError`：网络类异常都会归到这里，包括：找不到主机、连接拒绝、连接超时
    except OSError:
        # 捕获异常后，直接返回 False，代表 Milvus 端口不可访问
        return False

# 作用：文件名安全校验，防止恶意文件名攻击（路径穿越攻击）
# - 参数：filename：前端传过来的图片文件名
# - 返回：校验合法的原文件名
def _validate_filename(filename: str) -> str:

    # 判断：
    #  ① SAFE_FILENAME.match(filename)：从头匹配文件名，只允许字母数字下划线、横杠、空格，后缀只能是图片格式
    #  ② ".." in filename：检测有没有两点..，用来防御路径穿越攻击（比如恶意文件名 ../../etc/passwd，..代表向上跳一级目录）
    if not SAFE_FILENAME.match(filename) or ".." in filename:

        # 抛出FastAPI内置异常，HTTP状态码400（代表客户端提交参数非法），返回提示信息
        raise HTTPException(status_code=400, detail=f"非法文件名: {filename}")

    # 校验全部通过，把原文件名返回出去，供后续保存文件使用
    return filename


# list_images(图片列表 /api/images)：扫描图库目录，返回全部图片列表，前端页面展示图片画廊
@router.get("/images")
def list_images():

    # 调用我们之前看过的`_scan_images()`函数：
    # 1. 先加载 caption 缓存（图片描述 txt）
    # 2. 遍历图片目录，筛选图片后缀
    # 3. 收集每张图片：文件名、静态访问 url、文件大小、修改时间，返回 list [dict]
    images = _scan_images()

    # 返回 JSON 结构，前端拿到直接渲染图库页面
    return {"total": len(images), "images": images}


# upload_image(上传入库 /api/images/upload):上传图片到图库目录 DEFAULT_IMAGE_DIR，返回图片信息
# - 入参：file: UploadFile 前端form-data上传的文件（FastAPI内置上传文件类型）
# - 返回：上传成功的信息，包含访问url、文件名、文件大小
# 注意:入库 ≠ 可检索。真实模式下还需调 /api/images/reindex 重建索引(响应里带提示)
@router.post("/images/upload")
async def upload_image(file: UploadFile):

    # # 获取原始文件名，如果为空就默认叫 upload.png
    original = file.filename or "upload.png"

    # 调用上面的校验函数，检查文件名是否合法，防路径穿越攻击，不合法直接抛400
    filename = _validate_filename(original)

    # 拼接完整保存路径：图库目录 + 合法文件名
    target = DEFAULT_IMAGE_DIR / filename
    # 记录文件字节大小，初始0
    size = 0

    try:
        # 以二进制写入模式打开目标文件
        with open(target, "wb") as out:
            # 循环读取文件块，每次最多读取1MB(1024*1024字节)，await是异步读取
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)   # 累加本次读到的数据长度，统计总大小
                out.write(chunk)     # 把读到的二进制块写入本地文件

    # 如果发生磁盘IO错误（磁盘满、权限不足等）
    except OSError as e:
        # 删除已经创建的残缺文件；missing_ok=True：文件不存在也不会报错
        target.unlink(missing_ok=True)  # 写一半失败不留残片
        # 返回500服务端错误，把异常信息传给前端
        raise HTTPException(status_code=500, detail=f"保存文件失败: {e}")

    # 无论成功还是报错，finally一定会执行：关闭上传文件流，释放资源
    finally:
        await file.close()

    # 上传成功返回JSON
    return {
        "status": "ok",
        "filename": filename,
        "sizeBytes": size,
        "url": f"/static/images/{filename}",
        "hint": "图片已入库。真实模式下需重建索引后才能被检索到",
    }


# delete_image(删图 /api/images/{filename})：删除图库中的图片，同时配套删除对应的caption描述缓存txt文件
# - 入参：filename 路径参数，图片文件名（如 test.png）
# - 返回：删除结果，标记是否同步删掉了caption缓存
# 索引中的残留向量在下次重建索引时自然消失,无需在此同步删 Milvus 数据
@router.delete("/images/{filename}")
def delete_image(filename: str):

    # 调用文件名安全校验，拦截非法文件名、包含..的路径穿越攻击
    _validate_filename(filename)
    # 拼接图片完整本地路径：图库目录 + 文件名
    target = DEFAULT_IMAGE_DIR / filename

    # 判断：如果这个图片文件不存在
    if not target.exists():
        # 抛出404，资源不存在
        raise HTTPException(status_code=404, detail="图片不存在")

    try:
        # Path.unlink() 删除磁盘文件
        target.unlink()
    except OSError as e:
        # IO异常：文件被占用、权限不足等，返回500
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")

    # 拼接caption缓存文件路径：用图片文件名不带后缀，拼接 txt 后缀
    caption_file = CAPTION_CACHE_DIR / f"{target.stem}.txt"

    # 标记caption文件是否成功删除，初始False
    caption_removed = False

    # 如果caption缓存txt存在
    if caption_file.exists():
        try:
            # unlink() 删除磁盘文件
            caption_file.unlink()
            # 标记为True
            caption_removed = True

        # 删除caption出错直接忽略，不阻断主流程（图片已经删掉了，caption删不掉不抛异常）
        except OSError:
            pass

    # 返回结果给前端，包含删除的文件名、是否成功删除caption
    return {"status": "ok", "deleted": filename, "captionRemoved": caption_removed}


# reindex(热重建索引 /api/images/reindex):不重启服务，用当前图片库目录全量重建多模态向量索引 --- Milvus 索引与 BM25。
# 三种结果:
#   ok      → 真实模式下重建完成(耗时与图片数成正比,接口同步等待)
#   skipped → 引擎未加载(如 Mock 模式):图片本身已入库,下次真实启动自动建索引
#   busy    → 已有重建任务在跑,拒绝并发触发
# 核心：加锁，**同一时间只能跑一次重建任务，防止并发建索引造成Milvus数据错乱**
@router.post("/images/reindex")
def reindex():

    # - `_reindex_lock = threading.Lock()` 前面定义的线程锁
    # - `acquire(blocking=False)`：非阻塞抢锁
    #   - 拿到锁 → 返回 True，继续执行重建
    #   - 拿不到锁（上一次重建还没跑完）→ 返回 False，直接返回 busy 提示，不阻塞等待
    # - 目的：禁止多人同时点击重建索引，并发写入 Milvus 会造成数据混乱、重复向量、崩溃
    if not _reindex_lock.acquire(blocking=False):
        return {"status": "busy", "message": "已有重建任务在进行中,请稍候"}

    try:
        # 延迟导入：只有触发重建索引的时候，才导入retrieval引擎（torch、transformers都是重量级包）
        # 好处：Mock模式启动服务时，不用加载大模型，启动速度快，省内存
        try:
            from backend.core.retrieval import retrieval_engine
            from backend.config import DEFAULT_IMAGE_DIR as _DIR
        except Exception as e:
            # 如果导入失败（模型缺失、依赖没装），直接返回 skipped，不会 500 崩溃
            return {
                "status": "skipped", 
                "message": f"检索引擎不可用({e}),图片已入库,启动真实模式时会自动建索引"
            }

        # 判断检索引擎是否完成初始化，Mock 模式下`retrieval_engine.initialized = False`，直接跳过重建，返回提示，避免在纯前端调试环境去加载大模型
        if not retrieval_engine.initialized:
            return {
                "status": "skipped",
                "message": "引擎未加载(可能为 Mock 模式),图片已入库,下次真实启动时自动建索引",
            }

        print("🔄 [REINDEX] 收到重建索引请求")

        # 手动把引擎状态标记为未初始化
        # initialize 方法内部开头会判断 if self.initialized: return。 如果标记还是True，initialize直接原地 return，不会执行重建逻辑！ 所以必须手动置False，强制让initialize完整跑一遍
        retrieval_engine.initialized = False
        # 调用initialize：内部逻辑会清空 Milvus 原有集合，遍历图库所有图片，批量生成多模态向量，存入 Milvus
        retrieval_engine.initialize(_DIR)

        # 扫描图片文件夹，统计图片数量，返回给前端重建结果
        image_count = len(_scan_images())
        print(f"✅ [REINDEX] 重建完成,当前图片 {image_count} 张")

        return {
            "status": "ok", 
            "indexed": image_count, 
            "message": f"重建完成,已索引 {image_count} 张图片"
        }

    # 不管重建索引正常结束、还是中途抛出异常崩溃，锁一定会释放
    finally:
        _reindex_lock.release()


# system_status(系统状态 /api/system/status)：系统状态巡检接口，前端仪表盘展示整个后端服务的各项状态
# engine.initialized 三态:true(已就绪) / false(未加载) / null(引擎模块不可用,如未装 torch)
# 探测过程全部 try/except 包裹:状态接口自己绝不能 500,坏了也要把"哪里坏了"报给前端
@router.get("/system/status")
def system_status():

    # 定义变量，初始值None（代表无法获取状态）
    engine_initialized = None
    try:
        # 延迟导入检索引擎
        from backend.core.retrieval import retrieval_engine
        # 读取引擎初始化标记：True=引擎就绪，False=未初始化，None=导入失败
        engine_initialized = retrieval_engine.initialized

    except Exception:
        # 导入/读取属性发生任何异常，捕获，保持engine_initialized=None
        engine_initialized = None

    try:
        # 调用之前写好的扫描图片函数，统计图库图片总数
        image_count = len(_scan_images())
    except Exception:
        # 扫描图片出错（文件夹不存在、权限不足），数量置为None
        image_count = None

    # 返回状态汇总JSON
    return {
        "backend": {"status": "ok", "version": "1.0.0"},
        "milvus": {"uri": MILVUS_URI, "connected": _probe_milvus(MILVUS_URI)},
        "engine": {"initialized": engine_initialized},
        "models": {
            "embedding": EMBEDDING_MODEL_MS,
            "reranker": RERANKER_MODEL_MS,
            "agentLlm": AGENT_LLM_MODEL,
            "visionLlm": VISION_LLM_MODEL,
        },
        "gallery": {"dir": str(DEFAULT_IMAGE_DIR), "count": image_count},
        "mockMode": os.environ.get("USE_MOCK_DATA", "false").lower() == "true",
        "timestamp": time.time(),
    }
