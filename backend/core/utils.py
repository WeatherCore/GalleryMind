# utils.py(图片处理工具):前端上传的 base64 图片落地、本地路径转 URL。
# 解决两个核心问题:
#  1. 上传落地:前端把图片编码成 base64 字符串塞进 JSON,后端需解码为 PNG 文件存到本地
#  2. URL 转换:检索引擎返回的是本地绝对路径,但前端只能通过 HTTP URL 访问,需转换
# 调用方:routers/search.py(上传图片时)、routers/agent.py(Agent 带图时)、core/retrieval.py(_format_results 时)

import base64
import uuid
from pathlib import Path
from PIL import Image
from io import BytesIO

from backend.config import DATA_DIR

# save_base64_image：把前端图片 base64 字符串解码并保存到文件夹 save_dir，返回在服务器上的完整本地路径
# 用 uuid4().hex 命名避免重名碰撞,所有上传图统一转 PNG(即使原图是 JPG,有轻微存储冗余但简化类型判断)
# 调用方:routers/search.py 与 routers/agent.py,在用户上传图片时调用
def save_base64_image(base64_str: str, save_dir: Path) -> Path:
    """如果 base64_str 包含 header (data:image/...), 去掉它并保存"""

    # 前端 FileReader.readAsDataURL 会产出 "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..." 格式
    # 逗号前面这一段是MIME 头部，不是图片二进制内容，解码必须删掉，
    if "," in base64_str:
        # 这里去掉 header 只保留 base64 部分
        base64_str = base64_str.split(",")[1]

    # b64decode()：把 base64 文本，解码还原成原始图片二进制字节，得到图片原始二进制数据
    image_data = base64.b64decode(base64_str)

    # BytesIO(image_data)：把二进制图片包成内存文件对象
    # Image.open()：相当于在内存虚拟出一个图片文件，Pillow 直接打开，不用先落地磁盘再打开，提升速度
    # 附带隐性校验：如果 base64 不是合法图片，`Image.open`会直接抛异常，不会生成空文件
    image = Image.open(BytesIO(image_data))

    # 生成唯一文件名(uuid4().hex 是 32 位十六进制,碰撞概率可忽略)
    filename = f"upload_{uuid.uuid4().hex}.png"

    # Path 类把 / 重新定义为"拼接路径"，所以写 save_dir / filename 时，真正执行的是 Path 内部的拼接逻辑
    file_path = save_dir / filename

    # 把图片对象写入磁盘，保存到上面拼接好的路径
    image.save(file_path)

    # 这个路径是服务器本地路径，不能直接返回给前端！浏览器访问不到服务器磁盘，要交给下面get_image_url()转换成 http 链接
    return file_path

# get_image_url(路径转 URL):把图片本地完整 Path 路径转为前端可访问的 /static/... URL 字符串。
# 前提假设:main.py 中 app.mount("/static", StaticFiles(directory=DATA_DIR)) 已挂载,
# FastAPI 把服务器本地DATA_DIR目录，映射到 web 路由前缀/static，
# 浏览器访问/static/xxx.png，FastAPI 自动去 DATA_DIR 下面找 xxx.png 返回图片
# 调用方:core/retrieval.py 的 _format_results,每条检索结果都调一次
def get_image_url(file_path: Path) -> str:
    """
    将本地文件路径转换为前端可访问的相对 URL (通过 /static 代理)

    假设:
    1. static 挂载在 /static
    2. static 指向 DATA_DIR (backend/data)
    """

    try:
        # file_path.relative_to(DATA_DIR)：计算 file_path 相对于 DATA_DIR 的相对路径
        #  - DATA_DIR=/xxx/GalleryMind/backend/data
        #  - file_path=/xxx/GalleryMind/backend/data/gallery/001.png
        #  relative_to执行后得到：gallery/001.png
        relative_path = file_path.relative_to(DATA_DIR)

        # 拼接得到 url：/static/gallery/001.png，前端<img src="/static/gallery/001.png"/>直接展示图片
        return f"/static/{relative_path}"

    # 当 file_path 不在 DATA_DIR 文件夹内部的时候！此时file_path.relative_to(DATA_DIR)会报错，进入 except 分支
    except ValueError:
        # 兜底方案：直接拿文件名，返回`/static/images/文件名.png`
        # FastAPI 的 StaticFiles 挂载的是 DATA_DIR，默认不存在 images 子文件夹，浏览器访问这个 url 会 404，图片加载失败，这是原项目的一个小 bug，后面可以优化
        return f"/static/images/{file_path.name}"
