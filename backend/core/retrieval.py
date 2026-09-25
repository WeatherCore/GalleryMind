# retrieval.py(检索引擎核心):项目的"大脑",封装多模态检索全流程。
# 解决四个核心问题:
#  1. 多模态嵌入:Qwen3-VL-Embedding-2B 把文本/图片编码为 512 维向量,L2 归一化
#  2. 双索引系统:Milvus 图片向量索引(图搜图)+ 混合索引(Qwen3-VL 向量 + BM25,文搜图)
#  3. 三路召回融合:Qwen3-VL 向量 + BM25 + (caption 向量,已禁用) → QueryFusionRetriever RRF 融合
#  4. 共享 Reranker:Qwen3-VL-Reranker-2B 单例复用,避免每次检索重新加载 2B 大模型
# 调用方:core/agent.py 的 search_images 工具、routers/search.py 的 search 接口

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import Field

# LlamaIndex Core
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, QueryBundle, Settings
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.embeddings import MultiModalEmbedding
from llama_index.core.schema import ImageNode, NodeWithScore
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.retrievers.bm25 import BM25Retriever

# 兼容处理：try-except 检测 modelscope 库是否安装
try:
    # modelscope 是阿里魔搭的 Python 库；snapshot_download 是魔搭提供的模型下载函数，可以把模型从魔搭平台下载到本地缓存目录（替代 HuggingFace 的hf_hub_download）
    from modelscope import snapshot_download
    # 如果环境已经安装 `modelscope` 包：导入成功，设置 `MODELSCOPE_AVAILABLE = True`
    MODELSCOPE_AVAILABLE = True
except ImportError:
    # 如果没装这个包（`pip install modelscope`没执行）：会抛出`ImportError`，
    # 进入 except 分支，`MODELSCOPE_AVAILABLE = False`
    MODELSCOPE_AVAILABLE = False

# MODELSCOPE_AVAILABLE就是这个开关
# - 为 True：优先从魔搭 ModelScope 下载模型（国内网络，速度快）
# - 为 False：降级，直接用 HuggingFace 的模型地址，从 hf 下载（国内网络容易超时）
# 这是工程设计思想：
# 环境自适应降级策略 国内用户推荐装 modelscope，加速模型下载；国外 / 不想装 modelscope 的用户，也能正常跑，只是下载源换成 HuggingFace，代码不用改

# 从配置文件读取全局常量，都是提前定义好的路径、模型标识、向量库连接地址
from backend.config import (
    MODEL_CACHE_DIR, CAPTION_CACHE_DIR, EMBEDDING_MODEL_MS, RERANKER_MODEL_MS,
    MILVUS_URI
)
from backend.core.utils import get_image_url

# 向量维度，多模态模型输出 512 维向量，Milvus 建索引必须和这个维度严格一致
EMBEDDING_DIM = 512

# ============================================================
# 辅助函数
# ============================================================

# download_model_from_modelscope(模型下载):从 ModelScope 魔搭平台下载模型到本地缓存。
# 返回值：成功返回本地磁盘模型路径；失败直接返回原始 model_id 字符串
def download_model_from_modelscope(model_id: str, cache_dir: Path) -> str:

    # model_id：魔搭平台模型 ID，例如Qwen/Qwen3-VL-Embedding-2B
    # cache_dir：模型缓存目录（就是之前配置的MODEL_CACHE_DIR，存模型文件）

    # 如果没安装 modelscope 包 → 直接返回 model_id 字符串
    if not MODELSCOPE_AVAILABLE:
        return model_id

    print(f"\n📥 从 ModelScope 下载: {model_id}")

    try:
        # 调用snapshot_download下载模型，返回本地模型文件夹路径
        model_dir = snapshot_download(model_id, cache_dir=str(cache_dir), revision='master')
        # 下载成功，返回本地模型路径，后续`AutoModel.from_pretrained`加载这个本地目录
        return model_dir

    # 捕获下载异常（网络、权限），降级返回 model_id，交给 huggingface‑transformers 尝试拉取
    except Exception as e:
        print(f"⚠️ ModelScope 下载失败: {e}, 尝试直接使用 ID")
        return model_id

# load_caption_cache(加载 caption 缓存):读取图片描述缓存文件夹，把 caption_cache/*.txt 加载为 {图片名: 描述} 字典
# 用途:文搜图时把图片文件名 + caption 拼成 ImageNode 的 text 字段,提升 BM25 关键词检索效果
# 设计背景：对图片生成文本 caption（图片内容描述）算力开销很大，不要每次启动服务重新跑一遍
def load_caption_cache(cache_dir: Path = CAPTION_CACHE_DIR) -> Dict[str, str]:

    # 入参：缓存文件夹，默认CAPTION_CACHE_DIR
    # 返回：字典{图片文件名不带后缀: "图片描述文本"}

    # 初始化空字典
    captions = {}

    # 如果缓存文件夹存在并且不为空，遍历全部*.txt文件，读取内容存入字典
    if cache_dir and cache_dir.exists():
        # 遍历文件夹里所有后缀`.txt`的文件
        for cache_file in cache_dir.glob("*.txt"):
            try:
                # stem 是文件名去扩展名,与图片文件名对应(如 foo.txt → foo.png)
                # 拿不带后缀文件名作为 key；读取文本作为 value 存入字典
                captions[cache_file.stem] = cache_file.read_text(encoding='utf-8')

            # 极简异常捕获：读取某个 txt 文件损坏、编码异常时，跳过这个文件，不打断整体加载。只忽略坏文件，其他缓存继续加载
            except:
                pass

    # 返回字典：{ "img_001":"一只猫坐在草地上", "img_002":"海边日落", ... }
    return captions
    # 后续构建索引的时候，优先读这个缓存；缓存不存在，才调用多模态模型生成 caption，生成完写入 txt 缓存，下次启动直接读取


# ============================================================
# 模型类定义
# ============================================================

# Qwen3VLEmbedding(多模态嵌入封装)：封装通义 Qwen3‑VL‑Embedding‑2B 多模态嵌入模型
# 解决两个核心问题:
#  1. encode_text()：输入文本，输出 512 维向量
#  2. encode_image()：输入本地图片路径，输出 512 维向量
# 上游：RetrievalEngine 使用这个类；下游输出向量存入 Milvus 向量库，用于文搜图、图搜图。关键点：这个不是直接继承 Llama‑Index 的MultiModalEmbedding，是原生模型封装，后面会做一层适配器对接 Llama‑Index
#   为什么写这个类？
# 1. 统一收口：把模型加载、图片预处理、文本预处理、向量提取全部打包在这一个类里面
# 2. 隔离底层huggingface代码：项目其他地方调用时，不用管processor、tokenizer、hidden_states这些底层细节
# 3. 对外暴露简洁方法 encode_text / encode_image，输入文本/图片，直接输出numpy向量
# 4. 统一设备管理（自动识别cuda/mps/cpu），封装逻辑，不用到处写重复代码
class Qwen3VLEmbedding:
    """Qwen3-VL-Embedding-2B 封装"""

    # __init__ 构造函数，实例化对象的时候执行，加载模型、processor，选择运行设备
    def __init__(self, model_path: str = None, output_dim: int = EMBEDDING_DIM, device: str = "auto"):

        # 入参：
        # model_path：模型本地路径 / 模型 ID；不传就自动从 ModelScope 下载
        # output_dim：输出向量维度，默认读取全局常量EMBEDDING_DIM=512
        # device="auto"：自动选择运行设备，可选 cuda / mps / cpu

        # 局部 import：延迟导入
        # AutoModel：加载模型权重
        # AutoProcessor：多模态处理器，处理图片、文本，转成模型需要的 tensor 张量
        from transformers import AutoModel, AutoProcessor
        # process_vision_info：Qwen‑VL 官方工具函数，处理图片信息，预处理图片
        from qwen_vl_utils import process_vision_info

        # 输出向量维度，保底全局常量EMBEDDING_DIM=512，Milvus 集合向量维度必须和这个严格匹配
        self.output_dim = output_dim
        # 把工具函数绑定为实例属性，后续方法直接调用
        self.process_vision_info = process_vision_info

        # 自动设备判断 + 数据类型 dtype 选择，适配 NVIDIA 显卡 / Mac M 系列 / CPU
        # - cuda：NVIDIA 显卡，使用bfloat16，显存占用小，适合大模型
        # - mps：苹果 Mac 芯片 (M1/M2/M3)，用float16
        # - cpu：CPU 运行，float32
        if device == "auto":
            if torch.cuda.is_available():
                # cuda + bfloat16：N 卡，bfloat16 显存占用更低，Qwen3‑VL 模型原生支持 bf16
                self.device, dtype = "cuda", torch.bfloat16
            elif torch.backends.mps.is_available():
                # mps + float16：Apple Silicon (Mac)，mps 后端
                self.device, dtype = "mps", torch.float16
            else:
                # cpu + float32：纯 CPU 运行，速度很慢，只适合 demo
                self.device, dtype = "cpu", torch.float32
        else:
            # 如果手动指定 device（比如手动传cpu），直接使用，dtype 固定 float32
            self.device, dtype = device, torch.float32

        print(f"🚀 初始化 Qwen3-VL-Embedding (设备: {self.device})")

        # model_path 可能是模型文件本地路径，也可能是 HuggingFace 模型名
        # 调用方没传路径，需要自动找到模型文件
        if model_path is None:

            # 如果装了 ModelScope（国内魔搭社区）
            if MODELSCOPE_AVAILABLE:
                # 从魔搭下载"qwen/Qwen3-VL-Embedding-2B"到本地 backend/data/models/
                model_path = download_model_from_modelscope(EMBEDDING_MODEL_MS, MODEL_CACHE_DIR)
                # 此时 model_path = "backend/data/models/qwen3-vl-embedding/"
            # 没有 modelscope → 直接使用 HuggingFace 模型 ID，从 hf 加载
            else:
                # modelscope 不可用，直接使用 HuggingFace 模型 ID，transformers 会在线拉取
                model_path = "Qwen/Qwen3-VL-Embedding-2B"
        # model_path 现在的值有两种可能性:
        # 1.本地磁盘路径：/xxx/model_cache/Qwen3‑VL‑Embedding‑2B
        # 2.模型 ID 字符串："Qwen/Qwen3‑VL‑Embedding‑2B"

        # AutoProcessor 来自 transformers,.from_pretrained(model_path) 读取模型目录下的配置文件，创建文本 + 图片处理器
        # - 如果 model_path 是本地文件夹路径：直接读取文件夹里面的配置文件 processor_config.json，不联网。
        # - 如果 model_path 是模型 ID 字符串：去网络（modelscope/huggingface）下载 processor 配置，缓存到本地
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # Qwen 系列模型，没有用 transformers 官方内置的 processor，自己写了一套自定义处理逻辑代码，放在模型仓库里面,trust_remote_code=True：允许执行模型仓库里面的自定义 Python 代码，构建 Processor

        # AutoModel.from_pretrained() —— 加载神经网络权重，AutoModel 是一个抽象类，具体使用哪个模型，由 model_path 决定
        # 返回：一个 PyTorch 的模型对象（模型实例），可以调用 forward() 方法，输入数据，得到输出
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None
            # device_map="auto"：只有 cuda 的时候开启，自动把模型分层分配到 GPU 显存；CPU/MPS 不使用 device_map，所以赋值 None
        )

        # 非 cuda 设备手动迁移模型
        # 原因：device_map="auto"仅支持 cuda。如果是 mps 或者 cpu，需要手动.to()把模型放到对应设备上  
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        # 模型切换为推理模式
        # 关闭 dropout、batchnorm 训练相关逻辑，推理更快，结果稳定
        # 做向量检索必须加 eval ()，不然每次向量结果会随机波动
        self.model.eval()

    # _get_embedding_from_model(前向推理)：统一处理文本 / 图文多模态消息，送入 Qwen3-VL-Emb 模型，输出归一化向量 (numpy 数组)
    # 输入：之前 encode_text/encode_image 构造的消息列表，格式遵循 Qwen 的对话消息规范
    # 返回：np.ndarray 一维 numpy 浮点数组，L2 归一化后的向量（给 Milvus 存 / 检索用）
    def _get_embedding_from_model(self, messages: list) -> np.ndarray:

        # apply_chat_template：Qwen 系列 processor 内置方法，把 messages 对话列表拼接成模型认识的完整字符串 prompt
        # - tokenize=False：只输出字符串，不转 token 编号（后面 processor 再统一做 tokenize）
        # - add_generation_prompt=False：不要追加 “assistant:” 生成提示！因为这是向量嵌入模型，不是对话生成模型，不需要让模型 “接着回答”，所以关掉生成 prompt。如果打开，向量效果会变差
        # 输出text：拼接完成的 prompt 字符串
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )

        # process_vision_info 是 Qwen‑VL 工具函数
        # - 遍历 messages，提取里面所有图片对象，做图片预处理（缩放、转 RGB 等）
        # - 返回：图片列表image_inputs、视频列表video_inputs，这里项目只用图片，video 一般是空。如果是纯文本检索，image_inputs = []
        image_inputs, video_inputs = self.process_vision_info(messages)

        # processor 干两件大事：对文本 + 图片统一编码，生成模型输入张量
        # inputs 是字典，包含 input_ids、attention_mask、pixel_values 这些模型需要的张量
        inputs = self.processor(
            text=[text],               # 传入上面拼接好的字符串，转成 token id 张量；套列表变成 batch
            images=image_inputs,       # 传入提取好的图片，图片做 resize、归一化、转张量
            videos=video_inputs,       # 
            return_tensors="pt",       # 输出 PyTorch tensor 张量
            padding=True               # 自动补 padding，保证 batch 内长度对齐
        ).to(self.device) 
        # .to(self.device)：把所有张量（input_ids、pixel_values 等）搬运到 cuda/mps/cpu，和模型在同一个设备，必须加这一步！否则会出现 “数据和模型不在同一个设备” 的报错，导致推理失败

        # PyTorch 默认会记录每一步计算，为反向传播（训练）做准备，但推理阶段不需要反向传播、不需要算梯度；关闭节省显存、速度更快
        # torch.no_grad()：就是告诉 PyTorch："下面这段代码只是推理，别记了，省点资源。"
        with torch.no_grad():

            # 把输入字典解包，送入模型前向推理
            # - output_hidden_states=True：要求模型返回每一层网络的隐状态；不加这个，hidden_states为 None 拿不到数据，普通对话模型拿 logits；Embedding 模型拿最后一层 hidden_states 特征
            outputs = self.model(**inputs, output_hidden_states=True)
            # outputs：模型输出对象，里面存放各类张量，类型是模型专属输出结构体

            # .hidden_states：元组，存每一层输出张量；hidden_states[-1]取神经网络 transformer 最后一层输出 shape：[1, 80, 1024] → [batch, seq_len, hidden_size]
            # .mean(dim=1)：在序列维度求均值，把整段图文所有 token 向量压缩成单个向量 dim=1 代表 seq_len 维度，把所有 token 做平均池化，
            # shape 变成：[1,1024]，把一整串 token 压缩成 1 条向量
            # .squeeze()：消除多余的 batch 维度，去掉 shape 里的 1，从[1, hidden_dim]变成[hidden_dim]一维向量，
            # shape 变成：[1024]，一维 PyTorch 张量
            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()
            # 注意：此时embedding仍然是PyTorch tensor，还不是 numpy 数组，还在 GPU 显存上

        # 链式调用，一步一步转换：PyTorch tensor → numpy ndarray → 32 位浮点数
        # .cpu()：张量从 GPU 显存复制到系统内存 CPU；GPU 上的 tensor 不能直接转 numpy。
        # .numpy()：PyTorch tensor → numpy ndarray（Milvus 需要 numpy 格式向量）
        # .astype(np.float32)：强制转为 float32。Milvus 向量库只接受 float32；bfloat16、float16 存进去直接报错
        embedding = embedding.cpu().numpy().astype(np.float32)

        # 求向量 L2 范数（向量模长），再 L2 归一化
        # Milvus 做向量相似度检索（内积）**必须归一化！** 归一化后内积等价余弦相似度。判断`norm>0`：防止全零向量除以 0 报错
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # 模型原生输出 1024 维大于我们设定的`EMBEDDING_DIM=512`，取前 512 个元素，截断。
        # shape 由(1024,) → (512,)，和 Milvus 建库时定义向量维度对齐
        if self.output_dim and len(embedding) > self.output_dim:
            embedding = embedding[:self.output_dim]

        # 返回最终归一化后的固定维度向量，回到 encode_text / encode_image，最后存 Milvus 或者检索
        return embedding

    # encode_text(文本嵌入)：把单条文本编码为 L2 归一化的 512 维向量。
    # 调用方:文搜图时由 Qwen3VLMultiModalEmbedding._get_text_embedding 调用
    def encode_text(self, text: str) -> np.ndarray:
        # 构造 messages
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

        # 把拼装好的 messages 丢给我们刚刚精读的私有函数，
        # 走完整套 template、processor、模型前向、池化、归一化、截断逻辑，直接返回最终 numpy 向量
        return self._get_embedding_from_model(messages)

    # encode_image(图片嵌入)：把单张图片编码为 L2 归一化 512 维图片向量
    # 调用方:图搜图时由 Qwen3VLMultiModalEmbedding._get_image_embedding 调用
    def encode_image(self, image_path: str) -> np.ndarray:

        # 方法内部局部导入：延迟导入
        from PIL import Image

        # Image.open(image_path)：从磁盘打开图片文件，得到 PIL.Image 对象；支持 jpg/png 等格式
        # .convert("RGB") 强制转为 RGB 三通道 ，防止 png 透明通道（RGBA4 通道）、灰度单通道图片导致模型预处理报错
        image = Image.open(image_path).convert("RGB")  

        # 构造 messages
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."}
            ]
        }]
        # 同样交给底层私有函数，走完整一套流程，返回图片向量
        return self._get_embedding_from_model(messages)


# Qwen3VLMultiModalEmbedding(LlamaIndex 适配器):把 Qwen3VLEmbedding 方法 encode_text/encode_image 包装成 LlamaIndex 标准 MultiModalEmbedding 接口，之后就能被 LlamaIndex 框架无缝使用，这个类本身没有任何逻辑，所有方法都是"转发"
# Llama‑Index 要求多模态嵌入必须继承框架提供的抽象基类 MultiModalEmbedding，实现框架规定的接口方法，框架才会自动调用它去生成文本 / 图片向量
# MultiModalEmbedding：Llama‑Index 的抽象基类（ABC），它声明了"子类必须实现 _get_text_embedding、_get_image_embedding 等接口"，继承它之后，Llama‑Index 的 MultiModalVectorStoreIndex、检索器就可以把这个类当做标准多模态 Embedding 组件使用
class Qwen3VLMultiModalEmbedding(MultiModalEmbedding):
    """Qwen3-VL 的 LlamaIndex 多模态适配器"""

    # model_config：Pydantic 配置（LlamaIndex 底层基于 Pydantic v2），允许属性存放自定义 Python 对象
    # - arbitrary_types_allowed = True：允许 Pydantic 模型里存放任意自定义对象类型，否则实例化直接抛 Pydantic 校验异常，Pydantic 默认只认 int/str/列表这些，不认自建类
    # - extra="allow"：额外属性直接放行，不校验报错
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
    # 坑点：不加这一行，self.qwen3_embedder = xxx 会直接报错，很多人写自定义 llama‑index embedding 踩这个坑

    # 构造函数：
    # 入参：qwen3_embedder，传入我们已经实例化完成的 Qwen3VLEmbedding 对象（已经加载好模型、processor），注意：不是在适配器内部新建模型，而是外部实例好传进来，做到模型全局单例，避免重复加载 2B 模型、重复占用显存
    def __init__(self, qwen3_embedder: Qwen3VLEmbedding):

        # 调用父类 MultiModalEmbedding 的构造函数
        # - embed_batch_size=1：批次大小设置 1；我们 Qwen3VLEmbedding 没有实现批量编码，只支持单条文本 / 单张图片，所以 batch 强制 1
        # - model_name：给 Llama‑Index 内部标记模型名字，日志、元数据会用到，业务逻辑不影响
        super().__init__(
            embed_batch_size=1, 
            model_name="Qwen3-VL-Embedding-2B"
        )

        # ❗为什么不直接写 self.qwen3_embedder = qwen3_embedder？
        # 因为父类是 Pydantic Model，普通 self.xxx = value 会走 Pydantic 字段校验逻辑。
        # object.__setattr__() 绕开 Pydantic 的字段校验机制，强行挂载我们的自定义 embedder 实例
        # 等价底层原生对象赋值，避开 Pydantic 拦截。这是 Pydantic 自定义组件非常经典写法
        object.__setattr__(
            self, 
            'qwen3_embedder', 
            qwen3_embedder
        )

    # _get_text_embedding(文本嵌入):LlamaIndex 标准接口实现,转发给 qwen3_embedder，获取文本向量
    def _get_text_embedding(self, text: str) -> List[float]:
        return self.qwen3_embedder.encode_text(text).tolist()
        # .tolist()：numpy 数组 → Python list [float]

    # _get_query_embedding(查询嵌入):LlamaIndex 把 query 与 document embedding 分开接口,
    # 这里与 _get_text_embedding 一致(Qwen3-VL 不区分 query/document)
    def _get_query_embedding(self, query: str) -> List[float]:
        return self.qwen3_embedder.encode_text(query).tolist()

    # _get_image_embedding(图片嵌入):LlamaIndex 多模态接口,转发给 qwen3_embedder.encode_image，查询文本的向量
    def _get_image_embedding(self, img_file_path: str) -> List[float]:
        return self.qwen3_embedder.encode_image(img_file_path).tolist()


    # 异步版本方法（aget前缀 = async get）
    # LlamaIndex 支持异步调用，它会优先调用带a开头的异步接口：当前代码只是简单包装，并没有真正异步
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_image_embedding(self, img_file_path: str) -> List[float]:
        return self._get_image_embedding(img_file_path)


# Qwen3VLNodePostprocessor(Llama‑Index 后置处理器（Reranker 精排组件）)：封装 Qwen3-VL-Reranker-2B,LlamaIndex NodePostprocessor 接口实现。
# 作用：向量检索 + BM25 召回一批候选图片之后，调用 Qwen3-VL-Reranker-2B 多模态重排模型，同时接收【文本 query / 图片 query】，对召回结果做二次打分重排，把真正相关的图片排到前面。
# 前置链路：Milvus 向量库 + BM25 融合检索拿到粗召回`List[NodeWithScore]` → 送入本 Postprocessor → 更新每个 node 的 score，按分数降序，截取 top‑n 返回
class Qwen3VLNodePostprocessor(BaseNodePostprocessor):
    """Qwen3-VL-Reranker-2B 的 LlamaIndex NodePostprocessor 实现"""

    # Pydantic 声明字段
    # 默认返回前 5 条精排结果
    top_n: int = Field(default=5, description="Number of top results to return")
    # 用来保存Reranker 模型实例，类型标注 Any（因为是自定义模型）
    model: Any = Field(default=None, description="Reranker model instance")

    # __init__ 双路径:
    #   1.model_instance 不传 -- Qwen3VLNodePostprocessor() → 加载权重文件、实例化底层 Reranker 模型，只做一次，放到引擎全局变量self.reranker_model缓存，全局共享这个模型，避免重复加载、占用显存(首次创建时走这条)
    #   2.model_instance 传入 -- Qwen3VLNodePostprocessor(top_n=recall_top_k, model_instance=self.reranker_model) → 新建一个 LlamaIndex 的后处理器实例，直接复用上面已经加载好的模型，不再重新加载权重，专门交给 QueryEngine 做检索后的精排
    def __init__(self, model_path: str = None, top_n: int = 5, model_instance: Any = None, **kwargs):
        super().__init__(top_n=top_n, **kwargs)

        # 如果外部提前加载好了 reranker 模型实例传进来，直接挂载到self.model，不用重新加载
        # 用处：开发调试时可以复用已经加载好的模型，避免重复加载大模型，节省显存
        if model_instance:
            object.__setattr__(self, 'model', model_instance)
            return

        print(f"\n🔧 加载 Qwen3-VL-Reranker-2B")

        # 如果没有传入模型本地路径
        if model_path is None:
            # 优先从 modelscope 自动下载模型到缓存目录
            if MODELSCOPE_AVAILABLE:
                model_path = download_model_from_modelscope(RERANKER_MODEL_MS, MODEL_CACHE_DIR)
            # 不能用 modelscope 就直接用 hf 模型 id 字符串
            else:
                model_path = "qwen/Qwen3-VL-Reranker-2B"

        # 导入两个标准库：
        # - `sys`：操作 Python 模块全局注册表`sys.modules`
        # - `importlib.util`：Python 官方动态导入模块工具，用来从任意文件路径加载`.py`脚本
        import sys, importlib.util

        # 拼接路径：在模型目录下找 scripts/qwen3_vl_reranker.py，这个文件里面定义了官方Qwen3VLReranker类
        script_path = Path(model_path) / "scripts" / "qwen3_vl_reranker.py"

        # 兜底打印警告：就算脚本还是找不到，只打印警告，程序不直接崩溃（后面执行 importlib 的时候才会报错）
        if not script_path.exists():
            # Fallback detection if scripts folder is structure differently after download
            print(f"⚠️ Reranker script not found at {script_path}, checking root...")

        # 动态 import:把 scripts/qwen3_vl_reranker.py 加载为模块
        # 从磁盘上任意路径的 .py 文件，动态加载里面的类，等价于 from xxx import Qwen3VLReranker，但是文件不在 Python 默认搜索路径下，只能用这套方案
        # spec = importlib.util.spec_from_file_location(模块名称, 文件绝对路径)
        # - 第一个参数 `"qwen3_vl_reranker_script"`：自定义模块名，随便起名，用来在 Python 内部标识这个模块。不是文件名，只是一个名字
        # - 第二个参数 `script_path`：`qwen3_vl_reranker.py` 的完整本地路径（Path 对象）
        # - 返回值 `spec`：模块规格对象，里面保存了这个 py 文件在哪里、用什么加载器读取文件
        spec = importlib.util.spec_from_file_location("qwen3_vl_reranker_script", script_path)

        # 根据刚才的 spec，创建一个空的 Python 模块对象，等待灌入代码
        reranker_module = importlib.util.module_from_spec(spec)

        # 注册到 `sys.modules`
        # sys.modules 是 Python 全局字典，缓存所有已经导入成功的模块，后面代码如果再次写 `import qwen3_vl_reranker_script`，Python 会直接从`sys.modules`拿现成模块，不会重新读取、执行 py 文件
        sys.modules["qwen3_vl_reranker_script"] = reranker_module

        # 这是整套逻辑最关键一步，加载器读取磁盘上 qwen3_vl_reranker.py 的全部源码并执行，执行完成后，`reranker_module` 模块里面就有了 `Qwen3VLReranker` 这个类
        spec.loader.exec_module(reranker_module)

        # 从动态加载进来的 py 文件里拿到 Reranker类，然后 new 出这个模型实例，再强行挂到 self 对象上
        OfficialReranker = reranker_module.Qwen3VLReranker
        object.__setattr__(self, 'model', OfficialReranker(model_name_or_path=model_path))

    # 类方法，属于类本身，不是实例方法，不需要实例化就可以调用
    # LlamaIndex 框架规范：给组件注册名字，序列化 / 日志 / 配置读取的时候使用，用来标识这个后处理器
    # 作用：框架内部可以通过字符串 "Qwen3VLNodePostprocessor" 找到这个类，LlamaIndex 组件的标准写法，业务逻辑无关，属于框架约定
    @classmethod
    def class_name(cls) -> str:
        return "Qwen3VLNodePostprocessor"

    # _postprocess_nodes(节点精排):LlamaIndex 标准接口,对召回的 nodes 用 Reranker 重新打分排序。
    # 支持三种 query 模式:纯文本 / 纯图片 / 图文混合,根据 query_bundle 内容动态构造
    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]:
        """支持文本和图片query的精排"""

        # 防御性判断，兜底保护，
        # 如果候选节点为空，或者用户查询为空，直接返回前 top_n 条，不执行精排，避免后面代码遍历空列表而报错
        if not nodes or query_bundle is None:
            return nodes[:self.top_n]

        # 构造 documents:从每个 node 提取图片路径
        # 兼容两种 node 结构:image_path 属性 / file_path in metadata
        documents = []

        for node in nodes:
            # 尝试从node里面拿到图片路径（多模态节点，存图片）
            if hasattr(node.node, 'image_path') and node.node.image_path:
                # 情况1：node对象直接自带image_path属性
                image_path = node.node.image_path
            elif 'file_path' in node.node.metadata:
                # 情况2：图片路径存在node的metadata元数据字典里面
                image_path = node.node.metadata['file_path']
            else:
                # 这个节点拿不到图片，跳过，不送入reranker
                continue
            # 转成绝对路径，放进documents列表，给reranker使用
            # 最终`documents`是一个列表，里面每个元素是字典：`{"image": "xxx/xxx.png"}`，就是给 reranker 的候选图库
            documents.append({"image": str(Path(image_path).resolve())})

        # 如果遍历完，没有提取到任何图片，直接返回原来的前 top_n 结果，不再走 reranker
        if not documents:
            return nodes[:self.top_n]

        # 构造query字典,支持文本和图片
        query_dict = {}

        # 尝试从query_bundle拿到用户查询图片（存储在metadata中）
        # 三种 query 来源优先级:image_path 属性 > custom_embedding_strs > query_str
        if hasattr(query_bundle, 'image_path') and query_bundle.image_path:
            # 情况1：query_bundle直接挂载image_path（图搜场景：用户上传图片检索），直接使用
            query_dict["image"] = str(Path(query_bundle.image_path).resolve())
        elif hasattr(query_bundle, 'custom_embedding_strs') and query_bundle.custom_embedding_strs:
            # 情况2：多模态查询，图片放在custom_embedding_strs（LlamaIndex多模态常用方式）
            # [补充] 图搜图场景:retrieval 把图片路径塞进 custom_embedding_strs 而非 query_str
            for item in query_bundle.custom_embedding_strs:
                if isinstance(item, str) and Path(item).exists() and Path(item).suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
                    query_dict["image"] = str(Path(item).resolve())
                    break

        # 取出用户的文本查询
        if query_bundle.query_str:
            query_dict["text"] = query_bundle.query_str

        # 这段：组装用户的查询信息，支持 2 种检索模式
        # - 文搜图：用户只输入文字，query_dict={"text": "描述文字"}
        # - 图搜图：用户上传图片，可以附带文字，query_dict={"image":"xxx.png", "text":"描述"}

        # 极端情况：既没有查询文本，也没有查询图片，直接返回前 top_n 个 node(不精排)
        if not query_dict:
            return nodes[:self.top_n]

        # 把用户 query + 候选图片列表，打包成 Reranker 输入格式(官方约定):
        inputs = {
            # instruction: 任务说明 / query: 含 text 或 image / documents: 待打分的图片列表 / fps: 帧率(视频用)
            "instruction": "Retrieve images or text relevant to the user's query.",
            "query": query_dict,
            "documents": documents,
            "fps": 1.0
        }

        # 调用 reranker 的推理函数，一次性给所有候选图片打分 :scores 是与 documents 等长的分数列表
        scores = self.model.process(inputs)

        # 把 reranker 算出来的新分数，覆盖掉原来向量粗召回的相似度分数
        for i, score in enumerate(scores):
            nodes[i].score = float(score)

        # 按分数降序排序,取前 top_n
        nodes.sort(key=lambda x: x.score or 0.0, reverse=True)

        # 截取前top_n条，返回给 LlamaIndex 框架，完成精排
        return nodes[:self.top_n]

# ============================================================
# Core Retrieval Engine
# ============================================================

# Qwen3VLRetrievalEngine(检索引擎单例，保证全项目从头到尾只有 1 个检索引擎对象，Embedding 模型、Reranker、Milvus 索引只加载一次，不重复占用显存和内存)：项目核心,封装双索引系统与统一检索入口。
# 设计要点:
#  1. 单例模式:__new__ 拦截类创建,首次设 initialized=False,后续返回同一实例
#  2. 双索引系统:_build_indices 构建 [1] 图片向量索引 + [2] 混合检索索引
#  3. 共享 Reranker:initialize 时加载一次,后续每次检索创建轻量 postprocessor 复用
# 调用方:routers/search.py、core/agent.py 的 search_images 工具
class Qwen3VLRetrievalEngine:
    """封装多模态检索逻辑"""

    # 类静态变量（属于类，不属于实例），用来保存唯一的实例
    _instance = None

    # __new__ 的职责：分配内存，创建空对象外壳；__init__ 是给这个外壳填充属性
    def __new__(cls):
        # 判断全局有没有已经建好的实例
        if cls._instance is None:
            # super(...).__new__(cls)：调用父类object的__new__，
            # 在内存新建一个空的引擎对象存入类静态变量 `_instance` 永久保存
            cls._instance = super(Qwen3VLRetrievalEngine, cls).__new__(cls)
            # 给空实例打上标记：对象外壳建好，但是模型 / 索引资源还没加载
            cls._instance.initialized = False

        # 第二次、第三次再调用Qwen3VLRetrievalEngine()：_instance不为空，直接返回之前创建好的对象，不会新建内存
        return cls._instance

    # 实例初始化，填充成员变量
    def __init__(self):

        # 幂等保护:已初始化的实例不再重置字段
        if self.initialized:
            return
        
        # 多模态Embedding模型实例，用于图片/文本向量化
        self.embedder = None
        # LlamaIndex适配包装器，把原生embedder转为框架可用的嵌入模型
        self.embed_adapter = None

        # 双索引系统
        self.multimodal_index = None  # 多模态向量索引（底层Milvus），图搜图专用
        self.qwen_index = None        # Qwen向量索引，文搜图混合检索
        self.caption_index = None     # 图片caption文本的向量索引，文搜图混合检索
        self.bm25_retriever = None    # BM25文本关键词检索器，用于关键词召回，文搜图混合检索

        # Reranker精排模型实例
        self.reranker_model = None    
        # 内存缓存字典：存放图片描述caption，减少磁盘IO
        self.caption_cache = {}

    # initialize(初始化引擎)：启动时调用一次,加载所有组件。
    # 步骤:加载 embedder → 设置全局 Settings → 加载 caption → 读取图片 → 构建双索引 → 加载 Reranker
    # 调用方:main.py 的 lifespan 启动钩子;routers/search.py 懒初始化兜底
    def initialize(self, image_dir: Path, use_reranker: bool = True):
        """初始化主要组件"""

        # 保护判断：已经初始化完成，不再重复执行
        if self.initialized:
            print("⚠️ Engine already initialized, skipping.")
            return

        print("\n⚙️  初始化检索引擎...")

        # 实例化Qwen3-VL Embedding模型，设置向量输出维度
        self.embedder = Qwen3VLEmbedding(output_dim=EMBEDDING_DIM)
        # 套上LlamaIndex适配器，适配框架接口
        self.embed_adapter = Qwen3VLMultiModalEmbedding(self.embedder)

        # LlamaIndex全局配置：指定整个项目使用这个多模态Embedding，防止 LlamaIndex fallback 到 OpenAI
        # 关键:不设这行,LlamaIndex 在 RetrieverQueryEngine.query 等路径会调默认 OpenAI embedding,
        # 消耗 API 额度且维度不匹配(Qwen3-VL 512 维 vs OpenAI text-embedding 1536 维)
        Settings.embed_model = self.embed_adapter
        print("   ✅ 全局 embed_model 已设置为 Qwen3-VL (本地模型)")

        # 加载本地所有图片caption文本，存入内存缓存
        self.caption_cache = load_caption_cache()
        print(f"   加载 {len(self.caption_cache)} 条 Caption 缓存")

        # LlamaIndex工具：读取image_dir目录下的图片，转成Document文档对象
        documents = SimpleDirectoryReader(
            input_dir=str(image_dir),
            required_exts=[".png", ".jpg", ".jpeg", ".gif"]
        ).load_data()
        print(f"   加载 {len(documents)} 张图片")

        # 4. 构建双索引系统：根据图片文档，构建Milvus向量索引、caption索引、BM25索引
        self._build_indices(documents, image_dir)

        # 判断是否启用Reranker精排模型，加载 Reranker Model (Shared)
        if use_reranker:

            print(f"\n[3/3] 加载 Reranker 模型 (Shared)...")
            # 临时实例化Postprocessor，内部会加载Reranker权重
            temp_processor = Qwen3VLNodePostprocessor()
            # 挂载到引擎实现全局复用
            self.reranker_model = temp_processor.model
            print("      ✅ Reranker 模型加载完成")

        # 标记引擎初始化全部完成，后续不会再重复初始化
        self.initialized = True
        print("✅ 检索引擎就绪")

    # _build_indices(构建双索引):项目核心架构决策，构建两套Milvus向量索引 + BM25关键词检索器，实现双索引多路召回
    # [1] 图片向量索引:仅图片 embedding,给图搜图用(Milvus collection=qwen3_vl_image_only)
    # [2] 混合检索索引:Qwen3-VL 向量 + BM25,给文搜图三路召回融合用
    # 分开的原因:图搜图只需图片向量比对，文搜图需要跨模态+关键词双路召回，数据结构需求不同
    def _build_indices(self, documents, image_dir):

        print("   构建双索引系统 (统一512维)...")

        print("   [1/2] 图片向量索引（图搜图）...")

        # 扫描图库目录，获取所有图片路径 png/jpg/jpeg
        image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
        print(f"      📊 待处理图片数: {len(image_paths)}")

        # 使用 ImageNode（避免文本 embedding）
        # 关键:ImageNode 只含 image_path 不含 text，LlamaIndex 不会为它生成 text embedding
        image_nodes = []

        # 循环图片，构造 LlamaIndex 的 ImageNode 多模态节点
        for img_path in image_paths:
            node = ImageNode(
                image_path=str(img_path),
                metadata={"file_path": str(img_path), "file_name": img_path.name}
            )
            image_nodes.append(node)


        # 实例化Milvus向量库，创建集合【qwen3_vl_image_only】专门用于图搜图
        print("      ⏳ 正在创建 Milvus 存储...")
        image_store = MilvusVectorStore(
            uri=MILVUS_URI,
            collection_name="qwen3_vl_image_only",     # 集合名：纯图片检索用
            dim=EMBEDDING_DIM,                         # 向量维度 512，和Qwen3-VL embedding输出对齐
            overwrite=True,                            # 重建索引时覆盖旧集合（reindex的时候会清空重建）
            similarity_metric="IP"                     # 内积相似度，归一化向量等价余弦相似度
        )
        print("      ✅ Milvus 存储创建完成")


        print("      ⏳ 正在向量化图片（仅图片，无文本）...")

        # 这一步耗时与图片数成正比(GPU 1-3s/张,CPU 30-60s/张)
        import time
        start_time = time.time()

        # StorageContext：LlamaIndex存储上下文，绑定向量库
        storage_context = StorageContext.from_defaults(vector_store=image_store)

        # 构建向量索引：送入图片节点，使用我们的Qwen多模态embedding做向量化
        self.multimodal_index = VectorStoreIndex(
            nodes=image_nodes,
            storage_context=storage_context,
            embed_model=self.embed_adapter,      # 使用前面初始化好的Qwen3VL适配器
            show_progress=True                   # 终端显示进度条
        )

        elapsed = time.time() - start_time
        print(f"      ✅ 图搜图索引完成 (耗时: {elapsed:.1f}秒)")


        print("   [2/2] 混合检索索引（文搜图三路召回）...")

        # 构建 ImageNode(与 [1] 不同:这里 text 字段含文件名+caption,给 BM25 用)
        nodes = []

        # 再次扫描图片，用于构建文搜图的混合索引
        image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

        for image_path in image_paths:
            # 从内存caption_cache读取图片描述文本（之前load_caption_cache加载的）
            caption = self.caption_cache.get(image_path.stem, "")
            # 拼接文本：文件名 + 图片caption描述；没有caption就只用文件名
            text = f"{image_path.name} {caption}" if caption else image_path.name
            # 构造ImageNode，同时携带【文本描述】+【图片路径】，用于文本查询召回图片
            node = ImageNode(
                text=text,
                image_path=str(image_path),
                metadata={"file_name": image_path.name, "file_path": str(image_path)}
            )
            nodes.append(node)

        # 创建第二个Milvus集合【qwen3_vl_hybrid_agent】，用于文搜图
        qwen_store = MilvusVectorStore(
            uri=MILVUS_URI,
            collection_name="qwen3_vl_hybrid_agent",
            dim=EMBEDDING_DIM,
            overwrite=True,
            similarity_metric="IP"
        )

        # 构建文搜图向量索引
        self.qwen_index = VectorStoreIndex(
            nodes=nodes,
            storage_context=StorageContext.from_defaults(vector_store=qwen_store),
            embed_model=self.embed_adapter
        )

        # 路径2: Caption 文本向量 [已禁用 - 太慢]
        # 原因：Qwen3-VL 处理 37622 个文本块需要 5+ 小时
        # 替代：使用 BM25 进行文本关键词搜索
        # [补充] 设计权衡:caption 向量索引虽能提升语义检索效果,
        # 但首次构建耗时 5+ 小时不可接受,改用 BM25 关键词检索替代,性能与效果平衡更好
        """
        caption_nodes = [node for node in nodes if self.caption_cache.get(Path(node.image_path).stem)]
        if caption_nodes:
            caption_store = MilvusVectorStore(
                uri=MILVUS_URI,
                collection_name="qwen3_vl_caption_agent",
                dim=EMBEDDING_DIM,
                overwrite=True,
                similarity_metric="IP"
            )
            self.caption_index = VectorStoreIndex(
                caption_nodes,
                storage_context=StorageContext.from_defaults(vector_store=caption_store),
                embed_model=self.embed_adapter
            )
        """

        # 关闭caption独立索引，不启用
        self.caption_index = None

        # 路径3: BM25（文本关键词搜索）
        # 构建BM25关键词检索器：纯文本关键词召回，和向量检索做混合检索
        self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=30)

        print("      ✅ 混合检索索引完成")

    # search(统一检索入口):根据 mode 分发到对应检索方法。
    # 调用方:routers/search.py 与 core/agent.py 的 search_images 工具
    def search(self, mode: str, query: str = None, image_path: Path = None,recall_top_k: int = 30, rerank_top_k: int = 5, score_threshold: float = 0.0) -> List[Dict]:
        """统一检索入口"""

        # 校验引擎状态，如果还没初始化，直接抛出运行时异常
        if not self.initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # 根据mode检索模式，分发路由到对应的检索函数
        if mode == '文搜图':
            # 文搜图：传入文本query，调用文本检索函数
            return self.text_to_image_search(query, 
                                             recall_top_k, 
                                             rerank_top_k, 
                                             score_threshold
                                             )
        elif mode == '图搜图':
            # 图搜图：必须提供图片路径，没有图片直接抛参数异常
            if not image_path: raise ValueError("Image path required for image search")
            return self.image_to_image_search(str(image_path), 
                                              query=None,
                                              recall_top_k=recall_top_k, 
                                              rerank_top_k=rerank_top_k, 
                                              score_threshold=score_threshold
                                              )
        elif mode == '混合搜索':
            # 混合搜索：上传图片 + 附加文本描述，一起作为检索条件
            if not image_path: raise ValueError("Image path required for hybrid search")
            return self.image_to_image_search(str(image_path), query=query,
                                              recall_top_k=recall_top_k, 
                                              rerank_top_k=rerank_top_k, 
                                              score_threshold=score_threshold
                                              )
        else:
            # 传入未知mode，抛出参数错误
            raise ValueError(f"Unknown mode: {mode}")

    # text_to_image_search(文搜图):多路召回 → 结果融合 → Reranker 精排（可选） → 分数过滤 → 截断 → 格式化返回
    # 检索链路:Qwen3-VL 向量召回 + BM25 召回 → QueryFusionRetriever RRF 融合 → Reranker 全量精排 → 阈值过滤 → 截断
    def text_to_image_search(self, 
                             query: str, 
                             recall_top_k: int = 30, 
                             rerank_top_k: int = 5, 
                             score_threshold: float = 0.0
                            ) -> List[Dict]:
        
        print(f"🔍 文搜图: {query} (Recall={recall_top_k}, Rerank={rerank_top_k}, Thresh={score_threshold})")
        
        # 组装多个检索器：qwen向量索引检索器 + BM25关键词检索器
        retrievers = [self.qwen_index.as_retriever(similarity_top_k=recall_top_k), self.bm25_retriever]

        # 如果开启了caption独立索引，就把caption检索器也加入进来（当前项目注释掉了caption_index，这里不会执行）
        if self.caption_index:
            retrievers.append(self.caption_index.as_retriever(similarity_top_k=recall_top_k))

        # 多路检索融合器：把上面多个检索器的结果融合，使用 reciprocal_rerank 倒数排名融合算法
        fusion_retriever = QueryFusionRetriever(
            retrievers=retrievers,              # 传入一组检索器
            similarity_top_k=recall_top_k,      # 每个检索器最多召回30条候选
            num_queries=1,                      # 不做查询扩展，只使用原始query
            mode="reciprocal_rerank",           # 融合算法：倒数排名融合，综合多个检索器的排名打分
            use_async=False                     # 关闭异步检索，同步执行
        )

        # 判断是否加载了Reranker精排模型
        if self.reranker_model:

            # 这里的其实是使用了两次 Qwen3VLNodePostprocessor()
            # 实例化精排处理器，复用引擎全局已经加载好的reranker模型
            reranker = Qwen3VLNodePostprocessor(top_n=recall_top_k, model_instance=self.reranker_model)

            # 构建查询引擎：融合检索器 + 后置精排处理器
            # query(query) 会执行:retriever 召回 → postprocessors 精排 → 返回 response
            query_engine = RetrieverQueryEngine(
                retriever=fusion_retriever,
                node_postprocessors=[reranker]
            )

            # 执行检索+精排，输入查询文本
            response = query_engine.query(query)

            # 取出检索得到的节点（图片节点）
            results = response.source_nodes
        else:
            # 无 Reranker 时，不走精排，直接拿召回结果
            results = fusion_retriever.retrieve(query)[:rerank_top_k]

        # 过滤：剔除分数低于阈值的候选结果
        filtered_results = [r for r in results if (r.score or 0.0) >= score_threshold]

        # 截断，保留指定数量的最终结果
        final_results = filtered_results[:rerank_top_k]

        print(f"   📊 最终结果: {len(final_results)} (过滤前: {len(results)})")

        # 格式化ImageNode节点，转为字典数组，给前端接口返回
        return self._format_results(final_results)

    # image_to_image_search(图搜图/混合搜):图片 embedding + Milvus 检索 + Reranker 精排。
    # 检索链路:query 图 embedding → multimodal_index 检索 → 过滤自身 → Reranker 精排(图文混合或纯图) → 阈值过滤
    def image_to_image_search(self, 
                              image_path: str, 
                              query: str = None,
                              recall_top_k: int = 30, 
                              rerank_top_k: int = 5, 
                              score_threshold: float = 0.0
                              ) -> List[Dict]:

        # 把传入的图片路径字符串转为Path对象，方便后续路径对比
        query_path = Path(image_path)
        print(f"🖼️ 图搜图: {query_path.name} (Query='{query}', Recall={recall_top_k}, Rerank={rerank_top_k}, Thresh={score_threshold})")

        # ========= 1. 提取查询图片的向量 =========
        # 使用多模态Embedding模型，对参考图片编码，得到图片向量
        query_embedding = self.embed_adapter._get_image_embedding(str(query_path))

        # ========= 2. 动态决定粗召回数量 =========
        # 如果有reranker精排模型：粗召回取 recall_top_k（比如30条，给精排模型挑选）
        # 如果没有reranker：粗召回直接少拿一点，只需要比最终返回数多1个即可，节省算力
        top_k_retrieve = recall_top_k if self.reranker_model else (rerank_top_k + 1)

        # 使用multimodal_index（Milvus多模态向量索引）构建检索器
        retriever = self.multimodal_index.as_retriever(similarity_top_k=top_k_retrieve)

        # 构造QueryBundle：LlamaIndex封装查询的载体，这里传入图片向量，query_str填空字符串
        query_bundle = QueryBundle(query_str="", embedding=query_embedding)

        # 执行向量粗召回，拿到一批相似图片节点
        results = retriever.retrieve(query_bundle)

        # ========= 【重要逻辑】过滤：把上传的原图从结果中删掉 =========
        # 场景：用户上传图片A搜相似图，向量库里面本身就存了图片A，会召回自己，这里排除
        results = [r for r in results if Path(r.node.metadata.get('file_path', '')).resolve() != query_path.resolve()]

        # ========= 3. 如果加载了Reranker精排模型，执行多模态精排 =========
        if self.reranker_model:

            # 新建精排处理器，复用全局已经加载好的reranker权重，不重复加载模型
            reranker = Qwen3VLNodePostprocessor(top_n=rerank_top_k, model_instance=self.reranker_model)

            if query:

                # 分支1：用户【图片+文字】一起检索（图文混合检索）
                print(f"   🔄 图文混合精排 (文本约束: '{query}')")
                query_bundle = QueryBundle(query_str=query)
                # 给query_bundle额外挂载图片路径，传给reranker做图文联合打分
                query_bundle.image_path = str(query_path)
            else:

                # 分支2：纯图搜图，没有附加文字描述
                print(f"   🔄 使用图片query进行精排")
                # custom_embedding_strs 是约定俗成的"放图片路径"的字段,Reranker 从这里读
                query_bundle = QueryBundle(query_str="", custom_embedding_strs=[str(query_path)])

            # 调用精排器的后置处理方法，对粗召回的候选图片重新打分、重排序
            results = reranker._postprocess_nodes(results, query_bundle)

        # ========= 4. 分数阈值过滤：低于阈值的全部丢弃 =========
        filtered_results = [r for r in results if (r.score or 0.0) >= score_threshold]

        # 截断，只保留前 rerank_top_k 个最高分结果
        final_results = filtered_results[:rerank_top_k]

        print(f"   📊 最终结果: {len(final_results)} (过滤前: {len(results)})")

        # 调用格式化私有方法，转为前端需要的JSON结构并返回
        return self._format_results(final_results)

    # _format_results(结果格式化):把 LlamaIndex 返回的 NodeWithScore 对象列表，转换成前端可以直接渲染的字典数组
    # 调用方:text_to_image_search 与 image_to_image_search 末尾调用
    def _format_results(self, results) -> List[Dict]:

        # 创建空列表，用来存放格式化之后的结果字典
        formatted = []

        # enumerate(results, 1)：遍历检索结果，序号i从1开始（给前端展示排名）
        # `results`：就是前面检索 / 精排得到的`NodeWithScore`列表，每一个`res`包含`node`（图片节点）+`score`（相似度分数）
        for i, res in enumerate(results, 1):

            # 从节点metadata取出图片本地路径
            file_path = Path(res.node.metadata.get("file_path"))

            # 根据本地文件路径，生成前端可访问的图片url（后端静态资源路由）
            image_url = get_image_url(file_path)

            # 组装前端需要的json结构
            formatted.append({
                "id": str(i),                                    # 排名序号，转字符串
                "imageUrl": image_url,                           # 前端图片预览地址
                "title": res.node.metadata.get("file_name"),     # 图片文件名，用作标题
                "score": round(float(res.score) * 100, 2),       # 展示用分数：乘以100保留2位小数，方便看百分比
                "relevanceScore": res.score,                     # 原始相似度分数（后端内部保留，用于调试）
                "metadata": res.node.metadata                    # 原始完整元数据，备用
            })

        return formatted

# 单例导出:模块级实例化,routers 与 agent 模块都引用这个单例
retrieval_engine = Qwen3VLRetrievalEngine()
