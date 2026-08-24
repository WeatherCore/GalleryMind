<div align="center">

# 🎨 GalleryMind

**多模态图像检索 + Agentic Chat 系统 —— 让 AI 看图找图、自然语言对话**

*Multimodal RAG with Qwen3-VL embedding, Milvus, BM25 fusion, and LangGraph Agent SSE streaming*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.120-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6-646CFF?style=flat-square&logo=vite&logoColor=white)](https://vitejs.dev/)
[![Milvus](https://img.shields.io/badge/Milvus-2.3-00A6FB?style=flat-square)](https://milvus.io/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.14-FF6F00?style=flat-square)](https://docs.llamaindex.ai/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0-1C3C3C?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![Qwen3-VL](https://img.shields.io/badge/Qwen3--VL-Embed%2FReranker-6E37C7?style=flat-square)](https://modelscope.cn/models/qwen/Qwen3-VL-Embedding-2B)

[快速开始](#-快速开始) · [架构总览](#-架构总览) · [技术亮点](#-核心技术亮点) · [项目结构](#-项目结构) · [深度导读](./ZHIDAO.md)

</div>

---

## 📝 项目简介

GalleryMind 是一个把"Qwen3-VL 多模态嵌入 + Milvus 向量库 + BM25 关键词检索 + Qwen3-VL Reranker 精排 + LangGraph Agent + SSE 流式输出"串成一条完整链路的多模态 RAG 系统。

> 它不是"一个搜索引擎",而是**一个有眼睛会看图、有手会调工具的图像管理员**——你给它一句话或一张图,它先理解(embedding)、再翻箱倒柜(Milvus + BM25 召回)、再挑出最像的几张让评委打分(Qwen3-VL Reranker 精排),最后用大白话告诉你找到了啥(Agent 总结)。

支持两种交互模式:

| 模式 | 接口 | 路径 | 体验 |
|---|---|---|---|
| **检索模式** | 同步 JSON | `POST /api/search` | 文搜图 / 图搜图 / 图文混合搜,三选一,带参数滑块 |
| **Agent 模式** | SSE 流式 | `POST /api/agent/chat` | 自然语言对话,LangGraph 自主调 search_images / describe_image 工具,过程透明可见 |

💡 **推荐路径**:首次体验从 Mock 模式开始(`USE_MOCK_DATA=true`),5 分钟跑通前后端联调;再切真实模式,加载 Qwen3-VL 模型体验完整能力。

---

## ✨ 核心技术亮点

- 🔍 **双索引系统**(`backend/core/retrieval.py`)——图搜图与文搜图分离:Milvus `qwen3_vl_image_only` 集合(仅图片向量)专给图搜图,`qwen3_vl_hybrid_agent` 集合 + BM25 给文搜图,避免一种索引兼顾两种检索模式互相干扰。

- 🧠 **三路召回 + RRF 融合**(`Qwen3VLRetrievalEngine.text_to_image_search`)——Qwen3-VL 跨模态向量 + BM25 关键词字面匹配 + (可选 caption 向量),通过 `QueryFusionRetriever(mode="reciprocal_rerank")` 融合,语义与字面互补提升召回率。

- ⚡ **共享 Reranker 模型单例**(`Qwen3VLRetrievalEngine.initialize`)——Qwen3-VL-Reranker-2B 是 2B 大模型,每次检索重新加载会拖慢响应。启动时通过临时 postprocessor 加载一次,后续每次检索创建轻量包装器复用同一模型实例。

- 📐 **L2 归一化 + 512 维截断**(`Qwen3VLEmbedding._get_embedding_from_model`)——嵌入向量除以模长转为单位向量,再截断到 512 维适配 Milvus 集合维度,配合 `similarity_metric="IP"` 实现余弦相似度。

- 🛡️ **LlamaIndex 全局 embed_model 覆盖**(`Qwen3VLRetrievalEngine.initialize`)——`Settings.embed_model = self.embed_adapter` 防止 LlamaIndex 在某些路径(如 `RetrieverQueryEngine.query`)fallback 到默认 OpenAI embedding,避免消耗 API 额度且维度不匹配。

- 🤖 **LangGraph Agent + SSE 6 类事件流**(`backend/core/agent.py`)——`create_agent` + `InMemorySaver` checkpointer,Agent 推理过程拆解为 `thinking → tool_call → process → results → summary → complete` 六类事件实时推送,让用户看见"AI 在思考、在调工具、在出结果"。

- 🚫 **图搜图自身过滤**(`image_to_image_search`)——检索结果中排除 query 图自身,防止返回"最像的就是你自己"。

- 🧪 **Mock 模式开关**(`routers/search.py`)——`USE_MOCK_DATA=true` 不加载任何模型即可跑通前后端联调,适合纯前端开发与 CI 测试。

---

## 🏗️ 架构总览

```mermaid
flowchart TB
    subgraph Frontend["Frontend (React + Vite :3000)"]
        App[App.tsx<br/>mode 切换 + 状态管理]
        Search[UnifiedSearchInput<br/>文本+图片+参数滑块]
        Agent[AgentChat<br/>SSE 事件消费]
        Results[ResultCard / ResultDetail<br/>ComparisonPanel]
    end

    subgraph Backend["Backend (FastAPI :3001)"]
        Router[路由层<br/>search / agent / health]
        Engine[Qwen3VLRetrievalEngine<br/>单例 + 双索引]
        AgentMgr[AgentManager<br/>单例 + LangGraph]
        SSE[SSE 流式输出<br/>6 类事件]
    end

    subgraph Models["AI Models (本地加载)"]
        Embed[Qwen3-VL-Embedding-2B<br/>文本+图片 → 512维向量]
        Rerank[Qwen3-VL-Reranker-2B<br/>共享单例精排]
        GPT[gpt-4o<br/>Agent LLM + 看图说话]
    end

    subgraph Storage["向量库与文件系统"]
        Milvus[(Milvus :19530<br/>2 个 collection)]
        BM25[BM25Retriever<br/>内存关键词索引]
        FS[backend/data/<br/>images / uploads / caption_cache]
    end

    App -->|检索模式| Search
    App -->|Agent 模式| Agent
    Search -->|POST /api/search| Router
    Agent -->|POST /api/agent/chat<br/>SSE| Router

    Router --> Engine
    Router --> AgentMgr
    AgentMgr --> SSE
    SSE -->|事件流| Agent
    AgentMgr -->|search_images 工具| Engine

    Engine --> Embed
    Engine --> Rerank
    AgentMgr --> GPT

    Engine --> Milvus
    Engine --> BM25
    Engine --> FS
    GPT -.->|OpenAI 兼容代理| External[外部 LLM 端点]
```

<details><summary><b>📊 核心检索链路时序图</b>(点击展开)</summary>

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant F as 前端
    participant R as /api/search
    participant E as RetrievalEngine
    participant M as Milvus
    participant B as BM25
    participant RR as Reranker

    U->>F: 输入文本 / 上传图片
    F->>R: POST /api/search
    R->>E: search(mode, query, image_path)

    alt 文搜图
        E->>M: Qwen3-VL 向量召回 Top-30
        E->>B: BM25 关键词召回 Top-30
        E->>E: QueryFusionRetriever RRF 融合
    else 图搜图
        E->>E: embed_adapter._get_image_embedding
        E->>M: multimodal_index 检索 Top-30
        E->>E: 过滤 query 图自身
    end

    E->>RR: 精排全量候选(Reranker 共享单例)
    RR-->>E: 重打分排序
    E->>E: 阈值过滤 + 截断到 rerankTopK
    E-->>R: List[Dict] 结果
    R-->>F: SearchResponse JSON
    F-->>U: 渲染结果卡片网格
```

</details>

---

## 🛠️ 技术栈

| 层级 | 技术 | 版本 | 用途 |
|---|---|---|---|
| **后端框架** | FastAPI + Uvicorn | 0.120 / 0.38 | 异步 Web 框架 + ASGI 服务器 |
| **RAG 框架** | LlamaIndex Core | 0.14.12 | VectorStoreIndex / QueryFusionRetriever / Reranker 适配 |
| **向量库** | Milvus + pymilvus | 2.3.21 / 2.6.3 | 向量存储与相似度检索 |
| **关键词检索** | llama-index-retrievers-bm25 | 0.6.5 | BM25 文本检索 |
| **Agent 框架** | LangChain + LangGraph | 1.2.1 / 1.0.5 | `create_agent` + `InMemorySaver` checkpointer |
| **嵌入/精排模型** | Qwen3-VL-Embedding-2B / Qwen3-VL-Reranker-2B | via ModelScope | 多模态嵌入(512维)+ 精排 |
| **LLM 调用** | langchain-openai (gpt-4o) | 1.1.6 | Agent 推理 + describe_image 看图说话 |
| **深度学习** | PyTorch + Transformers + qwen-vl-utils | 2.9 / 4.57 / 0.0.14 | 模型加载与前向推理 |
| **前端框架** | React + TypeScript | 18.3 / 5.x | 组件化 UI |
| **构建工具** | Vite + @vitejs/plugin-react-swc | 6.3 / 3.10 | 极速 HMR 构建 |
| **UI 组件库** | shadcn/ui (Radix UI) | 60+ 组件 | 可定制无样式组件 |
| **图标/通知** | lucide-react + sonner | 0.487 / 2.0 | 图标 + Toast 通知 |
| **容器化** | Docker Compose | — | Milvus 三件套(etcd + minio + milvus) |

<details><summary><b>📦 后端 Python 依赖全表</b>(点击展开)</summary>

```text
# Web 框架
fastapi==0.120.0
uvicorn[standard]==0.38.0
python-dotenv==1.1.1
pydantic==2.11.10

# LlamaIndex
llama-index-core==0.14.12
llama-index-llms-openai==0.6.13
llama-index-vector-stores-milvus==0.9.5
llama-index-retrievers-bm25==0.6.5

# LangChain & LangGraph
langchain==1.2.1
langchain-openai==1.1.6
langgraph==1.0.5

# AI/ML 模型
transformers==4.57.1
torch==2.9.0
torchvision==0.24.0
qwen-vl-utils==0.0.14
pillow==11.3.0
numpy==2.2.6
scipy==1.16.2

# 模型下载
modelscope==1.31.0

# 向量数据库
pymilvus==2.6.3

# 工具库
requests==2.32.5
```

</details>

---

## 📁 项目结构

```
GalleryMind/
├── backend/                          ⭐ 后端:FastAPI + 多模态 RAG 引擎
│   ├── main.py                       📋 FastAPI 入口,lifespan 预热引擎+Agent
│   ├── config.py                     🔧 配置加载(.env + 路径 + 模型ID + 端口)
│   ├── schemas.py                    📋 Pydantic 请求/响应模型(前端契约)
│   ├── core/                         ⭐ 核心业务逻辑
│   │   ├── retrieval.py              🔐 检索引擎(570 行,项目核心)
│   │   ├── agent.py                  🔐 Agent 管理(LangGraph + SSE 流式)
│   │   └── utils.py                  🔧 base64 图片落地 + URL 转换
│   ├── routers/                      📋 FastAPI 路由层
│   │   ├── search.py                 🔐 /api/search 同步检索接口
│   │   ├── agent.py                  🔐 /api/agent/chat SSE 流式接口
│   │   └── health.py                 🧪 /api/health + /api/ping 健康检查
│   └── data/                         📂 运行时数据(自动创建)
│       ├── images/                   📂 图片库(被检索对象)
│       ├── uploads/                  📂 用户上传的临时图片
│       ├── models/                   📂 ModelScope 模型缓存
│       └── caption_cache/            📂 图片文本描述缓存(.txt)
│
├── frontend/                         ⭐ 前端:React + Vite + TS + shadcn/ui
│   ├── package.json                  📋 依赖清单
│   ├── vite.config.ts                🔧 Vite 配置(含 /api /static 代理到 3001)
│   └── src/
│       ├── App.tsx                   🔐 顶层组件(mode 切换 + 状态管理)
│       ├── types/index.ts            📋 TypeScript 类型定义
│       ├── components/               ⭐ 业务组件(14 个)
│       │   ├── UnifiedSearchInput.tsx 🔐 统一搜索输入
│       │   ├── AgentChat.tsx         🔐 Agent 聊天容器
│       │   ├── AgentMessage.tsx      🔐 Agent 消息渲染
│       │   └── ...(11 个 UI 组件)
│       └── utils/                    📋 工具函数(演示模式模拟器)
│
├── volumes/                          📂 Milvus Docker 数据卷(自动生成)
├── docker-compose.yml                🔧 Milvus 三件套
├── ZHIDAO.md                         📖 项目导读地图(10 章黄金模板)
├── README.md                         📖 本文件
└── Description.md                    📖 项目名片(中英双版)
```

> 📖 **逐文件深度导读见 [ZHIDAO.md](./ZHIDAO.md)**——含运行流程全景图、逐文件代码导读、关键设计模式解析、配置系统详解、复刻路线与常见问题。

---

## 🚀 快速开始

### 0️⃣ 环境要求

| 组件 | 版本 | 默认端口 | 说明 |
|---|---|---|---|
| Python | 3.10+ | — | 见 `backend/requirements.txt` |
| Node.js | 18+ | — | 见 `frontend/package.json` |
| Docker | 20+ | — | 用于跑 Milvus |
| Milvus | v2.3.21 | 19530 | 向量数据库(Docker 启动) |
| GPU(可选)| CUDA 11.8+ | — | 无 GPU 则 CPU 推理(慢 10-50 倍) |

### 1️⃣ 启动 Milvus 向量数据库

```bash
cd D:/1/GalleryMind
docker-compose up -d

# 验证三个容器启动
docker ps
# 应看到:milvus-standalone / milvus-etcd / milvus-minio
```

### 2️⃣ 准备图片库

```bash
mkdir -p backend/data/images
# 把你的图片(.png / .jpg / .jpeg / .gif)放到 backend/data/images/
```

### 3️⃣ 配置后端环境变量

```bash
cd backend
cat > .env <<'EOF'
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MILVUS_URI=http://localhost:19530
USE_MOCK_DATA=false
EOF
```

### 4️⃣ 安装后端依赖

```bash
# 建议用 venv 隔离
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 5️⃣ 启动后端

```bash
python -m backend.main
# 或:python main.py
```

> ⚠️ **首次启动**会从 ModelScope 下载 `Qwen3-VL-Embedding-2B`(约 4GB)和 `Qwen3-VL-Reranker-2B`(约 4GB)到 `backend/data/models/`,需 10-30 分钟(取决于网络)。后续启动从本地缓存加载约 30-60 秒。
>
> 看到以下日志说明就绪:
> ```
> 🚀 Server starting on 0.0.0.0:3001
> ✅ 检索引擎和Agent就绪
> ```

<details><summary><b>🧪 Mock 模式</b>:不加载模型快速跑通前后端(点击展开)</summary>

```bash
# 设置环境变量启动 Mock 模式
USE_MOCK_DATA=true python -m backend.main
# 不下载模型,直接返回预设假数据,适合纯前端开发与联调
```

</details>

### 6️⃣ 启动前端

```bash
# 另开终端
cd frontend
npm install
npm run dev
# 浏览器自动打开 http://localhost:3000
```

### 7️⃣ 体验核心链路

**检索模式**:
- 顶部确保在"检索"模式
- 输入文本(如"红色跑车")或上传图片,或两者都有(混合搜)
- 调节召回 Top-K(默认 30)/ 精排 Top-K(默认 6)/ 阈值(默认 0.5)
- 点"开始检索"或 `⌘+Enter`,查看结果卡片网格

**Agent 模式**:
- 顶部切换到"Agent"
- 自然语言对话:"帮我找一张红色跑车的图片,并描述它的内容"
- 观察 Agent 实时思考链 → 工具调用 → 结果 → 总结

**直接调 API 测试**:

```bash
# 同步检索
curl -X POST http://localhost:3001/api/search \
  -H "Content-Type: application/json" \
  -d '{"searchMode":"文搜图","textQuery":"测试","recallTopK":20,"rerankTopK":5,"threshold":0.0}'

# Agent SSE 流式
curl -N -X POST http://localhost:3001/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"帮我找一张红色跑车的图片"}'
```

---

## ⚙️ 配置说明

| 变量 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | OpenAI / 兼容代理的 API Key |
| `OPENAI_BASE_URL` | ⬜ | `https://api.openai.com/v1` | LLM 端点,可指向国内代理 |
| `MILVUS_URI` | ⬜ | `http://localhost:19530` | Milvus 连接串 |
| `IMAGE_DIR` | ⬜ | `backend/data/images` | 被检索的图片库路径 |
| `MODELSCOPE_CACHE` | ⬜ | `backend/data/models` | ModelScope 模型缓存目录 |
| `USE_MOCK_DATA` | ⬜ | `false` | Mock 模式开关 |
| `HOST` / `PORT` | ⬜ | `0.0.0.0:3001` | 后端监听(代码硬编码) |
| `ALLOWED_ORIGINS` | ⬜ | `["*"]` | CORS 白名单(代码硬编码) |
| `EMBEDDING_DIM` | ⬜ | `512` | 嵌入向量维度(代码硬编码) |

💡 **最低可用配置**:仅需在 `backend/.env` 设置 `OPENAI_API_KEY=sk-xxx`,其余都有默认值;若用 Mock 模式连 Key 都不需要。

---

## 🧭 Roadmap

- [x] 双索引系统(图搜图 + 文搜图分离)
- [x] 三路召回 + RRF 融合(Qwen3-VL 向量 + BM25)
- [x] Qwen3-VL Reranker 精排(共享单例)
- [x] LangGraph Agent + SSE 6 类事件流
- [x] 检索模式 + Agent 模式双界面切换
- [x] Mock 模式快速联调
- [ ] Caption 索引重启用(性能优化)
- [ ] Agent 多用户会话隔离(替换 `thread_id="session_default"`)
- [ ] 检索过程真实进度上报(替代前端 simulator)
- [ ] 持久化 Checkpointer(PostgresSaver / RedisSaver)

---

<div align="center">

**🤝 参与贡献**

Fork → 创建分支 → 提交 PR

如果这个项目对你有帮助,欢迎 ⭐ Star 支持!

**📖 完整项目导读** · [ZHIDAO.md](./ZHIDAO.md) · **📋 项目名片** · [Description.md](./Description.md)

</div>
