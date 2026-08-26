# 📖 GalleryMind 项目导读指南

> 本文件是 `GalleryMind` 项目的中文导读，帮助你从零开始理解这个**多模态图像检索 + Agentic Chat 系统**的架构、代码和运行方式。
>
> 阅读建议：先看第 1-4 章建立全局心智模型，再按第 5 章「逐文件导读」的顺序深入源码；遇到不懂的概念查第 2 章，遇到配置问题查第 7 章。

---

## 1. 这个项目是干什么的？

**一句话定位**：GalleryMind 是一个把"Qwen3-VL 多模态嵌入 + Milvus 向量库 + BM25 关键词检索 + Qwen3-VL Reranker 精排 + LangGraph Agent + SSE 流式输出"串成一条完整链路的多模态 RAG 系统，让用户既能**文搜图 / 图搜图 / 图文混合搜**，也能用**自然语言跟 Agent 对话**让它自动调工具找图。

```
用户输入（文本/图片/混合）
        │
        ▼
┌───────────────────────────────────┐
│ 两条入口路径                       │
│                                   │
│ [A] /api/search  同步检索          │
│     ├─ 文搜图：三路召回 → Fusion → Rerank │
│     ├─ 图搜图：Qwen3-VL embedding → Milvus → Rerank │
│     └─ 混合搜：图片 embedding + 文本约束 Rerank │
│                                   │
│ [B] /api/agent/chat  SSE 流式 Agent │
│     └─ LangGraph create_agent      │
│        tools=[search_images,       │
│                describe_image]     │
│        事件: thinking→tool_call→   │
│        process→results→summary→    │
│        complete                    │
└───────────────────────────────────┘
        │
        ▼
   图片 URL 列表 + Agent 自然语言总结
   （图片本体由后端 /static 静态服务提供）
```

**类比**：它不是"一个搜索引擎"，而是**一个有眼睛会看图、有手会调工具的图像管理员**——你给它一句话或一张图，它先理解（embedding）、再翻箱倒柜（Milvus + BM25 召回）、再挑出最像的几张让评委打分（Qwen3-VL Reranker 精排），最后用大白话告诉你找到了啥（Agent 总结）。

**要点**：
- 既有**传统检索路径**（`/api/search` 同步返回 JSON），也有**Agent 路径**（`/api/agent/chat` SSE 流式），共享同一套检索引擎
- 检索引擎走**双索引 + 三路召回 + 共享 Reranker**，是一个完整的工业级多模态 RAG 架构样本
- 前端两套界面（检索面板 / Agent 聊天）通过 `currentMode` 切换，共用 `recallTopK` / `rerankTopK` 参数

---

## 2. 核心概念速览

读代码前必须先懂这 7 个概念，否则后续每一行都会卡。

| 概念 | 是什么 | 把它理解为… | 伪代码 |
|------|--------|-------------|--------|
| **Qwen3-VL-Embedding-2B** | 阿里通义千问视觉语言嵌入模型，2B 参数 | 一个**翻译官**：把图片和文本都翻成 512 维向量，让机器能"算相似度" | `embed(image) -> vec[512]` `embed(text) -> vec[512]` |
| **Milvus** | 开源向量数据库，存向量并支持相似度检索 | 一个**按"长得像不像"找图的图书馆**，跟传统 SQL 数据库按"等于不等于"找不一样 | `Milvus.search(query_vec, top_k=30)` |
| **BM25** | 经典文本关键词检索算法（TF-IDF 的进化版） | 一个**只认字面**的搜索引擎，"红色跑车"就去找包含"红色"和"跑车"的文档 | `BM25.retrieve("红色跑车", k=30)` |
| **QueryFusionRetriever** | LlamaIndex 的多检索器融合器 | 一个**收发室**：把多个检索器（向量+BM25）的结果按 RRF（Reciprocal Rank Fusion）合并排名 | `Fusion.retrieve(query) -> merged_list` |
| **Qwen3-VL-Reranker-2B** | 阿里通义千问视觉语言精排模型 | 一个**评委**：召回的 30 张图让评委看一遍（query+图一起喂），重新打分挑出真正最像的 5 张 | `Reranker.score(query, docs) -> sorted_docs` |
| **LangGraph create_agent** | LangChain 1.x 推出的 Agent 编排接口（封装 LangGraph） | 一个**带记忆的工具调度员**：LLM 决定调哪个工具，结果回传给 LLM 继续推理，循环直到给出最终答案 | `agent.stream({messages}) -> events` |
| **SSE (Server-Sent Events)** | HTTP 长连接流式推送协议 | 一个**单向广播**：服务器不断往前端推事件（thinking/tool_call/results...），前端只读不写 | `event: thinking\ndata: {...}\n\n` |

**要点**：
- **召回（Recall）vs 精排（Rerank）是两阶段**：召回快速捞一大批（top_k=30），精排慢但准挑少数（top_k=5）。这是工业 RAG 的标准范式
- **embedding 维度统一 512**：所有向量都截断/归一化到 512 维，否则 Milvus 集合维度不匹配会报错
- **Agent 不是必需路径**：项目本质是检索系统，Agent 只是套在检索之上的一层"自然语言交互外壳"

---

## 3. 项目目录结构详解

```
GalleryMind/
├── backend/                      ⭐ 后端：FastAPI + 多模态 RAG 引擎
│   ├── main.py                   📋 FastAPI 入口，lifespan 预热引擎+Agent
│   ├── config.py                 🔧 配置加载（.env + 路径 + 模型ID + 端口）
│   ├── schemas.py                📋 Pydantic 请求/响应模型（前端契约）
│   ├── requirements.txt          📋 Python 依赖清单
│   ├── core/                     ⭐ 核心业务逻辑
│   │   ├── retrieval.py          🔐 检索引擎（570行，项目核心）
│   │   ├── agent.py              🔐 Agent 管理（LangGraph + SSE 流式）
│   │   └── utils.py              🔧 base64 图片落地 + URL 转换
│   ├── routers/                  📋 FastAPI 路由层
│   │   ├── search.py             🔐 /api/search 同步检索接口
│   │   ├── agent.py              🔐 /api/agent/chat SSE 流式接口
│   │   └── health.py             🧪 /api/health + /api/ping 健康检查
│   ├── data/                     📂 运行时数据（自动创建）
│   │   ├── images/               📂 图片库（被检索的对象）
│   │   ├── uploads/              📂 用户上传的临时图片
│   │   ├── models/               📂 ModelScope 下载的模型缓存
│   │   └── caption_cache/        📂 图片文本描述缓存（.txt）
│   ├── debug_*.py                🧪 调试脚本（不参与主流程）
│   └── test_*.py                 🧪 测试脚本（不参与主流程）
│
├── frontend/                     ⭐ 前端：React + Vite + TS + shadcn/ui
│   ├── package.json              📋 依赖清单（name: 多模态RAG检索演示界面）
│   ├── vite.config.ts            🔧 Vite 配置（含 /api /static 代理到 3001）
│   ├── index.html                📋 HTML 入口
│   └── src/
│       ├── main.tsx              📋 React 入口（createRoot.render）
│       ├── App.tsx               🔐 顶层组件（mode 切换 + 状态管理）
│       ├── types/index.ts        📋 TypeScript 类型定义（SearchResult/Message）
│       ├── components/           ⭐ 业务组件
│       │   ├── Header.tsx        📋 顶部导航（检索/Agent 模式切换）
│       │   ├── UnifiedSearchInput.tsx 🔐 统一搜索输入（文本+图片+阈值）
│       │   ├── SearchInput.tsx   📋 旧版搜索输入（保留兼容）
│       │   ├── AgentChat.tsx     🔐 Agent 聊天容器（SSE 事件消费）
│       │   ├── AgentInput.tsx    📋 Agent 输入框
│       │   ├── AgentMessage.tsx  🔐 Agent 消息渲染（思考链+结果+总结）
│       │   ├── ResultCard.tsx    📋 结果卡片（带选中状态）
│       │   ├── ResultDetail.tsx  📋 结果详情弹窗
│       │   ├── ComparisonPanel.tsx 📋 对比面板（多选结果对比）
│       │   ├── ProcessVisualization.tsx 📋 检索过程可视化
│       │   ├── QuickExamples.tsx 📋 快捷示例查询
│       │   ├── ImageLightbox.tsx 📋 图片放大灯箱
│       │   ├── figma/ImageWithFallback.tsx 📋 带兜底的图片组件
│       │   └── ui/              📋 shadcn/ui 标准组件（60+ 个，本项目跳过注释）
│       └── utils/                📋 工具函数
│           ├── mockData.ts      📋 模拟检索结果（演示用）
│           ├── agentSimulator.ts 📋 Agent 响应模拟器（演示用）
│           └── processSimulator.ts 📋 进度条模拟器（演示用）
│
├── volumes/                      📂 Milvus Docker 数据卷（自动生成）
├── docker-compose.yml            🔧 Milvus 三件套（etcd + minio + milvus）
└── ZHIDAO.md / README.md / Description.md  📋 本套导读文档
```

**emoji 标记说明**：⭐ 重点目录 / 🔐 核心业务文件 / 🔧 配置/工具 / 📋 普通/契约文件 / 🧪 测试/调试 / 📂 运行时数据

**要点**：
- 真正承载项目意图的源码约 30 个（backend 9 + frontend 21），其余 60+ 个 `ui/*.tsx` 是 shadcn/ui 模板件
- `frontend/src/utils/*` 三个 simulator 是**演示模式用**的假数据生成器，真实检索走 `/api/search` 后端
- `backend/data/` 下的子目录在 `config.py` 中通过 `mkdir(parents=True, exist_ok=True)` 自动创建

---

## 4. 运行流程全景图

### 4.1 启动流程（lifespan）

```
用户执行 python -m backend.main
        │
        ▼
┌─────────────────────────────────────────┐
│ main.py: app = FastAPI(lifespan=lifespan)│
└─────────────────────────────────────────┘
        │
        ▼ lifespan 启动钩子
┌─────────────────────────────────────────┐
│ 1. 挂载 /static -> backend/data         │
│    （让前端能访问图片）                  │
│                                         │
│ 2. retrieval_engine.initialize(         │
│      DEFAULT_IMAGE_DIR)                 │
│    ├─ 加载 Qwen3-VL-Embedding（GPU/MPS/CPU 自动选择）│
│    ├─ 设置 LlamaIndex Settings.embed_model（防 fallback 到 OpenAI）│
│    ├─ 加载 caption_cache（图片文本描述缓存）│
│    ├─ SimpleDirectoryReader 扫描图片库   │
│    ├─ _build_indices:                   │
│    │   ├─ [1] 图片向量索引（Milvus collection=qwen3_vl_image_only）│
│    │   │      仅图搜图用，512 维 IP      │
│    │   └─ [2] 混合检索索引               │
│    │       ├─ Qwen3-VL 向量索引（collection=qwen3_vl_hybrid_agent）│
│    │       └─ BM25Retriever（关键词检索）│
│    └─ 加载 Qwen3-VL-Reranker（共享单例）  │
│                                         │
│ 3. agent_manager.initialize()           │
│    └─ create_agent(model=gpt-4o,        │
│        tools=[search_images,            │
│                describe_image],         │
│        checkpointer=InMemorySaver())   │
└─────────────────────────────────────────┘
        │
        ▼
   uvicorn.run :3001 就绪，等请求
```

**退出条件**：所有组件初始化成功（任一失败会冒泡抛异常导致启动失败）。

### 4.2 检索路径（/api/search 同步）

```
前端 fetch POST /api/search
        │
        ▼
routers/search.py: search(request: SearchRequest)
        │
        ├─ [Mock 模式] USE_MOCK_DATA=true → 返回 MOCK_RESULTS（不加载模型）
        │
        └─ [真实模式]
            │
            ▼
            retrieval_engine.search(mode, query, image_path, ...)
                │
                ├─ mode='文搜图' → text_to_image_search()
                │       │
                │       ├─ 构造 retrievers = [Qwen3-VL 向量 retriever, BM25 retriever]
                │       │   (caption_index 若存在也加入，本项目已禁用)
                │       ├─ QueryFusionRetriever(mode='reciprocal_rerank')
                │       ├─ 若有 reranker_model:
                │       │   └─ RetrieverQueryEngine(retriever, [Qwen3VLNodePostprocessor])
                │       │      query_engine.query(query) -> response.source_nodes
                │       │      (Rerank ALL recalled nodes, 不仅是 top K)
                │       ├─ 阈值过滤: r.score >= score_threshold
                │       └─ 截断到 rerank_top_k
                │
                ├─ mode='图搜图' → image_to_image_search()
                │       ├─ embed_adapter._get_image_embedding(query_path) -> vec[512]
                │       ├─ multimodal_index.as_retriever(similarity_top_k=recall_top_k)
                │       │   .retrieve(QueryBundle(embedding=vec))
                │       ├─ 过滤自身: Path != query_path
                │       ├─ 若有 reranker:
                │       │   └─ Qwen3VLNodePostprocessor._postprocess_nodes()
                │       │      (纯图搜图：query 用图片；混合搜：query 用图+文)
                │       └─ 阈值过滤 + 截断
                │
                └─ mode='混合搜索' → image_to_image_search(query=...)
                    (同图搜图，但 reranker 接收文本约束)
                │
                ▼
            _format_results(): 转换为前端契约
                id, imageUrl (/static/...), title, score (0-100),
                relevanceScore (0-1), metadata
                │
                ▼
            SearchResponse(results, totalTime)
```

**退出条件**：返回 `SearchResponse` JSON 给前端；任一异常被 `try/except` 捕获并 `raise HTTPException(500)`。

### 4.3 Agent 路径（/api/agent/chat SSE 流式）

```
前端 fetch POST /api/agent/chat
        │
        ▼
routers/agent.py: chat_sse(request: AgentChatRequest)
        │
        ├─ 若有 image (base64) → save_base64_image() 落地到 UPLOAD_DIR
        │
        └─ return StreamingResponse(agent_manager.chat_stream(...),
                                     media_type="text/event-stream")
                │
                ▼
            AgentManager.chat_stream(query, image_path)
                │
                ├─ yield SSE: event=thinking, data={"content":"正在分析..."}
                │
                └─ async for event in _run_agent_async(content):
                        │
                        ▼
                    agent.stream({messages:[{role:user, content}]},
                                  config={thread_id: "session_default"},
                                  stream_mode="values")
                        │
                        ▼  (LangGraph 内部循环：LLM 推理 → 调工具 → 结果回传 → 继续)
                        │
                        └─ for event in events:
                              msg = event["messages"][-1]
                              │
                              ├─ msg.type == "ai" 且有 tool_calls:
                              │   yield SSE: event=tool_call, {toolName, timestamp}
                              │   yield SSE: event=process, {stepId, progress:50, status:processing}
                              │
                              ├─ msg.type == "ai" 无 tool_calls:
                              │   yield SSE: event=summary, {content, done:true}
                              │
                              └─ msg.type == "tool":
                                  try json.loads(msg.content)
                                  ├─ success+results:
                                  │   yield SSE: event=results, {results:[...]}
                                  └─ process: {stepId, progress:100, status:completed}
                              │
                              └─ await asyncio.sleep(0.01)  # 让出控制权模拟流式
                │
                ▼
            yield SSE: event=complete, {}
```

**退出条件**：Agent 自然结束（无 tool_call 输出最终 summary）→ 发 `complete` 事件；异常被 catch 后发 thinking 错误事件 + complete 收尾。

**要点**：
- 三条路径**共用同一个 `retrieval_engine` 单例**和**同一个 `agent_manager` 单例**，启动时初始化一次，请求时复用
- Agent 路径的 `search_images` 工具内部调用 `retrieval_engine.image_to_image_search` / `text_to_image_search`，与 `/api/search` 共享检索能力
- SSE 6 类事件按顺序：`thinking → tool_call → process(50) → results → process(100) → summary → complete`

---

## 5. 逐文件代码导读

### 阅读顺序建议

| 阶段 | 顺序 | 文件 | 行数 | 耗时估计 | 阅读目标 |
|------|------|------|------|----------|----------|
| 入门 | 1 | `backend/config.py` | 40 | 5 分钟 | 看清配置项与路径布局 |
| 入门 | 2 | `backend/schemas.py` | 49 | 5 分钟 | 看清前后端数据契约 |
| 入门 | 3 | `backend/main.py` | 74 | 10 分钟 | 看清启动流程 |
| 入门 | 4 | `backend/core/utils.py` | 42 | 5 分钟 | 看清图片落地与 URL 转换 |
| 入门 | 5 | `backend/routers/health.py` | 15 | 2 分钟 | 健康检查（最简单的路由样板）|
| 进阶 | 6 | `backend/routers/search.py` | 140 | 15 分钟 | 同步检索接口（含 Mock）|
| 进阶 | 7 | `backend/routers/agent.py` | 36 | 10 分钟 | SSE 流式接口 |
| 核心 | 8 | `backend/core/agent.py` | 224 | 30 分钟 | Agent 编排与 SSE 事件分发 |
| 核心 | 9 | `backend/core/retrieval.py` | 570 | 60-90 分钟 | **项目大脑**，建议分块读 |
| 前端 | 10 | `frontend/src/types/index.ts` | ~50 | 5 分钟 | 前端类型契约 |
| 前端 | 11 | `frontend/src/App.tsx` | 282 | 20 分钟 | 前端顶层状态机 |
| 前端 | 12 | `frontend/src/components/UnifiedSearchInput.tsx` | 312 | 15 分钟 | 检索输入主组件 |
| 前端 | 13 | `frontend/src/components/AgentChat.tsx` | 269 | 20 分钟 | Agent 聊天容器 |
| 前端 | 14 | 其余业务组件 | ~1000 | 60 分钟 | 按需读 |
| 前端 | 15 | `frontend/src/utils/*` | ~200 | 15 分钟 | 演示模式模拟器 |

### 5.1 backend/config.py

**作用**：项目配置中枢，从 `.env` 加载敏感信息（API Key、Milvus URI），定义路径布局（图片库、上传、模型缓存、caption 缓存），并自动创建必要目录。

**关键数据结构**：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BASE_DIR` | `backend/` | 后端根目录 |
| `DATA_DIR` | `backend/data` | 运行时数据根 |
| `UPLOAD_DIR` | `backend/data/uploads` | 用户上传图片落地 |
| `MODEL_CACHE_DIR` | `backend/data/models` | ModelScope 模型缓存 |
| `CAPTION_CACHE_DIR` | `backend/data/caption_cache` | 图片文本描述缓存 |
| `DEFAULT_IMAGE_DIR` | `$IMAGE_DIR` 或 `data/images` | 被检索的图片库 |
| `OPENAI_API_KEY` | 来自 `.env` | LLM 凭证（gpt-4o）|
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | LLM 端点（可指向代理）|
| `MILVUS_URI` | `http://localhost:19530` | Milvus 连接串 |
| `EMBEDDING_MODEL_MS` | `qwen/Qwen3-VL-Embedding-2B` | 嵌入模型 ID |
| `RERANKER_MODEL_MS` | `qwen/Qwen3-VL-Reranker-2B` | 精排模型 ID |
| `HOST/PORT` | `0.0.0.0:3001` | 服务监听 |
| `ALLOWED_ORIGINS` | `["*"]` | CORS 全开（开发友好）|

**要点**：
- `load_dotenv(override=True)` 用 `.env` 覆盖系统环境变量
- 三个 `mkdir(parents=True, exist_ok=True)` 保证目录存在，避免运行时 `FileNotFoundError`
- `ALLOWED_ORIGINS=["*"]` 适合开发，生产环境应收紧

### 5.2 backend/schemas.py

**作用**：Pydantic 模型层，定义前后端 JSON 契约。FastAPI 用这些模型自动做请求校验和响应序列化。

**模型清单**：

| 模型 | 用途 | 关键字段 |
|------|------|----------|
| `SearchRequest` | `/api/search` 请求 | `searchMode`（文搜图/图搜图/混合）、`textQuery`、`uploadedImage`(base64)、`recallTopK=20`、`rerankTopK=5`、`threshold=0.0` |
| `SearchResult` | 单条检索结果 | `imageUrl`、`score`(0-100)、`relevanceScore`(0-1)、`rerankScore`、`metadata` |
| `SearchResponse` | `/api/search` 响应 | `results`、`processSteps`、`totalTime` |
| `ProcessStep` | 检索过程步骤 | `status`(pending/processing/completed)、`progress`、`duration` |
| `AgentChatRequest` | `/api/agent/chat` 请求 | `message`、`image`(base64) |
| `InitRequest/InitResponse` | （预留）初始化接口 | `milvus_uri` |

**要点**：
- `searchMode` 用中文字符串枚举（`'文搜图' | '图搜图' | '混合搜索'`），后端 `retrieval_engine.search()` 直接 if/elif 分发，简单直接但扩展性一般
- `score` 字段 0-100（前端展示用），`relevanceScore` 字段 0-1（原始相似度），两者并存是历史决策
- `AgentChatRequest` 极简，因为 Agent 真实数据走 SSE 流，Schema 主要用于请求校验和文档生成

### 5.3 backend/main.py

**作用**：FastAPI 应用入口，组装中间件、静态文件、路由，并通过 `lifespan` 在启动时预热检索引擎与 Agent。

**关键函数**：

| 函数 | 行号 | 职责 |
|------|------|------|
| `lifespan(app)` | ~23 | 启动时挂载静态文件、初始化 retrieval_engine 和 agent_manager；关闭时打印日志 |
| `root()` | ~69 | `GET /` 返回 `{"status":"ok"}` 用于探活 |

**要点**：
- 用 `lifespan` 替代弃用的 `@app.on_event("startup")`，是 FastAPI 0.93+ 推荐写法
- `app.include_router(health.router)` 优先注册，保证健康检查路由靠前
- `sys.path.insert(0, parent_dir)` 让 `python main.py` 和 `python -m backend.main` 两种启动方式都能正确解析 `from backend.config import ...`

### 5.4 backend/core/utils.py

**作用**：图片处理工具，负责把前端上传的 base64 图片落地为本地文件，以及把本地路径转换为前端可访问的 `/static/...` URL。

**关键函数**：

| 函数 | 行号 | 职责 |
|------|------|------|
| `save_base64_image(base64_str, save_dir)` | ~9 | 去掉 `data:image/...;base64,` header → `base64.b64decode` → PIL 打开 → 用 `uuid4().hex` 生成唯一名 → 保存为 PNG |
| `get_image_url(file_path)` | ~24 | `file_path.relative_to(DATA_DIR)` → `/static/<relative>`；若不在 DATA_DIR 下，回退到 `/static/images/<filename>` |

**要点**：
- 用 `uuid4().hex` 命名避免重名碰撞，但所有上传图都转 PNG（即使原图是 JPG），有轻微存储冗余
- `get_image_url` 假设 `StaticFiles(directory=DATA_DIR)` 挂载在 `/static`，与 `main.py` 中的 `app.mount("/static", ...)` 强耦合

### 5.5 backend/core/agent.py

**作用**：Agent 编排核心，封装 LangGraph `create_agent`，提供 SSE 流式输出。

**关键类与函数**：

| 名称 | 行号 | 职责 |
|------|------|------|
| `_agent_llm` / `_vision_llm` | ~246 / ~249 | 两个全局 ChatOpenAI 实例：前者给 Agent 用，后者给 `describe_image` 工具用 |
| `@tool search_images(query, image_path)` | ~35 | 工具：调 retrieval_engine 做检索，返回 JSON 字符串（含 success/results）|
| `@tool describe_image(image_path)` | ~62 | 工具：用 gpt-4o 看图说话，返回中文描述 |
| `class AgentManager` | ~88 | 单例（`__new__` 实现），管理 agent 实例与配置 |
| `AgentManager.initialize()` | ~98 | 创建 `_agent_llm`、注册 tools、定义 system_prompt、调用 `create_agent` + `InMemorySaver` checkpointer |
| `AgentManager.chat_stream(query, image_path)` | ~129 | SSE 流式入口：发 thinking → 调 `_run_agent_async` → 发 complete |
| `AgentManager._run_agent_async(content)` | ~157 | 遍历 `agent.stream(stream_mode="values")`，按消息类型分发 6 类 SSE 事件 |
| `AgentManager._format_sse(event_type, data)` | ~217 | 拼接 `event: X\ndata: Y\n\n` 格式字符串 |
| `_ts()` | ~221 | 返回 `str(time.time())` 时间戳字符串 |

**要点**：
- 两个 LLM 实例**全局化**避免重复创建（ChatOpenAI 构造有连接池开销）
- `InMemorySaver` checkpointer 让 Agent 具备多轮记忆能力（但 `thread_id="session_default"` 是全局共享的，多用户场景下会串话——这是教学项目的简化设计）
- SSE 事件类型与前端 `AgentChat.tsx` 的事件处理严格对应：`thinking/tool_call/process/results/summary/complete`
- `await asyncio.sleep(0.01)` 在事件循环中让出控制权，模拟流式效果（实际 LangGraph stream 是同步 generator）

### 5.6 backend/core/retrieval.py（项目核心，570 行）

**作用**：多模态检索引擎，封装 Qwen3-VL 嵌入、Milvus 双索引、BM25、Qwen3-VL Reranker，提供统一检索入口。

**辅助函数**：

| 函数 | 行号 | 职责 |
|------|------|------|
| `download_model_from_modelscope(model_id, cache_dir)` | ~38 | 从 ModelScope 下载模型，失败回退到原始 ID |
| `load_caption_cache(cache_dir)` | ~54 | 加载 `caption_cache/*.txt` 为 `{图片名: 描述}` 字典 |

**模型类**：

| 类 | 行号 | 职责 |
|---|------|------|
| `Qwen3VLEmbedding` | ~68 | 封装 Qwen3-VL-Embedding-2B：自动选 device（CUDA/MPS/CPU）、L2 归一化、截断到 `output_dim=512` |
| `Qwen3VLMultiModalEmbedding(MultiModalEmbedding)` | ~147 | LlamaIndex 适配器：把 `Qwen3VLEmbedding` 包装成 LlamaIndex 标准 MultiModalEmbedding 接口 |
| `Qwen3VLNodePostprocessor(BaseNodePostprocessor)` | ~175 | LlamaIndex 节点后处理器：封装 Qwen3-VL-Reranker-2B，支持文本/图片/混合 query 精排 |
| `Qwen3VLRetrievalEngine` | ~280 | **单例检索引擎**：构建双索引、提供 `search()` 统一入口 |

**Qwen3VLRetrievalEngine 关键方法**：

| 方法 | 行号 | 职责 |
|---|------|------|
| `__new__` | ~285 | 单例模式：第一次创建设 `initialized=False`，后续返回同一实例 |
| `initialize(image_dir, use_reranker=True)` | ~307 | 加载 embedder → 设置全局 `Settings.embed_model` → 加载 caption → 读取图片 → `_build_indices` → 加载共享 Reranker |
| `_build_indices(documents, image_dir)` | ~348 | 构建**双索引系统**：[1] 图片向量索引（图搜图）[2] 混合检索索引（Qwen3-VL 向量 + BM25）|
| `search(mode, query, image_path, ...)` | ~450 | 统一入口：按 mode 分发到 `text_to_image_search` 或 `image_to_image_search` |
| `text_to_image_search(query, ...)` | ~469 | 文搜图：`QueryFusionRetriever` 融合 Qwen3-VL 向量 + BM25，可选 Reranker 精排 |
| `image_to_image_search(image_path, query, ...)` | ~507 | 图搜图/混合搜：query 图 embedding → Milvus 检索 → 过滤自身 → Reranker 精排 |
| `_format_results(results)` | ~550 | 转换为前端契约：`id/imageUrl/title/score/relevanceScore/metadata` |

**要点**（项目最关键的设计决策都在这里）：
- **双索引系统**：图片向量索引（`qwen3_vl_image_only` 集合）专给图搜图用，混合索引（`qwen3_vl_hybrid_agent` 集合 + BM25）专给文搜图用，避免一种索引兼顾两种检索模式
- **共享 Reranker 模型**：`temp_processor = Qwen3VLNodePostprocessor()` 加载模型后提取 `self.reranker_model`，后续每次检索创建轻量 postprocessor 时通过 `model_instance=` 复用，避免重复加载 2B 大模型
- **L2 归一化 + 截断到 512 维**：`embedding / np.linalg.norm(embedding)` + `embedding[:output_dim]`，配合 Milvus 的 `similarity_metric="IP"`（内积）实现余弦相似度
- **全局 `Settings.embed_model` 覆盖**：`Settings.embed_model = self.embed_adapter` 防止 LlamaIndex 在某些路径（如 `query_engine.query`）fallback 到默认 OpenAI embedding，避免消耗 API 额度
- **Caption 索引已禁用**：代码注释解释了原因——Qwen3-VL 处理 37622 个文本块需 5+ 小时，改用 BM25 做文本关键词检索替代
- **图搜图过滤自身**：`Path(r.node.metadata['file_path']).resolve() != query_path.resolve()` 防止返回 query 图自己
- **Reranker 全量重排**：`top_n=recall_top_k`（不是 `rerank_top_k`），保证阈值过滤有足够候选

### 5.7 backend/routers/search.py

**作用**：同步检索 HTTP 接口，支持 Mock 模式（不加载模型）与真实模式（调 retrieval_engine）。

**关键函数**：

| 函数 | 行号 | 职责 |
|---|------|------|
| `search(request: SearchRequest)` | ~87 | 入口：Mock 模式直接返回 `MOCK_RESULTS[:rerankTopK]`；真实模式调 `retrieval_engine.search()` 转换为 `SearchResponse` |

**要点**：
- `USE_MOCK_DATA` 环境变量控制 Mock 开关，用于前端联调（不依赖 GPU 模型）
- 异常捕获 `traceback.print_exc() + raise HTTPException(500)`，便于排查
- `processSteps=[]` 始终返回空数组，前端 `ProcessVisualization` 的进度条其实走的是 `processSimulator.ts` 模拟

### 5.8 backend/routers/agent.py

**作用**：SSE 流式 Agent 聊天接口。

**关键函数**：

| 函数 | 行号 | 职责 |
|---|------|------|
| `chat_sse(request: AgentChatRequest)` | ~13 | 接收 message+image（base64），若有图片用 `save_base64_image` 落地，返回 `StreamingResponse(media_type="text/event-stream")` |

**要点**：
- 用 POST 而非 GET，因为图片 base64 payload 较大，GET 受 URL 长度限制
- 图片保存失败时 `try/except: pass` 静默降级，Agent 仍可纯文本对话
- `media_type="text/event-stream"` 是 SSE 标准 MIME，浏览器 `EventSource` 与 `fetch` 都可消费

### 5.9 backend/routers/health.py

**作用**：极简健康检查路由，用于前后端联通性测试。

**关键函数**：

| 函数 | 行号 | 职责 |
|---|------|------|
| `health_check()` | ~9 | `GET /api/health` 返回 `{"status":"ok","service":"multimodal-rag-backend"}` |
| `ping()` | ~13 | `GET /api/ping` 返回 `{"message":"pong"}` |

**要点**：两个接口功能类似，`/health` 用于 K8s/监控探活，`/ping` 用于前端调试。

### 5.10 frontend/src/App.tsx

**作用**：前端顶层组件，管理 `currentMode`（retrieval/agent）切换、检索状态、结果列表、对比模式。

**关键状态**：

| State | 默认值 | 用途 |
|---|---|---|
| `currentMode` | `'retrieval'` | 当前界面模式 |
| `textQuery` / `uploadedImage` | `''` / `null` | 检索输入 |
| `searchResults` | `[]` | 检索结果列表 |
| `recallTopK` / `rerankTopK` / `threshold` | `30 / 6 / 0.5` | 检索参数 |
| `compareMode` / `selectedResults` | `false` / `Set()` | 对比模式 |
| `processSteps` / `showProcess` | `[]` / `false` | 检索过程可视化 |

**关键函数**：

| 函数 | 行号 | 职责 |
|---|------|------|
| `getSearchMode()` | ~48 | 根据 textQuery/uploadedImage 推断 `searchMode`（文搜图/图搜图/混合）|
| `handleSearch()` | ~55 | 调 `POST /api/search`，更新进度状态与结果 |
| `handleResultSelect(id)` | ~116 | 对比模式下切换结果选中态 |
| `useEffect(键盘监听)` | ~134 | `⌘+Enter` 触发检索，`⌘+K` 聚焦输入框 |

**要点**：
- 检索走真实后端 `/api/search`（不走 mockData），但 `processSimulator` 用于模拟进度条
- `useEffect` 自动调整 `rerankTopK <= recallTopK`，防止参数倒挂
- 暗色主题 `bg-[#0F172A]`，shadcn/ui 组件库统一视觉风格

### 5.11 frontend/src/components/AgentChat.tsx

**作用**：Agent 聊天容器，消费 SSE 流，管理消息列表与流式渲染。

**关键状态**：

| State | 用途 |
|---|---|
| `messages: Message[]` | 消息历史列表 |
| `inputValue` / `uploadedImage` | 当前输入 |
| `isLoading` | 流式加载中 |

**关键函数**：

| 函数 | 职责 |
|---|------|
| `handleSendMessage()` | POST `/api/agent/chat`，用 `ReadableStream` 解析 SSE 事件，按 event 类型更新消息 |
| `updateMessage(id, updater)` | 不可变更新某条消息（React 状态更新模式）|

**要点**：
- SSE 解析用 `fetch` + `response.body.getReader()` 而非 `EventSource`，因为 `EventSource` 只支持 GET
- 6 类事件对应 `AgentMessage.tsx` 的 6 种 UI 渲染分支

### 5.12 frontend/src/components/UnifiedSearchInput.tsx

**作用**：统一搜索输入组件，集成文本输入、图片上传、召回/精排参数滑块、阈值调节。

**关键 Props**：

| Prop | 用途 |
|---|------|
| `textQuery` / `onTextChange` | 文本双向绑定 |
| `uploadedImage` / `onImageUpload` / `onImageRemove` | 图片上传/移除 |
| `recallTopK` / `rerankTopK` / `threshold` + onChange | 三参数滑块 |

**要点**：用 shadcn/ui 的 `Slider` `Tooltip` `Input` 组合，参数调节带 tooltip 提示，是参数化检索的典型 UI。

### 5.13 其余前端业务组件（按需读）

| 组件 | 行数 | 一句话职责 |
|---|---|---|
| `SearchInput.tsx` | 345 | 旧版搜索输入（保留兼容，新界面用 UnifiedSearchInput）|
| `AgentInput.tsx` | 135 | Agent 输入框 + 发送按钮 |
| `AgentMessage.tsx` | 324 | 单条消息渲染：思考链/结果卡片/总结/流式光标 |
| `ResultCard.tsx` | 159 | 结果卡片（带选中态、点击进详情）|
| `ResultDetail.tsx` | 185 | 结果详情弹窗（含图片大图与元数据）|
| `ComparisonPanel.tsx` | 144 | 多选结果对比面板 |
| `ProcessVisualization.tsx` | 115 | 检索过程三步骤进度条 |
| `QuickExamples.tsx` | 74 | 快捷示例查询 chips |
| `Header.tsx` | 50 | 顶部导航（检索/Agent 模式切换）|
| `ImageLightbox.tsx` | 62 | 图片灯箱（点击放大）|
| `figma/ImageWithFallback.tsx` | 28 | 带兜底的图片组件（加载失败显示占位图）|

### 5.14 frontend/src/utils/*（演示模式模拟器）

**作用**：三个工具文件用于**演示模式**——不连后端时也能在前端模拟检索/Agent 响应的进度与结果。

| 文件 | 关键导出 | 用途 |
|---|---|---|
| `mockData.ts` | `generateMockResults(count, query)` / `PRESET_MOCK_RESULTS` | 生成假检索结果 |
| `processSimulator.ts` | `simulateProgress()` / `simulateRetrievalProcess()` / `delay()` | 模拟进度条更新 |
| `agentSimulator.ts` | `simulateAgentResponse()` / `addThinkingItem()` | 模拟 Agent 思考链与流式输出 |

**要点**：真实检索走 `/api/search`，真实 Agent 走 `/api/agent/chat`，三个 simulator 主要服务于**离线演示/开发调试**。

---

## 6. 关键设计模式解析

### 6.1 单例模式（检索引擎 + Agent + LLM）

```
backend/main.py
    ├─ retrieval_engine = Qwen3VLRetrievalEngine()  ← 模块级单例
    └─ agent_manager = AgentManager()                ← 模块级单例
            │
            ▼
    lifespan 启动时 initialize() 一次
            │
            ▼
    所有请求共享同一实例（避免重复加载 2B 模型）
```

**意图**：Qwen3-VL-Embedding-2B 加载耗时数十秒、占用数 GB 显存，绝不能每个请求重建。`__new__` 实现的单例 + `initialized` 标志位双重保险。

### 6.2 双索引系统（图搜图 vs 文搜图分离）

```
                  图片库 (data/images/*.png)
                          │
                  ┌───────┴───────┐
                  ▼               ▼
        ┌──────────────┐  ┌──────────────────┐
        │ 图片向量索引  │  │ 混合检索索引      │
        │ Milvus:       │  │ ├─ Qwen3-VL 向量  │
        │ qwen3_vl_     │  │ │  Milvus:        │
        │ image_only    │  │ │  qwen3_vl_      │
        │ (仅 image     │  │ │  hybrid_agent   │
        │  embedding)   │  │ ├─ BM25 Retriever │
        └──────┬───────┘  │ │  (关键词检索)   │
               │          └──────┬───────────┘
               │                 │
        用于图搜图          用于文搜图
        image_to_image_search   text_to_image_search
```

**意图**：图搜图只需要图片向量比对，文搜图需要文本→图片跨模态 + 关键词双路召回，两种检索模式数据结构需求不同，分索引避免互相干扰。

### 6.3 共享 Reranker 模型

```
initialize():
    temp_processor = Qwen3VLNodePostprocessor()  ← 加载 Reranker 模型（耗时数秒）
    self.reranker_model = temp_processor.model  ← 提取模型实例
                │
                ▼
每次检索:
    reranker = Qwen3VLNodePostprocessor(
        top_n=recall_top_k,
        model_instance=self.reranker_model  ← 复用，不再重新加载
    )
```

**意图**：Qwen3-VL-Reranker-2B 是 2B 参数大模型，每次检索重新加载会拖慢响应。共享模型实例 + 轻量 postprocessor 包装是性能与代码复用的平衡。

### 6.4 三路召回 + RRF 融合

```
text_to_image_search(query):
    retrievers = [
        Qwen3-VL 向量 retriever,   ← 跨模态语义检索
        BM25 retriever,            ← 关键词字面检索
        # caption_index retriever  ← (本项目已禁用)
    ]
                │
                ▼
    QueryFusionRetriever(
        retrievers=retrievers,
        mode="reciprocal_rerank"  ← RRF 融合算法
    )
                │
                ▼
    Reranker 精排
                │
                ▼
    阈值过滤 + 截断到 rerank_top_k
```

**意图**：向量检索捕捉语义但漏字面匹配，BM25 反之，RRF 互补提升召回率。

### 6.5 SSE 事件流协议

```
后端 AgentManager.chat_stream         前端 AgentChat.handleSendMessage
        │                                       │
        │  event: thinking                      │
        ├──────────────────────────────────────►│ 渲染"正在分析..."
        │                                       │
        │  event: tool_call                    │
        ├──────────────────────────────────────►│ 渲染工具调用气泡
        │                                       │
        │  event: process (progress=50)        │
        ├──────────────────────────────────────►│ 进度条推进到 50%
        │                                       │
        │  event: results                      │
        ├──────────────────────────────────────►│ 渲染结果卡片网格
        │                                       │
        │  event: process (progress=100)       │
        ├──────────────────────────────────────►│ 进度条完成
        │                                       │
        │  event: summary                      │
        ├──────────────────────────────────────►│ 流式渲染总结文本
        │                                       │
        │  event: complete                     │
        └──────────────────────────────────────►│ 结束流式状态
```

**意图**：把 Agent 推理过程拆解为 6 类事件实时推送，让用户看见"AI 在思考、在调工具、在出结果"，而不是等几十秒后突然蹦出答案。这是 RAG 系统体验设计的关键模式。

### 6.6 LlamaIndex 全局 embed_model 覆盖

```python
# retrieval.py: initialize()
Settings.embed_model = self.embed_adapter  # 关键一行
```

**意图**：LlamaIndex 在某些路径（如 `RetrieverQueryEngine.query`）会 fallback 到默认 OpenAI embedding，消耗 API 额度且与本地 Qwen3-VL 维度不匹配。显式覆盖 `Settings.embed_model` 强制使用本地模型，是 LlamaIndex 多模态项目的必备配置。

---

## 7. 配置系统详解

### 7.1 配置项全表

| 配置项 | 来源 | 默认值 | 说明 |
|---|---|---|---|
| `OPENAI_API_KEY` | `.env` | — | OpenAI / 兼容代理的 API Key |
| `OPENAI_BASE_URL` | `.env` | `https://api.openai.com/v1` | LLM 端点，可指向国内代理 |
| `MILVUS_URI` | `.env` | `http://localhost:19530` | Milvus 连接串 |
| `IMAGE_DIR` | `.env` | `backend/data/images` | 被检索的图片库路径 |
| `MODELSCOPE_CACHE` | 环境变量 | `backend/data/models` | ModelScope 模型缓存目录 |
| `USE_MOCK_DATA` | 环境变量 | `false` | Mock 模式开关 |
| `HOST` / `PORT` | 代码硬编码 | `0.0.0.0:3001` | 后端监听 |
| `ALLOWED_ORIGINS` | 代码硬编码 | `["*"]` | CORS 白名单 |
| `EMBEDDING_DIM` | 代码硬编码 | `512` | 嵌入向量维度 |

### 7.2 加载优先级

```
代码硬编码默认值
    ↑ （最低）
.env 文件 (load_dotenv override=True)
    ↑ （覆盖代码默认）
系统环境变量
    ↑ （最高，因 override=True 不覆盖系统变量则反过来）
```

**要点**：`load_dotenv(override=True)` 的 `override` 参数表示 `.env` 文件**覆盖**已存在的环境变量；若想系统环境变量优先，改为 `override=False`。

### 7.3 .env 文件示例

```bash
# 必填
OPENAI_API_KEY=sk-xxxxx

# 可选（有默认值）
OPENAI_BASE_URL=https://api.openai.com/v1
MILVUS_URI=http://localhost:19530
IMAGE_DIR=/path/to/your/images
USE_MOCK_DATA=false
```

---

## 8. 如何运行和测试

### 8.1 环境要求

| 组件 | 版本 | 默认端口 | 说明 |
|---|---|---|---|
| Python | 3.10+ | — | 见 `backend/requirements.txt` |
| Node.js | 18+ | — | 见 `frontend/package.json` |
| Docker | 20+ | — | 用于跑 Milvus |
| Milvus | v2.3.21 | 19530 | 向量数据库 |
| GPU（可选）| CUDA 11.8+ | — | 无 GPU 则 CPU 推理（慢 10-50 倍）|

### 8.2 启动步骤

```bash
# 0️⃣ 克隆项目
cd D:/1/GalleryMind

# 1️⃣ 启动 Milvus（在项目根目录）
docker-compose up -d
# 验证：docker ps 看到 milvus-standalone / milvus-etcd / milvus-minio 三个容器

# 2️⃣ 准备图片库
mkdir -p backend/data/images
# 把你的图片（.png/.jpg/.jpeg/.gif）放到 backend/data/images/

# 3️⃣ 配置 .env
cd backend
cat > .env <<'EOF'
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MILVUS_URI=http://localhost:19530
USE_MOCK_DATA=false
EOF

# 4️⃣ 安装后端依赖（建议用 venv）
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# 5️⃣ 启动后端（首次启动会下载 Qwen3-VL 模型，约 4-8GB，需 10-30 分钟）
python -m backend.main
# 或：python main.py
# 看到以下日志说明就绪：
# 🚀 Server starting on 0.0.0.0:3001
# ✅ 检索引擎和Agent就绪

# 6️⃣ 启动前端（另开终端）
cd frontend
npm install
npm run dev
# 浏览器自动打开 http://localhost:3000

# 7️⃣ 体验检索
#    - 检索模式：输入文本或上传图片，点"开始检索"
#    - Agent 模式：顶部切换到 Agent，自然语言对话
```

### 8.3 测试与调试

```bash
# 健康检查
curl http://localhost:3001/api/health
# {"status":"ok","service":"multimodal-rag-backend"}

# 同步检索（Mock 模式，不加载模型）
USE_MOCK_DATA=true python -m backend.main
curl -X POST http://localhost:3001/api/search \
  -H "Content-Type: application/json" \
  -d '{"searchMode":"文搜图","textQuery":"测试","recallTopK":20,"rerankTopK":5,"threshold":0.0}'

# Agent SSE 流式
curl -N -X POST http://localhost:3001/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"帮我找一张红色跑车的图片"}'

# 用项目自带的调试脚本
cd backend
python debug_retrieval.py    # 测检索
python debug_agent_small.py # 测 Agent 小图
python debug_agent_post.py   # 测 Agent POST
```

**要点**：
- 首次启动后端会从 ModelScope 下载 `Qwen3-VL-Embedding-2B`（约 4GB）和 `Qwen3-VL-Reranker-2B`（约 4GB）到 `backend/data/models/`
- 无 GPU 时 `device="cpu"`，单次检索可能需 30-60 秒；有 GPU 时 1-3 秒
- Vite 开发服务器通过 `vite.config.ts` 的 `proxy` 把 `/api` 和 `/static` 转发到 `:3001`，前端代码可用相对路径 `fetch('/api/search')`

---

## 9. 复刻建议与学习路线

### 9.1 分阶段复刻路线

| 阶段 | 目标 | 耗时估计 | 关键技能 |
|---|---|---|---|
| 1 | 跑通 Mock 模式前后端 | 0.5 天 | React + FastAPI 基础 |
| 2 | 跑通 Milvus + 同步检索 | 1-2 天 | Docker + LlamaIndex + Milvus |
| 3 | 集成 Qwen3-VL 嵌入 + Reranker | 2-3 天 | PyTorch + Transformers + 模型加载 |
| 4 | 接入 LangGraph Agent + SSE | 2-3 天 | LangChain 1.x + LangGraph |
| 5 | 完整前端聊天+检索 UI | 3-5 天 | React 状态管理 + SSE 解析 + shadcn/ui |

### 9.2 学习资源表

| 主题 | 推荐资源 |
|---|---|
| LlamaIndex 多模态 | [LlamaIndex MultiModal Docs](https://docs.llamaindex.ai/en/stable/use_cases/multimodal/) |
| Milvus 向量库 | [Milvus Quickstart](https://milvus.io/docs/install_standalone-docker.md) |
| Qwen3-VL 模型 | [ModelScope Qwen3-VL-Embedding-2B](https://modelscope.cn/models/qwen/Qwen3-VL-Embedding-2B) |
| LangGraph Agent | [LangGraph create_agent Docs](https://langchain-ai.github.io/langgraph/) |
| SSE 协议 | [MDN Server-Sent Events](https://developer.mozilla.org/zh-CN/docs/Web/API/Server-sent_events) |
| RRF 融合算法 | [Reciprocal Rank Fusion 论文](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) |

### 9.3 推荐阅读路径

1. 先读 `ZHIDAO.md` 第 1-4 章建立全局心智模型（本文档前半部分）
2. 跑通 Mock 模式，理解前后端数据流（`config.py` → `schemas.py` → `main.py` → `routers/search.py`）
3. 深入 `core/retrieval.py`，理解双索引与三路召回（结合第 6 章设计模式）
4. 深入 `core/agent.py`，理解 LangGraph Agent 与 SSE 事件分发
5. 读前端 `App.tsx` 与 `AgentChat.tsx`，理解 SSE 事件消费与 UI 渲染
6. 按需读其余组件，理解 UI 细节

---

## 10. 常见问题

### Q1: 为什么启动后端第一次很慢？

**A**：首次启动会从 ModelScope 下载 `Qwen3-VL-Embedding-2B`（约 4GB）和 `Qwen3-VL-Reranker-2B`（约 4GB）到 `backend/data/models/`，下载速度取决于网络。后续启动从本地缓存加载，约 30-60 秒（加载模型到显存）。

### Q2: 没有 GPU 能跑吗？

**A**：能。`Qwen3VLEmbedding.__init__` 自动检测 device：CUDA → MPS → CPU。CPU 模式下单次检索可能 30-60 秒（vs GPU 的 1-3 秒），适合功能验证不适合高频调用。

### Q3: 为什么 Caption 索引被禁用了？

**A**：代码注释（`retrieval.py` ~L881-L897）解释了：Qwen3-VL 处理 37622 个文本块需要 5+ 小时（每次都要过模型前向），改用 BM25 做关键词检索替代，性能与效果平衡更好。

### Q4: Agent 多轮记忆能跨会话吗？

**A**：不能。`InMemorySaver` 是进程内存 checkpointer，重启即丢失；且 `thread_id="session_default"` 是全局共享的，多用户场景下会串话。生产环境应换 `PostgresSaver` 或 `RedisSaver` 并按用户分配 thread_id。

### Q5: Milvus 数据存在哪？怎么清理？

**A**：Docker Compose 把 Milvus 数据挂在 `./volumes/milvus/`，etcd 在 `./volumes/etcd/`，MinIO 在 `./volumes/minio/`。清理：`docker-compose down -v` 删容器与卷，或手动删 `volumes/` 目录。

### Q6: 前端能用 EventSource 而非 fetch 消费 SSE 吗？

**A**：不能。`EventSource` 只支持 GET 请求，而 Agent 接口需要 POST（带 image base64 payload 大）。所以 `AgentChat.tsx` 用 `fetch` + `response.body.getReader()` 手动解析 SSE 格式。

### Q7: 如何切换 OpenAI 兼容代理（如 Azure、国内中转）？

**A**：在 `.env` 设置 `OPENAI_BASE_URL=https://your-proxy.com/v1`，代码用 `ChatOpenAI(base_url=OPENAI_BASE_URL)` 自动适配，只要代理兼容 OpenAI Chat Completions API 即可。

### Q8: 为什么检索结果里 `score` 是 0-100 而 `relevanceScore` 是 0-1？

**A**：`_format_results` 中 `score = round(float(res.score) * 100, 2)`，前端展示用 0-100 更直观；`relevanceScore` 保留原始 0-1 相似度供程序使用。两者表达同一信息，是历史决策。

---

## 附录：关键术语对照表

| 英文 | 中文 | 说明 |
|---|---|---|
| Embedding | 嵌入向量 | 把文本/图片编码为定长向量 |
| Retriever | 检索器 | 从向量库/BM25 中召回候选 |
| Reranker | 精排器 | 对召回结果重新打分排序 |
| Recall | 召回 | 第一阶段快速捞大批候选 |
| Rerank | 精排 | 第二阶段慢但准挑少数 |
| Vector Store | 向量库 | 存向量并支持相似度检索的数据库 |
| Multi-Modal | 多模态 | 同时处理文本+图片等不同模态 |
| QueryFusionRetriever | 融合检索器 | LlamaIndex 的多检索器融合组件 |
| RRF | 倒数排名融合 | Reciprocal Rank Fusion，多检索器结果融合算法 |
| BM25 | 文本检索算法 | TF-IDF 进化版，基于词频与逆文档频率 |
| Agent | 智能体 | LLM + 工具调用的自主推理循环 |
| Checkpointer | 检查点保存器 | LangGraph 的状态持久化组件 |
| SSE | 服务器推送事件 | HTTP 长连接流式协议 |
| L2 Normalization | L2 归一化 | 向量除以其模长，转为单位向量 |
| IP | 内积相似度 | Inner Product，Milvus 的一种相似度度量 |

---

**要点（全文总结）**：
1. GalleryMind 是一个**完整的工业级多模态 RAG 样本项目**，覆盖检索与 Agent 两条路径
2. 核心架构 = **Qwen3-VL 嵌入 + Milvus 双索引 + BM25 + RRF 融合 + Qwen3-VL Reranker + LangGraph Agent + SSE 流式**
3. 推荐阅读顺序：`config.py` → `schemas.py` → `main.py` → `routers/*` → `core/agent.py` → `core/retrieval.py`（最大最关键）→ 前端 `App.tsx` → `AgentChat.tsx`
4. 跑通项目最关键的三步：`docker-compose up -d`（Milvus）+ 配置 `.env`（OpenAI Key）+ `python -m backend.main`（首次会下载模型）
