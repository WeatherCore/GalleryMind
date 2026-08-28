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
- 前端是「侧栏导航 + 四页面」SPA（React Router）：检索 `/`、Agent 对话 `/agent`、图库管理 `/gallery`、系统状态 `/status`，全部通过 `frontend/src/api.ts` 这一个客户端模块访问后端

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
│   │   ├── retrieval.py          🔐 检索引擎（800行，项目核心）
│   │   ├── agent.py              🔐 Agent 管理（LangGraph + SSE 流式）
│   │   └── utils.py              🔧 base64 图片落地 + URL 转换
│   ├── routers/                  📋 FastAPI 路由层（5 个模块）
│   │   ├── search.py             🔐 /api/search 同步检索接口（成功后写历史）
│   │   ├── agent.py              🔐 /api/agent/* SSE 流式接口 + 会话重置
│   │   ├── system.py             🔐 图库管理 + /api/system/status 状态巡检
│   │   ├── history.py            🔐 /api/history 检索历史（SQLite 持久化）
│   │   └── health.py             🧪 /api/health + /api/ping 健康检查
│   ├── data/                     📂 运行时数据（自动创建）
│   │   ├── images/               📂 图片库（被检索的对象）
│   │   ├── uploads/              📂 用户上传的临时图片
│   │   ├── models/               📂 ModelScope 下载的模型缓存
│   │   ├── caption_cache/        📂 图片文本描述缓存（.txt）
│   │   └── history.db            📂 检索历史数据库（SQLite，自动创建）
│   ├── .env.example              🧪 环境变量模板（含 Agent/Vision 独立配置示例）
│   ├── debug_*.py                🧪 调试脚本（不参与主流程）
│   └── test_*.py                 🧪 测试脚本（不参与主流程）
│
├── frontend/                     ⭐ 前端：React 18 + Vite + TS + Tailwind CSS 4（全新重写）
│   ├── package.json              📋 依赖清单（react-router-dom / tailwindcss / phosphor-icons）
│   ├── vite.config.ts            🔧 Vite 配置（:3000，/api 与 /static 代理到 3001）
│   ├── index.html                📋 HTML 入口
│   └── src/
│       ├── main.tsx              📋 React 入口（createRoot.render）
│       ├── App.tsx               🔐 应用骨架（侧栏布局 + 4 条路由）
│       ├── api.ts                🔐 后端 API 客户端（全部接口封装 + SSE 解析）
│       ├── types.ts              📋 TypeScript 类型定义（与 schemas.py 一一对齐）
│       ├── index.css             🎨 Tailwind 主题令牌（亮色极简单主题）
│       ├── components/           ⭐ 通用组件
│       │   ├── Sidebar.tsx       🔐 侧栏导航 + 后端健康呼吸灯
│       │   ├── ImageUploader.tsx 🔐 图片上传（预览/移除/紧凑模式）
│       │   └── Lightbox.tsx      🔐 大图查看（以图搜图/删除/元信息）
│       └── pages/                ⭐ 四大页面
│           ├── SearchPage.tsx    🔐 检索页（三模式 + 参数滑块 + 历史面板）
│           ├── AgentPage.tsx     🔐 Agent 对话页（思考链/工具调用/结果渲染）
│           ├── GalleryPage.tsx   🔐 图库页（上传/删除/热重建索引）
│           └── StatusPage.tsx    🔐 状态页（Milvus/引擎/模型/图库巡检）
│
├── volumes/                      📂 Milvus Docker 数据卷（自动生成）
├── docker-compose.yml            🔧 Milvus 三件套（etcd + minio + milvus）
└── ZHIDAO.md / README.md / Description.md  📋 本套导读文档
```

**emoji 标记说明**：⭐ 重点目录 / 🔐 核心业务文件 / 🔧 配置/工具 / 📋 普通/契约文件 / 🧪 测试/调试 / 📂 运行时数据

**要点**：
- 真正承载项目意图的源码约 27 个（backend 15 + frontend 12），前端无 UI 模板库依赖，全部页面级组件手写
- 前端**没有** mock/simulator 工具——"演示模式"由后端 `USE_MOCK_DATA=true` 承担（不加载模型直接返回预设数据），前端代码只写真实接口
- `backend/data/` 下的子目录在 `config.py` 中通过 `mkdir(parents=True, exist_ok=True)` 自动创建，`history.db` 由 `routers/history.py` 首次访问时建表

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
| 入门 | 1 | `backend/config.py` | 74 | 5 分钟 | 看清配置项与路径布局（含 Agent/Vision 双模型）|
| 入门 | 2 | `backend/schemas.py` | 70 | 5 分钟 | 看清前后端数据契约 |
| 入门 | 3 | `backend/main.py` | 131 | 10 分钟 | 看清启动流程（lifespan + 5 路由注册）|
| 入门 | 4 | `backend/core/utils.py` | 68 | 5 分钟 | 看清图片落地与 URL 转换 |
| 入门 | 5 | `backend/routers/health.py` | 23 | 2 分钟 | 健康检查（最简单的路由样板）|
| 进阶 | 6 | `backend/routers/search.py` | 162 | 15 分钟 | 同步检索接口（含 Mock + 历史落库）|
| 进阶 | 7 | `backend/routers/agent.py` | 58 | 10 分钟 | SSE 流式接口 + 会话重置 |
| 进阶 | 8 | `backend/routers/system.py` | 230 | 15 分钟 | 图库管理 + 只读状态巡检 |
| 进阶 | 9 | `backend/routers/history.py` | 99 | 5 分钟 | SQLite 检索历史 |
| 核心 | 10 | `backend/core/agent.py` | 382 | 30 分钟 | Agent 编排与 SSE 事件分发 |
| 核心 | 11 | `backend/core/retrieval.py` | 800 | 60-90 分钟 | **项目大脑**，建议分块读 |
| 前端 | 12 | `frontend/src/types.ts` | 97 | 5 分钟 | 前端类型契约 |
| 前端 | 13 | `frontend/src/api.ts` | 199 | 15 分钟 | API 客户端 + SSE 流解析（前端唯一出入口）|
| 前端 | 14 | `frontend/src/App.tsx` | 71 | 5 分钟 | 应用骨架与路由表 |
| 前端 | 15 | `frontend/src/pages/SearchPage.tsx` | 546 | 30 分钟 | 检索页（最重的页面）|
| 前端 | 16 | `frontend/src/pages/AgentPage.tsx` | 397 | 20 分钟 | Agent 对话页（SSE 事件渲染）|
| 前端 | 17 | 其余页面与组件（Gallery/Status/Sidebar 等）| ~700 | 40 分钟 | 按需读 |

### 5.1 backend/config.py

**作用**：项目配置中枢，从 `.env` 加载敏感信息（API Key、Milvus URI），定义路径布局（图片库、上传、模型缓存、caption 缓存），并自动创建必要目录；同时定义 Agent 与 Vision 两组可独立配置的 LLM 变量。

**关键数据结构**：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BASE_DIR` | `backend/` | 后端根目录 |
| `DATA_DIR` | `backend/data` | 运行时数据根 |
| `UPLOAD_DIR` | `backend/data/uploads` | 用户上传图片落地 |
| `MODEL_CACHE_DIR` | `$MODELSCOPE_CACHE` 或 `backend/data/models` | ModelScope 模型缓存 |
| `CAPTION_CACHE_DIR` | `backend/data/caption_cache` | 图片文本描述缓存 |
| `DEFAULT_IMAGE_DIR` | `$IMAGE_DIR` 或 `data/images` | 被检索的图片库 |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | 来自 `.env` / `https://api.openai.com/v1` | 旧版兼容配置（回退用）|
| `AGENT_LLM_MODEL` | `gpt-4o` | Agent 推理模型（需支持 function calling）|
| `AGENT_LLM_API_KEY` / `AGENT_LLM_BASE_URL` | 回退 `OPENAI_*` | Agent 模型凭证与端点 |
| `VISION_LLM_MODEL` | `gpt-4o` | Vision 视觉模型（describe_image 用）|
| `VISION_LLM_API_KEY` / `VISION_LLM_BASE_URL` | 回退 `OPENAI_*` | Vision 模型凭证与端点 |
| `MILVUS_URI` | `http://localhost:19530` | Milvus 连接串 |
| `EMBEDDING_MODEL_MS` | `qwen/Qwen3-VL-Embedding-2B` | 嵌入模型 ID |
| `RERANKER_MODEL_MS` | `qwen/Qwen3-VL-Reranker-2B` | 精排模型 ID |
| `HOST/PORT` | `0.0.0.0:3001` | 服务监听 |
| `ALLOWED_ORIGINS` | `["*"]` | CORS 全开（开发友好）|

**要点**：
- `load_dotenv(override=False)`：`.env` **只补缺**，已存在的系统环境变量优先——这是 `USE_MOCK_DATA=true python -m backend.main` 这类命令行临时开关能生效的原因（否则 `.env` 里写死的 `USE_MOCK_DATA=false` 会反过来覆盖命令行）
- `AGENT_LLM_*` / `VISION_LLM_*` 未设置时逐项回退 `OPENAI_API_KEY` / `OPENAI_BASE_URL`，兼容存量 `.env`
- 四个 `mkdir(parents=True, exist_ok=True)` 保证目录存在，避免运行时 `FileNotFoundError`
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
| `AgentChatRequest` | `/api/agent/chat` 请求 | `message`、`image`(base64)、`sessionId` |
| `SessionResetRequest` | `/api/agent/session/reset` 请求 | `sessionId` |
| `InitRequest/InitResponse` | （预留）初始化接口 | `milvus_uri` |

**要点**：
- `searchMode` 用中文字符串枚举（`'文搜图' | '图搜图' | '混合搜索'`），后端 `retrieval_engine.search()` 直接 if/elif 分发，简单直接但扩展性一般；前端 `types.ts` 的 `SEARCH_MODES` 常量与之严格对齐
- `score` 字段 0-100（前端展示用），`relevanceScore` 字段 0-1（原始相似度），两者并存是历史决策
- `AgentChatRequest` 带 `sessionId`，作为 LangGraph 的 `thread_id` 实现多轮记忆隔离；真实数据走 SSE 流，Schema 主要用于请求校验和文档生成

### 5.3 backend/main.py

**作用**：FastAPI 应用入口，组装中间件、静态文件、路由，并通过 `lifespan` 在启动时预热检索引擎与 Agent。

**关键函数**：

| 函数 | 行号 | 职责 |
|------|------|------|
| `lifespan(app)` | ~43 | 启动时初始化 retrieval_engine 和 agent_manager；`USE_MOCK_DATA=true` 时跳过（Mock 守卫，纯前端联调不依赖 GPU/Milvus）；关闭时打印日志 |
| `root()` | ~124 | `GET /` 返回 `{"status":"ok"}` 用于探活 |

**要点**：
- 用 `lifespan` 替代弃用的 `@app.on_event("startup")`，是 FastAPI 0.93+ 推荐写法
- 路由注册顺序：`health`（优先，保证探活靠前）→ `search` → `agent` → `system` → `history`
- `app.mount("/static", StaticFiles(directory=DATA_DIR))` 把 `backend/data` 公开成网址，与 `core/utils.py` 的 `get_image_url()` 强耦合（改挂载点必须两边同步）
- `sys.path.insert(0, parent_dir)` 让 `python main.py` 和 `python -m backend.main` 两种启动方式都能正确解析 `from backend.config import ...`

### 5.4 backend/core/utils.py

**作用**：图片处理工具，负责把前端上传的 base64 图片落地为本地文件，以及把本地路径转换为前端可访问的 `/static/...` URL。

**关键函数**：

| 函数 | 行号 | 职责 |
|------|------|------|
| `save_base64_image(base64_str, save_dir)` | ~18 | 去掉 `data:image/...;base64,` header → `base64.b64decode` → PIL 打开 → 用 `uuid4().hex` 生成唯一名 → 保存为 PNG |
| `get_image_url(file_path)` | ~47 | `file_path.relative_to(DATA_DIR)` → `/static/<relative>`；若不在 DATA_DIR 下，回退到 `/static/images/<filename>` |

**要点**：
- 用 `uuid4().hex` 命名避免重名碰撞，但所有上传图都转 PNG（即使原图是 JPG），有轻微存储冗余
- `get_image_url` 假设 `StaticFiles(directory=DATA_DIR)` 挂载在 `/static`，与 `main.py` 中的 `app.mount("/static", ...)` 强耦合

### 5.5 backend/core/agent.py

**作用**：Agent 编排核心，封装 LangGraph `create_agent`，提供 SSE 流式输出与会话重置。

**关键类与函数**：

| 名称 | 行号 | 职责 |
|------|------|------|
| `@tool search_images(query, image_path)` | ~56 | 工具：调 retrieval_engine 做检索，返回 JSON 字符串（含 success/results）|
| `@tool describe_image(image_path)` | ~94 | 工具：用 Vision LLM 看图说话，返回中文描述 |
| `class AgentManager` | ~127 | 单例（`__new__` 实现），管理 agent 实例与配置 |
| `AgentManager.initialize()` | ~146 | 创建 `_agent_llm`（读 `AGENT_LLM_*`）与 `_vision_llm`（读 `VISION_LLM_*`）、注册 tools、定义 system_prompt、调用 `create_agent` + `InMemorySaver` checkpointer |
| `AgentManager.chat_stream(query, image_path, session_id)` | ~194 | SSE 流式入口：发 thinking → 调 `_run_agent_async` → 发 complete；`session_id` 即 LangGraph `thread_id` |
| `AgentManager.reset_session(session_id)` | ~251 | 重置指定会话的记忆（配合前端"新对话"按钮）|
| `AgentManager._run_agent_async(content)` | ~268 | 遍历 `agent.stream(stream_mode="values")`，按消息类型分发 6 类 SSE 事件 |
| `AgentManager._format_sse(event_type, data)` | ~367 | 拼接 `event: X\ndata: Y\n\n` 格式字符串 |
| `_ts()` | ~377 | 返回 `str(time.time())` 时间戳字符串 |

**要点**：
- **Agent / Vision 双模型独立配置**：`initialize()` 用 `AGENT_LLM_MODEL/API_KEY/BASE_URL` 创建 Agent 推理模型，用 `VISION_LLM_*` 创建 `describe_image` 用的视觉模型；未设置的项逐级回退 `OPENAI_*`（见 5.1）
- `InMemorySaver` checkpointer 让 Agent 具备多轮记忆能力；`thread_id` 来自前端传入的 `sessionId`（前端"新对话"时生成新 id 并调 `POST /api/agent/session/reset`），比旧版全局共享 `session_default` 已进步，但仍未按用户体系隔离
- SSE 事件类型与前端 `api.ts` 的 `streamAgentChat` / `pages/AgentPage.tsx` 的事件处理严格对应：`thinking/tool_call/process/results/summary/complete`
- `await asyncio.sleep(0.01)` 在事件循环中让出控制权，模拟流式效果（实际 LangGraph stream 是同步 generator）

### 5.6 backend/core/retrieval.py（项目核心，800 行）

**作用**：多模态检索引擎，封装 Qwen3-VL 嵌入、Milvus 双索引、BM25、Qwen3-VL Reranker，提供统一检索入口。

**辅助函数**：

| 函数 | 行号 | 职责 |
|------|------|------|
| `download_model_from_modelscope(model_id, cache_dir)` | ~50 | 从 ModelScope 下载模型，失败回退到原始 ID |
| `load_caption_cache(cache_dir)` | ~70 | 加载 `caption_cache/*.txt` 为 `{图片名: 描述}` 字典 |

**模型类**：

| 类 | 行号 | 职责 |
|---|------|------|
| `Qwen3VLEmbedding` | ~100 | 封装 Qwen3-VL-Embedding-2B：自动选 device（CUDA/MPS/CPU）、L2 归一化、截断到 `output_dim=512` |
| `Qwen3VLMultiModalEmbedding(MultiModalEmbedding)` | ~268 | LlamaIndex 适配器：把 `Qwen3VLEmbedding` 包装成 LlamaIndex 标准 MultiModalEmbedding 接口（含 async 系列方法）|
| `Qwen3VLNodePostprocessor(BaseNodePostprocessor)` | ~322 | LlamaIndex 节点后处理器：封装 Qwen3-VL-Reranker-2B，支持文本/图片/混合 query 精排 |
| `Qwen3VLRetrievalEngine` | ~454 | **单例检索引擎**：构建双索引、提供 `search()` 统一入口 |

**Qwen3VLRetrievalEngine 关键方法**：

| 方法 | 行号 | 职责 |
|---|------|------|
| `__new__` | ~459 | 单例模式：第一次创建设 `initialized=False`，后续返回同一实例 |
| `initialize(image_dir, use_reranker=True)` | ~485 | 加载 embedder → 设置全局 `Settings.embed_model` → 加载 caption → 读取图片 → `_build_indices` → 加载共享 Reranker |
| `_build_indices(documents, image_dir)` | ~535 | 构建**双索引系统**：[1] 图片向量索引（图搜图）[2] 混合检索索引（Qwen3-VL 向量 + BM25）|
| `search(mode, query, image_path, ...)` | ~650 | 统一入口：按 mode 分发到 `text_to_image_search` 或 `image_to_image_search` |
| `text_to_image_search(query, ...)` | ~672 | 文搜图：`QueryFusionRetriever` 融合 Qwen3-VL 向量 + BM25，可选 Reranker 精排 |
| `image_to_image_search(image_path, query, ...)` | ~723 | 图搜图/混合搜：query 图 embedding → Milvus 检索 → 过滤自身 → Reranker 精排 |
| `_format_results(results)` | ~777 | 转换为前端契约：`id/imageUrl/title/score/relevanceScore/metadata` |

**要点**（项目最关键的设计决策都在这里）：
- **双索引系统**：图片向量索引（`qwen3_vl_image_only` 集合）专给图搜图用，混合索引（`qwen3_vl_hybrid_agent` 集合 + BM25）专给文搜图用，避免一种索引兼顾两种检索模式
- **共享 Reranker 模型**：`temp_processor = Qwen3VLNodePostprocessor()` 加载模型后提取 `self.reranker_model`，后续每次检索创建轻量 postprocessor 时通过 `model_instance=` 复用，避免重复加载 2B 大模型
- **L2 归一化 + 截断到 512 维**：`embedding / np.linalg.norm(embedding)` + `embedding[:output_dim]`，配合 Milvus 的 `similarity_metric="IP"`（内积）实现余弦相似度
- **全局 `Settings.embed_model` 覆盖**：`Settings.embed_model = self.embed_adapter` 防止 LlamaIndex 在某些路径（如 `query_engine.query`）fallback 到默认 OpenAI embedding，避免消耗 API 额度
- **Caption 索引已禁用**：代码注释解释了原因——Qwen3-VL 处理 37622 个文本块需 5+ 小时，改用 BM25 做文本关键词检索替代
- **图搜图过滤自身**：`Path(r.node.metadata['file_path']).resolve() != query_path.resolve()` 防止返回 query 图自己
- **Reranker 全量重排**：`top_n=recall_top_k`（不是 `rerank_top_k`），保证阈值过滤有足够候选

### 5.7 backend/routers/search.py

**作用**：同步检索 HTTP 接口，支持 Mock 模式（不加载模型）与真实模式（调 retrieval_engine），检索成功后写检索历史。

**关键函数**：

| 函数 | 行号 | 职责 |
|---|------|------|
| `search(request: SearchRequest)` | ~101 | 入口：Mock 模式直接返回 `MOCK_RESULTS[:rerankTopK]`；真实模式调 `retrieval_engine.search()` 转换为 `SearchResponse`；返回前调 `record_search()` 落库历史 |

**要点**：
- `USE_MOCK_DATA` 环境变量控制 Mock 开关，用于前端联调（不依赖 GPU 模型）
- 检索成功后调 `routers/history.py` 的 `record_search(query, mode, result_count, total_time, mock)` 记录一条历史
- 异常捕获 `traceback.print_exc() + raise HTTPException(500)`，便于排查
- `processSteps=[]` 始终返回空数组，检索进度展示由前端 SearchPage 的骨架屏/pipeline 状态承担

### 5.8 backend/routers/agent.py

**作用**：SSE 流式 Agent 聊天接口 + 会话重置。

**关键函数**：

| 函数 | 行号 | 职责 |
|---|------|------|
| `chat_sse(request: AgentChatRequest)` | ~22 | 接收 message + image（base64）+ sessionId，若有图片用 `save_base64_image` 落地，返回 `StreamingResponse(media_type="text/event-stream")` |
| `reset_session(req: SessionResetRequest)` | ~56 | `POST /api/agent/session/reset`：调 `agent_manager.reset_session(session_id)` 清空指定会话记忆 |

**要点**：
- 用 POST 而非 GET，因为图片 base64 payload 较大，GET 受 URL 长度限制
- 图片保存失败时 `try/except: pass` 静默降级，Agent 仍可纯文本对话
- `media_type="text/event-stream"` 是 SSE 标准 MIME，浏览器 `EventSource` 与 `fetch` 都可消费

### 5.9 backend/routers/system.py

**作用**：前端"图库页"与"状态页"的数据源，兼图片库管理（列表 / 上传 / 删除 / 热重建索引 / 状态巡检）。

**关键函数**：

| 函数 | 行号 | 职责 |
|---|------|------|
| `list_images()` | ~100 | `GET /api/images`：扫描 `DEFAULT_IMAGE_DIR`，返回文件名/URL/大小/时间/caption（与检索引擎同源，保证图库页看到的 = 引擎能搜到的）|
| `upload_image(file)` | ~108 | `POST /api/images/upload`：multipart 上传入库 |
| `delete_image(filename)` | ~138 | `DELETE /api/images/{filename}`：删图 |
| `reindex()` | ~167 | `POST /api/images/reindex`：热重建向量索引，`threading.Lock` 保证同一时刻只允许一个重建任务 |
| `system_status()` | ~202 | `GET /api/system/status`：汇总后端版本 / Milvus 连通性（socket 探测）/ 引擎初始化 / 四组模型配置 / 图库统计 |

**要点**：
- 文件名白名单正则 `^[\w\-. ]+\.(png|jpg|jpeg|gif)$` 校验，杜绝路径穿越（`../`）与非法字符
- `_load_captions()` 本地重实现而不 import retrieval——避免列表接口连带拉起 torch，保证"列表要轻、要快"
- 状态巡检**全部只读探测，绝不触发引擎初始化**（状态页不应该把服务拖去加载模型）；Milvus 探测用 socket 连通性检查

### 5.10 backend/routers/history.py

**作用**：用标准库 SQLite 持久化每次检索，给前端历史面板提供数据。

**关键函数**：

| 函数 | 行号 | 职责 |
|---|------|------|
| `record_search(query, mode, result_count, total_time, mock)` | ~49 | 写入一条检索记录；**任何异常都不抛**——历史是锦上添花，绝不能弄挂检索主流程 |
| `list_history(limit)` | ~64 | `GET /api/history?limit=20`：倒序返回最近检索 |
| `clear_history()` | ~93 | `DELETE /api/history`：清空 |

**要点**：
- 存储位置 `backend/data/history.db`，每次操作开短连接（无长连接、无外部依赖）
- `PRAGMA journal_mode=WAL` 降低"检索写入 + 前端轮询读取"并发的互锁概率
- 前端 SearchPage 历史面板点击条目可**一键重发**

### 5.11 backend/routers/health.py

**作用**：极简健康检查路由，用于前后端联通性测试。

**关键函数**：

| 函数 | 行号 | 职责 |
|---|------|------|
| `health_check()` | ~16 | `GET /api/health` 返回 `{"status":"ok","service":"multimodal-rag-backend"}` |
| `ping()` | ~21 | `GET /api/ping` 返回 `{"message":"pong"}` |

**要点**：两个接口功能类似，`/health` 用于 K8s/监控探活，`/ping` 用于前端调试；前端侧栏的在线呼吸灯轮询的就是 `/api/health`。

### 5.12 frontend/src/types.ts + api.ts（契约层与客户端）

**作用**：`types.ts` 是前后端契约层（与 `backend/schemas.py` 及 routers 返回结构一一对齐）；`api.ts` 是后端 HTTP 客户端——**前端所有页面只通过它访问后端**，请求全部走相对路径由 Vite proxy 转发到 3001。

**api.ts 关键导出**：

| 函数 | 行号 | 职责 |
|---|------|------|
| `searchImages(req)` | ~32 | `POST /api/search` 同步检索 |
| `fetchImages / uploadImage / deleteImage / reindexImages` | ~44-73 | 图库四件套：列表 / multipart 上传 / 删除 / 热重建索引 |
| `fetchHistory / clearHistory` | ~77-86 | 检索历史读取与清空 |
| `fetchStatus / pingHealth` | ~90-104 | 状态页数据 / 侧栏呼吸灯探活 |
| `resetAgentSession(sessionId)` | ~107 | 重置 Agent 会话记忆 |
| `streamAgentChat(body, onEvent, signal)` | ~151 | **SSE 流式对话核心**：`fetch` + `ReadableStream` 逐帧解析，按空行切帧、`event:`/`data:` 行解析，单帧 JSON 失败只丢该帧不让整个流崩 |
| `fileToBase64 / urlToBase64` | ~119-139 | 本地文件 / `/static` 图片 URL → base64 data URL（以图搜图跳转时用）|

**要点**：
- Agent SSE 用 `fetch` + `response.body.getReader()` 而非 `EventSource`，因为 `EventSource` 只支持 GET，而 Agent 接口必须 POST 传大体积 base64 图片
- `TextDecoder(stream: true)` 处理多字节中文字符被 TCP 包从中间截断的情况
- `types.ts` 顶部注释是团队约定："后端改字段，先改这里"

### 5.13 frontend/src/App.tsx + components/Sidebar.tsx（应用骨架）

**作用**：`App.tsx`（71 行）是应用骨架——`BrowserRouter` + 左侧栏布局，路由表 `/` 检索、`/agent` 对话、`/gallery` 图库、`/status` 状态，其余路径重定向到 `/`。`Sidebar.tsx`（87 行）是导航 + 健康指示。

**要点**：
- `<768px` 时侧栏隐藏，`MobileNav`（App.tsx 内）退化为顶部横滑导航，与侧栏共用同一份 `NAV_ITEMS`
- `useBackendHealth()` 钩子轮询 `pingHealth()`，在线绿灯 / 离线红灯
- 样式全走 Tailwind 原子类，亮色极简主题（`index.css` 的 `@theme` 定义中文字体栈与等宽字体，锁定单主题不做深浅切换）

### 5.14 frontend/src/pages/SearchPage.tsx（检索页，546 行）

**作用**：最重的页面。三检索模式（文搜图/图搜图/混合）+ 图片上传 + 召回/精排/阈值参数 + 结果网格 + 历史面板。

**要点**：
- 示例查询 chips 贴合图片库内容（架构图/流程图为主），点击即填入
- `searchByResultImage()`：点结果图 → `urlToBase64` → 填入上传区并切"图搜图"模式，实现"以图搜图"连环搜
- 历史面板：`fetchHistory` 加载最近 20 条，点击条目一键重发，支持一键清空
- 加载态用骨架屏（`SkeletonGrid`），不再有假进度条模拟器

### 5.15 frontend/src/pages/AgentPage.tsx（Agent 对话页，397 行）

**作用**：SSE 事件流消费与渲染：思考链 `ThinkingLine`、工具调用 `ToolLine`、检索结果卡片（Lightbox 查看）、总结文本。

**要点**：
- 通过 `api.ts` 的 `streamAgentChat` 消费 6 类事件，逐事件增量更新消息树（`Turn` 结构）
- "新对话" = `resetAgentSession(旧id)` + 生成新 sessionId，利用 LangGraph `thread_id` 实现会话记忆隔离
- 支持上传图片提问（`ImageUploader`），结果图可放大查看

### 5.16 frontend/src/pages/GalleryPage.tsx + StatusPage.tsx

**作用**：图库页（250 行）与状态页（193 行）。

| 页面 | 关键能力 |
|---|---|
| `GalleryPage` | 图片网格（文件名/大小/时间/caption 标记）；`uploadImage` 上传、`deleteImage` 删除、`reindexImages` 热重建索引（返回 ok/skipped/busy 三态）；`searchWithImage()` 以图搜图跳检索页 |
| `StatusPage` | 消费 `SystemStatus`：后端版本/Milvus 连通性/引擎初始化三态（`true/false/null`）/四组模型配置（Embedding/Reranker/Agent LLM/Vision LLM）/图库统计，`StatusPill` 红绿灯呈现 |

### 5.17 frontend/src/components/*（通用组件）

| 组件 | 行数 | 一句话职责 |
|---|---|---|
| `ImageUploader.tsx` | 146 | 图片上传（点击/拖拽选图、预览、移除、compact 紧凑模式），检索页与 Agent 页共用 |
| `Lightbox.tsx` | 135 | 大图灯箱：元信息展示（`formatBytes` 人性化字节数）、"以此图检索"、删除入口；检索页/图库页/Agent 页共用 |
| `Sidebar.tsx` | 87 | 侧栏导航 + 健康呼吸灯（见 5.13）|

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
后端 AgentManager.chat_stream         前端 api.ts.streamAgentChat → AgentPage
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
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | `.env` | — / `https://api.openai.com/v1` | 旧版兼容配置，Agent/Vision 未单独设置时回退使用 |
| `AGENT_LLM_MODEL` | `.env` | `gpt-4o` | Agent 推理模型（需支持 function calling）|
| `AGENT_LLM_API_KEY` / `AGENT_LLM_BASE_URL` | `.env` | 回退 `OPENAI_*` | Agent 模型凭证与端点 |
| `VISION_LLM_MODEL` | `.env` | `gpt-4o` | Vision 视觉模型（describe_image 用，需支持图片输入）|
| `VISION_LLM_API_KEY` / `VISION_LLM_BASE_URL` | `.env` | 回退 `OPENAI_*` | Vision 模型凭证与端点 |
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
.env 文件 (load_dotenv override=False，只补缺)
    ↑
系统环境变量（最高优先，不被 .env 覆盖）
```

**要点**：`load_dotenv(override=False)` 表示已存在的系统环境变量**优先**，`.env` 只补没有的项。这正是 `USE_MOCK_DATA=true python -m backend.main` 命令行临时开关能生效的原因——若改成 `override=True`，`.env` 里写死的 `USE_MOCK_DATA=false` 会反过来覆盖命令行传入的 `true`。

### 7.3 .env 文件示例

完整模板见 `backend/.env.example`（含 DeepSeek / 通义 / GLM / Ollama 多厂商示例）。最小可用配置：

```bash
# 简单模式：Agent 与 Vision 共用一个模型
OPENAI_API_KEY=sk-xxxxx

# 进阶模式：Agent 与 Vision 分别指定（均可省略回退到 OPENAI_*）
AGENT_LLM_MODEL=qwen3.5-omni-flash
AGENT_LLM_API_KEY=sk-your-dashscope-key
AGENT_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
VISION_LLM_MODEL=qwen3.5-omni-plus-2026-03-15
VISION_LLM_API_KEY=sk-your-dashscope-key
VISION_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 可选（有默认值）
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

# 3️⃣ 配置 .env（从模板复制后编辑，至少设置一个可用 Key）
cd backend
cp .env.example .env    # Windows CMD 用:copy .env.example .env
# 简单模式：只填 OPENAI_API_KEY；进阶模式：AGENT_LLM_* 与 VISION_LLM_* 分别指定

# 4️⃣ 安装后端依赖（推荐 uv，也可用 venv + pip）
# 在项目根目录执行
uv venv .venv
source .venv/Scripts/activate   # Git Bash
# .venv\Scripts\activate        # CMD / PowerShell
uv pip install -r requirements.txt

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

# 7️⃣ 体验四页面
#    - 检索页 /：输入文本或上传图片，点"开始检索"；历史面板一键重发
#    - Agent 页 /agent：自然语言对话，实时观察思考链与工具调用
#    - 图库页 /gallery：上传/删除图片后点"重建索引"（热更新）
#    - 状态页 /status：查看 Milvus/引擎/模型配置是否正常
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
| 5 | 完整前端四页面 UI（检索/Agent/图库/状态）| 3-5 天 | React Router + Tailwind CSS 4 + SSE 解析 |

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
5. 读前端 `api.ts`（SSE 解析）与 `pages/AgentPage.tsx`（事件渲染），理解流式链路两端
6. 按需读其余页面与组件（SearchPage / GalleryPage / StatusPage），理解 UI 细节

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

**A**：不能。`EventSource` 只支持 GET 请求，而 Agent 接口需要 POST（带 image base64 payload 大）。所以 `api.ts` 的 `streamAgentChat` 用 `fetch` + `response.body.getReader()` 手动解析 SSE 格式。

### Q7: 如何切换 OpenAI 兼容代理（如 Azure、国内中转）？

**A**：Agent 与 Vision 可分别配置——`.env` 里设置 `AGENT_LLM_BASE_URL` / `VISION_LLM_BASE_URL`（未设置时回退 `OPENAI_BASE_URL`），代码用 `ChatOpenAI(base_url=...)` 自动适配，只要代理兼容 OpenAI Chat Completions API 即可。注意 Agent 模型需支持 function calling，Vision 模型需支持图片输入。

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
1. GalleryMind 是一个**完整的工业级多模态 RAG 样本项目**，覆盖检索与 Agent 两条路径，并带图库管理、检索历史、系统状态巡检等配套能力
2. 核心架构 = **Qwen3-VL 嵌入 + Milvus 双索引 + BM25 + RRF 融合 + Qwen3-VL Reranker + LangGraph Agent + SSE 流式**；前端为 React Router 四页面 SPA（Tailwind CSS 4）
3. 推荐阅读顺序：`config.py` → `schemas.py` → `main.py` → `routers/*` → `core/agent.py` → `core/retrieval.py`（最大最关键）→ 前端 `api.ts` → `pages/*`
4. 跑通项目最关键的三步：`docker-compose up -d`（Milvus）+ 配置 `.env`（从 `.env.example` 复制，至少一个可用 Key）+ `python -m backend.main`（首次会下载模型）
