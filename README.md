# GalleryMind
GalleryMind 是基于 FastAPI 与 React 的多模态图像检索与 Agent Chat 系统,支持文搜图、图搜图、混合搜三种模式及 Agent 对话。含金量在检索与编排:Milvus 双索引分离图/文搜图、Qwen3-VL 向量与 BM25 经 RRF 三路融合、共享 Reranker 2B 单例复用、L2 归一化加 512 维截断、全局 embed_model 防 OpenAI fallback。Agent 用 LangGraph create_agent 配 InMemorySaver 与 SSE 六事件流让推理透明。适合作为多模态 RAG 学习样本
