# Description

## 中文版

GalleryMind 是一个多模态图像检索与 Agentic Chat 系统：输入一句话或一张图即可在本地图片库按视觉语义检索，也可与 Agent 对话完成找图与看图，并配套图库管理、检索历史与系统状态巡检。其含金量在检索与编排：Milvus 双索引分离图/文搜图避免互相干扰、Qwen3-VL 向量与 BM25 三路召回经 RRF 融合提升召回率、共享 Reranker 2B 单例避免重复加载拖慢响应、增删图片可热重建索引无需重启，LangGraph Agent 以 SSE 六类事件流让推理全程透明；前端为 React Router 四页面 + Tailwind 4，内置 Mock 模式无 GPU 即可联调，适合作为多模态 RAG 学习样本或二次开发图片管理应用

## English

GalleryMind is a multimodal image retrieval and agentic chat system: give it a text query or an image, and it finds visually similar images in a local library, then lets you talk to an agent to close the find-and-describe loop — complete with gallery management, SQLite-backed search history, and a read-only system status page. The depth lies in retrieval and orchestration — a dual-index Milvus setup keeps image-to-image and text-to-image search in separate collections so the two modes never interfere; Qwen3-VL vectors fused with BM25 via reciprocal rank fusion combine semantic and lexical recall; a shared 2B reranker singleton avoids costly per-request model reloads; re-indexing after uploads or deletions happens hot, without restarting the service; and a LangGraph agent streams six SSE event types (thinking, tool_call, process, results, summary, complete) to make every reasoning step visible to the user. Built on FastAPI, a React Router four-page frontend with Tailwind CSS 4, LlamaIndex and Milvus (Docker Compose), with a mock mode that runs the full flow without GPU or model downloads. A complete multimodal RAG sample, extendable into image-management or asset-retrieval applications.
