// types.ts(前后端契约层):与 backend/schemas.py 及 routers/system.py 的返回结构一一对齐。
// 后端改字段,先改这里;前端所有页面只依赖本文件的类型,不允许散落定义

// 检索模式:必须是这三个中文字符串,后端 retrieval_engine.search() 直接按字符串分发
export const SEARCH_MODES = ["文搜图", "图搜图", "混合搜索"] as const;
export type SearchMode = (typeof SEARCH_MODES)[number];

// POST /api/search 请求体
export interface SearchRequest {
  textQuery: string;
  uploadedImage?: string | null; // base64 data URL,后端 save_base64_image 会自己剥前缀
  recallTopK: number;
  rerankTopK: number;
  threshold: number;
  searchMode: SearchMode;
}

// 单条检索结果(score 0-100 给人看,relevanceScore 0-1 是真实相似度)
export interface SearchResult {
  id: string;
  imageUrl: string; // 形如 /static/images/xxx.png,dev 下由 vite proxy 转发到 3001
  title: string;
  description?: string | null;
  score: number;
  relevanceScore: number;
  rerankScore?: number | null;
  metadata?: Record<string, unknown> | null;
}

// POST /api/search 响应体
export interface SearchResponse {
  results: SearchResult[];
  processSteps?: unknown[]; // 后端恒为 [],真实进度由前端 pipeline 模拟
  totalTime: number;
}

// GET /api/images(routers/system.py)图片库列表项
export interface GalleryImage {
  filename: string;
  url: string;
  sizeBytes: number;
  modifiedAt: number;
  hasCaption: boolean;
  caption?: string | null;
}

// GET /api/history(routers/history.py)检索历史条目
export interface HistoryEntry {
  id: number;
  createdAt: number;
  query: string;
  mode: SearchMode | string;
  resultCount: number;
  totalTime: number;
  mock: boolean;
}

export interface ImagesResponse {
  total: number;
  images: GalleryImage[];
}

// GET /api/system/status(routers/system.py)
// engine.initialized 三态:true 已就绪 / false 未加载 / null 引擎模块不可用
export interface SystemStatus {
  backend: { status: string; version: string };
  milvus: { uri: string; connected: boolean };
  engine: { initialized: boolean | null };
  models: {
    embedding: string;
    reranker: string;
    agentLlm: string;
    visionLlm: string;
  };
  gallery: { dir: string; count: number | null };
  mockMode: boolean;
  timestamp: number;
}

// Agent SSE 六类事件(格式见 core/agent.py 的 _format_sse 与 chat_stream 注释)
export type AgentEventType =
  | "thinking"
  | "tool_call"
  | "process"
  | "results"
  | "summary"
  | "complete";

export interface AgentEventPayloads {
  thinking: { content: string; timestamp?: string };
  tool_call: { toolName: string; timestamp?: string };
  process: { stepId: string; progress: number; status: "processing" | "completed" };
  results: { results: SearchResult[] };
  summary: { content: string; done?: boolean };
  complete: Record<string, never>;
}
