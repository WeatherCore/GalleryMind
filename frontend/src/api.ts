// api.ts(后端 HTTP 客户端):所有请求走相对路径,由 vite proxy 转发到 3001(见 vite.config.ts)。
// 特殊点:Agent 聊天用 fetch + ReadableStream 手工解析 SSE,因为浏览器 EventSource 只支持 GET,
// 而 Agent 接口必须用 POST 传大体积 base64 图片(见 routers/agent.py 注释)

import type {
  AgentEventPayloads,
  AgentEventType,
  HistoryEntry,
  ImagesResponse,
  SearchRequest,
  SearchResponse,
  SystemStatus,
} from "./types";

const JSON_HEADERS = { "Content-Type": "application/json" };

// readError(错误信息提取):FastAPI 异常返回 {detail: "..."},尽量把人话翻出来
async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body?.detail) {
      return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    }
  } catch {
    /* 响应体不是 JSON,走下面兜底 */
  }
  return `后端返回 ${res.status} ${res.statusText}`;
}

// ---- 检索模式 ----

export async function searchImages(req: SearchRequest): Promise<SearchResponse> {
  const res = await fetch("/api/search", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

// ---- 图库 ----

export async function fetchImages(): Promise<ImagesResponse> {
  const res = await fetch("/api/images");
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

// uploadImage(上传入库):multipart 单文件上传,入库后需重建索引才能被真实检索
export async function uploadImage(file: File): Promise<{ filename: string; hint: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/images/upload", { method: "POST", body: form });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function deleteImage(filename: string): Promise<void> {
  const res = await fetch(`/api/images/${encodeURIComponent(filename)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await readError(res));
}

// reindex(热重建索引):真实模式下同步等待重建完成,可能耗时较长
export async function reindexImages(): Promise<{
  status: "ok" | "skipped" | "busy";
  message?: string;
  indexed?: number;
}> {
  const res = await fetch("/api/images/reindex", { method: "POST" });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

// ---- 检索历史 ----

export async function fetchHistory(limit = 20): Promise<{ total: number; items: HistoryEntry[] }> {
  const res = await fetch(`/api/history?limit=${limit}`);
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function clearHistory(): Promise<void> {
  const res = await fetch("/api/history", { method: "DELETE" });
  if (!res.ok) throw new Error(await readError(res));
}

// ---- 系统状态 ----

export async function fetchStatus(): Promise<SystemStatus> {
  const res = await fetch("/api/system/status");
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

// 侧边栏在线指示灯用:能通即在线,不做多余解析
export async function pingHealth(): Promise<boolean> {
  try {
    const res = await fetch("/api/health");
    return res.ok;
  } catch {
    return false;
  }
}

// resetAgentSession(重置会话):尽力清空旧会话记忆;失败不影响"换新 id"带来的隔离
export async function resetAgentSession(sessionId: string): Promise<void> {
  const res = await fetch("/api/agent/session/reset", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ sessionId }),
  });
  if (!res.ok) throw new Error(await readError(res));
}

// ---- 图片编码工具 ----

// fileToBase64:本地文件 → data URL(base64),检索/Agent 的图片字段都吃这个格式
export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("读取图片文件失败"));
    reader.readAsDataURL(file);
  });
}

// urlToBase64:后端 /static 的图片 URL → base64,图库"以图搜图"跳检索页时用
export async function urlToBase64(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error("读取图片失败");
  const blob = await res.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("读取图片失败"));
    reader.readAsDataURL(blob);
  });
}

// ---- Agent SSE 流 ----

export type AgentEventHandler = (
  type: AgentEventType,
  payload: AgentEventPayloads[AgentEventType],
) => void;

// streamAgentChat(流式对话):POST /api/agent/chat,逐帧解析 SSE。
// sessionId 决定多轮记忆挂在哪个会话(LangGraph thread_id),前端"新对话"时换新 id
// SSE 帧格式(后端 _format_sse): "event: thinking\ndata: {...}\n\n",帧间以空行分隔
export async function streamAgentChat(
  body: { message: string; image?: string | null; sessionId?: string },
  onEvent: AgentEventHandler,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/agent/chat", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw new Error(await readError(res));
  if (!res.body) throw new Error("后端未返回流式响应");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatchFrame = (frame: string) => {
    let eventType = "";
    const dataLines: string[] = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) eventType = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (!eventType) return; // 没有 event 行的帧不是契约内事件,忽略
    try {
      const payload = dataLines.length ? JSON.parse(dataLines.join("\n")) : {};
      onEvent(eventType as AgentEventType, payload);
    } catch {
      // 单帧 JSON 解析失败只丢这一帧,不能让整个流崩掉
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    // stream:true 处理多字节中文字符被 TCP 包从中间截断的情况
    buffer += decoder.decode(value, { stream: true });
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      dispatchFrame(buffer.slice(0, sep));
      buffer = buffer.slice(sep + 2);
    }
  }
  // 流正常结束后可能还剩最后一帧(没有以空行收尾)
  if (buffer.trim()) dispatchFrame(buffer.trim());
}
