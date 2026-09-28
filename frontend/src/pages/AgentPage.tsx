// AgentPage.tsx(Agent 模式):自然语言对话,对接 POST /api/agent/chat 的 SSE 六类事件流。
// 事件契约见 backend/core/agent.py:thinking → tool_call → process → results → summary → complete
// 多轮上下文由后端 LangGraph InMemorySaver(thread_id)维持,前端只需按顺序把消息发上去

import { useEffect, useRef, useState } from "react";
import {
  Aperture,
  ArrowBendUpLeft,
  CircleNotch,
  Eye,
  MagnifyingGlass,
  PaperPlaneRight,
  Stop,
  WarningCircle,
  Wrench,
} from "@phosphor-icons/react";
import { resetAgentSession, streamAgentChat } from "../api";
import type { SearchResult } from "../types";
import ImageUploader from "../components/ImageUploader";
import Lightbox from "../components/Lightbox";

// 一轮助手回复由若干"片段"按事件到达顺序组成,而不是一个纯文本气泡:
// 这样"正在分析→调工具→出图→给总结"的推理过程可以透明展示
type Part =
  | { kind: "thinking"; content: string; isError?: boolean }
  | { kind: "tool"; toolName: string; progress: number; status: "processing" | "completed" }
  | { kind: "results"; results: SearchResult[] }
  | { kind: "summary"; content: string };

interface Turn {
  id: number;
  role: "user" | "assistant";
  text?: string;
  image?: string | null;
  parts: Part[];
  streaming?: boolean;
}

// 工具名 → 展示文案(后端 tools 定义见 core/agent.py)
const TOOL_LABELS: Record<string, { label: string; icon: typeof Wrench }> = {
  search_images: { label: "检索图片库", icon: MagnifyingGlass },
  describe_image: { label: "识别图片内容", icon: Eye },
};

const SUGGESTIONS = ["帮我找几张系统架构图", "图片库里都有什么?", "描述一下这张图(先附一张图)"];

export default function AgentPage() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [image, setImage] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [lightbox, setLightbox] = useState<SearchResult | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const idRef = useRef(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  // 会话 ID:挂载时生成,"新对话"时更换——后端 LangGraph 按 thread_id 隔离多轮记忆
  const sessionRef = useRef<string>(crypto.randomUUID());

  const nextId = () => ++idRef.current;

  // startNewConversation(新对话):换 sessionId(旧上下文自然隔离) + 尽力清空旧会话记忆 + 清屏
  const startNewConversation = () => {
    if (streaming) abortRef.current?.abort();
    const oldId = sessionRef.current;
    sessionRef.current = crypto.randomUUID();
    setTurns([]);
    setInput("");
    setImage(null);
    // 清旧记忆只是释放,失败不影响新会话(已换 id,读不到旧上下文)
    void resetAgentSession(oldId).catch(() => undefined);
  };

  // 新事件到达 / 轮次结束时滚到底部
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  // patchTurn(不可变更新):往指定助手轮次追加/修改片段
  const patchTurn = (
    turnId: number,
    updater: (turn: Turn) => Turn,
  ) => {
    setTurns((prev) => prev.map((t) => (t.id === turnId ? updater(t) : t)));
  };

  const handleEvent = (turnId: number, type: string, payload: Record<string, unknown>) => {
    patchTurn(turnId, (turn) => {
      switch (type) {
        case "thinking": {
          const content = String(payload.content ?? "");
          return {
            ...turn,
            parts: [
              ...turn.parts,
              { kind: "thinking", content, isError: content.startsWith("发生错误") },
            ],
          };
        }
        case "tool_call": {
          const toolName = String(payload.toolName ?? "unknown_tool");
          return {
            ...turn,
            parts: [...turn.parts, { kind: "tool", toolName, progress: 50, status: "processing" }],
          };
        }
        case "process": {
          // process 事件通过 stepId 关联到最近的工具片段(后端单步只发一个工具,取末尾即可)
          const parts = [...turn.parts];
          for (let i = parts.length - 1; i >= 0; i--) {
            const p = parts[i];
            if (p.kind === "tool") {
              parts[i] = {
                ...p,
                progress: Number(payload.progress ?? p.progress),
                status: payload.status === "completed" ? "completed" : "processing",
              };
              break;
            }
          }
          return { ...turn, parts };
        }
        case "results": {
          const results = Array.isArray(payload.results) ? (payload.results as SearchResult[]) : [];
          if (!results.length) return turn;
          return { ...turn, parts: [...turn.parts, { kind: "results", results }] };
        }
        case "summary": {
          const content = String(payload.content ?? "");
          return { ...turn, parts: [...turn.parts, { kind: "summary", content }] };
        }
        case "complete": {
          return { ...turn, streaming: false };
        }
        default:
          return turn;
      }
    });
  };

  const send = async () => {
    const message = input.trim();
    if ((!message && !image) || streaming) return;

    const userTurn: Turn = { id: nextId(), role: "user", text: message, image, parts: [] };
    const assistantId = nextId();
    const assistantTurn: Turn = { id: assistantId, role: "assistant", parts: [], streaming: true };
    setTurns((prev) => [...prev, userTurn, assistantTurn]);
    setInput("");
    setImage(null);
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamAgentChat(
        { message: message || "请看看这张图", image, sessionId: sessionRef.current },
        (type, payload) => handleEvent(assistantId, type, payload as Record<string, unknown>),
        controller.signal,
      );
    } catch (e) {
      const aborted = e instanceof DOMException && e.name === "AbortError";
      patchTurn(assistantId, (turn) => ({
        ...turn,
        streaming: false,
        parts: aborted
          ? turn.parts
          : [
              ...turn.parts,
              {
                kind: "thinking",
                content: `请求失败: ${e instanceof Error ? e.message : "未知错误"}`,
                isError: true,
              },
            ],
      }));
    } finally {
      // 后端异常路径也会补发 complete;这里兜底确保 UI 不卡在流式态
      patchTurn(assistantId, (turn) => ({ ...turn, streaming: false }));
      setStreaming(false);
      abortRef.current = null;
    }
  };

  const stop = () => abortRef.current?.abort();

  return (
    <div className="mx-auto flex h-dvh max-w-3xl flex-col px-4 md:px-8">
      {/* 页头 */}
      <header className="flex items-end justify-between gap-4 py-5">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Agent 对话</h1>
          <p className="mt-1 text-sm text-zinc-500">
            用大白话说需求,Agent 自己调工具找图、看图、给结论
          </p>
        </div>
        {turns.length > 0 && (
          <button
            onClick={startNewConversation}
            className="flex shrink-0 items-center gap-1.5 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-700 transition-colors hover:bg-zinc-50"
          >
            <ArrowBendUpLeft size={15} />
            新对话
          </button>
        )}
      </header>

      {/* 消息区 */}
      <div className="flex-1 space-y-6 overflow-y-auto pb-4">
        {turns.length === 0 && (
          <div className="flex flex-col items-center pt-16 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-zinc-100">
              <Aperture size={26} className="text-zinc-400" />
            </div>
            <p className="mt-4 text-sm font-medium text-zinc-700">试着这样问它</p>
            <div className="mt-4 flex flex-col gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => setInput(s)}
                  className="rounded-full border border-zinc-200 bg-white px-4 py-2 text-sm text-zinc-600 transition-colors hover:border-zinc-300 hover:text-zinc-900"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((turn) =>
          turn.role === "user" ? (
            <div key={turn.id} className="flex justify-end">
              <div className="flex max-w-[80%] flex-col items-end gap-2">
                {turn.image && (
                  <img
                    src={turn.image}
                    alt="附带图片"
                    className="h-20 w-20 rounded-lg border border-zinc-200 object-cover"
                  />
                )}
                {turn.text && (
                  <div className="rounded-2xl rounded-br-md bg-zinc-900 px-4 py-2.5 text-sm leading-relaxed text-white">
                    {turn.text}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div key={turn.id} className="flex gap-3">
              <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-100">
                <Aperture size={17} className="text-zinc-500" />
              </div>
              <div className="min-w-0 flex-1 space-y-3">
                {turn.parts.length === 0 && turn.streaming && <ThinkingLine />}
                {turn.parts.map((part, i) => {
                  if (part.kind === "thinking") return <ThinkingLine key={i} part={part} />;
                  if (part.kind === "tool") return <ToolLine key={i} part={part} />;
                  if (part.kind === "results")
                    return (
                      <div
                        key={i}
                        className="rise-in grid grid-cols-3 gap-2 sm:grid-cols-4"
                        style={{ "--index": 0 } as React.CSSProperties}
                      >
                        {part.results.map((r) => (
                          <button
                            key={r.id}
                            onClick={() => setLightbox(r)}
                            className="aspect-square overflow-hidden rounded-lg border border-zinc-200 bg-white transition-shadow hover:shadow-md"
                          >
                            <img
                              src={r.imageUrl}
                              alt={r.title}
                              loading="lazy"
                              className="h-full w-full object-cover"
                            />
                          </button>
                        ))}
                      </div>
                    );
                  return (
                    <p
                      key={i}
                      className="rise-in text-sm leading-relaxed whitespace-pre-wrap text-zinc-800"
                    >
                      {part.content}
                    </p>
                  );
                })}
              </div>
            </div>
          ),
        )}
        <div ref={bottomRef} />
      </div>

      {/* 输入区 */}
      <div className="border-t border-zinc-200 bg-zinc-50 py-4">
        {image && (
          <div className="mb-2 flex items-center gap-2">
            <img
              src={image}
              alt="待发送图片"
              className="h-12 w-12 rounded-lg border border-zinc-200 object-cover"
            />
            <button
              onClick={() => setImage(null)}
              className="text-xs text-zinc-500 underline underline-offset-2 hover:text-zinc-800"
            >
              移除图片
            </button>
          </div>
        )}
        <div className="flex items-end gap-2 rounded-xl border border-zinc-200 bg-white p-2 shadow-sm focus-within:border-zinc-400">
          <ImageUploader value={image} onChange={setImage} compact />
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
            rows={1}
            placeholder="描述需求,Enter 发送,Shift+Enter 换行"
            className="max-h-32 flex-1 resize-none bg-transparent px-1 py-1.5 text-sm outline-none placeholder:text-zinc-400"
          />
          {streaming ? (
            <button
              onClick={stop}
              className="flex items-center gap-1.5 rounded-lg border border-zinc-300 px-3 py-2 text-sm text-zinc-700 transition-colors hover:bg-zinc-50"
            >
              <Stop size={14} weight="fill" />
              停止
            </button>
          ) : (
            <button
              onClick={() => void send()}
              disabled={!input.trim() && !image}
              className="flex items-center gap-1.5 rounded-lg bg-zinc-900 px-3.5 py-2 text-sm font-medium text-white transition-all hover:bg-zinc-700 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40"
            >
              <PaperPlaneRight size={15} weight="fill" />
              发送
            </button>
          )}
        </div>
      </div>

      {lightbox && <Lightbox image={lightbox} onClose={() => setLightbox(null)} />}
    </div>
  );
}

// ---- 片段渲染组件 ----

function ThinkingLine({ part }: { part?: { content: string; isError?: boolean } }) {
  if (!part) {
    // 首个 thinking 事件到达前的占位
    return (
      <p className="breathe text-sm text-zinc-400">正在思考</p>
    );
  }
  if (part.isError) {
    return (
      <p className="flex items-start gap-1.5 text-sm text-red-700">
        <WarningCircle size={16} className="mt-0.5 shrink-0" />
        {part.content}
      </p>
    );
  }
  return <p className="breathe text-sm text-zinc-400">{part.content}</p>;
}

function ToolLine({ part }: { part: Extract<Part, { kind: "tool" }> }) {
  const meta = TOOL_LABELS[part.toolName] ?? { label: part.toolName, icon: Wrench };
  const Icon = meta.icon;
  const done = part.status === "completed";
  return (
    <div
      className={`rise-in inline-flex w-fit items-center gap-2 rounded-lg border px-3 py-1.5 text-sm transition-colors ${
        done ? "border-zinc-200 bg-zinc-50 text-zinc-600" : "border-emerald-200 bg-emerald-50 text-emerald-800"
      }`}
      style={{ "--index": 0 } as React.CSSProperties}
    >
      <Icon size={15} />
      {meta.label}
      {done ? (
        <span className="text-xs text-zinc-400">完成</span>
      ) : (
        <CircleNotch size={14} className="animate-spin text-emerald-600" />
      )}
    </div>
  );
}
