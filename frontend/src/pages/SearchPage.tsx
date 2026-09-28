// SearchPage.tsx(检索模式):文搜图 / 图搜图 / 混合搜索三模式统一入口。
// 对接 POST /api/search(同步 JSON)。检索过程进度条为前端模拟(后端 processSteps 恒为空),
// 三个步骤对应真实管线阶段:向量召回 → 融合排序 → 精排重排,完成时刻以实际响应为准

import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  ArrowClockwise,
  ClockCounterClockwise,
  MagnifyingGlass,
  SlidersHorizontal,
  Trash,
  WarningCircle,
} from "@phosphor-icons/react";
import { clearHistory, fetchHistory, searchImages, urlToBase64 } from "../api";
import type { HistoryEntry, SearchMode, SearchResult } from "../types";
import { SEARCH_MODES } from "../types";
import ImageUploader from "../components/ImageUploader";
import Lightbox from "../components/Lightbox";

// 检索管线三阶段(与 retrieval.py 的真实流程对应,给用户过程透明感)
const PIPELINE_STEPS = ["向量召回", "融合排序", "精排重排"];

// 每种模式需要哪些输入,驱动按钮禁用态与输入区显隐
const modeRequires = (mode: SearchMode) => ({
  text: mode !== "图搜图",
  image: mode !== "文搜图",
});

// 示例查询:贴合图片库内容(架构/流程图为主)
const EXAMPLE_QUERIES = ["系统架构图", "数据流向图", "流程图"];

export default function SearchPage() {
  const [mode, setMode] = useState<SearchMode>("文搜图");
  const [textQuery, setTextQuery] = useState("");
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [recallTopK, setRecallTopK] = useState(20);
  const [rerankTopK, setRerankTopK] = useState(5);
  const [threshold, setThreshold] = useState(0);
  const [showParams, setShowParams] = useState(false);

  const [phase, setPhase] = useState<"idle" | "searching" | "done" | "error">("idle");
  const [activeStep, setActiveStep] = useState(0);
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [totalTime, setTotalTime] = useState(0);
  const [error, setError] = useState("");
  const [lightbox, setLightbox] = useState<SearchResult | null>(null);

  // 检索历史(后端 SQLite 持久化,见 routers/history.py)
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyOpen, setHistoryOpen] = useState(false);

  const navigate = useNavigate();
  const location = useLocation();
  const timersRef = useRef<ReturnType<typeof setInterval>[]>([]);

  // loadHistory:挂载时取一次,每次检索成功后刷新
  const loadHistory = async () => {
    try {
      const resp = await fetchHistory(12);
      setHistory(resp.items);
    } catch {
      // 历史是辅助信息,加载失败静默(面板显示为空即可)
      setHistory([]);
    }
  };

  useEffect(() => {
    void loadHistory();
  }, []);

  // 图库页"以此图检索"跳转过来:location.state 携带图片 base64,自动切到图搜图
  useEffect(() => {
    const state = location.state as { imageB64?: string } | null;
    if (state?.imageB64) {
      setImageB64(state.imageB64);
      setMode("图搜图");
      // 用完即清,避免刷新/前进后退时重复注入
      navigate(location.pathname, { replace: true, state: null });
    }
  }, [location.state, location.pathname, navigate]);

  // 搜索进行中:推进管线步骤动画(封顶在最后一步"处理中",等真实响应到达再全部完成)
  useEffect(() => {
    if (phase !== "searching") {
      timersRef.current.forEach(clearInterval);
      timersRef.current = [];
      return;
    }
    setActiveStep(0);
    const timer = setInterval(() => {
      setActiveStep((s) => Math.min(s + 1, PIPELINE_STEPS.length - 1));
    }, 900);
    timersRef.current.push(timer);
    return () => {
      timersRef.current.forEach(clearInterval);
      timersRef.current = [];
    };
  }, [phase]);

  const requires = modeRequires(mode);
  const canSearch =
    phase !== "searching" &&
    (!requires.text || textQuery.trim().length > 0) &&
    (!requires.image || imageB64 !== null);

  // handleSearch:发起检索。overrideMode/overrideQuery 供历史点击一键重发
  // (state 更新是异步的,重发路径必须用显式参数而不是刚 set 的 state)
  const handleSearch = async (overrideMode?: SearchMode, overrideQuery?: string) => {
    const m = overrideMode ?? mode;
    const q = (overrideQuery ?? textQuery).trim();
    const req = modeRequires(m);
    if (phase === "searching") return;
    if (req.text && !q) return;
    if (req.image && !imageB64) return;

    setPhase("searching");
    setError("");
    try {
      const resp = await searchImages({
        textQuery: q,
        uploadedImage: imageB64,
        recallTopK,
        rerankTopK,
        threshold,
        searchMode: m,
      });
      setResults(resp.results);
      setTotalTime(resp.totalTime);
      setPhase("done");
      void loadHistory();
    } catch (e) {
      setError(e instanceof Error ? e.message : "检索请求失败");
      setPhase("error");
    }
  };

  // rerunFromHistory:用历史条目的查询词+模式重发(图搜图历史不含图,退化为文搜图语义;
  // 后端记录的 mode 直接透传,若图片缺失后端会报错,由错误提示兜底)
  const rerunFromHistory = (entry: HistoryEntry) => {
    const m = (SEARCH_MODES as readonly string[]).includes(entry.mode)
      ? (entry.mode as SearchMode)
      : "文搜图";
    setMode(m);
    setTextQuery(entry.query);
    void handleSearch(m, entry.query);
  };

  const clearAllHistory = async () => {
    try {
      await clearHistory();
      setHistory([]);
    } catch {
      // 清历史失败不打断页面
    }
  };

  // 从检索结果再发起图搜图:拉图片转 base64,填入上传区
  const searchByResultImage = async (imageUrl: string) => {
    setLightbox(null);
    try {
      setImageB64(await urlToBase64(imageUrl));
      setMode("图搜图");
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch {
      setError("读取图片失败,无法以此图检索");
    }
  };

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 md:px-8">
      {/* 页头 */}
      <header className="mb-6">
        <h1 className="text-xl font-semibold tracking-tight">图像检索</h1>
        <p className="mt-1 text-sm text-zinc-500">
          一句话、一张图,或两者结合,从图片库中找回最相关的内容
        </p>
      </header>

      {/* 检索输入区 */}
      <section className="rounded-xl border border-zinc-200 bg-white p-5 md:p-6">
        {/* 模式切换 */}
        <div className="flex flex-wrap gap-1 rounded-lg bg-zinc-100 p-1" role="tablist">
          {SEARCH_MODES.map((m) => (
            <button
              key={m}
              role="tab"
              aria-selected={mode === m}
              onClick={() => setMode(m)}
              className={`flex-1 rounded-md px-4 py-2 text-sm transition-all ${
                mode === m
                  ? "bg-white font-medium text-zinc-900 shadow-sm"
                  : "text-zinc-500 hover:text-zinc-800"
              }`}
            >
              {m}
            </button>
          ))}
        </div>

        {/* 文本输入(文搜图/混合搜索显示) */}
        {requires.text && (
          <div className="mt-4">
            <label htmlFor="query" className="mb-1.5 block text-sm font-medium text-zinc-700">
              {mode === "混合搜索" ? "文本描述(与图片联合约束)" : "描述你要找的图"}
            </label>
            <input
              id="query"
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="例如:系统架构图"
              className="w-full rounded-lg border border-zinc-300 bg-white px-3.5 py-2.5 text-sm outline-none transition-colors placeholder:text-zinc-400 focus:border-zinc-900 focus:ring-2 focus:ring-zinc-900/10"
            />
          </div>
        )}

        {/* 图片输入(图搜图/混合搜索显示) */}
        {requires.image && (
          <div className={requires.text ? "mt-4" : "mt-4"}>
            <p className="mb-1.5 text-sm font-medium text-zinc-700">
              {mode === "混合搜索" ? "参考图片" : "上传一张相似的图"}
            </p>
            <ImageUploader value={imageB64} onChange={setImageB64} />
          </div>
        )}

        {/* 参数面板(可折叠) */}
        <div className="mt-4 border-t border-zinc-100 pt-3">
          <button
            onClick={() => setShowParams((v) => !v)}
            className="flex items-center gap-1.5 text-sm text-zinc-500 transition-colors hover:text-zinc-800"
            aria-expanded={showParams}
          >
            <SlidersHorizontal size={15} />
            检索参数
            <span className="font-mono text-xs text-zinc-400">
              召回 {recallTopK} · 精排 {rerankTopK}
            </span>
          </button>

          {showParams && (
            <div className="mt-4 grid gap-5 sm:grid-cols-3">
              <ParamSlider
                label="召回数量 recallTopK"
                hint="第一阶段快速捞取的候选数"
                min={5}
                max={50}
                step={1}
                value={recallTopK}
                onChange={setRecallTopK}
              />
              <ParamSlider
                label="精排数量 rerankTopK"
                hint="Reranker 精挑后返回的数量"
                min={1}
                max={10}
                step={1}
                value={rerankTopK}
                onChange={setRerankTopK}
              />
              <ParamSlider
                label="相似度阈值 threshold"
                hint="低于该分数的结果会被过滤"
                min={0}
                max={0.9}
                step={0.05}
                value={threshold}
                onChange={setThreshold}
              />
            </div>
          )}
        </div>

        {/* 动作行 */}
        <div className="mt-5 flex items-center justify-between gap-3">
          <p className="hidden text-xs text-zinc-400 sm:block">
            {phase === "done" && `本次耗时 ${totalTime.toFixed(2)}s`}
          </p>
          <button
            onClick={() => void handleSearch()}
            disabled={!canSearch}
            className="ml-auto flex items-center gap-2 rounded-lg bg-zinc-900 px-5 py-2.5 text-sm font-medium text-white transition-all hover:bg-zinc-700 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40"
          >
            <MagnifyingGlass size={16} weight="bold" />
            {phase === "searching" ? "检索中" : "开始检索"}
          </button>
        </div>
      </section>

      {/* 检索历史(可折叠,点击条目一键重发) */}
      {history.length > 0 && (
        <section className="mt-4 rounded-xl border border-zinc-200 bg-white">
          <div className="flex items-center gap-2 px-5 py-3">
            <button
              onClick={() => setHistoryOpen((v) => !v)}
              aria-expanded={historyOpen}
              className="flex items-center gap-1.5 text-sm text-zinc-500 transition-colors hover:text-zinc-800"
            >
              <ClockCounterClockwise size={15} />
              检索历史
              <span className="font-mono text-xs text-zinc-400">{history.length}</span>
            </button>
            {historyOpen && (
              <button
                onClick={() => void clearAllHistory()}
                className="ml-auto flex items-center gap-1 text-xs text-zinc-400 transition-colors hover:text-red-600"
              >
                <Trash size={13} />
                清空
              </button>
            )}
          </div>
          {historyOpen && (
            <ul className="divide-y divide-zinc-100 border-t border-zinc-100">
              {history.map((h) => (
                <li key={h.id}>
                  <button
                    onClick={() => rerunFromHistory(h)}
                    className="flex w-full items-center gap-3 px-5 py-2.5 text-left transition-colors hover:bg-zinc-50"
                    title="点击重新执行这次检索"
                  >
                    <span className="min-w-0 flex-1 truncate text-sm text-zinc-700">{h.query}</span>
                    <span className="shrink-0 rounded-full border border-zinc-200 px-2 py-0.5 text-xs text-zinc-500">
                      {h.mode}
                    </span>
                    {h.mock && (
                      <span className="shrink-0 rounded-full bg-zinc-100 px-2 py-0.5 text-xs text-zinc-500">
                        Mock
                      </span>
                    )}
                    <span className="shrink-0 font-mono text-xs text-zinc-400">
                      {h.resultCount} 项 · {h.totalTime.toFixed(2)}s
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {/* 管线进度(仅检索中/刚完成时显示) */}
      {phase !== "idle" && phase !== "error" && (
        <section className="mt-6" aria-label="检索进度">
          <ol className="flex items-center gap-2">
            {PIPELINE_STEPS.map((step, i) => {
              const completed = phase === "done" || i < activeStep;
              const active = phase === "searching" && i === activeStep;
              return (
                <li key={step} className="flex flex-1 items-center gap-2">
                  <span
                    className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-medium transition-colors ${
                      completed
                        ? "bg-emerald-600 text-white"
                        : active
                          ? "bg-zinc-900 text-white"
                          : "border border-zinc-300 text-zinc-400"
                    }`}
                  >
                    {completed ? "✓" : i + 1}
                  </span>
                  <span
                    className={`hidden text-sm sm:block ${
                      completed || active ? "text-zinc-800" : "text-zinc-400"
                    }`}
                  >
                    {step}
                  </span>
                  {i < PIPELINE_STEPS.length - 1 && (
                    <span className="h-px flex-1 bg-zinc-200" aria-hidden />
                  )}
                </li>
              );
            })}
          </ol>
        </section>
      )}

      {/* 错误提示 */}
      {(error || phase === "error") && (
        <div className="mt-6 flex items-start gap-2.5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          <WarningCircle size={18} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-medium">检索失败</p>
            <p className="mt-0.5 text-red-700">{error || "请检查后端是否启动"}</p>
            <button
              onClick={() => void handleSearch()}
              className="mt-2 flex items-center gap-1 text-sm font-medium text-red-800 underline underline-offset-2 hover:text-red-900"
            >
              <ArrowClockwise size={14} />
              重试
            </button>
          </div>
        </div>
      )}

      {/* 结果区 */}
      {phase === "searching" ? (
        <SkeletonGrid />
      ) : phase === "done" && results ? (
        results.length > 0 ? (
          <section className="mt-6" aria-label="检索结果">
            <div className="mb-3 flex items-baseline justify-between">
              <h2 className="text-sm font-medium text-zinc-700">
                检索到 <span className="font-mono">{results.length}</span> 张相关图片
              </h2>
              <p className="font-mono text-xs text-zinc-400">{totalTime.toFixed(2)}s</p>
            </div>
            <ResultGrid results={results} onSelect={setLightbox} />
          </section>
        ) : (
          <EmptyBlock
            title="没有找到相关图片"
            body="试试换一个描述,降低相似度阈值,或确认图片库中已有相关图片"
          />
        )
      ) : phase === "idle" ? (
        <InitialHints onPick={(q) => setTextQuery(q)} />
      ) : null}

      {lightbox && (
        <Lightbox
          image={lightbox}
          onClose={() => setLightbox(null)}
          onSearchByImage={searchByResultImage}
        />
      )}
    </div>
  );
}

// ---- 子组件 ----

function ParamSlider(props: {
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <label className="text-sm text-zinc-700">{props.label}</label>
        <span className="font-mono text-sm font-semibold text-zinc-900">{props.value}</span>
      </div>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
        className="w-full accent-emerald-600"
      />
      <p className="mt-1 text-xs text-zinc-400">{props.hint}</p>
    </div>
  );
}

function ResultGrid(props: { results: SearchResult[]; onSelect: (r: SearchResult) => void }) {
  return (
    <div className="grid grid-cols-1 gap-x-4 gap-y-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {props.results.map((r, i) => (
        <button
          key={r.id}
          onClick={() => props.onSelect(r)}
          className="rise-in group text-left"
          style={{ "--index": i } as React.CSSProperties}
        >
          <div className="aspect-[4/3] overflow-hidden rounded-lg border border-zinc-200 bg-white transition-shadow group-hover:shadow-md">
            <img
              src={r.imageUrl}
              alt={r.title}
              loading="lazy"
              className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03]"
            />
          </div>
          <div className="mt-2 flex items-center justify-between gap-2">
            <p className="truncate text-sm text-zinc-700">{r.title}</p>
            <span className="shrink-0 font-mono text-sm font-semibold text-emerald-700">
              {r.relevanceScore.toFixed(3)}
            </span>
          </div>
        </button>
      ))}
    </div>
  );
}

function SkeletonGrid() {
  return (
    <div
      className="mt-6 grid grid-cols-1 gap-x-4 gap-y-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
      aria-label="加载中"
    >
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i}>
          <div className="aspect-[4/3] animate-pulse rounded-lg bg-zinc-200/70" />
          <div className="mt-2 h-4 w-3/4 animate-pulse rounded bg-zinc-200/70" />
        </div>
      ))}
    </div>
  );
}

function InitialHints({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="mt-16 flex flex-col items-center text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-zinc-100">
        <MagnifyingGlass size={26} className="text-zinc-400" />
      </div>
      <p className="mt-4 text-sm font-medium text-zinc-700">先用一句话,或一张图</p>
      <p className="mt-1 max-w-sm text-sm text-zinc-500">
        描述你记忆中的图片内容,或者上传一张相似的图,系统会从图片库里找回最相关的结果
      </p>
      <div className="mt-5 flex flex-wrap justify-center gap-2">
        {EXAMPLE_QUERIES.map((q) => (
          <button
            key={q}
            onClick={() => onPick(q)}
            className="rounded-full border border-zinc-200 bg-white px-3.5 py-1.5 text-sm text-zinc-600 transition-colors hover:border-zinc-300 hover:bg-zinc-50 hover:text-zinc-900"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function EmptyBlock({ title, body }: { title: string; body: string }) {
  return (
    <div className="mt-16 flex flex-col items-center text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-zinc-100">
        <MagnifyingGlass size={26} className="text-zinc-400" />
      </div>
      <p className="mt-4 text-sm font-medium text-zinc-700">{title}</p>
      <p className="mt-1 max-w-sm text-sm text-zinc-500">{body}</p>
    </div>
  );
}
