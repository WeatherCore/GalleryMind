// StatusPage.tsx(系统状态):GET /api/system/status 的可视化,每 10s 自动刷新。
// 只读展示:后端版本、Milvus 连通、引擎就绪、模型配置、图片库规模、Mock 开关

import { useCallback, useEffect, useState } from "react";
import {
  ArrowsClockwise,
  Brain,
  Cpu,
  Database,
  HardDrives,
  Monitor,
  WarningCircle,
} from "@phosphor-icons/react";
import { fetchStatus } from "../api";
import type { SystemStatus } from "../types";

// 状态徽标的三种成色:正常(emerald)/异常(red)/未知或未启用(zinc)
type Level = "ok" | "bad" | "neutral";

const LEVEL_STYLES: Record<Level, { dot: string; pill: string }> = {
  ok: { dot: "bg-emerald-500", pill: "bg-emerald-50 text-emerald-800 border-emerald-200" },
  bad: { dot: "bg-red-500", pill: "bg-red-50 text-red-800 border-red-200" },
  neutral: { dot: "bg-zinc-400", pill: "bg-zinc-100 text-zinc-600 border-zinc-200" },
};

function StatusPill({ level, label }: { level: Level; label: string }) {
  const s = LEVEL_STYLES[level];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${s.pill}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${s.dot}`} aria-hidden />
      {label}
    </span>
  );
}

export default function StatusPage() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState("");
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);

  const load = useCallback(async () => {
    setError("");
    try {
      const data = await fetchStatus();
      setStatus(data);
      setUpdatedAt(new Date());
    } catch (e) {
      setError(e instanceof Error ? e.message : "状态接口请求失败");
    }
  }, []);

  useEffect(() => {
    void load();
    const timer = setInterval(() => void load(), 10_000);
    return () => clearInterval(timer);
  }, [load]);

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 md:px-8">
      <header className="mb-6 flex items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">系统状态</h1>
          <p className="mt-1 text-sm text-zinc-500">
            {updatedAt
              ? `每 10 秒自动刷新,更新于 ${updatedAt.toLocaleTimeString("zh-CN")}`
              : "正在读取系统状态"}
          </p>
        </div>
        <button
          onClick={() => void load()}
          className="flex shrink-0 items-center gap-1.5 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-700 transition-colors hover:bg-zinc-50"
        >
          <ArrowsClockwise size={15} />
          立即刷新
        </button>
      </header>

      {error && (
        <div className="mb-6 flex items-start gap-2.5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
          <WarningCircle size={18} className="mt-0.5 shrink-0" />
          <div>
            <p className="font-medium">无法读取系统状态</p>
            <p className="mt-0.5 text-red-700">{error},请确认后端(:3001)已启动</p>
          </div>
        </div>
      )}

      {status && (
        <div className="space-y-4">
          {/* 服务运行状态 */}
          <section className="rounded-xl border border-zinc-200 bg-white">
            <div className="divide-y divide-zinc-100">
              <StatusRow
                icon={<Monitor size={18} />}
                name="后端服务"
                desc={`FastAPI v${status.backend.version}${status.mockMode ? "(Mock 模式,检索返回示例数据)" : ""}`}
                right={
                  <div className="flex items-center gap-2">
                    {status.mockMode && <StatusPill level="neutral" label="Mock" />}
                    <StatusPill level="ok" label="运行中" />
                  </div>
                }
              />
              <StatusRow
                icon={<Database size={18} />}
                name="Milvus 向量库"
                desc={<span className="font-mono text-xs">{status.milvus.uri}</span>}
                right={
                  <StatusPill
                    level={status.milvus.connected ? "ok" : "bad"}
                    label={status.milvus.connected ? "已连接" : "未连接"}
                  />
                }
              />
              <StatusRow
                icon={<Cpu size={18} />}
                name="检索引擎"
                desc="Qwen3-VL Embedding / Reranker / 双索引"
                right={
                  status.engine.initialized === null ? (
                    <StatusPill level="neutral" label="未知" />
                  ) : status.engine.initialized ? (
                    <StatusPill level="ok" label="已就绪" />
                  ) : (
                    <StatusPill level="neutral" label="未加载" />
                  )
                }
              />
              <StatusRow
                icon={<HardDrives size={18} />}
                name="图片库"
                desc={<span className="font-mono text-xs">{status.gallery.dir}</span>}
                right={
                  <span className="font-mono text-sm text-zinc-700">
                    {status.gallery.count ?? "?"} 张
                  </span>
                }
              />
            </div>
          </section>

          {/* 模型配置(只读展示,不含任何密钥) */}
          <section className="rounded-xl border border-zinc-200 bg-white">
            <div className="border-b border-zinc-100 px-5 py-3">
              <h2 className="flex items-center gap-2 text-sm font-medium text-zinc-800">
                <Brain size={16} />
                模型配置
              </h2>
            </div>
            <div className="divide-y divide-zinc-100">
              <ModelRow label="嵌入模型" model={status.models.embedding} />
              <ModelRow label="重排模型" model={status.models.reranker} />
              <ModelRow label="Agent 推理" model={status.models.agentLlm} />
              <ModelRow label="视觉理解" model={status.models.visionLlm} />
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

function StatusRow(props: {
  icon: React.ReactNode;
  name: string;
  desc: React.ReactNode;
  right: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-3.5 px-5 py-4">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-zinc-100 text-zinc-500">
        {props.icon}
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-zinc-900">{props.name}</p>
        <p className="mt-0.5 truncate text-xs text-zinc-500">{props.desc}</p>
      </div>
      {props.right}
    </div>
  );
}

function ModelRow({ label, model }: { label: string; model: string }) {
  return (
    <div className="flex items-center justify-between gap-4 px-5 py-3">
      <span className="text-sm text-zinc-600">{label}</span>
      <span className="truncate font-mono text-xs text-zinc-500">{model}</span>
    </div>
  );
}
