// Sidebar.tsx(左侧导航):四个主视图的入口 + 后端在线指示灯。
// <768px 时整个侧栏隐藏,由 App.tsx 顶部渲染的移动端导航条接管(MobileNav 复用 NAV_ITEMS)

import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import { Aperture, Chats, Images, MagnifyingGlass, Pulse } from "@phosphor-icons/react";
import { pingHealth } from "../api";

export const NAV_ITEMS = [
  { to: "/", label: "检索", icon: MagnifyingGlass, end: true },
  { to: "/agent", label: "Agent 对话", icon: Chats, end: false },
  { to: "/gallery", label: "图片库", icon: Images, end: false },
  { to: "/status", label: "系统状态", icon: Pulse, end: false },
] as const;

// useBackendHealth(在线探活):每 15s ping 一次 /api/health,驱动侧栏底部指示灯。
// 指示灯是真实语义状态(不是装饰点),离线时用户能立刻明白为什么所有请求都失败
export function useBackendHealth(): boolean {
  const [online, setOnline] = useState(true);
  useEffect(() => {
    let alive = true;
    const check = async () => {
      const ok = await pingHealth();
      if (alive) setOnline(ok);
    };
    check();
    const timer = setInterval(check, 15_000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);
  return online;
}

export default function Sidebar() {
  const online = useBackendHealth();

  return (
    <aside className="hidden w-60 shrink-0 flex-col border-r border-zinc-200 bg-white md:flex">
      {/* 品牌区 */}
      <div className="flex items-center gap-3 px-5 pt-6 pb-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-zinc-900 text-white">
          <Aperture size={20} weight="fill" />
        </div>
        <div className="leading-tight">
          <p className="text-[15px] font-semibold tracking-tight">GalleryMind</p>
          <p className="text-xs text-zinc-500">多模态图像检索</p>
        </div>
      </div>

      {/* 主导航 */}
      <nav className="flex flex-col gap-1 px-3">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors ${
                isActive
                  ? "bg-zinc-900 font-medium text-white"
                  : "text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900"
              }`
            }
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* 底部:后端连接状态(真实语义状态指示) */}
      <div className="mt-auto border-t border-zinc-100 px-5 py-4">
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <span
            className={`h-2 w-2 rounded-full ${online ? "bg-emerald-500" : "bg-red-500"}`}
            aria-hidden
          />
          {online ? "后端已连接" : "后端离线"}
          <span className="font-mono text-[11px] text-zinc-400">:3001</span>
        </div>
      </div>
    </aside>
  );
}
