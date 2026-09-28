// App.tsx(应用骨架):左侧栏 + 主区布局,<768px 时侧栏退化为顶部导航条。
// 路由表: / 检索 · /agent Agent 对话 · /gallery 图片库 · /status 系统状态

import { BrowserRouter, Navigate, NavLink, Route, Routes } from "react-router-dom";
import { Aperture } from "@phosphor-icons/react";
import Sidebar, { NAV_ITEMS, useBackendHealth } from "./components/Sidebar";
import SearchPage from "./pages/SearchPage";
import AgentPage from "./pages/AgentPage";
import GalleryPage from "./pages/GalleryPage";
import StatusPage from "./pages/StatusPage";

// MobileNav(移动端顶部导航):图标 + 文字横排,与侧栏共用同一份 NAV_ITEMS
function MobileNav() {
  const online = useBackendHealth();
  return (
    <div className="sticky top-0 z-40 border-b border-zinc-200 bg-white/90 backdrop-blur md:hidden">
      <div className="flex items-center gap-3 px-4 pt-3">
        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-zinc-900 text-white">
          <Aperture size={15} weight="fill" />
        </div>
        <p className="text-sm font-semibold tracking-tight">GalleryMind</p>
        <span
          className={`ml-auto h-2 w-2 rounded-full ${online ? "bg-emerald-500" : "bg-red-500"}`}
          aria-label={online ? "后端在线" : "后端离线"}
        />
      </div>
      <nav className="flex gap-1 overflow-x-auto px-3 py-2">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `flex shrink-0 items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm transition-colors ${
                isActive
                  ? "bg-zinc-900 font-medium text-white"
                  : "text-zinc-600 hover:bg-zinc-100"
              }`
            }
          >
            <Icon size={15} />
            {label}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-dvh">
        <Sidebar />
        <div className="flex min-w-0 flex-1 flex-col">
          <MobileNav />
          <main className="flex-1">
            <Routes>
              <Route path="/" element={<SearchPage />} />
              <Route path="/agent" element={<AgentPage />} />
              <Route path="/gallery" element={<GalleryPage />} />
              <Route path="/status" element={<StatusPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}
