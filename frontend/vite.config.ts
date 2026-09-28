import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// 后端 FastAPI 固定跑在 3001(config.py PORT=3001),前端 dev server 用 3000;
// /api 与 /static 统一 proxy 到后端,前端代码里只写相对路径,不出现 localhost 硬编码。
// 注意写 127.0.0.1 而不是 localhost:Windows 下 uvicorn 只监听 IPv4,
// 而 Node 18 的 DNS 解析会把 localhost 优先解析为 IPv6(::1),导致代理连接失败
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://127.0.0.1:3001',
      '/static': 'http://127.0.0.1:3001',
    },
  },
})
