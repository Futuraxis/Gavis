import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // 开发模式: /api 代理到平台后端; 生产模式由 Python 服务直接托管 dist/
    proxy: {
      '/api': 'http://127.0.0.1:8770',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
