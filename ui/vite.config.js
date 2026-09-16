import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Defaults to the local API. Set VITE_API_TARGET to point at a remote
      // one (e.g. https://exo.dosi.io) when testing the UI without running a
      // local backend — useful on-device, and avoids starting a second
      // ExoScheduler against the same DATA_DIR.
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
