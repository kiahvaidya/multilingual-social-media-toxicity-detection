import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const apiPort = process.env.API_PORT || '8001';
const apiUrl = process.env.API_URL || `http://127.0.0.1:${apiPort}`;
const frontendPort = Number(process.env.FRONTEND_PORT || 3001);

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: frontendPort,
    host: '127.0.0.1',
    proxy: {
      '/api': {
        target: apiUrl,
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: frontendPort,
    host: '127.0.0.1',
    proxy: {
      '/api': {
        target: apiUrl,
        changeOrigin: true,
      },
    },
  },
});
