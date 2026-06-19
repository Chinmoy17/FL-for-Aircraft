import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite dev server runs on http://localhost:5173 by default.
// All /api/* calls are proxied to the FastAPI backend on http://localhost:8000
// so we don't need to think about CORS during local development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
