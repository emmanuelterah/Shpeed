/// <reference types="vitest" />
import { fileURLToPath, URL } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// SHPEED_BASE is set by the Pages workflow to "/<repo-name>/".
// SHPEED_API_PROXY points the dev server's /api proxy at the backend (Docker dev: http://api:8000).
// SHPEED_POLL=1 turns on polling, needed for hot reload on Windows bind mounts.
export default defineConfig({
  base: process.env.SHPEED_BASE ?? "/",
  plugins: [react()],
  server: {
    port: 5173,
    watch: process.env.SHPEED_POLL === "1" ? { usePolling: true, interval: 300 } : undefined,
    proxy: {
      "/api": {
        target: process.env.SHPEED_API_PROXY ?? "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
