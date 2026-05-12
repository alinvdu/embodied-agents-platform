import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const explorationTarget = env.VITE_EXPLORATION_API_TARGET || "http://127.0.0.1:8770";
  const agentTarget = env.VITE_AGENT_API_TARGET || "http://127.0.0.1:8765";

  return {
    plugins: [react()],
    server: {
      proxy: {
        "/exploration-api": {
          target: explorationTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/exploration-api/, ""),
        },
        "/agent-api": {
          target: agentTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/agent-api/, ""),
        },
      },
    },
  };
});
