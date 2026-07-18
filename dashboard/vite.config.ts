import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const ORCHESTRATOR = process.env.ORCHESTRATOR_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Same-origin in the browser, so the orchestrator needs no CORS config.
    proxy: Object.fromEntries(
      ["/health", "/capture", "/reconstruct", "/segment", "/generate-twin", "/plan",
       "/train", "/optimize", "/deploy", "/sync", "/status"]
        .map((route) => [route, { target: ORCHESTRATOR }])
        .concat([["/ws/status", { target: ORCHESTRATOR, ws: true }]]),
    ),
  },
});
