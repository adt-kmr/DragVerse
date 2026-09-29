import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const ORCHESTRATOR = process.env.ORCHESTRATOR_URL ?? "http://localhost:8000";

// Vercel hosts the public demo with no orchestrator behind it, so its builds use the
// simulated replay (src/sim.js). DASHBOARD_MODE=simulated does the same anywhere else.
const SIMULATED = process.env.VERCEL === "1" || process.env.DASHBOARD_MODE === "simulated";

export default defineConfig({
  plugins: [react()],
  define: { __SIMULATED__: JSON.stringify(SIMULATED) },
  server: {
    port: 5173,
    // Same-origin in the browser, so the orchestrator needs no CORS config.
    proxy: Object.fromEntries(
      ["/health", "/capture", "/reconstruct", "/segment", "/generate-twin", "/plan",
       "/train", "/optimize", "/deploy", "/sync", "/status", "/benchmarks"]
        .map((route) => [route, { target: ORCHESTRATOR }])
        .concat([["/ws/status", { target: ORCHESTRATOR, ws: true }]]),
    ),
  },
});
