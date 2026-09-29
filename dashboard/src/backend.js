// The console talks to the orchestrator through api.js. The public demo build has no
// orchestrator behind it, so it uses the simulated replay in sim.js instead, and the
// console says so on screen. SIMULATED is fixed at build time in vite.config.ts.
import * as live from "./api.js";
import * as sim from "./sim.js";

// eslint-disable-next-line no-undef
export const SIMULATED = __SIMULATED__;

export const {
  health, scanStatus, uploadScan, SCAN_EXTS, importScan, reconstruct, segment,
  generateTwin, plan, train, optimize, deploy, sync, benchmarks,
} = SIMULATED ? sim : live;
