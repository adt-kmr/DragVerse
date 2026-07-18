// Thin wrappers over the orchestrator REST surface (blueprint section 16).
// Same origin — Vite proxies these to the orchestrator in dev, see vite.config.ts.

async function send(path, options) {
  const response = await fetch(path, options);
  const text = await response.text();
  const body = text ? JSON.parse(text) : {};
  if (!response.ok) {
    // FastAPI puts the payload under `detail`; /train's sim gate returns an object
    // there rather than a string, and the operator needs to see the numbers.
    const detail = body.detail ?? response.statusText;
    const error = new Error(typeof detail === "string" ? detail : detail.error);
    error.status = response.status;
    error.detail = detail;
    throw error;
  }
  return body;
}

const post = (path, body) =>
  send(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });

export const health = () => send("/health");
export const scanStatus = (scanId) => send(`/capture/${scanId}`);
export const reconstruct = (scan_id, mode) => post("/reconstruct", { scan_id, mode });
export const segment = (mesh_id) => post("/segment", { mesh_id });
export const generateTwin = (mesh_id, objects_id) =>
  post("/generate-twin", { mesh_id, objects_id });
export const plan = (twin_id, text, lang) => post("/plan", { twin_id, text, lang });
export const train = (twin_id, task_graph_id) => post("/train", { twin_id, task_graph_id });
export const optimize = (policy_id, device_label) =>
  post("/optimize", { policy_id, device_label });
export const deploy = (artifact_id, kind) => post("/deploy", { artifact_id, kind });
export const sync = (twin_id, new_scan_id) => post("/sync", { twin_id, new_scan_id });
