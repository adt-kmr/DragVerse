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

/**
 * Upload frame chunks and close the scan. The first response mints the scan id, so the
 * chunks go up in sequence rather than in parallel — the id has to exist before the
 * second chunk can name it. Re-POSTing an index overwrites it, which is what makes a
 * half-finished upload resumable.
 *
 * FormData sets its own multipart content-type; overriding it drops the boundary.
 */
export async function uploadScan(files, onProgress) {
  let scanId = null;
  for (let index = 0; index < files.length; index += 1) {
    const form = new FormData();
    form.append("file", files[index]);
    const query = new URLSearchParams({ index });
    if (scanId) query.set("scan_id", scanId);
    ({ scan_id: scanId } = await send(`/capture?${query}`, { method: "POST", body: form }));
    onProgress?.(index + 1, files.length);
  }
  return send(`/capture/${scanId}/complete`, { method: "POST" });
}
/** Extensions the orchestrator will ingest, mirrored from capture/scaniverse.py. */
export const SCAN_EXTS = [".ply", ".obj"];

/**
 * Open a scan from a finished export (Scaniverse .ply/.obj) in one POST.
 *
 * XHR rather than fetch: a room scan runs to hundreds of megabytes and fetch reports
 * no upload progress, which would leave the operator staring at a dead button.
 */
export function importScan(file, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/capture/import");
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress?.(event.loaded, event.total);
    };
    xhr.onload = () => {
      let body = {};
      try {
        body = xhr.responseText ? JSON.parse(xhr.responseText) : {};
      } catch {
        return reject(new Error(`Malformed reply from the orchestrator (${xhr.status})`));
      }
      if (xhr.status >= 200 && xhr.status < 300) return resolve(body);
      const detail = body.detail ?? xhr.statusText;
      const error = new Error(typeof detail === "string" ? detail : detail.error);
      error.status = xhr.status;
      error.detail = detail;
      reject(error);
    };
    xhr.onerror = () => reject(new Error("Upload failed — is the orchestrator running?"));
    xhr.onabort = () => reject(new Error("Upload cancelled"));
    xhr.send(form);
  });
}

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
export const benchmarks = () => send("/benchmarks");
