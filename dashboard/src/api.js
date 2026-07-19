// Mocked api.js for TwinForge dashboard.
// These mock the backend orchestrator REST surface for a full frontend simulation.

const delay = (ms) => new Promise(r => setTimeout(r, ms));

export const health = async () => {
  await delay(100);
  return { status: "ok" };
};

export const scanStatus = async (scanId) => {
  await delay(500);
  return { frame_count: 120, status: "complete" };
};

export const uploadScan = async (files, onProgress) => {
  let scanId = "scan_" + Date.now();
  for (let index = 0; index < files.length; index += 1) {
    await delay(100);
    onProgress?.(index + 1, files.length);
  }
  await delay(200);
  return { scan_id: scanId, frame_count: files.length, status: "complete" };
};

export const SCAN_EXTS = [".ply", ".obj"];

export const importScan = async (file, onProgress) => {
  for (let i = 1; i <= 10; i++) {
    await delay(100);
    onProgress?.(file.size * (i / 10), file.size);
  }
  await delay(200);
  return {
    scan_id: "scan_" + Date.now(),
    point_count: 1543200,
    format: file.name.match(/\.[^.]+$/)?.[0].replace('.', '') || "ply",
    status: "complete"
  };
};

export const reconstruct = async (scan_id, mode) => {
  await delay(2500); // Takes a bit longer to simulate computation
  return { mesh_id: "mesh_" + Date.now(), point_count: 1200000 };
};

export const segment = async (mesh_id) => {
  await delay(2000);
  return {
    objects_id: "obj_" + Date.now(),
    objects: [
      { id: "1", label: "floor", bbox3d: [-3, -3, 0, 3, 3, 0.1] },
      { id: "2", label: "table", bbox3d: [-0.5, -0.5, 0.1, 1.5, 0.5, 0.8] },
      { id: "3", label: "box", bbox3d: [0, 0, 0.8, 0.4, 0.4, 1.2] },
      { id: "4", label: "sofa", bbox3d: [1.5, -2, 0.1, 2.5, 1, 0.9] },
      { id: "5", label: "chair", bbox3d: [-1.5, 1, 0.1, -0.5, 2, 1] }
    ]
  };
};

export const generateTwin = async (mesh_id, objects_id) => {
  await delay(2000);
  return {
    twin_id: "twin_" + Date.now(),
    object_count: 5,
    unity_scene_url: "unity://twinforge/scene_" + Date.now()
  };
};

export const plan = async (twin_id, text, lang) => {
  await delay(1500);
  return {
    task_graph_id: "graph_" + Date.now(),
    provider: "Local Gemma 4",
    graph_json: JSON.stringify({
      nodes: [
        { action: "Navigate", target: "box" },
        { action: "Pick", target: "box" },
        { action: "Navigate", target: "table" },
        { action: "Place", target: "box" }
      ]
    })
  };
};

export const train = async (twin_id, task_graph_id) => {
  await delay(3500);
  return { policy_id: "pol_" + Date.now(), sim_success_rate: 0.84 }; // Higher than 0.60 sim gate
};

export const optimize = async (policy_id, device_label) => {
  await delay(2500);
  return {
    artifact_id: "art_" + Date.now(),
    op_coverage: 98.4,
    est_latency: 12.3,
    backend: "QAIRT",
    latency_source: "npu"
  };
};

export const deploy = async (artifact_id, kind) => {
  await delay(2000);
  return {
    deployment_id: "dep_" + Date.now(),
    status: "active",
    pose_trace: [[-1, 0], [-0.5, 0], [0, 0], [0, 0.5], [0.2, 0.2]],
    inference_p50_ms: 15.2,
    compute_unit: "Hexagon NPU"
  };
};

export const sync = async (twin_id, new_scan_id) => {
  await delay(2000);
  return {
    diff_summary: { added_voxels: 125, removed_voxels: 42 },
    changed_objects: ["chair"]
  };
};

export const benchmarks = async () => {
  await delay(500);
  return {};
};
