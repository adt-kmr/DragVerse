// Simulated replay of the orchestrator REST surface, for the public demo build, which has
// no orchestrator behind it. The console labels every run from here as simulated.
//
// Values are the shapes the real endpoints return, never measurements: nothing here is
// profiled, so latency and NPU coverage stay null rather than inventing numbers.

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

export const health = async () => ({ status: "ok" });

export const scanStatus = async () => {
  await delay(300);
  return { frame_count: 120, status: "complete" };
};

export const uploadScan = async (files, onProgress) => {
  for (let index = 0; index < files.length; index += 1) {
    await delay(100);
    onProgress?.(index + 1, files.length);
  }
  return { scan_id: "sim_scan", frame_count: files.length, status: "complete" };
};

export const SCAN_EXTS = [".ply", ".obj"];

export const importScan = async (file, onProgress) => {
  for (let i = 1; i <= 10; i += 1) {
    await delay(100);
    onProgress?.(file.size * (i / 10), file.size);
  }
  return {
    scan_id: "sim_scan",
    point_count: 1543200,
    format: file.name.match(/\.([^.]+)$/)?.[1] ?? "ply",
    status: "complete",
  };
};

export const reconstruct = async () => {
  await delay(1500);
  return { mesh_id: "sim_mesh", point_count: 1200000 };
};

export const segment = async () => {
  await delay(1200);
  return {
    objects_id: "sim_objects",
    objects: [
      { id: "1", label: "floor", bbox3d: [-3, -3, 0, 3, 3, 0.1] },
      { id: "2", label: "table", bbox3d: [-0.5, -0.5, 0.1, 1.5, 0.5, 0.8] },
      { id: "3", label: "box", bbox3d: [0, 0, 0.8, 0.4, 0.4, 1.2] },
      { id: "4", label: "sofa", bbox3d: [1.5, -2, 0.1, 2.5, 1, 0.9] },
      { id: "5", label: "chair", bbox3d: [-1.5, 1, 0.1, -0.5, 2, 1] },
    ],
  };
};

export const generateTwin = async () => {
  await delay(1200);
  return { twin_id: "sim_twin", object_count: 5, unity_scene_url: "sim_twin/scene.json" };
};

export const plan = async () => {
  await delay(800);
  return {
    task_graph_id: "sim_graph",
    // The offline planner's real provider name; Sarvam only runs with an API key.
    provider: "keyword",
    graph_json: JSON.stringify({
      nodes: [
        { action: "Navigate", target: "box" },
        { action: "Pick", target: "box" },
        { action: "Navigate", target: "table" },
        { action: "Place", target: "box" },
      ],
    }),
  };
};

export const train = async () => {
  await delay(2000);
  return { policy_id: "sim_policy", sim_success_rate: 0.84 };
};

export const optimize = async () => {
  await delay(1200);
  return {
    artifact_id: "sim_artifact",
    op_coverage: null,
    est_latency: null,
    backend: "int8",
    latency_source: "simulated",
  };
};

export const deploy = async () => {
  await delay(1200);
  return {
    deployment_id: "sim_deployment",
    status: "simulated",
    pose_trace: [[-1, 0], [-0.5, 0], [0, 0], [0, 0.5], [0.2, 0.2]],
    inference_p50_ms: null,
    compute_unit: "CPU",
  };
};

export const sync = async () => {
  await delay(1200);
  return { diff_summary: { added_voxels: 125, removed_voxels: 42 }, changed_objects: ["chair"] };
};

export const benchmarks = async () => ({});
