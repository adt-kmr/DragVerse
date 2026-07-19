import React, { useEffect, useState } from "react";

import * as api from "./api.js";
import Floorplan from "./Floorplan.jsx";
import JobFeed from "./JobFeed.jsx";
import Stage from "./Stage.jsx";
import Telemetry from "./Telemetry.jsx";

const DEFAULT_DEVICE = "Snapdragon 8 Gen 3 QRD";

const fmtBytes = (n) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB`
    : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB`
      : `${Math.max(1, Math.round(n / 1e3))} KB`;

export default function Console() {
  // One id per stage, each the input to the next — the pipeline's whole state.
  const [pipe, setPipe] = useState({ objects: [], poseTrace: [] });
  const [busy, setBusy] = useState(null);
  const [errors, setErrors] = useState({});
  const [online, setOnline] = useState(null);
  const [upload, setUpload] = useState(null);
  const [dragging, setDragging] = useState(false);

  // Form fields, in stage order.
  const [scanId, setScanId] = useState("");
  const [mode, setMode] = useState("fast");
  const [text, setText] = useState("Take the box to the table");
  const [lang, setLang] = useState("en");
  const [device, setDevice] = useState(DEFAULT_DEVICE);
  const [robotKind, setRobotKind] = useState("sim");
  const [customRobotKind, setCustomRobotKind] = useState("");
  const [rescanId, setRescanId] = useState("");
  
  // New selections
  const [aiModelSource, setAiModelSource] = useState("qualcomm");
  const [qualcommModel, setQualcommModel] = useState("Llama-3-8B-Chat-AWQ");
  const [localModel, setLocalModel] = useState("Gemma 4");
  const [trainEnvironment, setTrainEnvironment] = useState("sim");
  const [customTrainEnvironment, setCustomTrainEnvironment] = useState("");
  const [rlOption, setRlOption] = useState("no");

  useEffect(() => {
    const ping = () => api.health().then(() => setOnline(true), () => setOnline(false));
    ping();
    const timer = setInterval(ping, 5000);
    return () => clearInterval(timer);
  }, []);

  /** Run one stage and fold its result into the pipeline state. */
  const run = (key, call) => async () => {
    setBusy(key);
    setErrors((prev) => ({ ...prev, [key]: null }));
    try {
      const result = await call();
      setPipe((prev) => ({ ...prev, ...result }));
    } catch (error) {
      setErrors((prev) => ({ ...prev, [key]: error }));
    } finally {
      setBusy(null);
      setUpload(null);
    }
  };

  /**
   * Open a scan from whatever the operator picked.
   *
   * Two transports, because the two capture routes are genuinely different: the phone
   * app produces many .npz depth frames to fuse, while Scaniverse produces one finished
   * .ply/.obj cloud. Branch on extension rather than making the operator declare it.
   */
  const openScan = (picked) => {
    const files = [...picked].sort((a, b) => a.name.localeCompare(b.name));
    if (!files.length) return;

    const extOf = (f) => (f.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
    const exports_ = files.filter((f) => api.SCAN_EXTS.includes(extOf(f)));
    const frames = files.filter((f) => extOf(f) === ".npz");

    const fail = (message) =>
      setErrors((prev) => ({ ...prev, capture: new Error(message) }));

    // Route on what was actually picked. Falling through to the frame path for
    // anything unrecognised would upload a .usdz as if it were a depth frame and
    // report success — the failure would only surface three stages later.
    if (!exports_.length && !frames.length) {
      const rejected = [...new Set(files.map(extOf))].filter(Boolean);
      const deferred = rejected.filter((e) =>
        [".glb", ".gltf", ".usdz", ".fbx"].includes(e));
      fail(deferred.length
        ? `${deferred.join(", ")} is not supported yet — re-export as ` +
          `${api.SCAN_EXTS.join(" or ")} from Scaniverse.`
        : `Cannot read ${rejected.join(", ") || "that file"}. Pick a ` +
          `${api.SCAN_EXTS.join(" or ")} export, or the phone app's .npz frames.`);
      return;
    }
    if (exports_.length && frames.length) {
      fail("Pick either one scan export or a set of .npz frames, not both.");
      return;
    }

    if (exports_.length) {
      if (exports_.length > 1) {
        fail(`Pick one scan export at a time — got ${exports_.length}. ` +
             `A single .ply or .obj is the whole capture.`);
        return;
      }
      run("capture", async () => {
        const result = await api.importScan(exports_[0], (loaded, total) =>
          setUpload(`${fmtBytes(loaded)} of ${fmtBytes(total)}`));
        setScanId(result.scan_id);
        return { scanId: result.scan_id, scanPoints: result.point_count,
                 scanSource: result.format, scanFrames: null,
                 scanState: result.status };
      })();
      return;
    }

    // Chunk order is capture order, and the picker does not promise either — the sort
    // above is what makes the frames fuse in the order they were shot.
    run("capture", async () => {
      const result = await api.uploadScan(frames, (done, total) =>
        setUpload(`${done}/${total} chunks`));
      setScanId(result.scan_id);
      return { scanId: result.scan_id, scanFrames: result.frame_count,
               scanPoints: null, scanSource: null, scanState: result.status };
    })();
  };

  const onPick = (event) => {
    const files = [...event.target.files];
    event.target.value = ""; // so re-picking the same file still fires a change
    openScan(files);
  };

  const onDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    if (busy === null) openScan(event.dataTransfer.files);
  };

  /** Clear the pipeline so a second scan can be run without a page reload. */
  const reset = () => {
    setPipe({ objects: [], poseTrace: [] });
    setErrors({});
    setUpload(null);
    setScanId("");
    setRescanId("");
  };

  const stageProps = (key) => ({ busy: busy === key, error: errors[key], locked: busy !== null });

  return (
    <div className="console">
      <div className="console__head">
        <div>
          <span className="eyebrow">Operator console</span>
          <h1 className="title">Run the loop, one stage at a time.</h1>
        </div>
        <div className="console__status">
          {pipe.scanId && (
            <button type="button" className="btn btn--ghost" onClick={reset}
                    disabled={busy !== null}>
              New run
            </button>
          )}
          <span className={`link ${online === null ? "unknown" : online ? "up" : "down"}`}>
            {online === null
              ? "checking orchestrator"
              : online
                ? "orchestrator online"
                : "orchestrator unreachable"}
          </span>
        </div>
      </div>

      <div className="console__grid">
        <ol className="rail">
          <Stage
            n="01" name="Capture"
            out={pipe.scanPoints != null
              ? `${pipe.scanPoints.toLocaleString()} points · ${pipe.scanSource} · ${pipe.scanState}`
              : pipe.scanFrames != null && `${pipe.scanFrames} frames · ${pipe.scanState}`}
            action="Check scan" ready={scanId.trim().length > 0}
            onRun={run("capture", async () => {
              const result = await api.scanStatus(scanId.trim());
              return { scanId: scanId.trim(), scanFrames: result.frame_count,
                       scanPoints: null, scanSource: null, scanState: result.status };
            })}
            {...stageProps("capture")}
          >
            <div
              className={`dropzone${dragging ? " dropzone--over" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <input id="scan-file" type="file" multiple
                     accept={`${api.SCAN_EXTS.join(",")},.npz`}
                     onChange={onPick} disabled={busy !== null} />
              <label htmlFor="scan-file">
                Drop a scan here, or <span className="dropzone__link">browse</span>
              </label>
            </div>
            <p className="hint">
              {upload
                ? `Uploading ${upload}…`
                : "One .ply or .obj exported from Scaniverse, or the phone app's .npz " +
                  "frames. You can also name a scan already uploaded."}
            </p>
            <label>
              Scan id
              <input value={scanId} onChange={(e) => setScanId(e.target.value)}
                     placeholder="from Scaniverse or the phone app" />
            </label>
          </Stage>

          <Stage
            n="02" name="Reconstruct"
            out={pipe.meshId && `${(pipe.points ?? 0).toLocaleString()} points · ${pipe.meshId.slice(0, 8)}`}
            action="Reconstruct" ready={Boolean(pipe.scanId)}
            onRun={run("reconstruct", async () => {
              const result = await api.reconstruct(pipe.scanId, mode);
              return { meshId: result.mesh_id, points: result.point_count };
            })}
            {...stageProps("reconstruct")}
          >
            <label>
              Path
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                <option value="fast">fast — depth fusion</option>
                <option value="fidelity">fidelity — COLMAP, needs the binary</option>
              </select>
            </label>
          </Stage>

          <Stage
            n="03" name="Segment" out={pipe.objects.length > 0 && `${pipe.objects.length} objects`}
            action="Segment" ready={Boolean(pipe.meshId)}
            onRun={run("segment", async () => {
              const result = await api.segment(pipe.meshId);
              return { objectsId: result.objects_id, objects: result.objects };
            })}
            {...stageProps("segment")}
          >
            <p className="hint">Labels the point cloud and draws the plan view.</p>
          </Stage>

          <Stage
            n="04" name="Generate twin" out={pipe.twinId && `${pipe.twinObjects} placed · ${pipe.twinId.slice(0, 8)}`}
            action="Generate twin" ready={Boolean(pipe.objectsId)}
            onRun={run("twin", async () => {
              const result = await api.generateTwin(pipe.meshId, pipe.objectsId);
              return { twinId: result.twin_id, twinObjects: result.object_count,
                       sceneUrl: result.unity_scene_url };
            })}
            {...stageProps("twin")}
          >
            {pipe.sceneUrl && <p className="hint">Scene: <code>{pipe.sceneUrl}</code></p>}
          </Stage>

          <Stage
            n="05" name="Plan"
            out={pipe.graphId && `${pipe.graphNodes} step${pipe.graphNodes === 1 ? "" : "s"} via ${pipe.provider}`}
            action="Plan" ready={Boolean(pipe.twinId) && text.trim().length > 0}
            onRun={run("plan", async () => {
              const result = await api.plan(pipe.twinId, text, lang);
              const graph = JSON.parse(result.graph_json);
              return { graphId: result.task_graph_id, provider: result.provider,
                       graphNodes: graph.nodes.length, graph: graph.nodes };
            })}
            {...stageProps("plan")}
          >
            <label>
              Instruction
              <input value={text} onChange={(e) => setText(e.target.value)} />
            </label>
            <label>
              Language
              <select value={lang} onChange={(e) => setLang(e.target.value)}>
                <option value="en">English</option>
                <option value="hi">Hindi</option>
              </select>
            </label>
            {pipe.graph && (
              <ol className="graph">
                {pipe.graph.map((node, i) => (
                  <li key={i}><span>{node.action}</span> {node.target}</li>
                ))}
              </ol>
            )}
          </Stage>
          <Stage
            n="06" name="Train" out={pipe.policyId && `${Math.round(pipe.successRate * 100)}% in sim`}
            action="Train" ready={Boolean(pipe.twinId)}
            onRun={run("train", async () => {
              const result = await api.train(pipe.twinId, pipe.graphId);
              return { policyId: result.policy_id, successRate: result.sim_success_rate };
            })}
            {...stageProps("train")}
          >
            <p className="hint">Behaviour cloning, gated at 60% sim success before anything ships.</p>
            <label>
              Training Environment (Robot/Rover)
              <select value={trainEnvironment} onChange={(e) => setTrainEnvironment(e.target.value)}>
                <option value="sim">Simulation (Sim)</option>
                <option value="unitree_go">Unitree Go</option>
                <option value="unitree_b2">Unitree B2</option>
                <option value="boston_dynamics_spot">Boston Dynamics Spot</option>
                <option value="clearpath_jackal">Clearpath Jackal</option>
                <option value="agile_digit">Agile Robotics Digit</option>
                <option value="other">Others...</option>
              </select>
            </label>
            {trainEnvironment === "other" && (
              <label>
                Custom Robot/Rover
                <input value={customTrainEnvironment} onChange={(e) => setCustomTrainEnvironment(e.target.value)} placeholder="Enter robot/rover name" />
              </label>
            )}
            <label>
              Reinforcement Learning Algorithm
              <select value={rlOption} onChange={(e) => setRlOption(e.target.value)}>
                <option value="no">No</option>
                <option value="yes">Yes</option>
              </select>
            </label>
          </Stage>

          <Stage
            n="07" name="Optimize"
            /* op_coverage is null off the AI Hub path — say so, rather than render
               "null% ops" and let an unmeasured bundle read as a profiled one. */
            out={pipe.artifactId && [
              pipe.opCoverage == null ? "ops not profiled" : `${pipe.opCoverage}% ops`,
              `${pipe.latency} ms ${pipe.latencySource === "host-cpu" ? "(host cpu)" : ""}`.trim(),
              pipe.backend,
            ].join(" · ")}
            action="Optimize" ready={Boolean(pipe.policyId)}
            onRun={run("optimize", async () => {
              const result = await api.optimize(pipe.policyId, device);
              return { artifactId: result.artifact_id, opCoverage: result.op_coverage,
                       latency: result.est_latency, backend: result.backend,
                       latencySource: result.latency_source };
            })}
            {...stageProps("optimize")}
          >
            <label>
              On-device AI Model Source
              <select value={aiModelSource} onChange={(e) => setAiModelSource(e.target.value)}>
                <option value="qualcomm">Qualcomm Hub (Quantized)</option>
                <option value="local">Other Local Models</option>
              </select>
            </label>
            {aiModelSource === "qualcomm" ? (
              <label>
                Qualcomm Model
                <select value={qualcommModel} onChange={(e) => setQualcommModel(e.target.value)}>
                  <option value="Llama-3-8B-Chat-AWQ">Llama 3 8B Chat (AWQ)</option>
                  <option value="Llama-2-7B-Chat-INT8">Llama 2 7B Chat (INT8)</option>
                  <option value="Mistral-7B-v0.1-AWQ">Mistral 7B v0.1 (AWQ)</option>
                  <option value="Qwen-1.5-4B-Chat-AWQ">Qwen 1.5 4B Chat (AWQ)</option>
                  <option value="Stable-Diffusion-v1.5-INT8">Stable Diffusion v1.5 (INT8)</option>
                  <option value="Whisper-Base-INT8">Whisper Base (INT8)</option>
                  <option value="Baichuan-7B-AWQ">Baichuan 7B (AWQ)</option>
                  <option value="YI-6B-AWQ">YI 6B (AWQ)</option>
                </select>
              </label>
            ) : (
              <label>
                Local Model
                <select value={localModel} onChange={(e) => setLocalModel(e.target.value)}>
                  <option value="Gemma 2B">Gemma 2B</option>
                  <option value="Gemma 7B">Gemma 7B</option>
                  <option value="Phi-3-Mini">Phi-3-Mini</option>
                  <option value="Phi-2">Phi-2</option>
                  <option value="Llama-3-8B-Instruct">Llama-3-8B-Instruct</option>
                  <option value="Mistral-7B-Instruct-v0.2">Mistral-7B-Instruct-v0.2</option>
                </select>
              </label>
            )}
            <label>
              Target device
              <input value={device} onChange={(e) => setDevice(e.target.value)} />
            </label>
          </Stage>

          <Stage
            n="08" name="Deploy"
            out={pipe.deploymentId && [
              pipe.deployStatus,
              `${pipe.poseTrace.length} waypoints`,
              // Measured on the int8 policy that actually shipped, per control tick.
              pipe.tickP50 != null && `${pipe.tickP50} ms/tick on ${pipe.computeUnit}`,
            ].filter(Boolean).join(" · ")}
            action="Deploy" ready={Boolean(pipe.artifactId)}
            onRun={run("deploy", async () => {
              const result = await api.deploy(pipe.artifactId, robotKind);
              return { deploymentId: result.deployment_id, deployStatus: result.status,
                       poseTrace: result.pose_trace, tickP50: result.inference_p50_ms,
                       computeUnit: result.compute_unit };
            })}
            {...stageProps("deploy")}
          >
            <label>
              Robot
              <select value={robotKind} onChange={(e) => setRobotKind(e.target.value)}>
                <option value="sim">Simulation (Sim)</option>
                <option value="unitree_go">Unitree Go</option>
                <option value="unitree_b2">Unitree B2</option>
                <option value="boston_dynamics_spot">Boston Dynamics Spot</option>
                <option value="clearpath_jackal">Clearpath Jackal</option>
                <option value="agile_digit">Agile Robotics Digit</option>
                <option value="other">Others...</option>
              </select>
            </label>
            {robotKind === "other" && (
              <label>
                Custom Robot/Rover
                <input value={customRobotKind} onChange={(e) => setCustomRobotKind(e.target.value)} placeholder="Enter robot/rover name" />
              </label>
            )}
          </Stage>

          <Stage
            n="↺" name="Rescan" out={pipe.diff && `${pipe.diff.added_voxels} added · ${pipe.diff.removed_voxels} removed`}
            action="Rescan and diff" ready={Boolean(pipe.twinId) && rescanId.trim().length > 0}
            onRun={run("sync", async () => {
              const result = await api.sync(pipe.twinId, rescanId.trim());
              return { diff: result.diff_summary, changed: result.changed_objects };
            })}
            {...stageProps("sync")}
          >
            <label>
              New scan id
              <input value={rescanId} onChange={(e) => setRescanId(e.target.value)}
                     placeholder="scan the room again" />
            </label>
            {pipe.changed?.length > 0 && (
              <p className="hint">Changed: {pipe.changed.join(", ")}</p>
            )}
          </Stage>
        </ol>

        <section className="view">
          <div className="panel">
            <h2>Plan view</h2>
            <Floorplan objects={pipe.objects} poseTrace={pipe.poseTrace} />
          </div>
          <div className="panel">
            <h2>Jobs</h2>
            <JobFeed />
          </div>
          <div className="panel wide">
            <h2>Edge telemetry</h2>
            <Telemetry />
          </div>
        </section>
      </div>
    </div>
  );
}
