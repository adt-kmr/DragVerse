import React, { useEffect, useState } from "react";

import * as api from "./api.js";
import Floorplan from "./Floorplan.jsx";
import JobFeed from "./JobFeed.jsx";
import Stage from "./Stage.jsx";

const DEFAULT_DEVICE = "Snapdragon 8 Gen 3 QRD";

export default function App() {
  // One id per stage, each the input to the next — the pipeline's whole state.
  const [pipe, setPipe] = useState({ objects: [], poseTrace: [] });
  const [busy, setBusy] = useState(null);
  const [errors, setErrors] = useState({});
  const [online, setOnline] = useState(null);

  // Form fields, in stage order.
  const [scanId, setScanId] = useState("");
  const [mode, setMode] = useState("fast");
  const [text, setText] = useState("Take the box to the table");
  const [lang, setLang] = useState("en");
  const [device, setDevice] = useState(DEFAULT_DEVICE);
  const [robotKind, setRobotKind] = useState("sim");
  const [rescanId, setRescanId] = useState("");

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
    }
  };

  const stageProps = (key) => ({ busy: busy === key, error: errors[key], locked: busy !== null });

  return (
    <div className="console">
      <header>
        <div>
          <h1>TwinForge</h1>
          <p className="eyebrow">Scan to policy, eight stages</p>
        </div>
        <span className={`link ${online === null ? "unknown" : online ? "up" : "down"}`}>
          {online === null ? "checking orchestrator" : online ? "orchestrator online" : "orchestrator unreachable"}
        </span>
      </header>

      <main>
        <ol className="rail">
          <Stage
            n="01" name="Capture" out={pipe.scanFrames != null && `${pipe.scanFrames} frames · ${pipe.scanState}`}
            action="Check scan" ready={scanId.trim().length > 0}
            onRun={run("capture", async () => {
              const result = await api.scanStatus(scanId.trim());
              return { scanId: scanId.trim(), scanFrames: result.frame_count, scanState: result.status };
            })}
            {...stageProps("capture")}
          >
            <label>
              Scan id
              <input value={scanId} onChange={(e) => setScanId(e.target.value)}
                     placeholder="from the phone app" />
            </label>
            <p className="hint">The companion app mints the id on its first chunk.</p>
          </Stage>

          <Stage
            n="02" name="Reconstruct" out={pipe.meshId && `${pipe.points.toLocaleString()} points · ${pipe.meshId.slice(0, 8)}`}
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
          </Stage>

          <Stage
            n="07" name="Optimize" out={pipe.artifactId && `${pipe.opCoverage}% ops · ${pipe.latency} ms · ${pipe.backend}`}
            action="Optimize" ready={Boolean(pipe.policyId)}
            onRun={run("optimize", async () => {
              const result = await api.optimize(pipe.policyId, device);
              return { artifactId: result.artifact_id, opCoverage: result.op_coverage,
                       latency: result.est_latency, backend: result.backend };
            })}
            {...stageProps("optimize")}
          >
            <label>
              Target device
              <input value={device} onChange={(e) => setDevice(e.target.value)} />
            </label>
          </Stage>

          <Stage
            n="08" name="Deploy" out={pipe.deploymentId && `${pipe.deployStatus} · ${pipe.poseTrace.length} waypoints`}
            action="Deploy" ready={Boolean(pipe.artifactId)}
            onRun={run("deploy", async () => {
              const result = await api.deploy(pipe.artifactId, robotKind);
              return { deploymentId: result.deployment_id, deployStatus: result.status,
                       poseTrace: result.pose_trace };
            })}
            {...stageProps("deploy")}
          >
            <label>
              Robot
              <select value={robotKind} onChange={(e) => setRobotKind(e.target.value)}>
                <option value="sim">sim</option>
                <option value="unoq">UNO Q</option>
              </select>
            </label>
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
        </section>
      </main>
    </div>
  );
}
