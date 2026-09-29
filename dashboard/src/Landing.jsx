import React, { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

import Preloader from "./Preloader.jsx";
import ReconCanvas from "./ReconCanvas.jsx";
import { Link } from "./router.jsx";
import { useReveal } from "./useReveal.js";

gsap.registerPlugin(ScrollTrigger);

const STEPS = [
  ["01", "Capture", "Chunked ARCore frames from the companion app.",
   "POST /capture → scan_id, frame_count"],
  ["02", "Reconstruct",
   "Two paths, chosen per run: depth fusion for speed, COLMAP and gsplat when the scan has to hold up under a camera move.",
   "POST /reconstruct → mesh_id, point_count"],
  ["03", "Segment",
   "Geometry first — the floor splits off as a z-slab, the rest clusters by voxel connectivity and is labelled by shape. No weights to download, which is what keeps the stage offline.",
   "POST /segment → objects_id, objects[]"],
  ["04", "Generate twin",
   "Each label maps to a Unity prefab and a collider type, then Unity builds the scene in batch mode.",
   "POST /generate-twin → twin_id, unity_scene_url"],
  ["05", "Plan",
   "Say the task in English or Hindi. Sarvam plans it when a key is set; FunctionGemma plans it on the device when one is not.",
   "POST /plan → task_graph_id, provider"],
  ["06", "Train",
   "Behaviour cloning against the twin, then twenty evaluation episodes. A policy below the gate cannot leave this step.",
   "POST /train → policy_id, sim_success_rate"],
  ["07", "Optimize",
   "Qualcomm AI Hub export, with local QAIRT conversion as the fallback when the token is absent.",
   "POST /optimize → artifact_id, op_coverage, est_latency"],
  ["08", "Deploy", "Same call, simulated robot or an UNO Q on the bench.",
   "POST /deploy → deployment_id, pose_trace"],
];

const TIERS = [
  ["Phone", "ARCore session, chunked upload. The only sensor in the system."],
  ["AI PC", "Reconstruction, segmentation, Unity batch build, behaviour cloning."],
  ["Snapdragon", "The quantized policy, executing on the NPU with the radios off."],
];

// NOTE: streamlined workflow steps focused on user onboarding perspective
const SCANIVERSE_STEPS = [
  ["01", "Capture Space (Scaniverse)", "Scan your physical room or environment using Scaniverse on your mobile device and export the scan as a .ply or .obj file.", "User Action: Scaniverse App → Export PLY/OBJ"],
  ["02", "Onboard & Import Scan", "Open the DragVerse dashboard, set up your project workspace, and drag-and-drop your exported Scaniverse 3D file.", "User Action: Workspace Onboarding → Upload Scan"],
  ["03", "Verify 3D Segmentation", "Inspect auto-detected floor planes and clustered 3D objects with interactive AI semantic labels.", "User Action: Review 3D Scene & Object Labels"],
  ["04", "Generate Digital Twin", "Convert labeled objects into a 1:1 simulation environment in Unity, complete with physics colliders.", "User Action: Generate Physics-Ready Twin"],
  ["05", "Instruct Task & Train", "Specify the robot's objective using voice or text prompts, then run RL policy training in simulation.", "User Action: Voice/Text Prompt → Train Policy"],
  ["06", "Deploy to Snapdragon", "Export the quantized neural policy directly to Snapdragon NPU silicon for zero-latency, offline execution.", "User Action: Deploy Binary to Edge Hardware"],
];

export default function Landing() {
  const pipelineRef = useReveal();
  const gateRef = useReveal();
  const edgeRef = useReveal();
  const startRef = useReveal();
  const scaniverseRef = useReveal();
  const meterFillRef = useRef(null);

  // The reading sweeps in once the gate section is reached — an instrument taking a
  // measurement, not a decorative bar filling up. useReveal already handles the section's
  // fade-in; this is the one earned moment layered on top of it.
  useEffect(() => {
    const fill = meterFillRef.current;
    if (!fill) return;

    const mm = gsap.matchMedia();
    mm.add("(prefers-reduced-motion: no-preference)", () => {
      gsap.from(fill, {
        scaleX: 0,
        duration: 0.9,
        ease: "power2.out",
        scrollTrigger: { trigger: fill, start: "top 85%" },
      });
    });

    return () => mm.revert();
  }, []);

  return (
    <>
      <Preloader />

      <ReconCanvas />

      <section className="band" id="pipeline" ref={pipelineRef}>
        <aside className="notes" data-reveal>
          <p>Z is up, metres.</p>
          <p>Each step consumes the id the step before it returned.</p>
          <p>One REST surface, one Python SDK.</p>
        </aside>

        <div className="band__body">
          <h2 className="title" data-reveal>Eight calls, scan to silicon.</h2>
          <p className="prose" data-reveal>
            The order is not a diagram. Every stage takes the identifier the last one
            returned, so the sequence is enforced by the data, not by the interface.
          </p>

          <ol className="steps">
            {STEPS.map(([n, name, body, sig]) => (
              <li className="step" key={n} data-reveal>
                <span className="step__n">{n}</span>
                <div className="step__body">
                  <h3>{name}</h3>
                  <p>{body}</p>
                  <code>{sig}</code>
                </div>
              </li>
            ))}
          </ol>

          <p className="steps__loop" data-reveal>
            <span className="steps__loopmark" aria-hidden="true">↺</span>
            <span>
              The room changes. <code>POST /sync</code> re-scans it and returns the voxel
              diff — added, removed, and which objects moved — so the twin follows the space
              instead of aging out of it.
            </span>
          </p>
        </div>
      </section>

      <section className="band band--gate" ref={gateRef}>
        <aside className="notes" data-reveal>
          <p>SIM_GATE, orchestrator/service.py</p>
          <p>20 evaluation episodes</p>
          <p>HTTP 409 on failure</p>
        </aside>
        <div className="band__body">
          <h2 className="title" data-reveal>An unvalidated policy never reaches hardware.</h2>
          <p className="prose" data-reveal>
            <code>/train</code> refuses to return a policy that misses the simulated success
            gate. There is no override flag and no way to hand <code>/deploy</code> an
            artifact that skipped it — the failing run returns the measured rate and the
            threshold it missed, and stops there.
          </p>
          <div className="simgate" data-reveal>
            <div className="simgate__meter" aria-hidden="true">
              <i ref={meterFillRef} style={{ "--pass": 0.6 }} />
            </div>
            <p className="simgate__legend">
              <b>0.60</b> minimum success rate in simulation, measured before export.
            </p>
          </div>
        </div>
      </section>

      <section className="band" id="edge" ref={edgeRef}>
        <aside className="notes" data-reveal>
          <p>No cloud GPU in the loop.</p>
          <p>Airplane mode is the demo.</p>
        </aside>
        <div className="band__body">
          <h2 className="title" data-reveal>Commodity edge hardware, end to end.</h2>
          <p className="prose" data-reveal>
            Every stage was built to survive a room with no uplink. The cloud is an
            optimization, not a dependency.
          </p>
          <ul className="edge-tiers">
            {TIERS.map(([role, body]) => (
              <li className="edge-tier" key={role} data-reveal>
                <span className="edge-tier__role">{role}</span>
                <p>{body}</p>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="band band--start" id="start" ref={startRef}>
        <aside className="notes" data-reveal>
          <p>pip install -e .</p>
          <p>orchestrator :8000</p>
        </aside>
        <div className="band__body">
          <h2 className="title" data-reveal>The whole loop is one method.</h2>
          <pre className="code" data-reveal>
            <code>
              <span className="c-k">from</span> dragverse <span className="c-k">import</span> DragVerse
              {"\n\n"}
              tf = DragVerse(<span className="c-s">"http://localhost:8000"</span>)
              {"\n"}
              result = tf.run_pipeline(
              {"\n    "}frames,
              {"\n    "}text=<span className="c-s">"Take the box to the table"</span>,
              {"\n    "}mode=<span className="c-s">"fast"</span>,
              {"\n"})
              {"\n"}
              <span className="c-f">print</span>(result[<span className="c-s">"deployment_id"</span>])
            </code>
          </pre>

          <div className="cta" data-reveal>
            <Link className="btn btn--solid" to="/dashboard">
              Open the console
            </Link>
            <a className="btn" href="https://github.com/adt-kmr/DragVerse">
              Read the source
            </a>
          </div>

          <p className="prose prose--small" data-reveal>
            Run it locally with <code>make install</code>, then{" "}
            <code>uvicorn orchestrator.service:app --port 8000</code>. The console drives the
            same endpoints, one stage at a time.
          </p>
        </div>
      </section>

      <section className="band" id="scaniverse" ref={scaniverseRef}>
        <aside className="notes" data-reveal>
          <p>Scaniverse onboarding workflow</p>
          <p>Direct 3D scan ingest</p>
          <p>User onboarding journey</p>
        </aside>
        <div className="band__body">
          <h2 className="title" data-reveal>User onboarding & Scaniverse workflow.</h2>
          <p className="prose" data-reveal>
            Step-by-step user onboarding journey: from scanning a physical environment with Scaniverse to deploying an autonomous policy onto Snapdragon edge hardware:
          </p>

          <ol className="steps">
            {SCANIVERSE_STEPS.map(([n, name, body, sig]) => (
              <li className="step" key={n} data-reveal>
                <span className="step__n">{n}</span>
                <div className="step__body">
                  <h3>{name}</h3>
                  <p>{body}</p>
                  <code>{sig}</code>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>
    </>
  );
}
