import React, { useEffect, useState } from "react";
import { benchmarks } from "./api";

// Blueprint section 14: below this share of layers on the NPU, a model needs a GPU
// fallback path rather than being called "NPU-deployed".
const NPU_GATE = 80;

/**
 * Where each workload runs, and why it belongs there. This is the argument the
 * benchmark numbers below are evidence for — the UNO Q row is deliberately "no NPU",
 * because a QRB2210 has an Adreno 702 and a low-power Hexagon DSP, not a tensor NPU.
 */
const TIERS = [
  { tier: "Perception", silicon: "OnePlus 15 · 8 Elite", unit: "Hexagon NPU",
    workload: "YOLO-World + MobileSAM, INT8" },
  { tier: "Orchestration", silicon: "X Elite AI PC", unit: "NPU",
    workload: "Reconstruction, twin gen, planning" },
  { tier: "Actuation", silicon: "Arduino UNO Q · QRB2210", unit: "CPU (no NPU)",
    workload: "INT8 linear policy" },
];

/** Prefer the measured p50 over the single-point estimate when the profile has runs. */
const latency = (r) => r?.latency_p50_ms ?? r?.latency_ms ?? null;

/**
 * NPU vs CPU — same compiled binary, same device, only the compute unit differs.
 * A ratio is what an efficiency claim needs; an absolute number alone proves nothing.
 */
function Compare({ npu, cpu, speedup }) {
  const npuMs = latency(npu);
  const cpuMs = latency(cpu);
  if (!npuMs || !cpuMs) return null;
  const worst = Math.max(npuMs, cpuMs);
  const bar = (ms) => `${Math.max(2, (ms / worst) * 100)}%`;
  return (
    <div className="compare">
      <div className="bar-row">
        <span className="bar-label">NPU</span>
        <span className="bar npu" style={{ width: bar(npuMs) }} />
        <span className="bar-value">{npuMs} ms</span>
      </div>
      <div className="bar-row">
        <span className="bar-label">CPU</span>
        <span className="bar cpu" style={{ width: bar(cpuMs) }} />
        <span className="bar-value">{cpuMs} ms</span>
      </div>
      {speedup && <p className="speedup">{speedup}× faster on the NPU</p>}
    </div>
  );
}

/** p50/p95/p99 across real runs — a control loop is judged on its tail. */
function Tail({ record }) {
  if (!record.latency_p95_ms) return null;
  return (
    <p className="tail">
      p50 {record.latency_p50_ms} · p95 {record.latency_p95_ms} · p99{" "}
      {record.latency_p99_ms} ms <span className="muted">({record.runs} runs)</span>
    </p>
  );
}

function ModelCard({ record }) {
  const npu = record.npu ?? {};
  const passed = record.meets_80pct_npu_gate;
  return (
    <article className="bench-card">
      <header>
        <h4>{record.model}</h4>
        <span className={passed ? "gate pass" : "gate fail"}>
          {npu.op_coverage_pct == null ? "not profiled" : `${npu.op_coverage_pct}% on NPU`}
        </span>
      </header>

      {npu.error ? (
        <p className="bench-error">compile failed — {npu.error}</p>
      ) : (
        <>
          <dl>
            <dt>Device</dt><dd>{npu.device ?? "—"}</dd>
            <dt>Runtime</dt><dd>{npu.runtime ?? "—"} · {npu.precision ?? "—"}</dd>
            <dt>Layers on NPU</dt>
            <dd>{npu.layers_on_npu ?? "—"} / {npu.layers_total ?? "—"}</dd>
            {/* Where the time goes, which layer count alone can hide. */}
            <dt>Time on NPU</dt>
            <dd>{npu.time_on_npu_pct != null ? `${npu.time_on_npu_pct}%` : "—"}</dd>
            {npu.fallback_units?.length > 0 && (
              <>
                <dt>Falls back to</dt><dd>{npu.fallback_units.join(", ")}</dd>
              </>
            )}
            <dt>Peak memory</dt>
            <dd>{npu.peak_memory_bytes
              ? `${(npu.peak_memory_bytes / 1e6).toFixed(1)} MB` : "—"}</dd>
          </dl>
          <Tail record={npu} />
          {record.components && (
            <details className="fallbacks">
              <summary>{Object.keys(record.components).length} components</summary>
              <ul>
                {Object.entries(record.components).map(([name, c]) => (
                  <li key={name}>
                    <code>{name}</code>{" "}
                    <span className="muted">
                      {c.op_coverage_pct}% NPU ·{" "}
                      {c.latency_p50_ms ?? c.latency_ms} ms
                    </span>
                  </li>
                ))}
              </ul>
            </details>
          )}
          {npu.top_fallback_layers?.length > 0 && (
            <details className="fallbacks">
              <summary>{npu.top_fallback_layers.length} layers off the NPU</summary>
              <ul>
                {npu.top_fallback_layers.map((l) => (
                  <li key={l.name}>
                    <code>{l.name}</code> <span className="muted">{l.type}</span>
                    {" → "}{l.unit} <span className="muted">{l.us}µs</span>
                  </li>
                ))}
              </ul>
            </details>
          )}
          <Compare npu={npu} cpu={record.cpu} speedup={record.speedup_vs_cpu} />
          {!passed && npu.op_coverage_pct != null && (
            <p className="bench-warn">Below the {NPU_GATE}% gate — needs a GPU fallback.</p>
          )}
        </>
      )}
    </article>
  );
}

/**
 * Measured on-device numbers from the AI Hub device cloud. Renders an explicit
 * "not profiled yet" rather than zeros when the harness has not been run — an
 * unprofiled model must never read as a fast one.
 */
export default function Telemetry() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    benchmarks().then(setData).catch((e) => setError(e.message));
  }, []);

  return (
    <section className="telemetry">
      <h3>Silicon utilization</h3>

      <table className="tiers">
        <thead>
          <tr><th>Tier</th><th>Silicon</th><th>Compute unit</th><th>Workload</th></tr>
        </thead>
        <tbody>
          {TIERS.map((t) => (
            <tr key={t.tier}>
              <td>{t.tier}</td><td>{t.silicon}</td>
              <td className={t.unit.includes("no NPU") ? "unit-cpu" : "unit-npu"}>
                {t.unit}
              </td>
              <td>{t.workload}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {error && <p className="bench-error">benchmarks unavailable — {error}</p>}

      {data && !data.profiled && (
        <p className="bench-empty">
          Not profiled yet. Run <code>python -m deployment.aihub_export.profile_models
          --model &lt;slug&gt; --compare</code> to measure on real hardware.
        </p>
      )}

      {data?.models?.length > 0 && (
        <>
          <div className="bench-grid">
            {data.models.map((m) => <ModelCard key={m.model} record={m} />)}
          </div>
          <p className="bench-source">
            Measured on physical devices in the Qualcomm AI Hub device cloud — not
            compile-time estimates.
          </p>
        </>
      )}
    </section>
  );
}
