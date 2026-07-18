import React from "react";

/**
 * One rung of the pipeline. A stage stays locked until the stage before it has produced
 * the id it consumes, which is the same precondition the orchestrator enforces — better
 * to grey the button out than to let the operator earn a 404 mid-demo.
 */
export default function Stage({ n, name, action, ready, busy, locked, error, out, onRun, children }) {
  const state = busy ? "busy" : error ? "failed" : out ? "done" : ready ? "ready" : "waiting";

  return (
    <li className={`stage ${state}`}>
      <div className="marker">{n}</div>
      <div className="body">
        <div className="head">
          <h3>{name}</h3>
          {out && <span className="out">{out}</span>}
        </div>
        <div className="fields">{children}</div>
        <button onClick={onRun} disabled={!ready || locked}>
          {busy ? "Working…" : action}
        </button>
        {error && (
          <div className="error">
            <strong>{error.message}</strong>
            {/* The sim gate refuses with the numbers attached; show them. */}
            {typeof error.detail === "object" && error.detail !== null && (
              <dl>
                {Object.entries(error.detail)
                  .filter(([key]) => key !== "error")
                  .map(([key, value]) => (
                    <React.Fragment key={key}>
                      <dt>{key.replace(/_/g, " ")}</dt>
                      <dd>{String(value)}</dd>
                    </React.Fragment>
                  ))}
              </dl>
            )}
          </div>
        )}
      </div>
    </li>
  );
}
