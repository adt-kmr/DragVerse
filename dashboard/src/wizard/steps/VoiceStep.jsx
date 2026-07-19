import React from "react";

/** Placeholder — Task 19 wires this to POST /plan (taskText, lang). */
export default function VoiceStep({ onNext }) {
  return (
    <div className="wizard__step">
      <h2>Say the task</h2>
      <p className="prose">
        English or Hindi, typed or spoken. This step will turn it into a task graph.
      </p>
      <button className="btn btn--solid" onClick={() => onNext({})}>
        Continue
      </button>
    </div>
  );
}
