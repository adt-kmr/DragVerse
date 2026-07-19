import React from "react";

/**
 * Placeholder — Task 18 wires this to a static AI model picker (semantic model
 * variant / RL warm-start checkpoint). The twin is already built by CaptureStep
 * (Task 16 chains reconstruct/segment/generate-twin) by the time the operator
 * reaches this step — do not re-run that chain here.
 */
export default function ModelStep({ onNext }) {
  return (
    <div className="wizard__step">
      <h2>Which AI model do you want to use?</h2>
      <p className="prose">
        Choose the perception/policy base for this run. This step will show a picker for the
        semantic model variant and any available warm-start checkpoint.
      </p>
      <button className="btn btn--solid" onClick={() => onNext({})}>
        Continue
      </button>
    </div>
  );
}
