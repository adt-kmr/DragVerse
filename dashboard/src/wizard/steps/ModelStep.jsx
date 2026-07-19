import React from "react";

/** Placeholder — Task 18 wires this to reconstruct/segment/generate-twin. */
export default function ModelStep({ onNext }) {
  return (
    <div className="wizard__step">
      <h2>Build the twin</h2>
      <p className="prose">
        Reconstruct, segment, and generate the Unity scene from the scan. This step will show
        that build running.
      </p>
      <button className="btn btn--solid" onClick={() => onNext({})}>
        Continue
      </button>
    </div>
  );
}
