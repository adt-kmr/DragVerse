import React from "react";

/** Placeholder — Task 20 wires this to POST /train and the sim-gate result. */
export default function ActivityStep({ onNext }) {
  return (
    <div className="wizard__step">
      <h2>Train the policy</h2>
      <p className="prose">
        Behaviour cloning against the twin, gated on simulated success. This step will show
        that run.
      </p>
      <button className="btn btn--solid" onClick={() => onNext({})}>
        Continue
      </button>
    </div>
  );
}
