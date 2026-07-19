import React from "react";

/** Placeholder — Task 21 wires this to POST /optimize then POST /deploy. */
export default function DeployStep({ onNext }) {
  return (
    <div className="wizard__step">
      <h2>Deploy</h2>
      <p className="prose">
        Export for the target device and send the policy to the robot. This step will show
        the deployment id.
      </p>
      <button className="btn btn--solid" onClick={() => onNext({})}>
        Finish
      </button>
    </div>
  );
}
