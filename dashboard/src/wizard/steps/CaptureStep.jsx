import React from "react";

/** Placeholder — Task 16 wires this to the companion app's scan_id (POST /capture). */
export default function CaptureStep({ onNext }) {
  return (
    <div className="wizard__step">
      <h2>Scan the space</h2>
      <p className="prose">
        Point the companion app at the room. This step will show scan progress and hand off
        a scan id.
      </p>
      <button className="btn btn--solid" onClick={() => onNext({})}>
        Continue
      </button>
    </div>
  );
}
