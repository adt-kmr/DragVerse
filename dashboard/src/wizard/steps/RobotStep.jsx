import React from "react";

/** Placeholder — Task 17 wires this to picking sim vs. UNO Q (robot_kind). */
export default function RobotStep({ onNext }) {
  return (
    <div className="wizard__step">
      <h2>Choose a robot</h2>
      <p className="prose">Simulated, or an UNO Q on the bench. This step will let you pick.</p>
      <button className="btn btn--solid" onClick={() => onNext({})}>
        Continue
      </button>
    </div>
  );
}
