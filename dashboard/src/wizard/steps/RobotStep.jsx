import React, { useState } from "react";

const ROBOT_OPTIONS = [
  {
    kind: "sim",
    label: "Simulated",
    description: "Test in the digital twin, no hardware required",
  },
  {
    kind: "unoq",
    label: "Buggy (UNO Q)",
    description: "Deploy to the physical UNO Q-driven buggy",
  },
];

export default function RobotStep({ onNext }) {
  const [selected, setSelected] = useState(null);

  const handleSelect = (kind) => {
    setSelected(kind);
    onNext({ robotKind: kind });
  };

  return (
    <div className="wizard__step">
      <h2>Choose a robot</h2>
      <p className="prose">Simulated, or an UNO Q on the bench.</p>

      <div className="robot-options">
        {ROBOT_OPTIONS.map((option) => (
          <button
            key={option.kind}
            className={`robot-option ${selected === option.kind ? "is-selected" : ""}`}
            onClick={() => handleSelect(option.kind)}
          >
            <strong>{option.label}</strong>
            <p>{option.description}</p>
          </button>
        ))}
      </div>
    </div>
  );
}
