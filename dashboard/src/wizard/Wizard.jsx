import React, { useState } from "react";

import ActivityStep from "./steps/ActivityStep.jsx";
import CaptureStep from "./steps/CaptureStep.jsx";
import DeployStep from "./steps/DeployStep.jsx";
import ModelStep from "./steps/ModelStep.jsx";
import RobotStep from "./steps/RobotStep.jsx";
import VoiceStep from "./steps/VoiceStep.jsx";

const STEPS = [
  { name: "Capture", Component: CaptureStep },
  { name: "Robot", Component: RobotStep },
  { name: "Model", Component: ModelStep },
  { name: "Voice", Component: VoiceStep },
  { name: "Activity", Component: ActivityStep },
  { name: "Deploy", Component: DeployStep },
];

/**
 * The guided front door to the same eight-call pipeline the console drives
 * one stage at a time — six steps here because "robot" and "model" bundle
 * two console stages apiece for someone who isn't reading the API. State is
 * one object, appended to as each step finishes; nothing reads it back until
 * a later step needs an earlier id (Tasks 16-21).
 */
export default function Wizard() {
  const [index, setIndex] = useState(0);
  const [state, setState] = useState({
    scanId: null,
    robotKind: null,
    modelChoice: null,
    taskText: "",
    lang: "en",
    activity: null,
    deploymentId: null,
  });

  const onNext = (partial) => {
    setState((prev) => ({ ...prev, ...partial }));
    setIndex((prev) => Math.min(prev + 1, STEPS.length - 1));
  };

  const { name, Component } = STEPS[index];

  return (
    <div className="wizard">
      <span className="eyebrow">Guided setup</span>
      <h1 className="title wizard__title">{name}</h1>

      <ol className="wizard__steps">
        {STEPS.map((step, i) => (
          <li key={step.name} className={i === index ? "is-here" : i < index ? "is-done" : ""}>
            <span className="wizard__marker">{i + 1}</span>
            {step.name}
          </li>
        ))}
      </ol>
      <p className="hint">
        Step {index + 1} of {STEPS.length}
      </p>

      <div className="wizard__body">
        <Component state={state} onNext={onNext} />
      </div>
    </div>
  );
}
