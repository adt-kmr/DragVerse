import React from "react";

import Console from "./Console.jsx";
import Landing from "./Landing.jsx";
import { Link, usePath } from "./router.jsx";
import Wizard from "./wizard/Wizard.jsx";

/**
 * The sheet all three routes are printed on: `/` is the overview, `/wizard` is the guided
 * setup, `/dashboard` is the operator console. Masthead, registration marks, and the
 * footer stamp are shared, which is what keeps the routes reading as one document rather
 * than three sites.
 */
export default function App() {
  const path = usePath();
  const onWizard = path.startsWith("/wizard");
  const onConsole = !onWizard && path.startsWith("/dashboard");
  const sheet = onWizard ? "02" : onConsole ? "03" : "01";

  return (
    <div className="sheet">
      <span className="reg reg--tl" aria-hidden="true" />
      <span className="reg reg--tr" aria-hidden="true" />
      <span className="reg reg--bl" aria-hidden="true" />
      <span className="reg reg--br" aria-hidden="true" />

      <header className="masthead">
        <Link className="masthead__mark" to="/">
          TwinForge
        </Link>

        <nav className="masthead__nav">
          <Link className={!onWizard && !onConsole ? "is-here" : ""} to="/">
            Overview
          </Link>
          <Link className={onWizard ? "is-here" : ""} to="/wizard">
            Wizard
          </Link>
          <Link className={onConsole ? "is-here" : ""} to="/dashboard">
            Console
          </Link>
        </nav>

        <dl className="masthead__meta">
          <div>
            <dt>Sheet</dt>
            <dd>{sheet} / 03</dd>
          </div>
          <div>
            <dt>Rev</dt>
            <dd>0.1.0</dd>
          </div>
          <div>
            <dt>Field</dt>
            <dd>Noida · Jul 2026</dd>
          </div>
        </dl>
      </header>

      <main>{onWizard ? <Wizard /> : onConsole ? <Console /> : <Landing />}</main>

      <footer className="stamp">
        <span>TwinForge</span>
        <span>Edge-first Physical AI operating layer</span>
        <span>Snapdragon Multiverse · Noida · July 2026</span>
      </footer>
    </div>
  );
}
