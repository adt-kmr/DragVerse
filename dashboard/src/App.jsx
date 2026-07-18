import React from "react";

import Console from "./Console.jsx";
import Landing from "./Landing.jsx";
import { Link, usePath } from "./router.jsx";

/**
 * The sheet both routes are printed on: `/` is the overview, `/dashboard` is the console.
 * Masthead, registration marks, and the footer stamp are shared, which is what keeps the
 * two routes reading as one document rather than two sites.
 */
export default function App() {
  const path = usePath();
  const onConsole = path.startsWith("/dashboard");

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
          <Link className={onConsole ? "" : "is-here"} to="/">
            Overview
          </Link>
          <Link className={onConsole ? "is-here" : ""} to="/dashboard">
            Console
          </Link>
        </nav>

        <dl className="masthead__meta">
          <div>
            <dt>Sheet</dt>
            <dd>{onConsole ? "02 / 02" : "01 / 02"}</dd>
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

      <main>{onConsole ? <Console /> : <Landing />}</main>

      <footer className="stamp">
        <span>TwinForge</span>
        <span>Edge-first Physical AI operating layer</span>
        <span>Snapdragon Multiverse · Noida · July 2026</span>
      </footer>
    </div>
  );
}
