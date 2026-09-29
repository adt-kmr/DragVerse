import React, { useEffect, useRef, useState } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

import { createScene, STAGES, PLAN_LOG } from "./recon.js";
import { createAperture } from "./aperture.js";

gsap.registerPlugin(ScrollTrigger);

/**
 * The signature element: one canvas carrying capture → deploy, scrubbed by scroll.
 *
 * The stage caption is React state because it changes six times. The point counter is
 * written straight to the DOM — it changes every frame, and re-rendering the tree at
 * scroll rate to move one number is how a smooth scrub turns into a janky one.
 */
export default function ReconCanvas() {
  const trackRef = useRef(null);
  const canvasRef = useRef(null);
  const apertureRef = useRef(null);
  const apertureCtl = useRef(null);
  const titleRef = useRef(null);
  const captionRef = useRef(null);
  const readoutRef = useRef(null);
  const pointsRef = useRef(null);
  const objectsRef = useRef(null);
  const planRefs = useRef([]);
  const [phase, setPhase] = useState(0);
  const [atRest, setAtRest] = useState(true);

  useEffect(() => {
    const scene = createScene(canvasRef.current);
    scene.resize();

    const aperture = createAperture(apertureRef.current, "#ff2900");
    aperture.resize();
    apertureCtl.current = aperture;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let lastPhase = -1;
    let lastRest = true;
    let rafId = null;
    let rafStart = null;

    const spinAperture = (now) => {
      if (rafStart === null) rafStart = now;
      aperture.render((now - rafStart) / 1000);
      rafId = requestAnimationFrame(spinAperture);
    };
    if (reduced) aperture.render(0);
    else rafId = requestAnimationFrame(spinAperture);

    const paint = (progress) => {
      const state = scene.render(progress);
      if (pointsRef.current) pointsRef.current.textContent = state.points.toLocaleString();
      if (objectsRef.current) objectsRef.current.textContent = String(state.objects);
      if (state.phase !== lastPhase) {
        lastPhase = state.phase;
        setPhase(state.phase);
      }
      if (state.rest !== lastRest) {
        lastRest = state.rest;
        setAtRest(state.rest);
      }
      if (state.plan) {
        state.plan.forEach((alpha, i) => {
          const el = planRefs.current[i];
          if (el) el.style.opacity = alpha;
        });
      }
    };

    // Reduced motion: no scrub. Show the finished twin and let the section be one screen.
    if (reduced) {
      gsap.set([titleRef.current, apertureRef.current], { opacity: 0 });
      gsap.set([captionRef.current, readoutRef.current], { opacity: 1 });
      paint(1);
      const onResize = () => {
        scene.resize();
        paint(1);
        aperture.resize();
        aperture.render(0);
      };
      window.addEventListener("resize", onResize);
      return () => {
        window.removeEventListener("resize", onResize);
        if (rafId !== null) cancelAnimationFrame(rafId);
        scene.dispose();
      };
    }

    const onResize = () => aperture.resize();
    window.addEventListener("resize", onResize);

    const trigger = ScrollTrigger.create({
      trigger: trackRef.current,
      start: "top top",
      end: "bottom bottom",
      // A short scrub lag (rather than 1:1 `true`) is what actually reads as "smooth" —
      // the scene eases toward the scroll position instead of snapping to it every tick.
      // Same value on the title/caption crossfade below so they never drift out of sync
      // with the canvas.
      scrub: 0.5,
      onUpdate: (self) => paint(self.progress),
      onRefresh: (self) => {
        scene.resize();
        paint(self.progress);
      },
    });

    // The title yields to the caption once scanning starts.
    const chrome = gsap
      .timeline({
        scrollTrigger: {
          trigger: trackRef.current,
          start: "top top",
          end: "8% top",
          scrub: 0.5,
        },
      })
      .to([titleRef.current, apertureRef.current], { opacity: 0, ease: "none" }, 0)
      .to(titleRef.current, { y: -24, ease: "none" }, 0)
      .to([captionRef.current, readoutRef.current], { opacity: 1, ease: "none" }, 0.4);

    paint(0);

    return () => {
      trigger.kill();
      chrome.scrollTrigger?.kill();
      chrome.kill();
      window.removeEventListener("resize", onResize);
      if (rafId !== null) cancelAnimationFrame(rafId);
      scene.dispose();
    };
  }, []);

  const [step, name, note] = STAGES[phase];

  return (
    <section className="recon" ref={trackRef}>
      <div className="recon__stage">
        <canvas className="recon__canvas" ref={canvasRef} />

        <canvas
          className={`recon__rest${atRest ? "" : " recon__rest--hidden"}`}
          aria-hidden="true"
          ref={apertureRef}
          onMouseEnter={() => apertureCtl.current?.setHover(true)}
          onMouseLeave={() => apertureCtl.current?.setHover(false)}
          onClick={() => apertureCtl.current?.pulse()}
        />

        <div className="recon__title" ref={titleRef}>
          <h1 className="display">
            Point a phone at a room.
            <br />
            <em>Get a robot that works in it.</em>
          </h1>
          <p className="lede">
            DragVerse turns a walk-through scan into a simulation-ready twin and trains a robot
            policy inside it. Its Unity-trained buggy policy runs on the Hexagon NPU of a
            Snapdragon AI PC with the network off.
          </p>
          <p className="scrollcue">Scroll to run the pipeline</p>
        </div>

        <figcaption className="recon__caption" ref={captionRef} aria-live="polite">
          <span className="recon__step">{step}</span>
          <span className="recon__name">{name}</span>
          <span className="recon__note">{note}</span>
        </figcaption>

        <div className="recon__plan" aria-live="polite">
          {PLAN_LOG.map((line, i) => (
            <span className="recon__plan-line" key={i} ref={(el) => (planRefs.current[i] = el)}>
              {line}
            </span>
          ))}
        </div>

        <div className="recon__readout" ref={readoutRef}>
          <div>
            <span>points</span>
            <b ref={pointsRef}>0</b>
          </div>
          <div>
            <span>objects</span>
            <b ref={objectsRef}>0</b>
          </div>
          <div>
            <span>stage</span>
            <b>{name.toLowerCase()}</b>
          </div>
        </div>
      </div>
      <p className="sr-only">
        An animation showing a scanned room becoming a labelled digital twin. Every stage it
        depicts is described in the pipeline table below.
      </p>
    </section>
  );
}
