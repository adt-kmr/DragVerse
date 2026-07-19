/* The material library: one physically-based material per label, matching the mission's
   palette (warm neutral walls, natural oak floor/furniture, light-gray robot). Signal
   colors (orange/teal/indigo) are deliberately NOT PBR — they're the pipeline's state
   language (see DESIGN.md's Measured Colour Rule) and stay unlit accents layered on top,
   never something a light bounces off. */

import * as THREE from "three";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";

// Straight from dashboard/DESIGN.md — the project's own token set.
export const SIGNAL = {
  orange: 0xff5b2e,
  teal: 0x14b09a,
  indigo: 0x22307e,
  vellum: 0xeef0ea,
};

// The flat, unlit "clay" state every mesh starts in during phase 3 before its material
// resolves — see the roughness/color crossfade in recon.js's phase 3 driver.
export const CLAY_COLOR = new THREE.Color(0x9aa39c);

export function createMaterials() {
  const oak = new THREE.MeshStandardMaterial({ color: 0xb98d54, roughness: 0.55, metalness: 0 });
  // A touch darker than the page's own vellum (#eef0ea) so walls read as a distinct
  // surface instead of nearly disappearing into the background at a glance.
  const drywall = new THREE.MeshStandardMaterial({ color: 0xdcd6c4, roughness: 0.92, metalness: 0 });
  const paintedWood = new THREE.MeshStandardMaterial({ color: 0xdcd6c6, roughness: 0.65, metalness: 0 });
  const fabric = new THREE.MeshStandardMaterial({ color: 0xc7bfa9, roughness: 0.95, metalness: 0 });
  const aluminum = new THREE.MeshStandardMaterial({ color: 0x9aa0a6, roughness: 0.32, metalness: 0.9 });
  // Cooler and a step darker than the walls/drywall — the robot needs to read as a
  // distinct object against the room, not blend into it, especially while it's small
  // and in motion.
  const plastic = new THREE.MeshStandardMaterial({ color: 0x9aa1a6, roughness: 0.38, metalness: 0.08 });
  const rubber = new THREE.MeshStandardMaterial({ color: 0x1c1c1c, roughness: 0.85, metalness: 0 });
  const doorWood = new THREE.MeshStandardMaterial({ color: 0x8a5a35, roughness: 0.5, metalness: 0 });

  return { oak, drywall, paintedWood, fabric, aluminum, plastic, rubber, doorWood };
}

/** Unlit accent materials for the signal layer: scan plane, semantic recolor, path, robot
    marker, quantize grid. Never PBR-lit — see the header note.

    orangeLine/indigoLine use LineMaterial (three's "fat line" renderer — screen-space
    quads via a shader) instead of LineBasicMaterial. Plain 1px GL_LINE primitives render
    unreliably without MSAA (confirmed empirically: valid, visible-should-be path geometry
    rasterized to zero fragments) — LineMaterial sidesteps that entirely and, as a bonus,
    gives the wireframe/path actual controllable width instead of a driver-dependent
    "thin as your GPU feels like." Callers must keep `.resolution` synced on resize. */
export function createSignalMaterials() {
  // toneMapped: false keeps these vivid against the ACES-tonemapped room, which is what
  // makes them read clearly as "the active signal" rather than blending into the lit scene.
  const m = {
    orange: new THREE.MeshBasicMaterial({ color: SIGNAL.orange, transparent: true }),
    teal: new THREE.MeshBasicMaterial({ color: SIGNAL.teal, transparent: true }),
    indigo: new THREE.MeshBasicMaterial({ color: SIGNAL.indigo, transparent: true }),
    orangeLine: new LineMaterial({ color: SIGNAL.orange, transparent: true, linewidth: 2, side: THREE.DoubleSide }),
    // The bolder of the two line widths — this is the one used directly for the traced
    // path (recon.js clones it again, thinner, for each object's wireframe reveal), so
    // "the executed route" reads visually heavier than "structural wireframe".
    indigoLine: new LineMaterial({ color: SIGNAL.indigo, transparent: true, linewidth: 2, side: THREE.DoubleSide }),
    // The direct line the planner considered and rejected — dashed and muted so it never
    // competes with the solid indigo route it gets replaced by.
    ghostLine: new LineMaterial({
      color: 0x4a524d,
      transparent: true,
      linewidth: 1.5,
      dashed: true,
      dashSize: 0.07,
      gapSize: 0.05,
      side: THREE.DoubleSide,
    }),
  };
  for (const mat of Object.values(m)) mat.toneMapped = false;
  return m;
}

/** A soft radial-gradient contact shadow, procedural (no asset fetch) — grounds furniture
    and the driving robot without the SSAOPass that had to be dropped (see pipeline.js). */
export function createShadowTexture() {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, "rgba(20, 20, 24, 0.38)");
  grad.addColorStop(0.6, "rgba(20, 20, 24, 0.16)");
  grad.addColorStop(1, "rgba(20, 20, 24, 0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}
