/* The lens: a circular ordered-dithering spiral aperture — the hero's primary visual
   anchor before the pipeline starts scrolling. Pure canvas, binary pixels, one accent
   color. Density comes only from dot spacing (ordered dither against a Bayer 4x4
   matrix) — no alpha gradients, no blur, no glow. A thin bezel and aperture ring give
   the halftone a mathematically exact circular edge, the way an instrument's housing
   frames its optics. */

const BAYER4 = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5],
].map((row) => row.map((v) => (v + 0.5) / 16));

const ARMS = 6;
const CELL = 4; // dot pitch, CSS px — finer grid reads as thousands of pixels, not hundreds

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smoothstep = (e0, e1, x) => {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};

export function createAperture(canvas, color) {
  const ctx = canvas.getContext("2d");
  let W = 0;
  let H = 0;
  let cx = 0;
  let cy = 0;
  let R = 0;
  let dpr = 1;

  // Instrument-grade motion state: rotation is integrated (not derived from
  // absolute time) so hover/click speed changes don't jump the angle.
  const ROT_SPEED = 0.065; // turns/sec at 1x — ~4deg/s effective across 6 arms
  let accumRot = 0;
  let lastT = null;
  let speedMul = 1;
  let pulseEnergy = 0;

  function setHover(on) {
    speedMul = on ? 1.1 : 1;
  }

  function pulse() {
    pulseEnergy = 1;
  }

  function resize() {
    dpr = Math.min(2, window.devicePixelRatio || 1);
    W = canvas.clientWidth;
    H = canvas.clientHeight;
    if (!W || !H) return;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    cx = canvas.width / 2;
    cy = canvas.height / 2;
    R = Math.min(cx, cy) - dpr; // leave room for the bezel stroke
  }

  /** render(t): t is seconds of animation time. Static instrument, not decoration —
      slow rotation, a breathing aperture radius, and a slow drift in spiral tightness,
      everything else procedural. */
  function render(t) {
    if (!W || !H) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (lastT === null) lastT = t;
    const dt = Math.max(0, t - lastT);
    lastT = t;
    accumRot += dt * ROT_SPEED * speedMul;
    pulseEnergy *= Math.exp(-dt * 8);

    const rot = accumRot + pulseEnergy * 0.12;
    const breathe = 0.5 + 0.5 * Math.sin(t * 0.28);
    const innerN = 0.19 + 0.015 * breathe; // aperture diameter, fraction of total diameter
    const inner = R * innerN;
    const tightness = 2.5 + 0.2 * Math.sin(t * 0.06); // log-spiral winding, drifts slowly
    const cell = CELL * dpr;

    ctx.fillStyle = color;
    for (let gy = 0; gy < canvas.height; gy += cell) {
      const by = Math.floor(gy / cell) % 4;
      for (let gx = 0; gx < canvas.width; gx += cell) {
        const dx = gx + cell / 2 - cx;
        const dy = gy + cell / 2 - cy;
        const r = Math.sqrt(dx * dx + dy * dy);
        if (r > R || r < inner) continue;

        const theta = Math.atan2(dy, dx);
        const rn = Math.max(r / R, 0.02);
        // A true logarithmic spiral — the curve an iris or a galaxy actually winds
        // in, tighter near the aperture and relaxing toward the rim — rotating
        // continuously over time.
        const phase = ARMS * theta - tightness * Math.log(rn) + rot * Math.PI * 2;
        const spiral = 0.5 + 0.5 * Math.sin(phase);
        // Smooth envelope in from the aperture and out toward the rim, so the
        // boundary reads as a clean taper rather than a hard dither cutoff.
        const envelope = smoothstep(innerN, innerN + 0.09, rn) * (1 - smoothstep(0.87, 1, rn));
        // A faint floor keeps the troughs between arms from going fully blank —
        // sensor grain, not empty space.
        const density = envelope * (0.16 + 0.84 * spiral);

        const bx = Math.floor(gx / cell) % 4;
        if (density > BAYER4[by][bx]) ctx.fillRect(gx, gy, cell, cell);
      }
    }
  }

  return { resize, render, setHover, pulse };
}
