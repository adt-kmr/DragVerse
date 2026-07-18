/* The reconstruction scene: one procedural room that runs the whole pipeline as the
   viewer scrolls. Points sweep in coloured by depth, cluster into labels, snap to a twin
   wireframe, get a route traced through them, and ship quantized.

   Pure canvas — no React, no GSAP. ReconCanvas.jsx owns the scroll wiring.
   This is illustration, not live API data. */

export const ROOM = { w: 8, d: 6, h: 2.7 };

// Label → collider pairs are the real ones from twin/rules/mapping.yaml.
const COLLIDER = {
  wall: "BoxCollider",
  floor: "BoxCollider",
  door: "MeshCollider",
  table: "BoxCollider",
  chair: "MeshCollider",
  shelf: "MeshCollider",
  cabinet: "BoxCollider",
  robot: "CapsuleCollider",
};

// Props the segmenter would cluster out of the cloud. x/y/z are the min corner, metres.
// tdx/tdy nudge each label off its box — the table group projects tightly, so the tags
// are staggered by hand rather than by a layout pass nobody would read the code for.
const PROPS = [
  { label: "table", x: 3.0, y: 2.4, z: 0, w: 1.6, d: 0.9, h: 0.75, tdx: 34, tdy: 4 },
  { label: "chair", x: 2.35, y: 2.55, z: 0, w: 0.5, d: 0.5, h: 0.9, tdx: -52, tdy: -6 },
  { label: "chair", x: 4.75, y: 2.55, z: 0, w: 0.5, d: 0.5, h: 0.9, tdx: 50, tdy: 16 },
  { label: "shelf", x: 0.3, y: 4.9, z: 0, w: 2.6, d: 0.45, h: 1.9, tdx: 0, tdy: -14 },
  { label: "cabinet", x: 6.75, y: 0.5, z: 0, w: 0.9, d: 1.2, h: 1.45, tdx: 0, tdy: -14 },
  { label: "robot", x: 1.0, y: 1.0, z: 0, w: 0.45, d: 0.45, h: 0.55, tdx: 0, tdy: -18 },
];

const PATH = [
  [1.22, 1.22],
  [1.7, 1.85],
  [2.7, 1.95],
  [3.85, 2.05],
];

const LABEL_COLOR = {
  floor: "#a2ab9e",
  wall: "#22307e",
  door: "#ff5b2e",
  table: "#14b09a",
  chair: "#14b09a",
  shelf: "#14b09a",
  cabinet: "#14b09a",
  robot: "#ff5b2e",
};

export const STAGES = [
  ["01", "Capture", "Depth frames stream in from the phone, coloured near to far."],
  ["02", "Reconstruct", "The frames fuse into one cloud and the scan noise settles out."],
  ["03", "Segment", "Clusters take labels: floor, wall, door, table, chair, shelf, cabinet."],
  ["04", "Generate twin", "Every label becomes a Unity prefab and the collider that fits it."],
  ["05", "Train", "A policy learns the route inside the twin, then has to clear the gate."],
  ["06", "Deploy", "The quantized policy drives the robot, with nothing upstream to call."],
];

/* ------------------------------------------------------------------- utilities */

/** Deterministic RNG — the cloud must look identical on every resize and reload. */
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const hexRgb = (hex) => [
  parseInt(hex.slice(1, 3), 16),
  parseInt(hex.slice(3, 5), 16),
  parseInt(hex.slice(5, 7), 16),
];

const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const smooth = (t) => t * t * (3 - 2 * t);

const RAMP = [hexRgb("#ff5b2e"), hexRgb("#14b09a"), hexRgb("#22307e")];

/** The depth colormap: near is hot, far is indigo. */
function depthRamp(d) {
  const t = clamp(d, 0, 1) * 2;
  const i = t < 1 ? 0 : 1;
  const k = t < 1 ? t : t - 1;
  return [
    lerp(RAMP[i][0], RAMP[i + 1][0], k),
    lerp(RAMP[i][1], RAMP[i + 1][1], k),
    lerp(RAMP[i][2], RAMP[i + 1][2], k),
  ];
}

/* -------------------------------------------------------------- cloud building */

const rand = mulberry32(20260718);

/** Sample points across a box's faces, skipping the bottom nobody sees. */
function sampleBox(box, label, density, out) {
  const faces = [
    [box.w, box.d, "z", box.h],
    [box.w, box.h, "y", 0],
    [box.w, box.h, "y", box.d],
    [box.d, box.h, "x", 0],
    [box.d, box.h, "x", box.w],
  ];
  for (const [u, v, axis, at] of faces) {
    const n = Math.max(6, Math.round(u * v * density));
    for (let i = 0; i < n; i++) {
      const a = rand() * u;
      const b = rand() * v;
      let p;
      if (axis === "z") p = [box.x + a, box.y + b, box.z + at];
      else if (axis === "y") p = [box.x + a, box.y + at, box.z + b];
      else p = [box.x + at, box.y + a, box.z + b];
      out.push({
        x: p[0],
        y: p[1],
        z: p[2],
        label,
        // Jitter is fixed at build time so noise settles smoothly rather than boiling.
        jx: (rand() - 0.5) * 0.09,
        jy: (rand() - 0.5) * 0.09,
        jz: (rand() - 0.5) * 0.06,
      });
    }
  }
}

function buildCloud() {
  const pts = [];
  const T = 0.12;

  const floorN = Math.round(ROOM.w * ROOM.d * 52);
  for (let i = 0; i < floorN; i++) {
    pts.push({
      x: rand() * ROOM.w,
      y: rand() * ROOM.d,
      z: 0,
      label: "floor",
      jx: (rand() - 0.5) * 0.09,
      jy: (rand() - 0.5) * 0.09,
      jz: (rand() - 0.5) * 0.05,
    });
  }

  // Walls. The front wall carries a door gap, so it is sampled in two runs.
  sampleBox({ x: 0, y: 0, z: 0, w: T, d: ROOM.d, h: ROOM.h }, "wall", 26, pts);
  sampleBox({ x: ROOM.w - T, y: 0, z: 0, w: T, d: ROOM.d, h: ROOM.h }, "wall", 26, pts);
  sampleBox({ x: 0, y: ROOM.d - T, z: 0, w: ROOM.w, d: T, h: ROOM.h }, "wall", 26, pts);
  sampleBox({ x: 0, y: 0, z: 0, w: 3.4, d: T, h: ROOM.h }, "wall", 26, pts);
  sampleBox({ x: 4.5, y: 0, z: 0, w: ROOM.w - 4.5, d: T, h: ROOM.h }, "wall", 26, pts);
  sampleBox({ x: 3.4, y: 0, z: 2.05, w: 1.1, d: T, h: 0.65 }, "door", 30, pts);

  for (const p of PROPS) sampleBox(p, p.label, 130, pts);
  return pts;
}

export const CLOUD = buildCloud();

/* ------------------------------------------------------------------ projection */

const ISO_C = Math.cos(Math.PI / 6);
const ISO_S = Math.sin(Math.PI / 6);
const isoX = (x, y) => (x - y) * ISO_C;
const isoY = (x, y, z) => (x + y) * ISO_S - z * 1.02;

/* ----------------------------------------------------------------- the renderer */

/** Wires a canvas to the scene. Returns { resize, render } — render(progress 0..1)
    draws the frame and reports what it drew, so the caption can follow it. */
export function createScene(canvas) {
  const ctx = canvas.getContext("2d");
  let W = 0;
  let H = 0;
  let scale = 1;
  let ox = 0;
  let oy = 0;
  let roomRect = { x0: 0, y0: 0, x1: 0, y1: 0 };

  const px = (x, y) => ox + isoX(x, y) * scale;
  const py = (x, y, z) => oy + isoY(x, y, z) * scale;

  function resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    W = canvas.clientWidth;
    H = canvas.clientHeight;
    if (!W || !H) return;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const xs = [isoX(0, 0), isoX(ROOM.w, 0), isoX(0, ROOM.d), isoX(ROOM.w, ROOM.d)];
    const ys = [
      isoY(0, 0, ROOM.h),
      isoY(ROOM.w, 0, ROOM.h),
      isoY(ROOM.w, ROOM.d, 0),
      isoY(0, ROOM.d, 0),
    ];
    const spanX = Math.max(...xs) - Math.min(...xs);
    const spanY = Math.max(...ys) - Math.min(...ys);
    const narrow = W < 900;
    scale = Math.min((W * (narrow ? 0.92 : 0.72)) / spanX, (H * (narrow ? 0.5 : 0.68)) / spanY);
    ox = W * (narrow ? 0.5 : 0.6) - ((Math.max(...xs) + Math.min(...xs)) / 2) * scale;
    oy = H * (narrow ? 0.38 : 0.44) - ((Math.max(...ys) + Math.min(...ys)) / 2) * scale;

    const pad = 18;
    roomRect = {
      x0: ox + Math.min(...xs) * scale - pad,
      y0: oy + Math.min(...ys) * scale - pad,
      x1: ox + Math.max(...xs) * scale + pad,
      y1: oy + Math.max(...ys) * scale + pad,
    };
  }

  function drawGrid(alpha) {
    ctx.strokeStyle = `rgba(203, 209, 199, ${alpha})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x <= ROOM.w; x++) {
      ctx.moveTo(px(x, 0), py(x, 0, 0));
      ctx.lineTo(px(x, ROOM.d), py(x, ROOM.d, 0));
    }
    for (let y = 0; y <= ROOM.d; y++) {
      ctx.moveTo(px(0, y), py(0, y, 0));
      ctx.lineTo(px(ROOM.w, y), py(ROOM.w, y, 0));
    }
    ctx.stroke();
  }

  /** Wireframe box with a stroke-in reveal driven by `t`. */
  function drawBox(b, t, color) {
    const c = [
      [b.x, b.y, b.z],
      [b.x + b.w, b.y, b.z],
      [b.x + b.w, b.y + b.d, b.z],
      [b.x, b.y + b.d, b.z],
      [b.x, b.y, b.z + b.h],
      [b.x + b.w, b.y, b.z + b.h],
      [b.x + b.w, b.y + b.d, b.z + b.h],
      [b.x, b.y + b.d, b.z + b.h],
    ].map(([x, y, z]) => [px(x, y), py(x, y, z)]);

    const edges = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.25;
    const shown = Math.round(edges.length * clamp(t, 0, 1));
    ctx.beginPath();
    for (let i = 0; i < shown; i++) {
      const [a, z] = edges[i];
      ctx.moveTo(c[a][0], c[a][1]);
      ctx.lineTo(c[z][0], c[z][1]);
    }
    ctx.stroke();
  }

  function drawTag(prop, alpha) {
    if (alpha <= 0.01) return;
    const cx = prop.x + prop.w / 2;
    const cy = prop.y + prop.d / 2;
    const ax = px(cx, cy);
    const ay = py(cx, cy, prop.z + prop.h);
    const x = ax + prop.tdx;
    const y = ay - 24 + prop.tdy;

    ctx.globalAlpha = alpha;
    // Leader from the box top, so a nudged label still points at its object.
    ctx.strokeStyle = "rgba(93, 101, 96, 0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(ax, ay - 2);
    ctx.lineTo(x, y + 5);
    ctx.stroke();

    ctx.textAlign = "center";
    ctx.font = '500 11px "JetBrains Mono", ui-monospace, monospace';
    ctx.fillStyle = "#12151a";
    ctx.fillText(prop.label, x, y);
    ctx.font = '400 9.5px "JetBrains Mono", ui-monospace, monospace';
    ctx.fillStyle = "#5d6560";
    ctx.fillText(COLLIDER[prop.label], x, y + 12);
    ctx.globalAlpha = 1;
  }

  function drawPath(t) {
    if (t <= 0) return;
    const pts = PATH.map(([x, y]) => [px(x, y), py(x, y, 0.02)]);
    ctx.strokeStyle = "rgba(34, 48, 126, 0.85)";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    const span = (pts.length - 1) * clamp(t, 0, 1);
    for (let i = 1; i < pts.length; i++) {
      const seg = clamp(span - (i - 1), 0, 1);
      if (seg <= 0) break;
      ctx.lineTo(lerp(pts[i - 1][0], pts[i][0], seg), lerp(pts[i - 1][1], pts[i][1], seg));
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawRobot(t) {
    const span = (PATH.length - 1) * clamp(t, 0, 1);
    const i = Math.min(PATH.length - 2, Math.floor(span));
    const k = span - i;
    const wx = lerp(PATH[i][0], PATH[i + 1][0], k);
    const wy = lerp(PATH[i][1], PATH[i + 1][1], k);
    const x = px(wx, wy);
    const y = py(wx, wy, 0.28);
    ctx.fillStyle = "#ff5b2e";
    ctx.beginPath();
    ctx.arc(x, y, 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 91, 46, 0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(x, y, 11 + Math.sin(span * 6) * 1.5, 0, Math.PI * 2);
    ctx.stroke();
  }

  /** The scan plane — the one place the hot end of the ramp is spent. */
  function drawScanPlane(sx) {
    const x = ROOM.w * sx;
    ctx.strokeStyle = "#ff5b2e";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px(x, 0), py(x, 0, ROOM.h));
    ctx.lineTo(px(x, 0), py(x, 0, 0));
    ctx.lineTo(px(x, ROOM.d), py(x, ROOM.d, 0));
    ctx.lineTo(px(x, ROOM.d), py(x, ROOM.d, ROOM.h));
    ctx.stroke();
  }

  /** int8 blocks settling over the twin, clipped to the room so it reads as the model
      being quantized rather than a grid dropped over the page. */
  function drawQuantize(t) {
    if (t <= 0.01) return;
    const step = 24;
    const { x0, y0, x1, y1 } = roomRect;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, y0, x1 - x0, y1 - y0);
    ctx.clip();
    ctx.globalAlpha = 0.22 * t;
    ctx.strokeStyle = "rgba(34, 48, 126, 0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = x0; x <= x1; x += step) {
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y1);
    }
    for (let y = y0; y <= y1; y += step) {
      ctx.moveTo(x0, y);
      ctx.lineTo(x1, y);
    }
    ctx.stroke();
    ctx.restore();

    ctx.globalAlpha = t;
    ctx.font = '500 10px "JetBrains Mono", ui-monospace, monospace';
    ctx.textAlign = "left";
    ctx.fillStyle = "#22307e";
    ctx.fillText("int8 · on-device", x0, y0 - 10);
    ctx.globalAlpha = 1;
  }

  function render(progress) {
    if (!W || !H) return { phase: 0, points: 0, objects: 0 };
    ctx.clearRect(0, 0, W, H);

    // Hold on the title, then run six phases across the rest of the track.
    const run = clamp((progress - 0.1) / 0.86, 0, 1);
    const f = run * 6;
    const phase = Math.min(5, Math.floor(f));
    const t = smooth(clamp(f - phase, 0, 1));

    // The empty room is present from the first frame — the space exists before it is scanned.
    drawGrid(phase >= 3 ? 1 : 0.8);

    const sweep = phase === 0 ? t : 1;
    const jitter = phase === 0 ? 1 : phase === 1 ? 1 - t : 0;
    const labelMix = phase === 2 ? t : phase > 2 ? 1 : 0;
    const fade = phase === 3 ? lerp(1, 0.22, t) : phase > 3 ? 0.22 : 1;
    const size = phase >= 1 ? 1.6 : 2;

    let visible = 0;
    for (const p of CLOUD) {
      const reveal = clamp((sweep - p.x / ROOM.w) * 7, 0, 1);
      if (reveal <= 0.01) continue;
      visible++;

      const x = p.x + p.jx * jitter;
      const y = p.y + p.jy * jitter;
      const z = p.z + p.jz * jitter;
      const depth = (p.x + p.y) / (ROOM.w + ROOM.d);

      let c = depthRamp(depth);
      if (labelMix > 0) {
        const l = hexRgb(LABEL_COLOR[p.label]);
        c = [
          lerp(c[0], l[0], labelMix),
          lerp(c[1], l[1], labelMix),
          lerp(c[2], l[2], labelMix),
        ];
      }

      ctx.fillStyle = `rgba(${c[0] | 0}, ${c[1] | 0}, ${c[2] | 0}, ${reveal * fade})`;
      ctx.fillRect(px(x, y) - size / 2, py(x, y, z) - size / 2, size, size);
    }

    if (phase === 0 && t > 0.01 && t < 0.99) drawScanPlane(t);

    if (phase >= 3) {
      const build = phase === 3 ? t : 1;
      PROPS.forEach((prop, i) => {
        const local = clamp(build * PROPS.length - i, 0, 1);
        drawBox(prop, local, prop.label === "robot" ? "#ff5b2e" : "#22307e");
        drawTag(prop, phase === 3 ? clamp(local * 1.5 - 0.4, 0, 1) : 1);
      });
    }

    if (phase === 4) drawPath(clamp((t - 0.15) / 0.85, 0, 1));
    if (phase === 5) {
      drawPath(1);
      drawQuantize(clamp((t - 0.25) / 0.75, 0, 1));
      drawRobot(t);
    }

    return { phase, points: visible, objects: phase >= 2 ? PROPS.length : 0 };
  }

  return { resize, render };
}
