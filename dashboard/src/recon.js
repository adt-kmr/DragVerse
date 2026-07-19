/* The reconstruction scene: one procedural room that runs the whole pipeline as the
   viewer scrolls. Points sweep in coloured by depth, cluster into labels, snap to a twin
   with real geometry and PBR materials, get a route traced through them, and ship
   quantized.

   This file owns the DATA (ROOM/PROPS/PATH/STAGES/CLOUD) and the createScene() contract —
   {resize, render(progress) -> {phase, points, objects, rest}} — exactly as before. The
   internals of createScene() are now a Three.js scene (see recon/*.js) instead of 2D
   canvas draws; ReconCanvas.jsx, which owns all scroll/GSAP wiring, needed no changes to
   keep working against this same contract. This is illustration, not live API data. */

import * as THREE from "three";

import { createMaterials, createSignalMaterials, createShadowTexture, CLAY_COLOR } from "./recon/materials.js";
import { buildProp, buildRoomShell } from "./recon/geometry.js";
import { createLighting } from "./recon/lighting.js";
import { createPointCloud } from "./recon/pointcloud.js";
import { createPipeline } from "./recon/pipeline.js";
import { createLabelLayer } from "./recon/labels.js";
import { planRoute } from "./recon/planner.js";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";

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

// The robot's dock (matches the "robot" PROP's own bbox center below) and a delivery
// point on the far side of the table+chair pair — chosen specifically so the straight
// line between them cuts through both, giving the planner something real to route
// around instead of an illustrative bend nobody computed.
const ROBOT_START = [1.225, 1.225];
const DELIVERY_GOAL = [6.51, 3.43];

// Computed once, at module load, against the same PROPS the room is built from — not
// hand-picked waypoints. See recon/planner.js for the visibility-graph search.
const { route: PATH, direct: DIRECT_PATH, blocked: BLOCKED } = planRoute(ROBOT_START, DELIVERY_GOAL, PROPS);

// The on-canvas decision log for the Train stage: what the planner sensed and did,
// nearest obstacle first, mono-aligned so it reads as instrument output.
const LOG_LABEL_WIDTH = 11;
export const PLAN_LOG = [
  `${"SENSING".padEnd(LOG_LABEL_WIDTH)}…`,
  ...BLOCKED.map(({ prop, dist }) => `${"OBSTACLE".padEnd(LOG_LABEL_WIDTH)}${prop.label} · ${dist.toFixed(1)}m`),
  `${"REPLANNING".padEnd(LOG_LABEL_WIDTH)}…`,
  `${"ROUTE".padEnd(LOG_LABEL_WIDTH)}clear`,
];

// Phase 4's ("Train") timeline, t in 0..1: the direct line appears, sensing cycles
// through BLOCKED nearest-first, the direct line is rejected, and the real route draws
// in over what's left. Phase 5 ("Deploy") is the drive along that same route.
const GHOST_IN_END = 0.15;
const SENSE_START = 0.2;
const SENSE_END = 0.56;
const REPLAN_AT = 0.58;
const GHOST_OUT_END = 0.64;
const DRAW_START = 0.64;

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

const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const smooth = (t) => t * t * (3 - 2 * t);

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

  const T = 0.12;
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

/* ----------------------------------------------------------------- the renderer */

/** Wires a canvas to the scene. Returns { resize, render, dispose } — render(progress
    0..1) draws the frame and reports what it drew, so the caption can follow it. */
export function createScene(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  // Transparent — the page's vellum shows through directly, so the canvas background
  // matches it exactly rather than an opaque clear color that ACES tonemapping would
  // otherwise shift slightly off the true vellum value. (SSAO/bloom didn't preserve alpha
  // correctly and caused a wash-out with this on; both are gone now — see pipeline.js.)
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  // No scene.fog: it fogs by real view-space camera distance, which for this orthographic
  // setup is the arbitrary ~20-unit camera offset (irrelevant to framing, since ortho
  // projection ignores distance for scale) — not the room's actual size, so any near/far
  // choice either does nothing or washes the whole room out. The point cloud gets its own
  // atmospheric fade directly in its shader (recon/pointcloud.js); solids don't need one at
  // this room's scale (always fully in frame, ~8m across).

  // Orthographic isometric camera: position along the true (1,1,1) isometric axis with
  // up=(0,0,-1) — this specific up vector (not the more obvious (0,0,1)) is what makes
  // Three's lookAt basis match the room's axis convention (x/y floor, z height) without a
  // mirror flip. The frustum itself is fit to the viewport in resize(), replicating the
  // old hand-rolled projection's span-fit logic exactly, just measured from the real
  // camera instead of a duplicated formula.
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 60);
  const center = new THREE.Vector3(ROOM.w / 2, ROOM.d / 2, ROOM.h / 2);
  const dir = new THREE.Vector3(1, 1, 1).normalize();
  camera.position.copy(center).addScaledVector(dir, 20);
  camera.up.set(0, 0, -1);
  camera.lookAt(center);
  camera.updateMatrixWorld(true);

  // Everything the pipeline builds lives in one group, scaled 2% taller on Z — the old
  // projection's isoY used a z*1.02 coefficient (a deliberate exaggeration, not true
  // isometric) instead of the mathematically "correct" 1.0. Baking it into the content
  // group reproduces that exact framing with real geometry.
  const content = new THREE.Group();
  content.scale.set(1, 1, 1.02);
  scene.add(content);

  const mats = createMaterials();
  const signal = createSignalMaterials();
  const lighting = createLighting(renderer, ROOM);
  lighting.addTo(scene);

  const roomShell = buildRoomShell(ROOM, mats, signal.indigoLine);
  content.add(roomShell.group);
  for (const m of roomShell.solids) m.material.transparent = true;
  roomShell.edgeMaterial.opacity = 0;
  for (const m of roomShell.solids) m.material.opacity = 0;

  // Soft contact-shadow decal under every object — grounds them without the SSAOPass that
  // had to be dropped (see pipeline.js). Fades in with the object's own build (phase 3).
  const shadowTexture = createShadowTexture();
  function makeShadow(cx, cy, rx, ry) {
    const material = new THREE.MeshBasicMaterial({ map: shadowTexture, transparent: true, depthWrite: false, opacity: 0 });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(rx * 2, ry * 2), material);
    mesh.position.set(cx, cy, 0.004);
    return mesh;
  }

  const objects = PROPS.map((prop) => {
    const built = buildProp(prop, mats, signal.indigoLine, signal.orange.clone());
    content.add(built.group);
    for (const m of built.solids) {
      m.material.transparent = true;
      m.material.opacity = 0;
      m.material.color.copy(CLAY_COLOR);
      m.material.roughness = 1;
      m.material.metalness = 0;
      m.castShadow = false;
      m.receiveShadow = false;
    }
    built.edgeMaterial.opacity = 0;
    if (built.light) built.light.material.opacity = 0;
    const shadow = makeShadow(prop.x + prop.w / 2, prop.y + prop.d / 2, prop.w * 0.85, prop.d * 0.85);
    content.add(shadow);
    return { prop, ...built, shadow };
  });

  // Fat-line (LineMaterial) instances need `.resolution` kept in sync with the canvas'
  // pixel size — collected here so resize() can update all of them in one place.
  const lineMaterials = [
    signal.orangeLine,
    signal.indigoLine,
    signal.ghostLine,
    roomShell.edgeMaterial,
    ...objects.map((o) => o.edgeMaterial),
  ];

  const cloud = createPointCloud(CLOUD, ROOM, LABEL_COLOR, mulberry32(9));
  content.add(cloud.points);

  // Scan plane — phase 0's sweeping capture front. Cloned materials throughout this
  // section so each signal-accent mesh's opacity is independent of its siblings'.
  const scanPlane = new THREE.Mesh(new THREE.PlaneGeometry(ROOM.d, ROOM.h), signal.orange.clone());
  scanPlane.rotation.y = Math.PI / 2;
  scanPlane.material.opacity = 0.5;
  scanPlane.visible = false;
  content.add(scanPlane);

  // Traced path — phase 4. Line2/LineGeometry (fat line) rather than plain THREE.Line —
  // see the note on edgesFor() in recon/geometry.js for why.
  const pathGeo = new LineGeometry();
  const pathLine = new Line2(pathGeo, signal.indigoLine);
  pathLine.visible = false;
  pathLine.frustumCulled = false;
  content.add(pathLine);

  function updatePathVertices(tFrac) {
    const span = (PATH.length - 1) * clamp(tFrac, 0, 1);
    const verts = [PATH[0]];
    for (let i = 1; i < PATH.length; i++) {
      const seg = clamp(span - (i - 1), 0, 1);
      if (seg <= 0) break;
      verts.push([lerp(PATH[i - 1][0], PATH[i][0], seg), lerp(PATH[i - 1][1], PATH[i][1], seg)]);
    }
    if (verts.length < 2) {
      pathLine.visible = false;
      return;
    }
    const flat = [];
    for (const [x, y] of verts) flat.push(x, y, 0.02);
    pathGeo.setPositions(flat);
    pathLine.computeLineDistances();
  }

  // The direct line the planner considered first and rejected — drawn once (it never
  // grows), just faded in, held, then faded out as the real route takes over. Sits a
  // hair below the real path in z so the two never z-fight during the crossfade.
  const ghostGeo = new LineGeometry();
  ghostGeo.setPositions([DIRECT_PATH[0][0], DIRECT_PATH[0][1], 0.019, DIRECT_PATH[1][0], DIRECT_PATH[1][1], 0.019]);
  const ghostLine = new Line2(ghostGeo, signal.ghostLine);
  ghostLine.computeLineDistances();
  ghostLine.visible = false;
  ghostLine.frustumCulled = false;
  content.add(ghostLine);

  // Sensing ray + ring: one shared pair, repositioned onto whichever blocked obstacle is
  // "active" this instant (see sensePhase()) rather than one mesh per obstacle — the
  // choreography only ever highlights one at a time, so there's nothing to gain from more.
  const senseRayGeo = new LineGeometry();
  senseRayGeo.setPositions([0, 0, 0.021, 0, 0, 0.021]);
  const senseRay = new Line2(senseRayGeo, signal.orangeLine);
  senseRay.visible = false;
  senseRay.frustumCulled = false;
  content.add(senseRay);

  // A lock-on reticle, built as a fat-line loop (same Line2 technique as the traced path)
  // rather than a filled ring mesh — floats above whichever obstacle is "active", clear
  // of every object's own silhouette so nothing in the room can occlude it.
  const RING_SEGMENTS = 40;
  const senseRingGeo = new LineGeometry();
  {
    const unit = [];
    for (let i = 0; i <= RING_SEGMENTS; i++) {
      const a = (i / RING_SEGMENTS) * Math.PI * 2;
      unit.push(Math.cos(a), Math.sin(a), 0);
    }
    senseRingGeo.setPositions(unit);
  }
  const senseRing = new Line2(senseRingGeo, signal.orangeLine);
  senseRing.computeLineDistances();
  senseRing.visible = false;
  senseRing.frustumCulled = false;
  content.add(senseRing);

  /** Which blocked obstacle (if any) is being "sensed" at this instant within phase 4,
      and how visible its ring/ray should be — fades in, holds, fades out, one obstacle
      at a time, in nearest-first order. */
  function sensePhase(t) {
    const n = BLOCKED.length;
    if (!n || t < SENSE_START || t > SENSE_END) return null;
    const slot = (SENSE_END - SENSE_START) / n;
    const index = Math.min(n - 1, Math.floor((t - SENSE_START) / slot));
    const within = clamp((t - SENSE_START - index * slot) / slot, 0, 1);
    const alpha = within < 0.25 ? smooth(within / 0.25) : within > 0.75 ? smooth((1 - within) / 0.25) : 1;
    return { index, alpha };
  }

  /** Reveal alphas for PLAN_LOG's lines: SENSING, one per blocked obstacle in order,
      REPLANNING, ROUTE CLEAR. Each line reveals once and holds — an accumulating log,
      not a status line that overwrites itself. */
  function planLogAlphas(t) {
    const n = BLOCKED.length;
    const a = new Array(PLAN_LOG.length).fill(0);
    a[0] = smooth(clamp((t - 0.1) / 0.08, 0, 1));
    const slot = (SENSE_END - SENSE_START) / Math.max(n, 1);
    for (let i = 0; i < n; i++) {
      const revealAt = SENSE_START + i * slot + slot * 0.15;
      a[1 + i] = smooth(clamp((t - revealAt) / 0.05, 0, 1));
    }
    a[n + 1] = smooth(clamp((t - REPLAN_AT) / 0.05, 0, 1));
    a[n + 2] = smooth(clamp((t - 0.92) / 0.06, 0, 1));
    return a;
  }

  // Phase 5: the robot prop itself drives the route — no separate abstract marker. It
  // spends phases 3-4 parked at its dock (prop.x/y/z, built like every other object);
  // driveRobot() repositions/reorients the same mesh, spins its wheels, and hands the
  // label layer a live anchor so the "robot" tag tracks it instead of staying pinned to
  // the empty dock.
  const robotObj = objects.find((o) => o.prop.label === "robot");
  const robotDock = new THREE.Vector3(robotObj.prop.x, robotObj.prop.y, robotObj.prop.z);
  const robotTagAnchor = new THREE.Vector3();
  const robotIndex = objects.indexOf(robotObj);

  function driveRobot(tFrac) {
    const span = (PATH.length - 1) * clamp(tFrac, 0, 1);
    const i = Math.min(PATH.length - 2, Math.floor(span));
    const k = span - i;
    const wx = lerp(PATH[i][0], PATH[i + 1][0], k);
    const wy = lerp(PATH[i][1], PATH[i + 1][1], k);
    const heading = Math.atan2(PATH[i + 1][1] - PATH[i][1], PATH[i + 1][0] - PATH[i][0]);

    // The group's local origin is the bbox's min corner (every object is built that way —
    // see geometry.js), not its visual center — so rotating and scaling the group directly
    // swings/drifts that corner's *center* away from (wx, wy) by a growing amount as
    // heading and scale move away from 0/1. Compensating the position by the rotated,
    // scaled center offset keeps the robot's actual visual center pinned to the path
    // point, exactly like every other object's center is pinned to its own bbox.
    const cx = robotObj.prop.w / 2;
    const cy = robotObj.prop.d / 2;
    // Eases up to 45% larger over the first sliver of the drive — deliberately bold,
    // because the robot's actual footprint (0.45m in an 8x6m room) is small enough that a
    // subtle emphasis doesn't read at all from the isometric distance this scene is
    // always viewed at. It's the only thing in the room that ever moves or scales, so a
    // clear size jump reads as "the active element", not as a glitch.
    const s = 1 + 0.45 * Math.min(tFrac * 5, 1);
    const cosH = Math.cos(heading);
    const sinH = Math.sin(heading);
    const offX = s * (cx * cosH - cy * sinH);
    const offY = s * (cx * sinH + cy * cosH);
    robotObj.group.position.set(wx - offX, wy - offY, robotObj.prop.z);
    robotObj.group.rotation.z = heading;
    robotObj.group.scale.setScalar(s);

    const spin = span * 8;
    for (let wi = 2; wi < robotObj.solids.length; wi++) robotObj.solids[wi].rotation.set(0, spin, Math.PI / 2);

    robotObj.shadow.position.set(wx, wy, 0.004);

    // Anchored to the beacon's height, not the body's — the beacon is the part that's
    // actually always visible (see the note in geometry.js's buildRobot), so that's what
    // the tag's leader line should point at.
    robotTagAnchor.set(wx, wy, robotObj.prop.h * 4);
    content.localToWorld(robotTagAnchor);
    labels.setAnchorOverride(robotIndex, robotTagAnchor);
  }

  function parkRobot() {
    robotObj.group.position.copy(robotDock);
    robotObj.group.rotation.z = 0;
    robotObj.group.scale.setScalar(1);
    robotObj.shadow.position.set(robotObj.prop.x + robotObj.prop.w / 2, robotObj.prop.y + robotObj.prop.d / 2, 0.004);
    labels.setAnchorOverride(robotIndex, null);
  }

  // Quantize grid — phase 5, int8 blocks settling over the twin.
  const cellSize = 0.32;
  const cols = Math.max(1, Math.round(ROOM.w / cellSize));
  const rows = Math.max(1, Math.round(ROOM.d / cellSize));
  const quantMat = signal.indigo.clone();
  quantMat.opacity = 0;
  const quantize = new THREE.InstancedMesh(
    new THREE.BoxGeometry(cellSize * 0.86, 0.01, cellSize * 0.86),
    quantMat,
    cols * rows
  );
  {
    const m4 = new THREE.Matrix4();
    let qi = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        m4.makeTranslation((c + 0.5) * cellSize, (r + 0.5) * cellSize, 0.015);
        quantize.setMatrixAt(qi++, m4);
      }
    }
  }
  quantize.visible = false;
  content.add(quantize);
  const quantizeAnchor = new THREE.Vector3(0, 0, 0);

  const pipeline = createPipeline(renderer, scene, camera);
  const labels = createLabelLayer(canvas, PROPS, COLLIDER, content);

  // Sorted reveal thresholds for an O(log n) visible-point-count readout, matching the
  // shader's own reveal test instead of walking every point on the CPU each frame.
  const revealSorted = CLOUD.map((p) => p.x / ROOM.w).sort((a, b) => a - b);
  function countVisible(sweep) {
    const threshold = sweep - 0.01 / 7;
    let lo = 0;
    let hi = revealSorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (revealSorted[mid] < threshold) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  let W = 0;
  let H = 0;
  let dpr = 1;

  function resize() {
    dpr = Math.min(2, window.devicePixelRatio || 1);
    W = canvas.clientWidth;
    H = canvas.clientHeight;
    if (!W || !H) return;

    renderer.setPixelRatio(dpr);
    renderer.setSize(W, H, false);

    for (const m of lineMaterials) m.resolution.set(W, H);

    // Measure the room's true camera-space extent, then fit it to the viewport with the
    // exact same span/centering logic (and breakpoint) the old resize() used — just
    // sourced from the real camera instead of a hand-duplicated projection formula.
    const view = camera.matrixWorld.clone().invert();
    const corners = [];
    for (const cx of [0, ROOM.w]) {
      for (const cy of [0, ROOM.d]) {
        for (const cz of [0, ROOM.h * 1.02]) {
          corners.push(new THREE.Vector3(cx, cy, cz).applyMatrix4(view));
        }
      }
    }
    const xs = corners.map((c) => c.x);
    const ys = corners.map((c) => c.y);
    const spanX = Math.max(...xs) - Math.min(...xs);
    const spanY = Math.max(...ys) - Math.min(...ys);
    const narrow = W < 900;
    const scale = Math.min((W * (narrow ? 0.92 : 0.72)) / spanX, (H * (narrow ? 0.5 : 0.68)) / spanY);
    const ox = W * (narrow ? 0.5 : 0.6) - ((Math.max(...xs) + Math.min(...xs)) / 2) * scale;
    const oy = H * (narrow ? 0.38 : 0.44) - ((Math.max(...ys) + Math.min(...ys)) / 2) * scale;

    camera.left = -ox / scale;
    camera.right = (W - ox) / scale;
    camera.top = -oy / scale;
    camera.bottom = (H - oy) / scale;
    camera.near = 0.1;
    camera.far = 60;
    camera.updateProjectionMatrix();
  }

  function render(progress) {
    if (!W || !H) return { phase: 0, points: 0, objects: 0, rest: true, plan: new Array(PLAN_LOG.length).fill(0) };

    const rest = progress <= 0.1;
    const run = clamp((progress - 0.1) / 0.86, 0, 1);
    const f = run * 6;
    const phase = Math.min(5, Math.floor(f));
    const t = smooth(clamp(f - phase, 0, 1));

    // Micro camera settle: each phase eases in from a hair of extra zoom instead of
    // snapping straight to rest — the "camera smoothing" the mission calls for, kept
    // small enough it reads as focus settling, never as the camera actually reframing.
    // Position/angle/frustum (set in resize()) are untouched — this is the one place a
    // camera property changes per frame, and it's ±1%.
    camera.zoom = 1 - 0.01 * (1 - t);
    camera.updateProjectionMatrix();

    const sweep = phase === 0 ? t : 1;
    const jitterMix = phase === 0 ? 1 : phase === 1 ? 1 - t : 0;
    const labelMix = phase === 2 ? t : phase > 2 ? 1 : 0;
    const fade = phase === 3 ? lerp(1, 0.22, t) : phase > 3 ? 0.22 : 1;
    const sizePx = (phase >= 1 ? 1.6 : 2) * dpr;
    cloud.setUniforms({ sweep, jitterMix, labelMix, fade, sizePx });

    // Room shell fades in alongside segmentation, matching the old floor grid's timing.
    // ponytail: a flat opacity ramp rather than the props' full wireframe->fill->material
    // sequence — the shell is the background the objects sit in, not a staged "object"
    // with its own tag; upgrade path is giving it the same per-object treatment as PROPS
    // if the room shell ever needs its own reveal beat.
    const shellAlpha = phase < 2 ? 0 : phase === 2 ? t * 0.8 : phase === 3 ? lerp(0.8, 1, t) : 1;
    const shellSettled = phase >= 3;
    for (const m of roomShell.solids) {
      m.material.opacity = shellAlpha;
      // Opaque once fully resolved: dozens of meshes staying `transparent` forever means
      // Three sorts them back-to-front by distance instead of depth-testing them properly,
      // which was silently hiding the thin path/marker signal accents behind "opaque-
      // looking" (opacity 1) walls that happened to sort afterward. Flipping back to
      // opaque once the fade-in is done fixes that and is cheaper to render besides.
      m.material.transparent = phase <= 3;
      m.castShadow = shellSettled;
      m.receiveShadow = shellSettled;
    }
    roomShell.edgeMaterial.opacity = shellAlpha * 0.6;

    scanPlane.visible = phase === 0 && t > 0.01 && t < 0.99;
    if (scanPlane.visible) scanPlane.position.set(ROOM.w * t, ROOM.d / 2, ROOM.h / 2);

    // Phase 3: points converge -> wireframe grows -> surface fills -> material resolves
    // -> lighting settles, staggered per object exactly as the original per-box reveal.
    // ponytail: the wireframe reveal fades the edge set's opacity in as one unit, rather
    // than growing edge-by-edge like the old 12-edge box did — with real furniture meshes
    // there can be 60+ edges per object, and matching that reveal exactly would need
    // consistent per-edge ordering across composite meshes for a subtle flourish nobody
    // would consciously notice; upgrade path is per-edge growth via setDrawRange if this
    // stage ever needs to read as more mechanical.
    const tagAlphas = new Array(objects.length).fill(0);
    if (phase >= 3) {
      const build = phase === 3 ? t : 1;
      objects.forEach((o, i) => {
        const local = clamp(build * objects.length - i, 0, 1);
        // smooth() on every sub-stage, not just the material crossfade — same three
        // windows (0-0.35 / 0.35-0.65 / 0.65-0.85), just eased ramps instead of linear
        // ones so the wireframe-grows -> surface-fills handoff doesn't read as a snap.
        const edgeT = smooth(clamp(local / 0.35, 0, 1));
        const fillT = smooth(clamp((local - 0.35) / 0.3, 0, 1));
        const materialT = smooth(clamp((local - 0.65) / 0.2, 0, 1));
        const settled = local >= 0.85;

        // Edges stay visible longer as the fill comes in (0.45 instead of the previous
        // 0.7 falloff) — the wireframe reveal should read as a distinct beat, not a flash.
        o.edgeMaterial.opacity = edgeT * (1 - fillT * 0.45);
        for (const m of o.solids) {
          m.material.opacity = fillT;
          m.material.transparent = phase === 3; // see the room-shell note above
          m.material.color.copy(CLAY_COLOR).lerp(m.userData.target.color, materialT);
          m.material.roughness = lerp(1, m.userData.target.roughness, materialT);
          m.material.metalness = lerp(0, m.userData.target.metalness, materialT);
          m.castShadow = settled;
          m.receiveShadow = settled;
        }
        if (o.light) o.light.material.opacity = fillT;
        o.shadow.material.opacity = fillT * 0.85;
        tagAlphas[i] = phase === 3 ? smooth(clamp(local * 1.5 - 0.4, 0, 1)) : 1;
      });
    } else {
      objects.forEach((o) => {
        o.edgeMaterial.opacity = 0;
        o.shadow.material.opacity = 0;
        if (o.light) o.light.material.opacity = 0;
        for (const m of o.solids) m.material.opacity = 0;
      });
    }

    let planAlphas;
    if (phase === 4) {
      // The planner's own beat: try the direct line, sense what's in the way one
      // obstacle at a time, reject the direct line, then draw the route it actually
      // computed. Skips straight to drawing if the scene ever has no BLOCKED obstacles.
      if (BLOCKED.length) {
        const ghostAlpha =
          t < GHOST_IN_END
            ? smooth(t / GHOST_IN_END)
            : t < REPLAN_AT
              ? 1
              : t < GHOST_OUT_END
                ? 1 - smooth((t - REPLAN_AT) / (GHOST_OUT_END - REPLAN_AT))
                : 0;
        ghostLine.visible = ghostAlpha > 0.01;
        signal.ghostLine.opacity = ghostAlpha * 0.85;

        const sense = sensePhase(t);
        if (sense) {
          const { prop } = BLOCKED[sense.index];
          const cx = prop.x + prop.w / 2;
          const cy = prop.y + prop.d / 2;
          // A floor-level ring sitting flush against the object's own footprint reads as
          // a shadow decal and is routinely occluded by the object's own body from this
          // isometric angle. Floating a small lock-on reticle above the obstacle instead
          // — clear of every object's silhouette, unmistakably a sensor reading rather
          // than ground decoration.
          const cz = prop.z + prop.h + 0.4;
          const r = 0.26;

          // senseRing and senseRay share signal.orangeLine, so one opacity write covers
          // both — they only ever appear together.
          signal.orangeLine.opacity = sense.alpha * 0.85;

          senseRing.visible = true;
          senseRing.position.set(cx, cy, cz);
          senseRing.scale.setScalar(r);

          senseRay.visible = true;
          senseRayGeo.setPositions([ROBOT_START[0], ROBOT_START[1], robotObj.prop.h + 0.1, cx, cy, cz]);
          senseRay.computeLineDistances();
        } else {
          senseRing.visible = false;
          senseRay.visible = false;
        }

        if (t >= DRAW_START) {
          pathLine.visible = true;
          updatePathVertices(clamp((t - DRAW_START) / (1 - DRAW_START), 0, 1));
        } else {
          pathLine.visible = false;
        }
      } else {
        ghostLine.visible = false;
        senseRing.visible = false;
        senseRay.visible = false;
        pathLine.visible = true;
        updatePathVertices(clamp((t - 0.15) / 0.85, 0, 1));
      }
      planAlphas = planLogAlphas(t);
    } else if (phase === 5) {
      pathLine.visible = true;
      updatePathVertices(1);
      ghostLine.visible = false;
      senseRing.visible = false;
      senseRay.visible = false;
      // The log stays put through the drive — it's the record of the decision the robot
      // is now executing, not a phase-4-only status line.
      planAlphas = new Array(PLAN_LOG.length).fill(1);
    } else {
      pathLine.visible = false;
      ghostLine.visible = false;
      senseRing.visible = false;
      senseRay.visible = false;
      planAlphas = new Array(PLAN_LOG.length).fill(0);
    }

    if (phase === 5) {
      const qt = clamp((t - 0.25) / 0.75, 0, 1);
      quantize.visible = qt > 0.01;
      quantMat.opacity = 0.22 * qt;
      quantizeAnchor.set(0, 0, ROOM.h * 1.02);
      content.localToWorld(quantizeAnchor);
      labels.renderQuantizeCaption(camera, quantizeAnchor, qt);
      driveRobot(t);
    } else {
      quantize.visible = false;
      labels.renderQuantizeCaption(camera, quantizeAnchor, 0);
      parkRobot();
    }

    labels.setAlpha(tagAlphas);
    labels.render(camera);

    pipeline.render();

    return { phase, points: countVisible(sweep), objects: phase >= 2 ? objects.length : 0, rest, plan: planAlphas };
  }

  function dispose() {
    lighting.dispose();
    cloud.dispose();
    labels.dispose();
    renderer.dispose();
  }

  content.updateMatrixWorld(true);
  return { resize, render, dispose };
}
