/* Procedural furniture + room shell. Every builder fits its output exactly inside the
   caller's existing bounding box ({x, y, z, w, d, h} — the same PROPS/ROOM data recon.js
   has always used) so footprint, position, and scale never move; only what's built inside
   that box gets real proportions instead of a solid primitive.

   Each returned group carries its own cloned edgeMaterial (not the shared template) so
   recon.js can fade one object's wireframe independently of every other object's, during
   the phase-3 per-object stagger. */

import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";

const BEVEL = 0.012; // metres — small, consistent corner radius, not a design choice per object
const SEG = 3; // bevel smoothness — kept low, this is product-viz scale, not hero-prop scale

// RoundedBoxGeometry's native axes are (width, height, depth) with height along local Y —
// but every caller passes args in this scene's own convention (x/y/z = width/depth/height,
// see recon.js), so h and d swap here once, for every call site, instead of at each one.
function roundedBox(w, h, d) {
  return new RoundedBoxGeometry(Math.max(w, 0.01), Math.max(d, 0.01), Math.max(h, 0.01), SEG, BEVEL);
}

/** A solid mesh tagged with the PBR values it should crossfade *to* during phase 3 —
    recon.js drives material.color/roughness/metalness from CLAY_COLOR toward these. */
function solid(geometry, material, x, y, z) {
  const mesh = new THREE.Mesh(geometry, material.clone());
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.userData.target = {
    color: material.color.clone(),
    roughness: material.roughness,
    metalness: material.metalness,
  };
  return mesh;
}

/** Wireframe overlay for the phase-3 reveal stage — see the ponytail note on
    edgeOpacity in recon.js for why this fades as a set rather than growing edge-by-edge.
    LineSegments2 (fat lines, screen-space quads) rather than plain LineSegments+
    EdgesGeometry — 1px GL_LINE primitives render unreliably without MSAA (confirmed
    empirically elsewhere in this scene), and fat lines also give an actual controllable
    width instead of "however thin your GPU/driver feels like". */
function edgesFor(geometry, x, y, z, material) {
  const segGeo = new LineSegmentsGeometry().fromEdgesGeometry(new THREE.EdgesGeometry(geometry, 20));
  const line = new LineSegments2(segGeo, material);
  line.position.set(x, y, z);
  line.computeLineDistances();
  line.frustumCulled = false;
  return line;
}

function assemble(prop, parts, edgeMaterialTemplate) {
  const group = new THREE.Group();
  group.position.set(prop.x, prop.y, prop.z);
  const edgeMaterial = edgeMaterialTemplate.clone(); // one clone per object — see header note
  // Thinner than the template (which is also used directly, un-thinned, for the traced
  // path) — structural wireframe reads as a lighter accent than "the executed route".
  edgeMaterial.linewidth = 1.4;
  const solids = [];
  const edgeLines = [];
  for (const { geometry, material, x, y, z } of parts) {
    const mesh = solid(geometry, material, x, y, z);
    group.add(mesh);
    solids.push(mesh);
    edgeLines.push(edgesFor(geometry, x, y, z, edgeMaterial));
  }
  for (const e of edgeLines) group.add(e);
  return { group, solids, edgeLines, edgeMaterial };
}

/* ------------------------------------------------------------------------ furniture */

function buildTable(prop, mats, edgeMaterial) {
  const legR = 0.035;
  const legH = prop.h - 0.05;
  const inset = 0.08;
  const legXs = [inset, prop.w - inset];
  const legYs = [inset, prop.d - inset];
  const parts = [];
  for (const lx of legXs) {
    for (const ly of legYs) {
      parts.push({ geometry: new THREE.CylinderGeometry(legR, legR, legH, 10), material: mats.oak, x: lx, y: ly, z: legH / 2 });
    }
  }
  parts.push({ geometry: roundedBox(prop.w, 0.05, prop.d), material: mats.oak, x: prop.w / 2, y: prop.d / 2, z: prop.h - 0.025 });
  const g = assemble(prop, parts, edgeMaterial);
  // Legs are cylinders — their length defaults to local Y, so they need rotating onto Z
  // (this scene's height axis) to stand up instead of lying along depth.
  for (let i = 0; i < 4; i++) g.solids[i].rotation.x = Math.PI / 2;
  return g;
}

function buildChair(prop, mats, edgeMaterial) {
  const legR = 0.022;
  const seatH = prop.h * 0.5;
  const inset = 0.05;
  const legXs = [inset, prop.w - inset];
  const legYs = [inset, prop.d - inset];
  const parts = [];
  for (const lx of legXs) {
    for (const ly of legYs) {
      parts.push({ geometry: new THREE.CylinderGeometry(legR, legR, seatH, 8), material: mats.paintedWood, x: lx, y: ly, z: seatH / 2 });
    }
  }
  parts.push({ geometry: roundedBox(prop.w - 0.04, 0.04, prop.d - 0.04), material: mats.fabric, x: prop.w / 2, y: prop.d / 2, z: seatH + 0.02 });
  parts.push({ geometry: roundedBox(prop.w - 0.06, prop.h - seatH, 0.03), material: mats.paintedWood, x: prop.w / 2, y: prop.d - 0.03, z: seatH + (prop.h - seatH) / 2 });
  const g = assemble(prop, parts, edgeMaterial);
  // Legs are cylinders — see the note in buildTable.
  for (let i = 0; i < 4; i++) g.solids[i].rotation.x = Math.PI / 2;
  return g;
}

function buildShelf(prop, mats, edgeMaterial) {
  const t = 0.03;
  const parts = [
    { geometry: roundedBox(prop.w, prop.h, t), material: mats.oak, x: prop.w / 2, y: t / 2, z: prop.h / 2 },
    { geometry: roundedBox(t, prop.h, prop.d), material: mats.oak, x: t / 2, y: prop.d / 2, z: prop.h / 2 },
    { geometry: roundedBox(t, prop.h, prop.d), material: mats.oak, x: prop.w - t / 2, y: prop.d / 2, z: prop.h / 2 },
  ];
  const shelves = 4;
  for (let i = 0; i <= shelves; i++) {
    const z = (prop.h - t) * (i / shelves) + t / 2;
    parts.push({ geometry: roundedBox(prop.w - t * 2, t, prop.d), material: mats.oak, x: prop.w / 2, y: prop.d / 2, z });
  }
  return assemble(prop, parts, edgeMaterial);
}

function buildCabinet(prop, mats, edgeMaterial) {
  const parts = [{ geometry: roundedBox(prop.w, prop.h, prop.d), material: mats.paintedWood, x: prop.w / 2, y: prop.d / 2, z: prop.h / 2 }];
  const handleR = 0.012;
  const handleH = prop.h * 0.16;
  const doorSplit = prop.w / 2;
  for (const hx of [doorSplit * 0.18, prop.w - doorSplit * 0.18]) {
    parts.push({
      geometry: new THREE.CylinderGeometry(handleR, handleR, handleH, 8),
      material: mats.aluminum,
      x: hx,
      y: -0.012,
      z: prop.h * 0.55,
    });
  }
  const g = assemble(prop, parts, edgeMaterial);
  // Handles read as vertical grips, not floor-mounted posts — rotate them onto the door face.
  g.solids[1].rotation.x = g.solids[2].rotation.x = Math.PI / 2;
  return g;
}

function buildRobot(prop, mats, edgeMaterial, signalOrange) {
  const wheelR = Math.min(prop.w, prop.d) * 0.16;
  const chassisH = prop.h * 0.55;
  const mastTopZ = wheelR + chassisH + prop.h * 0.3;
  const parts = [
    { geometry: roundedBox(prop.w * 0.86, chassisH, prop.d * 0.86), material: mats.plastic, x: prop.w / 2, y: prop.d / 2, z: wheelR + chassisH / 2 },
    { geometry: new THREE.CylinderGeometry(prop.w * 0.14, prop.w * 0.14, prop.h * 0.3, 12), material: mats.plastic, x: prop.w / 2, y: prop.d / 2, z: mastTopZ - prop.h * 0.15 },
  ];
  const wheelZ = wheelR;
  const wheelXs = [prop.w * 0.14, prop.w * 0.86];
  for (const wx of wheelXs) {
    parts.push({ geometry: new THREE.CylinderGeometry(wheelR, wheelR, 0.05, 14), material: mats.rubber, x: wx, y: prop.d * 0.14, z: wheelZ });
    parts.push({ geometry: new THREE.CylinderGeometry(wheelR, wheelR, 0.05, 14), material: mats.rubber, x: wx, y: prop.d * 0.86, z: wheelZ });
  }
  const g = assemble(prop, parts, edgeMaterial);
  g.solids[1].rotation.x = Math.PI / 2; // mast is a cylinder — see the note in buildTable.
  for (let i = 2; i < g.solids.length; i++) g.solids[i].rotation.z = Math.PI / 2;

  // Beacon, floating well above the tallest furniture in the room (cabinet: 1.45m, shelf:
  // 1.9m — the tallest, so 4x the robot's own 0.55m height is what actually clears it) —
  // unlit signal orange, not part of `solids` so it never gets pulled into the clay->PBR
  // crossfade. A real 3D room means real occlusion: the robot's actual body, at floor
  // height, regularly ends up entirely hidden behind the table/chairs/cabinet/shelf from
  // this isometric angle exactly as solid geometry should behave — confirmed by testing a
  // fixed marker at table height, which was mostly hidden behind the table too. A beacon
  // riding well above everything else is the standard fix real AMR fleets use for exactly
  // this problem, and it stays perfectly in sync with the body since it's a child of the
  // same group — it's always readable as "here's the robot," regardless of what's in
  // front of the body itself.
  const light = new THREE.Mesh(new THREE.SphereGeometry(prop.w * 0.16, 12, 10), signalOrange);
  light.position.set(prop.w / 2, prop.d / 2, prop.h * 4);
  light.frustumCulled = false;
  g.group.add(light);
  g.light = light;
  return g;
}

const BUILDERS = { table: buildTable, chair: buildChair, shelf: buildShelf, cabinet: buildCabinet, robot: buildRobot };

export function buildProp(prop, mats, edgeMaterial, signalOrange) {
  return BUILDERS[prop.label](prop, mats, edgeMaterial, signalOrange);
}

/* ---------------------------------------------------------------------- room shell */

const WALL_T = 0.12;

export function buildRoomShell(room, mats, edgeMaterial) {
  const walls = [
    { x: 0, y: 0, z: 0, w: WALL_T, d: room.d, h: room.h },
    { x: room.w - WALL_T, y: 0, z: 0, w: WALL_T, d: room.d, h: room.h },
    { x: 0, y: room.d - WALL_T, z: 0, w: room.w, d: WALL_T, h: room.h },
    { x: 0, y: 0, z: 0, w: 3.4, d: WALL_T, h: room.h },
    { x: 4.5, y: 0, z: 0, w: room.w - 4.5, d: WALL_T, h: room.h },
  ];
  const doorLintel = { x: 3.4, y: 0, z: 2.05, w: 1.1, d: WALL_T, h: 0.65 };
  const baseboardH = 0.09;

  const parts = [{ geometry: roundedBox(room.w, 0.02, room.d), material: mats.oak, x: room.w / 2, y: room.d / 2, z: -0.01 }];
  for (const w of walls) {
    parts.push({ geometry: roundedBox(w.w, w.h, w.d), material: mats.drywall, x: w.x + w.w / 2, y: w.y + w.d / 2, z: w.z + w.h / 2 });
    // Baseboard trim: a slightly darker strip along the wall's floor edge, per the mission's
    // "walls / trim / baseboards" surface-detail ask.
    parts.push({
      geometry: roundedBox(w.w + 0.01, baseboardH, w.d + 0.01),
      material: mats.paintedWood,
      x: w.x + w.w / 2,
      y: w.y + w.d / 2,
      z: w.z + baseboardH / 2,
    });
  }
  parts.push({ geometry: roundedBox(doorLintel.w, doorLintel.h, doorLintel.d), material: mats.doorWood, x: doorLintel.x + doorLintel.w / 2, y: doorLintel.y + doorLintel.d / 2, z: doorLintel.z + doorLintel.h / 2 });

  return assemble({ x: 0, y: 0, z: 0 }, parts, edgeMaterial);
}
