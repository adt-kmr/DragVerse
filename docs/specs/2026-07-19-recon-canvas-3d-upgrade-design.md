# ReconCanvas 3D Upgrade — Design

## Context

`ReconCanvas.jsx` (dashboard landing page) is the "signature element": a single
canvas scrubbed by scroll through six pipeline phases (Capture → Reconstruct →
Segment → Generate twin → Train → Deploy) over one procedurally generated room.
It is currently a **2D `<canvas>` isometric line drawing** — a hand-rolled
affine projection (`isoX = (x-y)·cos30°`, `isoY = (x+y)·sin30° - z·1.02`) with
points drawn via `ctx.fillRect` and boxes via `ctx.strokeStyle` paths. There is
no WebGL, no 3D library, no lighting model; `package.json` has zero 3D
dependencies.

A mission brief was provided asking for a "production-quality visualization"
upgrade — real PBR materials, HDR lighting, soft shadows, modeled furniture,
post-processing — while explicitly preserving scroll choreography, camera
path, stage ordering, timing, and semantic labeling. That brief assumes an
existing Three.js/WebGL renderer being polished. It doesn't exist yet.

The project's `dashboard/DESIGN.md` also documents a deliberate, doctrinal flat
design system ("Absolutely flat. Zero `box-shadow` declarations... explicitly
rejects the dark neon robotics site") that is close to the literal opposite of
the mission's PBR/shadow/bloom ask. This was surfaced to the user directly;
they confirmed they want the full 3D/PBR direction taken literally, overriding
`DESIGN.md`'s no-shadow doctrine for this one component. `DESIGN.md` is not
otherwise being touched by this work.

## Goal

Replace the 2D isometric renderer with a real Three.js WebGL scene —
procedurally modeled furniture, PBR materials, IBL lighting, soft shadows,
light post-processing — while the six-phase scroll experience, camera
framing, object positions/scale, timing, and semantic label colors stay
pixel-equivalent to today.

## Non-goals

- Redesigning the pipeline stages, storytelling, or scroll choreography.
- Touching `dashboard/DESIGN.md`'s doctrine for the rest of the site — this
  change is scoped to the ReconCanvas component only.
- LOD, Draco/KTX2 compression, texture atlasing — see "Explicitly trimmed"
  below; nothing in this scene warrants them.
- Loading external 3D assets (GLTF models, HDR files) from a CDN — everything
  is procedural, for both licensing and offline-friendliness reasons
  consistent with the rest of the project.

## Architecture

`ReconCanvas.jsx` is unchanged. It only depends on `createScene(canvas) →
{resize, render(progress) → {phase, points, objects, rest}}` — all GSAP
ScrollTrigger wiring, `prefers-reduced-motion` handling, and the DOM readout
writes live there already and need no changes.

`recon.js` keeps its public data exports (`STAGES`, `ROOM`, `PROPS`, `PATH`)
untouched — these are the source of truth for content, position, and scale.
Only `createScene()`'s internals change, from 2D canvas draws to a Three.js
scene, split into:

- `recon.js` — data exports + thin `createScene()` orchestrator (same
  `{resize, render}` contract as today).
- `recon/geometry.js` — procedural mesh builders: chair, table, cabinet,
  shelf, robot, walls/floor/door.
- `recon/materials.js` — the PBR material library, keyed by label.
- `recon/lighting.js` — light rig + `RoomEnvironment` IBL setup.
- `recon/pipeline.js` — post-processing chain (`EffectComposer` + passes).
- `recon/pointcloud.js` — GPU-instanced point cloud (`THREE.Points` +
  custom shader).

`aperture.js` (the resting hero dither) is untouched — separate element, not
part of this scene.

**New dependency:** `three` — the only one. Three.js ships procedural IBL
(`RoomEnvironment`), all needed post-processing passes, and `CSS2DRenderer`
in its own `examples/jsm`, so no `drei`, `postprocessing`, or `react-three-fiber`
packages are needed. Vanilla imperative Three.js also matches `recon.js`'s
existing documented style ("pure canvas — no React, no GSAP").

## Camera

Today's projection is static — one fixed 30° isometric angle, with two
responsive framings (narrow <900px vs wide) chosen at `resize()`. There is no
camera *movement* in the current implementation; only content sweeps in.

This becomes an `OrthographicCamera` positioned/rotated to reproduce that
exact isometric angle mathematically, with frustum bounds recomputed on
resize to mirror the current narrow/wide branching. Orthographic projection
keeps parallel edges parallel exactly as today's hand math does — no
perspective distortion introduced. The only additions are the mission's
explicitly "subtle" ones — a few pixels of micro-parallax/settle easing and a
very light bokeh pass — layered on top of the fixed frame, never moving the
camera's actual position or angle.

## Geometry

`PROPS` entries (`{label, x, y, z, w, d, h}`) remain the single source of
truth for position and scale. What changes is what gets built from an entry:
each `label` maps to a procedural builder in `geometry.js` that constructs a
real mesh (chair legs/seat/backrest, cabinet body/doors, robot chassis/wheels,
etc.) **fitted exactly inside that entry's existing bounding box** — same
footprint, position, and overall scale as today's plain box, with believable
internal proportions (e.g. thin legs, not a solid block). Walls/floor/door get
the same treatment (thin slabs with baseboard/trim) in place of the current
point-sampled planes.

Each mesh also carries an `EdgesGeometry` overlay for the wireframe-reveal
stage (phase 3), reusing the existing `edges.length * clamp(t, 0, 1)` reveal
math — just against real mesh edges instead of a 12-edge box.

Polygon budget: a handful of props + room shell, target low thousands of
triangles total — comfortably within "avoid excessively high polygon counts."

## Materials

One label → one `MeshStandardMaterial`/`MeshPhysicalMaterial`, per the
mission's material list and palette:

| label | material |
|---|---|
| floor | oak wood, subtle grain roughness variation |
| wall | painted drywall, matte, warm neutral |
| door | painted wood frame |
| table, shelf | oak wood (same family as floor) |
| chair | painted wood + fabric seat |
| cabinet | painted wood + brushed-aluminum handles |
| robot | matte plastic (light gray body) + rubber (wheels) |

**Carve-out (from the mission's own Color Palette section — "semantic
overlays retain current accent color... reserved for AI visualization"):**
the state-signal colors (Signal Orange = scanning/robot marker, Survey Teal =
labelled, Blueprint Indigo = settled/path/quantize) are not replaced by PBR
realism. Per `DESIGN.md`'s Measured Colour Rule, they stay as unlit/emissive
accents layered on top of the physically-lit scene — never a material a light
bounces off. Realism applies to the room/furniture; the colors that carry
pipeline meaning stay exactly as they read today.

Materials are only visible from phase 3 onward — before that, only points and
wireframe exist (see Phase mechanics).

## Lighting & shadows

- **IBL:** Three's built-in procedural `RoomEnvironment` (ships with `three`,
  zero asset fetch) through `PMREMGenerator`.
- **Key light:** one soft `DirectionalLight` standing in for daylight through
  the door opening; `PCFSoftShadowMap`; shadow camera frustum fit tightly to
  `ROOM`'s bounds.
- **Fill:** one `HemisphereLight` tinted from the existing vellum/graphite
  tones for ambient fill.
- **AO/contact shadow:** an SSAO post-process pass rather than a baked
  contact-shadow plane — the scene is fully procedural/dynamic, so
  screen-space AO needs no baking step.
- Kept low-contrast/low-intensity throughout — "never theatrical," consistent
  with both the mission and the project's existing understated visual
  language.

## Point cloud rendering

Today's per-point `ctx.fillRect` loop (CPU, one call per point, every frame)
becomes a single `THREE.Points` draw with per-point attributes (position,
depth, label-color, reveal-threshold, jitter-seed) and a custom
vertex/fragment shader handling reveal, jitter, depth-ramp↔label-color mix,
size attenuation, and atmospheric fade — all GPU-side, one draw call
regardless of point count. `CLOUD`'s deterministic build-time generation
(seeded `mulberry32` RNG, `sampleBox`) is unchanged — same point positions,
just uploaded as GPU buffers instead of walked in JS per frame.

## Phase-by-phase transition mechanics

The existing `render(progress)` math is unchanged: same `rest`/`run`/`phase`/
`t` derivation, same per-object stagger (`local = clamp(build * PROPS.length -
i, 0, 1)`), same timing windows for path/quantize. Only what happens *inside*
those windows gets richer:

- **Phase 0 Capture** — unchanged: points sweep in by x-position, colored by
  the depth ramp.
- **Phase 1 Reconstruct** — unchanged: jitter/noise settles (`jitter: 1 → 0`).
- **Phase 2 Segment** — unchanged: points recolor depth-ramp → semantic accent
  color (`labelMix: 0 → 1`). Points stay an unlit signal layer throughout.
- **Phase 3 Generate twin** — the mission's "converge → wireframe grows →
  surface fills → materials fade in → roughness resolves → lighting settles"
  sequence, subdivided within each object's existing `local` 0→1 window:
  edges stroke-in (0–0.35) → mesh scales/dissolves in (0.35–0.65) → material
  crossfades from flat unlit "clay" to full PBR while roughness relaxes to
  target (0.65–0.85) → shadow/AO contribution fades in last (0.85–1.0). Same
  tag/leader-line fade math as today.
- **Phase 4 Train** — unchanged timing; traced path becomes a thin tube/line
  mesh with a soft indigo emissive accent instead of a dashed 2D stroke.
- **Phase 5 Deploy** — unchanged timing; quantize overlay becomes a light
  `InstancedMesh` grid clipped to the room, fading in per `t`; robot mesh
  drives along `PATH` with the same lerp position math as today, plus
  wheel-spin rotation.

## Post-processing

`EffectComposer` chain: `RenderPass` → `SSAOPass` (subtle) → a small
`UnrealBloomPass` restricted to the emissive signal accents only (scan plane,
robot marker, path glow — the one "physically justified" bloom source) →
`SMAAPass` → `ACESFilmicToneMapping` set on the renderer directly (not a
pass). No vignette, chromatic aberration, grain, or lens flare.

## Explicitly trimmed from the mission's perf list

These don't apply at this scene's actual scale (one ~8×6m room, 6 props,
fixed camera that never moves or zooms) and would be unused scaffolding:

- **LOD** — no camera distance range exists to swap detail against.
- **Draco / KTX2** — both compress *loaded* assets; nothing here is loaded.
- **Texture atlases** — the handful of small procedural textures don't
  warrant it.

Kept: `THREE.Points` instancing for the point cloud, `InstancedMesh` for the
quantize grid, default frustum culling, capped pixel ratio (2, matching
today's DPR cap), `antialias: false` on the context (SMAA handles it more
cheaply), one tightly-fit shadow map. Target: 60fps.

## Labels

Today's `ctx.fillText` becomes a DOM overlay via `CSS2DRenderer`: each tag's
anchor projects from 3D world space to screen space (`camera.project()`
replacing the `px/py` iso formulas), then applies the same per-prop
`tdx/tdy` pixel offset as today. Label text renders as real DOM nodes reusing
the existing `.recon__step`/`.recon__name` etc. CSS classes already in
`styles.css` — same Archivo/JetBrains Mono type system, same `-ink` colors,
crisp at any DPI. Leader lines render as a thin synced SVG overlay, redrawn
each frame from the same projected anchor/offset pair. Same fade timing as
today.

## Testing

Manual verification per the mission's own phase gates: after each
implementation phase, confirm scroll behavior, camera framing, phase timing,
and `prefers-reduced-motion` fallback are unchanged from today, and check
frame rate. No automated visual-regression tooling exists in this repo today;
none is being added for this change (out of scope — a dashboard visual
polish pass, not a testing infrastructure project).
