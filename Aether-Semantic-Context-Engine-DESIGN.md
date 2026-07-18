---
version: "alpha"
name: "Aether — Semantic Context Engine"
description: "Aether Semantic Dashboard Section is designed for demonstrating application workflows and interface hierarchy. Key features include clear information density, modular panels, and interface rhythm. It is suitable for product showcases, admin panels, and analytics experiences."
colors:
  primary: "#FF7A3D"
  secondary: "#3A1E12"
  tertiary: "#FF3B30"
  neutral: "#140A08"
  background: "#1C0F0A"
  surface: "#FF7A3D"
  text-primary: "#FBEFE6"
  text-secondary: "#FF7A3D"
  border: "#FF7A3D"
  accent: "#FF7A3D"
typography:
  display-lg:
    fontFamily: "Chakra Petch"
    fontSize: "96px"
    fontWeight: 700
    lineHeight: "96px"
    letterSpacing: "-0.025em"
    textTransform: "uppercase"
  body-md:
    fontFamily: "Fira Code"
    fontSize: "14px"
    fontWeight: 300
    lineHeight: "20px"
  label-md:
    fontFamily: "Chakra Petch"
    fontSize: "12px"
    fontWeight: 600
    lineHeight: "16px"
    letterSpacing: "1.8px"
    textTransform: "uppercase"
rounded:
  md: "8px"
spacing:
  base: "8px"
  sm: "8px"
  md: "14px"
  lg: "16px"
  xl: "24px"
  gap: "4px"
  card-padding: "16px"
components:
  button-primary:
    textColor: "{colors.neutral}"
    typography: "{typography.label-md}"
    rounded: "{rounded.md}"
    padding: "14px"
  button-secondary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.primary}"
    typography: "{typography.label-md}"
    rounded: "{rounded.md}"
    padding: "14px"
  card:
    rounded: "12px"
    padding: "16px"
---

## Overview

- **Composition cues:**
  - Layout: Grid
  - Content Width: Bounded
  - Framing: Glassy
  - Grid: Strong

## Colors

The color system uses light mode with #FF7A3D as the main accent and #140A08 as the neutral foundation.

- **Primary (#FF7A3D):** Main accent and emphasis color.
- **Secondary (#3A1E12):** Supporting accent for secondary emphasis.
- **Tertiary (#FF3B30):** Reserved accent for supporting contrast moments.
- **Neutral (#140A08):** Neutral foundation for backgrounds, surfaces, and supporting chrome.

- **Usage:** Background: #1C0F0A; Surface: #FF7A3D; Text Primary: #FBEFE6; Text Secondary: #FF7A3D; Border: #FF7A3D; Accent: #FF7A3D

- **Gradients:** bg-gradient-to-r from-[#E2612F] to-[#FFA06B], bg-gradient-to-r from-[#8FE3A8]/70 to-[#8FE3A8]

## Typography

Typography pairs Chakra Petch for display hierarchy with Fira Code for supporting content and interface copy.

- **Display (`display-lg`):** Chakra Petch, 96px, weight 700, line-height 96px, letter-spacing -0.025em, uppercase.
- **Body (`body-md`):** Fira Code, 14px, weight 300, line-height 20px.
- **Labels (`label-md`):** Chakra Petch, 12px, weight 600, line-height 16px, letter-spacing 1.8px, uppercase.

## Layout

Layout follows a grid composition with reusable spacing tokens. Preserve the grid, bounded structural frame before changing ornament or component styling. Use 8px as the base rhythm and let larger gaps step up from that cadence instead of introducing unrelated spacing values.

Treat the page as a grid / bounded composition, and keep that framing stable when adding or remixing sections.

- **Layout type:** Grid
- **Content width:** Bounded
- **Base unit:** 8px
- **Scale:** 8px, 14px, 16px, 24px, 28px, 32px, 40px, 48px
- **Card padding:** 16px
- **Gaps:** 4px, 6px, 8px, 10px

## Elevation & Depth

Depth is communicated through glass, border contrast, and reusable shadow or blur treatments. Keep those recipes consistent across hero panels, cards, and controls so the page reads as one material system.

Surfaces should read as glass first, with borders, shadows, and blur only reinforcing that material choice.

- **Surface style:** Glass
- **Borders:** 1px #FF7A3D; 2px #FF7A3D; 1px #E5E7EB
- **Shadows:** rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(255, 59, 48, 0.8) 0px 0px 10px 0px; rgba(226, 97, 47, 0.35) 0px 4px 14px 0px, rgba(255, 236, 220, 0.8) 0px 1px 1px 0px inset, rgba(0, 0, 0, 0.2) 0px -2px 4px 0px inset; rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(0, 0, 0, 0) 0px 0px 0px 0px, rgba(255, 122, 61, 0.6) 0px 0px 8px 0px
- **Blur:** 10px

### Techniques
- **Gradient border shell:** Use a thin gradient border shell around the main card. Wrap the surface in an outer shell with 0px padding and a 0px radius. Drive the shell with linear-gradient(to top, rgba(20, 10, 8, 0.92), rgba(0, 0, 0, 0)) so the edge reads like premium depth instead of a flat stroke. Keep the actual stroke understated so the gradient shell remains the hero edge treatment. Inset the real content surface inside the wrapper with a slightly smaller radius so the gradient only appears as a hairline frame.

## Shapes

Shapes rely on a tight radius system anchored by 8px and scaled across cards, buttons, and supporting surfaces. Icon geometry should stay compatible with that soft-to-controlled silhouette.

Use the radius family intentionally: larger surfaces can open up, but controls and badges should stay within the same rounded DNA instead of inventing sharper or pill-only exceptions.

- **Corner radii:** 8px, 12px, 9999px
- **Icon treatment:** Linear
- **Icon sets:** Solar

## Components

Anchor interactions to the detected button styles. Reuse the existing card surface recipe for content blocks.

### Buttons
- **Primary:** text #140A08, radius 8px, padding 14px, border 0px solid rgb(229, 231, 235).
- **Secondary:** background #FF7A3D, text #FF7A3D, radius 8px, padding 14px, border 1px solid rgba(255, 122, 61, 0.35).

### Cards and Surfaces
- **Card surface:** background rgba(28, 15, 10, 0.7), border 1px solid rgba(255, 122, 61, 0.15), radius 12px, padding 16px, shadow none, blur 10px.

### Iconography
- **Treatment:** Linear.
- **Sets:** Solar.

## Do's and Don'ts

Use these constraints to keep future generations aligned with the current system instead of drifting into adjacent styles.

### Do
- Do use the primary palette as the main accent for emphasis and action states.
- Do keep spacing aligned to the detected 8px rhythm.
- Do reuse the Glass surface treatment consistently across cards and controls.
- Do keep corner radii within the detected 8px, 12px, 9999px family.

### Don't
- Don't introduce extra accent colors outside the core palette roles unless the page needs a new semantic state.
- Don't mix unrelated shadow or blur recipes that break the current depth system.
- Don't exceed the detected expressive motion intensity without a deliberate reason.

## Motion

Motion feels expressive but remains focused on interface, text, and layout transitions. Timing clusters around 900ms and 150ms. Easing favors ease and 1). Hover behavior focuses on shadow and color changes.

**Motion Level:** expressive

**Durations:** 900ms, 150ms, 1200ms, 9000ms, 1300ms, 30000ms

**Easings:** ease, 1), 1, cubic-bezier(0.19, 0.22, cubic-bezier(0.4

**Hover Patterns:** shadow, color, text

## WebGL

Reconstruct the graphics as a full-bleed background field using webgl, renderer, alpha, antialias, dpr clamp, custom shaders. The effect should read as retro-futurist, technical, and meditative: dot-matrix particle field with green on black and sparse spacing. Build it from dot particles + soft depth fade so the effect reads clearly. Animate it as slow breathing pulse. Interaction can react to the pointer, but only as a subtle drift. Preserve reduced motion + dom fallback.

**Id:** webgl

**Label:** WebGL

**Stack:** ThreeJS, WebGL

**Insights:**
  - **Scene:**
    - **Value:** Full-bleed background field
  - **Effect:**
    - **Value:** Dot-matrix particle field
  - **Primitives:**
    - **Value:** Dot particles + soft depth fade
  - **Motion:**
    - **Value:** Slow breathing pulse
  - **Interaction:**
    - **Value:** Pointer-reactive drift
  - **Render:**
    - **Value:** WebGL, Renderer, alpha, antialias, DPR clamp, custom shaders

**Techniques:** Dot matrix, Breathing pulse, Pointer parallax, Shader gradients, Noise fields

**Code Evidence:**
  - **HTML reference:**
    - **Language:** html
    - **Snippet:**
      ```html
      <!-- Background WebGL Canvas -->
      <div class="fixed inset-0 z-0 pointer-events-none">
          <canvas id="terrain-canvas" class="w-full h-full block"></canvas>
      </div>
      ```
  - **JS reference:**
    - **Language:** html
    - **Snippet:**
      ```html
      <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
      ```

## ThreeJS

Reconstruct the Three.js layer as a full-bleed background field with layered spatial depth that feels retro-futurist, volumetric, and technical. Use alpha, antialias, tone mapping, dpr clamp renderer settings, perspective, ~45deg fov, plane geometry, meshstandardmaterial + meshbasicmaterial materials, and directional + hemisphere lighting. Motion should read as slow orbital drift, with reduced motion + non-3d fallback.

**Id:** threejs

**Label:** ThreeJS

**Stack:** ThreeJS, WebGL

**Insights:**
  - **Scene:**
    - **Value:** Full-bleed background field with layered spatial depth
  - **Render:**
    - **Value:** alpha, antialias, tone mapping, DPR clamp
  - **Camera:**
    - **Value:** Perspective, ~45deg FOV
  - **Lighting:**
    - **Value:** directional + hemisphere
  - **Materials:**
    - **Value:** MeshStandardMaterial + MeshBasicMaterial
  - **Geometry:**
    - **Value:** plane
  - **Motion:**
    - **Value:** Slow orbital drift

**Techniques:** PBR shading, Timeline beats, alpha, antialias, tone mapping, DPR clamp, Reduced motion + non-3D fallback

**Code Evidence:**
  - **HTML reference:**
    - **Language:** html
    - **Snippet:**
      ```html
      <!-- Background WebGL Canvas -->
      <div class="fixed inset-0 z-0 pointer-events-none">
          <canvas id="terrain-canvas" class="w-full h-full block"></canvas>
      </div>
      ```
  - **JS reference:**
    - **Language:** html
    - **Snippet:**
      ```html
      <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
      ```
