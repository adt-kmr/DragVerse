/* Object tags (phase 3+): real DOM text positioned by projecting each prop's anchor
   through the camera, replacing today's ctx.fillText. Same anchor/offset math as the old
   drawTag() — center-top of the bbox, nudged by the prop's own tdx/tdy — just sourced from
   camera.project() instead of the hand-rolled iso formulas. A synced SVG line stands in for
   the canvas-stroked leader. */

import * as THREE from "three";

const anchor = new THREE.Vector3();

export function createLabelLayer(hostCanvas, props, colliderMap, contentGroup) {
  const host = hostCanvas.parentElement;

  const container = document.createElement("div");
  container.className = "recon__tags";
  host.appendChild(container);

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");
  svg.style.position = "absolute";
  svg.style.inset = "0";
  container.appendChild(svg);

  const entries = props.map((prop) => {
    const tag = document.createElement("div");
    tag.className = "recon__tag";
    const label = document.createElement("span");
    label.className = "recon__tag-label";
    label.textContent = prop.label;
    const collider = document.createElement("span");
    collider.className = "recon__tag-collider";
    collider.textContent = colliderMap[prop.label];
    tag.append(label, collider);
    container.appendChild(tag);

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("class", "recon__tag-leader");
    svg.appendChild(line);

    return { prop, tag, line };
  });

  // Per-index world-position override — used to keep the robot's tag tracking its mesh
  // once it leaves its dock and drives the route, instead of staying pinned to the static
  // bbox anchor like every other (stationary) object's tag.
  const anchorOverride = new Map();
  function setAnchorOverride(index, worldPosOrNull) {
    if (worldPosOrNull) anchorOverride.set(index, worldPosOrNull);
    else anchorOverride.delete(index);
  }

  function render(camera) {
    const w = hostCanvas.clientWidth;
    const h = hostCanvas.clientHeight;
    if (!w || !h) return;
    entries.forEach(({ prop, tag, line }, i) => {
      const override = anchorOverride.get(i);
      if (override) {
        anchor.copy(override);
      } else {
        anchor.set(prop.x + prop.w / 2, prop.y + prop.d / 2, prop.z + prop.h);
        contentGroup.localToWorld(anchor);
      }
      anchor.project(camera);
      const ax = (anchor.x * 0.5 + 0.5) * w;
      const ay = (1 - (anchor.y * 0.5 + 0.5)) * h;
      const x = ax + prop.tdx;
      const y = ay - 24 + prop.tdy;

      tag.style.left = `${x}px`;
      tag.style.top = `${y}px`;
      line.setAttribute("x1", ax);
      line.setAttribute("y1", ay - 2);
      line.setAttribute("x2", x);
      line.setAttribute("y2", y + 5);
    });
  }

  function setAlpha(alphas) {
    entries.forEach(({ tag, line }, i) => {
      const a = alphas[i];
      tag.style.opacity = a;
      line.style.opacity = a;
    });
  }

  // The one-off "int8 · on-device" quantize caption (phase 5) — same DOM/projection
  // approach as the per-prop tags, just a single fixed-position element.
  const quantizeCaption = document.createElement("div");
  quantizeCaption.className = "recon__tag-collider";
  quantizeCaption.style.position = "absolute";
  quantizeCaption.style.opacity = 0;
  quantizeCaption.textContent = "int8 · on-device";
  container.appendChild(quantizeCaption);

  function renderQuantizeCaption(camera, worldAnchor, alpha) {
    const w = hostCanvas.clientWidth;
    const h = hostCanvas.clientHeight;
    if (!w || !h) return;
    anchor.copy(worldAnchor).project(camera);
    const x = (anchor.x * 0.5 + 0.5) * w;
    const y = (1 - (anchor.y * 0.5 + 0.5)) * h - 10;
    quantizeCaption.style.left = `${x}px`;
    quantizeCaption.style.top = `${y}px`;
    quantizeCaption.style.opacity = alpha;
  }

  function dispose() {
    container.remove();
  }

  return { render, setAlpha, setAnchorOverride, renderQuantizeCaption, dispose };
}
