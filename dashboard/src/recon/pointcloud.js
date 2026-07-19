/* The scan point cloud as one GPU-instanced THREE.Points draw call — reveal, jitter,
   depth-ramp-to-label-color mix, size, and atmospheric fade all run in the shader instead
   of a per-point ctx.fillRect loop. CLOUD's positions are untouched (still the deterministic
   mulberry32-seeded build from recon.js); only how they get to the screen changes. */

import * as THREE from "three";

const RAMP = [0xff5b2e, 0x14b09a, 0x22307e]; // must match recon.js's depth ramp / DESIGN.md
const FOG_COLOR = new THREE.Color(0xeef0ea); // vellum — matches the page the canvas sits on

const VERTEX = /* glsl */ `
  attribute vec3 aJitter;
  attribute vec3 aLabelColor;
  attribute float aReveal; // x / ROOM.w — drives the sweep-reveal threshold
  attribute float aDepth;  // (x + y) / (ROOM.w + ROOM.d) — drives the depth ramp + fog
  attribute float aBrightness;

  uniform float uSweep;
  uniform float uJitter;
  uniform float uLabelMix;
  uniform float uFade;
  uniform float uPointPx;

  varying float vAlpha;
  varying vec3 vColor;

  vec3 rampColor(float t) {
    vec3 c0 = vec3(${(RAMP[0] >> 16 & 255) / 255}, ${(RAMP[0] >> 8 & 255) / 255}, ${(RAMP[0] & 255) / 255});
    vec3 c1 = vec3(${(RAMP[1] >> 16 & 255) / 255}, ${(RAMP[1] >> 8 & 255) / 255}, ${(RAMP[1] & 255) / 255});
    vec3 c2 = vec3(${(RAMP[2] >> 16 & 255) / 255}, ${(RAMP[2] >> 8 & 255) / 255}, ${(RAMP[2] & 255) / 255});
    float k = clamp(t, 0.0, 1.0) * 2.0;
    return k < 1.0 ? mix(c0, c1, k) : mix(c1, c2, k - 1.0);
  }

  void main() {
    vec3 pos = position + aJitter * uJitter;
    float reveal = clamp((uSweep - aReveal) * 7.0, 0.0, 1.0);
    vAlpha = reveal * uFade;

    vec3 base = mix(rampColor(aDepth), aLabelColor, uLabelMix);
    vec3 fogged = mix(base, ${`vec3(${FOG_COLOR.r}, ${FOG_COLOR.g}, ${FOG_COLOR.b})`}, aDepth * 0.14);
    vColor = fogged * aBrightness;

    vec4 mv = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uPointPx;
  }
`;

const FRAGMENT = /* glsl */ `
  varying float vAlpha;
  varying vec3 vColor;
  void main() {
    if (vAlpha <= 0.01) discard;
    vec2 c = gl_PointCoord - 0.5;
    if (dot(c, c) > 0.25) discard; // round points read as scan samples, not squares
    gl_FragColor = vec4(vColor, vAlpha);
  }
`;

export function createPointCloud(cloud, room, labelColorHex, seededRandom) {
  const n = cloud.length;
  const positions = new Float32Array(n * 3);
  const jitter = new Float32Array(n * 3);
  const labelColor = new Float32Array(n * 3);
  const reveal = new Float32Array(n);
  const depth = new Float32Array(n);
  const brightness = new Float32Array(n);

  const tmp = new THREE.Color();
  for (let i = 0; i < n; i++) {
    const p = cloud[i];
    positions[i * 3] = p.x;
    positions[i * 3 + 1] = p.y;
    positions[i * 3 + 2] = p.z;
    jitter[i * 3] = p.jx;
    jitter[i * 3 + 1] = p.jy;
    jitter[i * 3 + 2] = p.jz;
    tmp.set(labelColorHex[p.label]);
    labelColor[i * 3] = tmp.r;
    labelColor[i * 3 + 1] = tmp.g;
    labelColor[i * 3 + 2] = tmp.b;
    reveal[i] = p.x / room.w;
    depth[i] = (p.x + p.y) / (room.w + room.d);
    brightness[i] = 0.9 + seededRandom() * 0.2;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aJitter", new THREE.BufferAttribute(jitter, 3));
  geometry.setAttribute("aLabelColor", new THREE.BufferAttribute(labelColor, 3));
  geometry.setAttribute("aReveal", new THREE.BufferAttribute(reveal, 1));
  geometry.setAttribute("aDepth", new THREE.BufferAttribute(depth, 1));
  geometry.setAttribute("aBrightness", new THREE.BufferAttribute(brightness, 1));

  const material = new THREE.ShaderMaterial({
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    transparent: true,
    depthWrite: false,
    uniforms: {
      uSweep: { value: 0 },
      uJitter: { value: 1 },
      uLabelMix: { value: 0 },
      uFade: { value: 1 },
      uPointPx: { value: 3.2 },
    },
  });

  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;

  function setUniforms({ sweep, jitterMix, labelMix, fade, sizePx }) {
    const u = material.uniforms;
    u.uSweep.value = sweep;
    u.uJitter.value = jitterMix;
    u.uLabelMix.value = labelMix;
    u.uFade.value = fade;
    if (sizePx !== undefined) u.uPointPx.value = sizePx;
  }

  function dispose() {
    geometry.dispose();
    material.dispose();
  }

  return { points, setUniforms, dispose };
}
