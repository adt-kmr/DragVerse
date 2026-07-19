/* The light rig: procedural IBL (no HDR file to fetch) + one soft key light standing in
   for daylight through the door + one hemisphere fill. Kept deliberately low-contrast —
   "never theatrical" — matching both the mission and the project's understated doctrine. */

import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

export function createLighting(renderer, room) {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envTexture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();

  const key = new THREE.DirectionalLight(0xfff2df, 1.85);
  key.position.set(room.w * 0.35, -room.d * 0.6, room.h * 2.6);
  key.target.position.set(room.w * 0.5, room.d * 0.5, 0);
  key.castShadow = true;
  key.shadow.mapSize.set(1536, 1536);
  key.shadow.bias = -0.0015;
  key.shadow.normalBias = 0.02;
  const pad = 0.6;
  const cam = key.shadow.camera;
  cam.left = -room.w / 2 - pad;
  cam.right = room.w / 2 + pad;
  cam.top = room.d / 2 + pad;
  cam.bottom = -room.d / 2 - pad;
  cam.near = 0.1;
  cam.far = room.h * 6;
  cam.updateProjectionMatrix();

  // Fill kept lower relative to the key than a typical "soft" rig — more of the shading
  // comes from the directional light so surfaces read with a clear, legible gradient
  // instead of the flat/washed-out look a heavier ambient fill produces.
  const hemi = new THREE.HemisphereLight(0xeef0ea, 0x4a524d, 0.42);

  function addTo(scene) {
    scene.environment = envTexture;
    scene.add(key, key.target, hemi);
  }

  function dispose() {
    envTexture.dispose();
  }

  return { addTo, dispose, key, hemi };
}
