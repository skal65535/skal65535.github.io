// glb.js: a mesh written as .glb (16-bit positions, byte colors) reads back within the quantization
// step, with its faces and colors, in both index widths, and with an up vector turned to +Y.
import { writeGlb, readGlb } from '../src/glb.js';
import { assert } from './util.js';

function mesh(nv) {
  let s = 11;
  const r = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const positions = new Float32Array(3 * nv).map((_, i) => (i % 3 === 1 ? 3 : 10) * r() - 2);
  const colors = new Uint8Array(3 * nv).map(() => Math.floor(256 * r()));
  const faces = new Uint32Array(3 * (nv - 2)).map((_, i) => (Math.floor(i / 3) + (i % 3)) % nv);
  return { positions, colors, faces };
}

for (const nv of [1000, 70000]) {
  Deno.test(`glb: ${nv} vertices (${nv > 65535 ? 32 : 16}-bit indices) round trip`, () => {
    const m = mesh(nv), bytes = writeGlb(m), back = readGlb(bytes.buffer);
    assert(bytes.length % 4 === 0 && new DataView(bytes.buffer).getUint32(8, true) === bytes.length, 'glb length');
    assert(back.positions.length === m.positions.length && back.faces.length === m.faces.length, 'counts');
    let worst = 0;
    for (let i = 0; i < m.positions.length; i++) worst = Math.max(worst, Math.abs(back.positions[i] - m.positions[i]));
    assert(worst <= 0.5 * 10 / 65535 * 1.01, `position error ${worst}`);  // half a step of the 10-wide axis
    for (let i = 0; i < m.faces.length; i++) assert(back.faces[i] === m.faces[i], `face index ${i}`);
    for (let i = 0; i < m.colors.length; i++) assert(back.colors[i] === m.colors[i], `color ${i}`);
    const js = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + new DataView(bytes.buffer).getUint32(12, true))));
    assert(js.extensionsRequired.includes('KHR_mesh_quantization') && js.accessors[0].componentType === 5123, 'quantized positions');
    assert(js.accessors.at(-1).componentType === (nv > 65535 ? 5125 : 5123), 'index width');
  });
}

Deno.test('glb: up = (0, -1, 0) (OpenCV-style y down) comes out with y flipped; (0, 0, 1) turns z to +Y; read back in the original frame', () => {
  const m = { positions: new Float32Array([0, 0, 0, 1, 2, 3, -1, 0.5, 2]), faces: new Uint32Array([0, 1, 2]) };
  const flip = readGlb(writeGlb(m, { up: [0, -1, 0] }).buffer, { gltfFrame: true }).positions;
  for (let i = 0; i < 3; i++) assert(Math.abs(flip[3 * i + 1] + m.positions[3 * i + 1]) < 1e-3, `y of ${i}: ${flip[3 * i + 1]}`);
  const zup = readGlb(writeGlb(m, { up: [0, 0, 1] }).buffer, { gltfFrame: true }).positions;
  for (let i = 0; i < 3; i++) assert(Math.abs(zup[3 * i + 1] - m.positions[3 * i + 2]) < 1e-3, `z -> y of ${i}: ${zup[3 * i + 1]}`);
  const up = [0.3, -0.9, 0.2], back = readGlb(writeGlb(m, { up }).buffer).positions;
  for (let i = 0; i < 9; i++) assert(Math.abs(back[i] - m.positions[i]) < 1e-3, `original frame ${i}: ${back[i]} vs ${m.positions[i]}`);
  assert(readGlb(writeGlb(m).buffer).colors === null, 'no colors written, none read');
});

Deno.test('glb: a textured mesh keeps its uvs (16 bits) and image, without vertex colors', () => {
  const m = mesh(500), uvs = new Float32Array(1000).map((_, i) => ((i * 7919) % 1000) / 999);
  const image = { bytes: Uint8Array.from({ length: 333 }, (_, i) => (i * 31) & 255), mimeType: 'image/jpeg' };
  const bytes = writeGlb({ ...m, uvs }, { texture: image }), back = readGlb(bytes.buffer);
  assert(back.colors === null, 'vertex colors left out');
  let worst = 0;
  for (let i = 0; i < uvs.length; i++) worst = Math.max(worst, Math.abs(back.uvs[i] - uvs[i]));
  assert(worst <= 0.5 / 65535 * 1.01, `uv error ${worst}`);
  assert(back.texture.mimeType === 'image/jpeg' && back.texture.bytes.length === 333 && back.texture.bytes.every((b, i) => b === image.bytes[i]), 'image bytes');
  for (let i = 0; i < m.faces.length; i++) assert(back.faces[i] === m.faces[i], `face index ${i}`);
  const js = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + new DataView(bytes.buffer).getUint32(12, true))));
  const prim = js.meshes[0].primitives[0];
  assert(prim.attributes.TEXCOORD_0 !== undefined && prim.attributes.COLOR_0 === undefined && js.materials[0].pbrMetallicRoughness.baseColorTexture.index === 0, 'attributes, material');
  assert(readGlb(writeGlb(m).buffer).uvs === null && readGlb(writeGlb(m).buffer).texture === null, 'untextured: no uvs, no texture');
});
