// glb.js — Triangle meshes as binary glTF 2.0 (.glb), DESIGN.md §9 E5: positions quantized to 16 bits
// (KHR_mesh_quantization: unsigned shorts, the node's transform scales them back), vertex colors as
// bytes, or a texture (an image in the .glb, 16-bit uvs), indices as 16 or 32 bits, an unlit material
// (KHR_materials_unlit, optional) so the colors show as baked. Readable by any glTF viewer (Blender,
// three.js); readGlb reads these back (and plain float meshes) for the page's viewers.
"use strict";

const ARRAY_BUFFER = 34962, ELEMENT_ARRAY_BUFFER = 34963;
const UNSIGNED_BYTE = 5121, UNSIGNED_SHORT = 5123, UNSIGNED_INT = 5125, FLOAT = 5126;

// The rotation (row-major 3x3) taking unit vector `up` to +Y by the shortest arc
export function rotationToY(up) {
  const u = up.map(x => x / Math.hypot(...up)), c = u[1];
  const ax = [-u[2], 0, u[0]], s = Math.hypot(...ax);  // up x Y
  if (s < 1e-9) return c > 0 ? [1, 0, 0, 0, 1, 0, 0, 0, 1] : [1, 0, 0, 0, -1, 0, 0, 0, -1];
  const [x, y, z] = ax.map(v => v / s), C = 1 - c;
  return [c + x * x * C, x * y * C - z * s, x * z * C + y * s,
          y * x * C + z * s, c + y * y * C, y * z * C - x * s,
          z * x * C - y * s, z * y * C + x * s, c + z * z * C];
}

// mesh: { positions: Float32Array 3V, faces: Uint32Array 3F, colors?: Uint8Array 3V, uvs?: Float32Array 2V }
// up: the world's up vector, turned to glTF's +Y (default: none, the coordinates as they are)
// texture: { bytes: Uint8Array (a PNG or JPEG file), mimeType } with uvs in [0, 1] (v down, as glTF's):
// the base color, on 16-bit uvs; the vertex colors are then left out (glTF multiplies them into it).
export function writeGlb({ positions, faces, colors, uvs }, { up = null, texture = null } = {}) {
  const nv = positions.length / 3;
  if (texture && !uvs) throw new Error('writeGlb: a texture needs uvs');
  if (texture) colors = null;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], positions[3 * i + k]); hi[k] = Math.max(hi[k], positions[3 * i + k]); }
  const scale = lo.map((l, k) => Math.max(hi[k] - l, 1e-30) / 65535);
  // layout: positions (8 bytes per vertex: 3 shorts + padding), colors (4 bytes) or uvs (2 shorts), indices, image
  const wide = nv > 65535, idxBytes = (wide ? 4 : 2) * faces.length;
  const posOff = 0, attOff = 8 * nv, idxOff = attOff + (colors || texture ? 4 * nv : 0), imgOff = Math.ceil((idxOff + idxBytes) / 4) * 4;
  const binLen = Math.ceil((imgOff + (texture ? texture.bytes.length : 0)) / 4) * 4, bin = new ArrayBuffer(binLen);
  const q = new Uint16Array(bin, posOff, 4 * nv), qmin = [65535, 65535, 65535], qmax = [0, 0, 0];
  for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) {
    const v = Math.max(0, Math.min(65535, Math.round((positions[3 * i + k] - lo[k]) / scale[k])));
    q[4 * i + k] = v; qmin[k] = Math.min(qmin[k], v); qmax[k] = Math.max(qmax[k], v);
  }
  if (colors) {
    const c = new Uint8Array(bin, attOff, 4 * nv);
    for (let i = 0; i < nv; i++) { c[4 * i] = colors[3 * i]; c[4 * i + 1] = colors[3 * i + 1]; c[4 * i + 2] = colors[3 * i + 2]; c[4 * i + 3] = 255; }
  }
  if (texture) {
    const u = new Uint16Array(bin, attOff, 2 * nv);
    for (let i = 0; i < 2 * nv; i++) u[i] = Math.max(0, Math.min(65535, Math.round(uvs[i] * 65535)));
    new Uint8Array(bin, imgOff, texture.bytes.length).set(texture.bytes);
  }
  (wide ? new Uint32Array(bin, idxOff, faces.length) : new Uint16Array(bin, idxOff, faces.length)).set(faces);
  // node matrix (column-major): x_world = rot (lo + scale * q)
  const R = up ? rotationToY(up) : [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const t = [0, 1, 2].map(r => R[3 * r] * lo[0] + R[3 * r + 1] * lo[1] + R[3 * r + 2] * lo[2]);
  const matrix = [R[0] * scale[0], R[3] * scale[0], R[6] * scale[0], 0, R[1] * scale[1], R[4] * scale[1], R[7] * scale[1], 0,
                  R[2] * scale[2], R[5] * scale[2], R[8] * scale[2], 0, t[0], t[1], t[2], 1];
  const views = [{ buffer: 0, byteOffset: posOff, byteLength: 8 * nv, byteStride: 8, target: ARRAY_BUFFER }];
  const accessors = [{ bufferView: 0, componentType: UNSIGNED_SHORT, count: nv, type: 'VEC3', min: qmin, max: qmax }];
  const attributes = { POSITION: 0 };
  if (colors || texture) {
    views.push({ buffer: 0, byteOffset: attOff, byteLength: 4 * nv, byteStride: 4, target: ARRAY_BUFFER });
    accessors.push(colors ? { bufferView: 1, componentType: UNSIGNED_BYTE, normalized: true, count: nv, type: 'VEC4' }
                          : { bufferView: 1, componentType: UNSIGNED_SHORT, normalized: true, count: nv, type: 'VEC2' });
    attributes[colors ? 'COLOR_0' : 'TEXCOORD_0'] = 1;
  }
  views.push({ buffer: 0, byteOffset: idxOff, byteLength: idxBytes, target: ELEMENT_ARRAY_BUFFER });
  accessors.push({ bufferView: views.length - 1, componentType: wide ? UNSIGNED_INT : UNSIGNED_SHORT, count: faces.length, type: 'SCALAR' });
  const pbr = { baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 1 };
  const json = {
    asset: { version: '2.0', generator: 'gaussopt mesh.js' },
    extensionsUsed: ['KHR_mesh_quantization', 'KHR_materials_unlit'], extensionsRequired: ['KHR_mesh_quantization'],
    scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0, matrix, ...(up && { extras: { gaussopt_up: up } }) }],
    materials: [{ pbrMetallicRoughness: texture ? { ...pbr, baseColorTexture: { index: 0 } } : pbr, doubleSided: true, extensions: { KHR_materials_unlit: {} } }],
    meshes: [{ primitives: [{ attributes, indices: accessors.length - 1, material: 0 }] }],
    buffers: [{ byteLength: binLen }], bufferViews: views, accessors,
  };
  if (texture) {
    views.push({ buffer: 0, byteOffset: imgOff, byteLength: texture.bytes.length });
    json.images = [{ bufferView: views.length - 1, mimeType: texture.mimeType }];
    json.samplers = [{ magFilter: 9729, minFilter: 9987, wrapS: 33071, wrapT: 33071 }];  // linear, trilinear mipmaps, clamped
    json.textures = [{ source: 0, sampler: 0 }];
  }
  let js = new TextEncoder().encode(JSON.stringify(json));
  const jsLen = Math.ceil(js.length / 4) * 4, out = new Uint8Array(12 + 8 + jsLen + 8 + binLen), dv = new DataView(out.buffer);
  dv.setUint32(0, 0x46546c67, true); dv.setUint32(4, 2, true); dv.setUint32(8, out.length, true);
  dv.setUint32(12, jsLen, true); dv.setUint32(16, 0x4e4f534a, true);  // 'JSON'
  out.set(js, 20); out.fill(0x20, 20 + js.length, 20 + jsLen);
  dv.setUint32(20 + jsLen, binLen, true); dv.setUint32(24 + jsLen, 0x004e4942, true);  // 'BIN\0'
  out.set(new Uint8Array(bin), 28 + jsLen);
  return out;
}

// The first mesh primitive of a .glb, in world coordinates (its node's matrix applied), turned back from
// glTF's +Y up to the frame writeGlb had (extras.gaussopt_up) unless gltfFrame:
// { positions: Float32Array, faces: Uint32Array, colors: Uint8Array | null, uvs: Float32Array | null,
// texture: { bytes, mimeType } | null (the base color's image) }. Positions: float or
// (KHR_mesh_quantization) integer shorts / bytes, normalized or not; colors: float or normalized bytes /
// shorts, VEC3 or VEC4; indices of any width (or none: a triangle list in vertex order).
export function readGlb(buffer, { gltfFrame = false } = {}) {
  const dv = new DataView(buffer);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a .glb (no glTF magic)');
  let off = 12, json = null, bin = null;
  while (off + 8 <= dv.byteLength) {
    const len = dv.getUint32(off, true), type = dv.getUint32(off + 4, true);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, off + 8, len)));
    else if (type === 0x004e4942) bin = new Uint8Array(buffer, off + 8, len);
    off += 8 + len;
  }
  if (!json || !bin) throw new Error('.glb without a JSON or BIN chunk');
  const node = json.nodes?.find(n => n.mesh !== undefined) ?? { mesh: 0 };
  const prim = json.meshes[node.mesh].primitives[0];
  if ((prim.mode ?? 4) !== 4) throw new Error('.glb: only triangle lists are read');
  const SIZE = { [UNSIGNED_BYTE]: 1, 5120: 1, [UNSIGNED_SHORT]: 2, 5122: 2, [UNSIGNED_INT]: 4, [FLOAT]: 4 };
  const COMPS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
  const read = (i) => {  // accessor i as Float64Array of count x comps (normalized ints scaled to [0, 1] / [-1, 1])
    const a = json.accessors[i], bv = json.bufferViews[a.bufferView], n = COMPS[a.type], sz = SIZE[a.componentType];
    const stride = bv.byteStride ?? n * sz, base = bin.byteOffset + (bv.byteOffset ?? 0) + (a.byteOffset ?? 0);
    const d = new DataView(bin.buffer), out = new Float64Array(a.count * n);
    const get = { 5120: (o) => d.getInt8(o), [UNSIGNED_BYTE]: (o) => d.getUint8(o), 5122: (o) => d.getInt16(o, true),
                  [UNSIGNED_SHORT]: (o) => d.getUint16(o, true), [UNSIGNED_INT]: (o) => d.getUint32(o, true), [FLOAT]: (o) => d.getFloat32(o, true) }[a.componentType];
    const norm = a.normalized ? { 5120: 127, [UNSIGNED_BYTE]: 255, 5122: 32767, [UNSIGNED_SHORT]: 65535 }[a.componentType] : 1;
    for (let k = 0; k < a.count; k++) for (let c = 0; c < n; c++) out[k * n + c] = get(base + k * stride + c * sz) / norm;
    return { data: out, n };
  };
  const pos = read(prim.attributes.POSITION).data, nv = pos.length / 3;
  const M = node.matrix ?? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  if (node.translation || node.rotation || node.scale) throw new Error('.glb: node TRS not read (matrix only)');
  const positions = new Float32Array(3 * nv), up = !gltfFrame && node.extras?.gaussopt_up;
  const Rt = up ? rotationToY(up) : null;  // applied transposed: glTF frame -> writeGlb's
  for (let i = 0; i < nv; i++) {
    const [x, y, z] = [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]];
    const w = [0, 1, 2].map(r => M[r] * x + M[4 + r] * y + M[8 + r] * z + M[12 + r]);
    for (let r = 0; r < 3; r++) positions[3 * i + r] = Rt ? Rt[r] * w[0] + Rt[3 + r] * w[1] + Rt[6 + r] * w[2] : w[r];
  }
  let colors = null;
  if (prim.attributes.COLOR_0 !== undefined) {
    const c = read(prim.attributes.COLOR_0);
    colors = new Uint8Array(3 * nv);
    for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) colors[3 * i + k] = Math.round(255 * Math.min(1, Math.max(0, c.data[c.n * i + k])));
  }
  const faces = prim.indices !== undefined ? Uint32Array.from(read(prim.indices).data) : Uint32Array.from({ length: nv }, (_, i) => i);
  let uvs = null, texture = null;
  const tex = json.materials?.[prim.material ?? -1]?.pbrMetallicRoughness?.baseColorTexture;
  if (tex && prim.attributes[`TEXCOORD_${tex.texCoord ?? 0}`] !== undefined) {
    uvs = Float32Array.from(read(prim.attributes[`TEXCOORD_${tex.texCoord ?? 0}`]).data);
    const im = json.images?.[json.textures?.[tex.index]?.source], bv = im && json.bufferViews[im.bufferView];
    if (!bv) throw new Error('.glb: only textures in the .glb itself are read');
    texture = { bytes: bin.slice(bv.byteOffset ?? 0, (bv.byteOffset ?? 0) + bv.byteLength), mimeType: im.mimeType };
  }
  return { positions, faces, colors, uvs, texture };
}
