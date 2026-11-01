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

// A .glb being built: binary data appended (4-byte aligned) as buffer views, accessors over them, and the
// JSON's arrays (meshes, nodes, ...) filled by the caller; finish() packs the JSON and BIN chunks
export class GlbWriter {
  constructor() { this.parts = []; this.length = 0; this.views = []; this.accessors = []; }
  // bytes (a typed array) as a buffer view; its index
  view(bytes, { target, byteStride } = {}) {
    const u8 = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    this.views.push({ buffer: 0, byteOffset: this.length, byteLength: u8.length, ...(byteStride && { byteStride }), ...(target && { target }) });
    this.parts.push([this.length, u8]);
    this.length = Math.ceil((this.length + u8.length) / 4) * 4;
    return this.views.length - 1;
  }
  accessor(a) { this.accessors.push(a); return this.accessors.length - 1; }
  // a float32 accessor of `n` components (VEC3 positions: with their min and max, which glTF requires)
  floats(data, n, { target = ARRAY_BUFFER, minMax = false } = {}) {
    const a = { bufferView: this.view(Float32Array.from(data), { target }), componentType: FLOAT, count: data.length / n, type: TYPES[n] };
    if (minMax) {
      a.min = Array(n).fill(Infinity); a.max = Array(n).fill(-Infinity);
      for (let i = 0; i < data.length; i++) { const k = i % n, v = Math.fround(data[i]); if (v < a.min[k]) a.min[k] = v; if (v > a.max[k]) a.max[k] = v; }
      if (!data.length) { a.min.fill(0); a.max.fill(0); }
    }
    return this.accessor(a);
  }
  finish(json) {
    const bin = new Uint8Array(this.length);
    for (const [off, u8] of this.parts) bin.set(u8, off);
    json.buffers = [{ byteLength: bin.length }];
    json.bufferViews = this.views;
    if (this.accessors.length) json.accessors = this.accessors;  // glTF: no empty arrays
    const js = new TextEncoder().encode(JSON.stringify(json));
    const jsLen = Math.ceil(js.length / 4) * 4, out = new Uint8Array(12 + 8 + jsLen + 8 + bin.length), dv = new DataView(out.buffer);
    dv.setUint32(0, 0x46546c67, true); dv.setUint32(4, 2, true); dv.setUint32(8, out.length, true);
    dv.setUint32(12, jsLen, true); dv.setUint32(16, 0x4e4f534a, true);  // 'JSON'
    out.set(js, 20); out.fill(0x20, 20 + js.length, 20 + jsLen);
    dv.setUint32(20 + jsLen, bin.length, true); dv.setUint32(24 + jsLen, 0x004e4942, true);  // 'BIN\0'
    out.set(bin, 28 + jsLen);
    return out;
  }
}
const TYPES = { 1: 'SCALAR', 2: 'VEC2', 3: 'VEC3', 4: 'VEC4' };

// mesh: { positions: Float32Array 3V, faces: Uint32Array 3F, colors?: Uint8Array 3V, uvs?: Float32Array 2V }
// up: the world's up vector, turned to glTF's +Y (default: none, the coordinates as they are)
// texture: { bytes: Uint8Array (a PNG or JPEG file), mimeType } with uvs in [0, 1] (v down, as glTF's):
// the base color, on 16-bit uvs; the vertex colors are then left out (glTF multiplies them into it).
// voxel: the TSDF grid's voxel the mesh was made at (extras.gaussopt_voxel; meshVoxel)
export function writeGlb(mesh, { up = null, texture = null, voxel = null } = {}) {
  const w = new GlbWriter(), json = {
    asset: { version: '2.0', generator: 'gaussopt mesh.js' },
    extensionsUsed: ['KHR_mesh_quantization', 'KHR_materials_unlit'], extensionsRequired: ['KHR_mesh_quantization'],
    scene: 0, scenes: [{ nodes: [0] }], nodes: [], materials: [], meshes: [],
  };
  const { node } = addMesh(w, json, mesh, { up, texture });
  if (up || voxel) node.extras = { ...(up && { gaussopt_up: up }), ...(voxel && { gaussopt_voxel: voxel }) };
  return w.finish(json);
}

// The mesh into w and json (its mesh, material, texture, and a node, json.nodes' last): positions quantized to 16 bits
// (KHR_mesh_quantization: the node's matrix scales them back and turns `up` to +Y), colors as bytes or a texture.
// Returns { node, mesh: its index in json.meshes }
export function addMesh(w, json, { positions, faces, colors, uvs }, { up = null, texture = null } = {}) {
  const nv = positions.length / 3;
  if (texture && !uvs) throw new Error('writeGlb: a texture needs uvs');
  if (texture) colors = null;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], positions[3 * i + k]); hi[k] = Math.max(hi[k], positions[3 * i + k]); }
  const scale = lo.map((l, k) => Math.max(hi[k] - l, 1e-30) / 65535);
  const q = new Uint16Array(4 * nv), qmin = [65535, 65535, 65535], qmax = [0, 0, 0];  // 3 shorts + padding a vertex
  for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) {
    const v = Math.max(0, Math.min(65535, Math.round((positions[3 * i + k] - lo[k]) / scale[k])));
    q[4 * i + k] = v; qmin[k] = Math.min(qmin[k], v); qmax[k] = Math.max(qmax[k], v);
  }
  const attributes = { POSITION: w.accessor({ bufferView: w.view(q, { byteStride: 8, target: ARRAY_BUFFER }), componentType: UNSIGNED_SHORT, count: nv, type: 'VEC3', min: qmin, max: qmax }) };
  if (colors) {
    const c = new Uint8Array(4 * nv);
    for (let i = 0; i < nv; i++) { c[4 * i] = colors[3 * i]; c[4 * i + 1] = colors[3 * i + 1]; c[4 * i + 2] = colors[3 * i + 2]; c[4 * i + 3] = 255; }
    attributes.COLOR_0 = w.accessor({ bufferView: w.view(c, { byteStride: 4, target: ARRAY_BUFFER }), componentType: UNSIGNED_BYTE, normalized: true, count: nv, type: 'VEC4' });
  } else if (texture) {
    const u = new Uint16Array(2 * nv);
    for (let i = 0; i < 2 * nv; i++) u[i] = Math.max(0, Math.min(65535, Math.round(uvs[i] * 65535)));
    attributes.TEXCOORD_0 = w.accessor({ bufferView: w.view(u, { byteStride: 4, target: ARRAY_BUFFER }), componentType: UNSIGNED_SHORT, normalized: true, count: nv, type: 'VEC2' });
  }
  const wide = nv > 65535;
  const indices = w.accessor({ bufferView: w.view(wide ? Uint32Array.from(faces) : Uint16Array.from(faces), { target: ELEMENT_ARRAY_BUFFER }),
                               componentType: wide ? UNSIGNED_INT : UNSIGNED_SHORT, count: faces.length, type: 'SCALAR' });
  // node matrix (column-major): x_world = rot (lo + scale * q)
  const R = up ? rotationToY(up) : [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const t = [0, 1, 2].map(r => R[3 * r] * lo[0] + R[3 * r + 1] * lo[1] + R[3 * r + 2] * lo[2]);
  const matrix = [R[0] * scale[0], R[3] * scale[0], R[6] * scale[0], 0, R[1] * scale[1], R[4] * scale[1], R[7] * scale[1], 0,
                  R[2] * scale[2], R[5] * scale[2], R[8] * scale[2], 0, t[0], t[1], t[2], 1];
  const pbr = { baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 1 };
  if (texture) {
    const image = (json.images ??= []).push({ bufferView: w.view(texture.bytes), mimeType: texture.mimeType }) - 1;
    const sampler = (json.samplers ??= []).push({ magFilter: 9729, minFilter: 9987, wrapS: 33071, wrapT: 33071 }) - 1;  // linear, trilinear mipmaps, clamped
    pbr.baseColorTexture = { index: (json.textures ??= []).push({ source: image, sampler }) - 1 };
  }
  const material = json.materials.push({ pbrMetallicRoughness: pbr, doubleSided: true, extensions: { KHR_materials_unlit: {} } }) - 1;
  const meshIdx = json.meshes.push({ primitives: [{ attributes, indices, material }] }) - 1;
  const node = { mesh: meshIdx, matrix };
  json.nodes.push(node);
  return { node, mesh: meshIdx };
}

// The JSON and BIN chunks of a .glb: { json, bin: Uint8Array }
export function parseGlb(buffer) {
  const dv = new DataView(buffer);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a .glb (no glTF magic)');
  let off = 12, json = null, bin = null;
  while (off + 8 <= dv.byteLength) {
    const len = dv.getUint32(off, true), type = dv.getUint32(off + 4, true);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, off + 8, len)));
    else if (type === 0x004e4942) bin = new Uint8Array(buffer, off + 8, len);
    off += 8 + len;
  }
  if (!json) throw new Error('.glb without a JSON chunk');
  return { json, bin };
}

// Accessor i of a parsed .glb as { data: Float64Array (count x n; normalized integers scaled to [0, 1] / [-1, 1]), n }
export function readAccessor({ json, bin }, i) {
  const a = json.accessors[i], bv = json.bufferViews[a.bufferView], n = COMPS[a.type], sz = SIZE[a.componentType];
  const stride = bv.byteStride ?? n * sz, base = bin.byteOffset + (bv.byteOffset ?? 0) + (a.byteOffset ?? 0);
  const d = new DataView(bin.buffer), out = new Float64Array(a.count * n);
  const get = { 5120: (o) => d.getInt8(o), [UNSIGNED_BYTE]: (o) => d.getUint8(o), 5122: (o) => d.getInt16(o, true),
                [UNSIGNED_SHORT]: (o) => d.getUint16(o, true), [UNSIGNED_INT]: (o) => d.getUint32(o, true), [FLOAT]: (o) => d.getFloat32(o, true) }[a.componentType];
  const norm = a.normalized ? { 5120: 127, [UNSIGNED_BYTE]: 255, 5122: 32767, [UNSIGNED_SHORT]: 65535 }[a.componentType] : 1;
  for (let k = 0; k < a.count; k++) for (let c = 0; c < n; c++) out[k * n + c] = get(base + k * stride + c * sz) / norm;
  return { data: out, n };
}
// A buffer view's bytes (a view into the .glb's buffer)
export const viewBytes = ({ json, bin }, i) => { const bv = json.bufferViews[i]; return bin.subarray(bv.byteOffset ?? 0, (bv.byteOffset ?? 0) + bv.byteLength); };
const SIZE = { [UNSIGNED_BYTE]: 1, 5120: 1, [UNSIGNED_SHORT]: 2, 5122: 2, [UNSIGNED_INT]: 4, [FLOAT]: 4 };
const COMPS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

// The first mesh primitive of a .glb, in world coordinates (its node's matrix applied), turned back from
// glTF's +Y up to the frame writeGlb had (extras.gaussopt_up) unless gltfFrame:
// { positions: Float32Array, faces: Uint32Array, colors: Uint8Array | null, uvs: Float32Array | null,
// texture: { bytes, mimeType } | null (the base color's image), voxel: extras.gaussopt_voxel | null }. Positions: float or
// (KHR_mesh_quantization) integer shorts / bytes, normalized or not; colors: float or normalized bytes /
// shorts, VEC3 or VEC4; indices of any width (or none: a triangle list in vertex order). In a scene .glb (scene_glb.js)
// the first node with triangles, in the scene's own frame (its parent's leveling not applied).
export function readGlb(buffer, { gltfFrame = false } = {}) {
  const g = parseGlb(buffer), { json } = g;
  if (!g.bin) throw new Error('.glb without a JSON or BIN chunk');
  const node = json.nodes?.find(n => n.mesh !== undefined && (json.meshes[n.mesh].primitives[0].mode ?? 4) === 4) ?? { mesh: 0 };
  const m = readMeshNode(g, node);
  const up = !gltfFrame && node.extras?.gaussopt_up;
  if (up) {
    const Rt = rotationToY(up), P = m.positions;  // applied transposed: glTF frame -> writeGlb's
    for (let i = 0; i < P.length; i += 3) {
      const [x, y, z] = [P[i], P[i + 1], P[i + 2]];
      for (let r = 0; r < 3; r++) P[i + r] = Rt[r] * x + Rt[3 + r] * y + Rt[6 + r] * z;
    }
  }
  return m;
}

// A node's triangle mesh (its first primitive), its matrix applied: the fields of readGlb
export function readMeshNode(g, node) {
  const { json, bin } = g, prim = json.meshes[node.mesh].primitives[0];
  if ((prim.mode ?? 4) !== 4) throw new Error('.glb: only triangle lists are read');
  const pos = readAccessor(g, prim.attributes.POSITION).data, nv = pos.length / 3;
  const M = node.matrix ?? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  if (node.translation || node.rotation || node.scale) throw new Error('.glb: node TRS not read (matrix only)');
  const positions = new Float32Array(3 * nv);
  for (let i = 0; i < nv; i++) {
    const [x, y, z] = [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]];
    for (let r = 0; r < 3; r++) positions[3 * i + r] = M[r] * x + M[4 + r] * y + M[8 + r] * z + M[12 + r];
  }
  let colors = null;
  if (prim.attributes.COLOR_0 !== undefined) {
    const c = readAccessor(g, prim.attributes.COLOR_0);
    colors = new Uint8Array(3 * nv);
    for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) colors[3 * i + k] = Math.round(255 * Math.min(1, Math.max(0, c.data[c.n * i + k])));
  }
  const faces = prim.indices !== undefined ? Uint32Array.from(readAccessor(g, prim.indices).data) : Uint32Array.from({ length: nv }, (_, i) => i);
  let uvs = null, texture = null;
  const tex = json.materials?.[prim.material ?? -1]?.pbrMetallicRoughness?.baseColorTexture;
  if (tex && prim.attributes[`TEXCOORD_${tex.texCoord ?? 0}`] !== undefined) {
    uvs = Float32Array.from(readAccessor(g, prim.attributes[`TEXCOORD_${tex.texCoord ?? 0}`]).data);
    const im = json.images?.[json.textures?.[tex.index]?.source];
    if (im?.bufferView === undefined) throw new Error('.glb: only textures in the .glb itself are read');
    texture = { bytes: viewBytes(g, im.bufferView).slice(), mimeType: im.mimeType };
  }
  return { positions, faces, colors, uvs, texture, voxel: node.extras?.gaussopt_voxel ?? null };
}

// The TSDF voxel a Make mesh was made at: its own (writeGlb's voxel), else its bounds' longest side / 383,
// the grid's at the default mesh_res 384 (the grid spans about the mesh's bounds)
export function meshVoxel({ positions: P, voxel = null }) {
  if (voxel) return voxel;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < P.length; i++) { const k = i % 3; if (P[i] < lo[k]) lo[k] = P[i]; if (P[i] > hi[k]) hi[k] = P[i]; }
  return P.length ? Math.max(...hi.map((h, k) => h - lo[k])) / 383 : 0;
}
