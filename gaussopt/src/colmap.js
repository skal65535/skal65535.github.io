// colmap.js — COLMAP binary parsers (cameras.bin, images.bin, points3D.bin), dataset loaders,
//             and 3DGS .ply / .splat checkpoint import & export (.ply may embed the Loose Octree).
"use strict";

import { sigmoid, logit, quatToMat3, makeViewMatrix, makeProjectionMatrix } from './math.js';

export const SH_C0 = 0.28209479177387814;
export const SPLAT_FLOATS = 16; // 64 bytes/splat:
// [0..3]:   x, y, z, logit_opacity
// [4..7]:   log_sx, log_sy, log_sz, leaf_id
// [8..11]:  qw, qx, qy, qz
// [12..15]: f_dc_0, f_dc_1, f_dc_2, flags

export const rgbToSh0  = (r01) => (r01 - 0.5) / SH_C0;
export const sh0ToRgb01 = (dc) => Math.min(1, Math.max(0, 0.5 + SH_C0 * dc));

// Fill splat i of `params` (quaternion normalized; leaf_id and flags left at 0)
function setSplat(params, i, x, y, z, op, sx, sy, sz, qw, qx, qy, qz, dc0, dc1, dc2) {
  const qn = Math.hypot(qw, qx, qy, qz) || 1;
  params.set([x, y, z, op, sx, sy, sz, 0, qw / qn, qx / qn, qy / qn, qz / qn, dc0, dc1, dc2, 0], i * SPLAT_FLOATS);
}

// ── COLMAP binary parsers ────────────────────────────────────────────────────

// model_id -> [name, #params]; params are (f | fx, fy), cx, cy, distortion... (distortion ignored:
// images are expected to be undistorted already)
const CAMERA_MODELS = {
  0: ['SIMPLE_PINHOLE', 3],  // f, cx, cy
  1: ['PINHOLE',        4],  // fx, fy, cx, cy
  2: ['SIMPLE_RADIAL',  4],  // f, cx, cy, k
  3: ['RADIAL',         5],  // f, cx, cy, k1, k2
  4: ['OPENCV',         8],  // fx, fy, cx, cy, k1, k2, p1, p2
};
const u64 = (dv, off) => Number(dv.getBigUint64(off, true));

export function parseCamerasBin(buffer) {
  const dv = new DataView(buffer);
  const cameras = new Map();
  let off = 8;
  for (let i = 0, n = u64(dv, 0); i < n; i++) {
    const id      = dv.getInt32(off, true);
    const modelId = dv.getInt32(off + 4, true);
    const width   = u64(dv, off + 8);
    const height  = u64(dv, off + 16);
    off += 24;
    const info = CAMERA_MODELS[modelId];
    if (!info) throw new Error(`Unsupported COLMAP camera model_id=${modelId}`);
    const p = [];
    for (let k = 0; k < info[1]; k++, off += 8) p.push(dv.getFloat64(off, true));
    const [fx, fy, cx, cy] = (modelId === 1 || modelId === 4) ? p : [p[0], p[0], p[1], p[2]];
    cameras.set(id, { id, model: info[0], width, height, fx, fy, cx, cy });
  }
  return cameras;
}

// Images sorted by name. R (row-major) / t are world -> camera (x_cam = R x + t).
// obs2D = flat [u, v, point3D_id] of the observations that have a 3D point.
export function parseImagesBin(buffer, cameras) {
  const dv = new DataView(buffer);
  const u8 = new Uint8Array(buffer);
  const decoder = new TextDecoder();
  const images = [];
  let off = 8;
  for (let i = 0, n = u64(dv, 0); i < n; i++) {
    const id = dv.getInt32(off, true);
    const [qw, qx, qy, qz, tx, ty, tz] = Array.from({ length: 7 }, (_, k) => dv.getFloat64(off + 4 + 8 * k, true));
    const cameraId = dv.getInt32(off + 60, true);
    off += 64;

    let end = off;
    while (end < u8.length && u8[end] !== 0) end++;
    const name = decoder.decode(u8.subarray(off, end));
    off = end + 1;

    const numPoints2D = u64(dv, off); off += 8;
    const obs2D = [];
    for (let k = 0; k < numPoints2D; k++, off += 24) {
      const pid = Number(dv.getBigInt64(off + 16, true));  // -1 = no 3D point
      if (pid >= 0) obs2D.push(dv.getFloat64(off, true), dv.getFloat64(off + 8, true), pid);
    }

    const R = quatToMat3([qw, qx, qy, qz]);
    const t = [tx, ty, tz];
    const center = [  // C = -R^T t
      -(R[0] * tx + R[3] * ty + R[6] * tz),
      -(R[1] * tx + R[4] * ty + R[7] * tz),
      -(R[2] * tx + R[5] * ty + R[8] * tz),
    ];
    const forward = [R[6], R[7], R[8]];  // camera +Z in world = 3rd row of R
    const cam = cameras.get(cameraId) || cameras.values().next().value;
    images.push({
      id, name, cameraId, camera: cam,
      q: [qw, qx, qy, qz], t, R, center, forward,
      view: makeViewMatrix(R, t),
      proj: makeProjectionMatrix(cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height),
      obs2D: new Float64Array(obs2D),
      zMin: 0.2, zMax: 50.0,  // refined by computeCameraDepthStats()
      isValidation: false,
    });
  }
  images.sort((a, b) => a.name.localeCompare(b.name));
  return images;
}

export function parsePoints3DBin(buffer) {
  const dv = new DataView(buffer);
  const count = u64(dv, 0);
  const positions = new Float32Array(count * 3);
  const colors    = new Uint8Array(count * 3);
  const errors    = new Float32Array(count);
  const pointIndexById = new Map();
  let off = 8;
  for (let i = 0; i < count; i++) {
    pointIndexById.set(u64(dv, off), i);
    for (let k = 0; k < 3; k++) {
      positions[3 * i + k] = dv.getFloat64(off + 8 + 8 * k, true);
      colors[3 * i + k] = dv.getUint8(off + 32 + k);
    }
    errors[i] = dv.getFloat64(off + 35, true);
    const trackLen = u64(dv, off + 43);
    off += 51 + trackLen * 8;  // skip track: [image_id (i32), point2D_idx (i32)]
  }
  return { count, positions, colors, errors, pointIndexById };
}

// Per-camera 5th / 95th percentile depth of its observed points (zMin feeds the near-camera
// floater carve, DESIGN §5.2) + mean COLMAP reprojection error (sanity check).
export function computeCameraDepthStats(images, points3D) {
  const { positions, pointIndexById } = points3D;
  let totalReprojErr = 0, totalObs = 0;
  for (const img of images) {
    const depths = [];
    const { R, t, camera, obs2D } = img;
    for (let k = 0; k < obs2D.length; k += 3) {
      const idx = pointIndexById.get(obs2D[k + 2]);
      if (idx === undefined) continue;
      const X = positions[3 * idx], Y = positions[3 * idx + 1], Z = positions[3 * idx + 2];
      const xc = R[0] * X + R[1] * Y + R[2] * Z + t[0];
      const yc = R[3] * X + R[4] * Y + R[5] * Z + t[1];
      const zc = R[6] * X + R[7] * Y + R[8] * Z + t[2];
      if (zc <= 0.01) continue;
      depths.push(zc);
      totalReprojErr += Math.hypot(camera.fx * (xc / zc) + camera.cx - obs2D[k],
                                   camera.fy * (yc / zc) + camera.cy - obs2D[k + 1]);
      totalObs++;
    }
    if (depths.length > 0) {
      depths.sort((a, b) => a - b);
      img.zMin = depths[Math.floor(depths.length * 0.05)];
      img.zMax = depths[Math.floor(depths.length * 0.95)];
    }
  }
  return { meanReprojError: totalObs > 0 ? totalReprojErr / totalObs : 0, totalObs };
}

// ── .ply / .splat checkpoints ────────────────────────────────────────────────

// PLY scalar type -> [bytes, DataView getter]
const PLY_TYPES = {
  char:  [1, 'getInt8'],    int8:    [1, 'getInt8'],
  uchar: [1, 'getUint8'],   uint8:   [1, 'getUint8'],
  short: [2, 'getInt16'],   int16:   [2, 'getInt16'],
  ushort: [2, 'getUint16'], uint16:  [2, 'getUint16'],
  int:   [4, 'getInt32'],   int32:   [4, 'getInt32'],
  uint:  [4, 'getUint32'],  uint32:  [4, 'getUint32'],
  float: [4, 'getFloat32'], float32: [4, 'getFloat32'],
  double: [8, 'getFloat64'], float64: [8, 'getFloat64'],
};
// Exported vertex layout: PLY property -> SPLAT_FLOATS slot (opacity logit, scales log)
const PLY_VERTEX_PROPS = [
  ['x', 0], ['y', 1], ['z', 2], ['f_dc_0', 12], ['f_dc_1', 13], ['f_dc_2', 14], ['opacity', 3],
  ['scale_0', 4], ['scale_1', 5], ['scale_2', 6], ['rot_0', 8], ['rot_1', 9], ['rot_2', 10], ['rot_3', 11],
];
// `element octree_node` layout (32 bytes): [PLY property, type, LooseOctree.serializeNodes() field]
const PLY_OCTREE_PROPS = [
  ['cx', 'float', 'cx'], ['cy', 'float', 'cy'], ['cz', 'float', 'cz'], ['half_size', 'float', 'halfSize'],
  ['is_leaf', 'int', 'isLeaf'], ['first_child_or_start', 'int', 'firstChildOrStart'],
  ['count_or_depth', 'int', 'countOrDepth'], ['state', 'int', 'state'],
];
const OCTREE_NODE_BYTES = 4 * PLY_OCTREE_PROPS.length;

// Binary little-endian .ply with arbitrary vertex property order (e.g. Brush's alphabetical one).
// Missing fields get defaults; without f_dc_*, red/green/blue (0..255) are used.
// Returns { count, params, octreeData: { meta, nodes } | null }.
export function parsePlyBuffer(buffer) {
  const u8 = new Uint8Array(buffer);
  const decoder = new TextDecoder();
  let pos = 0;
  const readLine = () => {
    let end = pos;
    while (end < u8.length && u8[end] !== 10) end++;
    const line = decoder.decode(u8.subarray(pos, end)).trim();
    pos = end + 1;
    return line;
  };
  if (readLine() !== 'ply') throw new Error('Invalid PLY file (missing "ply" magic header)');

  let numVertices = 0, numOctreeNodes = 0, octreeMeta = null, format = '', headerEnded = false;
  let element = '';
  const prop = {};  // vertex property name -> { offset, get }
  let stride = 0;
  while (pos < u8.length) {
    const parts = readLine().split(/\s+/);
    if (parts[0] === 'end_header') { headerEnded = true; break; }
    if (parts[0] === 'format') {
      format = parts[1] || '';
    } else if (parts[0] === 'comment' && parts[1] === 'gaussopt_octree_v1') {
      octreeMeta = { beta: 0.20, maxLeafSize: 256, maxDepth: 8, sceneCenter: [0, 0, 0], sceneRadius: 2.0 };
      for (const kv of parts.slice(2)) {
        const [k, v] = kv.split('=');
        const num = parseFloat(v);
        if (k === 'beta') octreeMeta.beta = num;
        else if (k === 'max_leaf') octreeMeta.maxLeafSize = num | 0;
        else if (k === 'radius') octreeMeta.sceneRadius = num;
        else if (k === 'cx' || k === 'cy' || k === 'cz') octreeMeta.sceneCenter['xyz'.indexOf(k[1])] = num;
      }
    } else if (parts[0] === 'element') {
      element = parts[1];
      if (element === 'vertex') numVertices = parseInt(parts[2], 10);
      else if (element === 'octree_node') numOctreeNodes = parseInt(parts[2], 10);
    } else if (parts[0] === 'property' && element === 'vertex') {
      const type = PLY_TYPES[parts[1]];
      if (!type) throw new Error(`Unsupported PLY vertex property type "${parts[1]}"`);
      prop[parts[2]] = { offset: stride, get: type[1] };
      stride += type[0];
    }
  }
  if (!headerEnded) throw new Error('Invalid PLY file (truncated header, missing "end_header")');
  if (format && format !== 'binary_little_endian') {
    throw new Error(`Unsupported PLY format "${format}" (expected binary_little_endian)`);
  }
  if (!(numVertices > 0)) throw new Error(`Invalid PLY vertex count (${numVertices})`);
  if (!prop.x || !prop.y || !prop.z) throw new Error('Invalid PLY file (missing vertex x/y/z properties)');
  const vertexBytes = numVertices * stride;
  if (pos + vertexBytes > buffer.byteLength) {
    throw new Error(`Truncated PLY file (expected ${pos + vertexBytes} bytes, got ${buffer.byteLength})`);
  }

  const dv = new DataView(buffer, pos);
  const params = new Float32Array(numVertices * SPLAT_FLOATS);
  const hasShDc = 'f_dc_0' in prop;
  for (let i = 0; i < numVertices; i++) {
    const row = i * stride;
    const get = (name, def = 0) => {
      const p = prop[name];
      return p ? dv[p.get](row + p.offset, true) : def;
    };
    const dc = hasShDc ? [get('f_dc_0'), get('f_dc_1'), get('f_dc_2')]
                       : [get('red', 128), get('green', 128), get('blue', 128)].map(c => rgbToSh0(c / 255));
    setSplat(params, i, get('x'), get('y'), get('z'), get('opacity', logit(0.5)),
             get('scale_0', -4), get('scale_1', -4), get('scale_2', -4),
             get('rot_0', 1), get('rot_1'), get('rot_2'), get('rot_3'), ...dc);
  }

  // Embedded octree (fixed layout, right after the vertices). Leaves get sequential leaf_ids.
  let octreeData = null;
  if (numOctreeNodes > 0 && octreeMeta && pos + vertexBytes + numOctreeNodes * OCTREE_NODE_BYTES <= buffer.byteLength) {
    const nodes = new Array(numOctreeNodes);
    let leafId = 0;
    for (let i = 0; i < numOctreeNodes; i++) {
      const no = vertexBytes + i * OCTREE_NODE_BYTES;
      const n = {};
      PLY_OCTREE_PROPS.forEach(([, type, field], k) =>
        n[field] = (type === 'float') ? dv.getFloat32(no + 4 * k, true) : dv.getInt32(no + 4 * k, true));
      nodes[i] = n;
      if (n.isLeaf && n.countOrDepth > 0) {
        const end = Math.min(numVertices, n.firstChildOrStart + n.countOrDepth);
        for (let s = n.firstChildOrStart; s < end; s++) params[s * SPLAT_FLOATS + 7] = leafId;
        leafId++;
      }
    }
    octreeData = { meta: octreeMeta, nodes };
  }
  return { count: numVertices, params, octreeData };
}

// antimatter15 .splat: 32 bytes/splat = 3f pos, 3f linear scale, rgba u8, quat u8 ((q + 1) * 128)
const SPLAT_BYTES = 32;

export function parseSplatBuffer(buffer) {
  const count = Math.floor(buffer.byteLength / SPLAT_BYTES);
  const f32 = new Float32Array(buffer, 0, count * SPLAT_BYTES / 4);
  const u8  = new Uint8Array(buffer);
  const params = new Float32Array(count * SPLAT_FLOATS);
  for (let i = 0; i < count; i++) {
    const fo = i * 8, uo = i * SPLAT_BYTES;
    const logS = (k) => Math.log(Math.max(1e-6, f32[fo + 3 + k]));
    const q = (k) => (u8[uo + 28 + k] - 128) / 128;
    const dc = (k) => rgbToSh0(u8[uo + 24 + k] / 255);
    setSplat(params, i, f32[fo], f32[fo + 1], f32[fo + 2], logit(u8[uo + 27] / 255),
             logS(0), logS(1), logS(2), q(0), q(1), q(2), q(3), dc(0), dc(1), dc(2));
  }
  return { count, params, octreeData: null };
}

// SH0 .ply (14 floats/vertex, PLY_VERTEX_PROPS order) + optional `element octree_node` block.
export function exportPlyBuffer(params, count, octree = null) {
  const nodes = octree ? octree.serializeNodes() : [];
  const header = ['ply', 'format binary_little_endian 1.0', 'comment Exported from Gauss-Opt', 'comment Vertical axis: y'];
  if (nodes.length > 0) {
    const [cx, cy, cz] = octree.sceneCenter;
    header.push(`comment gaussopt_octree_v1 beta=${octree.beta.toFixed(4)} max_leaf=${octree.maxLeafSize} ` +
                `cx=${cx.toFixed(5)} cy=${cy.toFixed(5)} cz=${cz.toFixed(5)} radius=${octree.sceneRadius.toFixed(5)}`);
  }
  header.push(`element vertex ${count}`, ...PLY_VERTEX_PROPS.map(([name]) => `property float ${name}`));
  if (nodes.length > 0) {
    header.push(`element octree_node ${nodes.length}`, ...PLY_OCTREE_PROPS.map(([name, type]) => `property ${type} ${name}`));
  }
  header.push('end_header\n');
  const headerBytes = new TextEncoder().encode(header.join('\n'));

  const vertexBytes = count * PLY_VERTEX_PROPS.length * 4;
  const out = new ArrayBuffer(headerBytes.byteLength + vertexBytes + nodes.length * OCTREE_NODE_BYTES);
  new Uint8Array(out).set(headerBytes);
  const dv = new DataView(out, headerBytes.byteLength);
  let o = 0;
  for (let i = 0; i < count; i++) {
    const b = i * SPLAT_FLOATS;
    for (const [, slot] of PLY_VERTEX_PROPS) { dv.setFloat32(o, params[b + slot], true); o += 4; }
  }
  for (const n of nodes) {
    for (const [, type, field] of PLY_OCTREE_PROPS) {
      if (type === 'float') dv.setFloat32(o, n[field], true); else dv.setInt32(o, n[field], true);
      o += 4;
    }
  }
  return out;
}

export function exportSplatBuffer(params, count) {
  const out = new ArrayBuffer(count * SPLAT_BYTES);
  const f32 = new Float32Array(out);
  const u8  = new Uint8Array(out);
  for (let i = 0; i < count; i++) {
    const b = i * SPLAT_FLOATS, fo = i * 8, uo = i * SPLAT_BYTES;
    for (let k = 0; k < 3; k++) {
      f32[fo + k]     = params[b + k];
      f32[fo + 3 + k] = Math.exp(params[b + 4 + k]);
      u8[uo + 24 + k] = Math.round(sh0ToRgb01(params[b + 12 + k]) * 255);
    }
    u8[uo + 27] = Math.round(sigmoid(params[b + 3]) * 255);
    const qn = Math.hypot(params[b + 8], params[b + 9], params[b + 10], params[b + 11]) || 1;
    for (let k = 0; k < 4; k++) {
      u8[uo + 28 + k] = Math.min(255, Math.max(0, Math.round((params[b + 8 + k] / qn + 1) * 128)));
    }
  }
  return out;
}

// ── Dataset loaders (HTTP base URL or dropped folder) ────────────────────────

// Parse the sparse model; every 8th view (i % 8 == 4) is held out for validation when >= 8 views.
// points3D.bin is optional (ptBuf = null): an empty point set, see fallbackPoints3D().
function buildDataset([camBuf, imgBuf, ptBuf]) {
  const cameras  = parseCamerasBin(camBuf);
  const images   = parseImagesBin(imgBuf, cameras);
  const points3D = ptBuf ? parsePoints3DBin(ptBuf) : fallbackPoints3D(images);
  const stats    = computeCameraDepthStats(images, points3D);
  images.forEach((im, i) => { im.isValidation = images.length >= 8 && (i % 8) === 4; });
  return { cameras, images, points3D, stats };
}
const SPARSE_FILES = ['sparse/cameras.bin', 'sparse/images.bin', 'sparse/points3D.bin'];
const OPTIONAL_FILE = 'sparse/points3D.bin';

// Stand-in init cloud when points3D.bin is missing: gray points scattered along each camera's
// view frustum, at depths ~ the spread of the camera centers (no scale information otherwise).
function fallbackPoints3D(images, perCam = 200) {
  const n = images.length * perCam;
  const positions = new Float32Array(3 * n), colors = new Uint8Array(3 * n).fill(128);
  const c0 = [0, 1, 2].map(k => images.reduce((s, im) => s + im.center[k], 0) / Math.max(1, images.length));
  const spread = Math.max(1e-3, Math.sqrt(images.reduce((s, im) =>
    s + (im.center[0] - c0[0]) ** 2 + (im.center[1] - c0[1]) ** 2 + (im.center[2] - c0[2]) ** 2, 0) / Math.max(1, images.length)));
  let p = 0;
  for (const { R, center, camera } of images) {
    for (let j = 0; j < perCam; j++, p += 3) {
      const z = spread * (0.5 + 2.5 * Math.random());
      const x = (Math.random() - 0.5) * z * camera.width / camera.fx;
      const y = (Math.random() - 0.5) * z * camera.height / camera.fy;
      for (let k = 0; k < 3; k++) positions[p + k] = center[k] + R[k] * x + R[3 + k] * y + R[6 + k] * z;  // R^T [x y z]
    }
  }
  return { count: n, positions, colors, errors: new Float32Array(n), pointIndexById: new Map(), fallback: true };
}

export async function loadColmapFromUrl(baseUrl, { onProgress } = {}) {
  const base = baseUrl.replace(/\/$/, '');
  onProgress?.('Fetching COLMAP sparse model...');
  let info = { plys: [] };
  try {
    const r = await fetch(`${base}/info.json`);
    if (r.ok) info = await r.json();
  } catch {}
  const ds = buildDataset(await Promise.all(SPARSE_FILES.map(async (f) => {
    const r = await fetch(`${base}/${f}`);
    if (!r.ok) {
      if (f === OPTIONAL_FILE) return null;
      throw new Error(`Missing ${base}/${f}`);
    }
    return r.arrayBuffer();
  })));
  for (const im of ds.images) im.imageUrl = `${base}/images/${im.name}`;
  return { baseUrl: base, ...ds, availablePlys: info.plys || [] };
}

export async function loadColmapFromFiles(fileList, { onProgress } = {}) {
  const byRel = new Map();  // path relative to the dropped root, and bare file name
  for (const f of fileList) {
    byRel.set((f.webkitRelativePath || f.name).replace(/^[^/]+\//, ''), f);
    byRel.set(f.name, f);
  }
  onProgress?.('Parsing dropped COLMAP files...');
  const ds = buildDataset(await Promise.all(SPARSE_FILES.map((rel) => {
    const f = byRel.get(rel) || byRel.get(`gs/${rel}`);
    if (!f) {
      if (rel === OPTIONAL_FILE) return null;
      throw new Error(`Missing ${rel} in dropped files`);
    }
    return f.arrayBuffer();
  })));
  for (const im of ds.images) {
    const f = byRel.get(`images/${im.name}`) || byRel.get(im.name);
    if (f) im.fileBlob = f;
  }
  const availablePlys = [];
  for (const [name, file] of byRel) {
    if (name.endsWith('.ply') || name.endsWith('.splat')) availablePlys.push({ name, file });
  }
  return { baseUrl: null, ...ds, availablePlys };
}
