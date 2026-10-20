// gaussopt-colmap.js — COLMAP binary parser (cameras.bin, images.bin, points3D.bin)
//                      + 3DGS .ply / .splat checkpoint import & export (with embedded Loose Octree).
"use strict";

export const SH_C0 = 0.28209479177387814;
export const SPLAT_FLOATS = 16; // 64 bytes/splat:
// [0..3]:  x, y, z, logit_opacity
// [4..7]:  log_sx, log_sy, log_sz, leaf_id (as float/uint)
// [8..11]: qw, qx, qy, qz
// [12..15]: f_dc_0, f_dc_1, f_dc_2, flags

const CAMERA_MODEL_PARAMS = {
  0: { name: 'SIMPLE_PINHOLE', numParams: 3 }, // f, cx, cy
  1: { name: 'PINHOLE',        numParams: 4 }, // fx, fy, cx, cy
  2: { name: 'SIMPLE_RADIAL',  numParams: 4 }, // f, cx, cy, k
  3: { name: 'RADIAL',         numParams: 5 }, // f, cx, cy, k1, k2
  4: { name: 'OPENCV',         numParams: 8 }, // fx, fy, cx, cy, k1, k2, p1, p2
};

export function sigmoid(x) {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

export function logit(p) {
  const c = Math.min(0.995, Math.max(0.001, p));
  return Math.log(c / (1 - c));
}

export function rgbToSh0(r01) {
  return (r01 - 0.5) / SH_C0;
}

export function sh0ToRgb01(dc) {
  return Math.min(1, Math.max(0, 0.5 + SH_C0 * dc));
}

// Convert quaternion [qw, qx, qy, qz] to 3x3 row-major rotation matrix R.
export function quatToMat3([qw, qx, qy, qz]) {
  const n = Math.hypot(qw, qx, qy, qz) || 1;
  const w = qw / n, x = qx / n, y = qy / n, z = qz / n;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
    2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
  ];
}

// Build column-major 4x4 view matrix (world -> OpenCV camera: +X right, +Y down, +Z forward)
export function makeViewMatrix(R, t) {
  return new Float32Array([
    R[0], R[3], R[6], 0,
    R[1], R[4], R[7], 0,
    R[2], R[5], R[8], 0,
    t[0], t[1], t[2], 1,
  ]);
}

// Build column-major 4x4 WebGPU projection matrix from OpenCV intrinsics (fx, fy, cx, cy, W, H)
// Maps OpenCV camera space (+X right, +Y down, +Z forward in [znear, zfar])
// to WebGPU NDC: x in [-1, 1], y in [-1, 1] (+Y up), z in [0, 1].
export function makeProjectionMatrix(fx, fy, cx, cy, W, H, znear = 0.05, zfar = 200.0) {
  const A = zfar / (zfar - znear);
  const B = -(zfar * znear) / (zfar - znear);
  return new Float32Array([
    2 * fx / W,  0,           0,  0,
    0,          -2 * fy / H,  0,  0,
    2 * (cx / W) - 1, 1 - 2 * (cy / H), A, 1,
    0,           0,           B,  0,
  ]);
}

// ── COLMAP Binary Parsers ────────────────────────────────────────────────────

export function parseCamerasBin(buffer) {
  const dv = new DataView(buffer);
  let off = 0;
  const numCameras = Number(dv.getBigUint64(off, true)); off += 8;
  const cameras = new Map();

  for (let i = 0; i < numCameras; i++) {
    const cameraId = dv.getInt32(off, true); off += 4;
    const modelId  = dv.getInt32(off, true); off += 4;
    const width    = Number(dv.getBigUint64(off, true)); off += 8;
    const height   = Number(dv.getBigUint64(off, true)); off += 8;
    const info     = CAMERA_MODEL_PARAMS[modelId];
    if (!info) throw new Error(`Unsupported COLMAP camera model_id=${modelId}`);

    const p = [];
    for (let k = 0; k < info.numParams; k++) {
      p.push(dv.getFloat64(off, true)); off += 8;
    }
    const [fx, fy, cx, cy] = (modelId === 1 || modelId === 4)
      ? [p[0], p[1], p[2], p[3]]
      : [p[0], p[0], p[1], p[2]];

    cameras.set(cameraId, { id: cameraId, model: info.name, width, height, fx, fy, cx, cy });
  }
  return cameras;
}

export function parseImagesBin(buffer, cameras) {
  const dv = new DataView(buffer);
  const u8 = new Uint8Array(buffer);
  let off = 0;
  const numImages = Number(dv.getBigUint64(off, true)); off += 8;
  const images = [];

  for (let i = 0; i < numImages; i++) {
    const id = dv.getInt32(off, true); off += 4;
    const qw = dv.getFloat64(off, true); off += 8;
    const qx = dv.getFloat64(off, true); off += 8;
    const qy = dv.getFloat64(off, true); off += 8;
    const qz = dv.getFloat64(off, true); off += 8;
    const tx = dv.getFloat64(off, true); off += 8;
    const ty = dv.getFloat64(off, true); off += 8;
    const tz = dv.getFloat64(off, true); off += 8;
    const cameraId = dv.getInt32(off, true); off += 4;

    let end = off;
    while (end < u8.length && u8[end] !== 0) end++;
    const name = new TextDecoder().decode(u8.subarray(off, end));
    off = end + 1;

    const numPoints2D = Number(dv.getBigUint64(off, true)); off += 8;
    const point3DIds = [];
    const obs2D = [];
    for (let k = 0; k < numPoints2D; k++) {
      const px = dv.getFloat64(off, true);
      const py = dv.getFloat64(off + 8, true);
      const pid = Number(dv.getBigInt64(off + 16, true));
      off += 24;
      if (pid >= 0) {
        point3DIds.push(pid);
        obs2D.push(px, py, pid);
      }
    }

    const R = quatToMat3([qw, qx, qy, qz]);
    const t = [tx, ty, tz];
    // Camera optical center in world space: C = -R^T * t
    const center = [
      -(R[0] * tx + R[3] * ty + R[6] * tz),
      -(R[1] * tx + R[4] * ty + R[7] * tz),
      -(R[2] * tx + R[5] * ty + R[8] * tz),
    ];
    // Optical axis (+Z in camera space) in world space: third row of R
    const forward = [R[6], R[7], R[8]];
    const cam = cameras.get(cameraId) || cameras.values().next().value;
    const view = makeViewMatrix(R, t);
    const proj = makeProjectionMatrix(cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height);

    images.push({
      id, name, cameraId, camera: cam,
      q: [qw, qx, qy, qz], t, R, center, forward,
      view, proj,
      point3DIds,
      obs2D: new Float64Array(obs2D),
      zMin: 0.2,
      zMax: 50.0,
      isValidation: false,
    });
  }
  images.sort((a, b) => a.name.localeCompare(b.name));
  return images;
}

export function parsePoints3DBin(buffer) {
  const dv = new DataView(buffer);
  let off = 0;
  const numPoints = Number(dv.getBigUint64(off, true)); off += 8;

  const positions = new Float32Array(numPoints * 3);
  const colors    = new Uint8Array(numPoints * 3);
  const errors    = new Float32Array(numPoints);
  const pointIndexById = new Map();

  for (let i = 0; i < numPoints; i++) {
    const id = Number(dv.getBigUint64(off, true)); off += 8;
    const x  = dv.getFloat64(off, true); off += 8;
    const y  = dv.getFloat64(off, true); off += 8;
    const z  = dv.getFloat64(off, true); off += 8;
    const r  = dv.getUint8(off);
    const g  = dv.getUint8(off + 1);
    const b  = dv.getUint8(off + 2);
    off += 3;
    const err = dv.getFloat64(off, true); off += 8;
    const trackLen = Number(dv.getBigUint64(off, true)); off += 8;
    off += trackLen * 8; // skip [image_id (i32), point2D_idx (i32)]

    positions[3 * i + 0] = x;
    positions[3 * i + 1] = y;
    positions[3 * i + 2] = z;
    colors[3 * i + 0] = r;
    colors[3 * i + 1] = g;
    colors[3 * i + 2] = b;
    errors[i] = err;
    pointIndexById.set(id, i);
  }
  return { count: numPoints, positions, colors, errors, pointIndexById };
}

// Compute per-camera near/far observed depths (for Section 5.2 near-camera floater carving)
// and mean COLMAP reprojection error verification (Stage 1 check).
export function computeCameraDepthStats(images, points3D) {
  const { positions, pointIndexById } = points3D;
  let totalReprojErr = 0;
  let totalObs = 0;

  for (const img of images) {
    const depths = [];
    const { R, t, camera, obs2D } = img;
    for (let k = 0; k < obs2D.length; k += 3) {
      const u = obs2D[k], v = obs2D[k + 1], pid = obs2D[k + 2];
      const idx = pointIndexById.get(pid);
      if (idx === undefined) continue;
      const X = positions[3 * idx], Y = positions[3 * idx + 1], Z = positions[3 * idx + 2];
      const xc = R[0] * X + R[1] * Y + R[2] * Z + t[0];
      const yc = R[3] * X + R[4] * Y + R[5] * Z + t[1];
      const zc = R[6] * X + R[7] * Y + R[8] * Z + t[2];
      if (zc > 0.01) {
        depths.push(zc);
        const uProj = camera.fx * (xc / zc) + camera.cx;
        const vProj = camera.fy * (yc / zc) + camera.cy;
        totalReprojErr += Math.hypot(uProj - u, vProj - v);
        totalObs++;
      }
    }
    if (depths.length > 0) {
      depths.sort((a, b) => a - b);
      img.zMin = depths[Math.floor(depths.length * 0.05)];
      img.zMax = depths[Math.floor(depths.length * 0.95)];
    }
  }
  const meanReprojError = totalObs > 0 ? totalReprojErr / totalObs : 0;
  return { meanReprojError, totalObs };
}

// ── 3DGS .ply / .splat Checkpoint Parser & Exporter (with Embedded Octree) ───

export function parsePlyBuffer(buffer) {
  const u8 = new Uint8Array(buffer);
  let pos = 0;
  const readLine = () => {
    let end = pos;
    while (end < u8.length && u8[end] !== 10) end++;
    const line = new TextDecoder().decode(u8.subarray(pos, end)).replace(/\r$/, '');
    pos = end + 1;
    return line;
  };

  const magic = readLine().trim();
  if (magic !== 'ply') throw new Error('Not a valid PLY file');

  let numVertices = 0;
  let numOctreeNodes = 0;
  let octreeMeta = null;
  const props = [];
  let currentElement = '';

  while (pos < u8.length) {
    const line = readLine().trim();
    if (line === 'end_header') break;
    const parts = line.split(/\s+/);
    if (parts[0] === 'comment' && parts[1] === 'gaussopt_octree_v1') {
      octreeMeta = { beta: 0.20, maxLeafSize: 256, maxDepth: 8, sceneCenter: [0, 0, 0], sceneRadius: 2.0 };
      for (let i = 2; i < parts.length; i++) {
        const [k, v] = parts[i].split('=');
        const num = parseFloat(v);
        if (k === 'beta') octreeMeta.beta = num;
        else if (k === 'max_leaf') octreeMeta.maxLeafSize = num | 0;
        else if (k === 'cx') octreeMeta.sceneCenter[0] = num;
        else if (k === 'cy') octreeMeta.sceneCenter[1] = num;
        else if (k === 'cz') octreeMeta.sceneCenter[2] = num;
        else if (k === 'radius') octreeMeta.sceneRadius = num;
      }
    } else if (parts[0] === 'element') {
      currentElement = parts[1];
      if (currentElement === 'vertex') numVertices = parseInt(parts[2], 10);
      else if (currentElement === 'octree_node') numOctreeNodes = parseInt(parts[2], 10);
    } else if (parts[0] === 'property' && currentElement === 'vertex') {
      props.push({ type: parts[1], name: parts[2] });
    }
  }

  const propOffset = {};
  let strideBytes = 0;
  for (const p of props) {
    propOffset[p.name] = { offset: strideBytes, type: p.type };
    if (p.type === 'float' || p.type === 'int' || p.type === 'uint') strideBytes += 4;
    else if (p.type === 'uchar' || p.type === 'uint8') strideBytes += 1;
    else if (p.type === 'double') strideBytes += 8;
    else strideBytes += 4;
  }

  const dv = new DataView(buffer, pos);
  const getF = (rowBase, name, defVal = 0) => {
    const info = propOffset[name];
    if (!info) return defVal;
    if (info.type === 'float') return dv.getFloat32(rowBase + info.offset, true);
    if (info.type === 'uchar' || info.type === 'uint8') return dv.getUint8(rowBase + info.offset);
    return dv.getFloat64(rowBase + info.offset, true);
  };

  const hasShDc = ('f_dc_0' in propOffset);
  const hasScale = ('scale_0' in propOffset);
  const params = new Float32Array(numVertices * SPLAT_FLOATS);

  for (let i = 0; i < numVertices; i++) {
    const row = i * strideBytes;
    const x = getF(row, 'x', 0);
    const y = getF(row, 'y', 0);
    const z = getF(row, 'z', 0);
    const op = ('opacity' in propOffset) ? getF(row, 'opacity', 0) : logit(0.5);

    const sx = hasScale ? getF(row, 'scale_0', -4) : -4;
    const sy = hasScale ? getF(row, 'scale_1', -4) : -4;
    const sz = hasScale ? getF(row, 'scale_2', -4) : -4;

    let qw = getF(row, 'rot_0', 1);
    let qx = getF(row, 'rot_1', 0);
    let qy = getF(row, 'rot_2', 0);
    let qz = getF(row, 'rot_3', 0);
    const qn = Math.hypot(qw, qx, qy, qz) || 1;
    qw /= qn; qx /= qn; qy /= qn; qz /= qn;

    let dc0, dc1, dc2;
    if (hasShDc) {
      dc0 = getF(row, 'f_dc_0', 0);
      dc1 = getF(row, 'f_dc_1', 0);
      dc2 = getF(row, 'f_dc_2', 0);
    } else {
      dc0 = rgbToSh0(getF(row, 'red', 128) / 255);
      dc1 = rgbToSh0(getF(row, 'green', 128) / 255);
      dc2 = rgbToSh0(getF(row, 'blue', 128) / 255);
    }

    const base = i * SPLAT_FLOATS;
    params[base + 0]  = x;
    params[base + 1]  = y;
    params[base + 2]  = z;
    params[base + 3]  = op;
    params[base + 4]  = sx;
    params[base + 5]  = sy;
    params[base + 6]  = sz;
    params[base + 7]  = 0;
    params[base + 8]  = qw;
    params[base + 9]  = qx;
    params[base + 10] = qy;
    params[base + 11] = qz;
    params[base + 12] = dc0;
    params[base + 13] = dc1;
    params[base + 14] = dc2;
    params[base + 15] = 0;
  }

  // Parse embedded `element octree_node` table if present
  let octreeData = null;
  const octreeByteOffset = numVertices * strideBytes;
  if (numOctreeNodes > 0 && octreeMeta && (pos + octreeByteOffset + numOctreeNodes * 32 <= buffer.byteLength)) {
    const nodes = new Array(numOctreeNodes);
    let lid = 0;
    for (let i = 0; i < numOctreeNodes; i++) {
      const no = octreeByteOffset + i * 32;
      const cx = dv.getFloat32(no + 0, true);
      const cy = dv.getFloat32(no + 4, true);
      const cz = dv.getFloat32(no + 8, true);
      const halfSize          = dv.getFloat32(no + 12, true);
      const isLeaf            = dv.getInt32(no + 16, true);
      const firstChildOrStart = dv.getInt32(no + 20, true);
      const countOrDepth      = dv.getInt32(no + 24, true);
      const state             = dv.getInt32(no + 28, true);
      nodes[i] = { cx, cy, cz, halfSize, isLeaf, firstChildOrStart, countOrDepth, state };
      if (isLeaf && countOrDepth > 0) {
        for (let k = 0; k < countOrDepth; k++) {
          const sIdx = firstChildOrStart + k;
          if (sIdx < numVertices) params[sIdx * SPLAT_FLOATS + 7] = lid;
        }
        lid++;
      }
    }
    octreeData = { meta: octreeMeta, nodes };
  }

  return { count: numVertices, params, octreeData };
}

export function parseSplatBuffer(buffer) {
  const STRIDE = 32; // 3f pos + 3f scale + 4u8 rgba + 4u8 quat
  const count = Math.floor(buffer.byteLength / STRIDE);
  const f32 = new Float32Array(buffer);
  const u8  = new Uint8Array(buffer);
  const params = new Float32Array(count * SPLAT_FLOATS);

  for (let i = 0; i < count; i++) {
    const fo = i * 8;
    const uo = i * 32;
    const x = f32[fo + 0], y = f32[fo + 1], z = f32[fo + 2];
    const sx = Math.log(Math.max(1e-6, f32[fo + 3]));
    const sy = Math.log(Math.max(1e-6, f32[fo + 4]));
    const sz = Math.log(Math.max(1e-6, f32[fo + 5]));

    const r = u8[uo + 24] / 255;
    const g = u8[uo + 25] / 255;
    const b = u8[uo + 26] / 255;
    const a = u8[uo + 27] / 255;

    let qw = (u8[uo + 28] - 128) / 128;
    let qx = (u8[uo + 29] - 128) / 128;
    let qy = (u8[uo + 30] - 128) / 128;
    let qz = (u8[uo + 31] - 128) / 128;
    const qn = Math.hypot(qw, qx, qy, qz) || 1;
    qw /= qn; qx /= qn; qy /= qn; qz /= qn;

    const base = i * SPLAT_FLOATS;
    params[base + 0]  = x;
    params[base + 1]  = y;
    params[base + 2]  = z;
    params[base + 3]  = logit(a);
    params[base + 4]  = sx;
    params[base + 5]  = sy;
    params[base + 6]  = sz;
    params[base + 7]  = 0;
    params[base + 8]  = qw;
    params[base + 9]  = qx;
    params[base + 10] = qy;
    params[base + 11] = qz;
    params[base + 12] = rgbToSh0(r);
    params[base + 13] = rgbToSh0(g);
    params[base + 14] = rgbToSh0(b);
    params[base + 15] = 0;
  }
  return { count, params, octreeData: null };
}

export function exportPlyBuffer(params, count, octree = null) {
  const nodes = octree ? octree.serializeNodes() : [];
  const headerLines = [
    'ply',
    'format binary_little_endian 1.0',
    'comment Exported from Gauss-Opt',
    'comment Vertical axis: y',
  ];
  if (octree && nodes.length > 0) {
    const [cx, cy, cz] = octree.sceneCenter;
    headerLines.push(
      `comment gaussopt_octree_v1 beta=${octree.beta.toFixed(4)} max_leaf=${octree.maxLeafSize} cx=${cx.toFixed(5)} cy=${cy.toFixed(5)} cz=${cz.toFixed(5)} radius=${octree.sceneRadius.toFixed(5)}`
    );
  }
  headerLines.push(
    `element vertex ${count}`,
    'property float x',
    'property float y',
    'property float z',
    'property float f_dc_0',
    'property float f_dc_1',
    'property float f_dc_2',
    'property float opacity',
    'property float scale_0',
    'property float scale_1',
    'property float scale_2',
    'property float rot_0',
    'property float rot_1',
    'property float rot_2',
    'property float rot_3'
  );
  if (nodes.length > 0) {
    headerLines.push(
      `element octree_node ${nodes.length}`,
      'property float cx',
      'property float cy',
      'property float cz',
      'property float half_size',
      'property int is_leaf',
      'property int first_child_or_start',
      'property int count_or_depth',
      'property int state'
    );
  }
  headerLines.push('end_header\n');
  const header = headerLines.join('\n');

  const headerBytes = new TextEncoder().encode(header);
  const floatCount = 14;
  const vertexBytes = count * floatCount * 4;
  const octreeBytes = nodes.length * 32;
  const out = new ArrayBuffer(headerBytes.byteLength + vertexBytes + octreeBytes);
  new Uint8Array(out, 0, headerBytes.byteLength).set(headerBytes);
  const dv = new DataView(out, headerBytes.byteLength);

  for (let i = 0; i < count; i++) {
    const b = i * SPLAT_FLOATS;
    const o = i * floatCount * 4;
    dv.setFloat32(o +  0, params[b + 0], true);  // x
    dv.setFloat32(o +  4, params[b + 1], true);  // y
    dv.setFloat32(o +  8, params[b + 2], true);  // z
    dv.setFloat32(o + 12, params[b + 12], true); // f_dc_0
    dv.setFloat32(o + 16, params[b + 13], true); // f_dc_1
    dv.setFloat32(o + 20, params[b + 14], true); // f_dc_2
    dv.setFloat32(o + 24, params[b + 3], true);  // opacity (logit)
    dv.setFloat32(o + 28, params[b + 4], true);  // scale_0 (log)
    dv.setFloat32(o + 32, params[b + 5], true);  // scale_1 (log)
    dv.setFloat32(o + 36, params[b + 6], true);  // scale_2 (log)
    dv.setFloat32(o + 40, params[b + 8], true);  // rot_0 (qw)
    dv.setFloat32(o + 44, params[b + 9], true);  // rot_1 (qx)
    dv.setFloat32(o + 48, params[b + 10], true); // rot_2 (qy)
    dv.setFloat32(o + 52, params[b + 11], true); // rot_3 (qz)
  }

  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    const no = vertexBytes + i * 32;
    dv.setFloat32(no +  0, n.cx, true);
    dv.setFloat32(no +  4, n.cy, true);
    dv.setFloat32(no +  8, n.cz, true);
    dv.setFloat32(no + 12, n.halfSize, true);
    dv.setInt32(no + 16, n.isLeaf, true);
    dv.setInt32(no + 20, n.firstChildOrStart, true);
    dv.setInt32(no + 24, n.countOrDepth, true);
    dv.setInt32(no + 28, n.state, true);
  }
  return out;
}

export function exportSplatBuffer(params, count) {
  const out = new ArrayBuffer(count * 32);
  const f32 = new Float32Array(out);
  const u8  = new Uint8Array(out);
  for (let i = 0; i < count; i++) {
    const b = i * SPLAT_FLOATS;
    const fo = i * 8;
    const uo = i * 32;
    f32[fo + 0] = params[b + 0];
    f32[fo + 1] = params[b + 1];
    f32[fo + 2] = params[b + 2];
    f32[fo + 3] = Math.exp(params[b + 4]);
    f32[fo + 4] = Math.exp(params[b + 5]);
    f32[fo + 5] = Math.exp(params[b + 6]);
    u8[uo + 24] = Math.round(sh0ToRgb01(params[b + 12]) * 255);
    u8[uo + 25] = Math.round(sh0ToRgb01(params[b + 13]) * 255);
    u8[uo + 26] = Math.round(sh0ToRgb01(params[b + 14]) * 255);
    u8[uo + 27] = Math.round(sigmoid(params[b + 3]) * 255);
    const qw = params[b + 8], qx = params[b + 9], qy = params[b + 10], qz = params[b + 11];
    const qn = Math.hypot(qw, qx, qy, qz) || 1;
    u8[uo + 28] = Math.min(255, Math.max(0, Math.round((qw / qn + 1) * 128)));
    u8[uo + 29] = Math.min(255, Math.max(0, Math.round((qx / qn + 1) * 128)));
    u8[uo + 30] = Math.min(255, Math.max(0, Math.round((qy / qn + 1) * 128)));
    u8[uo + 31] = Math.min(255, Math.max(0, Math.round((qz / qn + 1) * 128)));
  }
  return out;
}

// ── Dataset Loader (HTTP URL or Dropped Folder) ──────────────────────────────

export async function loadColmapFromUrl(baseUrl, { onProgress } = {}) {
  const base = baseUrl.replace(/\/$/, '');
  onProgress?.('Fetching COLMAP sparse model...');

  let info = { plys: [] };
  try {
    const r = await fetch(`${base}/info.json`);
    if (r.ok) info = await r.json();
  } catch {}

  const [camBuf, imgBuf, ptBuf] = await Promise.all([
    fetch(`${base}/sparse/cameras.bin`).then(r => {
      if (!r.ok) throw new Error(`Missing ${base}/sparse/cameras.bin`);
      return r.arrayBuffer();
    }),
    fetch(`${base}/sparse/images.bin`).then(r => {
      if (!r.ok) throw new Error(`Missing ${base}/sparse/images.bin`);
      return r.arrayBuffer();
    }),
    fetch(`${base}/sparse/points3D.bin`).then(r => {
      if (!r.ok) throw new Error(`Missing ${base}/sparse/points3D.bin`);
      return r.arrayBuffer();
    }),
  ]);

  const cameras  = parseCamerasBin(camBuf);
  const images   = parseImagesBin(imgBuf, cameras);
  const points3D = parsePoints3DBin(ptBuf);
  const stats    = computeCameraDepthStats(images, points3D);

  // Hold out every 8th view as validation (if >= 8 views)
  for (let i = 0; i < images.length; i++) {
    images[i].isValidation = (images.length >= 8) && ((i % 8) === 4);
    images[i].imageUrl = `${base}/images/${images[i].name}`;
  }

  return { baseUrl: base, cameras, images, points3D, stats, availablePlys: info.plys || [] };
}

export async function loadColmapFromFiles(fileList, { onProgress } = {}) {
  const byRel = new Map();
  for (const f of fileList) {
    const rel = (f.webkitRelativePath || f.name).replace(/^[^/]+\//, '');
    byRel.set(rel, f);
    byRel.set(f.name, f);
  }
  const getBuf = async (rel) => {
    const f = byRel.get(rel) || byRel.get(`gs/${rel}`);
    if (!f) throw new Error(`Missing ${rel} in dropped files`);
    return f.arrayBuffer();
  };

  onProgress?.('Parsing dropped COLMAP files...');
  const [camBuf, imgBuf, ptBuf] = await Promise.all([
    getBuf('sparse/cameras.bin'),
    getBuf('sparse/images.bin'),
    getBuf('sparse/points3D.bin'),
  ]);

  const cameras  = parseCamerasBin(camBuf);
  const images   = parseImagesBin(imgBuf, cameras);
  const points3D = parsePoints3DBin(ptBuf);
  const stats    = computeCameraDepthStats(images, points3D);

  for (let i = 0; i < images.length; i++) {
    images[i].isValidation = (images.length >= 8) && ((i % 8) === 4);
    const imgFile = byRel.get(`images/${images[i].name}`) || byRel.get(images[i].name);
    if (imgFile) images[i].fileBlob = imgFile;
  }

  const availablePlys = [];
  for (const [k, f] of byRel.entries()) {
    if (k.endsWith('.ply') || k.endsWith('.splat')) availablePlys.push({ name: k, file: f });
  }

  return { baseUrl: null, cameras, images, points3D, stats, availablePlys };
}
