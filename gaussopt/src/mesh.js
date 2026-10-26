// mesh.js — A triangle mesh from the trained splats (DESIGN.md §9 E5): each camera's rendered
// median depth fused into a truncated signed distance grid (TSDF), its zero level set extracted
// with surface nets, the vertices colored from the views that see them, written as a .ply.
// The CPU core (bounds, Tsdf, colorVertices, meshToPly) is DOM- and GPU-free; meshFromTrainer
// drives it from a Trainer's forward renders.
"use strict";

import { atlas, bakeView, finishTexture, texelsOf } from './mesh_texture.js';
import { decodeImage, mipLevels } from './mesh_image.js';

// A depth view: { R (world -> camera, row-major 3x3), t, fx, fy, cx, cy, W, H,
//                 depth: Float32Array W*H (camera z; 0 = no depth), rgb: Uint8Array RGBA rows of `pitch` bytes }

// Per-axis [lo, hi] percentiles of the views' depth points (every `stride`-th pixel), padded by `pad` of the extent
export function depthBounds(views, { lo = 0.02, hi = 0.98, pad = 0.05, stride = 7 } = {}) {
  const pts = [[], [], []];
  for (const v of views) {
    const { R, t, fx, fy, cx, cy, W, depth } = v;
    for (let p = 0; p < depth.length; p += stride) {
      const z = depth[p];
      if (!(z > 0)) continue;
      const xc = ((p % W) + 0.5 - cx) / fx * z - t[0], yc = (((p / W) | 0) + 0.5 - cy) / fy * z - t[1], zc = z - t[2];
      for (let k = 0; k < 3; k++) pts[k].push(R[k] * xc + R[3 + k] * yc + R[6 + k] * zc);  // R^T (x_cam - t)
    }
  }
  if (!pts[0].length) return null;
  const out = pts.map(a => {
    a.sort((x, y) => x - y);
    const a0 = a[Math.floor(lo * (a.length - 1))], a1 = a[Math.ceil(hi * (a.length - 1))], e = (a1 - a0) * pad;
    return [a0 - e, a1 + e];
  });
  return { min: out.map(r => r[0]), max: out.map(r => r[1]) };
}

export class Tsdf {
  // res: voxels along the longest axis of [min, max]; trunc: the truncation width in voxels
  // freeWeight: the weight of a view's vote for free space beyond the band (sdf > mu): 1 counts a view that sees
  // past a surface as much as one that sees it (one view seeing through the splats of a plain wall carves it)
  constructor({ min, max }, { res = 256, trunc = 3, freeWeight = 1 } = {}) {
    this.freeWeight = freeWeight;
    const ext = [0, 1, 2].map(k => max[k] - min[k]);
    this.voxel = Math.max(...ext) / (res - 1);
    this.n = ext.map(e => Math.max(2, Math.ceil(e / this.voxel) + 1));
    this.origin = min.slice();
    this.mu = trunc * this.voxel;
    const N = this.n[0] * this.n[1] * this.n[2];
    this.tsdf = new Float32Array(N);
    this.weight = new Float32Array(N);
  }

  // Running average of min(1, (depth - z) / mu) over the voxels with depth - z >= -mu: in front of the
  // depth (up to free space, 1) and up to mu behind it; voxels further behind are left as they are.
  fuse({ R, t, fx, fy, cx, cy, W, H, depth }) {
    const [nx, ny, nz] = this.n, s = this.voxel, mu = this.mu, o = this.origin, D = this.tsdf, Wt = this.weight, fw = this.freeWeight;
    // camera coordinates of voxel (0, j, k), stepping by R's columns times the voxel size
    const ax = [R[0] * s, R[3] * s, R[6] * s], ay = [R[1] * s, R[4] * s, R[7] * s], az = [R[2] * s, R[5] * s, R[8] * s];
    const c0 = [0, 1, 2].map(r => R[3 * r] * o[0] + R[3 * r + 1] * o[1] + R[3 * r + 2] * o[2] + t[r]);
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) {
      let x = c0[0] + j * ay[0] + k * az[0], y = c0[1] + j * ay[1] + k * az[1], z = c0[2] + j * ay[2] + k * az[2];
      let idx = nx * (j + ny * k);
      for (let i = 0; i < nx; i++, idx++, x += ax[0], y += ax[1], z += ax[2]) {
        if (z <= 1e-6) continue;
        const u = Math.floor(fx * x / z + cx), v = Math.floor(fy * y / z + cy);
        if (u < 0 || v < 0 || u >= W || v >= H) continue;
        const d = depth[v * W + u];
        if (!(d > 0)) continue;
        const sdf = d - z;
        if (sdf < -mu) continue;
        const w = Wt[idx], dw = sdf > mu ? fw : 1;
        D[idx] = (D[idx] * w + dw * Math.min(1, sdf / mu)) / (w + dw);
        Wt[idx] = w + dw;
      }
    }
  }

  // At world point p: the signed distance to the surface (> 0 in front of it), the unit normal (the
  // tsdf's gradient, towards free space), the smallest view count of the cell's 8 corners and the
  // tsdf; null outside the grid or where tsdf does not change. The distance is tsdf / |gradient|
  // (a Newton step): tsdf x mu is measured along the cameras' axes, 1.7-2x the true distance on the
  // test sphere.
  sample(p) {
    const [nx, ny] = this.n, f = [0, 1, 2].map(k => (p[k] - this.origin[k]) / this.voxel);
    const c = f.map(Math.floor);
    if (c.some((ck, k) => ck < 0 || ck >= this.n[k] - 1)) return null;
    const [u, v, w] = f.map((fk, k) => fk - c[k]), base = c[0] + nx * (c[1] + ny * c[2]);
    let d = 0, minW = Infinity;
    const g = [0, 0, 0];
    for (let q = 0; q < 8; q++) {
      const i = q & 1, j = (q >> 1) & 1, k = (q >> 2) & 1, idx = base + i + nx * (j + ny * k);
      const D = this.tsdf[idx], wu = i ? u : 1 - u, wv = j ? v : 1 - v, ww = k ? w : 1 - w;
      d += D * wu * wv * ww;
      g[0] += D * (i ? 1 : -1) * wv * ww; g[1] += D * (j ? 1 : -1) * wu * ww; g[2] += D * (k ? 1 : -1) * wu * wv;
      minW = Math.min(minW, this.weight[idx]);
    }
    const gl = Math.hypot(...g);
    return gl > 0 ? { dist: d * this.voxel / gl, normal: g.map(x => x / gl), views: minW, tsdf: d } : null;
  }

  // Surface nets on the zero level set: { positions: Float32Array 3V, faces: Uint32Array 3F }. A cell
  // with a corner of weight < minWeight has no vertex; an edge crossing whose ends differ by >= 1.5
  // (of at most 2: an occlusion boundary's jump from free space to behind a surface, not a surface;
  // a limit of 1 opened holes in a sphere, where oblique views pull the outside values to ~1) does not count. Faces
  // are counter-clockwise seen from the positive (free space) side.
  extract({ minWeight = 1 } = {}) {
    const [nx, ny, nz] = this.n, D = this.tsdf, Wt = this.weight, s = this.voxel, o = this.origin;
    const cx = nx - 1, cy = ny - 1, cz = nz - 1;
    const cellVert = new Int32Array(cx * cy * cz).fill(-1);
    const corner = (c) => [c & 1, (c >> 1) & 1, (c >> 2) & 1];
    const EDGES = [];
    for (let a = 0; a < 8; a++) for (let b = a + 1; b < 8; b++) if ([1, 2, 4].includes(a ^ b)) EDGES.push([a, b]);
    const CORNERS = Array.from({ length: 8 }, (_, c) => corner(c));
    const OFF = CORNERS.map(([i, j, k]) => i + nx * (j + ny * k));
    const crosses = (a, b) => (a < 0) !== (b < 0) && Math.abs(a - b) < 1.5;
    const pos = [];
    const val = new Float64Array(8);
    for (let k = 0; k < cz; k++) for (let j = 0; j < cy; j++) for (let i = 0; i < cx; i++) {
      const base = i + nx * (j + ny * k);
      let ok = true, neg = 0;
      for (let c = 0; c < 8; c++) {
        const q = base + OFF[c];
        if (Wt[q] < minWeight) { ok = false; break; }
        val[c] = D[q];
        if (D[q] < 0) neg++;
      }
      if (!ok || neg === 0 || neg === 8) continue;
      let px = 0, py = 0, pz = 0, m = 0;
      for (const [a, b] of EDGES) {
        if (!crosses(val[a], val[b])) continue;
        const f = val[a] / (val[a] - val[b]), A = CORNERS[a], B = CORNERS[b];
        px += A[0] + f * (B[0] - A[0]); py += A[1] + f * (B[1] - A[1]); pz += A[2] + f * (B[2] - A[2]);
        m++;
      }
      if (!m) continue;
      cellVert[i + cx * (j + cy * k)] = pos.length / 3;
      pos.push(o[0] + (i + px / m) * s, o[1] + (j + py / m) * s, o[2] + (k + pz / m) * s);
    }
    // a quad per crossing grid edge, around it: the 4 cells sharing the edge (axis a, the other two
    // axes u = a+1, w = a+2 in cyclic order, so the cycle (0,0) (1,0) (1,1) (0,1) turns around +a)
    const faces = [];
    const cellAt = (ijk) => (ijk[0] < 0 || ijk[1] < 0 || ijk[2] < 0 || ijk[0] >= cx || ijk[1] >= cy || ijk[2] >= cz)
      ? -1 : cellVert[ijk[0] + cx * (ijk[1] + cy * ijk[2])];
    const step = [1, nx, nx * ny];
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const q = i + nx * (j + ny * k), ijk = [i, j, k];
      if (Wt[q] < minWeight) continue;
      for (let a = 0; a < 3; a++) {
        if (ijk[a] + 1 >= this.n[a]) continue;
        const r = q + step[a];
        if (Wt[r] < minWeight || !crosses(D[q], D[r])) continue;
        const u = (a + 1) % 3, w = (a + 2) % 3, quad = [];
        for (const [du, dw] of [[0, 0], [1, 0], [1, 1], [0, 1]]) {
          const c = ijk.slice();
          c[u] += du - 1; c[w] += dw - 1;
          quad.push(cellAt(c));
        }
        if (quad.includes(-1)) continue;
        // D[q] < 0: inside at q, free space towards +a, so the normal is +a
        if (D[q] < 0) faces.push(quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]);
        else faces.push(quad[0], quad[2], quad[1], quad[0], quad[3], quad[2]);
      }
    }
    return { positions: new Float32Array(pos), faces: new Uint32Array(faces) };
  }
}

// Vertex colors: the mean of the views' pixels where a vertex is within `tol` of the view's depth
// (seen, not occluded). acc: Float64Array 4V (r, g, b, count), filled one view at a time.
export function colorVertices(mesh, view, tol, acc) {
  const { R, t, fx, fy, cx, cy, W, H, depth, rgb, pitch } = view, P = mesh.positions;
  for (let i = 0, n = P.length / 3; i < n; i++) {
    const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
    const zc = R[6] * x + R[7] * y + R[8] * z + t[2];
    if (zc <= 1e-6) continue;
    const u = Math.floor(fx * (R[0] * x + R[1] * y + R[2] * z + t[0]) / zc + cx);
    const v = Math.floor(fy * (R[3] * x + R[4] * y + R[5] * z + t[1]) / zc + cy);
    if (u < 0 || v < 0 || u >= W || v >= H || !(Math.abs(depth[v * W + u] - zc) < tol)) continue;
    const p = v * pitch + 4 * u;
    acc[4 * i] += rgb[p]; acc[4 * i + 1] += rgb[p + 1]; acc[4 * i + 2] += rgb[p + 2]; acc[4 * i + 3]++;
  }
}

export function finishColors(acc) {
  const n = acc.length / 4, out = new Uint8Array(3 * n).fill(128);  // gray: seen by no view
  for (let i = 0; i < n; i++) if (acc[4 * i + 3]) for (let c = 0; c < 3; c++) out[3 * i + c] = Math.round(acc[4 * i + c] / acc[4 * i + 3]);
  return out;
}

// Binary little-endian .ply: float x y z, uchar red green blue, faces as uchar-counted int lists
export function meshToPly({ positions, faces, colors }) {
  const nv = positions.length / 3, nf = faces.length / 3;
  const header = new TextEncoder().encode(['ply', 'format binary_little_endian 1.0', 'comment gaussopt mesh (TSDF of rendered depth)',
    `element vertex ${nv}`, 'property float x', 'property float y', 'property float z',
    'property uchar red', 'property uchar green', 'property uchar blue',
    `element face ${nf}`, 'property list uchar int vertex_indices', 'end_header', ''].join('\n'));
  const out = new Uint8Array(header.length + nv * 15 + nf * 13), dv = new DataView(out.buffer);
  out.set(header);
  let o = header.length;
  for (let i = 0; i < nv; i++, o += 15) {
    for (let k = 0; k < 3; k++) dv.setFloat32(o + 4 * k, positions[3 * i + k], true);
    for (let k = 0; k < 3; k++) out[o + 12 + k] = colors ? colors[3 * i + k] : 128;
  }
  for (let f = 0; f < nf; f++, o += 13) {
    out[o] = 3;
    for (let k = 0; k < 3; k++) dv.setInt32(o + 1 + 4 * k, faces[3 * f + k], true);
  }
  return out;
}

// View `i` of a Trainer as a depth view (forward render; the caller holds the step lock), its rgb the
// splat render, or the photo (at training size) when `photo`.
// maxSpread: only pixels whose depth spread (pix_state.w, (z_q3 - z_q1) / z_med) is below it.
async function renderDepthView(tr, i, maxSpread, photo = false) {
  const W = tr.trainW, H = tr.trainH;
  tr._lastHatKey = '';
  const out = await tr._renderViewForward(i);
  const r = out && await tr._readback('mesh_readback', [{ buf: tr.pixStateBuf, bytes: W * H * 16 }, { tex: photo ? out.gtTex : tr.hatTex }]);
  if (!r) return null;
  const [pix, rgb] = r, depth = new Float32Array(W * H), img = tr.images[i], k = tr._intrinsics(img);
  for (let p = 0; p < W * H; p++) if (pix[4 * p] < 0.5 && pix[4 * p + 3] < maxSpread) depth[p] = pix[4 * p + 1];
  return { R: img.R, t: img.t, fx: k.fx, fy: k.fy, cx: k.cx, cy: k.cy, W, H, depth, rgb, pitch: tr.rowPitch };
}

// The Tsdf of a Trainer's current splats from views `ids` (image indices). lock(fn) runs fn under
// the step lock (identity when the caller already holds it, as the rebalance does).
export async function fuseTrainer(tr, ids, { res = 256, trunc = 3, freeWeight = 1, maxSpread = Infinity, lock = (fn) => fn(), onView = null } = {}) {
  const render = (i) => lock(() => renderDepthView(tr, i, maxSpread));
  const sample = [];
  for (let n = 0; n < ids.length; n += Math.max(1, Math.floor(ids.length / 24))) {
    const v = await render(ids[n]);
    if (v) sample.push(v);
  }
  const bounds = depthBounds(sample);
  if (!bounds) throw new Error('mesh: no rendered depth');
  const grid = new Tsdf(bounds, { res, trunc, freeWeight });
  for (let n = 0; n < ids.length; n++) {
    const v = await render(ids[n]);
    if (v) grid.fuse(v);
    await onView?.(n + 1, ids.length);
  }
  return grid;
}

// meshToPly's .ply read back ({ positions, faces, colors }), or null when the .ply has no faces (splats)
export function meshFromPly(buffer) {
  const u8 = new Uint8Array(buffer), head = new TextDecoder().decode(u8.subarray(0, 1024));
  const end = head.indexOf('end_header\n'), nf = Number(head.match(/element face (\d+)/)?.[1] ?? 0);
  if (end < 0 || !nf) return null;
  if (!head.includes('format binary_little_endian') || !head.includes('property uchar blue\nelement face') ||
      !head.includes('property list uchar int vertex_indices')) throw new Error('mesh .ply: only the layout of Make mesh is read');
  const nv = Number(head.match(/element vertex (\d+)/)[1]), o = end + 11, dv = new DataView(buffer);
  const positions = new Float32Array(3 * nv), colors = new Uint8Array(3 * nv), faces = new Uint32Array(3 * nf);
  for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) { positions[3 * i + k] = dv.getFloat32(o + 15 * i + 4 * k, true); colors[3 * i + k] = u8[o + 15 * i + 12 + k]; }
  for (let f = 0; f < nf; f++) for (let k = 0; k < 3; k++) faces[3 * f + k] = dv.getInt32(o + 15 * nv + 13 * f + 1 + 4 * k, true);
  return { positions, faces, colors };
}

// The mesh of a Trainer's current splats, from the forward render of its views (every `stride`-th).
// Returns { mesh: { positions, faces, colors }, info }; onProgress(phase, done, total) may be async.
export async function meshFromTrainer(trainer, { res = 256, trunc = 3, freeWeight = 0.1, maxSpread = Infinity, stride = 1, minWeight = 1, onProgress = null } = {}) {
  const tr = trainer, lock = (fn) => tr._withStepLock(fn);
  const ids = tr.images.map((im, i) => i).filter(i => i % stride === 0);
  const t0 = performance.now();
  try {
    const grid = await fuseTrainer(tr, ids, { res, trunc, freeWeight, maxSpread, lock, onView: (n, of) => onProgress?.('fuse', n, of) });
    const mesh = grid.extract({ minWeight });
    const acc = new Float64Array(4 * mesh.positions.length / 3);
    for (let n = 0; n < ids.length; n++) {
      const v = await lock(() => renderDepthView(tr, ids[n], maxSpread));
      if (v) colorVertices(mesh, v, grid.mu, acc);
      await onProgress?.('color', n + 1, ids.length);
    }
    mesh.colors = finishColors(acc);
    return { mesh, info: { views: ids.length, grid: grid.n, voxel: grid.voxel, trunc, bounds: { min: grid.origin, max: grid.n.map((n, k) => grid.origin[k] + (n - 1) * grid.voxel) },
                           vertices: mesh.positions.length / 3, faces: mesh.faces.length / 3, ms: performance.now() - t0 } };
  } finally {
    tr._invalidateCaches();
  }
}

// Image `i` of a Trainer at its own size (the SfM's undistorted JPEG, or the dataset's file; larger
// than the training size), with its 2 x 2 mean levels; null without a file
async function photoLevels(tr, i) {
  const im = tr.images[i];
  let blob = im.fileBlob;
  if (!blob && im.imageUrl) {
    const r = await fetch(im.imageUrl);
    if (r.ok) blob = await r.blob();
  }
  return blob ? mipLevels(await decodeImage(blob)) : null;
}

// A texture for `mesh` (mesh_texture.js) from a Trainer's views `ids` (default all): the atlas (or
// the mesh's own uvs), each view's photo (`source` 'photo': at its own size unless `full` is false,
// then at the training size) or splat render ('render') baked into it where its rendered depth is
// within `tol` of the texel. Returns { mesh (with uvs), image: RGBA bytes, size, info }.
export async function textureFromTrainer(trainer, mesh, { size = 2048, tol, ids = null, source = 'photo', full = true, maxSpread = Infinity, weight = 'footprint', power = 1, onProgress = null } = {}) {
  const tr = trainer, lock = (fn) => tr._withStepLock(fn), t0 = performance.now();
  ids ??= tr.images.map((im, i) => i);
  const at = mesh.uvs ? { mesh, info: {} } : atlas(mesh, { size });
  const texels = texelsOf(at.mesh, size), acc = new Float64Array(4 * texels.index.length);  // power > 1: weights beyond float32
  try {
    for (let n = 0; n < ids.length; n++) {
      const v = await lock(() => renderDepthView(tr, ids[n], maxSpread, source === 'photo'));
      if (v && source === 'photo' && full) v.levels = await photoLevels(tr, ids[n]);  // null: no file, the training size
      if (v) bakeView(texels, v, tol, acc, { weight, power });
      await onProgress?.('texture', n + 1, ids.length);
    }
  } finally {
    tr._invalidateCaches();
  }
  const { image, seen } = finishTexture(texels, size, acc);
  return { mesh: at.mesh, image, size, info: { ...at.info, texels: texels.index.length, used: texels.index.length / (size * size), overlap: texels.overlap,
                                                seen, views: ids.length, ms: performance.now() - t0 } };
}
