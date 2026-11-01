// mesh_grow.js — The mesh behind the splats (DESIGN.md §9 E5), grown from the splats themselves and carved by
// the cameras' rays, instead of src/mesh.js's TSDF of their rendered depth (which sits at the splats' median depth:
// a depth pre-pass at it hides much of what the splats draw). On a voxel grid:
// 1. occupancy: the splats' summed alpha (each to 3 sigma, its covariance + (voxel / 2)^2) reaches `tau`
// 2. closing: the occupancy dilated by `close` voxels (gaps under twice that closed), except around the camera path
// 3. seen exterior: the path (the camera centers, joined unless a step is over 5x the median) and the cells each
//    camera's rays (rayW x rayW * 9 / 16 a camera) cross before their first closed cell (through runs of closed cells
//    with no splats longer than `passage` voxels: narrow slots seen lengthwise, not holes nor a wall's closed band seen
//    at a grazing angle); solid: the cells within
//    `depth` voxels of a ray's first hit, not crossed by a ray. The rest is the unseen exterior.
// 4. surface: the level `close + inset` voxels from the seen exterior (behind the splats' front), surface nets;
//    the unseen exterior cuts the solid there without an inset, and the faces where it does are dropped (a sheet
//    behind the splats, open behind it). Faces at the grid's border, and faces with no splats within the color blur
//    (close + inset + 1 voxels: the edges of what the rays saw, in empty space), dropped too.
// Vertex colors: the splats' alpha-weighted color, box-blurred over that. DOM- and GPU-free (worker and node).
"use strict";

import { quatToMat3 } from './math.js';
import { SPLAT_FLOATS } from './colmap.js';
import { Tsdf } from './mesh.js';

const SH_C0 = 0.28209479177387814;
export const BACKING = { close: 2, inset: 4, depth: 12, passage: 24 };  // growBacking's defaults, in voxels

// Squared Euclidean distance transform (Felzenszwalb & Huttenlocher 2012), in voxels^2, of an [nx, ny, nz]
// grid to its cells where isFeature(q), into `out` (Float32Array N)
function edt([nx, ny, nz], isFeature, out) {
  const INF = 1e20, N = nx * ny * nz, D = out;
  for (let q = 0; q < N; q++) D[q] = isFeature(q) ? 0 : INF;
  const L = Math.max(nx, ny, nz), f = new Float64Array(L), d = new Float64Array(L), vv = new Int32Array(L), zz = new Float64Array(L + 1);
  const pass = (len, base, step) => {
    for (let q = 0; q < len; q++) f[q] = D[base + q * step];
    let k = 0; vv[0] = 0; zz[0] = -INF; zz[1] = INF;
    for (let q = 1; q < len; q++) {
      let s;
      while (true) { const r = vv[k]; s = ((f[q] + q * q) - (f[r] + r * r)) / (2 * q - 2 * r); if (s <= zz[k] && k > 0) k--; else break; }
      if (s <= zz[k]) { vv[0] = q; zz[0] = -INF; zz[1] = INF; k = 0; continue; }
      k++; vv[k] = q; zz[k] = s; zz[k + 1] = INF;
    }
    k = 0;
    for (let q = 0; q < len; q++) { while (zz[k + 1] < q) k++; const r = vv[k]; d[q] = (q - r) * (q - r) + f[r]; }
    for (let q = 0; q < len; q++) D[base + q * step] = Math.min(d[q], INF);
  };
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) pass(nx, nx * (j + ny * k), 1);
  for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) pass(ny, i + nx * ny * k, nx);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) pass(nz, i + nx * j, nx * ny);
  return D;
}

// Box sums over [q - K, q + K] along each axis in turn, in place
function boxBlur([nx, ny, nz], A, K) {
  const pre = new Float64Array(Math.max(nx, ny, nz) + 1);
  const run = (len, base, step) => {
    pre[0] = 0;
    for (let q = 0; q < len; q++) pre[q + 1] = pre[q] + A[base + q * step];
    for (let q = 0; q < len; q++) A[base + q * step] = pre[Math.min(len, q + K + 1)] - pre[Math.max(0, q - K)];
  };
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) run(nx, nx * (j + ny * k), 1);
  for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) run(ny, i + nx * ny * k, nx);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) run(nz, i + nx * j, nx * ny);
}

// splats: { params: Float32Array (SPLAT_FLOATS a splat, colmap.js), count }; cameras: [{ center, view (world ->
// camera, column-major 4 x 4), fx, fy, cx, cy, width, height }], `path` their order along the capture (default as
// given); bounds: { min, max }, the grid's (res voxels along the longest side, as Tsdf). Returns { mesh:
// { positions, faces, colors }, info } with info.voxel the grid's.
export function growBacking({ params, count }, cameras, bounds, { path = null, res = 384, tau = 0.5, close = BACKING.close, inset = BACKING.inset, depth = BACKING.depth,
                                                                  rayStride = 1, rayW = 192, cap = 16, passage = BACKING.passage, onProgress = null } = {}) {
  const t0 = performance.now(), times = {}, phase = (name) => { times[name] = Math.round(performance.now() - t0); onProgress?.(name); };
  const grid = new Tsdf(bounds, { res }), v = grid.voxel, n = grid.n, [nx, ny, nz] = n, N = nx * ny * nz, o = grid.origin;
  const at = (i, j, k) => i + nx * (j + ny * k);
  const cellOf = (x) => {
    const i = Math.round((x[0] - o[0]) / v), j = Math.round((x[1] - o[1]) / v), k = Math.round((x[2] - o[2]) / v);
    return i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz ? at(i, j, k) : -1;
  };

  // 1. summed alpha and alpha-weighted color
  const acc = new Float32Array(N), col = [0, 1, 2].map(() => new Float32Array(N));
  for (let s = 0; s < count; s++) {
    const p = params.subarray(s * SPLAT_FLOATS, (s + 1) * SPLAT_FLOATS), op = 1 / (1 + Math.exp(-p[3]));
    if (op < 0.02) continue;
    const rgb = [12, 13, 14].map(k => Math.min(1, Math.max(0, 0.5 + SH_C0 * p[k])));
    const Rm = quatToMat3([p[8], p[9], p[10], p[11]]), s2 = [4, 5, 6].map(k => Math.exp(2 * p[k]));
    const C = [0, 1, 2].map(a => [0, 1, 2].map(b => Rm[3 * a] * Rm[3 * b] * s2[0] + Rm[3 * a + 1] * Rm[3 * b + 1] * s2[1] + Rm[3 * a + 2] * Rm[3 * b + 2] * s2[2] + (a === b ? (v / 2) ** 2 : 0)));
    const i00 = C[1][1] * C[2][2] - C[1][2] * C[2][1], i01 = C[0][2] * C[2][1] - C[0][1] * C[2][2], i02 = C[0][1] * C[1][2] - C[0][2] * C[1][1];
    const det = C[0][0] * i00 + C[1][0] * i01 + C[2][0] * i02;
    const a00 = i00 / det, a01 = i01 / det, a02 = i02 / det, a11 = (C[0][0] * C[2][2] - C[0][2] * C[2][0]) / det;
    const a12 = (C[0][2] * C[1][0] - C[0][0] * C[1][2]) / det, a22 = (C[0][0] * C[1][1] - C[0][1] * C[1][0]) / det;
    const lo = [], hi = [];  // 3 sigma along each axis: sqrt(9 C_kk), capped
    for (let k = 0; k < 3; k++) {
      const e = Math.min(cap, 3 * Math.sqrt(C[k][k]) / v), c = (p[k] - o[k]) / v;
      lo.push(Math.max(0, Math.ceil(c - e))); hi.push(Math.min(n[k] - 1, Math.floor(c + e)));
    }
    for (let k = lo[2]; k <= hi[2]; k++) { const dz = o[2] + k * v - p[2];
      for (let j = lo[1]; j <= hi[1]; j++) { const dy = o[1] + j * v - p[1];
        for (let i = lo[0]; i <= hi[0]; i++) { const dx = o[0] + i * v - p[0];
          const q = a00 * dx * dx + a11 * dy * dy + a22 * dz * dz + 2 * (a01 * dx * dy + a02 * dx * dz + a12 * dy * dz);
          if (q > 9) continue;
          const c = at(i, j, k), a = op * Math.exp(-0.5 * q);
          acc[c] += a; col[0][c] += a * rgb[0]; col[1][c] += a * rgb[1]; col[2][c] += a * rgb[2];
        } } }
  }
  const occ = new Uint8Array(N);
  let nOcc = 0;
  for (let q = 0; q < N; q++) if (acc[q] >= tau) { occ[q] = 1; nOcc++; }
  // the colors, blurred now (acc and col then go): K voxels reach the splats in front of the surface
  const K = Math.ceil(close + inset + 1), rgb8 = new Uint8Array(3 * N), near = new Uint8Array(N);  // near: splats within K
  for (const A of [acc, ...col]) boxBlur(n, A, K);
  for (let q = 0; q < N; q++) {
    if (acc[q] >= tau) near[q] = 1;
    for (let c = 0; c < 3; c++) rgb8[3 * q + c] = acc[q] > 0 ? Math.round(255 * col[c][q] / acc[q]) : 128;
  }
  phase('splats');

  // the path: camera centers in order, joined unless a step is over 5x the median
  const order = path ?? cameras.map((_, i) => i), centers = order.map(i => cameras[i].center);
  const steps = centers.slice(1).map((c, i) => Math.hypot(c[0] - centers[i][0], c[1] - centers[i][1], c[2] - centers[i][2]));
  const medStep = steps.slice().sort((a, b) => a - b)[steps.length >> 1] ?? 0;
  const seed = new Uint8Array(N);
  for (let i = 0; i < centers.length; i++) {
    const a = centers[i], b = centers[i + 1], m = b && steps[i] <= 5 * medStep ? Math.ceil(steps[i] / (0.5 * v)) : 0;
    for (let s = 0; s <= m; s++) {
      const t = m ? s / m : 0, q = cellOf([0, 1, 2].map(k => a[k] + t * ((b?.[k] ?? a[k]) - a[k])));
      if (q >= 0) seed[q] = 1;
    }
  }

  // 2. closing: the dilated occupancy, except within `close` of the path (the raw occupancy there)
  const dist = new Float32Array(N), blocked = new Uint8Array(N), R2 = close * close;
  edt(n, q => seed[q] === 1, dist);
  for (let q = 0; q < N; q++) blocked[q] = dist[q] <= R2 ? 2 : 0;
  edt(n, q => occ[q] === 1, dist);
  for (let q = 0; q < N; q++) blocked[q] = seed[q] ? 0 : blocked[q] === 2 ? occ[q] : dist[q] <= R2 ? 1 : 0;
  phase('closing');

  // 3. the seen exterior (seen: the path, the cells rays cross) and the rays' first hits. A ray stops at its first closed
  // cell, but a run of closed cells with no splats in them (occ 0) longer than `passage` voxels is a passage the closing
  // filled (a narrow slot seen lengthwise: the closing would lay a lid over it), crossed: carved, on to its end
  const seen = seed, hit = new Uint8Array(N), RH = Math.round(rayW * 9 / 16);
  const reach = 4 * Math.max(nx, ny, nz) * v;
  let passages = 0;
  for (let ci = 0; ci < cameras.length; ci += rayStride) {
    const c = cameras[ci], w = c.view, C = c.center;
    for (let py = 0; py < RH; py++) for (let px = 0; px < rayW; px++) {
      const x = ((px + 0.5) * c.width / rayW - c.cx) / c.fx, y = ((py + 0.5) * c.height / RH - c.cy) / c.fy;
      const d = [0, 1, 2].map(k => w[4 * k] * x + w[4 * k + 1] * y + w[4 * k + 2]), dl = Math.hypot(...d);
      const crossed = [];
      let free = false, inside = false;
      const cellAt = (t) => cellOf([C[0] + t * d[0] / dl, C[1] + t * d[1] / dl, C[2] + t * d[2] / dl]);
      for (let t = 0; t < reach; t += 0.5 * v) {
        const q = cellAt(t);
        if (q < 0) { if (inside) break; continue; }
        inside = true;
        if (blocked[q]) {
          if (!free) continue;  // out of the closed cells it starts in first
          if (!occ[q]) {
            const run = [];
            let u = t;
            for (; u < reach; u += 0.5 * v) { const r = cellAt(u); if (r < 0 || !blocked[r] || occ[r]) break; run.push(r); }
            if (0.5 * run.length > passage) { crossed.push(...run); t = u - 0.5 * v; passages++; continue; }
          }
          hit[q] = 1;
          break;
        }
        free = true;
        crossed.push(q);
      }
      for (const q of crossed) seen[q] = 1;
    }
  }
  phase('rays');
  // solid (reusing blocked): within `depth` voxels of a hit and not seen
  edt(n, q => hit[q] === 1, dist);
  let nSolid = 0;
  for (let q = 0; q < N; q++) { blocked[q] = !seen[q] && dist[q] <= depth * depth ? 1 : 0; nSolid += blocked[q]; }
  const solid = blocked, unseen = (q) => !solid[q] && !seen[q];

  // 4. the surface close + inset voxels from the seen exterior; the unseen exterior outside (+1)
  edt(n, q => seen[q] === 1, dist);
  const L = close + inset, band = 2;
  for (let q = 0; q < N; q++) {
    grid.tsdf[q] = unseen(q) ? 1 : Math.max(-1, Math.min(1, (L - Math.sqrt(dist[q])) / band));
    const i = q % nx, j = ((q / nx) | 0) % ny, k = (q / (nx * ny)) | 0;
    grid.weight[q] = i && j && k && i < nx - 1 && j < ny - 1 && k < nz - 1 ? 1 : 0;
  }
  const mesh = grid.extract({ minWeight: 1 }), P = mesh.positions, nv = P.length / 3;
  phase('surface');
  // dropped: vertices with a cell corner in the unseen exterior, or no splats within K
  const drop = new Uint8Array(nv), colors = new Uint8Array(3 * nv);
  let cut = 0, empty = 0;
  for (let i = 0; i < nv; i++) {
    const c = [0, 1, 2].map(k => Math.floor((P[3 * i + k] - o[k]) / v));
    for (let m = 0; m < 8 && !drop[i]; m++) {
      const ii = c[0] + (m & 1), jj = c[1] + ((m >> 1) & 1), kk = c[2] + ((m >> 2) & 1);
      if (ii < nx && jj < ny && kk < nz && unseen(at(ii, jj, kk))) drop[i] = 1;
    }
    const q = cellOf(P.subarray(3 * i, 3 * i + 3));
    if (q >= 0) for (let k = 0; k < 3; k++) colors[3 * i + k] = rgb8[3 * q + k];
    if (!drop[i] && !(q >= 0 && near[q])) drop[i] = 2;
  }
  const F = mesh.faces, kept = [];
  for (let f = 0; f < F.length; f += 3) {
    const a = drop[F[f]], b = drop[F[f + 1]], c = drop[F[f + 2]];
    if (!a && !b && !c) kept.push(F[f], F[f + 1], F[f + 2]);
    else if (a === 1 || b === 1 || c === 1) cut++; else empty++;
  }
  mesh.faces = new Uint32Array(kept);
  mesh.colors = colors;
  phase('faces');
  return { mesh, info: { times, grid: n, voxel: v, occupied: nOcc / N, solid: nSolid / N, passages, cut, empty,
                         vertices: nv, faces: kept.length / 3, ms: performance.now() - t0 } };
}
