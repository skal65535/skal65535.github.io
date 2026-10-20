// stipple.js — stipple-guided seeding of new splats (experiment E4).
//
// At rebalance time, for one training view:
//   1. density = residual |GT - hat| above its mean (only clearly under-fit pixels)
//   2. sites   = importance sampling + a few weighted Lloyd steps (../stipple/ Voronoi)
//   3. depth   = rendered z_med where the pixel is covered (T_final < 0.5), else the
//                nearest COLMAP point observed in that view, else the view's median depth
//   4. splats  = unprojected sites, GT color, low opacity, isotropic scale from cell area
// Pure CPU: the trainer does the GPU readbacks (GaussOptTrainer._stippleSeeds()).
"use strict";

import { computeVoronoi } from '../../stipple/stipple.js';
import { SPLAT_FLOATS, rgbToSh0 } from './colmap.js';
import { clamp, logit, makeRng } from './math.js';

// Density for the sampler: mean-RGB residual minus its image mean, clamped at 0.
// gt / hat: RGBA8 rows of `bytesPerRow` bytes.
export function residualDensity(gt, hat, W, H, bytesPerRow = W * 4) {
  const res = new Float32Array(W * H);
  let sum = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o = y * bytesPerRow + x * 4;
      const r = (Math.abs(gt[o] - hat[o]) + Math.abs(gt[o + 1] - hat[o + 1]) + Math.abs(gt[o + 2] - hat[o + 2])) / 765;
      res[y * W + x] = r;
      sum += r;
    }
  }
  const mean = sum / (W * H);
  for (let i = 0; i < res.length; i++) res[i] = Math.max(0, res[i] - mean);
  return res;
}

// N stipple sites on `density` (W x H, >= 0): inverse-CDF importance sampling, then
// `lloydIters` density^2-weighted Lloyd steps (site density stays ~ density).
// Returns unit coords and Voronoi cell areas in px (counting only pixels with density > 0,
// so border sites don't get huge cells).
export function stippleSites(density, W, H, N, { lloydIters = 4, seed = 1234 } = {}) {
  const cdf = new Float64Array(W * H);
  let acc = 0;
  for (let i = 0; i < W * H; i++) { acc += density[i]; cdf[i] = acc; }
  if (!(acc > 0) || N <= 0) return { xs: new Float32Array(0), ys: new Float32Array(0), area: new Float32Array(0) };

  const rand = makeRng(seed);
  const pts = [];
  for (let n = 0; n < N; n++) {
    const u = rand() * acc;
    let lo = 0, hi = W * H - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid] < u) lo = mid + 1; else hi = mid;
    }
    pts.push({ x: ((lo % W) + rand()) / W, y: (((lo / W) | 0) + rand()) / H });
  }

  let area = null;
  for (let it = 0; it <= lloydIters; it++) {
    const owner = computeVoronoi(W, H, pts);
    const m = new Float64Array(pts.length), mx = new Float64Array(pts.length), my = new Float64Array(pts.length);
    area = new Float32Array(pts.length);
    for (let y = 0, i = 0; y < H; y++) {
      for (let x = 0; x < W; x++, i++) {
        const c = owner[i], r = density[i];
        if (r > 0) area[c]++;  // cell area restricted to under-fit pixels
        // Centroids weighted by r^2: a 2D CVT has site density ~ sqrt(weight), so this keeps
        // seeds proportional to the error mass r (as the importance sampling started) while
        // Lloyd still regularizes their spacing.
        const w = r * r;
        m[c] += w; mx[c] += w * (x + 0.5); my[c] += w * (y + 0.5);
      }
    }
    if (it === lloydIters) break;  // last pass only measures the cells
    // A site whose cell holds no pixel mass (sub-pixel cells in dense clusters) stays put
    for (let n = 0; n < pts.length; n++) {
      if (m[n] > 0) pts[n] = { x: mx[n] / m[n] / W, y: my[n] / m[n] / H };
    }
  }
  const xs = new Float32Array(pts.length), ys = new Float32Array(pts.length);
  for (let n = 0; n < pts.length; n++) { xs[n] = pts[n].x; ys[n] = pts[n].y; }
  return { xs, ys, area };
}

// Coarse per-view depth grid from the COLMAP points *observed* in that view (obs2D), so
// occluded points don't leak through. Cells hold the median depth (0 = empty).
// (sx, sy) map native image pixels to training pixels.
export function buildViewDepthGrid(img, points3D, sx, sy, W, H, cell = 16) {
  const gw = Math.ceil(W / cell), gh = Math.ceil(H / cell);
  const buckets = Array.from({ length: gw * gh }, () => []);
  const { positions, pointIndexById } = points3D;
  const v = img.view;
  const all = [];
  for (let k = 0; k < img.obs2D.length; k += 3) {
    const idx = pointIndexById.get(img.obs2D[k + 2]);
    if (idx === undefined) continue;
    const X = positions[3 * idx], Y = positions[3 * idx + 1], Z = positions[3 * idx + 2];
    const z = v[2] * X + v[6] * Y + v[10] * Z + v[14];
    if (!(z > 0.01)) continue;
    const gx = Math.floor(img.obs2D[k] * sx / cell), gy = Math.floor(img.obs2D[k + 1] * sy / cell);
    if (gx < 0 || gy < 0 || gx >= gw || gy >= gh) continue;
    buckets[gx + gy * gw].push(z);
    all.push(z);
  }
  const median = a => { a.sort((p, q) => p - q); return a[a.length >> 1]; };
  const grid = new Float32Array(gw * gh);
  for (let i = 0; i < grid.length; i++) if (buckets[i].length) grid[i] = median(buckets[i]);
  return { grid, gw, gh, cell, median: all.length ? median(all) : 0 };
}

// Depth at training pixel (u, v): nearest non-empty grid cell (ring search), else median.
export function lookupDepth(dg, u, v, maxRing = 6) {
  const cx = clamp(Math.floor(u / dg.cell), 0, dg.gw - 1);
  const cy = clamp(Math.floor(v / dg.cell), 0, dg.gh - 1);
  for (let r = 0; r <= maxRing; r++) {
    let best = 0, bestD = Infinity;
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;  // ring only
        const x = cx + dx, y = cy + dy;
        if (x < 0 || y < 0 || x >= dg.gw || y >= dg.gh) continue;
        const z = dg.grid[x + y * dg.gw], d = dx * dx + dy * dy;
        if (z > 0 && d < bestD) { bestD = d; best = z; }
      }
    }
    if (best > 0) return { z: best, fromPoints: true };
  }
  return { z: dg.median, fromPoints: false };
}

// Training pixel (u, v) at camera depth z -> world position (img.R is row-major world->cam)
export function unprojectPixel(img, fx, fy, cx, cy, u, v, z) {
  const xc = (u - cx) / fx * z, yc = (v - cy) / fy * z;
  const R = img.R, c = img.center;
  return [
    c[0] + R[0] * xc + R[3] * yc + R[6] * z,
    c[1] + R[1] * xc + R[4] * yc + R[7] * z,
    c[2] + R[2] * xc + R[5] * yc + R[8] * z,
  ];
}

// Seed splats for one view. pix: Float32Array [T, z_med, last_k, 0] per pixel; gt: RGBA8 rows.
// Returns { params: Float32Array(n * SPLAT_FLOATS), count, covered, fromPoints, fromMedian }.
export function makeSeedSplats({
  img, sites, pix, gt, bytesPerRow, W, H, fx, fy, cx, cy, depthGrid,
  opacity = 0.2, maxRadiusPx = 8,
}) {
  const n = sites.xs.length;
  const params = new Float32Array(n * SPLAT_FLOATS);
  const f = 0.5 * (fx + fy);
  let count = 0, covered = 0, fromPoints = 0, fromMedian = 0;
  for (let k = 0; k < n; k++) {
    const u = sites.xs[k] * W, v = sites.ys[k] * H;
    const px = Math.min(W - 1, u | 0), py = Math.min(H - 1, v | 0);
    const p = (py * W + px) * 4;
    let z;
    if (pix[p] < 0.5 && pix[p + 1] > 0.05) {
      z = pix[p + 1];
      covered++;
    } else {
      const d = lookupDepth(depthGrid, u, v);
      z = d.z;
      if (d.fromPoints) fromPoints++; else fromMedian++;
    }
    if (!(z > 0.05)) continue;
    const pos = unprojectPixel(img, fx, fy, cx, cy, u, v, z);
    const rPx = clamp(Math.sqrt(sites.area[k] / Math.PI), 0.5, maxRadiusPx);
    const o = py * bytesPerRow + px * 4;
    const b = count * SPLAT_FLOATS;
    params.set(pos, b);
    params[b + 3]  = logit(opacity);
    params[b + 4]  = params[b + 5] = params[b + 6] = Math.log(rPx * z / f);
    params[b + 8]  = 1.0;  // identity quaternion
    for (let c = 0; c < 3; c++) params[b + 12 + c] = rgbToSh0(gt[o + c] / 255);
    count++;
  }
  return { params: params.subarray(0, count * SPLAT_FLOATS), count, covered, fromPoints, fromMedian };
}
