// edit.js — The editor's floater tools (ui/edit_panel.js, DESIGN.md E16): the splat under a pixel, the cluster grown
// from it, and every free-standing cluster of the scene. CPU, on the trainer's splats (SPLAT_FLOATS rows).
// Splats i and j touch when either's centre is within `reach` sigmas of the other in the other's own shape (its
// Mahalanobis distance: a flat splat reaches far along its plane, little across it). A cluster is the splats
// touching, through each other; a floater is a cluster of at most maxSize splats, apart from the scene (the largest
// cluster). A splat's r, its largest axis (1 sigma), bounds how far it reaches.
import { SPLAT_FLOATS } from './colmap.js';
import { unionFind } from './math.js';

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
// isolation: Find floater clusters keeps the clusters still apart from the scene at this many times the reach;
// minSplats: and of this many splats at least (a scene has thousands of lone splats apart)
export const EDIT_DEFAULTS = { reach: 2, minAlpha: 0.02, maxShare: 0.02, minSize: 100, isolation: 2, minSplats: 3 };

// Splat i's opacity (as the render: at most 0.99) and its reach r (exp of its largest log scale, x scaleMul)
export const splatAlpha = (params, i) => Math.min(0.99, sigmoid(params[i * SPLAT_FLOATS + 3]));
export function splatRadius(params, i, scaleMul = 1) {
  const b = i * SPLAT_FLOATS;
  return scaleMul * Math.exp(Math.min(4, Math.max(params[b + 4], params[b + 5], params[b + 6])));
}

// The largest cluster a floater may be: maxShare of the splats, at least minSize
export const maxClusterSize = (count, { maxShare = EDIT_DEFAULTS.maxShare, minSize = EDIT_DEFAULTS.minSize } = {}) =>
  Math.max(minSize, Math.floor(maxShare * count));

// The splat seen at pixel (px, py) of the view (world-to-camera `view`, column-major 4x4; fx, fy, cx, cy; W x H): of
// the splats whose 2.5-sigma ellipse (splats_render's EWA, its 0.3 px^2 dilation) covers it, the one adding the most
// to the pixel's colour, front to back (alpha x the transmittance before it). { index, depth, weight } | null.
export function pickSplat(params, count, { view, fx, fy, cx, cy, W, H }, px, py, { scaleMul = 1, znear = 0.05 } = {}) {
  const v = view, limX = 1.3 * Math.max(cx, W - cx) / fx, limY = 1.3 * Math.max(cy, H - cy) / fy;
  const hits = [];
  for (let i = 0; i < count; i++) {
    const b = i * SPLAT_FLOATS, x = params[b], y = params[b + 1], z = params[b + 2];
    const vz = v[2] * x + v[6] * y + v[10] * z + v[14];
    if (!(vz > znear)) continue;
    const vx = v[0] * x + v[4] * y + v[8] * z + v[12], vy = v[1] * x + v[5] * y + v[9] * z + v[13];
    const dx = px - (fx * vx / vz + cx), dy = py - (fy * vy / vz + cy);
    // a cheap bound first: 2.5 sigma of the largest axis, at that depth (and the dilation)
    const r = splatRadius(params, i, scaleMul), rp = 2.5 * (Math.max(fx, fy) * r / vz + 1);
    if (dx * dx + dy * dy > rp * rp) continue;
    const c = cov2d(params, b, v, vx, vy, vz, fx, fy, limX, limY, scaleMul);
    const det = c[0] * c[2] - c[1] * c[1];
    if (!(det > 0)) continue;
    const m = (c[2] * dx * dx - 2 * c[1] * dx * dy + c[0] * dy * dy) / det;  // Mahalanobis^2
    if (m > 6.25) continue;
    const a = splatAlpha(params, i) * Math.exp(-0.5 * m);
    if (a >= 1 / 255) hits.push([vz, i, a]);
  }
  hits.sort((p, q) => p[0] - q[0]);
  let T = 1, best = null;
  for (const [depth, index, a] of hits) {
    const w = a * T;
    if (!best || w > best.weight) best = { index, depth, weight: w };
    T *= 1 - a;
    if (T < 1e-3) break;
  }
  return best;
}

// Splat b's screen covariance [xx, xy, yy] (+ 0.3 px^2), as splats_render: T = J W, T Sigma T^T
function cov2d(p, b, v, vx, vy, vz, fx, fy, limX, limY, scaleMul) {
  const n = Math.hypot(p[b + 8], p[b + 9], p[b + 10], p[b + 11]) || 1;
  const w = p[b + 8] / n, x = p[b + 9] / n, y = p[b + 10] / n, z = p[b + 11] / n;
  const R = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
             2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
             2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
  const s = [0, 1, 2].map(k => (scaleMul * Math.exp(Math.min(4, Math.max(-12, p[b + 4 + k])))) ** 2);
  const iz = 1 / vz, tx = Math.min(limX, Math.max(-limX, vx * iz)), ty = Math.min(limY, Math.max(-limY, vy * iz));
  const J0 = [fx * iz, 0, -fx * tx * iz], J1 = [0, fy * iz, -fy * ty * iz];
  // T rows: J row x W (W[r][c] = v[c * 4 + r])
  const T0 = [0, 1, 2].map(c => J0[0] * v[c * 4] + J0[1] * v[c * 4 + 1] + J0[2] * v[c * 4 + 2]);
  const T1 = [0, 1, 2].map(c => J1[0] * v[c * 4] + J1[1] * v[c * 4 + 1] + J1[2] * v[c * 4 + 2]);
  // (T R) diag(s) (T R)^T
  const A0 = [0, 1, 2].map(k => T0[0] * R[k] + T0[1] * R[3 + k] + T0[2] * R[6 + k]);
  const A1 = [0, 1, 2].map(k => T1[0] * R[k] + T1[1] * R[3 + k] + T1[2] * R[6 + k]);
  let xx = 0.3, xy = 0, yy = 0.3;
  for (let k = 0; k < 3; k++) { xx += A0[k] * A0[k] * s[k]; xy += A0[k] * A1[k] * s[k]; yy += A1[k] * A1[k] * s[k]; }
  return [xx, xy, yy];
}

// The splats of opacity minAlpha or more in a grid of levels: level L of cell size h0 x 2^L holds the splats of r up
// to h0 x 2^L / reach, h0 from the 1st percentile of r (at least the median's / 256), so a splat's touching splats
// of each level are in the cells within reach x max(its r, that level's largest r) of it (a touch is within reach x
// the larger r: 3 cells a side at its own level and above). neighbours(i, fn) calls fn(j) for each splat j touching i
// (j != i); with coarser, only those of i's level and above: each touching pair found from its smaller splat (enough
// for components).
export class SplatGrid {
  constructor(params, count, { reach = EDIT_DEFAULTS.reach, minAlpha = EDIT_DEFAULTS.minAlpha, scaleMul = 1 } = {}) {
    this.params = params; this.count = count; this.reach = reach;
    this.r = new Float32Array(count);
    this.inv = new Float32Array(count * 6);  // each splat's inverse covariance: xx, yy, zz, xy, xz, yz
    this.active = new Uint8Array(count);
    const rs = [];
    for (let i = 0; i < count; i++) {
      this.r[i] = splatRadius(params, i, scaleMul);
      invCov(params, i, scaleMul, this.inv, i * 6);
      if (splatAlpha(params, i) >= minAlpha) { this.active[i] = 1; rs.push(this.r[i]); }
    }
    rs.sort((a, b) => a - b);
    const rLo = rs.length ? Math.max(rs[Math.floor(0.01 * (rs.length - 1))], rs[rs.length >> 1] / 256) : 1;
    this.h0 = reach * rLo || 1;
    this.levels = [];  // { h, rMax, cells: Map(key -> [i...]) }
    this.level = new Uint8Array(count);
    for (let i = 0; i < count; i++) {
      const L = this.level[i] = Math.min(255, Math.max(0, Math.ceil(Math.log2(reach * this.r[i] / this.h0))));
      if (!this.active[i]) continue;
      while (this.levels.length <= L) this.levels.push({ h: this.h0 * 2 ** this.levels.length, rMax: 0, cells: new Map() });
      const lv = this.levels[L];
      lv.rMax = Math.max(lv.rMax, this.r[i]);
      const k = this._key(lv.h, i);
      const cell = lv.cells.get(k);
      if (cell) cell.push(i); else lv.cells.set(k, [i]);
    }
  }
  // cell coordinates clamped to +-65535 (17 bits each: one exact double); a clamped cell holds more, checked exactly
  static key(ix, iy, iz) {
    const c = (t) => Math.min(65535, Math.max(-65535, t)) + 65536;
    return (c(ix) * 131072 + c(iy)) * 131072 + c(iz);
  }
  _key(h, i) {
    const p = this.params, b = i * SPLAT_FLOATS;
    return SplatGrid.key(Math.floor(p[b] / h), Math.floor(p[b + 1] / h), Math.floor(p[b + 2] / h));
  }
  touch(i, j) {
    const p = this.params, a = i * SPLAT_FLOATS, b = j * SPLAT_FLOATS, R2 = this.reach * this.reach;
    const dx = p[a] - p[b], dy = p[a + 1] - p[b + 1], dz = p[a + 2] - p[b + 2], rr = Math.max(this.r[i], this.r[j]);
    if (dx * dx + dy * dy + dz * dz > R2 * rr * rr) return false;  // beyond either's largest axis
    const m = (k) => { const q = this.inv;
      return q[k] * dx * dx + q[k + 1] * dy * dy + q[k + 2] * dz * dz + 2 * (q[k + 3] * dx * dy + q[k + 4] * dx * dz + q[k + 5] * dy * dz); };
    return m(i * 6) <= R2 || m(j * 6) <= R2;
  }
  neighbours(i, fn, coarser = false) {
    const p = this.params, b = i * SPLAT_FLOATS, x = p[b], y = p[b + 1], z = p[b + 2];
    for (let L = coarser ? this.level[i] : 0; L < this.levels.length; L++) {
      const lv = this.levels[L];
      if (!lv.cells.size) continue;
      const rad = this.reach * Math.max(this.r[i], lv.rMax), h = lv.h;
      const x0 = Math.floor((x - rad) / h), x1 = Math.floor((x + rad) / h);
      const y0 = Math.floor((y - rad) / h), y1 = Math.floor((y + rad) / h);
      const z0 = Math.floor((z - rad) / h), z1 = Math.floor((z + rad) / h);
      const visit = (cell) => { for (const j of cell) if (j !== i && this.touch(i, j)) fn(j); };
      if ((x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1) > lv.cells.size) {
        for (const cell of lv.cells.values()) visit(cell);  // fewer cells held than in range
        continue;
      }
      for (let ix = x0; ix <= x1; ix++) for (let iy = y0; iy <= y1; iy++) for (let iz = z0; iz <= z1; iz++) {
        const cell = lv.cells.get(SplatGrid.key(ix, iy, iz));
        if (cell) visit(cell);
      }
    }
  }
}

// Splat i's inverse covariance R diag(1 / s^2) R^T into out[o..o+5] (xx, yy, zz, xy, xz, yz)
function invCov(p, i, scaleMul, out, o) {
  const b = i * SPLAT_FLOATS, n = Math.hypot(p[b + 8], p[b + 9], p[b + 10], p[b + 11]) || 1;
  const w = p[b + 8] / n, x = p[b + 9] / n, y = p[b + 10] / n, z = p[b + 11] / n;
  const R = [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
             2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
             2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
  const t = [0, 1, 2].map(k => (scaleMul * Math.exp(Math.min(4, Math.max(-12, p[b + 4 + k])))) ** -2);
  const e = (r, c) => R[3 * r] * R[3 * c] * t[0] + R[3 * r + 1] * R[3 * c + 1] * t[1] + R[3 * r + 2] * R[3 * c + 2] * t[2];
  out[o] = e(0, 0); out[o + 1] = e(1, 1); out[o + 2] = e(2, 2); out[o + 3] = e(0, 1); out[o + 4] = e(0, 2); out[o + 5] = e(1, 2);
}

// The cluster grown from splat `seed` through the touching splats: { indices (sorted), attached } where attached is
// true when it grew past maxSize splats (it touches the scene; indices then stop at maxSize + 1). The seed is in it
// whatever its opacity.
export function growCluster(grid, seed, maxSize) {
  const seen = new Set([seed]), queue = [seed];
  for (let q = 0; q < queue.length && seen.size <= maxSize; q++) {
    grid.neighbours(queue[q], (j) => {
      if (seen.size > maxSize || seen.has(j)) return;
      seen.add(j); queue.push(j);
    });
  }
  return { indices: Int32Array.from(seen).sort(), attached: seen.size > maxSize };
}

// The grid's components (union-find over the touching pairs): root[i] (-1 for the splats left out), size[root]
export function components(grid) {
  const n = grid.count, { find, union } = unionFind(n);
  for (let i = 0; i < n; i++) {
    if (!grid.active[i]) continue;
    grid.neighbours(i, (j) => { if (grid.active[j]) union(i, j); }, true);
  }
  const root = new Int32Array(n).fill(-1), size = new Int32Array(n);
  for (let i = 0; i < n; i++) if (grid.active[i]) size[root[i] = find(i)]++;
  return { root, size };
}

// The floaters: the clusters of the grid's splats of minSplats to maxSize splats and not the largest (the scene), with
// `isolation` (a grid of a larger reach) only those whose cluster there is no larger than maxSize either (fragments
// of a surface join it there, floaters in the air do not); each { indices, mass, center, extent, near, seen }, the most
// seen first, at most `limit`. seen: with `cameras` (their centres), how much of the nearest camera's view it covers
// (the sum of alpha x r^2 over the cluster's distance to that camera squared: a floater near the path before a far
// sky splat); without,
// its mass (the sum of alpha x r^2). With `small`, also the isolated floaters: every splat of the groups of fewer than
// minSplats, apart as those (small: Int32Array, smallGroups: their count).
export function findClusters(grid, maxSize, { limit = 200, isolation = null, cameras = null, minSplats = 1, small = false } = {}) {
  const { root, size } = components(grid), iso = isolation && components(isolation);
  let largest = -1, groups = 0;
  for (let i = 0; i < grid.count; i++) if (root[i] === i) { groups++; if (largest < 0 || size[i] > size[largest]) largest = i; }
  const members = new Map();
  for (let i = 0; i < grid.count; i++) {
    const r = root[i];
    if (r < 0 || r === largest || size[r] > maxSize || (size[r] < minSplats && !small)) continue;
    const g = members.get(r);
    if (g) g.push(i); else members.set(r, [i]);
  }
  const out = [], lone = [];
  let joined = 0, smallGroups = 0;
  for (const g of members.values()) {
    if (iso && iso.size[iso.root[g[0]]] > maxSize) { if (g.length >= minSplats) joined++; continue; }
    if (g.length < minSplats) { smallGroups++; lone.push(...g); continue; }
    const c = clusterInfo(grid.params, Int32Array.from(g), grid.r);
    c.near = Infinity;
    for (const q of cameras ?? []) c.near = Math.min(c.near, Math.hypot(q[0] - c.center[0], q[1] - c.center[1], q[2] - c.center[2]));
    c.seen = isFinite(c.near) ? c.mass / Math.max(1e-12, c.near * c.near) : c.mass;
    out.push(c);
  }
  out.sort((a, b) => b.seen - a.seen);
  return { clusters: out.slice(0, limit), total: out.length, joined, groups, largest: largest < 0 ? 0 : size[largest],
           ...(small && { small: Int32Array.from(lone).sort(), smallGroups }) };
}

// A cluster's mass (sum of alpha x r^2), its centre (alpha-weighted) and extent (the largest axis of its box, with
// its splats' reach)
export function clusterInfo(params, indices, r = null) {
  let mass = 0, wsum = 0;
  const c = [0, 0, 0], lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const i of indices) {
    const b = i * SPLAT_FLOATS, a = splatAlpha(params, i), ri = r ? r[i] : splatRadius(params, i);
    mass += a * ri * ri; wsum += a;
    for (let k = 0; k < 3; k++) {
      c[k] += a * params[b + k];
      lo[k] = Math.min(lo[k], params[b + k] - ri); hi[k] = Math.max(hi[k], params[b + k] + ri);
    }
  }
  const center = c.map(v => v / (wsum || 1));
  return { indices, mass, center, extent: Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) };
}
