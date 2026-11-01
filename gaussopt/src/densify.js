// densify.js — The outer loop that grows and prunes the splats at each rebalance (reconcileAndGrow; DESIGN.md
// §5): WLBG prune / split / clone, floater removal, Monte-Carlo's relocation, overlap merge, stipple seeds, the cap;
// then the octree is built again over the result.
"use strict";

import { SPLAT_FLOATS } from './colmap.js';
import { clamp, logit, sigmoid, quatToMat3, projectPinhole } from './math.js';
import { MOMENT_FLOATS } from './octree.js';

// Returns { count, params (new, compacted), srcIdx, splitVectors, stats }. srcIdx[k] is the
// input splat whose optimizer state new splat k inherits (-1 = fresh), in the order BEFORE the
// final build(); compose with octree.lastPerm. `params` is also modified in place (opacity of
// MCMC teleport parents and of decayed occluders), so the caller should switch to the result.
export function reconcileAndGrow(octree, params, count, moments, images, opts = {}) {
  const {
    maxSplats      = 100000,
    pruneOpacity   = 0.008,
    hysteresis     = 0.35,
    enableFloaters = true,
    floaterDepth   = 'observed',  // nearCameraCone's depth
    floaterSpare   = null,        // nearCameraCone's spare: { positions, count, radius } (the SfM points)
    splitMinViews  = 0,           // > 0: no split / clone of a splat seen from fewer views (N_eff / visitsPerView)
    visitsPerView  = 1,           // steps per training view since the moments were zeroed: N_eff = W^2 / S counts
                                  // view visits (S grows each step), so a photo set's one view counts K / n times
    splitMode      = 'error',     // 'error': along the error moments (splitSplat); 'shape': within the splat (splitByShape)
    enableSplit    = true,
    enableMerge    = true,
    optMethod      = 'adam',  // 'adam' | 'monte_carlo'
    relocateBy     = 'opacity',  // monte_carlo: the live splats the dead ones move onto, drawn ~ 'opacity' or 'error' (R)
    relocateMode   = 'copy',  // ... as copies in place (3DGS-MCMC) or 'shape': split within their shape (brush, relocate)
    seedSplats     = null,    // Float32Array of splats to append (stipple seeding)
    repairMask     = null,    // Uint8Array(count), 1: in a stuck region (regions.js RegionTracker, E7): its splits
    repairBoost    = 0,       // need R > Tu / (1 + repairBoost), and monte_carlo relocates onto it (1 + repairBoost) as often
    keepScore      = null,    // Float32Array(count): over maxSplats (the cap lease's consolidation, schedule.js
                              // splatBudget), the rows of the lowest scores go (default: the window's W)
    bounds         = null,    // { center, halfSize }: a center out of this cube is dead, as a transparent splat
                              // (monte_carlo: its noise has no bound, trainer.js sceneBox)
  } = opts;
  const err = errorStats(moments, count, hysteresis);
  // stipple seeds come before splits: the room they need is kept free (at the cap, the room the prunes free)
  const nSeeds = seedSplats ? seedSplats.length / SPLAT_FLOATS : 0;
  const out = new KeptSplats(params, count);
  const stats = { split: 0, cloned: 0, pruned: 0, floaters: 0, floaterRules: [0, 0, 0], merged: 0, teleported: 0, seeded: 0 };
  // monte_carlo: the dead splats (pruned, floaters) are not dropped but moved onto live ones, after the pass
  let dead = 0;
  const teleport = () => optMethod === 'monte_carlo' && ++dead > 0;
  const nearCamera = nearCameraCone(images, { depth: floaterDepth, spare: floaterSpare });

  for (let i = 0; i < count; i++) {
    const b = i * SPLAT_FLOATS, m = i * MOMENT_FLOATS;
    const alpha = sigmoid(params[b + 3]);
    const maxS = Math.exp(Math.max(params[b + 4], params[b + 5], params[b + 6]));
    const leaf = octree.leaves[clamp(params[b + 7] | 0, 0, octree.leaves.length - 1)];
    const W = moments[m], S = moments[m + 1], R = moments[m + 2];

    // A. Prune transparent / oversized splats, and those out of `bounds`
    const away = bounds && Math.max(Math.abs(params[b] - bounds.center[0]), Math.abs(params[b + 1] - bounds.center[1]),
                                    Math.abs(params[b + 2] - bounds.center[2])) > bounds.halfSize;
    if (alpha < pruneOpacity || maxS > octree.sceneRadius * 0.5 || away) {
      if (!teleport()) stats.pruned++;
      continue;
    }
    // B. Floaters (removed, or teleported); high-error occluders lose opacity
    if (enableFloaters) {
      const rule = isFloater(params, b, alpha, leaf, W, S, R, nearCamera);
      if (rule) {
        stats.floaterRules[rule - 1]++;
        if (!teleport()) stats.floaters++;
        continue;
      }
      if (W > 1.0 && (R / W) > 0.22 && alpha > 0.15) params[b + 3] = logit(alpha * 0.65);
    }
    // C. Split when R > Tu.
    const Tu = repairMask?.[i] ? err.Tu / (1 + repairBoost) : err.Tu;
    let shouldSplit = R > Tu && err.Tu > 1e-4;
    if (shouldSplit && splitMinViews > 0 && !(S > 1e-8 && (W * W) / S / visitsPerView >= splitMinViews)) {
      shouldSplit = false;
      stats.splitTooFewViews = (stats.splitTooFewViews ?? 0) + 1;
    }
    // budget: splats kept so far + the ones still to visit + this one's extra child + the seeds, or the
    // dead ones relocated (monte_carlo) when there are more of them
    if (enableSplit && shouldSplit && out.n + (count - i) + 1 + Math.max(nSeeds, dead) <= maxSplats) {
      const how = splitMode === 'shape' ? splitByShape(out, i) : splitSplat(out, i, moments, maxS, octree.sceneRadius);
      if (how) { stats[how]++; continue; }
    }
    out.add(i, i);
  }

  // monte_carlo: the dead become the seeds first, the others copies of live splats, within the cap
  const toCopy = Math.max(0, Math.min(dead - nSeeds, maxSplats - out.n - nSeeds));
  if (toCopy > 0) stats.teleported = relocate(out, toCopy, relocateBy === 'error' ? moments : null, repairMask, repairBoost, relocateMode);
  stats.pruned += Math.max(0, dead - nSeeds) - stats.teleported;
  if (enableMerge) stats.merged = mergeOverlaps(out);
  stats.trimmed = trimToCap(out, maxSplats, keepScore ?? moments, keepScore ? 1 : MOMENT_FLOATS);
  const res = compactWithSeeds(out, seedSplats, maxSplats);
  stats.seeded = res.nSeeds;
  octree.lastStats = stats;
  octree.build(res.params, res.count, images);
  return { count: res.count, params: res.params, srcIdx: res.srcIdx, splitVectors: new Float32Array(out.splitVectors), stats };
}

// Clamp a world-space offset o from the center of splat p (params slice) to the box of
// +-c * scale along each of the splat's own axes, so a child moved by o stays on the splat:
// a flat disc's children stay in its plane, a needle's along its length.
function clampToSplat(p, o, c) {
  const Rm = quatToMat3([p[8], p[9], p[10], p[11]]);  // row-major; axis k = column k
  const out = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const lim = c * Math.exp(p[4 + k]);
    const ok = clamp(Rm[k] * o[0] + Rm[3 + k] * o[1] + Rm[6 + k] * o[2], -lim, lim);
    out[0] += Rm[k] * ok; out[1] += Rm[3 + k] * ok; out[2] += Rm[6 + k] * ok;
  }
  return out;
}

// ── reconcileAndGrow() phases ────────────────────────────────────────────────

// Mean error mass over visible splats (W > 1e-3) -> the split threshold Tu, (1 + hysteresis) times it
function errorStats(moments, count, hysteresis) {
  let totalR = 0, activeCount = 0;
  for (let i = 0; i < count; i++) {
    const W = moments[i * MOMENT_FLOATS], R = moments[i * MOMENT_FLOATS + 2];
    if (!(W > 1e-3)) continue;
    totalR += R;
    activeCount++;
  }
  const avgR = activeCount > 0 ? totalR / activeCount : 0;
  return { avgR, Tu: (1 + hysteresis) * avgR };
}

// Output splats of the pass, in one growable buffer (not an array per splat: too slow at
// 100k+ splats): row k of `buf`, its source input splat src[k] (-1 = new) and
// alive[k] (0 = merged away); keptOf maps an input splat to its row.
class KeptSplats {
  constructor(params, count) {
    this.params = params;
    this.n = 0;
    this._grow(count + (count >> 2) + 64);
    this.keptOf = new Int32Array(count).fill(-1);
    this.splitVectors = [];  // split axes, for the 3D overlay
  }
  _grow(cap) {
    const buf = new Float32Array(cap * SPLAT_FLOATS), src = new Int32Array(cap);
    if (this.buf) { buf.set(this.buf.subarray(0, this.n * SPLAT_FLOATS)); src.set(this.src.subarray(0, this.n)); }
    this.buf = buf; this.src = src; this.cap = cap;
  }
  // a new row (its src -1: a new splat), the buffers grown when full; returns it
  newRow() {
    if (this.n === this.cap) this._grow(2 * this.cap);
    this.src[this.n] = -1;
    return this.n++;
  }
  // a copy of input splat i as a new row (src = i: its kept copy, -1: a new splat); returns the row
  add(i, src) {
    const k = this.newRow();
    this.buf.set(this.params.subarray(i * SPLAT_FLOATS, (i + 1) * SPLAT_FLOATS), k * SPLAT_FLOATS);
    this.src[k] = src;
    if (src >= 0) this.keptOf[src] = k;
    return k;
  }
  row(k) { return this.buf.subarray(k * SPLAT_FLOATS, (k + 1) * SPLAT_FLOATS); }
}

// monte_carlo (3DGS-MCMC, Kheradmand et al. 2024, §4.2): `n` dead splats moved onto live rows of `out`,
// drawn with probability ~ their opacity (moments null) or their error mass R (the moments; new rows: 0).
// A row drawn j times becomes N = j + 1 copies at its place whose opacity and scale keep the look:
// o' = 1 - (1 - o)^(1/N), s' = s o / sum_{i=1..N} sum_{k=0..i-1} C(i-1, k) (-1)^k o'^(k+1) / sqrt(k+1).
// The copies differ in their sort order, so their gradients split them. Returns how many moved.
// mask (by input splat, E7's stuck regions): those rows drawn (1 + boost) times as often.
// mode 'shape' (brush): each of n distinct rows drawn splits into two halves within its shape (splitRowByShape);
// a row is drawn again only when there are fewer rows than n.
export function relocate(out, n, moments = null, mask = null, boost = 0, mode = 'copy') {
  const rows = [], cdf = [];
  let acc = 0;
  for (let k = 0; k < out.n; k++) {
    let w = moments ? (out.src[k] >= 0 ? moments[out.src[k] * MOMENT_FLOATS + 2] : 0) : sigmoid(out.buf[k * SPLAT_FLOATS + 3]);
    if (mask && out.src[k] >= 0 && mask[out.src[k]]) w *= 1 + boost;
    if (!(w > 0)) continue;
    acc += w; rows.push(k); cdf.push(acc);
  }
  if (!(acc > 0)) return 0;
  const draw = () => {
    const u = Math.random() * acc;
    let lo = 0, hi = cdf.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
    return rows[lo];
  };
  if (mode === 'shape') {
    const picked = new Set();
    for (let tries = 0; picked.size < Math.min(n, rows.length) && tries < 20 * n; tries++) picked.add(draw());
    const order = [...picked];
    for (let j = 0; j < n; j++) splitRowByShape(out, order[j % order.length]);
    return n;
  }
  const picks = new Map();
  for (let j = 0; j < n; j++) { const k = draw(); picks.set(k, (picks.get(k) ?? 0) + 1); }
  for (const [k, j] of picks) {
    const N = j + 1, row = out.row(k), [o, ds] = relocatedOpacityScale(sigmoid(row[3]), N);
    row[3] = logit(o);
    for (let c = 4; c < 7; c++) row[c] += Math.log(ds);
    for (let c = 0; c < j; c++) {
      const r = out.newRow();
      out.buf.copyWithin(r * SPLAT_FLOATS, k * SPLAT_FLOATS, (k + 1) * SPLAT_FLOATS);
    }
  }
  return n;
}

// relocate's opacity and scale factor for one splat of opacity o split into N
export function relocatedOpacityScale(o, N) {
  const on = 1 - Math.pow(1 - o, 1 / N);
  let sum = 0;
  for (let i = 1; i <= N; i++) {
    let binom = 1;  // C(i-1, k)
    for (let k = 0; k < i; k++) {
      sum += binom * (k % 2 ? -1 : 1) * Math.pow(on, k + 1) / Math.sqrt(k + 1);
      binom = binom * (i - 1 - k) / (k + 1);
    }
  }
  return [Math.max(on, 1e-4), o / sum];
}

// Rule 1 of the floater removal: (x, y, z) -> in front of a training camera, closer than 0.65 z(c).
// depth: 'observed', z = zMin (of the points the camera observed: a camera walking past things
// carves the ones other cameras saw near it), or 'view', z = zView (of all the points in its view,
// colmap.js computeCameraDepthStats).
// spare: { positions, count, radius }: a splat within `radius` of one of these points (the SfM's) is not in
// any cone ('observed' counts only the points a camera itself saw, so it can carve an object it passed by
// without observing).
export function nearCameraCone(images, { depth = 'observed', spare = null } = {}) {
  const trainImgs = images ? images.filter(im => !im.isValidation) : [];
  const uv = [0, 0];
  const near = depth === 'view' ? (im) => im.zView ?? im.zMin : (im) => im.zMin;
  const inCone = (im, x, y, z) => {
    const { R, t, camera } = im, zc = projectPinhole(R, t, camera, x, y, z, uv);
    return zc > 0 && zc < 0.65 * near(im) && uv[0] >= 0 && uv[0] <= camera.width && uv[1] >= 0 && uv[1] <= camera.height;
  };
  // Rather than testing every camera for every splat, each camera goes into the grid cells its cone (a pyramid of depth D = 0.65 z)
  // can reach, so a splat only tests the few cameras of its cell. A cell is skipped when it lies
  // entirely outside one of the pyramid's planes n.(R X + t) >= 0 (camera frame), a conservative
  // test; the pyramid's bounding box bounds the cells visited.
  const grid = new Map(), cones = [];
  for (const im of trainImgs) {
    const { R, t, camera: c } = im, D = 0.65 * near(im);
    if (!(D > 0)) continue;
    const planes = [[0, 0, 1, 0], [0, 0, -1, D], [c.fx, 0, c.cx, 0], [-c.fx, 0, c.width - c.cx, 0],
                    [0, c.fy, c.cy, 0], [0, -c.fy, c.height - c.cy, 0]].map(([x, y, z, d]) => {
      const w = [0, 1, 2].map(k => R[k] * x + R[3 + k] * y + R[6 + k] * z);  // R^T n: the plane in the world
      return [w[0], w[1], w[2], x * t[0] + y * t[1] + z * t[2] + d, Math.abs(w[0]) + Math.abs(w[1]) + Math.abs(w[2])];
    });
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const [u, v, z] of [[0, 0, 0], [0, 0, 1], [c.width, 0, 1], [0, c.height, 1], [c.width, c.height, 1]]) {
      const pc = [(u - c.cx) / c.fx * D * z, (v - c.cy) / c.fy * D * z, D * z];  // apex, base corners
      for (let k = 0; k < 3; k++) {
        const X = R[k] * (pc[0] - t[0]) + R[3 + k] * (pc[1] - t[1]) + R[6 + k] * (pc[2] - t[2]);
        lo[k] = Math.min(lo[k], X); hi[k] = Math.max(hi[k], X);
      }
    }
    cones.push({ im, planes, lo, hi, size: Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) });
  }
  if (!cones.length) return () => false;
  // cells of half the median cone: smaller cells cost more to fill than they save
  const sizes = cones.map(s => s.size).sort((a, b) => a - b), cell = sizes[sizes.length >> 1] / 2, h = 0.5 * cell;
  const key = (i, j, k) => (i * 73856093) ^ (j * 19349663) ^ (k * 83492791);  // collisions only add candidates
  for (const { im, planes, lo, hi } of cones) {
    const a = lo.map(v => Math.floor(v / cell)), b = hi.map(v => Math.floor(v / cell));
    for (let i = a[0]; i <= b[0]; i++) for (let j = a[1]; j <= b[1]; j++) for (let k = a[2]; k <= b[2]; k++) {
      const q = [(i + 0.5) * cell, (j + 0.5) * cell, (k + 0.5) * cell];
      if (planes.some(([x, y, z, d, l1]) => x * q[0] + y * q[1] + z * q[2] + d + h * l1 < 0)) continue;
      const hk = key(i, j, k);
      if (!grid.has(hk)) grid.set(hk, []);
      grid.get(hk).push(im);
    }
  }
  const spared = spare && spare.radius > 0 ? nearPoints(spare.positions, spare.count, spare.radius) : () => false;
  return (x, y, z) => {
    const list = grid.get(key(Math.floor(x / cell), Math.floor(y / cell), Math.floor(z / cell)));
    return !!list && list.some(im => inCone(im, x, y, z)) && !spared(x, y, z);
  };
}

// (x, y, z) -> whether one of the `count` points (xyz at positions[3 i]) is within r: a hash of cells of side r,
// the 27 around (x, y, z) tested
function nearPoints(positions, count, r) {
  const cells = new Map(), key = (i, j, k) => `${i},${j},${k}`, r2 = r * r;
  for (let p = 0; p < count; p++) {
    const k = key(Math.floor(positions[3 * p] / r), Math.floor(positions[3 * p + 1] / r), Math.floor(positions[3 * p + 2] / r));
    if (!cells.has(k)) cells.set(k, []);
    cells.get(k).push(p);
  }
  return (x, y, z) => {
    const i = Math.floor(x / r), j = Math.floor(y / r), k = Math.floor(z / r);
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) for (let dk = -1; dk <= 1; dk++) {
      const list = cells.get(key(i + di, j + dj, k + dk));
      if (list?.some(p => (positions[3 * p] - x) ** 2 + (positions[3 * p + 1] - y) ** 2 + (positions[3 * p + 2] - z) ** 2 <= r2)) return true;
    }
    return false;
  };
}

// Floaters: near-camera carve; single-view billboards (N_eff = W^2 / S < 1.45) in well-seen
// cells; faint splats in isolated leaves.
// The rule that removes the splat: 1 in a camera's near cone, 2 seen from too few directions for
// its error, 3 faint in an empty leaf; 0 for none
function isFloater(params, b, alpha, leaf, W, S, R, nearCamera) {
  const nEff = S > 1e-8 ? (W * W) / S : 1.0;
  if (nearCamera(params[b], params[b + 1], params[b + 2])) return 1;
  if (leaf && leaf.numCameras >= 4 && W > 0.5 && nEff < 1.45 && (R / W) > 0.12) return 2;
  if (leaf && leaf.state === 3 && alpha < 0.25) return 3;
  return 0;
}

// Split splat i along its error covariance: a large one into two children, a small one keeps
// itself and clones at the error centroid. Returns 'split' | 'cloned', or null (nothing pushed).
function splitSplat(out, i, moments, maxS, sceneRadius) {
  const b = i * SPLAT_FLOATS, m = i * MOMENT_FLOATS, params = out.params;
  const x = params[b], y = params[b + 1], z = params[b + 2], R = moments[m + 2];
  // error centroid offset d and covariance E (regularized diagonal)
  const invR = 1 / Math.max(1e-8, R);
  const dx = moments[m + 3] * invR, dy = moments[m + 4] * invR, dz = moments[m + 5] * invR;
  const epsIso = 1e-4 * maxS * maxS;
  const exx = Math.max(epsIso, moments[m + 6] * invR - dx * dx);
  const eyy = Math.max(epsIso, moments[m + 7] * invR - dy * dy);
  const ezz = Math.max(epsIso, moments[m + 8] * invR - dz * dz);
  const exy = moments[m + 9]  * invR - dx * dy;
  const exz = moments[m + 10] * invR - dx * dz;
  const eyz = moments[m + 11] * invR - dy * dz;
  // principal axis v by 4 power iterations
  let vx = exx >= eyy ? 1 : 0, vy = exx < eyy ? 1 : 0, vz = 0.5;
  for (let it = 0; it < 4; it++) {
    const nx = exx * vx + exy * vy + exz * vz;
    const ny = exy * vx + eyy * vy + eyz * vz;
    const nz = exz * vx + eyz * vy + ezz * vz;
    const len = Math.hypot(nx, ny, nz) || 1;
    vx = nx / len; vy = ny / len; vz = nz / len;
  }
  const lambdaMax = Math.max(0,
    vx * (exx * vx + exy * vy + exz * vz) +
    vy * (exy * vx + eyy * vy + eyz * vz) +
    vz * (exz * vx + eyz * vy + ezz * vz));
  const errStd = clamp(Math.sqrt(lambdaMax), maxS * 0.35, maxS * 1.2);
  const shiftNorm = Math.min(maxS, Math.hypot(dx, dy, dz));

  if (maxS > sceneRadius * 0.006) {
    // large: two children +-0.65 errStd along v (kept within 0.65 sigma on each splat
    // axis, so v's component off a flat splat's plane doesn't lift them off it), scales / 1.55
    const p = params.subarray(b, b + SPLAT_FLOATS);
    const [sx, sy, sz] = clampToSplat(p, [vx * errStd * 0.65, vy * errStd * 0.65, vz * errStd * 0.65], 0.65);
    const logShrink = Math.log(1 / 1.55);
    for (const sign of [-1, 1]) {
      const c = out.row(out.add(i, -1));
      c[0] = x + sign * sx; c[1] = y + sign * sy; c[2] = z + sign * sz;
      c[4] += logShrink; c[5] += logShrink; c[6] += logShrink;
    }
    out.splitVectors.push(x - sx, y - sy, z - sz, x + sx, y + sy, z + sz);
    return 'split';
  }
  if (shiftNorm > 1e-5) {
    // small: keep the parent, clone at the error centroid (within 1 sigma on each axis)
    out.add(i, i);
    const child = out.row(out.add(i, -1));
    const [ox, oy, oz] = clampToSplat(child, [dx, dy, dz], 1.0);
    child[0] = x + ox; child[1] = y + oy; child[2] = z + oz;
    return 'cloned';
  }
  return null;
}

// Split splat i as brush (v0.3, train.rs refine_splats): the parent and a fresh child at -o / +o, o drawn from the
// splat's own shape (N(0, 0.5) along each axis x its scale), both scales / sqrt(2), both opacities
// 1 - sqrt(1 - o) (together they cover what the parent did). Returns 'split'.
function splitByShape(out, i) {
  return splitRowByShape(out, out.add(i, i));
}

// row k of `out` split as splitByShape, the second half a new row (src -1); also monte_carlo's relocation with
// relocateMode 'shape' (relocate)
function splitRowByShape(out, k) {
  const kc = out.newRow();
  const p = out.row(k), c = out.row(kc);  // rows after the growth
  const Rm = quatToMat3([p[8], p[9], p[10], p[11]]);  // row-major; axis k = column k
  const e = [0, 1, 2].map(a => 0.5 * gaussRand() * Math.exp(p[4 + a]));
  const o = [0, 1, 2].map(r => Rm[3 * r] * e[0] + Rm[3 * r + 1] * e[1] + Rm[3 * r + 2] * e[2]);
  const a = 1 - Math.sqrt(1 - sigmoid(p[3]));
  for (let c = 4; c < 7; c++) p[c] -= 0.5 * Math.LN2;
  p[3] = logit(Math.max(a, 1e-4));
  c.set(p);
  for (let r = 0; r < 3; r++) { c[r] = p[r] + o[r]; p[r] -= o[r]; }
  out.splitVectors?.push(p[0], p[1], p[2], c[0], c[1], c[2]);
  return 'split';
}

function gaussRand() {
  return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
}

// D. Merge consecutive (same leaf neighborhood) overlapping splats of similar color:
//    opacity-weighted position / color, combined opacity 1 - (1 - a)(1 - b). Returns the count.
function mergeOverlaps(out) {
  const { buf, src, n } = out, F = SPLAT_FLOATS;
  out.alive = new Uint8Array(n).fill(1);
  if (n <= 16) return 0;
  let nMerged = 0;
  for (let i = 0; i + 1 < n; i++) {
    if (src[i] < 0 || src[i + 1] < 0) continue;  // don't undo this pass's new splats
    const a = i * F, b = a + F;
    const dx = buf[a] - buf[b], dy = buf[a + 1] - buf[b + 1], dz = buf[a + 2] - buf[b + 2];
    const sa = Math.exp(Math.max(buf[a + 4], buf[a + 5], buf[a + 6]));
    const sb = Math.exp(Math.max(buf[b + 4], buf[b + 5], buf[b + 6]));
    const colDiff = Math.abs(buf[a + 12] - buf[b + 12]) + Math.abs(buf[a + 13] - buf[b + 13]) + Math.abs(buf[a + 14] - buf[b + 14]);
    if (dx * dx + dy * dy + dz * dz < 0.08 * (sa * sa + sb * sb) && colDiff < 0.25) {
      const wa = sigmoid(buf[a + 3]), wb = sigmoid(buf[b + 3]);
      const wSum = wa + wb + 1e-6;
      for (const k of [0, 1, 2, 12, 13, 14]) buf[a + k] = (wa * buf[a + k] + wb * buf[b + k]) / wSum;
      buf[a + 3] = logit(1 - (1 - wa) * (1 - wb));
      out.alive[i + 1] = 0;
      nMerged++;
      i++;
    }
  }
  return nMerged;
}

// Over the cap (the count above maxSplats: the lease's consolidation, or max_splats lowered), the rows of the lowest score go (alive = 0).
// score[src * stride]: the input splat's; new rows of this pass (src -1: children, copies) stay. Returns how many went.
function trimToCap(out, maxSplats, score, stride) {
  out.alive ??= new Uint8Array(out.n).fill(1);
  const rows = [];
  for (let k = 0; k < out.n; k++) if (out.alive[k] && out.src[k] >= 0) rows.push(k);
  let alive = 0;
  for (let k = 0; k < out.n; k++) alive += out.alive[k];
  const excess = Math.min(rows.length, alive - maxSplats);
  if (!(excess > 0)) return 0;
  const key = Float32Array.from(rows, k => score[out.src[k] * stride]);
  const order = Uint32Array.from(rows.keys()).sort((a, b) => key[a] - key[b]);
  for (let j = 0; j < excess; j++) out.alive[rows[order[j]]] = 0;
  return excess;
}

// E. Compact the kept splats and append stipple seeds within the maxSplats budget.
// Returns { count, params, srcIdx, nSeeds }.
function compactWithSeeds({ buf, src, n, alive }, seedSplats, maxSplats) {
  const F = SPLAT_FLOATS;
  let nKept = 0;
  for (let k = 0; k < n; k++) nKept += alive ? alive[k] : 1;
  const nSeeds = seedSplats ? Math.max(0, Math.min(seedSplats.length / F, maxSplats - nKept)) : 0;
  const count = nKept + nSeeds;
  const params = new Float32Array(count * F), srcIdx = new Int32Array(count).fill(-1);
  for (let k = 0, j = 0; k < n; k++) {
    if (alive && !alive[k]) continue;
    params.set(buf.subarray(k * F, (k + 1) * F), j * F);
    srcIdx[j++] = src[k];
  }
  if (nSeeds > 0) params.set(seedSplats.subarray(0, nSeeds * F), nKept * F);
  return { count, params, srcIdx, nSeeds };
}
