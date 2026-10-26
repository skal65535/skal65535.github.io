// regions.js — The scene cut in fixed cells (a res^3 grid over the splat centers, fixed at its first use), and the
// error of each cell over a rebalance window: which parts of the scene still learn (DESIGN.md §9 E7). The cells
// stay put across rebalances, unlike the octree's leaves. Pure CPU (no DOM, no GPU).
"use strict";

import { SPLAT_FLOATS } from './colmap.js';
import { MOMENT_FLOATS } from './octree.js';

// The grid: the 2nd..98th percentile box of the centers on each axis, res cells per axis (splats outside go to
// the border cells)
export function makeRegionGrid(params, count, res = 8) {
  const lo = [0, 0, 0], hi = [0, 0, 0], v = new Float32Array(count);
  for (let a = 0; a < 3; a++) {
    for (let i = 0; i < count; i++) v[i] = params[i * SPLAT_FLOATS + a];
    v.sort();
    lo[a] = v[Math.floor(0.02 * (count - 1))];
    hi[a] = v[Math.ceil(0.98 * (count - 1))];
    if (!(hi[a] > lo[a])) hi[a] = lo[a] + 1;
  }
  return { res, lo, hi };
}

export function regionOf(grid, x, y, z) {
  const { res, lo, hi } = grid, p = [x, y, z];
  let c = 0;
  for (let a = 2; a >= 0; a--) {
    const k = Math.min(res - 1, Math.max(0, Math.floor((p[a] - lo[a]) / (hi[a] - lo[a]) * res)));
    c = c * res + k;
  }
  return c;  // x + res (y + res z)
}

// Per cell over the window: [W (visibility mass), R (error mass), n (splats)], 3 floats a cell
export function regionStats(grid, params, count, moments) {
  const cells = new Float32Array(3 * grid.res ** 3);
  for (let i = 0; i < count; i++) {
    const b = i * SPLAT_FLOATS, c = 3 * regionOf(grid, params[b], params[b + 1], params[b + 2]);
    cells[c] += moments[i * MOMENT_FLOATS];
    cells[c + 1] += moments[i * MOMENT_FLOATS + 2];
    cells[c + 2] += 1;
  }
  return cells;
}

// E7's repair: the cells stuck over the last 6 windows, i.e. whose error is above the scene's (e_c > e) and fell
// slower than it, log(e_c(last 3) / e_c(first 3)) > log(e(last 3) / e(first 3)) (3 windows at each end: a video's
// window does not visit every view), each end with at least `minShare` of the visibility mass
export class RegionTracker {
  constructor(res = 8, { minShare = 0.002 } = {}) { this.res = res; this.minShare = minShare; this.grid = null; this.windows = []; }

  // a window's moments in; returns the stuck cells (Uint8Array, 1: stuck), or null before 6 windows
  update(params, count, moments) {
    this.grid ??= makeRegionGrid(params, count, this.res);
    this.windows.push(regionStats(this.grid, params, count, moments));
    if (this.windows.length > 6) this.windows.shift();
    if (this.windows.length < 6) return (this.stuck = null);
    const nc = this.res ** 3, sum = (ws) => {
      const c = new Float32Array(3 * nc);
      for (const w of ws) for (let k = 0; k < c.length; k++) c[k] += w[k];
      return c;
    };
    const a = sum(this.windows.slice(0, 3)), b = sum(this.windows.slice(3)), tot = (c, o) => { let s = 0; for (let k = o; k < c.length; k += 3) s += c[k]; return s; };
    const [aW, aR, bW, bR] = [tot(a, 0), tot(a, 1), tot(b, 0), tot(b, 1)], scene = Math.log((bR / bW) / (aR / aW));
    const stuck = new Uint8Array(nc);
    for (let c = 0; c < nc; c++) {
      const w0 = a[3 * c], r0 = a[3 * c + 1], w1 = b[3 * c], r1 = b[3 * c + 1];
      if (!(w0 >= this.minShare * aW && w1 >= this.minShare * bW && r0 > 0 && r1 > 0)) continue;
      if (r1 / w1 > bR / bW && Math.log((r1 / w1) / (r0 / w0)) > scene) stuck[c] = 1;
    }
    return (this.stuck = stuck);
  }

  // per splat: 1 in a stuck cell (Uint8Array(count)), or null
  splatMask(params, count) {
    if (!this.stuck) return null;
    const m = new Uint8Array(count);
    for (let i = 0; i < count; i++) {
      const b = i * SPLAT_FLOATS;
      m[i] = this.stuck[regionOf(this.grid, params[b], params[b + 1], params[b + 2])];
    }
    return m;
  }
}
