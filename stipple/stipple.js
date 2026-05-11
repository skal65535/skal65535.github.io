// stipple.js - Weighted LBG Stippling
// API: iter = new StipplingIterator(grays, params)
//      iter.step() -> bool (true = done)
//      iter.points, iter.done, iter.progress
"use strict";

const log = (...a) => globalThis.STIPPLE_DEBUG && console.log(...a);

export class Point {
  constructor(x = 0, y = 0, c = 0) { this.x = x; this.y = y; this.c = c; }
  dist2(p) { const dx = p.x - this.x, dy = p.y - this.y; return dx*dx + dy*dy; }
}

export class StipplingParams {
  num_points    = 80000;
  num_iters     = 10;
  num_Lloyd_iters = 4;
  rho           = 0;
  seed          = 91651088029;
}

class Cell {
  acc = 0; r_acc = 0;
  rx_acc = 0; ry_acc = 0;
  rxx_acc = 0; rxy_acc = 0; ryy_acc = 0;

  reset() {
    this.acc = this.r_acc = 0;
    this.rx_acc = this.ry_acc = this.rxx_acc = this.rxy_acc = this.ryy_acc = 0;
  }
  add(x, y, r) {
    ++this.acc; this.r_acc += r;
    this.rx_acc += r*x; this.ry_acc += r*y;
    this.rxx_acc += r*x*x; this.rxy_acc += r*x*y; this.ryy_acc += r*y*y;
  }
  addRho(x, y, r) { this.r_acc += r; this.rx_acc += r*x; this.ry_acc += r*y; }
  // After average() rx_acc/ry_acc are <r·x>/<r> (r-weighted mean of x).
  // rxx_acc/rxy_acc/ryy_acc are <r·x²>/<r> etc. mainDirection uses these
  // consistently r-weighted; computing exx as rxx − rx² (not against an
  // unweighted mean) keeps the variance estimator coherent.
  average() {
    if (!this.acc || this.r_acc <= 0) return this.acc > 0;
    const rn = 1/this.r_acc;
    this.rx_acc *= rn; this.ry_acc *= rn;
    this.rxx_acc *= rn; this.rxy_acc *= rn; this.ryy_acc *= rn;
    return true;
  }
  averageRho() {
    if (!this.r_acc) return false;
    const rn = 1/this.r_acc;
    this.rx_acc *= rn; this.ry_acc *= rn;
    return true;
  }
  mainDirection() {
    const exx = this.rxx_acc - this.rx_acc * this.rx_acc;
    const eyy = this.ryy_acc - this.ry_acc * this.ry_acc;
    const num = this.rxy_acc - this.rx_acc * this.ry_acc;
    const den = exx - eyy;
    const t = (num*num + den*den > 0) ? 0.5 * Math.atan2(2*num, den) : 0;
    const radius = 0.5 * Math.sqrt(this.acc / Math.PI);
    return { dx: radius * Math.cos(t), dy: radius * Math.sin(t) };
  }
}

// Grid-accelerated Voronoi. For N points in [0,1]^2, uses a gs x gs spatial
// grid so each pixel needs O(k) checks (k~1-9 cells) instead of O(sqrt(N)).
// Packs points into Float32Arrays and uses a CSR-flattened grid to keep the
// inner loop free of object property loads and iterator allocations.
export function computeVoronoi(W, H, points) {
  const N = points.length;
  const out = new Uint32Array(W * H);
  if (!N) return out;

  const gs = Math.max(1, Math.ceil(Math.sqrt(N / 4)));
  const inv_gs2 = 1 / (gs * gs);
  const G = gs * gs;

  // Pack point coords (object loads are expensive in the hot loop).
  const xs = new Float32Array(N), ys = new Float32Array(N);
  for (let i = 0; i < N; i++) { xs[i] = points[i].x; ys[i] = points[i].y; }

  // Build CSR grid: counts[c..c+1] bracket bucket c in gridIdx.
  const counts = new Int32Array(G + 1);
  const cellOf = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    let gx = (xs[i] * gs) | 0; if (gx < 0) gx = 0; else if (gx >= gs) gx = gs - 1;
    let gy = (ys[i] * gs) | 0; if (gy < 0) gy = 0; else if (gy >= gs) gy = gs - 1;
    const c = gx + gy * gs;
    cellOf[i] = c;
    counts[c + 1]++;
  }
  for (let i = 1; i <= G; i++) counts[i] += counts[i - 1];
  const gridIdx = new Int32Array(N);
  const cursor = new Int32Array(G);
  for (let i = 0; i < N; i++) {
    const c = cellOf[i];
    gridIdx[counts[c] + cursor[c]++] = i;
  }

  const inv_W = 1 / W, inv_H = 1 / H;

  for (let y = 0; y < H; y++) {
    const Y = y * inv_H;
    let gy0 = (Y * gs) | 0; if (gy0 >= gs) gy0 = gs - 1;
    const rowOff = y * W;
    for (let x = 0; x < W; x++) {
      const X = x * inv_W;
      let gx0 = (X * gs) | 0; if (gx0 >= gs) gx0 = gs - 1;
      let best = 0, best_d2 = Infinity;

      // r=0: home cell.
      {
        const home = gx0 + gy0 * gs;
        const lo = counts[home], hi = counts[home + 1];
        for (let k = lo; k < hi; k++) {
          const i = gridIdx[k];
          const ddx = X - xs[i], ddy = Y - ys[i];
          const d2 = ddx*ddx + ddy*ddy;
          if (d2 < best_d2) { best_d2 = d2; best = i; }
        }
      }

      // Expanding Chebyshev rings; iterate perimeter only (no Math.abs).
      for (let r = 1; r <= gs; r++) {
        if (r >= 2 && (r - 1) * (r - 1) * inv_gs2 >= best_d2) break;
        const ny_t = gy0 - r, ny_b = gy0 + r;
        const nx_l = gx0 - r, nx_r = gx0 + r;
        const cx_lo = nx_l < 0 ? 0 : nx_l;
        const cx_hi = nx_r >= gs ? gs - 1 : nx_r;

        // Top + bottom rows.
        if (ny_t >= 0) {
          const base = ny_t * gs;
          const lo = counts[base + cx_lo], hi = counts[base + cx_hi + 1];
          for (let k = lo; k < hi; k++) {
            const i = gridIdx[k];
            const ddx = X - xs[i], ddy = Y - ys[i];
            const d2 = ddx*ddx + ddy*ddy;
            if (d2 < best_d2) { best_d2 = d2; best = i; }
          }
        }
        if (ny_b < gs) {
          const base = ny_b * gs;
          const lo = counts[base + cx_lo], hi = counts[base + cx_hi + 1];
          for (let k = lo; k < hi; k++) {
            const i = gridIdx[k];
            const ddx = X - xs[i], ddy = Y - ys[i];
            const d2 = ddx*ddx + ddy*ddy;
            if (d2 < best_d2) { best_d2 = d2; best = i; }
          }
        }
        // Left + right columns (excluding corners already covered above).
        const cy_lo = (ny_t + 1) < 0 ? 0 : ny_t + 1;
        const cy_hi = (ny_b - 1) >= gs ? gs - 1 : ny_b - 1;
        if (nx_l >= 0) {
          for (let ny = cy_lo; ny <= cy_hi; ny++) {
            const cell = nx_l + ny * gs;
            const lo = counts[cell], hi = counts[cell + 1];
            for (let k = lo; k < hi; k++) {
              const i = gridIdx[k];
              const ddx = X - xs[i], ddy = Y - ys[i];
              const d2 = ddx*ddx + ddy*ddy;
              if (d2 < best_d2) { best_d2 = d2; best = i; }
            }
          }
        }
        if (nx_r < gs) {
          for (let ny = cy_lo; ny <= cy_hi; ny++) {
            const cell = nx_r + ny * gs;
            const lo = counts[cell], hi = counts[cell + 1];
            for (let k = lo; k < hi; k++) {
              const i = gridIdx[k];
              const ddx = X - xs[i], ddy = Y - ys[i];
              const d2 = ddx*ddx + ddy*ddy;
              if (d2 < best_d2) { best_d2 = d2; best = i; }
            }
          }
        }
      }
      out[x + rowOff] = best;
    }
  }
  return out;
}

// Walk every pixel and accumulate it into the cell that owns it.
// full=false → addRho only (used during Lloyd step).
// full=true  → add (full second-moments, used at split/merge time).
function accumulate(cells, out, pixels, W, H, full) {
  for (let y = 0, idx = 0; y < H; y++) {
    const Y = y / H;
    for (let x = 0; x < W; x++, idx++) {
      const c = cells[out[idx]], xf = x/W, r = pixels[idx];
      if (full) c.add(xf, Y, r);
      else      c.addRho(xf, Y, r);
    }
  }
}

// grays: { W, H, pixels: Uint8ClampedArray, average: number (sum of all pixel values) }
export class StipplingIterator {
  _grays; _params; _points; _ops; _max_ops; _rand;

  constructor(grays, params, existing_points = null) {
    this._grays = grays;
    this._params = params;
    this._max_ops = params.num_iters * params.num_Lloyd_iters;
    this._ops = 0;

    let seed = params.seed ?? 91651088029;
    this._rand = () => {
      let t = seed += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };

    if (existing_points) {
      this._points = existing_points.map(p => new Point(p.x, p.y, p.c));
    } else {
      const n = params.num_points >> 1;
      this._points = Array.from({length: n}, () => new Point(this._rand(), this._rand()));
    }
  }

  get points()   { return this._points; }
  get done()     { return this._ops >= this._max_ops; }
  get progress() { return this._ops / this._max_ops; }

  // One Lloyd sub-iteration. Returns true when fully converged.
  step() {
    if (this.done) return true;
    const {_grays: g, _params: p} = this;
    const {W, H, pixels} = g;
    const N = this._points.length;
    const out = computeVoronoi(W, H, this._points);
    const cells = Array.from({length: N}, () => new Cell());

    accumulate(cells, out, pixels, W, H, /*full=*/false);
    let d2 = 0, cnt = 0;
    for (let n = 0; n < N; n++) {
      const c = cells[n];
      if (!c.averageRho()) continue;
      const pt = new Point(c.rx_acc, c.ry_acc);
      d2 += this._points[n].dist2(pt);
      this._points[n] = pt;
      cnt++;
    }
    const converged = cnt > 0 && (d2 * W * H / cnt) < 1e-3;
    if (converged) this._ops = this._max_ops;
    else           this._ops++;

    if (this._ops % p.num_Lloyd_iters === 0 || this.done) {
      this._splitMerge(out, cells, g);
    }
    if (this._points.length > p.num_points) this._ops = this._max_ops;
    return this.done;
  }

  _splitMerge(out, cells, g) {
    const {W, H} = g;
    const p = this._params;
    const avg_rho = Math.floor(p.rho * 255 + g.average / p.num_points);
    const hysteresis = 0.01 + 0.6 * this._ops / this._max_ops;
    const Tu = Math.ceil((1 + hysteresis) * avg_rho);
    const Tl = Math.floor((1 - hysteresis) * avg_rho);
    const eps_W = 1/W, eps_H = 1/H;

    for (const c of cells) c.reset();
    accumulate(cells, out, g.pixels, W, H, /*full=*/true);
    const prev = cells.length;
    const Nmax = p.num_points;
    const new_pts = [];
    const inUnit = v => v >= 0 && v <= 1;
    for (const c of cells) {
      if (!c.average()) continue;
      const rho = Math.round(c.r_acc);
      if (rho < Tl) continue;
      const color = c.r_acc / c.acc;
      const xf = c.rx_acc, yf = c.ry_acc;
      if (c.acc > 1 && rho > Tu && new_pts.length + 2 <= Nmax) {
        const d = c.mainDirection();
        const x1 = xf - d.dx, y1 = yf - d.dy;
        const x2 = xf + d.dx, y2 = yf + d.dy;

        if ((Math.abs(d.dx) > eps_W || Math.abs(d.dy) > eps_H) &&
            inUnit(x1) && inUnit(y1) && inUnit(x2) && inUnit(y2)) {
          new_pts.push(new Point(x1, y1, color));
          new_pts.push(new Point(x2, y2, color));
          continue;
        }
      }
      if (new_pts.length < Nmax) new_pts.push(new Point(xf, yf, color));
    }
    this._points = new_pts;
    log(`iter #${this._ops}/${this._max_ops}: T=[${Tl},${Tu}] pts:${prev}=>${new_pts.length}`);
  }
}
