// stipple.js - Weighted LBG Stippling
// API: iter = new StipplingIterator(grays, params)
//      iter.step() -> bool (true = done)
//      iter.points, iter.done, iter.progress
"use strict";

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
  x_acc = 0; y_acc = 0;
  rx_acc = 0; ry_acc = 0;
  rxx_acc = 0; rxy_acc = 0; ryy_acc = 0;

  reset() {
    this.acc = this.r_acc = this.x_acc = this.y_acc = 0;
    this.rx_acc = this.ry_acc = this.rxx_acc = this.rxy_acc = this.ryy_acc = 0;
  }
  add(x, y, r) {
    ++this.acc;
    this.x_acc += x; this.y_acc += y; this.r_acc += r;
    this.rx_acc += r*x; this.ry_acc += r*y;
    this.rxx_acc += r*x*x; this.rxy_acc += r*x*y; this.ryy_acc += r*y*y;
  }
  addRho(x, y, r) { this.r_acc += r; this.rx_acc += r*x; this.ry_acc += r*y; }
  average() {
    if (!this.acc) return false;
    const n = 1/this.acc;
    this.x_acc *= n; this.y_acc *= n;
    if (this.r_acc > 0) {
      const rn = 1/this.r_acc;
      this.rx_acc *= rn; this.ry_acc *= rn;
      this.rxx_acc *= rn; this.rxy_acc *= rn; this.ryy_acc *= rn;
    }
    return true;
  }
  averageRho() {
    if (!this.r_acc) return false;
    const rn = 1/this.r_acc;
    this.rx_acc *= rn; this.ry_acc *= rn;
    return true;
  }
  mainDirection() {
    const exx = this.rxx_acc - this.x_acc * this.x_acc;
    const eyy = this.ryy_acc - this.y_acc * this.y_acc;
    const num = this.rxy_acc - this.x_acc * this.y_acc;
    const den = exx - eyy;
    const t = (num*num + den*den > 0) ? 0.5 * Math.atan2(2*num, den) : 0;
    const radius = 0.5 * Math.sqrt(this.acc / Math.PI);
    return { dx: radius * Math.cos(t), dy: radius * Math.sin(t) };
  }
}

// Grid-accelerated Voronoi. For N points in [0,1]^2, uses a gs x gs spatial
// grid so each pixel needs O(k) checks (k~1-9 cells) instead of O(sqrt(N)).
// grays: {W, H} dimensions of the output map.
export function computeVoronoi(W, H, points) {
  const N = points.length;
  const out = new Uint32Array(W * H);
  if (!N) return out;

  const gs = Math.max(1, Math.ceil(Math.sqrt(N / 4)));
  const inv_gs = 1 / gs;
  const grid = new Array(gs * gs).fill(null).map(() => []);
  for (let i = 0; i < N; i++) {
    const gx = Math.max(0, Math.min((points[i].x * gs) | 0, gs - 1));
    const gy = Math.max(0, Math.min((points[i].y * gs) | 0, gs - 1));
    grid[gx + gy * gs].push(i);
  }

  for (let y = 0; y < H; y++) {
    const Y = y / H;
    const gy0 = Math.min((Y * gs) | 0, gs - 1);
    for (let x = 0; x < W; x++) {
      const X = x / W;
      const gx0 = Math.min((X * gs) | 0, gs - 1);
      let best = 0, best_d2 = Infinity;

      for (let r = 0; r <= gs; r++) {
        // cells at Chebyshev ring r>=2 are at Euclidean dist >= (r-1)/gs
        if (r >= 2 && (r-1)*(r-1)*inv_gs*inv_gs >= best_d2) break;
        for (let dy = -r; dy <= r; dy++) {
          for (let dx = -r; dx <= r; dx++) {
            if (r > 0 && Math.abs(dx) !== r && Math.abs(dy) !== r) continue;
            const nx = gx0 + dx, ny = gy0 + dy;
            if (nx < 0 || nx >= gs || ny < 0 || ny >= gs) continue;
            for (const i of grid[nx + ny * gs]) {
              const ddx = X - points[i].x, ddy = Y - points[i].y;
              const d2 = ddx*ddx + ddy*ddy;
              if (d2 < best_d2) { best_d2 = d2; best = i; }
            }
          }
        }
      }
      out[x + y * W] = best;
    }
  }
  return out;
}

// grays: { W, H, pixels: Uint8ClampedArray, average: number (sum of all pixel values) }
export class StipplingIterator {
  _grays; _params; _points; _ops; _max_ops; _rand;

  constructor(grays, params) {
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

    const n = params.num_points >> 1;
    this._points = Array.from({length: n}, () => new Point(this._rand(), this._rand()));
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

    for (let y = 0, idx = 0; y < H; y++) {
      const Y = y / H;
      for (let x = 0; x < W; x++, idx++) {
        cells[out[idx]].addRho(x/W, Y, pixels[idx]);
      }
    }
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
    for (let y = 0, idx = 0; y < H; y++) {
      const Y = y / H;
      for (let x = 0; x < W; x++, idx++) {
        cells[out[idx]].add(x/W, Y, g.pixels[idx]);
      }
    }
    const prev = cells.length;
    const new_pts = [];
    for (const c of cells) {
      if (!c.average()) continue;
      const rho = Math.round(c.r_acc);
      if (rho < Tl) continue;
      const color = c.r_acc / c.acc;
      const xf = c.rx_acc, yf = c.ry_acc;
      if (c.acc > 1 && rho > Tu) {
        const d = c.mainDirection();
        const x1 = xf - d.dx, y1 = yf - d.dy;
        const x2 = xf + d.dx, y2 = yf + d.dy;
        const inBounds = (val) => val >= 0 && val <= 1;

        if ((Math.abs(d.dx) > eps_W || Math.abs(d.dy) > eps_H) &&
            inBounds(x1) && inBounds(y1) && inBounds(x2) && inBounds(y2)) {
          new_pts.push(new Point(x1, y1, color));
          new_pts.push(new Point(x2, y2, color));
          continue;
        }
      }
      new_pts.push(new Point(xf, yf, color));
    }
    this._points = new_pts;
    console.log(`iter #${this._ops}/${this._max_ops}: T=[${Tl},${Tu}] pts:${prev}=>${new_pts.length}`);
  }
}
