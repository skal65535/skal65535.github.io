// exposure.js — Each training view's exposure, learned with the splats (DESIGN.md E19 1): the render through a per-view
// color affine before the loss, a gain (as its log) and an offset per channel about mid-grey, c' = g (c - 1/2) + 1/2 + o:
// the shader's c' = g c + b with b = o + (1 - g) / 2. tile_backward sums the view's gradient in its terms (S_g =
// dL/dc' (c' - b) = dL/dlog g of c' = g c + b, S_b = dL/dc', over the pixels) at each of its steps; here dL/dlog g = S_g -
// g S_b / 2, dL/do = S_b, and Adam on these 6 numbers, one update a visit. About 0 the gain and the offset nearly trade
// (the colors near 1/2: g up and b down give nearly the same image) and Adam, a step per coordinate, crawled along that:
// the room's photos through known gains (+-10%), 150 visits a view, the gains 4.3-4.7% off and the offsets 0.032-0.034,
// 1.6-1.8% and 0.017-0.018 so (the splats' residual weighs otherwise in a brighter photo: exposure.e2e.js); colors in
// [0.4, 0.6], 100 visits, 0.82% against 0.05% (tests/exposure_test.js). Gauge: the splats' colors
// and every gain can trade a common factor, so the mean log-gain and the mean offset over the views are kept at 0 (the
// splats keep the scene's colors). A view never trained (held out) takes its neighbours' in name order (a video's
// frames: in time), interpolated by the frame number in the names (else halfway), the first or last one's past the
// ends: long-garden's exposures drift slowly, a frame's within 0.004 of its two neighbours' mean where they spread by
// 0.056 (mid-grey's image, mean |deviation|); scored as is (identity), its held-out PSNR fell 1.3 dB under none
// learned. With none learned at all, as is.
"use strict";

export const EXPO_FLOATS = 8;  // tile_backward's sums a step: dL/dlog g (rgb), pad, dL/db (rgb), pad
const B1 = 0.9, B2 = 0.999, EPS = 1e-8, PIVOT = 0.5;
const byName = (a, b) => a.localeCompare(b, undefined, { numeric: true });
const frameNumber = (name) => { const m = name.match(/(\d+)\D*$/); return m ? Number(m[1]) : null; };

export class Exposure {
  constructor() { this.views = new Map(); }  // image name -> { p: [log g rgb, o rgb], m, v, t }

  // TrainUniforms.expo_g / expo_b of view `name`: [g rgb, 1, b rgb, 0] (c' = g c + b); a view with none its neighbours'
  // (the header), identity when no view has one
  uniforms(name) {
    const p = this.views.get(name)?.p ?? this._between(name);
    if (!p) return [1, 1, 1, 1, 0, 0, 0, 0];
    const g = [0, 1, 2].map(c => Math.exp(p[c]));
    return [...g, 1, ...g.map((x, c) => p[3 + c] + PIVOT * (1 - x)), 0];
  }

  // The parameters of a view without its own: its neighbours' in name order, interpolated by frame number
  _between(name) {
    if (!this.views.size) return null;
    if (this._sorted?.length !== this.views.size) this._sorted = [...this.views.keys()].sort(byName);
    const names = this._sorted;
    let lo = 0, hi = names.length;  // the first name after `name`
    while (lo < hi) { const mid = (lo + hi) >> 1; if (byName(names[mid], name) < 0) lo = mid + 1; else hi = mid; }
    if (lo === 0) return this.views.get(names[0]).p;
    if (lo === names.length) return this.views.get(names.at(-1)).p;
    const a = names[lo - 1], b = names[lo], [fa, fb, f] = [a, b, name].map(frameNumber);
    const t = fa !== null && fb !== null && f !== null && fb > fa ? Math.min(1, Math.max(0, (f - fa) / (fb - fa))) : 0.5;
    const pa = this.views.get(a).p, pb = this.views.get(b).p;
    return pa.map((x, j) => (1 - t) * x + t * pb[j]);
  }

  // One Adam step of view `name` on its gradient (EXPO_FLOATS, as tile_backward sums it) at rate lr
  update(name, grad, lr) {
    let s = this.views.get(name);
    if (!s) this.views.set(name, s = { p: new Float64Array(6), m: new Float64Array(6), v: new Float64Array(6), t: 0 });
    s.t++;
    const bc1 = 1 - B1 ** s.t, bc2 = 1 - B2 ** s.t;
    for (let j = 0; j < 6; j++) {
      const g = j < 3 ? grad[j] - PIVOT * Math.exp(s.p[j]) * grad[4 + j] : grad[j + 1];
      s.m[j] = B1 * s.m[j] + (1 - B1) * g;
      s.v[j] = B2 * s.v[j] + (1 - B2) * g * g;
      s.p[j] -= lr * (s.m[j] / bc1) / (Math.sqrt(s.v[j] / bc2) + EPS);
    }
  }

  // The gauge: the mean log-gain and offset over the views removed
  center() {
    const n = this.views.size;
    if (!n) return;
    const mean = new Float64Array(6);
    for (const s of this.views.values()) for (let j = 0; j < 6; j++) mean[j] += s.p[j] / n;
    for (const s of this.views.values()) for (let j = 0; j < 6; j++) s.p[j] -= mean[j];
  }

  // Each view's gain and offset as the shader's c' = g c + b ({ name: [g rgb, b rgb] }), and back (a snapshot's; Adam's
  // moments start over)
  toJSON() {
    return Object.fromEntries([...this.views.keys()].map(k => { const u = this.uniforms(k); return [k, [...u.slice(0, 3), ...u.slice(4, 7)]]; }));
  }
  static fromJSON(obj) {
    const e = new Exposure();
    for (const [k, a] of Object.entries(obj ?? {})) {
      const p = Float64Array.from([0, 1, 2, 0, 1, 2].map((c, j) => j < 3 ? Math.log(a[c]) : a[3 + c] - PIVOT * (1 - a[c])));
      e.views.set(k, { p, m: new Float64Array(6), v: new Float64Array(6), t: 0 });
    }
    return e;
  }
}
