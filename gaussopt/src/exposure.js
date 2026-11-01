// exposure.js — Each training view's exposure, learned with the splats (DESIGN.md E19 1): the render through a per-view
// color affine before the loss, a gain (as its log) and an offset per channel about mid-grey, c' = g (c - 1/2) + 1/2 + o:
// the shader's c' = g c + b with b = o + (1 - g) / 2. tile_backward sums the view's gradient in its terms (S_g =
// dL/dc' (c' - b) = dL/dlog g of c' = g c + b, S_b = dL/dc', over the pixels) at each of its steps; here dL/dlog g = S_g -
// g S_b / 2, dL/do = S_b, and Adam on these 6 numbers, one update a visit. The pivot at mid-grey: about 0, g c + b's
// gain and offset nearly trade (the colors near 1/2: g up and b down give nearly the same image) and Adam, a step per
// coordinate, crawls along that (exposure.e2e.js, tests/exposure_test.js). Gauge: the splats' colors
// and every gain can trade a common factor, so the mean log-gain and the mean offset over the views are kept at 0 (the
// splats keep the scene's colors). A view never trained (held out) takes its neighbours' in name order (a video's
// frames: in time), interpolated by the frame number in the names (else halfway), the first or last one's past the
// ends: exposures drift slowly along a capture, and a held-out view scored as is (identity) does worse than with
// none learned. With none learned at all, as is.
"use strict";

import { adamState, adamStep, byName, neighbours } from './per_view.js';

export const EXPO_FLOATS = 8;  // tile_backward's sums a step: dL/dlog g (rgb), pad, dL/db (rgb), pad
const EPS = 1e-8, PIVOT = 0.5;

export class Exposure {
  constructor() { this.views = new Map(); }  // image name -> { p: [log g rgb, o rgb], m, v, t }

  // TrainUniforms.expo_g / expo_b of view `name`: [g rgb, 1, b rgb, 0] (c' = g c + b); a view with none its neighbours'
  // (the header), identity when no view has one
  uniforms(name) {
    const p = this.views.get(name)?.p ?? this.between(name);
    if (!p) return [1, 1, 1, 1, 0, 0, 0, 0];
    const g = [0, 1, 2].map(c => Math.exp(p[c]));
    return [...g, 1, ...g.map((x, c) => p[3 + c] + PIVOT * (1 - x)), 0];
  }

  // The parameters of a view without its own: its neighbours' in name order, interpolated by frame number
  between(name) {
    if (!this.views.size) return null;
    if (this._sorted?.length !== this.views.size) this._sorted = [...this.views.keys()].sort(byName);
    const { a, b, t } = neighbours(this._sorted, name), pa = this.views.get(a).p, pb = this.views.get(b).p;
    return pa.map((x, j) => (1 - t) * x + t * pb[j]);
  }

  // One Adam step of view `name` on its gradient (EXPO_FLOATS, as tile_backward sums it) at rate lr
  update(name, grad, lr) {
    let s = this.views.get(name);
    if (!s) this.views.set(name, s = adamState(6));
    s.t++;
    for (let j = 0; j < 6; j++) adamStep(s, j, j < 3 ? grad[j] - PIVOT * Math.exp(s.p[j]) * grad[4 + j] : grad[j + 1], lr, EPS);
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
      e.views.set(k, { ...adamState(6), p });
    }
    return e;
  }
}
