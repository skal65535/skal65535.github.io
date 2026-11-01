// poses.js — The cameras refined by the splats (DESIGN.md E19 2, 3): each training view's pose and each lens's focal
// and principal point, from adam_step's sums (shaders.wgsl: dL/drho, dL/dphi of a camera-frame increment
// x_c -> x_c + rho + phi x x_c, dL/dlog f, dL/dcx, dL/dcy at training resolution), one Adam step a view a visit, a lens
// one a read (the sum of its views' visits). A view's pose is its SfM pose through its increment: R = Exp(phi) R0,
// t = Exp(phi) t0 + rho (rho in units of the scene radius for Adam's rate). Gauge: the splats and every camera can move
// together (a similarity), so the increments' mean world rotation, translation and scale about the cameras' centre are
// removed after each read: the scene stays in the SfM's frame. The pull to the SfM's values (pose_prior, sigma in
// radians): after each Adam step, p -= lr p / sigma (decoupled, as AdamW's decay): Adam's step is at most lr, so an
// increment stays within sigma (a steady gradient holds it at sigma times its mean over its RMS; noise, at 0); the
// move in units of the radius, the focal relative, the principal point x the width: the same sigma for all. A view
// never trained (held out) takes its neighbours' in name order (exposure.js's), as a world rotation and a move of its
// centre. The SfM's values are kept on the image
// (img.sfmPose) and the camera (camera.sfmLens); the refined ones are written into them (colmap.js setPose).
"use strict";

import { setPose } from './colmap.js';
import { mul3 } from './sfm/linalg.js';
import { adamState, adamStep, byName, neighbours } from './per_view.js';

const EPS = 1e-12;

// Rodrigues: the rotation matrix (row-major 3 x 3) of the rotation vector w
export function expRot(w) {
  const th = Math.hypot(w[0], w[1], w[2]);
  const a = th < 1e-12 ? 1 : Math.sin(th) / th, b = th < 1e-12 ? 0.5 : (1 - Math.cos(th)) / (th * th);
  const [x, y, z] = w;
  return [1 - b * (y * y + z * z), -a * z + b * x * y, a * y + b * x * z,
          a * z + b * x * y, 1 - b * (x * x + z * z), -a * x + b * y * z,
          -a * y + b * x * z, a * x + b * y * z, 1 - b * (x * x + y * y)];
}
const mv = (A, v) => [0, 1, 2].map(i => A[3 * i] * v[0] + A[3 * i + 1] * v[1] + A[3 * i + 2] * v[2]);
const mtv = (A, v) => [0, 1, 2].map(i => A[i] * v[0] + A[3 + i] * v[1] + A[6 + i] * v[2]);  // A^T v
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

const adam = (s, j, g, lr) => adamStep(s, j, g, lr, EPS);
// The pull to the SfM's values: p -= lr p / sigma (lr: the state's rate over the units' scale, the same for each)
function pull(s, lr, sigma) {
  if (!(sigma > 0)) return;
  const k = Math.max(0, 1 - lr / sigma);
  for (let j = 0; j < s.p.length; j++) s.p[j] *= k;
}
// The mean r^2 = (x / z)^2 + (y / z)^2 over a lens's image, at the SfM's focal (fixed: updateLens's a)
const meanR2 = (cam) => ((cam.width ** 2 + cam.height ** 2) / (12 * (cam.sfmLens ?? cam).fx ** 2)) || 1 / 3;

export class Poses {
  constructor(radius = 1) {
    this.radius = radius;      // the scene's: rho's unit
    this.views = new Map();    // image name -> state of [rho / radius xyz, phi xyz]
    this.lenses = new Map();   // camera (object) -> state of [log f, cx, cy (full resolution px), k1 (the residual's)]
  }

  // One Adam step of view `img`'s pose on its gradient (g: dL/drho xyz, dL/dphi xyz) at rates lrRot (rad) and
  // lrRot x the radius, then the pull to the SfM's (sigma: rad, 0 = none)
  update(img, g, lrRot, sigma = 0) {
    let s = this.views.get(img.name);
    if (!s) this.views.set(img.name, s = adamState(6));
    s.t++;
    for (let j = 0; j < 6; j++) adam(s, j, j < 3 ? g[j] * this.radius : g[j], lrRot);
    pull(s, lrRot, sigma);
  }

  // One Adam step of a lens on the sum of its views' gradients (dL/dlog f, dL/dcx, dL/dcy, cx and cy in full
  // resolution px, dL/dk1: the radial distortion the SfM's undistortion left, shaders.wgsl distort_jac). Its state:
  // [a, cx, cy, k1], a = log f + log(1 + k1 m), the image's mean magnification (m: the mean r^2 over the image), so
  // that k1 bends the radial profile without the scale the focal has (learned as log f and k1, they trade one for
  // the other and creep: lens.e2e.js). a at rate lr, the principal point at lr x the image's width, k1 at lr over r^2
  // at the image's corner (3 m: the corner moves as fast as with a); then the pull (sigma: rad; the principal point's x
  // the width, k1's over the corner's r^2)
  updateLens(camera, g, lr, sigma = 0) {
    let s = this.lenses.get(camera);
    if (!s) this.lenses.set(camera, s = adamState(4));
    s.t++;
    const m = meanR2(camera), gk = (g[3] ?? 0) - g[0] * m / (1 + s.p[3] * m);  // dL/dk1 at a fixed
    const rate = [lr, lr * camera.width, lr * camera.width, lr / (3 * m)];
    [g[0], g[1], g[2], gk].forEach((x, j) => adam(s, j, x, rate[j]));
    pull(s, lr, sigma);
  }

  // The gauge (the header) over the views refined; imagesByName: name -> image
  center(imagesByName) {
    const vs = [...this.views].map(([n, s]) => ({ s, img: imagesByName.get(n) })).filter(x => x.img);
    if (vs.length < 3) return;
    const sfm = (img) => img.sfmPose ?? { R: img.R, center: img.center };
    const c0 = [0, 1, 2].map(k => vs.reduce((a, x) => a + sfm(x.img).center[k], 0) / vs.length);
    // world rotation and centre move of each: w = R0^T phi, dc = -R0^T rho
    for (const x of vs) {
      const R0 = sfm(x.img).R;
      x.w = mtv(R0, [x.s.p[3], x.s.p[4], x.s.p[5]]);
      x.dc = mtv(R0, [x.s.p[0], x.s.p[1], x.s.p[2]].map(v => -v * this.radius));
      x.d = sfm(x.img).center.map((v, k) => v - c0[k]);
    }
    const w = [0, 1, 2].map(k => vs.reduce((a, x) => a + x.w[k], 0) / vs.length), q = w.map(v => -v);
    const tau = [0, 1, 2].map(k => vs.reduce((a, x) => a + x.dc[k], 0) / vs.length);
    let num = 0, den = 0;
    for (const x of vs) {
      const r = cross(q, x.d).map((v, k) => x.dc[k] - tau[k] - v);
      num += r[0] * x.d[0] + r[1] * x.d[1] + r[2] * x.d[2];
      den += x.d[0] ** 2 + x.d[1] ** 2 + x.d[2] ** 2;
    }
    const sig = den > 0 ? num / den : 0;
    for (const x of vs) {
      const R0 = sfm(x.img).R, qd = cross(q, x.d);
      const phi = mv(R0, x.w.map((v, k) => v - w[k]));
      const rho = mv(R0, x.dc.map((v, k) => -(v - qd[k] - tau[k] - sig * x.d[k]) / this.radius));
      x.s.p.set([...rho, ...phi]);
    }
  }

  // The increment of a view without its own: its neighbours' in name order, as world quantities, in its frame
  between(img) {
    if (!this.views.size) return null;
    if (this._sorted?.length !== this.views.size) this._sorted = [...this.views.keys()].sort(byName);
    const { a, b, t } = neighbours(this._sorted, img.name);
    const world = (n) => {  // [w, dc] of view n (its SfM rotation: this._R0 by name)
      const p = this.views.get(n).p, R0 = this._R0.get(n);
      return [mtv(R0, [p[3], p[4], p[5]]), mtv(R0, [p[0], p[1], p[2]])];
    };
    if (!this._R0?.has(a) || !this._R0?.has(b)) return null;
    const [wa, da] = world(a), [wb, db] = world(b), R0 = (img.sfmPose ?? img).R;
    const lerp = (x, y) => x.map((v, k) => (1 - t) * v + t * y[k]);
    return [...mv(R0, lerp(da, db)), ...mv(R0, lerp(wa, wb))];
  }

  // Each image's pose and each lens's intrinsics from the SfM's through the increments (held-out views: between)
  apply(images) {
    this._R0 = new Map(images.map(im => [im.name, (im.sfmPose ?? im).R]));
    for (const img of images) {
      const p = this.views.get(img.name)?.p ?? this.between(img);
      if (!p) continue;
      img.sfmPose ??= { R: img.R, t: img.t, center: img.center };
      const E = expRot([p[3], p[4], p[5]]), { R, t } = img.sfmPose;
      setPose(img, mul3(E, R), mv(E, t).map((v, k) => v + p[k] * this.radius));
    }
    for (const [cam, s] of this.lenses) {
      cam.sfmLens ??= { fx: cam.fx, fy: cam.fy, cx: cam.cx, cy: cam.cy };
      const L = cam.sfmLens, k = Math.exp(s.p[0]) / (1 + s.p[3] * meanR2(cam));  // updateLens's a
      Object.assign(cam, { fx: L.fx * k, fy: L.fy * k, cx: L.cx + s.p[1], cy: L.cy + s.p[2], k1Residual: s.p[3] });
    }
  }

  // The increments by image name (rho in the scene's units: a trainer made again has another radius) and the lenses by
  // the name of their first image (a snapshot's), and back
  toJSON(images) {
    const lensName = new Map();
    for (const im of images) if (!lensName.has(im.camera)) lensName.set(im.camera, im.name);
    return { views: Object.fromEntries([...this.views].map(([n, s]) => [n, Array.from(s.p, (x, j) => j < 3 ? x * this.radius : x)])),
             lenses: Object.fromEntries([...this.lenses].filter(([c]) => lensName.has(c)).map(([c, s]) => [lensName.get(c), [...s.p]])) };
  }
  static fromJSON(obj, images, radius) {
    const P = new Poses(radius), byImg = new Map(images.map(im => [im.name, im]));
    for (const [n, p] of Object.entries(obj?.views ?? {})) { const s = adamState(6); s.p.set(p.map((x, j) => j < 3 ? x / radius : x)); P.views.set(n, s); }
    for (const [n, p] of Object.entries(obj?.lenses ?? {})) { const im = byImg.get(n); if (im) { const s = adamState(4); s.p.set(p); P.lenses.set(im.camera, s); } }  // k1 0 in older ones
    return P;
  }
}
