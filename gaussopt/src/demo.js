// demo.js — "Demo" free camera: a slow, hands-off flight that loosely follows the capture path
// (DemoFlight); and the gallery's fly-by, on the path itself (PathFlight, below).
//
// The path is the dataset's camera centers in path order (capture (name) order, or the shortest path:
// app.js _pathImages, DESIGN-sfm §13.3 P7), smoothed, walked by arc
// length at `speed` scene radii per second; it turns back at the ends, or loops when the path
// closes on itself (no path: a circle around the scene). The camera hangs on a slightly
// underdamped spring from a target on the path plus a smooth wander offset (in units of the
// cameras' near depth), and looks at a point that follows the original cameras' view direction,
// on a spring of its own: it lags, overshoots a little and drifts, instead of replaying the track.
"use strict";

import { clamp, dot, cross, sub, madd, scale, normalize, viewFromAxes, orbitBasis, viewMatToQuat, viewFromQuat, quatSlerp } from './math.js';

// Smooth, non-repeating wander in [-1, 1]: three incommensurate sines per channel
const wander = (t, k) => (Math.sin(0.11 * t + 1.7 * k) + 0.6 * Math.sin(0.27 * t + 4.1 * k) + 0.35 * Math.sin(0.63 * t + 2.3 * k)) / 1.95;

const qdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
const normalize4 = (q) => { const l = Math.hypot(...q) || 1; return q.map(x => x / l); };

// Damped spring step: x'' = w^2 (target - x) - 2 zeta w x'
function spring(x, v, target, w, zeta, dt) {
  const a = madd(scale(sub(target, x), w * w), v, -2 * zeta * w);
  const v2 = madd(v, a, dt);
  return [madd(x, v2, dt), v2];
}

export class DemoFlight {
  constructor() {
    this.speed = 0.05;   // scene radii per second along the path
    this.wander = 0.5;   // wander amplitude, in units of the cameras' near depth (see start)
  }

  // images: the dataset images in path order ({ center, view, zMin, zMax }); from: the pose on screen
  start(images, { center, view }, up, sceneCenter, sceneRadius) {
    this.up = normalize(up);
    this.R = sceneRadius || 1;
    // wander unit: the median near depth (5th percentile) of the cameras, about their clearance
    // above the content; the scene radius is set by far points and is much larger for a drone
    const zs = images.map(im => im.zMin).filter(z => z > 0).sort((a, b) => a - b);
    this.near = clamp(zs.length ? zs[zs.length >> 1] : 0.2 * this.R, 0.02 * this.R, this.R);
    this._buildPath(images, sceneCenter);
    // start at the path point nearest to the current camera, heading on
    let best = 0, bestD = Infinity;
    this.pts.forEach((p, i) => { const d = dot(sub(p, center), sub(p, center)); if (d < bestD) { bestD = d; best = i; } });
    this.s = this.arc[best];
    this.dir = 1;
    this.t = 0;
    this.pos = [...center]; this.vel = [0, 0, 0];
    const f = [view[2], view[6], view[10]];
    this.look = madd(center, f, this._sample(this.s).depth); this.lookVel = [0, 0, 0];
  }

  _buildPath(images, sceneCenter) {
    const R = this.R;
    let pts = images.map(im => [...im.center]);
    let fwd = images.map(im => [im.view[2], im.view[6], im.view[10]]);
    let depth = images.map(im => clamp(0.5 * ((im.zMin ?? 0.2 * R) + (im.zMax ?? R)), 0.2 * R, 2 * R));
    const len = (a) => a.reduce((L, p, i) => i ? L + Math.hypot(...sub(p, a[i - 1])) : 0, 0);
    if (pts.length < 2 || len(pts) < 0.05 * R) {  // no usable path: a circle around the scene
      const { U, e1, e2 } = orbitBasis(this.up);
      pts = []; fwd = []; depth = [];
      for (let k = 0; k <= 48; k++) {
        const a = 2 * Math.PI * k / 48, p = madd(madd(madd([...sceneCenter], e1, 1.6 * R * Math.cos(a)), e2, 1.6 * R * Math.sin(a)), U, 0.5 * R);
        pts.push(p); fwd.push(normalize(sub(sceneCenter, p))); depth.push(1.6 * R);
      }
    }
    // moving average over ~1/60 of the path (at least 2 neighbours): hand-held jitter out
    const w = Math.max(2, Math.round(pts.length / 60));
    const avg = (a, i) => {
      const s = [0, 0, 0]; let n = 0;
      for (let j = Math.max(0, i - w); j <= Math.min(a.length - 1, i + w); j++, n++) for (let k = 0; k < 3; k++) s[k] += a[j][k];
      return s.map(x => x / n);
    };
    this.pts = pts.map((_, i) => avg(pts, i));
    this.fwd = fwd.map((_, i) => normalize(avg(fwd, i)));
    this.depth = depth;
    this.arc = [0];
    for (let i = 1; i < this.pts.length; i++) this.arc.push(this.arc[i - 1] + Math.max(1e-9, Math.hypot(...sub(this.pts[i], this.pts[i - 1]))));
    this.length = this.arc.at(-1);
    this.closed = Math.hypot(...sub(this.pts[0], this.pts.at(-1))) < 0.1 * this.length;
  }

  // Path point, view direction and look depth at arc length s
  _sample(s) {
    const a = this.arc;
    s = clamp(s, 0, this.length);
    let lo = 0, hi = a.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (a[m] <= s) lo = m; else hi = m; }
    const u = (s - a[lo]) / Math.max(1e-9, a[hi] - a[lo]);
    const mix = (x, y) => x.map((v, k) => v + (y[k] - v) * u);
    return { p: mix(this.pts[lo], this.pts[hi]), f: normalize(mix(this.fwd[lo], this.fwd[hi])),
             depth: this.depth[lo] + (this.depth[hi] - this.depth[lo]) * u };
  }

  update(dt) {
    dt = clamp(dt, 0, 0.05);
    const R = this.R;
    this.t += dt;
    this.s += this.dir * this.speed * R * dt;
    if (this.closed) this.s = ((this.s % this.length) + this.length) % this.length;
    else if (this.s > this.length || this.s < 0) { this.dir = -this.dir; this.s = clamp(this.s, 0, this.length); }
    const q = this._sample(this.s), t = this.t, A = this.wander * this.near;
    // wander: sideways in full; vertically only up, and a third of it, so it drifts more than it
    // bobs and never sinks below the path towards the ground
    const off = [wander(t, 0), wander(t, 1), wander(t, 2)].map(v => v * A);
    const vert = dot(off, this.up);
    const target = madd(madd(q.p, off, 1), this.up, 0.3 * Math.abs(vert) - vert);
    [this.pos, this.vel] = spring(this.pos, this.vel, target, 0.7, 0.6, dt);
    const look = madd(madd(q.p, q.f, q.depth), [wander(t, 3), wander(t, 4), wander(t, 5)], 0.8 * A);
    [this.look, this.lookVel] = spring(this.look, this.lookVel, look, 0.9, 0.7, dt);
  }

  // View matrix looking from the camera at the look point (OpenCV axes, `up` kept up)
  view() {
    let f = normalize(sub(this.look, this.pos));
    if (Math.abs(dot(f, this.up)) > 0.995) f = normalize(madd(f, orbitBasis(this.up).e1, 0.1));
    const r = normalize(cross(f, this.up)), d = cross(f, r);
    return viewFromAxes(r, d, f, this.pos);
  }

  hud() {
    const pct = this.length > 0 ? 100 * this.s / this.length : 0;
    return `demo: ${pct.toFixed(0)}% of the path ${this.closed ? '(loop)' : this.dir > 0 ? '→' : '←'}`;
  }
}

// A flight on the path itself (the gallery's fly-by). The curve: through each camera's center at its
// orientation, so the view passes through every frustum drawn; between two cameras a cubic Hermite curve
// over time, of the center and of the orientation's quaternion (normalized), with Catmull-Rom tangents
// capped by the neighbouring steps' rates (no overshoot, none at all next to a turn in place). A step
// lasts its length at `speed` scene radii per second, or its turn at `speed` / 0.05 x 30 deg per s,
// whichever is longer. smooth (0 to 1): the curve sampled every SMOOTH_DT s, its samples averaged over a
// Gaussian window of smooth x SMOOTH_SECONDS of flight (the cameras' jitter, and the changes of speed
// between steps, out), near the cameras rather than through them; 0: the curve itself. It turns back at
// the ends. Starting from the pose on screen, it blends into the path over the first `blend` s.
const SMOOTH_SECONDS = 2, SMOOTH_DT = 1 / 30;
export class PathFlight {
  constructor() {
    this.speed = 0.05;  // scene radii per second along the path
    this.smooth = 0.5;  // 0: through each camera; 1: averaged over SMOOTH_SECONDS of flight
    this.blend = 1.5;   // seconds from the pose on screen to the path
  }

  // images: the cameras in path order ({ center, view }, views upright for the screen); from: the pose on screen
  start(images, { center, view }, sceneRadius) {
    this.R = sceneRadius || 1;
    this.c = images.map(im => [...im.center]);
    this.q = [];
    for (const im of images) {  // each on the side of the previous one: the curve and averages take the short way
      const q = viewMatToQuat(im.view), p = this.q.at(-1);
      this.q.push(p && qdot(q, p) < 0 ? q.map(x => -x) : q);
    }
    this._build();
    let best = 0;
    this.c.forEach((p, i) => { if (dot(sub(p, center), sub(p, center)) < dot(sub(this.c[best], center), sub(this.c[best], center))) best = i; });
    this.t = this.T[best];
    this.dir = 1;
    this.age = 0;
    this.from = { c: [...center], q: viewMatToQuat(view) };
    this.update(0);
  }

  // The curve: knot times T (seconds from the first camera), tangents m (centers), mq (quaternions); with
  // smooth, its averaged samples
  _build() {
    const n = this.c.length, rate = 30 * Math.PI / 180 / 0.05;
    this.T = [0];
    for (let k = 1; k < n; k++) {
      const d = Math.hypot(...sub(this.c[k], this.c[k - 1])), a = 2 * Math.acos(clamp(Math.abs(qdot(this.q[k], this.q[k - 1])), 0, 1));
      this.T.push(this.T[k - 1] + Math.max(d / (this.speed * this.R), a / (this.speed * rate), 1e-3));
    }
    this.duration = this.T[n - 1];
    const tangents = (x) => {
      const d = (a, b) => a.map((v, i) => v - b[i]), len = (a) => Math.hypot(...a);
      const v = (k) => len(d(x[k + 1], x[k])) / (this.T[k + 1] - this.T[k]);  // rate of step k
      return x.map((_, k) => {
        if (k === 0 || k === n - 1) return x[k].map(() => 0);
        const m = d(x[k + 1], x[k - 1]).map(a => a / (this.T[k + 1] - this.T[k - 1])), l = len(m);
        return l > 0 ? m.map(a => a * Math.min(1, Math.min(v(k - 1), v(k)) / l)) : m;
      });
    };
    this.m = tangents(this.c);
    this.mq = tangents(this.q);
    this.samples = null;
    const sigma = this.smooth * SMOOTH_SECONDS;
    if (sigma > 0 && n > 1) {
      const N = Math.ceil(this.duration / SMOOTH_DT) + 1, raw = Array.from({ length: N }, (_, i) => this._curve(i * SMOOTH_DT));
      const r = Math.ceil(3 * sigma / SMOOTH_DT), w = Array.from({ length: r + 1 }, (_, i) => Math.exp(-0.5 * (i * SMOOTH_DT / sigma) ** 2));
      this.samples = raw.map((_, i) => {
        const c = [0, 0, 0], q = [0, 0, 0, 0];
        let W = 0;
        for (let j = Math.max(0, i - r); j <= Math.min(N - 1, i + r); j++) {
          const wj = w[Math.abs(j - i)];  // the samples' signs follow each other along the curve (start): no flip here
          for (let k = 0; k < 3; k++) c[k] += wj * raw[j].c[k];
          for (let k = 0; k < 4; k++) q[k] += wj * raw[j].q[k];
          W += wj;
        }
        return { c: scale(c, 1 / W), q: normalize4(q) };
      });
    }
    this.built = { speed: this.speed, smooth: this.smooth };
  }

  // The curve's center and orientation at time t
  _curve(t) {
    const T = this.T, n = T.length;
    if (n < 2) return { c: [...this.c[0]], q: this.q[0] };
    t = clamp(t, 0, this.duration);
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (T[m] <= t) lo = m; else hi = m; }
    const h = T[hi] - T[lo], u = (t - T[lo]) / h, u2 = u * u, u3 = u2 * u;
    const [h00, h10, h01, h11] = [2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u, -2 * u3 + 3 * u2, u3 - u2];
    const hermite = (x, m) => x[lo].map((_, i) => h00 * x[lo][i] + h10 * h * m[lo][i] + h01 * x[hi][i] + h11 * h * m[hi][i]);
    const q = hermite(this.q, this.mq), l = Math.hypot(...q);
    return { c: hermite(this.c, this.m), q: l > 1e-9 ? q.map(x => x / l) : quatSlerp(this.q[lo], this.q[hi], u) };
  }

  // The path flown at time t: the curve, or with smooth its averaged samples (interpolated)
  _sample(t) {
    if (!this.samples) return this._curve(t);
    const S = this.samples, x = clamp(t, 0, this.duration) / SMOOTH_DT, i = Math.min(Math.floor(x), S.length - 2), u = clamp(x - i, 0, 1);
    if (i < 0) return S[0];
    return { c: S[i].c.map((v, k) => v + (S[i + 1].c[k] - v) * u), q: quatSlerp(S[i].q, S[i + 1].q, u) };
  }

  update(dt) {
    dt = clamp(dt, 0, 0.05);
    if (this.speed !== this.built.speed || this.smooth !== this.built.smooth) {  // the same place on the new knots
      const f = this.duration > 0 ? this.t / this.duration : 0;
      this._build();
      this.t = f * this.duration;
    }
    this.t += this.dir * dt;
    if (this.t > this.duration || this.t < 0) { this.dir = -this.dir; this.t = clamp(this.t, 0, this.duration); }
    this.age += dt;
    let { c, q } = this._sample(this.t);
    if (this.age < this.blend) {
      const u = this.age / this.blend, w = u * u * (3 - 2 * u);
      c = c.map((x, i) => this.from.c[i] + (x - this.from.c[i]) * w);
      q = quatSlerp(this.from.q, q, w);
    }
    this.pos = c;
    this.quat = q;
  }

  view() { return viewFromQuat(this.quat, this.pos); }

  // The step's two cameras and the fraction of it flown: { lo, hi, u }
  where() {
    const T = this.T;
    let lo = 0;
    while (lo < T.length - 2 && T[lo + 1] <= this.t) lo++;
    const hi = Math.min(lo + 1, T.length - 1);
    return { lo, hi, u: hi > lo ? clamp((this.t - T[lo]) / (T[hi] - T[lo]), 0, 1) : 0 };
  }

  hud() {
    const { lo, hi, u } = this.where();
    return `fly-by: camera ${(u < 0.5 ? lo : hi) + 1} of ${this.c.length} ${this.dir > 0 ? '→' : '←'}`;
  }
}
