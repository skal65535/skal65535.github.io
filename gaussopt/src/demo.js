// demo.js — "Demo" free camera: a slow, hands-off flight that loosely follows the capture path.
//
// The path is the dataset's camera centers in capture (name) order, smoothed, walked by arc
// length at `speed` scene radii per second; it turns back at the ends, or loops when the path
// closes on itself (no path: a circle around the scene). The camera hangs on a slightly
// underdamped spring from a target on the path plus a smooth wander offset (in units of the
// cameras' near depth), and looks at a point that follows the original cameras' view direction,
// on a spring of its own: it lags, overshoots a little and drifts, instead of replaying the track.
"use strict";

import { clamp, dot, cross, sub, madd, normalize, viewFromAxes, orbitBasis } from './math.js';

// Smooth, non-repeating wander in [-1, 1]: three incommensurate sines per channel
const wander = (t, k) => (Math.sin(0.11 * t + 1.7 * k) + 0.6 * Math.sin(0.27 * t + 4.1 * k) + 0.35 * Math.sin(0.63 * t + 2.3 * k)) / 1.95;

// Damped spring step: x'' = w^2 (target - x) - 2 zeta w x'
function spring(x, v, target, w, zeta, dt) {
  const a = madd(madd([0, 0, 0], sub(target, x), w * w), v, -2 * zeta * w);
  const v2 = madd(v, a, dt);
  return [madd(x, v2, dt), v2];
}

export class DemoFlight {
  constructor() {
    this.speed = 0.05;   // scene radii per second along the path
    this.wander = 0.5;   // wander amplitude, in units of the cameras' near depth (see start)
  }

  // images: name-sorted dataset images ({ center, view, zMin, zMax }); from: the pose on screen
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
