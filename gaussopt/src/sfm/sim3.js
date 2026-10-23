// sim3.js — Similarity transforms {s, R, t}: x -> s R x + t (R flat row-major), for loop closure.
// A camera pose {R, t} (world -> camera, camera.js) is the Sim(3) with s = 1; a Sim(3) world ->
// camera transform maps back to a pose as {R, t / s} (projection ignores the camera frame's scale).
"use strict";

import { mul3, transpose3, mulVec3, expSO3, logSO3 } from './linalg.js';

export const fromPose = ({ R, t }) => ({ s: 1, R: [...R], t: [...t] });
export const toPose = ({ s, R, t }) => ({ R: [...R], t: t.map(x => x / s) });

export function apply({ s, R, t }, x) {
  const r = mulVec3(R, x);
  return [s * r[0] + t[0], s * r[1] + t[1], s * r[2] + t[2]];
}

// (a o b)(x) = a(b(x))
export function compose(a, b) {
  const Rt = mulVec3(a.R, b.t);
  return { s: a.s * b.s, R: mul3(a.R, b.R), t: [a.s * Rt[0] + a.t[0], a.s * Rt[1] + a.t[1], a.s * Rt[2] + a.t[2]] };
}

export function inverse({ s, R, t }) {
  const Rt = transpose3(R), r = mulVec3(Rt, t);
  return { s: 1 / s, R: Rt, t: [-r[0] / s, -r[1] / s, -r[2] / s] };
}

// Small perturbation d = [w (3), tau (3), sigma] applied on the left: exp(d) o T
export function perturb(T, d) {
  return compose({ s: Math.exp(d[6]), R: expSO3([d[0], d[1], d[2]]), t: [d[3], d[4], d[5]] }, T);
}

// 7-vector error of a near-identity Sim(3): [log R (3), t (3), log s]
export const error = ({ s, R, t }) => [...logSO3(R), t[0], t[1], t[2], Math.log(s)];
