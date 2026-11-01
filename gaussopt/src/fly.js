// fly.js — The camera's fly-to animation (app.js flyToCamera).
"use strict";

import { clamp, sub, madd, normalize, viewMatToQuat, quatSlerp, viewFromQuat, fitIntrinsics } from './math.js';

// Fly-to animation: cubic Bezier for the camera center (bowed outward from `sceneCenter`
// and slightly towards -Y, so it arcs instead of cutting through the scene), slerp for the
// orientation, linear blend of the intrinsics, over FLY_MS.
const FLY_MS = 680;
export function createCameraSplineAnimation({
  fromView, fromCenter, fromFx, fromFy, fromCx, fromCy,
  toImg, W, H, sceneCenter = [0, 0, 0],
}) {
  const p0 = [...fromCenter], p3 = [...toImg.center];
  const d = sub(p3, p0);
  const chord = Math.hypot(...d);
  const out = normalize(sub(madd(p0, d, 0.5), sceneCenter));
  const ctrl = (a) => {
    const p = madd(madd(p0, d, a), out, 0.18 * chord);
    p[1] -= 0.04 * chord;
    return p;
  };
  const { fx: toFx, fy: toFy, cx: toCx, cy: toCy } = fitIntrinsics(toImg.camera, W, H);
  return {
    startTime: performance.now(),
    p0, p1: ctrl(0.33), p2: ctrl(0.67), p3,
    q0: viewMatToQuat(fromView), q1: viewMatToQuat(toImg.view),
    fromFx, fromFy, fromCx, fromCy, toFx, toFy, toCx, toCy,
  };
}

export function evaluateCameraSpline(anim) {
  const t = clamp((performance.now() - anim.startTime) / FLY_MS, 0, 1);
  const s = t * t * t * (t * (t * 6 - 15) + 10);  // quintic smoothstep (C2 easing)
  const u = 1 - s;
  const b = [u * u * u, 3 * u * u * s, 3 * u * s * s, s * s * s];
  const camCenter = [0, 1, 2].map(k => b[0] * anim.p0[k] + b[1] * anim.p1[k] + b[2] * anim.p2[k] + b[3] * anim.p3[k]);
  const lerp = (a, z) => a + (z - a) * s;
  return {
    done: t >= 1,
    view: viewFromQuat(quatSlerp(anim.q0, anim.q1, s), camCenter),
    camCenter,
    fx: lerp(anim.fromFx, anim.toFx), fy: lerp(anim.fromFy, anim.toFy),
    cx: lerp(anim.fromCx, anim.toCx), cy: lerp(anim.fromCy, anim.toCy),
  };
}
