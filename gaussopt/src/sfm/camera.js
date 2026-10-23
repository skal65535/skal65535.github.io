// camera.js — Camera models and poses for the SfM (COLMAP conventions):
//   - models SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV with COLMAP's parameter order
//   - distortion / iterative undistortion in normalized image coordinates
//   - poses {R, t}: world -> camera (x_cam = R x + t), R flat row-major; center C = -R^T t
// Pixel coordinates follow COLMAP: the top-left pixel's center is (0.5, 0.5).
"use strict";

import { mulVec3, transpose3 } from './linalg.js';

// name -> [COLMAP model_id, parameter names]
export const CAMERA_MODELS = {
  SIMPLE_PINHOLE: [0, ['f', 'cx', 'cy']],
  PINHOLE:        [1, ['fx', 'fy', 'cx', 'cy']],
  SIMPLE_RADIAL:  [2, ['f', 'cx', 'cy', 'k1']],
  RADIAL:         [3, ['f', 'cx', 'cy', 'k1', 'k2']],
  OPENCV:         [4, ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2']],
};
// Index of a named parameter ('f', 'fx', 'fy', 'k1', ...) in a model's params, or -1
export const paramIndex = (model, name) => CAMERA_MODELS[model][1].indexOf(name);

export function makeCamera(model, width, height, params) {
  const info = CAMERA_MODELS[model];
  if (!info) throw new Error(`Unsupported camera model ${model}`);
  if (params.length !== info[1].length) throw new Error(`${model} takes ${info[1].length} params, got ${params.length}`);
  return { model, width, height, params: Array.from(params) };
}

// COLMAP's default when nothing is known: f = 1.2 max(w, h), principal point at the center
export function defaultCamera(model, width, height) {
  const f = 1.2 * Math.max(width, height), cx = width / 2, cy = height / 2;
  const params = { SIMPLE_PINHOLE: [f, cx, cy], PINHOLE: [f, f, cx, cy], SIMPLE_RADIAL: [f, cx, cy, 0],
                   RADIAL: [f, cx, cy, 0, 0], OPENCV: [f, f, cx, cy, 0, 0, 0, 0] }[model];
  return makeCamera(model, width, height, params);
}

// [fx, fy, cx, cy] of any model
export function intrinsics({ model, params: p }) {
  return model === 'PINHOLE' || model === 'OPENCV' ? [p[0], p[1], p[2], p[3]] : [p[0], p[0], p[1], p[2]];
}

export const meanFocal = (cam) => { const [fx, fy] = intrinsics(cam); return (fx + fy) / 2; };

// The same camera with focal f (fx = f; fy keeps its ratio to fx)
export function withFocal(cam, f) {
  const params = [...cam.params], iy = paramIndex(cam.model, 'fy');
  if (iy >= 0) params[iy] = f * params[iy] / params[0];
  params[0] = f;
  return { ...cam, params };
}

// [k1, k2, p1, p2] of any model (zeros where the model has none)
export function distortionParams({ model, params: p }) {
  const get = (name) => { const i = paramIndex(model, name); return i < 0 ? 0 : p[i]; };
  return [get('k1'), get('k2'), get('p1'), get('p2')];
}

// Distortion offset (dx, dy) at normalized ideal coordinates (x, y): OPENCV's radial + tangential
// terms, which cover the other models with their missing coefficients at zero
function distortionDelta(cam, x, y) {
  const [k1, k2, p1, p2] = distortionParams(cam);
  const r2 = x * x + y * y, radial = k1 * r2 + k2 * r2 * r2;
  return [x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
          y * radial + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)];
}

export function distort(cam, x, y) {
  const [dx, dy] = distortionDelta(cam, x, y);
  return [x + dx, y + dy];
}

// Inverse of distort(): Newton iterations with a numeric Jacobian (as COLMAP's IterativeUndistortion)
export function undistort(cam, xd, yd) {
  if (cam.model === 'SIMPLE_PINHOLE' || cam.model === 'PINHOLE') return [xd, yd];
  let x = xd, y = yd;
  const h = 1e-7;
  for (let it = 0; it < 100; it++) {
    const [fx, fy] = distort(cam, x, y);
    const ex = fx - xd, ey = fy - yd;
    if (ex * ex + ey * ey < 1e-26) break;
    const [ax, ay] = distort(cam, x + h, y), [bx, by] = distort(cam, x, y + h);
    const j00 = (ax - fx) / h, j10 = (ay - fy) / h, j01 = (bx - fx) / h, j11 = (by - fy) / h;
    const det = j00 * j11 - j01 * j10;
    if (Math.abs(det) < 1e-20) break;
    x -= (j11 * ex - j01 * ey) / det;
    y -= (j00 * ey - j10 * ex) / det;
  }
  return [x, y];
}

// Normalized ideal coordinates -> pixel, and back
export function imageFromNormalized(cam, x, y) {
  const [fx, fy, cx, cy] = intrinsics(cam);
  const [u, v] = distort(cam, x, y);
  return [fx * u + cx, fy * v + cy];
}

export function normalizedFromImage(cam, u, v) {
  const [fx, fy, cx, cy] = intrinsics(cam);
  return undistort(cam, (u - cx) / fx, (v - cy) / fy);
}

// World point -> [u, v, depth]; depth <= 0 means behind the camera. Polynomial distortion folds
// back far outside the field of view (k2 r^4 wins at large r), so test inFieldOfView() before
// trusting (u, v) of a point that may lie outside the frame.
export function project(cam, pose, X) {
  const p = mulVec3(pose.R, X);
  const z = p[2] + pose.t[2];
  const [u, v] = imageFromNormalized(cam, (p[0] + pose.t[0]) / z, (p[1] + pose.t[1]) / z);
  return [u, v, z];
}

// True when the ideal (undistorted) projection lies within the frame enlarged by `margin`
// (fraction of the size). Distortion is assumed monotonic over that region.
export function inFieldOfView(cam, pose, X, margin = 0.1) {
  const p = mulVec3(pose.R, X);
  const z = p[2] + pose.t[2];
  if (z <= 0) return false;
  const [fx, fy, cx, cy] = intrinsics(cam);
  const u = fx * (p[0] + pose.t[0]) / z + cx, v = fy * (p[1] + pose.t[1]) / z + cy;
  const mx = margin * cam.width, my = margin * cam.height;
  return u >= -mx && v >= -my && u <= cam.width + mx && v <= cam.height + my;
}

export function poseCenter({ R, t }) {
  const c = mulVec3(transpose3(R), t);
  return [-c[0], -c[1], -c[2]];
}

export function poseFromCenter(R, C) {
  const rc = mulVec3(R, C);
  return { R: Array.from(R), t: [-rc[0], -rc[1], -rc[2]] };
}
