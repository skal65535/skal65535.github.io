// math.js — Shared math for Gauss-Opt (plain JS arrays, no dependencies):
//   - scalars: clamp, wrapPi, approach, sigmoid / logit, random numbers
//   - vec3 helpers, Rodrigues rotation
//   - quaternions / rotation matrices, symmetric 3x3 eigen-decomposition
//   - camera geometry: OpenCV view matrices, pinhole projection, orbit frame, world-up, intrinsics fit
// Conventions: 3x3 matrices are flat row-major; 4x4 matrices are flat column-major (WGSL mat4x4f).
// Quaternions are [w, x, y, z]. Camera space is OpenCV/COLMAP: +X right, +Y down, +Z forward.
"use strict";

// ── Scalars ──────────────────────────────────────────────────────────────────
export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
// angle wrapped to (-pi, pi]
export const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));
// first-order approach of x to target at `rate` (1/s)
export const approach = (x, target, rate, dt) => x + (target - x) * Math.min(1, rate * dt);

export function sigmoid(x) {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}

export function logit(p) {
  const c = Math.min(0.995, Math.max(0.001, p));
  return Math.log(c / (1 - c));
}

export function randn() {
  const u1 = Math.max(1e-7, Math.random());
  const u2 = Math.random();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

// Deterministic RNG in [0, 1) (mulberry32)
export function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    let t = s = (s + 0x6D2B79F5) >>> 0;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// ── vec3 ─────────────────────────────────────────────────────────────────────
export const dot   = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const sub   = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const madd  = (a, b, s) => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s];  // a + b * s
export const normalize = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

// Rotate v around unit axis k by angle a (Rodrigues)
export function rotate(v, k, a) {
  const c = Math.cos(a), s = Math.sin(a);
  const kxv = cross(k, v);
  const kdv = dot(k, v) * (1 - c);
  return [v[0] * c + kxv[0] * s + k[0] * kdv, v[1] * c + kxv[1] * s + k[1] * kdv, v[2] * c + kxv[2] * s + k[2] * kdv];
}

// ── Rotations ────────────────────────────────────────────────────────────────
// Quaternion (normalized here) -> row-major rotation matrix R
export function quatToMat3([qw, qx, qy, qz]) {
  const n = Math.hypot(qw, qx, qy, qz) || 1;
  const w = qw / n, x = qx / n, y = qy / n, z = qz / n;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
    2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
  ];
}

// Row-major rotation matrix -> unit quaternion (Shepperd; inverse of quatToMat3)
export function mat3ToQuat(m) {
  const tr = m[0] + m[4] + m[8];
  let w, x, y, z;
  if (tr > 0) {
    const S = Math.sqrt(tr + 1) * 2;
    w = 0.25 * S; x = (m[7] - m[5]) / S; y = (m[2] - m[6]) / S; z = (m[3] - m[1]) / S;
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const S = Math.sqrt(1 + m[0] - m[4] - m[8]) * 2;
    w = (m[7] - m[5]) / S; x = 0.25 * S; y = (m[1] + m[3]) / S; z = (m[2] + m[6]) / S;
  } else if (m[4] > m[8]) {
    const S = Math.sqrt(1 + m[4] - m[0] - m[8]) * 2;
    w = (m[2] - m[6]) / S; x = (m[1] + m[3]) / S; y = 0.25 * S; z = (m[5] + m[7]) / S;
  } else {
    const S = Math.sqrt(1 + m[8] - m[0] - m[4]) * 2;
    w = (m[3] - m[1]) / S; x = (m[2] + m[6]) / S; y = (m[5] + m[7]) / S; z = 0.25 * S;
  }
  const l = Math.hypot(w, x, y, z) || 1;
  return [w / l, x / l, y / l, z / l];
}

// Unit quaternion of the world -> camera rotation of a 4x4 view matrix
export function viewMatToQuat(v) {
  return mat3ToQuat([v[0], v[4], v[8], v[1], v[5], v[9], v[2], v[6], v[10]]);
}

// Shortest-arc slerp (lerp when nearly parallel), renormalized
export function quatSlerp(a, b, t) {
  let cosom = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const sign = (cosom < 0) ? -1 : 1;
  cosom *= sign;
  let s0 = 1 - t, s1 = t;
  if (1 - cosom > 1e-5) {
    const omega = Math.acos(Math.min(1, cosom));
    const sinom = Math.sin(omega);
    s0 = Math.sin(s0 * omega) / sinom;
    s1 = Math.sin(t * omega) / sinom;
  }
  s1 *= sign;
  const q = [0, 1, 2, 3].map(k => s0 * a[k] + s1 * b[k]);
  const len = Math.hypot(...q) || 1;
  return q.map(v => v / len);
}

// Eigen-decomposition of a symmetric 3x3 matrix [a00, a01, a02, a11, a12, a22] (cyclic Jacobi).
// Returns { vals: [3], vecs: flat row-major 3x3 whose COLUMNS are the eigenvectors }.
export function symEigen3(a00, a01, a02, a11, a12, a22) {
  const A = [[a00, a01, a02], [a01, a11, a12], [a02, a12, a22]];
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 12; sweep++) {
    const off = A[0][1] * A[0][1] + A[0][2] * A[0][2] + A[1][2] * A[1][2];
    if (off < 1e-30) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(A[p][q]) < 1e-30) continue;
      const th = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]);
      const c = Math.cos(th), sn = Math.sin(th);
      for (let k = 0; k < 3; k++) {  // A = J^T A J
        const akp = A[k][p], akq = A[k][q];
        A[k][p] = c * akp - sn * akq; A[k][q] = sn * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = A[p][k], aqk = A[q][k];
        A[p][k] = c * apk - sn * aqk; A[q][k] = sn * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - sn * vkq; V[k][q] = sn * vkp + c * vkq;
      }
    }
  }
  return { vals: [A[0][0], A[1][1], A[2][2]], vecs: [...V[0], ...V[1], ...V[2]] };
}

// ── Camera geometry (OpenCV: +X right, +Y down, +Z forward) ─────────────────
// 4x4 view matrix (world -> camera) from row-major R and t: x_cam = R x + t
export function makeViewMatrix(R, t) {
  return new Float32Array([
    R[0], R[3], R[6], 0,
    R[1], R[4], R[7], 0,
    R[2], R[5], R[8], 0,
    t[0], t[1], t[2], 1,
  ]);
}

// 4x4 view matrix from world-space camera axes (right, down, forward) and center c
export function viewFromAxes(r, d, f, c) {
  return new Float32Array([
    r[0], d[0], f[0], 0,
    r[1], d[1], f[1], 0,
    r[2], d[2], f[2], 0,
    -dot(r, c), -dot(d, c), -dot(f, c), 1,
  ]);
}

// 4x4 view matrix from the world -> camera rotation quaternion q and camera center c
export function viewFromQuat(q, c) {
  const R = quatToMat3(q);
  return viewFromAxes(R.slice(0, 3), R.slice(3, 6), R.slice(6, 9), c);
}

// World point -> camera space [x, y, z] with a 4x4 view matrix
export function viewTransform(v, p) {
  return [0, 1, 2].map(k => v[k] * p[0] + v[4 + k] * p[1] + v[8 + k] * p[2] + v[12 + k]);
}

// World point (x, y, z) -> pixel [u, v] (into `out`) of pinhole camera `cam` ({ fx, fy, cx, cy })
// at pose x_cam = R x + t; returns the camera-space depth z. u, v divide by max(zFloor, z).
export function projectPinhole(R, t, cam, x, y, z, out, zFloor = -Infinity) {
  const zc = R[6] * x + R[7] * y + R[8] * z + t[2], zd = Math.max(zFloor, zc);
  out[0] = cam.fx * ((R[0] * x + R[1] * y + R[2] * z + t[0]) / zd) + cam.cx;
  out[1] = cam.fy * ((R[3] * x + R[4] * y + R[5] * z + t[1]) / zd) + cam.cy;
  return zc;
}

// Orthonormal orbit frame around world `up`: e1, e2 span the horizontal plane (e1 ~ world +X).
// COLMAP worlds are typically Y-down, hence the [0, -1, 0] default (see estimateWorldUp()).
export function orbitBasis(up = [0, -1, 0]) {
  const U  = normalize(up);
  const ref = (Math.abs(U[0]) > 0.9) ? [0, 0, 1] : [1, 0, 0];
  const e1 = normalize(madd(ref, U, -dot(ref, U)));
  return { U, e1, e2: cross(U, e1) };
}

// World up = -(mean camera +Y axis): COLMAP cameras are OpenCV (+Y down in image),
// and people mostly hold cameras upright, so the average "down" axis is world-down.
export function estimateWorldUp(images, fallback = [0, -1, 0]) {
  let s = [0, 0, 0];
  for (const im of images || []) s = sub(s, [im.R[3], im.R[4], im.R[5]]);
  const l = Math.hypot(s[0], s[1], s[2]);
  return (images?.length > 0 && l > 1e-3 * images.length) ? normalize(s) : [...fallback];
}

// Inverse of the orbit position: camera center -> { radius, theta, phi } (degrees) around target
export function orbitAnglesFromCenter(target, center, up = [0, -1, 0]) {
  const { U, e1, e2 } = orbitBasis(up);
  const o = sub(center, target);
  const r = Math.hypot(o[0], o[1], o[2]) || 1e-6;
  return {
    radius: r,
    phi:    Math.asin(clamp(dot(o, U) / r, -1, 1)) * 180 / Math.PI,
    theta:  Math.atan2(dot(o, e2), dot(o, e1)) * 180 / Math.PI,
  };
}

// Free-View orbit camera looking at `target`: phi > 0 raises it along `up`, theta turns around it.
// Returns { view, center }.
export function buildOrbitViewMatrix(target, radius, thetaDeg, phiDeg, up = [0, -1, 0]) {
  const theta = thetaDeg * Math.PI / 180;
  const phi   = phiDeg   * Math.PI / 180;
  const { U, e1, e2 } = orbitBasis(up);
  const ch = radius * Math.cos(phi);
  const c = madd(madd(madd(target, e1, ch * Math.cos(theta)), e2, ch * Math.sin(theta)), U, radius * Math.sin(phi));

  const f = normalize(sub(target, c));  // forward (+Z): camera -> target
  const r = normalize(cross(f, U));     // right   (+X)
  const d = cross(f, r);                // down    (+Y, OpenCV)

  return { view: viewFromAxes(r, d, f, c), center: c };
}

// COLMAP intrinsics mapped onto a W x H canvas with one uniform scale (no anisotropic
// squash when the image and canvas aspect ratios differ): the whole image fits, centered.
export function fitIntrinsics(camera, W, H) {
  const s = Math.min(W / camera.width, H / camera.height);
  return {
    fx: camera.fx * s,
    fy: camera.fy * s,
    cx: camera.cx * s + 0.5 * (W - camera.width  * s),
    cy: camera.cy * s + 0.5 * (H - camera.height * s),
  };
}
