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
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
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

// The view `v` turned about its forward axis by the quarter turns that bring its down axis closest to
// -up: a photo rotated by 90° to the set's orientation (photos.js) shown upright. { view, turns }: with
// odd turns, the image's width and height (fx, fy) swap.
export function uprightView(v, up) {
  const a = -(v[0] * up[0] + v[4] * up[1] + v[8] * up[2]), b = -(v[1] * up[0] + v[5] * up[1] + v[9] * up[2]);
  const turns = ((Math.round(Math.atan2(a, b) / (0.5 * Math.PI)) % 4) + 4) % 4;
  const c = [1, 0, -1, 0][turns], s = [0, 1, 0, -1][turns], w = new Float32Array(v);
  for (let j = 0; j < 16; j += 4) {  // right' = c right - s down, down' = c down + s right
    w[j] = c * v[j] - s * v[j + 1];
    w[j + 1] = c * v[j + 1] + s * v[j];
  }
  return { view: w, turns };
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

// World up from the cameras (OpenCV: +X right, +Y down in the image). People keep the horizon level
// but tilt the camera up and down, so the mean "down" axis leans with the mean pitch (5-12° on the
// photo sets): up is instead the direction most perpendicular to the photos' horizontal axes. A
// photo's horizontal axis is its X, or its Y when it was rotated by 90° to the set's orientation
// (photos.js; its down is then -X): the one closer to horizontal. The up that levels the most photos,
// among the ups perpendicular to two photos' horizontal axes, is refit on those (smallest eigenvector of
// the sum of h h^T). With every horizontal axis about parallel (photos facing one way), only the roll is fixed:
// -(mean +Y) made perpendicular to them.
const LEVEL_SIN = Math.sin(10 * Math.PI / 180), UP_SEEDS = 32;
export function estimateWorldUp(images, fallback = [0, -1, 0]) {
  const cams = (images || []).filter(im => im?.R).map(im => ({ x: [im.R[0], im.R[1], im.R[2]], y: [im.R[3], im.R[4], im.R[5]] }));
  if (!cams.length) return [...fallback];
  const meanDown = cams.reduce((s, c) => madd(s, c.y, 1), [0, 0, 0]);
  if (cams.length < 3) return Math.hypot(...meanDown) > 1e-3 ? normalize(scale(meanDown, -1)) : [...fallback];
  // per camera: its horizontal axis h and its down axis under up u
  const axes = (c, u) => Math.abs(dot(c.x, u)) <= Math.abs(dot(c.y, u)) ? { h: c.x, down: c.y } : { h: c.y, down: scale(c.x, -1) };
  // level under u: the horizontal axis within 10° of horizontal, the down axis within 60° of down (a
  // photo shot steeply down has its X about level both ways round: only its down axis tells)
  const level = (u) => cams.filter(c => { const a = axes(c, u); return Math.abs(dot(a.h, u)) < LEVEL_SIN && dot(a.down, u) < -0.5; });
  // candidates: -(mean +Y), and the up perpendicular to two photos' horizontal axes (each X or Y, either
  // sign); the one that levels the most photos
  let best = null;
  const tryUp = (u) => {
    if (Math.hypot(...u) < 0.2) return;  // two about parallel axes
    u = normalize(u);
    for (const w of [u, scale(u, -1)]) {
      const n = level(w).length;
      if (!best || n > best.n) best = { u: w, n };
    }
  };
  tryUp(scale(meanDown, -1));
  const step = cams.length / Math.min(cams.length, UP_SEEDS), seeds = [];  // evenly spaced (video: hundreds of keyframes)
  for (let k = 0; k < Math.min(cams.length, UP_SEEDS); k++) seeds.push(cams[Math.floor(k * step)]);
  for (let i = 0; i < seeds.length; i++) for (let j = i + 1; j < seeds.length; j++) {
    for (const a of [seeds[i].x, seeds[i].y]) for (const b of [seeds[j].x, seeds[j].y]) tryUp(cross(a, b));
  }
  let up = best.u;
  for (let it = 0; it < 3; it++) {
    const hs = (level(up).length >= 3 ? level(up) : cams).map(c => axes(c, up).h);
    const M = [0, 0, 0, 0, 0, 0];
    for (const h of hs) { M[0] += h[0] * h[0]; M[1] += h[0] * h[1]; M[2] += h[0] * h[2]; M[3] += h[1] * h[1]; M[4] += h[1] * h[2]; M[5] += h[2] * h[2]; }
    const { vals, vecs } = symEigen3(...M), o = [0, 1, 2].sort((a, b) => vals[a] - vals[b]);
    const v = (k) => [vecs[o[k]], vecs[3 + o[k]], vecs[6 + o[k]]];
    if (vals[o[1]] < 0.05 * vals[o[2]]) {  // the horizontal axes about parallel: level the roll only
      const h = v(2), s = scale(meanDown, -1);
      const p = madd(s, h, -dot(s, h));
      return Math.hypot(...p) > 1e-6 ? normalize(p) : [...fallback];
    }
    up = dot(v(0), up) < 0 ? scale(v(0), -1) : v(0);
  }
  return up;
}

// The world up turned about the view direction `forward` so the picture turns by `deg` (> 0:
// clockwise on screen): the orbit camera rebuilt with it at the same center and forward is the one
// on screen rolled by `deg` (the view popover's Level, app.js tiltView)
export const tiltUp = (up, forward, deg) => normalize(rotate(up, normalize(forward), -deg * Math.PI / 180));

// The scene's global transform (project.json `transform`): a 4x4 row-major matrix from the scene's
// world to the leveled one, identity by default; a rotation R only for now (Level). The views' up is
// R^T applied to the cameras' estimated up, so the identity keeps the estimate.
export const IDENTITY4 = Object.freeze([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
export const upFromTransform = (T, up0) => normalize([0, 1, 2].map(j => T[j] * up0[0] + T[4 + j] * up0[1] + T[8 + j] * up0[2]));
// T after a Level turn of `deg` about `forward` (tiltUp): R' = R rot(forward, deg), so that
// upFromTransform(R', up0) = tiltUp(upFromTransform(R, up0), forward, deg)
export function levelTransform(T, forward, deg) {
  const k = normalize(forward), a = deg * Math.PI / 180;
  const cols = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map(e => rotate(e, k, a));  // rot's columns
  const out = [...T];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) out[4 * i + j] = T[4 * i] * cols[j][0] + T[4 * i + 1] * cols[j][1] + T[4 * i + 2] * cols[j][2];
  return out;
}
// A transform read from JSON: 16 finite numbers, or the identity
export const parseTransform = (T) => Array.isArray(T) && T.length === 16 && T.every(Number.isFinite) ? [...T] : [...IDENTITY4];

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

// The largest w x h box of the image's aspect inside a W x H canvas, centered: { x, y, w, h } in
// canvas pixels (the 2D compare modes draw the photo there, app.js maps the split wipe through it)
export function fitRect(width, height, W, H) {
  const s = Math.min(W / width, H / height), w = Math.round(width * s), h = Math.round(height * s);
  return { x: Math.floor(0.5 * (W - w)), y: Math.floor(0.5 * (H - h)), w, h };
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
