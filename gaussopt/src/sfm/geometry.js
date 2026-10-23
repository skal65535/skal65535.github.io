// geometry.js — Multi-view geometry for the SfM (plain JS, f64):
//   - fundamental matrix (normalized 8-point) and homography (normalized DLT), Sampson / transfer errors
//   - adaptive RANSAC with local refits on the inliers
//   - essential matrix decomposition with cheirality, relative pose from matches, linear triangulation
//   - P3P (Grunert's quartic, coefficients built by polynomial algebra) and RANSAC PnP
// Image points are [u, v] pixels; rays are normalized ideal coordinates [x, y] (camera.js
// normalizedFromImage); poses are world -> camera {R, t} as in camera.js.
"use strict";

import { dot, sub } from '../math.js';
import { nullVector, svd, matMul, transpose, mul3, transpose3, mulVec3, det3, rotationAngle } from './linalg.js';
import { umeyama } from './align.js';
import { project, normalizedFromImage, meanFocal } from './camera.js';
import { angleBetween, deg } from './util.js';

// ── Polynomials (coefficients highest degree first) ─────────────────────────

const polyMul = (a, b) => {
  const r = new Array(a.length + b.length - 1).fill(0);
  a.forEach((x, i) => b.forEach((y, j) => { r[i + j] += x * y; }));
  return r;
};
const polyAdd = (a, b) => {
  const n = Math.max(a.length, b.length);
  const pa = [...new Array(n - a.length).fill(0), ...a], pb = [...new Array(n - b.length).fill(0), ...b];
  return pa.map((x, i) => x + pb[i]);
};
const polyScale = (a, s) => a.map(x => x * s);
const polyEval = (a, x) => a.reduce((s, c) => s * x + c, 0);
const polyDeriv = (a) => a.slice(0, -1).map((c, i) => c * (a.length - 1 - i));

// Real roots of a polynomial: roots of the derivative split the line into monotone intervals,
// each bracketing at most one root (bisection + Newton polish). Robust for the low degrees used here.
export function realRoots(coeffs) {
  let a = [...coeffs];
  while (a.length > 1 && Math.abs(a[0]) < 1e-14 * Math.max(...a.map(Math.abs))) a.shift();
  if (a.length <= 1) return [];
  if (a.length === 2) return [-a[1] / a[0]];
  const bound = 1 + Math.max(...a.slice(1).map(c => Math.abs(c / a[0])));  // Cauchy bound
  const crit = realRoots(polyDeriv(a)).filter(x => Math.abs(x) < bound).sort((x, y) => x - y);
  const edges = [-bound, ...crit, bound];
  const roots = [];
  for (let k = 0; k + 1 < edges.length; k++) {
    let lo = edges[k], hi = edges[k + 1];
    let flo = polyEval(a, lo), fhi = polyEval(a, hi);
    if (flo === 0) { roots.push(lo); continue; }
    if (flo * fhi > 0) continue;
    for (let it = 0; it < 200 && hi - lo > 1e-15 * Math.max(1, Math.abs(lo)); it++) {
      const mid = (lo + hi) / 2, fm = polyEval(a, mid);
      if (fm === 0) { lo = hi = mid; break; }
      if (fm * flo < 0) { hi = mid; } else { lo = mid; flo = fm; }
    }
    roots.push((lo + hi) / 2);
  }
  return roots.filter((x, i) => i === 0 || Math.abs(x - roots[i - 1]) > 1e-12 * Math.max(1, Math.abs(x)));
}

// ── Normalization ────────────────────────────────────────────────────────────

// Hartley: translate to the centroid, scale to mean distance sqrt(2). Returns T (3x3) and points.
function hartley(pts) {
  const n = pts.length;
  let mx = 0, my = 0;
  for (const [x, y] of pts) { mx += x; my += y; }
  mx /= n; my /= n;
  let d = 0;
  for (const [x, y] of pts) d += Math.hypot(x - mx, y - my);
  const s = Math.SQRT2 / (d / n || 1);
  return { T: [s, 0, -s * mx, 0, s, -s * my, 0, 0, 1], q: pts.map(([x, y]) => [s * (x - mx), s * (y - my)]) };
}

// ── Fundamental matrix ───────────────────────────────────────────────────────

// x2^T F x1 = 0 from >= 8 correspondences, rank 2 enforced
export function fundamental8(x1, x2) {
  const n = x1.length;
  const a = hartley(x1), b = hartley(x2);
  const A = new Float64Array(n * 9);
  for (let i = 0; i < n; i++) {
    const [u1, v1] = a.q[i], [u2, v2] = b.q[i];
    A.set([u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, 1], i * 9);
  }
  const f = nullVector(A, n, 9);
  const { U, S, V } = svd(Float64Array.from(f), 3, 3);
  const F2 = matMul(matMul(U, [S[0], 0, 0, 0, S[1], 0, 0, 0, 0], 3, 3, 3), transpose(V, 3, 3), 3, 3, 3);
  const F = mul3(mul3(transpose3(b.T), Array.from(F2)), a.T);  // denormalize: T2^T F T1
  const norm = Math.hypot(...F) || 1;
  return F.map(x => x / norm);
}

// Squared Sampson distance (pixels^2) of x1 <-> x2 to F
export function sampsonError(F, [u1, v1], [u2, v2]) {
  const f1 = F[0] * u1 + F[1] * v1 + F[2], f2 = F[3] * u1 + F[4] * v1 + F[5], f3 = F[6] * u1 + F[7] * v1 + F[8];
  const g1 = F[0] * u2 + F[3] * v2 + F[6], g2 = F[1] * u2 + F[4] * v2 + F[7];
  const e = u2 * f1 + v2 * f2 + f3;
  return e * e / (f1 * f1 + f2 * f2 + g1 * g1 + g2 * g2 || 1e-300);
}

// ── Homography ───────────────────────────────────────────────────────────────

// x2 ~ H x1 from >= 4 correspondences
export function homographyDLT(x1, x2) {
  const n = x1.length;
  const a = hartley(x1), b = hartley(x2);
  const A = new Float64Array(2 * n * 9);
  for (let i = 0; i < n; i++) {
    const [x, y] = a.q[i], [u, v] = b.q[i];
    A.set([-x, -y, -1, 0, 0, 0, u * x, u * y, u], 18 * i);
    A.set([0, 0, 0, -x, -y, -1, v * x, v * y, v], 18 * i + 9);
  }
  const h = nullVector(A, 2 * n, 9);
  const Tb = b.T, s = Tb[0];
  const TbInv = [1 / s, 0, -Tb[2] / s, 0, 1 / s, -Tb[5] / s, 0, 0, 1];
  const H = mul3(mul3(TbInv, h), a.T);
  return H.map(x => x / (H[8] || Math.hypot(...H)));
}

// Squared transfer error |x2 - H x1|^2 (pixels^2)
export function transferError(H, [u1, v1], [u2, v2]) {
  const w = H[6] * u1 + H[7] * v1 + H[8];
  const u = (H[0] * u1 + H[1] * v1 + H[2]) / w, v = (H[3] * u1 + H[4] * v1 + H[5]) / w;
  return (u - u2) ** 2 + (v - v2) ** 2;
}

// ── RANSAC ───────────────────────────────────────────────────────────────────

function sampleIndices(rng, n, k) {
  const s = new Set();
  while (s.size < k) s.add(Math.floor(rng() * n));
  return [...s];
}

// Generic adaptive RANSAC with local refits ("LO" lite): fit(indices) -> [models], error(model, i)
// -> squared error. Returns { model, inliers: Uint8Array, numInliers } or null.
export function ransac({ n, sampleSize, fit, error, threshold, rng, confidence = 0.999, maxIterations = 2000, minIterations = 20 }) {
  if (n < sampleSize) return null;
  const t2 = threshold * threshold;
  const score = (model) => {
    const inl = new Uint8Array(n);
    let c = 0;
    for (let i = 0; i < n; i++) if (error(model, i) <= t2) { inl[i] = 1; c++; }
    return { inl, c };
  };
  let best = null;
  let needed = maxIterations;
  for (let it = 0; it < Math.min(needed, maxIterations) || it < minIterations; it++) {
    for (const model of fit(sampleIndices(rng, n, sampleSize))) {
      if (!model) continue;
      const { inl, c } = score(model);
      if (!best || c > best.numInliers) {
        best = { model, inliers: inl, numInliers: c };
        const p = (c / n) ** sampleSize;  // all-inlier sample probability; 0 when it underflows
        needed = p >= 1 ? 0 : p > 0 ? Math.ceil(Math.log(1 - confidence) / Math.log1p(-p)) : maxIterations;
      }
    }
  }
  if (!best) return null;
  for (let refit = 0; refit < 3; refit++) {  // refit on all inliers while it gains support
    const idx = [];
    best.inliers.forEach((v, i) => { if (v) idx.push(i); });
    if (idx.length <= sampleSize) break;
    const model = fit(idx)[0];
    if (!model) break;
    const { inl, c } = score(model);
    if (c < best.numInliers) break;
    const gained = c > best.numInliers;
    best = { model, inliers: inl, numInliers: c };
    if (!gained) break;
  }
  return best;
}

export const ransacFundamental = (x1, x2, { rng, threshold = 2, ...opts }) => ransac({
  n: x1.length, sampleSize: 8, rng, threshold, ...opts,
  fit: (idx) => [fundamental8(idx.map(i => x1[i]), idx.map(i => x2[i]))],
  error: (F, i) => sampsonError(F, x1[i], x2[i]),
});

export const ransacHomography = (x1, x2, { rng, threshold = 4, ...opts }) => ransac({
  n: x1.length, sampleSize: 4, rng, threshold, ...opts,
  fit: (idx) => [homographyDLT(idx.map(i => x1[i]), idx.map(i => x2[i]))],
  error: (H, i) => transferError(H, x1[i], x2[i]),
});

// ── Essential matrix, triangulation ──────────────────────────────────────────

// E = K2^T F K1 for pixel intrinsics K = [fx 0 cx; 0 fy cy; 0 0 1]
export function essentialFromFundamental(F, [fx1, fy1, cx1, cy1], [fx2, fy2, cx2, cy2] = [fx1, fy1, cx1, cy1]) {
  const K1 = [fx1, 0, cx1, 0, fy1, cy1, 0, 0, 1], K2 = [fx2, 0, cx2, 0, fy2, cy2, 0, 0, 1];
  return mul3(mul3(transpose3(K2), F), K1);
}

// The 4 (R, t) of camera 2 relative to camera 1 (|t| = 1)
function decomposeEssential(E) {
  const { U, V } = svd(Float64Array.from(E), 3, 3);
  let Uf = Array.from(U), Vf = Array.from(V);
  if (det3(Uf) < 0) Uf = Uf.map((x, i) => (i % 3 === 2 ? -x : x));
  if (det3(Vf) < 0) Vf = Vf.map((x, i) => (i % 3 === 2 ? -x : x));
  const W = [0, -1, 0, 1, 0, 0, 0, 0, 1];
  const Vt = transpose3(Vf);
  const R1 = mul3(mul3(Uf, W), Vt), R2 = mul3(mul3(Uf, transpose3(W)), Vt);
  const t = [Uf[2], Uf[5], Uf[8]];
  const nt = t.map(x => -x);
  return [{ R: R1, t }, { R: R1, t: nt }, { R: R2, t }, { R: R2, t: nt }];
}

// Linear triangulation from >= 2 views: poses [{R, t}], rays [[x, y]] (normalized). Returns [X, Y, Z] or null.
export function triangulate(poses, rays) {
  const n = poses.length;
  const A = new Float64Array(2 * n * 4);
  for (let k = 0; k < n; k++) {
    const { R, t } = poses[k], [x, y] = rays[k];
    const P = [R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2]];
    for (let c = 0; c < 4; c++) {
      A[8 * k + c] = x * P[8 + c] - P[c];
      A[8 * k + 4 + c] = y * P[8 + c] - P[4 + c];
    }
  }
  const h = nullVector(A, 2 * n, 4);
  if (Math.abs(h[3]) < 1e-12) return null;
  return [h[0] / h[3], h[1] / h[3], h[2] / h[3]];
}

export const depthIn = ({ R, t }, X) => R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];

// Angle (radians) at X between the rays from camera centers C1 and C2
export const triangulationAngle = (C1, C2, X) => angleBetween(sub(X, C1), sub(X, C2));

// Relative pose of camera 2 (camera 1 = [I | 0]) from E and normalized rays, by cheirality:
// returns { R, t, front: Uint8Array (both depths > 0), numFront }
export function poseFromEssential(E, rays1, rays2) {
  const P1 = { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] };
  let best = null;
  for (const P2 of decomposeEssential(E)) {
    const front = new Uint8Array(rays1.length);
    let c = 0;
    for (let i = 0; i < rays1.length; i++) {
      const X = triangulate([P1, P2], [rays1[i], rays2[i]]);
      if (X && X[2] > 0 && depthIn(P2, X) > 0) { front[i] = 1; c++; }
    }
    if (!best || c > best.numFront) best = { ...P2, front, numFront: c };
  }
  return best;
}

// Relative pose of frame 2 from matched pixels x1 <-> x2: RANSAC essential matrix on the
// undistorted normalized rays (8-point F there, threshold in pixels / f), cheirality. Returns
// { F (RANSAC result, normalized coordinates), idx (inlier indices), rays1, rays2 (inliers,
// normalized), R, t, front, numFront } or null.
export function relativePose(x1, x2, camera, { rng, threshold = 2, minInliers = 8 }) {
  const r1 = x1.map(p => normalizedFromImage(camera, ...p)), r2 = x2.map(p => normalizedFromImage(camera, ...p));
  const F = ransacFundamental(r1, r2, { rng, threshold: threshold / meanFocal(camera) });
  if (!F || F.numInliers < minInliers) return null;
  const idx = [...F.inliers.keys()].filter(k => F.inliers[k]);
  const rays1 = idx.map(k => r1[k]), rays2 = idx.map(k => r2[k]);
  return { F, idx, rays1, rays2, ...poseFromEssential(F.model, rays1, rays2) };
}

// Relative pose { R, t } of frame 2 from the matches of a pair alone (and the camera), or null when
// it is not trustworthy: F is degenerate when a homography explains the matches (planar ground
// seen from a drone, rotation only), and then two RANSAC runs disagree. At most maxMatches
// (evenly sampled) decide it. Arbiter of the model's and COLMAP's poses (quality.js, reference.js).
export function twoViewPose(x1, x2, camera, { rng, maxHomographyRatio = 0.7, maxSpreadDeg = 0.3, maxMatches = 600 }) {
  const stride = Math.ceil(x1.length / maxMatches);
  if (stride > 1) { x1 = x1.filter((_, m) => m % stride === 0); x2 = x2.filter((_, m) => m % stride === 0); }
  const rel = relativePose(x1, x2, camera, { rng, threshold: 1, minInliers: 50 });
  if (!rel) return null;
  const H = ransacHomography(x1, x2, { rng, threshold: 2, maxIterations: 300 });  // a gate, not an estimate
  if (H && H.numInliers > maxHomographyRatio * rel.F.numInliers) return null;
  const again = relativePose(x1, x2, camera, { rng, threshold: 1, minInliers: 50 });
  if (!again || deg(rotationAngle(rel.R, again.R)) > maxSpreadDeg) return null;
  return { R: rel.R, t: rel.t };
}

// ── P3P / PnP ────────────────────────────────────────────────────────────────

// Grunert's P3P: unit bearings b[3] (camera frame) and world points X[3] -> up to 4 poses {R, t}.
// With s2 = u s1, s3 = v s1, the law of cosines gives two quadratics in u with the same u^2
// coefficient; their difference is linear in u, and substituting back gives a quartic in v.
export function p3p(b, X) {
  const d = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
  const a2 = d(X[1], X[2]) ** 2, b2 = d(X[0], X[2]) ** 2, c2 = d(X[0], X[1]) ** 2;
  if (a2 < 1e-24 || b2 < 1e-24 || c2 < 1e-24) return [];
  const ca = dot(b[1], b[2]), cb = dot(b[0], b[2]), cg = dot(b[0], b[1]);
  const K = [1, -2 * cb, 1];                                         // 1 + v^2 - 2 v cb
  const N = polyAdd(polyScale([-1, 0, 1], b2), polyScale(K, a2 - c2)); // u = N / D
  const D = [-2 * b2 * ca, 2 * b2 * cg];
  // b2 (u^2 - 2 u cg + 1) - c2 K = 0, multiplied by D^2
  const quartic = polyAdd(polyAdd(polyScale(polyMul(N, N), b2), polyScale(polyMul(N, D), -2 * b2 * cg)),
                          polyMul(polyAdd(polyScale([b2], 1), polyScale(K, -c2)), polyMul(D, D)));
  const poses = [];
  for (const v of realRoots(quartic)) {
    if (!(v > 0)) continue;
    const den = polyEval(D, v);
    if (Math.abs(den) < 1e-14) continue;
    const u = polyEval(N, v) / den;
    if (!(u > 0)) continue;
    const q = 1 + u * u - 2 * u * cg;
    if (!(q > 0)) continue;
    const s1 = Math.sqrt(c2 / q), s = [s1, u * s1, v * s1];
    const P = b.map((bi, k) => bi.map(x => x * s[k]));
    const { R, t } = umeyama(X, P, { withScale: false });  // camera = R X + t
    poses.push({ R, t });
  }
  return poses;
}

const bearing = ([x, y]) => { const n = Math.hypot(x, y, 1); return [x / n, y / n, 1 / n]; };

// RANSAC PnP: rays (normalized), pixels [u, v], world points, camera (for pixel errors).
// Returns { pose, inliers: Uint8Array, numInliers } or null.
export function ransacPnP(rays, pixels, points, camera, { rng, threshold = 4, ...opts }) {
  const b = rays.map(bearing);
  return ransac({
    n: rays.length, sampleSize: 3, rng, threshold, ...opts,
    fit: (idx) => idx.length === 3 ? p3p(idx.map(i => b[i]), idx.map(i => points[i]))
                                   : [pnpLinear(idx.map(i => rays[i]), idx.map(i => points[i]))],
    error: (pose, i) => {
      const [u, v, z] = project(camera, pose, points[i]);
      return z > 0 ? (u - pixels[i][0]) ** 2 + (v - pixels[i][1]) ** 2 : Infinity;
    },
  });
}

// Linear (DLT) pose from >= 6 correspondences, rotation projected onto SO(3); used for RANSAC refits
export function pnpLinear(rays, points) {
  const n = rays.length;
  if (n < 6) return null;
  const A = new Float64Array(2 * n * 12);
  for (let i = 0; i < n; i++) {
    const [x, y] = rays[i], [X, Y, Z] = points[i];
    A.set([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x], 24 * i);
    A.set([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y], 24 * i + 12);
  }
  const p = nullVector(A, 2 * n, 12);
  const M = [p[0], p[1], p[2], p[4], p[5], p[6], p[8], p[9], p[10]];
  const { U, S, V } = svd(Float64Array.from(M), 3, 3);
  let R = Array.from(matMul(U, transpose(V, 3, 3), 3, 3, 3));
  let scale = (S[0] + S[1] + S[2]) / 3;
  if (det3(R) < 0) { R = R.map(x => -x); scale = -scale; }
  let t = [p[3] / scale, p[7] / scale, p[11] / scale];
  // the det fix above resolved the sign of p (-p flips det R): mostly behind means a bad sample
  let front = 0;
  for (const X of points) front += (mulVec3(R, X)[2] + t[2]) > 0;
  if (front < n / 2) return null;
  return { R, t };
}

