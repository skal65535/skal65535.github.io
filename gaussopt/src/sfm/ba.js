// ba.js — Bundle adjustment: robust Levenberg-Marquardt with a Schur complement on the points
// (DESIGN-sfm D2, JS f64 reference backend). Sized for windows (tens of cameras, ~1e5 points):
// the reduced camera system is dense and solved by Cholesky.
//
// Parameters: free camera poses (camera-frame update q' = exp(dw) q + dt: 6 each), the intrinsics
// of each camera model shared by its poses (any of 'f', 'k1', 'k2'; fy keeps its ratio to fx,
// principal point and tangential terms stay fixed; a model per lens and zoom, DESIGN-sfm §13 P8),
// and all observed points (3 each). Fixed poses still constrain the points.
"use strict";

import { expSO3, mul3, mulVec3, choleskySolve } from './linalg.js';
import { intrinsics, distortionParams, paramIndex, withFocal } from './camera.js';

const INTRINSICS = ['f', 'k1', 'k2'];

// Projection of camera-space point q with Jacobians: out.uv, out.dq (2 x 3, d(uv)/dq) and
// out.dk (2 x 3, d(uv)/d[f, k1, k2], with fy = f * fy/fx)
function projectCameraPoint(cam, q, out) {
  const [fx, fy, cx, cy] = intrinsics(cam), [k1, k2, p1, p2] = distortionParams(cam);
  const iz = 1 / q[2], x = q[0] * iz, y = q[1] * iz;
  const r2 = x * x + y * y;
  const rad = 1 + k1 * r2 + k2 * r2 * r2;
  const xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
  const yd = y * rad + 2 * p2 * x * y + p1 * (r2 + 2 * y * y);
  out.uv[0] = fx * xd + cx;
  out.uv[1] = fy * yd + cy;
  const dr = 2 * k1 + 4 * k2 * r2;  // 2 d(rad)/d(r2)
  const dxd_dx = rad + x * dr * x + 2 * p1 * y + 6 * p2 * x, dxd_dy = x * dr * y + 2 * p1 * x + 2 * p2 * y;
  const dyd_dx = y * dr * x + 2 * p2 * y + 2 * p1 * x,      dyd_dy = rad + y * dr * y + 2 * p2 * x + 6 * p1 * y;
  // d(x, y)/dq = [iz 0 -x iz; 0 iz -y iz]
  const a00 = fx * dxd_dx, a01 = fx * dxd_dy, a10 = fy * dyd_dx, a11 = fy * dyd_dy;
  const dq = out.dq;
  dq[0] = a00 * iz; dq[1] = a01 * iz; dq[2] = -(a00 * x + a01 * y) * iz;
  dq[3] = a10 * iz; dq[4] = a11 * iz; dq[5] = -(a10 * x + a11 * y) * iz;
  const dk = out.dk;
  dk[0] = xd;               dk[3] = (fy / fx) * yd;
  dk[1] = fx * x * r2;      dk[4] = fy * y * r2;
  dk[2] = fx * x * r2 * r2; dk[5] = fy * y * r2 * r2;
}

const scratch = () => ({ uv: new Float64Array(2), dq: new Float64Array(6), dk: new Float64Array(6) });

// One observation: residual into r[ro..ro+1], pose Jacobian (2 x 6, row-major) into Jc[co..],
// point Jacobian (2 x 3) into Jp[po..]; tmp.dk holds the intrinsics Jacobian. Returns false when
// the point is behind the camera. The update q' = exp(dw) q + dt gives d(uv)/d(dw) = -dq [q]x
// (the row vector d times -[q]x is q x d) and d(uv)/d(dt) = dq; d(uv)/dX = dq R.
function linearize(cam, pose, X, u, v, tmp, r, ro, Jc, co, Jp, po) {
  const RX = mulVec3(pose.R, X), q = [RX[0] + pose.t[0], RX[1] + pose.t[1], RX[2] + pose.t[2]];
  if (!(q[2] > 1e-9)) return false;
  projectCameraPoint(cam, q, tmp);
  r[ro] = tmp.uv[0] - u; r[ro + 1] = tmp.uv[1] - v;
  const dq = tmp.dq, R = pose.R;
  for (let row = 0; row < 2; row++) {
    const d0 = dq[3 * row], d1 = dq[3 * row + 1], d2 = dq[3 * row + 2], c = co + 6 * row;
    Jc[c] = q[1] * d2 - q[2] * d1; Jc[c + 1] = q[2] * d0 - q[0] * d2; Jc[c + 2] = q[0] * d1 - q[1] * d0;
    Jc[c + 3] = d0; Jc[c + 4] = d1; Jc[c + 5] = d2;
    for (let k = 0; k < 3; k++) Jp[po + 3 * row + k] = d0 * R[k] + d1 * R[3 + k] + d2 * R[6 + k];
  }
  return true;
}

// Residual and Jacobians of one observation as bundleAdjust sees them, for tests:
// { r: [2], Jpose: 2x6 (dw, dt), Jpoint: 2x3, Jintr: 2x3 (f, k1, k2) }
export function linearizeObservation(camera, pose, X, uv) {
  const tmp = scratch(), r = [0, 0], Jpose = new Array(12), Jpoint = new Array(6);
  linearize(camera, pose, X, uv[0], uv[1], tmp, r, 0, Jpose, 0, Jpoint, 0);
  return { r, Jpose, Jpoint, Jintr: Array.from(tmp.dk) };
}

// Huber loss: [rho(s), rho'(s)] for squared residual s and scale d
const rho = (s, d) => (s <= d * d ? [s, 1] : [2 * d * Math.sqrt(s) - d * d, d / Math.sqrt(s)]);

// problem: {
//   camera: { model, width, height, params },  one camera model for every pose, or
//   cameras: [camera] and poseCam: Int32Array (the model of each pose); refined ones are replaced
//   poses: [{ R, t }] (free ones replaced), free: Uint8Array per pose
//   points: Float64Array(3 n) (updated in place),
//   obsCam: Int32Array, obsPoint: Int32Array, obsUV: Float64Array(2 m)
// }
// opts: { refine: ['f', 'k1'], refineCameras: [bool] per camera model (default: all of them),
// lossScale: 2 (px, of the Huber loss), maxIterations: 30, fixPoints: false }; stops when an accepted step lowers the cost by
// less than 1e-7 (relative)
// Returns { initialCost, cost, iterations, rmse (px, of the non-robust residuals) }.
export function bundleAdjust(problem, opts = {}) {
  const { refine = [], refineCameras = null, lossScale = 2, maxIterations = 30, fixPoints = false } = opts;
  const { poses, free, points, obsCam, obsPoint, obsUV } = problem;
  const single = !problem.cameras, poseCam = problem.poseCam ?? new Int32Array(poses.length);
  let cams = single ? [problem.camera] : [...problem.cameras];
  const model = cams[0].model;
  if (cams.some(c => c.model !== model)) throw new Error('bundleAdjust: the camera models must be of one type');
  const slot = (name) => (name === 'f' ? Math.max(paramIndex(model, 'f'), paramIndex(model, 'fx')) : paramIndex(model, name));
  const kIdx = INTRINSICS.map((name, k) => [name, k]).filter(([name]) => refine.includes(name) && slot(name) >= 0).map(([, k]) => k);
  const ni = kIdx.length;
  const numObs = obsCam.length, numPoints = points.length / 3;
  const camBlock = new Int32Array(poses.length).fill(-1);
  let nc = 0;
  for (let c = 0; c < poses.length; c++) if (free[c]) camBlock[c] = nc++;
  // the intrinsics blocks: one per refined camera model that some pose of the problem uses
  const used = new Set(poseCam), intrBlock = new Int32Array(cams.length).fill(-1);
  let nm = 0;
  if (ni) cams.forEach((_, k) => { if (used.has(k) && (!refineCameras || refineCameras[k])) intrBlock[k] = nm++; });
  const m = 6 * nc + ni * nm, iOff = 6 * nc;  // reduced system: free poses, then intrinsics per model
  const camOfObs = (o) => cams[poseCam[obsCam[o]]], intrOfObs = (o) => intrBlock[poseCam[obsCam[o]]];

  // Observations of each point
  const ptStart = new Int32Array(numPoints + 1);
  for (let o = 0; o < numObs; o++) ptStart[obsPoint[o] + 1]++;
  for (let p = 0; p < numPoints; p++) ptStart[p + 1] += ptStart[p];
  const ptObs = new Int32Array(numObs);
  { const fill = ptStart.slice(0, numPoints); for (let o = 0; o < numObs; o++) ptObs[fill[obsPoint[o]]++] = o; }

  // Linearization: per observation r (2), Jc (2 x 6), Jp (2 x 3), Ji (2 x ni), weight
  const res = new Float64Array(2 * numObs), wgt = new Float64Array(numObs);
  const Jc = new Float64Array(12 * numObs), Jp = new Float64Array(6 * numObs), Ji = new Float64Array(2 * ni * numObs);
  const tmp = scratch();
  const pointOf = (X, p) => [X[3 * p], X[3 * p + 1], X[3 * p + 2]];

  const evalCost = (C, P, X) => {
    let cost = 0, sq = 0, n = 0;
    const r = [0, 0], junk = new Float64Array(18);  // Jacobians are not needed here
    for (let o = 0; o < numObs; o++) {
      if (!linearize(C[poseCam[obsCam[o]]], P[obsCam[o]], pointOf(X, obsPoint[o]), obsUV[2 * o], obsUV[2 * o + 1], tmp, r, 0, junk, 0, junk, 12)) {
        cost += rho(1e6, lossScale)[0];  // behind the camera: large constant penalty
        continue;
      }
      const s = r[0] * r[0] + r[1] * r[1];
      cost += rho(s, lossScale)[0]; sq += s; n++;
    }
    return { cost, rmse: Math.sqrt(sq / Math.max(1, n)) };
  };

  const linearizeAll = () => {
    for (let o = 0; o < numObs; o++) {
      if (!linearize(camOfObs(o), poses[obsCam[o]], pointOf(points, obsPoint[o]), obsUV[2 * o], obsUV[2 * o + 1],
                     tmp, res, 2 * o, Jc, 12 * o, Jp, 6 * o)) { wgt[o] = 0; continue; }
      wgt[o] = rho(res[2 * o] ** 2 + res[2 * o + 1] ** 2, lossScale)[1];
      for (let k = 0; k < ni; k++) { Ji[2 * ni * o + k] = tmp.dk[kIdx[k]]; Ji[2 * ni * o + ni + k] = tmp.dk[3 + kIdx[k]]; }
    }
  };

  // The non-point parameter blocks of observation o: [offset in the reduced system, size, J, J offset]
  const blocksOf = (o) => {
    const out = [];
    if (camBlock[obsCam[o]] >= 0) out.push([6 * camBlock[obsCam[o]], 6, Jc, 12 * o]);
    if (ni && intrOfObs(o) >= 0) out.push([iOff + ni * intrOfObs(o), ni, Ji, 2 * ni * o]);
    return out;
  };

  // Damped normal equations reduced on the poses + intrinsics: { dCam (m), dPts (3 n) } or null
  const solve = (lambda) => {
    const S = new Float64Array(m * m), g = new Float64Array(m), dPts = new Float64Array(3 * numPoints);
    for (let o = 0; o < numObs; o++) {  // J^T w J and J^T w r of the pose / intrinsics blocks
      const w = wgt[o];
      if (!w) continue;
      const blocks = blocksOf(o);
      for (const [oa, na, A, ao] of blocks) {
        for (const [ob, nb, B, bo] of blocks) {
          for (let i = 0; i < na; i++) for (let j = 0; j < nb; j++)
            S[(oa + i) * m + ob + j] += w * (A[ao + i] * B[bo + j] + A[ao + na + i] * B[bo + nb + j]);
        }
        for (let i = 0; i < na; i++) g[oa + i] += w * (A[ao + i] * res[2 * o] + A[ao + na + i] * res[2 * o + 1]);
      }
    }
    for (let i = 0; i < m; i++) S[i * m + i] += lambda * Math.max(S[i * m + i], 1e-6);
    // Schur complement over the points: S -= W V^-1 W^T, g -= W V^-1 gp
    const perPoint = new Array(numPoints);
    const V = new Float64Array(9), gp = new Float64Array(3), Vinv = new Float64Array(9);
    for (let p = 0; p < (fixPoints ? 0 : numPoints); p++) {
      V.fill(0); gp.fill(0);
      const W = new Map();  // reduced-system offset -> { size, M (size x 3) }
      for (let k = ptStart[p]; k < ptStart[p + 1]; k++) {
        const o = ptObs[k], w = wgt[o];
        if (!w) continue;
        const J = Jp.subarray(6 * o, 6 * o + 6);
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) V[3 * i + j] += w * (J[i] * J[j] + J[3 + i] * J[3 + j]);
          gp[i] += w * (J[i] * res[2 * o] + J[3 + i] * res[2 * o + 1]);
        }
        for (const [off, size, A, ao] of blocksOf(o)) {
          let e = W.get(off);
          if (!e) W.set(off, e = { size, M: new Float64Array(3 * size) });
          for (let i = 0; i < size; i++) for (let j = 0; j < 3; j++)
            e.M[3 * i + j] += w * (A[ao + i] * J[j] + A[ao + size + i] * J[3 + j]);
        }
      }
      for (let i = 0; i < 3; i++) V[4 * i] += lambda * Math.max(V[4 * i], 1e-9);
      if (!invert3(V, Vinv)) continue;
      const list = [...W.entries()];
      perPoint[p] = { list, Vinv: Float64Array.from(Vinv), gp: Float64Array.from(gp) };
      const WV = list.map(([, { size, M }]) => {
        const r = new Float64Array(3 * size);
        for (let i = 0; i < size; i++) for (let j = 0; j < 3; j++)
          r[3 * i + j] = M[3 * i] * Vinv[j] + M[3 * i + 1] * Vinv[3 + j] + M[3 * i + 2] * Vinv[6 + j];
        return r;
      });
      list.forEach(([oa, { size: na }], a) => {
        const A = WV[a];
        for (const [ob, { size: nb, M: B }] of list) {
          for (let i = 0; i < na; i++) for (let j = 0; j < nb; j++)
            S[(oa + i) * m + ob + j] -= A[3 * i] * B[3 * j] + A[3 * i + 1] * B[3 * j + 1] + A[3 * i + 2] * B[3 * j + 2];
        }
        for (let i = 0; i < na; i++) g[oa + i] -= A[3 * i] * gp[0] + A[3 * i + 1] * gp[1] + A[3 * i + 2] * gp[2];
      });
    }
    const dCam = m ? choleskySolve(S, g.map(x => -x), m) : new Float64Array(0);
    if (!dCam) return null;
    perPoint.forEach((d, p) => {  // back-substitution: dX = -V^-1 (gp + W^T dCam)
      const rhs = Float64Array.from(d.gp);
      for (const [off, { size, M }] of d.list)
        for (let i = 0; i < size; i++) for (let j = 0; j < 3; j++) rhs[j] += M[3 * i + j] * dCam[off + i];
      for (let j = 0; j < 3; j++) dPts[3 * p + j] = -(d.Vinv[3 * j] * rhs[0] + d.Vinv[3 * j + 1] * rhs[1] + d.Vinv[3 * j + 2] * rhs[2]);
    });
    return { dCam, dPts };
  };

  const apply = ({ dCam, dPts }) => {
    const P = poses.map((pose, c) => {
      if (camBlock[c] < 0) return pose;
      const d = dCam.subarray(6 * camBlock[c], 6 * camBlock[c] + 6);
      const dR = expSO3([d[0], d[1], d[2]]), t = mulVec3(dR, pose.t);
      return { R: mul3(dR, pose.R), t: [t[0] + d[3], t[1] + d[4], t[2] + d[5]] };
    });
    const X = Float64Array.from(points, (x, i) => x + dPts[i]);
    const C = cams.map((cam, c) => {
      if (intrBlock[c] < 0) return cam;
      kIdx.forEach((k, j) => {
        const step = dCam[iOff + ni * intrBlock[c] + j];
        if (k === 0) cam = withFocal(cam, cam.params[0] + step);
        else { const params = [...cam.params]; params[slot(INTRINSICS[k])] += step; cam = { ...cam, params }; }
      });
      return cam;
    });
    return { P, X, C };
  };

  let { cost, rmse } = evalCost(cams, poses, points);
  const initialCost = cost;
  let lambda = 1e-4, it = 0;
  for (; it < maxIterations; it++) {
    linearizeAll();
    let accepted = false, converged = false;
    for (let tries = 0; tries < 10 && !accepted; tries++) {
      const step = solve(lambda);
      if (!step) { lambda *= 10; continue; }
      const cand = apply(step);
      const e = evalCost(cand.C, cand.P, cand.X);
      if (e.cost >= cost) { lambda *= 5; continue; }
      converged = (cost - e.cost) / Math.max(cost, 1e-300) < 1e-7;
      cand.P.forEach((p, c) => { poses[c] = p; });
      points.set(cand.X);
      cams = cand.C;
      if (single) problem.camera = cams[0];
      else problem.cameras = cams;
      cost = e.cost; rmse = e.rmse;
      lambda = Math.max(1e-10, lambda / 3);
      accepted = true;
    }
    if (!accepted || converged) { it += accepted ? 1 : 0; break; }  // no decrease even with heavy damping, or converged
  }
  return { initialCost, cost, iterations: it, rmse };
}

function invert3(A, out) {
  const [a, b, c, d, e, f, g, h, i] = A;
  const A0 = e * i - f * h, A1 = -(d * i - f * g), A2 = d * h - e * g;
  const det = a * A0 + b * A1 + c * A2;
  if (!(Math.abs(det) > 1e-300)) return false;
  const s = 1 / det;
  out[0] = A0 * s; out[1] = -(b * i - c * h) * s; out[2] = (b * f - c * e) * s;
  out[3] = A1 * s; out[4] = (a * i - c * g) * s; out[5] = -(a * f - c * d) * s;
  out[6] = A2 * s; out[7] = -(a * h - b * g) * s; out[8] = (a * e - b * d) * s;
  return true;
}

// Reprojection error (px) of every observation of a problem, without optimizing
export function reprojectionErrors(problem) {
  const { poses, points, obsCam, obsPoint, obsUV } = problem;
  const cams = problem.cameras ?? [problem.camera], poseCam = problem.poseCam ?? new Int32Array(poses.length);
  const out = new Float64Array(obsCam.length), tmp = scratch(), r = [0, 0], junk = new Float64Array(18);
  for (let o = 0; o < obsCam.length; o++) {
    const p = obsPoint[o];
    out[o] = linearize(cams[poseCam[obsCam[o]]], poses[obsCam[o]], [points[3 * p], points[3 * p + 1], points[3 * p + 2]],
                       obsUV[2 * o], obsUV[2 * o + 1], tmp, r, 0, junk, 0, junk, 12) ? Math.hypot(r[0], r[1]) : Infinity;
  }
  return out;
}
