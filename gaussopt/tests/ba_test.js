import { bundleAdjust, linearizeObservation, reprojectionErrors } from '../src/sfm/ba.js';
import { makeCamera, poseFromCenter, poseCenter } from '../src/sfm/camera.js';
import { expSO3, mul3, mulVec3, transpose3, rotationAngle } from '../src/sfm/linalg.js';
import { makeSyntheticScene, gaussian } from '../src/sfm/synth.js';
import { assert, assertClose, rng, problemFrom, fyOff } from './util.js';

const CAM = makeCamera('OPENCV', 1920, 1080, [1230, 1240, 960, 540, -0.05, 0.01, 0.001, -0.0005]);

Deno.test('BA Jacobians match central finite differences', () => {
  const r = rng(30);
  const h = 1e-6;
  for (let trial = 0; trial < 50; trial++) {
    const pose = poseFromCenter(expSO3([0.3 * gaussian(r), 0.3 * gaussian(r), 0.3 * gaussian(r)]), [gaussian(r), gaussian(r), -5]);
    const C = poseCenter(pose);
    const X = mulVec3(transpose3(pose.R), [gaussian(r), gaussian(r) * 0.5, 4 + 4 * r()]).map((x, i) => x + C[i]);
    const lin = linearizeObservation(CAM, pose, X, [0, 0]);
    const res = (cam, P, Y) => linearizeObservation(cam, P, Y, [0, 0]).r;
    const fd = (f) => { const a = f(h), b = f(-h); return [(a[0] - b[0]) / (2 * h), (a[1] - b[1]) / (2 * h)]; };
    for (let k = 0; k < 6; k++) {  // pose: q' = exp(dw) q + dt
      const d = fd(e => {
        const dv = [0, 0, 0, 0, 0, 0]; dv[k] = e;
        const dR = expSO3(dv.slice(0, 3));
        const R = mul3(dR, pose.R);
        const t = mulVec3(dR, pose.t).map((x, i) => x + dv[3 + i]);
        return res(CAM, { R, t }, X);
      });
      assertClose([lin.Jpose[k], lin.Jpose[6 + k]], d, 1e-4 * (1 + Math.abs(d[0]) + Math.abs(d[1])), `pose ${k}`);
    }
    for (let k = 0; k < 3; k++) {
      const d = fd(e => res(CAM, pose, X.map((x, i) => x + (i === k ? e : 0))));
      assertClose([lin.Jpoint[k], lin.Jpoint[3 + k]], d, 1e-4 * (1 + Math.abs(d[0]) + Math.abs(d[1])), `point ${k}`);
    }
    const intr = [
      (e) => { const p = [...CAM.params]; p[0] += e; p[1] = p[0] * CAM.params[1] / CAM.params[0]; return p; },
      (e) => { const p = [...CAM.params]; p[4] += e; return p; },
      (e) => { const p = [...CAM.params]; p[5] += e; return p; },
      (e) => { const p = [...CAM.params]; p[6] += e; return p; },
      (e) => { const p = [...CAM.params]; p[7] += e; return p; },
      (e) => { const p = [...CAM.params]; p[1] += e; return p; },  // 'fy'
    ];
    intr.forEach((mk, k) => {
      const d = fd(e => res({ ...CAM, params: mk(e) }, pose, X));
      assertClose([lin.Jintr[k], lin.Jintr[intr.length + k]], d, 1e-4 * (1 + Math.abs(d[0]) + Math.abs(d[1])), `intrinsic ${k}`);
    });
  }
});


const SCENE = makeSyntheticScene({ numFrames: 15, numGround: 3000, numBoxes: 4, numFar: 300, maxFeatures: 1500 });

// a lens of gare2's tangential distortion (COLMAP's p2 there: -0.020; without it, its own model refit to 2.9 px from 1.16)
const TANGENTIAL = makeSyntheticScene({ numFrames: 15, numGround: 3000, numBoxes: 4, numFar: 300, maxFeatures: 1500,
                                               params: [1230, 1230, 960, 540, -0.004, 0.002, 0.003, -0.02] });

// p1, p2 refined from 0: found, the residuals at the noise floor (0.5 px); without them the fit stays off
Deno.test('BA refines the tangential terms p1, p2', () => {
  const run = (refine) => {
    const prob = problemFrom(TANGENTIAL, rng(31));
    prob.camera.params[6] = prob.camera.params[7] = 0;
    bundleAdjust(prob, { refine, maxIterations: 50 });
    const e = Array.from(reprojectionErrors(prob)).filter(Number.isFinite);
    return { cam: prob.camera.params, rmse: Math.sqrt(e.reduce((a, x) => a + x * x, 0) / e.length) };
  };
  const a = run(['f', 'k1', 'k2', 'p1', 'p2']), b = run(['f', 'k1', 'k2']);
  assert(Math.abs(a.cam[7] + 0.02) < 1e-3 && Math.abs(a.cam[6] - 0.003) < 1e-3 && a.rmse < 0.8, `p1 ${a.cam[6]}, p2 ${a.cam[7]}, RMSE ${a.rmse}`);
  assert(b.rmse > 2 * a.rmse, `without p1, p2: RMSE ${b.rmse} (with ${a.rmse})`);
});

Deno.test('BA converges from perturbed poses, points and focal to the noise floor', () => {
  const r = rng(31);
  const prob = problemFrom(SCENE, r);
  const out = bundleAdjust(prob, { refine: ['f', 'k1'], maxIterations: 50 });
  assert(out.cost < out.initialCost * 1e-2, `cost ${out.initialCost} -> ${out.cost}`);
  assert(out.rmse < 0.8, `rmse ${out.rmse} px (noise 0.5 px per axis)`);
  assertClose(prob.camera.params[0], 1230, 1230 * 2e-3, 'focal');
  for (let c = 2; c < prob.poses.length; c++) {
    assert(rotationAngle(prob.poses[c].R, SCENE.truth.images[c].R) < 1e-3, `rotation ${c}`);
    const e = Math.hypot(...poseCenter(prob.poses[c]).map((x, i) => x - poseCenter(SCENE.truth.images[c])[i]));
    assert(e < 0.02, `center ${c}: ${e}`);
  }
});

Deno.test('BA with a Huber loss is robust to 5% outlier observations', () => {
  const r = rng(32);
  const prob = problemFrom(SCENE, r, { outliers: 0.05 });
  bundleAdjust(prob, { refine: ['f'], lossScale: 2, maxIterations: 50 });
  const errs = Array.from(reprojectionErrors(prob)).sort((a, b) => a - b);
  assert(errs[Math.floor(0.5 * errs.length)] < 1, `median reprojection ${errs[errs.length >> 1]}`);
  assertClose(prob.camera.params[0], 1230, 1230 * 5e-3, 'focal');
});


// COLMAP's OPENCV model: fx and fy apart. fy 1% off fx's ratio: 'fy' finds it, 'f' alone (fy at fx's ratio) cannot
Deno.test('BA refines fy apart from fx', () => {
  const run = (refine) => { const prob = fyOff(rng(31)); bundleAdjust(prob, { refine, maxIterations: 50 }); return prob.camera.params; };
  const a = run(['f', 'k1', 'fy']), b = run(['f', 'k1']);
  assert(Math.abs(a[0] / 1230 - 1) < 1e-3 && Math.abs(a[1] / 1230 - 1) < 1e-3, `with fy: fx ${a[0]}, fy ${a[1]} (truth 1230)`);
  assert(Math.abs(b[1] / b[0] - 1) > 5e-3, `without fy: fx ${b[0]}, fy ${b[1]}`);
});

// 5% outliers: least squares (trivial) pulled off by them, soft L1 at 1 px not; an unknown loss refused
Deno.test('BA with the soft L1 loss is robust to 5% outliers, least squares is not', () => {
  const run = (loss) => {
    const prob = problemFrom(SCENE, rng(32), { outliers: 0.05 });
    bundleAdjust(prob, { refine: ['f'], loss, lossScale: 1, maxIterations: 50 });
    const errs = Array.from(reprojectionErrors(prob)).sort((a, b) => a - b);
    return { median: errs[errs.length >> 1], f: prob.camera.params[0] };
  };
  const soft = run('softl1'), ls = run('trivial');
  assert(soft.median < 1 && Math.abs(soft.f / 1230 - 1) < 5e-3, `soft L1: median ${soft.median} px, f ${soft.f}`);
  assert(ls.median > 2 * soft.median, `least squares: median ${ls.median} px, f ${ls.f}`);
  let threw = false;
  try { bundleAdjust(problemFrom(SCENE, rng(32)), { loss: 'cauchy' }); } catch { threw = true; }
  assert(threw, 'an unknown loss taken');
});

// fixedPoints: the held points' positions kept to the bit, the others adjusted with the poses
Deno.test('BA holds the fixedPoints', () => {
  const prob = problemFrom(SCENE, rng(33)), n = prob.points.length / 3, before = prob.points.slice();
  const fixedPoints = Uint8Array.from({ length: n }, (_, p) => p % 2);
  const out = bundleAdjust(prob, { refine: [], fixedPoints, maxIterations: 20 });
  let held = 0, moved = 0;
  for (let p = 0; p < n; p++) {
    const same = [0, 1, 2].every(k => prob.points[3 * p + k] === before[3 * p + k]);
    if (fixedPoints[p]) held += same ? 1 : 0; else moved += same ? 0 : 1;
  }
  assert(held === n >> 1 && moved > 0.9 * (n - (n >> 1)) && out.cost < 0.5 * out.initialCost, `${held} held, ${moved} moved, cost ${out.initialCost} -> ${out.cost}`);
});

Deno.test('BA with its reduced system by conjugate gradients (denseMax 0) lands where the dense Cholesky does', () => {
  const a = problemFrom(SCENE, rng(31)), b = problemFrom(SCENE, rng(31));
  const oa = bundleAdjust(a, { refine: ['f', 'k1'], maxIterations: 50 });
  const ob = bundleAdjust(b, { refine: ['f', 'k1'], maxIterations: 50, denseMax: 0 });
  assert(ob.profile.cg > 0 && !oa.profile.cg, `${ob.profile.cg} conjugate gradient iterations`);
  let rot = 0, cen = 0;
  a.poses.forEach((p, k) => {
    rot = Math.max(rot, rotationAngle(p.R, b.poses[k].R) * 180 / Math.PI);
    cen = Math.max(cen, Math.hypot(...poseCenter(p).map((x, i) => x - poseCenter(b.poses[k])[i])));
  });
  assert(rot < 1e-5 && cen < 1e-5, `poses apart by ${rot} deg, ${cen} in centre`);
  assertClose(ob.cost, oa.cost, 1e-7 * oa.cost, 'cost');
  assertClose(b.camera.params[0], a.camera.params[0], 1e-4, 'focal');
});
