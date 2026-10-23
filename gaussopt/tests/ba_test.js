import { bundleAdjust, linearizeObservation, reprojectionErrors } from '../src/sfm/ba.js';
import { makeCamera, poseFromCenter, poseCenter } from '../src/sfm/camera.js';
import { expSO3, mul3, mulVec3, transpose3, rotationAngle } from '../src/sfm/linalg.js';
import { makeSyntheticScene, gaussian } from '../src/sfm/synth.js';
import { assert, assertClose, rng } from './util.js';

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
    ];
    intr.forEach((mk, k) => {
      const d = fd(e => res({ ...CAM, params: mk(e) }, pose, X));
      assertClose([lin.Jintr[k], lin.Jintr[3 + k]], d, 1e-4 * (1 + Math.abs(d[0]) + Math.abs(d[1])), `intrinsic ${k}`);
    });
  }
});

// BA problem from a synthetic scene, with perturbed poses / points / focal
function problemFrom(scene, r, { poseNoise = 0.01, pointNoise = 0.05, focalScale = 1.03, outliers = 0 } = {}) {
  const { truth, input } = scene;
  const used = new Map();
  const obsCam = [], obsPoint = [], obsUV = [];
  truth.images.forEach((im, c) => {
    im.pointIds.forEach((pid, k) => {
      if (!used.has(pid)) used.set(pid, used.size);
      obsCam.push(c); obsPoint.push(used.get(pid));
      const out = r() < outliers;
      obsUV.push(out ? 1920 * r() : input.images[c].keypoints[2 * k], out ? 1080 * r() : input.images[c].keypoints[2 * k + 1]);
    });
  });
  const points = new Float64Array(3 * used.size);
  for (const [pid, i] of used) truth.points[pid].forEach((x, k) => { points[3 * i + k] = x + pointNoise * gaussian(r); });
  const poses = truth.images.map(im => ({ R: [...im.R], t: [...im.t] }));
  // rotate and shift the free poses
  for (let c = 2; c < poses.length; c++) {
    const dR = expSO3([0, 1, 2].map(() => poseNoise * gaussian(r)));
    poses[c] = poseFromCenter(mul3(dR, truth.images[c].R), poseCenter(truth.images[c]).map(x => x + 10 * poseNoise * gaussian(r)));
  }
  const camera = { ...truth.camera, params: [...truth.camera.params] };
  camera.params[0] *= focalScale; camera.params[1] *= focalScale;
  const free = Uint8Array.from(poses, (_, c) => (c < 2 ? 0 : 1));  // two fixed poses fix the gauge
  return { camera, poses, free, points, obsCam: Int32Array.from(obsCam), obsPoint: Int32Array.from(obsPoint), obsUV: Float64Array.from(obsUV) };
}

const SCENE = makeSyntheticScene({ numFrames: 15, numGround: 3000, numBoxes: 4, numFar: 300, maxFeatures: 1500 });

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
  bundleAdjust(prob, { refine: ['f'], loss: 'huber', lossScale: 2, maxIterations: 50 });
  const errs = Array.from(reprojectionErrors(prob)).sort((a, b) => a - b);
  assert(errs[Math.floor(0.5 * errs.length)] < 1, `median reprojection ${errs[errs.length >> 1]}`);
  assertClose(prob.camera.params[0], 1230, 1230 * 5e-3, 'focal');
});
