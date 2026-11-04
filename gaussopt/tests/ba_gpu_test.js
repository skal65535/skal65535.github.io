// ba_gpu_test.js — The GPU bundle adjustment (src/sfm/ba_gpu.js) against ba.js from the same start, and its conditioning.
import { bundleAdjust, reprojectionErrors, linearizeObservation } from '../src/sfm/ba.js';
import { GpuBundleAdjuster } from '../src/sfm/ba_gpu.js';
import { poseCenter } from '../src/sfm/camera.js';
import { rotationAngle, choleskyFactor, invert3 } from '../src/sfm/linalg.js';
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { assert, assertClose, rng, problemFrom, hasGPU } from './util.js';

const SCENE = makeSyntheticScene({ numFrames: 15, numGround: 3000, numBoxes: 4, numFar: 300, maxFeatures: 1500 });
const gpu = hasGPU ? await GpuBundleAdjuster.create() : null;
const test = (name, fn) => Deno.test({ name, ignore: !hasGPU, fn });

// the largest pose differences of two solved copies: [rotation (deg), centre (scene units)]
function apart(a, b) {
  let rot = 0, cen = 0;
  a.poses.forEach((p, k) => {
    rot = Math.max(rot, rotationAngle(p.R, b.poses[k].R) * 180 / Math.PI);
    cen = Math.max(cen, Math.hypot(...poseCenter(p).map((x, i) => x - poseCenter(b.poses[k])[i])));
  });
  return [rot, cen];
}

for (const [name, refine] of [['poses, points, f and k1', ['f', 'k1']], ['poses and points, the intrinsics fixed', []]]) {
  test(`BA on the GPU lands where f64 does: ${name}`, async () => {
    const a = problemFrom(SCENE, rng(31)), b = problemFrom(SCENE, rng(31));
    const oa = bundleAdjust(a, { refine, maxIterations: 50 }), ob = await gpu.bundleAdjust(b, { refine, maxIterations: 50 });
    const [rot, cen] = apart(a, b);
    assert(rot < 1e-4 && cen < 1e-4, `poses apart by ${rot} deg, ${cen} in centre`);
    assertClose(ob.cost, oa.cost, 1e-5 * oa.cost, 'cost');
    assertClose(b.camera.params[0], a.camera.params[0], 0.01, 'focal');
  });
}

// the problem with n lenses: pose k on lens k % n, each starting at the problem's camera
function withLenses(prob, n) {
  const { camera, ...rest } = prob;
  return { ...rest, cameras: Array.from({ length: n }, () => ({ ...camera, params: [...camera.params] })), poseCam: Int32Array.from(prob.poses, (_, k) => k % n) };
}

test('BA on the GPU with 6 lenses lands where f64 does', async () => {
  const a = withLenses(problemFrom(SCENE, rng(33)), 6), b = withLenses(problemFrom(SCENE, rng(33)), 6);
  const opts = { refine: ['f', 'k1', 'k2'], maxIterations: 50 };
  bundleAdjust(a, opts);
  await gpu.bundleAdjust(b, opts);
  const [rot, cen] = apart(a, b);
  assert(rot < 1e-3 && cen < 1e-3, `poses apart by ${rot} deg, ${cen} in centre`);
  a.cameras.forEach((c, k) => assertClose(b.cameras[k].params[0], c.params[0], 0.05, `focal ${k}`));
});

test('BA on the GPU with a Huber loss is robust to 5% outlier observations', async () => {
  const prob = problemFrom(SCENE, rng(32), { outliers: 0.05 });
  await gpu.bundleAdjust(prob, { refine: ['f'], lossScale: 2, maxIterations: 50 });
  const errs = Array.from(reprojectionErrors(prob)).sort((a, b) => a - b);
  assert(errs[errs.length >> 1] < 1, `median reprojection ${errs[errs.length >> 1]}`);
  assertClose(prob.camera.params[0], 1230, 1230 * 5e-3, 'focal');
});

test('BA on the GPU with the points fixed: they stay, the poses land where f64 puts them', async () => {
  const a = problemFrom(SCENE, rng(35), { pointNoise: 0 }), b = problemFrom(SCENE, rng(35), { pointNoise: 0 }), X = b.points.slice();
  bundleAdjust(a, { fixPoints: true, maxIterations: 20 });
  await gpu.bundleAdjust(b, { fixPoints: true, maxIterations: 20 });
  assert(b.points.every((x, i) => x === X[i]), 'points moved');  // as they were, not through f32
  const [rot, cen] = apart(a, b);
  assert(rot < 1e-3 && cen < 1e-3, `poses apart by ${rot} deg, ${cen} in centre`);
});

// the f32 cost resolves about 1e-6 of itself: the solve ends where the model predicts a decrease under 1e-7 (ba.js's
// threshold; its points' part, gp^T V^-1 gp, with them), not where the cost stops resolving: the poses right and the
// points off, they are solved; the points fixed, the poses within 1e-3 deg of f64's, no tries wasted at the end
test('BA on the GPU: the solve ended by the model\'s predicted decrease', async () => {
  const a = problemFrom(SCENE, rng(36), { poseNoise: 0 }), b = problemFrom(SCENE, rng(36), { poseNoise: 0 });
  const oa = bundleAdjust(a, { maxIterations: 50 }), ob = await gpu.bundleAdjust(b, { maxIterations: 50 });
  assert(ob.iterations > 1, `${ob.iterations} iterations`);
  assertClose(ob.cost, oa.cost, 1e-5 * oa.cost, 'cost');
  const c = problemFrom(SCENE, rng(35), { pointNoise: 0 }), d = problemFrom(SCENE, rng(35), { pointNoise: 0 });
  bundleAdjust(c, { fixPoints: true, maxIterations: 50 });
  const od = await gpu.bundleAdjust(d, { fixPoints: true, maxIterations: 50 });
  const [rot] = apart(c, d);
  assert(rot < 1e-3 && od.profile.solves <= od.iterations + 2, `poses ${rot} deg from f64's, ${od.profile.solves} solves for ${od.iterations} iterations`);
});

// cond: a free pose seeing 2 points (4 equations for 6 unknowns) is weak, none of a full window; one fixed pose: gauge free
test('BA on the GPU: a pose of 2 observations is weak, the others not; one fixed pose leaves the gauge free', async () => {
  const full = problemFrom(SCENE, rng(31), { pointNoise: 0 });
  const a = (await gpu.bundleAdjust(problemFrom(SCENE, rng(31), { pointNoise: 0 }), { maxIterations: 20 })).cond;
  assert(a.minPivot > 1e-2 && a.weakPoses === 0 && !a.gaugeFree, `full: ${JSON.stringify(a)}`);
  const keep = [], seen = new Map();
  for (let o = 0; o < full.obsCam.length; o++) {
    const k = full.obsCam[o], n = seen.get(k) ?? 0;
    if (k === 7 && n >= 2) continue;
    seen.set(k, n + 1); keep.push(o);
  }
  const thin = { ...full, obsCam: Int32Array.from(keep, o => full.obsCam[o]), obsPoint: Int32Array.from(keep, o => full.obsPoint[o]),
                 obsUV: Float64Array.from(keep.flatMap(o => [full.obsUV[2 * o], full.obsUV[2 * o + 1]])) };
  const b = (await gpu.bundleAdjust(thin, { maxIterations: 20 })).cond;
  assert(b.minPivot < 1e-4 && b.weakPoses === 1, `pose 7 on 2 observations: ${JSON.stringify(b)}`);
  const one = problemFrom(SCENE, rng(31), { pointNoise: 0 });
  one.free[1] = 1;
  assert((await gpu.bundleAdjust(one, { maxIterations: 5 })).cond.gaugeFree, 'one fixed pose');
});

// a GPU validation error (here a pipeline whose bindings are not the pass's) is thrown, not solved through
test('BA on the GPU: a validation error is thrown', async () => {
  const bad = new GpuBundleAdjuster(gpu.device, { ...gpu.pipelines, back: gpu.pipelines.cost });
  let error = null;
  try { await bad.bundleAdjust(problemFrom(SCENE, rng(31)), { maxIterations: 2 }); } catch (e) { error = e; }
  assert(/GpuBundleAdjuster/.test(error?.message), `${error}`);
});

// the device as the solver sees it, with its queue's submits counted (failing from the n-th) and its buffers tracked
function wrapped(device, { failAt = Infinity, limits = device.limits } = {}) {
  const made = [];
  let submits = 0;
  const queue = new Proxy(device.queue, { get: (q, k) => k === 'submit' ? (cbs) => { if (++submits >= failAt) throw new Error('submit failed'); q.submit(cbs); }
                                                                     : typeof q[k] === 'function' ? q[k].bind(q) : q[k] });
  const dev = new Proxy(device, { get: (d, k) => {
    if (k === 'queue') return queue;
    if (k === 'limits') return limits;
    if (k === 'createBuffer') return (desc) => { const b = d.createBuffer(desc), destroy = b.destroy.bind(b); b.destroyed = false; b.destroy = () => { b.destroyed = true; destroy(); }; made.push(b); return b; };
    return typeof d[k] === 'function' ? d[k].bind(d) : d[k];
  } });
  return { dev, made };
}

// a failure inside the solve: thrown, its buffers destroyed and its error scope popped (none left for the device)
test('BA on the GPU: a failure inside the solve leaves no buffer nor error scope behind', async () => {
  const { dev, made } = wrapped(gpu.device, { failAt: 3 });
  let error = null;
  try { await new GpuBundleAdjuster(dev, gpu.pipelines).bundleAdjust(problemFrom(SCENE, rng(31)), { maxIterations: 5 }); } catch (e) { error = e; }
  assert(/submit failed/.test(error?.message) && made.length > 0 && made.every(b => b.destroyed), `${error}, ${made.filter(b => !b.destroyed).length} of ${made.length} buffers left`);
  let leftover = true;
  await gpu.device.popErrorScope().catch(() => { leftover = false; });
  assert(!leftover, 'an error scope left pushed');
});

// a window over the device's dispatch limit: refused before any buffer is made
test('BA on the GPU: a pass over maxComputeWorkgroupsPerDimension refused', async () => {
  const { dev, made } = wrapped(gpu.device, { limits: { ...gpu.device.limits, maxComputeWorkgroupsPerDimension: 16 } });
  let error = null;
  try { await new GpuBundleAdjuster(dev, gpu.pipelines).bundleAdjust(problemFrom(SCENE, rng(31)), { maxIterations: 2 }); } catch (e) { error = e; }
  assert(/workgroups a pass/.test(error?.message) && made.length === 0, `${error}, ${made.length} buffers`);
});

// nothing to solve (every pose fixed, no intrinsics): no zero-length writes (Deno's WebGPU dies on them), the poses kept
test('BA on the GPU: every pose fixed, nothing refined', async () => {
  const prob = problemFrom(SCENE, rng(31));
  prob.free.fill(0);
  const before = JSON.stringify(prob.poses), out = await gpu.bundleAdjust(prob, { maxIterations: 3 });
  assert(JSON.stringify(prob.poses) === before && Number.isFinite(out.cost), JSON.stringify(out.profile));
});

// the shared solver: made again once its device is lost
test('BA on the GPU: the shared solver made again after its device is lost', async () => {
  const a = await GpuBundleAdjuster.shared();
  a.device.destroy();
  await a.device.lost;
  assert(a.lost, 'lost flag');
  const b = await GpuBundleAdjuster.shared();
  assert(b !== a && !b.lost && Number.isFinite((await b.bundleAdjust(problemFrom(SCENE, rng(31)), { maxIterations: 2 })).cost), 'a new solver');
});

// cond's pivot against f64 at ba.js's solution: per free pose its Schur block, sum over its observations of
// Jc^T Q Jc with Q = (I + Jp V_o^-1 Jp^T)^-1 (V_o: the point's other observations; L2, so w = 1), Jacobi-scaled, its
// smallest Cholesky pivot. Near points and far points only (the latter: f32's cancellation, as a far point's V is near singular)
function f64Pivot(prob) {
  const { poses, free, points, obsCam, obsPoint, obsUV, camera } = prob, byPt = Array.from({ length: points.length / 3 }, () => []);
  const lin = Array.from(obsCam, (c, o) => linearizeObservation(camera, poses[c], Array.from(points.subarray(3 * obsPoint[o], 3 * obsPoint[o] + 3)), [obsUV[2 * o], obsUV[2 * o + 1]]));
  obsCam.forEach((_, o) => byPt[obsPoint[o]].push(o));
  const V = (list) => {  // sum of Jp^T Jp, regularized by 1e-9 of its trace
    const v = new Array(9).fill(0);
    for (const o of list) for (let i = 0; i < 9; i++) v[i] += lin[o].Jpoint[(i / 3) | 0] * lin[o].Jpoint[i % 3] + lin[o].Jpoint[3 + ((i / 3) | 0)] * lin[o].Jpoint[3 + i % 3];
    const tr = v[0] + v[4] + v[8] || 1;
    for (let i = 0; i < 3; i++) v[4 * i] += 1e-9 * tr;
    return v;
  };
  const M = poses.map(() => new Float64Array(36)), Vi = new Array(9);
  for (const list of byPt) for (const o of list) {
    if (!free[obsCam[o]]) continue;
    invert3(V(list.filter(x => x !== o)), Vi);
    const { Jpose: Jc, Jpoint: Jp } = lin[o], A = [[1, 0], [0, 1]];
    for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) A[a][b] += Jp[3 * a + i] * Vi[3 * i + j] * Jp[3 * b + j];
    const d = A[0][0] * A[1][1] - A[0][1] * A[1][0], Q = [[A[1][1] / d, -A[0][1] / d], [-A[1][0] / d, A[0][0] / d]];
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) M[obsCam[o]][6 * i + j] += Jc[6 * a + i] * Q[a][b] * Jc[6 * b + j];
  }
  let min = 1;
  M.forEach((m, c) => {
    if (!free[c]) return;
    const d = Array.from({ length: 6 }, (_, i) => Math.sqrt(m[7 * i])), Ms = new Float64Array(36), L = new Float64Array(36);
    for (let i = 0; i < 6; i++) for (let j = 0; j <= i; j++) Ms[6 * i + j] = m[6 * i + j] / (d[i] * d[j]);
    let piv = 0;
    if (choleskyFactor(Ms, 0, L, 0, 6)) { piv = Infinity; for (let i = 0; i < 6; i++) piv = Math.min(piv, L[7 * i] ** 2); }
    min = Math.min(min, piv);
  });
  return min;
}

test('BA on the GPU: its conditioning as f64 gives it, near points and far points only', async () => {
  const far = makeSyntheticScene({ numFrames: 15, maxFeatures: 1500, numGround: 0, numBoxes: 0, numFar: 3000 });
  for (const [name, scene] of [['near', SCENE], ['far', far]]) {
    const opts = { maxIterations: 20, lossScale: 1e9 }, a = problemFrom(scene, rng(31), { pointNoise: 0 });
    bundleAdjust(a, opts);
    const ref = f64Pivot(a), { cond } = await gpu.bundleAdjust(problemFrom(scene, rng(31), { pointNoise: 0 }), opts);
    assert(Math.abs(cond.minPivot / ref - 1) < 0.05, `${name}: pivot ${cond.minPivot}, f64 ${ref}`);
  }
});
