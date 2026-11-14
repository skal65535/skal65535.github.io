// ba_gpu_test.js — The GPU bundle adjustment (src/sfm/ba_gpu.js) against ba.js from the same start, and its conditioning.
import { bundleAdjust, reprojectionErrors, linearizeObservation } from '../src/sfm/ba.js';
import { GpuBundleAdjuster, pad16 } from '../src/sfm/ba_gpu.js';
import { poseCenter } from '../src/sfm/camera.js';
import { rotationAngle, choleskyFactor, invert3 } from '../src/sfm/linalg.js';
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { assert, assertClose, rng, problemFrom, hasGPU, fyOff } from './util.js';

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

test('BA on the GPU, its reduced system by conjugate gradients (denseMax 0), lands where its dense Cholesky does', async () => {
  const a = problemFrom(SCENE, rng(31)), b = problemFrom(SCENE, rng(31)), beats = [];
  const oa = await gpu.bundleAdjust(a, { refine: ['f', 'k1'], maxIterations: 50 });
  const ob = await gpu.bundleAdjust(b, { refine: ['f', 'k1'], maxIterations: 50, denseMax: 0, cgTol: 1e-8, alive: () => beats.push(1) });  // solved as exactly as the Cholesky
  const [rot, cen] = apart(a, b);
  assert(ob.profile.cg > 0 && !oa.profile.cg, `${ob.profile.cg} conjugate gradient iterations`);
  assert(rot < 1e-4 && cen < 1e-4, `poses apart by ${rot} deg, ${cen} in centre`);
  assertClose(ob.cost, oa.cost, 1e-5 * oa.cost, 'cost');
  assertClose(b.camera.params[0], a.camera.params[0], 0.01, 'focal');
  assert(ob.cond.minPivot === oa.cond.minPivot || Math.abs(ob.cond.minPivot - oa.cond.minPivot) < 1e-6, `min pivot ${ob.cond.minPivot} vs ${oa.cond.minPivot}`);
  assert(beats.length >= 2 * ob.profile.solves, `${beats.length} heartbeats for ${ob.profile.solves} solves`);
});

// A solve on the CPU is followed by a heartbeat before anything is awaited (the next readback's comes later): a solve
// long enough is not taken for a stalled GPU
test('BA on the GPU: a heartbeat right after each solve of its reduced system', async () => {
  const { ReducedSystem } = await import('../src/sfm/rcs.js');
  const solve = ReducedSystem.prototype.solve;
  let solves = 0, after = 0, pending = false;
  ReducedSystem.prototype.solve = function (b) {
    const x = solve.call(this, b);
    solves++; pending = true;
    queueMicrotask(() => { pending = false; });  // gone at the first await
    return x;
  };
  try {
    await gpu.bundleAdjust(problemFrom(SCENE, rng(31)), { refine: ['f'], maxIterations: 20, alive: () => { if (pending) after++; pending = false; } });
  } finally { ReducedSystem.prototype.solve = solve; }
  assert(solves > 0 && after === solves, `${solves} solves, ${after} with a heartbeat before the next await`);
});

// the problem with n lenses: pose k on lens k % n, each starting at the problem's camera
function withLenses(prob, n) {
  const { camera, ...rest } = prob;
  return { ...rest, cameras: Array.from({ length: n }, () => ({ ...camera, params: [...camera.params] })), poseCam: Int32Array.from(prob.poses, (_, k) => k % n) };
}

// p1, p2 refined from 0 on a lens of gare2's tangential distortion
test('BA on the GPU refining p1, p2 lands where f64 does', async () => {
  const scene = makeSyntheticScene({ numFrames: 15, numGround: 3000, numBoxes: 4, numFar: 300, maxFeatures: 1500,
                                     params: [1230, 1230, 960, 540, -0.004, 0.002, 0.003, -0.02] });
  const refine = ['f', 'k1', 'k2', 'p1', 'p2'];
  const a = problemFrom(scene, rng(31)), b = problemFrom(scene, rng(31));
  for (const p of [a, b]) p.camera.params[6] = p.camera.params[7] = 0;
  const oa = bundleAdjust(a, { refine, maxIterations: 50 }), ob = await gpu.bundleAdjust(b, { refine, maxIterations: 50 });
  const [rot, cen] = apart(a, b);
  assert(rot < 1e-3 && cen < 1e-3, `poses apart by ${rot} deg, ${cen} in centre`);
  assertClose(ob.cost, oa.cost, 1e-4 * oa.cost, 'cost');
  assert(Math.abs(b.camera.params[7] - a.camera.params[7]) < 1e-4 && Math.abs(b.camera.params[6] - a.camera.params[6]) < 1e-4,
         `p1, p2: GPU ${b.camera.params.slice(6)}, f64 ${a.camera.params.slice(6)}`);
});

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
  assert(a.minPivot > 1e-2 && !a.gaugeFree, `full: ${JSON.stringify(a)}`);
  const keep = [], seen = new Map();
  for (let o = 0; o < full.obsCam.length; o++) {
    const k = full.obsCam[o], n = seen.get(k) ?? 0;
    if (k === 7 && n >= 2) continue;
    seen.set(k, n + 1); keep.push(o);
  }
  const thin = { ...full, obsCam: Int32Array.from(keep, o => full.obsCam[o]), obsPoint: Int32Array.from(keep, o => full.obsPoint[o]),
                 obsUV: Float64Array.from(keep.flatMap(o => [full.obsUV[2 * o], full.obsUV[2 * o + 1]])) };
  const b = (await gpu.bundleAdjust(thin, { maxIterations: 20 })).cond;
  assert(b.minPivot < 1e-4, `pose 7 on 2 observations: ${JSON.stringify(b)}`);
  const one = problemFrom(SCENE, rng(31), { pointNoise: 0 });
  one.free[1] = 1;
  const c = (await gpu.bundleAdjust(one, { maxIterations: 5 })).cond;
  assert(c.gaugeFree && c.minPivot === null, `one fixed pose: ${JSON.stringify(c)}`);
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

// passes over maxComputeWorkgroupsPerDimension (here 32: about 350 workgroups a linearize) dispatched in 2D: the same solve
test('BA on the GPU: passes over maxComputeWorkgroupsPerDimension dispatched in 2D, the same solve', async () => {
  const { dev } = wrapped(gpu.device, { limits: { ...gpu.device.limits, maxComputeWorkgroupsPerDimension: 32 } });
  const a = problemFrom(SCENE, rng(31)), b = problemFrom(SCENE, rng(31)), opts = { refine: ['f', 'k1'], maxIterations: 5 };
  assert(Math.ceil(a.obsCam.length / 64) > 32, `${a.obsCam.length} observations`);
  const oa = await gpu.bundleAdjust(a, opts), ob = await new GpuBundleAdjuster(dev, gpu.pipelines).bundleAdjust(b, opts);
  assert(ob.cost === oa.cost && JSON.stringify(b.poses) === JSON.stringify(a.poses), `cost ${ob.cost} vs ${oa.cost}`);
});

// nothing to solve (every pose fixed, no intrinsics): no zero-length writes (Deno's WebGPU dies on them), the poses kept
test('BA on the GPU: every pose fixed, nothing refined', async () => {
  const prob = problemFrom(SCENE, rng(31));
  prob.free.fill(0);
  const before = JSON.stringify(prob.poses), out = await gpu.bundleAdjust(prob, { maxIterations: 3 });
  assert(JSON.stringify(prob.poses) === before && Number.isFinite(out.cost), JSON.stringify(out.profile));
});

// the shared solver: made again once its device is lost, or once its making failed (no adapter for a while)
test('BA on the GPU: the shared solver made again after its device is lost or its making failed', async () => {
  const a = await GpuBundleAdjuster.shared();
  a.device.destroy();
  await a.device.lost;
  assert(a.lost, 'lost flag');
  const b = await GpuBundleAdjuster.shared();
  assert(b !== a && !b.lost && Number.isFinite((await b.bundleAdjust(problemFrom(SCENE, rng(31)), { maxIterations: 2 })).cost), 'a new solver');
  b.device.destroy();
  await b.device.lost;
  navigator.gpu.requestAdapter = async () => null;
  let error = null;
  try { await GpuBundleAdjuster.shared(); } catch (e) { error = e; } finally { delete navigator.gpu.requestAdapter; }
  assert(/adapter/.test(error?.message), `${error}`);
  const c = await GpuBundleAdjuster.shared().catch(e => e);
  assert(c instanceof GpuBundleAdjuster && !c.lost, `after a failed making: ${c}`);
});

// an f32 overflow in the reduced system (here a NaN loss scale): thrown, not a solve that quietly takes no step
test('BA on the GPU: a reduced system not finite is thrown', async () => {
  let error = null;
  try { await gpu.bundleAdjust(problemFrom(SCENE, rng(31)), { maxIterations: 3, lossScale: NaN }); } catch (e) { error = e; }
  assert(/not finite/.test(error?.message), `${error}`);
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

// the buffers kept from one solve to the next: a second solve (a smaller window, in the first one's larger buffers)
// makes none and lands where a new solver does; a solve beside another makes its own, destroyed at its end
test('BA on the GPU: the buffers kept between solves, the same solve in them', async () => {
  const small = makeSyntheticScene({ numFrames: 8, numGround: 800, numBoxes: 2, numFar: 100, maxFeatures: 600 });
  const { dev, made } = wrapped(gpu.device), solver = new GpuBundleAdjuster(dev, gpu.pipelines), opts = { refine: ['f', 'k1'], maxIterations: 5 };
  await solver.bundleAdjust(problemFrom(SCENE, rng(31)), opts);
  const first = made.length, a = problemFrom(small, rng(33)), b = problemFrom(small, rng(33));
  const oa = await solver.bundleAdjust(a, opts), ob = await new GpuBundleAdjuster(gpu.device, gpu.pipelines).bundleAdjust(b, opts);
  assert(made.length === first && first > 0, `${made.length - first} buffers made by the second solve (${first} by the first)`);
  assert(made.every(x => !x.destroyed), 'a kept buffer destroyed');
  assert(oa.cost === ob.cost && JSON.stringify(a.poses) === JSON.stringify(b.poses), `cost ${oa.cost} vs ${ob.cost}`);
  const [c, d] = [problemFrom(small, rng(34)), problemFrom(small, rng(34))], [oc, od] = await Promise.all([solver.bundleAdjust(c, opts), solver.bundleAdjust(d, opts)]);
  const own = made.slice(first);
  assert(own.length > 0 && own.every(x => x.destroyed), `the solve beside: ${own.length} buffers, ${own.filter(x => !x.destroyed).length} left`);
  assert(oc.cost === od.cost && JSON.stringify(c.poses) === JSON.stringify(d.poses), `side by side: cost ${oc.cost} vs ${od.cost}`);
  solver.release();
  assert(solver.pool.size === 0 && made.every(x => x.destroyed), 'released');
});

// the realm's solver's buffers dropped once it solves nothing (the worker after each action)
test('BA on the GPU: releaseShared drops the shared solver\'s buffers', async () => {
  const s = await GpuBundleAdjuster.shared();
  await s.bundleAdjust(problemFrom(SCENE, rng(31)), { maxIterations: 2 });
  assert(s.pool.size > 0, 'nothing kept');
  await GpuBundleAdjuster.releaseShared();
  assert(s.pool.size === 0, `${s.pool.size} kept`);
  assert(Number.isFinite((await s.bundleAdjust(problemFrom(SCENE, rng(31)), { maxIterations: 2 })).cost), 'a solve after');
});

// a window of most frames fixed (the mapper's large ones: 3.7M of 4.15M observations by fixed frames): where f64 lands,
// the pose and lens rows on the GPU for the free frames' observations only (under 200 bytes an observation made: 417 before, 4500 of the 22500 by free frames)
test('BA on the GPU: most frames fixed, the same solve, the fixed frames\' observations without pose rows', async () => {
  const fixedMost = (p) => { p.free = Uint8Array.from(p.poses, (_, k) => +(k >= p.poses.length - 3)); return p; };
  const a = fixedMost(problemFrom(SCENE, rng(37))), b = fixedMost(problemFrom(SCENE, rng(37)));
  const { dev, made } = wrapped(gpu.device), opts = { maxIterations: 50 };
  const oa = bundleAdjust(a, opts), ob = await new GpuBundleAdjuster(dev, gpu.pipelines).bundleAdjust(b, opts);
  const [rot, cen] = apart(a, b);  // 1.1e-4 deg apart in the J layout before as in this one: f32's, in this window
  assert(rot < 3e-4 && cen < 1e-4, `poses apart by ${rot} deg, ${cen} in centre`);
  assertClose(ob.cost, oa.cost, 1e-5 * oa.cost, 'cost');
  const bytes = made.reduce((s, x) => s + x.size, 0), n = b.obsCam.length;
  assert(ob.profile.freeObs < 0.3 * n && bytes < 200 * n, `${(bytes / n).toFixed(0)} bytes an observation, ${ob.profile.freeObs} of ${n} by free frames`);
});

// a solve its caller gave up (a read-back that never comes back) does not keep the buffers from the next solves once
// the solver is idle: the next solve keeps its own, and lands where a new solver does
test('BA on the GPU: after a stalled solve, idle gives the next solve the kept buffers', async () => {
  let stall = true;
  const { dev, made } = wrapped(gpu.device);
  const stalling = new Proxy(dev, { get: (d, k) => k !== 'createBuffer' ? d[k] : (desc) => {
    const b = d.createBuffer(desc);
    if (stall && desc.usage & GPUBufferUsage.MAP_READ) b.mapAsync = () => new Promise(() => {});
    return b;
  } });
  const solver = new GpuBundleAdjuster(stalling, gpu.pipelines);
  solver.bundleAdjust(problemFrom(SCENE, rng(31)), { maxIterations: 2 }).catch(() => {});  // never ends
  await new Promise(r => setTimeout(r, 200));
  stall = false;
  solver.idle();
  const first = made.length, a = problemFrom(SCENE, rng(32)), b = problemFrom(SCENE, rng(32));
  const oa = await solver.bundleAdjust(a, { maxIterations: 5 }), ob = await gpu.bundleAdjust(b, { maxIterations: 5 });
  const own = made.slice(first);
  assert(own.length > 0 && own.every(x => !x.destroyed) && solver.pool.size > 0 && solver.solving === null,
         `the next solve: ${own.length} buffers, ${own.filter(x => x.destroyed).length} destroyed, ${solver.pool.size} kept`);
  assert(oa.cost === ob.cost && JSON.stringify(a.poses) === JSON.stringify(b.poses), `cost ${oa.cost} vs ${ob.cost}`);
});

// sizes rounded by arithmetic: from 2^31 bytes on, & ~15 wrapped (a buffer of 16 bytes made for 1.75e9)
test('BA on the GPU: pad16 past 2^31 bytes', () => {
  assert(pad16(2 ** 31 + 1) === 2 ** 31 + 16 && pad16(1.25 * 1.4e9) === 1.75e9 && pad16(0) === 16 && pad16(17) === 32, 'pad16');
});

// every pose fixed, the lens refined: every observation with its lens half only
test('BA on the GPU: every pose fixed, the lens refined, where f64 lands', async () => {
  const a = problemFrom(SCENE, rng(38)), b = problemFrom(SCENE, rng(38)), opts = { refine: ['f', 'k1'], maxIterations: 50 };
  a.free.fill(0); b.free.fill(0);
  const oa = bundleAdjust(a, opts), ob = await gpu.bundleAdjust(b, opts);
  assertClose(ob.cost, oa.cost, 1e-5 * oa.cost, 'cost');
  assertClose(b.camera.params[0], a.camera.params[0], 0.01, 'focal');
});

// fy apart from fx (refine 'fy', the sixth intrinsic): where ba.js lands
test('BA on the GPU lands where f64 does: fy apart from fx', async () => {
  const refine = ['f', 'k1', 'k2', 'p1', 'p2', 'fy'], a = fyOff(rng(31)), b = fyOff(rng(31));
  const oa = bundleAdjust(a, { refine, maxIterations: 50 }), ob = await gpu.bundleAdjust(b, { refine, maxIterations: 50 });
  assertClose(ob.cost, oa.cost, 1e-5 * oa.cost, 'cost');
  assert(Math.abs(b.camera.params[1] - a.camera.params[1]) < 0.05 && Math.abs(b.camera.params[1] / 1230 - 1) < 1e-3, `fy ${b.camera.params[1]}, f64 ${a.camera.params[1]}`);
});

// what the GPU solver does not have: refused, not solved otherwise
test('BA on the GPU refuses fixedPoints and another loss than Huber', async () => {
  for (const opts of [{ fixedPoints: new Uint8Array(1) }, { loss: 'softl1' }]) {
    let threw = false;
    try { await gpu.bundleAdjust(problemFrom(SCENE, rng(31)), opts); } catch { threw = true; }
    assert(threw, JSON.stringify(Object.keys(opts)));
  }
});
