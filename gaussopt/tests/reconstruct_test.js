// Full S2..S4 pipeline (DESIGN-sfm M2): loop closure, segment merging, refinement sweeps and
// retriangulation, against exact ground truth; (slow) a dumped video (SFM_DUMP) against COLMAP with
// COLMAP's own defects excluded.
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { reconstruct, timingsLine } from '../src/sfm/reconstruct.js';
import { copySnapshot, SequentialMapper } from '../src/sfm/mapper.js';
import { compareModels, compareLocal, checkThresholds, THRESHOLDS } from '../src/sfm/compare.js';
import { assert, SLOW, DUMP, HAS_REF, loadInput, assessColmap, ORBIT, hasGPU } from './util.js';

for (const [name, options] of Object.entries({
  'orbit with revisits': {},
  'cut: no sequential pair across frame 20, revisits only': { cut: 0.4 },
  'cut + 20% outlier matches': { cut: 0.6, outlierFrac: 0.2 },
})) {
  Deno.test(`reconstruct, synthetic: ${name}`, async () => {
    const scene = makeSyntheticScene({ ...ORBIT, ...options });
    const { model, stats } = await reconstruct(scene.input, {});
    const m = compareModels(model, scene.truth.model);
    const tag = `RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg, focal ${(100 * m.focalErr).toFixed(2)}%, ` +
                `${stats.numSegments} segment(s), ${stats.loops.loopEdges} loops, ${stats.loops.merges} merges`;
    assert(stats.numSegments === 1, `${tag}: one segment after merging`);
    assert(stats.loops.loopEdges > 0, `${tag}: loops verified`);
    const fails = checkThresholds(m, 'truth');
    assert(fails.length === 0, `${tag}: ${fails.join(', ')}`);
    assert(compareLocal(model, scene.truth.model).medianRelRotDeg < 0.05, tag);
  });
}

// The time by stage (pipeline.js logs it): each stage's window BAs within it, and together all the mapper's BA time
Deno.test('reconstruct, synthetic: the time by stage, the BA within each and in all', async () => {
  const t = performance.now();
  const { stats } = await reconstruct(makeSyntheticScene(ORBIT).input, {});
  const total = performance.now() - t, { timings, time } = stats;
  const stages = ['map', 'loops', 'flowJoin', 'refine', 'retriangulate', 'lateMerge'];
  let ms = 0, ba = 0;
  for (const name of stages) {
    assert(timings[name] >= 0 && timings[`${name}.ba`] >= 0, `${name}: ${timings[name]} ms, BA ${timings[`${name}.ba`]}`);
    assert(timings[`${name}.ba`] <= timings[name], `${name}: BA ${timings[`${name}.ba`]} ms of ${timings[name]}`);
    ms += timings[name]; ba += timings[`${name}.ba`];
  }
  assert(ms <= total && ms > 0.9 * total, `stages ${ms.toFixed(0)} ms of ${total.toFixed(0)}`);
  assert(time.baCalls > 0 && Math.abs(ba - time.ba) < 1e-6 * time.ba + 1e-9, `BA by stage ${ba} ms, in all ${time.ba}`);
  assert(time.solve <= time.ba && time.init + time.focal <= timings.map, JSON.stringify(time));
  // the solve's parts (ba.js profile) within each bundleAdjust and nearly all of it
  const parts = ['linearize', 'normal', 'schur', 'cholesky', 'back', 'cost'];
  const partsMs = time.windows.reduce((a, w) => a + parts.reduce((b, k) => b + w[k], 0), 0);
  assert(time.windows.length === time.baCalls && time.windows.every(w => w.obs >= w.freeObs && w.freeObs > 0 && w.free > 0 && w.solves > 0),
         `${time.windows.length} windows of ${time.baCalls}`);
  assert(partsMs <= time.solve && partsMs > 0.8 * time.solve, `the parts ${partsMs.toFixed(0)} ms of the solves' ${time.solve.toFixed(0)}`);
  const line = timingsLine(stats);
  for (const name of [...stages, ...parts]) assert(line.includes(name), `${name} not in '${line}'`);
});

// The time with shadow solves (mapper.js baShadow): theirs apart from the map's BA, whose parts still make it up; each
// from its window's start (f64 shadowing f64: the map's poses exactly, in as many iterations)
Deno.test('reconstruct, synthetic: shadow solves timed apart, from the window\'s start', async () => {
  const { stats } = await reconstruct(makeSyntheticScene({ ...ORBIT, numFrames: 24 }).input, { baShadow: 'f64' });
  const { time } = stats, parts = ['linearize', 'normal', 'schur', 'cholesky', 'back', 'cost'];
  const partsMs = time.windows.reduce((a, w) => a + parts.reduce((b, k) => b + w[k], 0), 0);
  assert(time.shadowMs > 0 && partsMs > 0.8 * time.solve, `the parts ${partsMs.toFixed(0)} ms of ${time.solve.toFixed(0)}, shadows ${time.shadowMs}`);
  const w = time.shadow.f64;
  assert(w.length === time.baCalls && w.every(x => x.rot === 0 && x.cen === 0 && x.shadowIterations === x.iterations), JSON.stringify(w.find(x => x.rot || x.shadowIterations !== x.iterations)));
  assert(time.initBA > 0 && time.init >= 0, `the initial pairs' BA ${time.initBA} ms, outside it ${time.init}`);
});

// A reconstruction's stats are its own: the mapper's later window BAs (a session's repairs) leave them as they are
Deno.test('reconstruct, synthetic: the stats do not change with later window BAs', async () => {
  const { mapper, stats } = await reconstruct(makeSyntheticScene({ ...ORBIT, numFrames: 24 }).input, {});
  const n = stats.time.windows.length, ba = stats.time.ba;
  await mapper.localBA(mapper.poses.findIndex(Boolean));
  assert(stats.time.windows.length === n && stats.time.ba === ba, `${n} -> ${stats.time.windows.length} windows`);
});

// baPrecision 'gpu' (mapper.js): the GPU solver (ba_gpu.js) for a window of baGpuMinObs observations, ba.js otherwise, and
// ba.js from the start where the GPU solve fails (illReason, an error) or there is no GPU solver
Deno.test({ name: 'reconstruct, synthetic: gpu, each window BA routed by its size, a failed GPU solve redone in f64 from its start', ignore: !hasGPU, fn: async () => {
  const { mapper, stats } = await reconstruct(makeSyntheticScene({ ...ORBIT, numFrames: 24 }).input, { baPrecision: 'gpu', baGpuMinObs: 3000 });
  const { gpu, cpu } = stats.time.route;
  assert(gpu > 0 && cpu > 0, `${gpu} windows on the GPU, ${cpu} on ba.js`);
  assert(/on the GPU solver/.test(timingsLine(stats)) && /build/.test(timingsLine(stats)), timingsLine(stats));
  const snap = mapper.snapshot(), center = mapper.poses.findIndex((p, k) => p && k > 8);
  // restore() takes the snapshot's arrays as they are: a copy each time
  const run = async (o) => {
    mapper.restore(copySnapshot(snap)); Object.assign(mapper.o, { baGpu: null, ...o }); mapper._gpu = undefined; mapper.time.route = { gpu: 0, cpu: 0 };
    await mapper.localBA(center, { iterations: 10 });
    return [JSON.stringify(mapper.poses), { ...mapper.time.route }];
  };
  const [f64] = await run({ baPrecision: 'f64' });
  const [kept, r1] = await run({ baPrecision: 'gpu', baGpuMinObs: 0, baIllPivot: 0 }), [redone] = await run({ baPrecision: 'gpu', baGpuMinObs: 0, baIllPivot: 2 });
  const [small, r2] = await run({ baPrecision: 'gpu', baGpuMinObs: 1e9 });
  assert(r1.gpu === 1 && kept !== f64, 'a window of baGpuMinObs observations: on the GPU');
  assert(redone === f64, 'the GPU solve failing: the poses ba.js gives from the same start');
  assert(r2.cpu === 1 && small === f64, 'under baGpuMinObs: ba.js');
  // a GPU error: redone by ba.js from the start; no GPU solver: ba.js
  const n = mapper.time.fallback?.length ?? 0;
  const [broken] = await run({ baPrecision: 'gpu', baGpuMinObs: 0, baIllPivot: 0, baGpu: { bundleAdjust: async () => { throw new Error('device lost'); } } });
  assert(broken === f64 && mapper.time.fallback.at(-1).reason === 'GPU error' && mapper.time.fallback.length === n + 1, 'a GPU error');
  const [none, r3] = await run({ baPrecision: 'gpu', baGpuMinObs: 0, baGpu: Promise.reject(new Error('no WebGPU')) });
  assert(none === f64 && r3.cpu === 1, 'no GPU solver');
  // baShadow 'gpu': the GPU solve on a copy (its own: ba.js's would give the map's poses exactly), the map ba.js's
  mapper.time.shadow = undefined;
  const [shadowed] = await run({ baPrecision: 'f64', baShadow: 'gpu' }), sh = mapper.time.shadow?.gpu;
  assert(shadowed === f64 && sh?.length === 1 && sh[0].rot > 0 && sh[0].rot < 0.01 && sh[0].rmse / sh[0].ref < 1.001, `the GPU shadow ${JSON.stringify(sh?.[0])}`);
  // a GPU shadow failing: no shadow of the window, the map's solve as it is
  mapper.time.shadow = undefined;
  const [unshadowed] = await run({ baPrecision: 'f64', baShadow: 'f64,gpu', baGpu: { bundleAdjust: async () => { throw new Error('device lost'); } } });
  assert(unshadowed === f64 && !mapper.time.shadow, `shadows ${JSON.stringify(mapper.time.shadow)}`);
  // its device lost: the GPU off for the mapper's later windows
  let calls = 0;
  const lost = { lost: true, bundleAdjust: async () => { calls++; throw new Error('device lost'); } };
  await run({ baPrecision: 'gpu', baGpuMinObs: 0, baGpu: lost });
  await mapper.localBA(center, { iterations: 10 });
  assert(calls === 1 && mapper.time.route.cpu === 1 && mapper.time.route.gpu === 1, `${calls} GPU calls, route ${JSON.stringify(mapper.time.route)}`);
} });

// the solvers a mapper takes: f64 and gpu, for the map and its shadows
Deno.test('mapper: a solver it does not have refused', () => {
  const input = makeSyntheticScene({ ...ORBIT, numFrames: 6 }).input;
  for (const o of [{ baPrecision: 'ba32' }, { baShadow: 'f64,f32' }]) {
    let error = null;
    try { new SequentialMapper(input, o); } catch (e) { error = e; }
    assert(/no window BA solver/.test(error?.message), `${JSON.stringify(o)}: ${error}`);
  }
  new SequentialMapper(input, { baPrecision: 'gpu', baShadow: 'f64,gpu' });
});

// illReason (mapper.js): f32 lost the solve, a direction left free, or the RMSE up; each rule on its own; gpuFriendly
Deno.test('mapper: why an f32 solve fails; which windows go to the GPU', async () => {
  const { mapper } = await reconstruct(makeSyntheticScene({ ...ORBIT, numFrames: 12 }).input, {});
  const ok = { minPivot: 0.05, gaugeFree: false, rmse0: 2, rmse: 0.5 };
  const why = (c) => mapper.illReason({ ...ok, ...c });
  assert(why({}) === '', why({}));
  assert(why({ rmse: NaN }) === 'not finite' && why({ rmse: Infinity }) === 'not finite', why({ rmse: NaN }));
  assert(why({ minPivot: 1e-6 }) === 'pivot' && why({ minPivot: 1e-6, gaugeFree: true }) === '', 'the pivot, its gauge fixed');
  assert(why({ rmse: 3 }) === 'RMSE up', why({ rmse: 3 }));
  // the routing: baGpuMinObs observations, whatever the gauge
  const prob = (fixed, n) => ({ free: Uint8Array.from({ length: 6 }, (_, k) => +(k >= fixed)), obsCam: new Int32Array(n) });
  Object.assign(mapper.o, { baGpuMinObs: 100 });
  assert(mapper.gpuFriendly(prob(2, 100)) && mapper.gpuFriendly(prob(1, 100)) && !mapper.gpuFriendly(prob(2, 99)), 'gpuFriendly');
});

Deno.test({ name: 'reconstruct, jardin5 (SFM_DUMP): segments merged, every consistent COLMAP piece within the colmap thresholds', ignore: !SLOW || !DUMP || !HAS_REF, fn: async () => {
  const input = await loadInput();
  const { model, stats } = await reconstruct(input, {});
  const { ref, local, fails } = await assessColmap(model, input);
  console.log(`  ${model.images.length}/${input.images.length} registered, ${stats.numSegments} segment(s) after ${stats.loops.merges} merges, ` +
              `${stats.loops.loopEdges} loops; local rotation ${local.medianRelRotDeg.toFixed(3)} deg`);
  assert(model.images.length >= 0.99 * input.images.length, 'registered');
  // a stretch after a hover joins only once refinement extends the tracks (the late merge, DESIGN-sfm R5)
  assert(stats.numSegments === 1 && stats.loops.merges >= 3, `${stats.numSegments} segments, ${stats.loops.merges} merges`);
  assert(Math.abs(model.focal / ref.focal - 1) < THRESHOLDS.colmap.focalErr, 'focal');
  assert(fails.length === 0, fails.join('; '));
} });
