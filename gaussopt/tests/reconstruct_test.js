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
  assert(time.route.gpu + time.route.cpu === time.baCalls && time.initBA > 0 && time.init >= 0, JSON.stringify(time));
  // the solve's parts (ba.js's profile, ba_gpu.js's) within each bundleAdjust and nearly all of it
  const parts = ['linearize', 'normal', 'schur', 'cholesky', 'back', 'cost'];
  const partsMs = time.windows.reduce((a, w) => a + [...parts, 'setup', 'build', 'eval'].reduce((b, k) => b + (w[k] ?? 0), 0), 0);
  assert(time.windows.length === time.baCalls && time.windows.every(w => w.obs >= w.freeObs && w.freeObs > 0 && w.free > 0 && w.solves > 0),
         `${time.windows.length} windows of ${time.baCalls}`);
  assert(partsMs <= time.solve && partsMs > 0.8 * time.solve, `the parts ${partsMs.toFixed(0)} ms of the solves' ${time.solve.toFixed(0)}`);
  const line = timingsLine(stats);
  for (const name of [...stages, ...parts]) assert(line.includes(name), `${name} not in '${line}'`);
});

// A reconstruction's stats are its own: the mapper's later window BAs (a session's repairs) leave them as they are
Deno.test('reconstruct, synthetic: the stats do not change with later window BAs', async () => {
  const { mapper, stats } = await reconstruct(makeSyntheticScene({ ...ORBIT, numFrames: 24 }).input, {});
  const n = stats.time.windows.length, ba = stats.time.ba;
  await mapper.localBA(mapper.poses.findIndex(Boolean));
  assert(stats.time.windows.length === n && stats.time.ba === ba, `${n} -> ${stats.time.windows.length} windows`);
});

// localBA's runs of one window from the same snapshot: run(options) -> [the poses (JSON), the route]; restore() takes the
// snapshot's arrays as they are: a copy each time
function windowRuns(mapper) {
  const snap = mapper.snapshot(), center = mapper.poses.findIndex((p, k) => p && k > 8);
  const run = async (o) => {
    mapper.restore(copySnapshot(snap)); Object.assign(mapper.o, { baGpu: null, ...o }); mapper._gpu = undefined; mapper.time.route = { gpu: 0, cpu: 0 };
    await mapper.localBA(center, { iterations: 10 });
    return [JSON.stringify(mapper.poses), { ...mapper.time.route }];
  };
  return { run, again: () => mapper.localBA(center, { iterations: 10 }) };
}

// The window BAs (mapper.js localBA): the GPU solver (ba_gpu.js) for a window of baGpuMinObs observations, ba.js otherwise,
// and ba.js from the start where the GPU's solve fails (illReason)
Deno.test({ name: 'reconstruct, synthetic: each window BA routed by its size, an ill GPU solve redone by ba.js from its start', ignore: !hasGPU, fn: async () => {
  const { mapper, stats } = await reconstruct(makeSyntheticScene({ ...ORBIT, numFrames: 24 }).input, { baGpuMinObs: 3000 });
  const { gpu, cpu } = stats.time.route;
  assert(gpu > 0 && cpu > 0, `${gpu} windows on the GPU, ${cpu} on ba.js`);
  assert(/on the GPU solver/.test(timingsLine(stats)) && /build/.test(timingsLine(stats)), timingsLine(stats));
  const { run } = windowRuns(mapper);
  const [f64, r0] = await run({ baGpuMinObs: 1e9 });
  const [kept, r1] = await run({ baGpuMinObs: 0, baIllPivot: 0 }), [redone] = await run({ baGpuMinObs: 0, baIllPivot: 2 });
  assert(r0.cpu === 1 && r0.gpu === 0, 'under baGpuMinObs: ba.js');
  assert(r1.gpu === 1 && kept !== f64, 'a window of baGpuMinObs observations: on the GPU');
  assert(redone === f64, 'the GPU solve failing: the poses ba.js gives from the same start');
} });

// The GPU solver failing (fakes: no WebGPU needed): an error, none, its device lost, a solve or the solver's making that
// never ends (baGpuTimeoutMs); each window redone by ba.js from its start, the GPU off from a loss or a stall on
Deno.test('reconstruct, synthetic: a failing or stalled GPU solver, its windows on ba.js', async () => {
  const { mapper } = await reconstruct(makeSyntheticScene({ ...ORBIT, numFrames: 24 }).input, { baGpuMinObs: 1e9 });
  const { run, again } = windowRuns(mapper);
  const [f64] = await run({ baGpuMinObs: 1e9 });
  const n = mapper.time.fallback.length;
  const [broken] = await run({ baGpuMinObs: 0, baIllPivot: 0, baGpu: { bundleAdjust: async () => { throw new Error('device lost'); } } });
  assert(broken === f64 && mapper.time.fallback.at(-1) === 'GPU error' && mapper.time.fallback.length === n + 1, 'a GPU error');
  const [none, r3] = await run({ baGpuMinObs: 0, baGpu: Promise.reject(new Error('no WebGPU')) });
  assert(none === f64 && r3.cpu === 1, 'no GPU solver');
  // its device lost, or a solve stalled: the GPU off for the mapper's later windows
  for (const [what, bundleAdjust, o] of [['lost', async () => { throw new Error('device lost'); }, {}],
                                         ['stalled', () => new Promise(() => {}), { baGpuTimeoutMs: 50 }]]) {
    let calls = 0;
    const [poses] = await run({ baGpuMinObs: 0, baIllPivot: 0, ...o, baGpu: { lost: what === 'lost', bundleAdjust: (...a) => { calls++; return bundleAdjust(...a); } } });
    await again();
    assert(poses === f64 && calls === 1 && mapper.time.route.cpu === 1 && mapper.time.route.gpu === 1, `${what}: ${calls} GPU calls, route ${JSON.stringify(mapper.time.route)}`);
  }
  // the solver's making stalled: ba.js
  const [unmade, r4] = await run({ baGpuMinObs: 0, baGpuTimeoutMs: 50, baGpu: new Promise(() => {}) });
  assert(unmade === f64 && r4.cpu === 1, 'the GPU solver never made');
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
