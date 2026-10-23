// Full S2..S4 pipeline (DESIGN-sfm M2): loop closure, segment merging, refinement sweeps and
// retriangulation, against exact ground truth; (slow) jardin5 against COLMAP with COLMAP's own
// defects excluded.
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { reconstruct } from '../src/sfm/reconstruct.js';
import { compareModels, compareLocal, checkThresholds, THRESHOLDS } from '../src/sfm/compare.js';
import { assert, SLOW, DUMP, HAS_REF, loadInput, assessColmap, ORBIT } from './util.js';

for (const [name, options] of Object.entries({
  'orbit with revisits': {},
  'cut: no sequential pair across frame 20, revisits only': { cut: 0.4 },
  'cut + 20% outlier matches': { cut: 0.6, outlierFrac: 0.2 },
})) {
  Deno.test(`reconstruct, synthetic: ${name}`, () => {
    const scene = makeSyntheticScene({ ...ORBIT, ...options });
    const { model, stats } = reconstruct(scene.input, {});
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

Deno.test({ name: 'reconstruct, jardin5 (SFM_DUMP): segments merged, every consistent COLMAP piece within the colmap thresholds', ignore: !SLOW || !DUMP || !HAS_REF, fn: async () => {
  const input = await loadInput();
  const { model, stats } = reconstruct(input, {});
  const { ref, local, fails } = await assessColmap(model, input);
  console.log(`  ${model.images.length}/${input.images.length} registered, ${stats.numSegments} segment(s) after ${stats.loops.merges} merges, ` +
              `${stats.loops.loopEdges} loops; local rotation ${local.medianRelRotDeg.toFixed(3)} deg`);
  assert(model.images.length >= 0.99 * input.images.length, 'registered');
  // the stretch after the hover at frames 362-373 joins only once refinement extends the tracks
  // (the late merge, DESIGN-sfm R5)
  assert(stats.numSegments === 1 && stats.loops.merges >= 3, `${stats.numSegments} segments, ${stats.loops.merges} merges`);
  assert(Math.abs(model.focal / ref.focal - 1) < THRESHOLDS.colmap.focalErr, 'focal');
  assert(fails.length === 0, fails.join('; '));
} });
