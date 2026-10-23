import { estimateFocal } from '../src/sfm/focal.js';
import { ransacFundamental } from '../src/sfm/geometry.js';
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { matchedPixels } from '../src/sfm/util.js';
import { THRESHOLDS } from '../src/sfm/compare.js';
import { assert, rng, SLOW, env, DUMP, loadInput } from './util.js';

// Fundamental matrices of sequential pairs `gap` keyframes apart
function sceneFs(scene, gap, r) {
  const ims = scene.input.images, Fs = [];
  for (const { i, j, matches } of scene.input.pairs) {
    if (j - i !== gap || matches.length < 32) continue;
    const [x1, x2] = matchedPixels(ims[i].keypoints, ims[j].keypoints, matches);
    const res = ransacFundamental(x1, x2, { rng: r, threshold: 2 });
    if (res && res.numInliers > 50) Fs.push(res.model);
  }
  return Fs;
}

Deno.test('focal self-calibration recovers the synthetic focal (900-1800 px) from F matrices alone', () => {
  for (const [seed, focal] of [[1, 1230], [2, 900], [3, 1800]]) {
    const scene = makeSyntheticScene({ seed, numFrames: 40, params: [focal, focal, 960, 540, 0.005, -0.002, 0.001, 0.0002],
                                       numGround: 6000, numBoxes: 6, numFar: 800, outlierFrac: 0.1 });
    const Fs = sceneFs(scene, 4, rng(seed));
    assert(Fs.length >= 20, `${Fs.length} pairs`);
    const est = estimateFocal(Fs, 1920, 1080);
    const err = Math.abs(est.focal / focal - 1);
    assert(err < 0.02, `seed ${seed}: focal ${est.focal.toFixed(1)} vs ${focal} (${(100 * err).toFixed(2)}%)`);
  }
});

Deno.test({ name: "focal from COLMAP's stored F matrices matches COLMAP's final focal (jardin5: 1231 px)", ignore: !SLOW || !DUMP, fn: async () => {
  const dump = await loadInput();
  const Fs = dump.pairs.filter(p => p.type === 'seq' && p.F && p.config === 'UNCALIBRATED' && p.matches.length > 200).map(p => p.F);
  const est = estimateFocal(Fs, dump.camera.width, dump.camera.height);
  const ref = Number(env('SFM_REF_FOCAL') ?? 1231);
  console.log(`  ${Fs.length} pairs -> focal ${est.focal.toFixed(1)} px (reference ${ref})`);
  assert(Math.abs(est.focal / ref - 1) < THRESHOLDS.colmap.focalErr, `focal ${est.focal.toFixed(1)} vs ${ref}`);
} });
