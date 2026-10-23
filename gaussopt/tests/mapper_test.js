// Sequential mapper (DESIGN-sfm M1): synthetic scenes against exact ground truth, and (slow) the
// jardin5 COLMAP database against COLMAP's model, after excluding that model's own defects
// (scale breaks, misregistered frames): never match COLMAP where it is wrong.
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { SequentialMapper } from '../src/sfm/mapper.js';
import { compareModels, compareLocal, checkThresholds, THRESHOLDS } from '../src/sfm/compare.js';
import { assert, SLOW, DUMP, HAS_REF, loadInput, assessColmap } from './util.js';

const BASE = { numFrames: 30, maxFeatures: 1000, numGround: 8000, numBoxes: 8, numFar: 1000 };

function mapSynthetic(options, mapperOptions = {}) {
  const scene = makeSyntheticScene({ ...BASE, ...options });
  const mapper = new SequentialMapper(scene.input, mapperOptions);
  const model = mapper.run();
  return { scene, mapper, model, m: compareModels(model, scene.truth.model), l: compareLocal(model, scene.truth.model) };
}

const CASES = {
  'orbit (focal prior 1.87x off)': {},
  '20% outlier matches': { outlierFrac: 0.2 },
  'yaw-in-place segment': { spin: { at: 0.5, frames: 6, deg: 60 } },
  'wide lens (f = 800, k1 = -0.08)': { params: [800, 800, 960, 540, -0.08, 0.01, 0, 0] },
  'straight forward flight': { path: 'line', pitchDeg: 45 },
};

for (const [name, options] of Object.entries(CASES)) {
  Deno.test(`mapper, synthetic: ${name}`, () => {
    const { scene, mapper, m, l } = mapSynthetic(options);
    const tag = `${name}: RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg, focal ${(100 * m.focalErr).toFixed(2)}%`;
    assert(m.numCommon === scene.truth.images.length && mapper.nextSegmentId === 1, `${tag}: ${m.numCommon} registered, ${mapper.nextSegmentId} segments`);
    const fails = checkThresholds(m, 'truth');  // DESIGN-sfm §8.3 against ground truth
    assert(fails.length === 0, `${tag}: ${fails.join(', ')}`);
    assert(l.medianRelRotDeg < 0.05, `${tag}, local ${l.medianRelRotDeg}`);
  });
}

// Critical motion: translation without rotation (a nadir mapping pass) leaves the focal
// unobservable from F. The mapper must say so and fall back, and a given HFOV must work.
const NADIR = { path: 'line', pitchDeg: 85, length: 60, numGround: 30000, numFar: 0 };

Deno.test('mapper, synthetic nadir pass: unobservable focal is detected and falls back', () => {
  const { scene, mapper, m } = mapSynthetic(NADIR);
  assert(mapper.stats.focalSource === 'fallback', `focal source ${mapper.stats.focalSource}, sharpness ${mapper.stats.focalSharpness}`);
  assert(m.numCommon === scene.truth.images.length, `${m.numCommon} registered`);
});

// Rotations stay looser over near-flat ground: small rotations and translations are nearly
// interchangeable there (bas-relief ambiguity).
Deno.test('mapper, synthetic nadir pass with the HFOV given', () => {
  const hfovDeg = 2 * Math.atan(960 / 1230) * 180 / Math.PI;
  const { m } = mapSynthetic(NADIR, { hfovDeg });
  const tag = `RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg, focal ${(100 * m.focalErr).toFixed(2)}%`;
  assert(m.focalErr < 0.01 && m.rmsCenterRel < 0.005 && m.medianRotDeg < 1.5, tag);
});

// Against COLMAP with COLMAP's own defects excluded (reference.js), at the strict §8.3 thresholds
Deno.test({ name: 'mapper, jardin5 (SFM_DUMP): every consistent COLMAP piece within the colmap thresholds', ignore: !SLOW || !DUMP || !HAS_REF, fn: async () => {
  const input = await loadInput();
  const mapper = new SequentialMapper(input, {});
  const model = mapper.run();
  const { ref, defects, local, fails } = await assessColmap(model, input);
  console.log(`  ${model.images.length}/${input.images.length} registered in ${mapper.nextSegmentId} segments, focal ${model.focal.toFixed(1)} vs ${ref.focal.toFixed(1)}, ` +
              `COLMAP defects excluded: ${defects.badFrames.size} frames, ${defects.breaks.size} breaks; local rotation ${local.medianRelRotDeg.toFixed(3)} deg`);
  assert(model.images.length >= 0.99 * input.images.length, 'registered');
  assert(Math.abs(model.focal / ref.focal - 1) < THRESHOLDS.colmap.focalErr, 'focal');
  assert(fails.length === 0, fails.join('; '));
} });
