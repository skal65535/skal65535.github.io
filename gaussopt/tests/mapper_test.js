// Sequential mapper (DESIGN-sfm M1): synthetic scenes against exact ground truth, and (slow) the
// jardin5 COLMAP database against COLMAP's model, after excluding that model's own defects
// (scale breaks, misregistered frames): never match COLMAP where it is wrong.
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { SequentialMapper } from '../src/sfm/mapper.js';
import { reconstruct } from '../src/sfm/reconstruct.js';
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

// The fallback field of view is across the long side: a portrait frame (a phone video held upright,
// desk.mov) starts from the same focal as a landscape one, not 1.78x short of it
Deno.test('mapper, synthetic nadir pass in portrait: the fallback HFOV is across the long side', () => {
  const scene = makeSyntheticScene({ ...BASE, ...NADIR, width: 1080, height: 1920, params: [1230, 1230, 540, 960, 0, 0, 0, 0] });
  const mapper = new SequentialMapper(scene.input);
  const f = mapper.calibrateFocal(), expected = 960 / Math.tan(38 * Math.PI / 180);
  assert(mapper.stats.focalSource === 'fallback', `focal source ${mapper.stats.focalSource}, sharpness ${mapper.stats.focalSharpness}`);
  assert(Math.abs(f / expected - 1) < 1e-9, `fallback focal ${f.toFixed(1)} px, expected ${expected.toFixed(1)}`);
});

// Rotations stay looser over near-flat ground: small rotations and translations are nearly
// interchangeable there (bas-relief ambiguity).
Deno.test('mapper, synthetic nadir pass with the HFOV given', () => {
  const hfovDeg = 2 * Math.atan(960 / 1230) * 180 / Math.PI;
  const { m } = mapSynthetic(NADIR, { hfovDeg });
  const tag = `RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg, focal ${(100 * m.focalErr).toFixed(2)}%`;
  assert(m.focalErr < 0.01 && m.rmsCenterRel < 0.005 && m.medianRotDeg < 1.5, tag);
});

// A thin link (photo sets, DESIGN-sfm §13.2): frames 0-5 and 6-11 are joined by the pair 5-6
// alone, whose matches are on points frames 0-4 never see, but for 10. Frame 6 then has 10 2D-3D
// correspondences, too few for PnP: the map splits in two unless registerRelative places frame 6
// from the pair's essential matrix and those few correspondences. `wrong` more matches on points
// frames 0-4 see go to random keypoints of frame 6: with 35 correspondences PnP is tried, and fails.
function thinLink({ wrong = 0 } = {}) {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 12, turns: 0.5, overlap: 11, minLoopMatches: 1 });
  const ids = scene.truth.images.map(im => im.pointIds);
  const seenBefore = new Set(ids.slice(0, 5).flatMap(p => [...p]));
  const pairs = [];
  for (const p of scene.input.pairs) {
    if ((p.i < 6) === (p.j < 6)) { pairs.push(p); continue; }
    if (p.i !== 5 || p.j !== 6) continue;
    const m = [];
    let old = 0;
    for (let k = 0; k < p.matches.length; k += 2) {
      const known = seenBefore.has(ids[5][p.matches[k]]);
      if (!known || old++ < 10) m.push(p.matches[k], p.matches[k + 1]);
      else if (old <= 10 + wrong) m.push(p.matches[k], (p.matches[k + 1] * 7919 + 13) % ids[6].length);
    }
    pairs.push({ ...p, matches: Uint32Array.from(m) });
  }
  return { ...scene, input: { ...scene.input, pairs } };
}

for (const [relativeRegistration, wrong] of [[false, 0], [true, 0], [true, 25]]) {
  Deno.test(`mapper, synthetic thin link, relativeRegistration ${relativeRegistration}, ${wrong} wrong matches`, () => {
    const scene = thinLink({ wrong });
    const mapper = new SequentialMapper(scene.input, { relativeRegistration });
    const model = mapper.run(), m = compareModels(model, scene.truth.model);
    const tag = `${m.numCommon} registered, ${mapper.nextSegmentId} segments, RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg`;
    if (!relativeRegistration) {  // the case registerRelative exists for
      assert(mapper.nextSegmentId === 2, `without registerRelative the map should split: ${tag}`);
      return;
    }
    assert(m.numCommon === 12 && mapper.nextSegmentId === 1, tag);
    const fails = checkThresholds(m, 'truth');
    assert(fails.length === 0, `${tag}: ${fails.join(', ')}`);
  });
}

// regrowSegments: a map split in two (the thin link mapped without registerRelative) becomes one
// once the smaller segment's frames are offered to the larger one, registerRelative on
Deno.test('mapper, synthetic thin link: regrowSegments joins the two segments', () => {
  const scene = thinLink();
  const mapper = new SequentialMapper(scene.input, { relativeRegistration: false });
  mapper.run();
  assert(new Set(mapper.segment).size === 2, `two segments first: ${[...new Set(mapper.segment)]}`);
  mapper.o.relativeRegistration = true;
  mapper.regrowSegments();
  const m = compareModels(mapper.model(), scene.truth.model);
  const tag = `${m.numCommon} registered, segments ${[...new Set(mapper.segment)]}, RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg`;
  assert(m.numCommon === 12 && new Set(mapper.segment).size === 1, tag);
  const fails = checkThresholds(m, 'truth');
  assert(fails.length === 0, `${tag}: ${fails.join(', ')}`);
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

// Two lenses (DESIGN-sfm §13 P8): an orbit where one photo in three is taken with a wide lens
// (f 700, more distortion), each camera model starting from a focal 3% low (EXIF on the Pixel 10,
// §13.2). With a camera per lens every frame registers in one segment, within the truth thresholds,
// each focal within 1%; the same photos as one camera do not.
function twoLenses() {
  return makeSyntheticScene({ ...BASE, numFrames: 30, turns: 0.6, overlap: 6,
    cameras: [[1230, 1230, 960, 540, 0.005, -0.002, 0, 0], [700, 700, 960, 540, -0.03, 0.004, 0, 0]],
    cameraOf: (k) => (k % 3 === 2 ? 1 : 0), priorFocals: [1230 * 0.97, 700 * 0.97] });
}

Deno.test('mapper, synthetic two lenses: a camera per lens', () => {
  const scene = twoLenses();
  const mapper = new SequentialMapper(scene.input, { refine: ['f', 'k1', 'k2'] });
  const model = mapper.run(), m = compareModels(model, scene.truth.model);
  const focals = mapper.cameras.map(c => c.params[0]), err = focals.map((f, k) => f / scene.truth.cameras[k].params[0] - 1);
  const tag = `${m.numCommon} registered, ${mapper.nextSegmentId} segments, RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg, focals ${focals.map(f => f.toFixed(1))}`;
  assert(m.numCommon === 30 && mapper.nextSegmentId === 1, tag);
  const fails = checkThresholds(m, 'truth');
  assert(fails.length === 0, `${tag}: ${fails.join(', ')}`);
  assert(err.every(e => Math.abs(e) < 0.01), `${tag}: focal errors ${err.map(e => (100 * e).toFixed(2))}%`);
});

// regrowSegments with no segment at all (no pair initializes: out_scene2, out_scene3 with every lens)
// ends with nothing registered instead of throwing
Deno.test('mapper: regrowSegments with no segment registers nothing, does not throw', () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 6 });
  const mapper = new SequentialMapper({ ...scene.input, pairs: [] }, { regrowSegments: true });
  const model = mapper.run();
  assert(model.images.length === 0 && mapper.nextSegmentId === 0, `${model.images.length} registered`);
});

// After the regrow, the largest segment retries the frames still unregistered (outside-mixed: a
// frame registered by a regrow that was undone was lost): one segment and a frame taken out of it
Deno.test('mapper: regrowSegments ends with the unregistered frames tried on the largest segment', () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 12, turns: 0.5, overlap: 11 });
  const mapper = new SequentialMapper(scene.input, { relativeRegistration: true });
  mapper.run();
  assert(mapper.poses.every(Boolean), 'all registered first');
  mapper.unregister([6]);
  mapper.regrowSegments();
  const m = compareModels(mapper.model(), scene.truth.model);
  assert(!!mapper.poses[6] && m.numCommon === 12, `frame 6 ${mapper.poses[6] ? 'back' : 'still out'}, ${m.numCommon} registered`);
  assert(checkThresholds(m, 'truth').length === 0, checkThresholds(m, 'truth').join(', '));
});

// The final sweeps fit the lens to the whole sequence, not to its last windows (desk.mov: k1 = 0.02 over
// the first half, 0.28 over the frames after the whip pans, 0.29 exported). A forward flight whose last 9
// of 30 frames are seen through k1 = 0.1, mapped as one camera: the lens of the other 21 (k1 = 0) wins.
// Refined window by window: k1 = 0.061.
Deno.test('reconstruct, synthetic: the lens is fitted to the whole sequence, not its last windows', () => {
  const scene = makeSyntheticScene({ ...BASE, path: 'line', pitchDeg: 45,
    cameras: [[1230, 1230, 960, 540, 0, 0, 0, 0], [1230, 1230, 960, 540, 0.1, 0, 0, 0]], cameraOf: (k) => (k >= 21 ? 1 : 0) });
  const { camera, images, pairs } = scene.input;
  const { mapper } = reconstruct({ camera, images: images.map(({ camera, ...im }) => im), pairs }, { sweeps: 2 });
  const k1 = mapper.camera.params[4];
  assert(mapper.poses.every(Boolean) && Math.abs(k1) < 0.03, `${mapper.poses.filter(Boolean).length} registered, k1 ${k1.toFixed(4)}`);
});

// A lens with 3 photos, its prior ratio 7% off (the Pixel 10 ultrawide's EXIF, DESIGN-sfm §13 P8): kept
// while mapping (fewer than minCameraFrames), fitted to the whole map in the final sweeps (minCameraFramesSweeps)
Deno.test('reconstruct, synthetic: a lens with 3 photos is fitted in the final sweeps', () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 30, turns: 0.6, overlap: 8,
    cameras: [[1230, 1230, 960, 540, 0.005, -0.002, 0, 0], [700, 700, 960, 540, -0.03, 0.004, 0, 0]],
    cameraOf: (k) => ([5, 15, 25].includes(k) ? 1 : 0), priorFocals: [1230 * 0.97, 700 * 0.97 * 1.07] });
  // a photo set's window spans every photo (worker.js run): a window BA sees the lens's 3 photos together
  const { mapper } = reconstruct(scene.input, { sweeps: 2, refine: ['f', 'k1', 'k2'], window: 29 });
  const err = mapper.cameras.map((c, k) => c.params[0] / scene.truth.cameras[k].params[0] - 1);
  const reg = mapper.poses.filter(Boolean).length;
  assert(reg === 30 && Math.abs(err[0]) < 0.01 && Math.abs(err[1]) < 0.02, `${reg} registered, focal errors ${err.map(e => (100 * e).toFixed(2))}%`);
});
