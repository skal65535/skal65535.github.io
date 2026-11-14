// Sequential mapper (DESIGN-sfm M1): synthetic scenes against exact ground truth, and (slow) a
// video's COLMAP database (SFM_DUMP) against COLMAP's model, after excluding that model's own
// defects (scale breaks, misregistered frames): never match COLMAP where it is wrong.
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { SequentialMapper } from '../src/sfm/mapper.js';
import { reconstruct } from '../src/sfm/reconstruct.js';
import { compareModels, compareLocal, checkThresholds, THRESHOLDS } from '../src/sfm/compare.js';
import { assert, SLOW, DUMP, HAS_REF, loadInput, assessColmap } from './util.js';

const BASE = { numFrames: 30, maxFeatures: 1000, numGround: 8000, numBoxes: 8, numFar: 1000 };

async function mapSynthetic(options, mapperOptions = {}) {
  const scene = makeSyntheticScene({ ...BASE, ...options });
  const mapper = new SequentialMapper(scene.input, mapperOptions);
  const model = await mapper.run();
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
  Deno.test(`mapper, synthetic: ${name}`, async () => {
    const { scene, mapper, m, l } = await mapSynthetic(options);
    const tag = `${name}: RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg, focal ${(100 * m.focalErr).toFixed(2)}%`;
    assert(m.numCommon === scene.truth.images.length && mapper.nextSegmentId === 1, `${tag}: ${m.numCommon} registered, ${mapper.nextSegmentId} segments`);
    const fails = checkThresholds(m, 'truth');  // DESIGN-sfm §8.3 against ground truth
    assert(fails.length === 0, `${tag}: ${fails.join(', ')}`);
    assert(l.medianRelRotDeg < 0.05, `${tag}, local ${l.medianRelRotDeg}`);
  });
}

// A lens of gare2's tangential distortion (COLMAP's p2 there: -0.020), its focal observed by F: k2, p1, p2 refined in every
// window find it; with f and k1 refined over the first 50 frames only (the old policy), the path is 1.5% and 3.4 deg off
Deno.test('reconstruct, synthetic tangential lens: the intrinsics, p1 and p2 among them, refined in every window', async () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 60, params: [1230, 1230, 960, 540, -0.004, 0.002, 0.003, -0.02] });
  const { mapper, model } = await reconstruct(scene.input, {});
  const m = compareModels(model, scene.truth.model), [, , , , , , p1, p2] = mapper.camera.params;
  const tag = `RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg, focal ${(100 * m.focalErr).toFixed(2)}%, p1 ${p1}, p2 ${p2}`;
  assert(mapper.stats.focalSource === 'calibrated' && m.numCommon === 60, `${mapper.stats.focalSource}, ${m.numCommon} registered`);
  assert(m.rmsCenterRel < 0.001 && m.medianRotDeg < 0.05 && m.focalErr < 0.002 && Math.abs(p2 + 0.02) < 5e-4 && Math.abs(p1 - 0.003) < 5e-4, tag);
});

// Critical motion: translation without rotation (a nadir mapping pass) leaves the focal
// unobservable from F. The mapper must say so and fall back, and a given HFOV must work.
const NADIR = { path: 'line', pitchDeg: 85, length: 60, numGround: 30000, numFar: 0 };

Deno.test('mapper, synthetic nadir pass: unobservable focal is detected and falls back', async () => {
  const { scene, mapper, m } = await mapSynthetic(NADIR);
  assert(mapper.stats.focalSource === 'fallback', `focal source ${mapper.stats.focalSource}, sharpness ${mapper.stats.focalSharpness}`);
  assert(m.numCommon === scene.truth.images.length, `${m.numCommon} registered`);
});

// The fallback field of view is across the long side: a portrait frame (a phone video held upright)
// starts from the same focal as a landscape one, not 1.78x short of it
Deno.test('mapper, synthetic nadir pass in portrait: the fallback HFOV is across the long side', () => {
  const scene = makeSyntheticScene({ ...BASE, ...NADIR, width: 1080, height: 1920, params: [1230, 1230, 540, 960, 0, 0, 0, 0] });
  const mapper = new SequentialMapper(scene.input);
  const f = mapper.calibrateFocal(), expected = 960 / Math.tan(38 * Math.PI / 180);
  assert(mapper.stats.focalSource === 'fallback', `focal source ${mapper.stats.focalSource}, sharpness ${mapper.stats.focalSharpness}`);
  assert(Math.abs(f / expected - 1) < 1e-9, `fallback focal ${f.toFixed(1)} px, expected ${expected.toFixed(1)}`);
});

// Rotations stay looser over near-flat ground: small rotations and translations are nearly
// interchangeable there (bas-relief ambiguity).
Deno.test('mapper, synthetic nadir pass with the HFOV given', async () => {
  const hfovDeg = 2 * Math.atan(960 / 1230) * 180 / Math.PI;
  const { m } = await mapSynthetic(NADIR, { hfovDeg });
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
  Deno.test(`mapper, synthetic thin link, relativeRegistration ${relativeRegistration}, ${wrong} wrong matches`, async () => {
    const scene = thinLink({ wrong });
    const mapper = new SequentialMapper(scene.input, { relativeRegistration });
    const model = await mapper.run(), m = compareModels(model, scene.truth.model);
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
Deno.test('mapper, synthetic thin link: regrowSegments joins the two segments', async () => {
  const scene = thinLink();
  const mapper = new SequentialMapper(scene.input, { relativeRegistration: false });
  await mapper.run();
  assert(new Set(mapper.segment).size === 2, `two segments first: ${[...new Set(mapper.segment)]}`);
  mapper.o.relativeRegistration = true;
  await mapper.regrowSegments();
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
  const model = await mapper.run();
  const { ref, defects, local, fails } = await assessColmap(model, input);
  console.log(`  ${model.images.length}/${input.images.length} registered in ${mapper.nextSegmentId} segments, focal ${model.focal.toFixed(1)} vs ${ref.focal.toFixed(1)}, ` +
              `COLMAP defects excluded: ${defects.badFrames.size} frames, ${defects.breaks.size} breaks; local rotation ${local.medianRelRotDeg.toFixed(3)} deg`);
  assert(model.images.length >= 0.99 * input.images.length, 'registered');
  assert(Math.abs(model.focal / ref.focal - 1) < THRESHOLDS.colmap.focalErr, 'focal');
  assert(fails.length === 0, fails.join('; '));
} });

// Two lenses (DESIGN-sfm §13 P8): an orbit where one photo in three is taken with a wide lens
// (f 700, more distortion), each camera model starting from a focal 3% low (as a phone's EXIF,
// §13.2). With a camera per lens every frame registers in one segment, within the truth thresholds,
// each focal within 1%; the same photos as one camera do not.
function twoLenses() {
  return makeSyntheticScene({ ...BASE, numFrames: 30, turns: 0.6, overlap: 6,
    cameras: [[1230, 1230, 960, 540, 0.005, -0.002, 0, 0], [700, 700, 960, 540, -0.03, 0.004, 0, 0]],
    cameraOf: (k) => (k % 3 === 2 ? 1 : 0), priorFocals: [1230 * 0.97, 700 * 0.97] });
}

Deno.test('mapper, synthetic two lenses: a camera per lens', async () => {
  const scene = twoLenses();
  const mapper = new SequentialMapper(scene.input, { refine: ['f', 'k1', 'k2'] });
  const model = await mapper.run(), m = compareModels(model, scene.truth.model);
  const focals = mapper.cameras.map(c => c.params[0]), err = focals.map((f, k) => f / scene.truth.cameras[k].params[0] - 1);
  const tag = `${m.numCommon} registered, ${mapper.nextSegmentId} segments, RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg, focals ${focals.map(f => f.toFixed(1))}`;
  assert(m.numCommon === 30 && mapper.nextSegmentId === 1, tag);
  const fails = checkThresholds(m, 'truth');
  assert(fails.length === 0, `${tag}: ${fails.join(', ')}`);
  assert(err.every(e => Math.abs(e) < 0.01), `${tag}: focal errors ${err.map(e => (100 * e).toFixed(2))}%`);
});

// regrowSegments with no segment at all (no pair initializes) ends with nothing registered instead of throwing
// A camera turning in place (a shaky shot): no pair gives an initial pair. Each pair's relative pose is computed once
// however many angles and segments are tried, the angles are tried once each, and the search reports its progress
Deno.test('mapper: an initial pair search over pairs that only rotate tries each pair once, with its progress', async () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 1, spin: { at: 0, frames: 12, deg: 40 } });
  const seen = [];
  const mapper = new SequentialMapper(scene.input, { progress: (stage, done, total) => { if (stage === 'init') seen.push([done, total]); } });
  const model = await mapper.run();
  const pairs = mapper.adj.flatMap((list, i) => list.filter(e => e.other > i));
  const strong = pairs.filter(e => e.a.length >= 100).length;
  assert(model.images.length === 0 && strong >= 20, `${model.images.length} registered, ${strong} pairs of 100 matches`);
  assert(mapper.stats.initTried === strong, `${mapper.stats.initTried} relative poses for ${strong} pairs`);
  const angles = mapper.initAngles();
  assert(angles.length === new Set(angles).size && angles.length === 2, `angles ${angles}`);
  assert(seen.length >= angles.length && seen.every(([d, t]) => t === pairs.length && d < t), `progress ${JSON.stringify(seen)}`);
});

Deno.test('mapper: regrowSegments with no segment registers nothing, does not throw', async () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 6 });
  const mapper = new SequentialMapper({ ...scene.input, pairs: [] }, { regrowSegments: true });
  const model = await mapper.run();
  assert(model.images.length === 0 && mapper.nextSegmentId === 0, `${model.images.length} registered`);
});

// After the regrow, the largest segment retries the frames still unregistered (a frame registered
// by a regrow that was undone is not lost): one segment and a frame taken out of it
Deno.test('mapper: regrowSegments ends with the unregistered frames tried on the largest segment', async () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 12, turns: 0.5, overlap: 11 });
  const mapper = new SequentialMapper(scene.input, { relativeRegistration: true });
  await mapper.run();
  assert(mapper.poses.every(Boolean), 'all registered first');
  mapper.unregister([6]);
  await mapper.regrowSegments();
  const m = compareModels(mapper.model(), scene.truth.model);
  assert(!!mapper.poses[6] && m.numCommon === 12, `frame 6 ${mapper.poses[6] ? 'back' : 'still out'}, ${m.numCommon} registered`);
  assert(checkThresholds(m, 'truth').length === 0, checkThresholds(m, 'truth').join(', '));
});

// The final sweeps fit the lens to the whole sequence, not to its last windows. A forward flight whose
// last 9 of 30 frames are seen through k1 = 0.1, mapped as one camera: the lens of the other 21 (k1 = 0)
// wins (refined window by window, k1 ends near 0.06).
Deno.test('reconstruct, synthetic: the lens is fitted to the whole sequence, not its last windows', async () => {
  const scene = makeSyntheticScene({ ...BASE, path: 'line', pitchDeg: 45,
    cameras: [[1230, 1230, 960, 540, 0, 0, 0, 0], [1230, 1230, 960, 540, 0.1, 0, 0, 0]], cameraOf: (k) => (k >= 21 ? 1 : 0) });
  const { camera, images, pairs } = scene.input;
  const { mapper } = await reconstruct({ camera, images: images.map(({ camera, ...im }) => im), pairs }, { sweeps: 2 });
  const k1 = mapper.camera.params[4];
  assert(mapper.poses.every(Boolean) && Math.abs(k1) < 0.03, `${mapper.poses.filter(Boolean).length} registered, k1 ${k1.toFixed(4)}`);
});

// A lens with 3 photos, its prior ratio 7% off (as a phone ultrawide's EXIF, DESIGN-sfm §13 P8): kept
// while mapping (fewer than minCameraFrames), fitted to the whole map in the final sweeps (minCameraFramesSweeps)
Deno.test('reconstruct, synthetic: a lens with 3 photos is fitted in the final sweeps', async () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 30, turns: 0.6, overlap: 8,
    cameras: [[1230, 1230, 960, 540, 0.005, -0.002, 0, 0], [700, 700, 960, 540, -0.03, 0.004, 0, 0]],
    cameraOf: (k) => ([5, 15, 25].includes(k) ? 1 : 0), priorFocals: [1230 * 0.97, 700 * 0.97 * 1.07] });
  // a photo set's window spans every photo (worker.js run): a window BA sees the lens's 3 photos together
  const { mapper } = await reconstruct(scene.input, { sweeps: 2, refine: ['f', 'k1', 'k2'], window: 29 });
  const err = mapper.cameras.map((c, k) => c.params[0] / scene.truth.cameras[k].params[0] - 1);
  const reg = mapper.poses.filter(Boolean).length;
  assert(reg === 30 && Math.abs(err[0]) < 0.01 && Math.abs(err[1]) < 0.02, `${reg} registered, focal errors ${err.map(e => (100 * e).toFixed(2))}%`);
});

// A video's zooms (DESIGN-sfm §5.1, frames.js zoomEvents): an orbit filmed at f 1230, zoomed in to
// 2400 for frames 12-17, then back out to 1230; the worker makes each stretch between zooms a camera
// (here 0, 1, 2) whose prior is the image scale across the zooms, which walking also changes: here
// 0.5 for the stretch back at the start's focal. scanFocal finds camera 1's focal from its first
// frame, and camera 2 joins camera 0.
function zoomedOrbit() {
  const lens = [960, 540, 0.005, -0.002, 0, 0];
  return makeSyntheticScene({ ...BASE, numFrames: 30, turns: 0.6, overlap: 6,
    cameras: [[1230, 1230, ...lens], [2400, 2400, ...lens], [1230, 1230, ...lens]],
    cameraOf: (k) => (k < 12 ? 0 : k < 18 ? 1 : 2), priorFocals: [1230, 1230 * 1.84, 1230 * 0.5] });
}

Deno.test('mapper, synthetic zooms: a scanned focal per zoom, a camera back at an earlier focal joins it', async () => {
  const scene = zoomedOrbit();
  const mapper = new SequentialMapper(scene.input, { scanFocal: true });
  const model = await mapper.run(), m = compareModels(model, scene.truth.model);
  const err = (k) => mapper.cameras[k].params[0] / scene.truth.cameras[k].params[0] - 1;
  const tag = `${m.numCommon} registered, ${mapper.nextSegmentId} segments, cameras of frames ${[...new Set(mapper.camOf)]}, ` +
              `focal errors ${[0, 1].map(k => (100 * err(k)).toFixed(2))}%, RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg`;
  assert(m.numCommon === 30 && mapper.nextSegmentId === 1, tag);
  assert(mapper.camOf.slice(18).every(c => c === 0) && mapper.camOf.slice(12, 18).every(c => c === 1), tag);
  assert(Math.abs(err(0)) < 0.01 && Math.abs(err(1)) < 0.01, tag);
  const fails = checkThresholds(m, 'truth');
  assert(fails.length === 0, `${tag}: ${fails.join(', ')}`);
});

// Track triangulation: a slow turn (the camera moving on 3% of a step a frame) matched only to the next frame: no pair
// of the turn is wide enough for a point (minAngleDeg), its tracks across it are. Without triangulateTrack, no point
// seen by the turn's frames only (gare2's pan: its pavement)
Deno.test('mapper, synthetic slow turn with neighbour pairs: points of the turn triangulated from their tracks', async () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 40, overlap: 1, spin: { at: 0.5, frames: 20, deg: 60, step: 0.05 } });
  const mapper = new SequentialMapper(scene.input);
  const model = await mapper.run(), m = compareModels(model, scene.truth.model);
  const turn = (k) => k >= 20 && k < 40;  // the 20 frames inserted at the path's middle
  const own = mapper.points.filter(P => P && [...P.obs.keys()].every(turn));
  const worst = Math.max(...own.flatMap(P => [...P.obs].map(([img, kp]) => mapper.reprojErr(img, kp, P.X))));
  const tag = `${own.length} points of the turn's frames only (worst ${worst.toFixed(2)} px), ${m.numCommon} registered, RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg`;
  assert(own.length >= 30 && own.every(P => P.obs.size >= 3) && worst <= mapper.o.maxReproj, tag);
  assert(m.numCommon === 60 && checkThresholds(m, 'truth').length === 0, tag);
});

// The same slow turn matched to 3 frames on: a keypoint has several neighbours too narrow for it, its track walked once
// per triangulateNew though, not once per neighbour, and the turn's points still made
Deno.test('mapper, synthetic slow turn with 3 neighbours: a failed track walked once per triangulateNew', async () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 40, overlap: 3, spin: { at: 0.5, frames: 20, deg: 60, step: 0.05 } });
  const mapper = new SequentialMapper(scene.input);
  const tn = mapper.triangulateNew, tt = mapper.triangulateTrack;
  let walks = null, repeats = 0, calls = 0;
  mapper.triangulateNew = function (img) { walks = new Map(); try { return tn.call(this, img); } finally { walks = null; } };
  mapper.triangulateTrack = function (img, a, minAngle) {
    calls++;
    if (walks) { const key = `${img},${a}`, n = (walks.get(key) ?? 0) + 1; walks.set(key, n); if (n > 1) repeats++; }
    return tt.call(this, img, a, minAngle);
  };
  const model = await mapper.run(), m = compareModels(model, scene.truth.model);
  const turn = (k) => k >= 20 && k < 40;
  const own = mapper.points.filter(P => P && [...P.obs.keys()].every(turn));
  const tag = `${calls} track walks, ${repeats} repeated within a triangulateNew, ${own.length} points of the turn's frames only, ` +
              `${m.numCommon} registered, RMS ${(100 * m.rmsCenterRel).toFixed(3)}%`;
  assert(calls > 0 && repeats === 0, tag);
  assert(own.length >= 30 && m.numCommon === 60 && checkThresholds(m, 'truth').length === 0, tag);
});

// A pass over many frames (a global BA's retriangulation, extendTracks): each frame pair done once, from its first
// frame's side; a pass again with the same pairs done does nothing
Deno.test('mapper, synthetic slow turn: triangulateNew with a pass\'s pairs does each frame pair once', async () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 40, overlap: 3, spin: { at: 0.5, frames: 20, deg: 60, step: 0.05 } });
  const mapper = new SequentialMapper(scene.input);
  await mapper.run();
  const reg = [...mapper.poses.keys()].filter(k => mapper.poses[k]), pairs = new Set();
  for (const k of reg) for (const e of mapper.adj[k]) if (mapper.poses[e.other] && mapper.segment[e.other] === mapper.segment[k]) pairs.add(Math.min(k, e.other) * mapper.n + Math.max(k, e.other));
  const seen = new Set();
  for (const k of reg) mapper.triangulateNew(k, seen);
  let reproj = 0, walks = 0;
  const re = mapper.reprojErr, tt = mapper.triangulateTrack;
  mapper.reprojErr = function (...a) { reproj++; return re.apply(this, a); };
  mapper.triangulateTrack = function (...a) { walks++; return tt.apply(this, a); };
  let created = 0;
  for (const k of reg) created += mapper.triangulateNew(k, seen);
  const tag = `${pairs.size} frame pairs, ${seen.size} done; again: ${created} points, ${reproj} reprojections, ${walks} track walks`;
  assert(pairs.size > reg.length && seen.size === pairs.size, tag);
  assert(created === 0 && reproj === 0 && walks === 0, tag);
});

// The pass with a minRatio (COLMAP's re_min_ratio): only the frame pairs whose matches are under that much one point already
Deno.test('mapper, synthetic slow turn: triangulateNew with a minRatio does only the pairs under it', async () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 40, overlap: 3, spin: { at: 0.5, frames: 20, deg: 60, step: 0.05 } });
  const mapper = new SequentialMapper(scene.input);
  await mapper.run();
  const reg = [...mapper.poses.keys()].filter(k => mapper.poses[k]), all = new Set(), under = new Set();
  for (const k of reg) for (const e of mapper.adj[k]) {
    if (k > e.other || !mapper.poses[e.other] || mapper.segment[e.other] !== mapper.segment[k]) continue;
    let n = 0;
    for (let m = 0; m < e.a.length; m++) { const p = mapper.kpPoint[k][e.a[m]]; if (p >= 0 && p === mapper.kpPoint[e.other][e.b[m]]) n++; }
    all.add(k * mapper.n + e.other);
    if (n < 0.9 * e.a.length) under.add(k * mapper.n + e.other);
  }
  const seen = new Set();
  for (const k of reg) mapper.triangulateNew(k, seen, 0.9);
  const tag = `${all.size} frame pairs, ${under.size} under 0.9 triangulated, ${seen.size} done`;
  assert(under.size > 0 && under.size < all.size, tag);
  assert(seen.size === under.size && [...seen].every(p => under.has(p)), tag);
});

// Global BAs: each time a segment has grown by globalBARatio, from globalBAMinFrames on (30 frames: at 20, 22, 25, 28)
Deno.test('mapper, synthetic: a global BA each time the segment grows by 10%, from 20 frames', async () => {
  const { mapper, m } = await mapSynthetic({});
  const sizes = (mapper.stats.globalBAs ?? []).map(g => g.frames);
  assert(JSON.stringify(sizes) === '[20,22,25,28]' && mapper.stats.globalAt[0] === 28, `global BAs at ${JSON.stringify(sizes)}`);
  assert(checkThresholds(m, 'truth').length === 0, `RMS ${(100 * m.rmsCenterRel).toFixed(3)}%`);
  const { mapper: off } = await mapSynthetic({}, { globalBARatio: null });
  assert(!off.stats.globalBAs, 'globalBARatio null: none');
});
