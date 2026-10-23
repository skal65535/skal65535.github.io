// Regression tests of review fixes in src/sfm: adaptive RANSAC at tiny inlier ratios, registered
// count after repairs, the trainer's point filter, deterministic geometric loop candidates, matched
// pixels from either match layout.
import { ransac } from '../src/sfm/geometry.js';
import { SequentialMapper } from '../src/sfm/mapper.js';
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { datasetFromMapper } from '../src/sfm/export.js';
import { geometricCandidates } from '../src/sfm/loopdetect.js';
import { matchedPixels } from '../src/sfm/util.js';
import { assert, rng } from './util.js';

Deno.test('adaptive RANSAC keeps sampling when the all-inlier probability underflows', () => {
  let fits = 0;
  // 10 inliers of 2000 (w^8 ~ 4e-19: 1 - w^8 rounds to 1)
  const res = ransac({ n: 2000, sampleSize: 8, threshold: 1, rng: rng(1), maxIterations: 200,
                       fit: () => { fits++; return ['m']; }, error: (_, i) => (i < 10 ? 0 : 100) });
  assert(res.numInliers === 10 && fits >= 200, `${fits} fits`);
});

const scene = makeSyntheticScene({ numFrames: 16, maxFeatures: 800, numGround: 6000, numBoxes: 6, numFar: 500 });
const mapper = new SequentialMapper(scene.input, {});
mapper.run();

Deno.test('trainer points: >= 3 observations, each within 2 px', () => {
  const all = datasetFromMapper(mapper, { minObs: 2, maxReprojPx: Infinity, minAngleDeg: 0 });
  const ds = datasetFromMapper(mapper);
  assert(all.points.length === mapper.numPoints(), `${all.points.length} of ${mapper.numPoints()} unfiltered`);
  assert(ds.points.length > 1000 && ds.points.length < all.points.length, `${ds.points.length} of ${all.points.length} kept`);
  assert(ds.points.every(p => p.track.length >= 3 && p.error <= 2), 'filtered tracks');
  const refs = new Map(ds.images.map(im => [im.id, im.points2D.length]));
  assert(ds.points.every(p => p.track.every(([id, idx]) => idx < refs.get(id))), 'track indices into points2D');
});

Deno.test('registered count follows unregister', () => {
  const m = new SequentialMapper(scene.input, {});
  m.restore(mapper.snapshot());
  m.unregister([3, 4, 4]);
  assert(m.stats.registered === m.poses.filter(Boolean).length, `${m.stats.registered} registered`);
});

Deno.test('geometric loop candidates need an rng', () => {
  let threw = false;
  try { geometricCandidates(mapper, {}); } catch { threw = true; }
  assert(threw, 'no rng accepted');
  const a = geometricCandidates(mapper, { minGap: 4, rng: rng(3) }), b = geometricCandidates(mapper, { minGap: 4, rng: rng(3) });
  assert(JSON.stringify(a) === JSON.stringify(b), 'deterministic');
});

Deno.test('matchedPixels: interleaved matches and { a, b } index arrays agree', () => {
  const kpA = Float32Array.from({ length: 40 }, (_, i) => i), kpB = Float32Array.from({ length: 40 }, (_, i) => 100 + i);
  const matches = Uint32Array.of(0, 5, 3, 2, 19, 7);
  const [x1, x2] = matchedPixels(kpA, kpB, matches);
  const [y1, y2] = matchedPixels(kpA, kpB, { a: Int32Array.of(0, 3, 19), b: Int32Array.of(5, 2, 7) });
  assert(JSON.stringify([x1, x2]) === JSON.stringify([y1, y2]) && x2[1][0] === 104, JSON.stringify([x1, x2]));
});
