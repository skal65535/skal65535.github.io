// Reference defects (src/sfm/reference.js): a ground-truth model is corrupted the way COLMAP's
// jardin5 model is (a scale break, a weakly supported misregistered frame, a bent step); detection
// must find exactly those, and the consistent pieces must then compare as exact.
import { referenceDefects, comparePieces, exclusions } from '../src/sfm/reference.js';
import { compareLocal } from '../src/sfm/compare.js';
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { poseCenter, poseFromCenter, project } from '../src/sfm/camera.js';
import { mul3, expSO3 } from '../src/sfm/linalg.js';
import { median } from '../src/sfm/util.js';
import { assert, rng } from './util.js';

const scene = makeSyntheticScene({ numFrames: 40, maxFeatures: 1000, numGround: 8000, numBoxes: 8, numFar: 1000 });

// Reference model in COLMAP form (numObs, medianDepth) from poses
function asReference(images) {
  return {
    focal: scene.truth.model.focal,
    images: images.map((p, k) => ({
      name: p.name, R: p.R, t: p.t, numObs: scene.truth.images[k].pointIds.length,
      medianDepth: median(Array.from(scene.truth.images[k].pointIds, id => project(scene.truth.camera, p, scene.truth.points[id])[2])),
    })),
  };
}

Deno.test('referenceDefects finds scale breaks, weak frames and arbitrated steps; pieces are exact', () => {
  const truth = scene.truth.images.map(im => ({ name: im.name, R: im.R, t: im.t }));
  const bad = truth.map(im => ({ ...im }));
  // frames 20..: scaled 40x about frame 19's center (a scale break between 19 and 20)
  const c19 = poseCenter(truth[19]);
  for (let k = 20; k < 40; k++) bad[k] = { name: truth[k].name, ...poseFromCenter(truth[k].R, poseCenter(truth[k]).map((x, i) => c19[i] + 40 * (x - c19[i]))) };
  // frame 30: misregistered by 10 deg
  bad[30] = { name: truth[30].name, ...poseFromCenter(mul3(expSO3([0, 0.17, 0]), bad[30].R), poseCenter(bad[30])) };
  const ref = asReference(bad);
  ref.images[30].numObs = 20;  // ... from few observations
  // frame 10: a bent step (relative rotation off by 1 deg), well supported: only arbitration can tell
  ref.images[10] = { ...ref.images[10], ...poseFromCenter(mul3(expSO3([0.0175, 0, 0]), ref.images[10].R), poseCenter(ref.images[10])) };

  const est = { ...scene.truth.model, camera: scene.truth.camera, images: truth.map(im => ({ ...im, segment: 0 })) };
  const d = referenceDefects(ref, { est, input: scene.input, rng: rng(40) });
  assert(d.badFrames.size === 1 && d.badFrames.has(truth[30].name), `weak frames: ${[...d.badFrames.keys()]}`);
  assert(d.breaks.has(truth[20].name), `scale break before frame 20: ${[...d.breaks.keys()]}`);
  assert(d.breaks.has(truth[10].name) && d.breaks.has(truth[11].name), `bent frame 10 (both steps): ${[...d.breaks.entries()].join(' | ')}`);
  const pieces = comparePieces(est, ref, d).filter(p => !p.skipped);
  assert(pieces.length >= 2, `${pieces.length} pieces`);
  for (const p of pieces) assert(p.metrics.rmsCenterRel < 1e-9 && p.metrics.medianRotDeg < 1e-6, `piece ${p.first}..${p.last}`);
  const l = compareLocal(est, ref, { exclude: exclusions(d) });
  assert(l.p90RelRotDeg < 1e-6, 'local metric is exact once defects are excluded');
});
