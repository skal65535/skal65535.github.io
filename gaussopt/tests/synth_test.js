import { makeSyntheticScene } from '../src/sfm/synth.js';
import { project, poseCenter } from '../src/sfm/camera.js';
import { assert, assertClose } from './util.js';

const SMALL = { numFrames: 60, numGround: 4000, numBoxes: 4, numFar: 500 };

Deno.test('scenes are deterministic per seed', () => {
  const a = makeSyntheticScene({ ...SMALL, seed: 3 }), b = makeSyntheticScene({ ...SMALL, seed: 3 });
  const c = makeSyntheticScene({ ...SMALL, seed: 4 });
  assertClose(a.input.images[7].keypoints, b.input.images[7].keypoints, 0, 'same seed');
  assert(a.input.pairs.length === b.input.pairs.length, 'same pairs');
  assert(a.input.images[7].keypoints[0] !== c.input.images[7].keypoints[0], 'other seed differs');
});

Deno.test('noise-free keypoints reproject exactly from the ground truth', () => {
  const { input, truth } = makeSyntheticScene({ ...SMALL, noisePx: 0 });
  for (let k = 0; k < truth.images.length; k += 7) {
    const im = truth.images[k], kp = input.images[k].keypoints;
    for (let i = 0; i < im.pointIds.length; i++) {
      const [u, v, z] = project(truth.camera, im, truth.points[im.pointIds[i]]);
      assert(z > 0, 'in front');
      assertClose([kp[2 * i], kp[2 * i + 1]], [u, v], 2e-3, `frame ${k} keypoint ${i}`);  // f32 storage
    }
  }
});

Deno.test('matches link keypoints of the same point; outliers are injected at the requested rate', () => {
  const clean = makeSyntheticScene(SMALL);
  for (const { i, j, matches } of clean.input.pairs) {
    const pi = clean.truth.images[i].pointIds, pj = clean.truth.images[j].pointIds;
    for (let m = 0; m < matches.length; m += 2) assert(pi[matches[m]] === pj[matches[m + 1]], `pair ${i}-${j}`);
  }
  const dirty = makeSyntheticScene({ ...SMALL, outlierFrac: 0.2 });
  let bad = 0, total = 0;
  for (const { i, j, matches } of dirty.input.pairs) {
    const pi = dirty.truth.images[i].pointIds, pj = dirty.truth.images[j].pointIds;
    for (let m = 0; m < matches.length; m += 2, total++) bad += pi[matches[m]] !== pj[matches[m + 1]];
  }
  assertClose(bad / total, 0.2, 0.02, 'outlier rate');  // slots are drawn with replacement
});

Deno.test('orbit paths revisit (loop pairs), straight paths do not', () => {
  const orbit = makeSyntheticScene({ ...SMALL, turns: 1.25 });
  // A nadir mapping pass without the far ring: a forward-looking camera, or distant points, keep the
  // same content in view for most of a straight flight, which pairs far-apart frames without a revisit
  const line = makeSyntheticScene({ ...SMALL, path: 'line', numFar: 0, pitchDeg: 85, length: 120, numGround: 40000 });
  const loops = orbit.input.pairs.filter(p => p.type === 'loop');
  assert(loops.length > 20, `orbit loop pairs: ${loops.length}`);
  assert(loops.some(p => p.j - p.i > 40), 'loops close across the turn');
  assert(line.input.pairs.every(p => p.type === 'seq'), 'no loops on a straight flight');
  for (const s of [orbit, line]) {
    for (const im of s.input.images) {
      assert(im.keypoints.length / 2 >= 300 && im.keypoints.length / 2 <= 4096, `${im.keypoints.length / 2} keypoints`);
    }
  }
});

Deno.test('a spin segment keeps the camera center fixed while it yaws', () => {
  const { truth } = makeSyntheticScene({ ...SMALL, spin: { at: 0.5, frames: 8, deg: 90 } });
  assert(truth.images.length === SMALL.numFrames + 8, 'frames inserted');
  const at = Math.round(0.5 * (SMALL.numFrames - 1));
  const c0 = poseCenter(truth.images[at]);
  for (let k = at + 1; k <= at + 8; k++) assertClose(poseCenter(truth.images[k]), c0, 1e-9, `spin frame ${k}`);
  const names = truth.images.map(im => im.name);
  assert(names.every((n, k) => k === 0 || n > names[k - 1]), 'names sort in time order');
});
