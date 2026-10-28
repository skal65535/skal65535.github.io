// Pieces joined across a turn in place by the optical flow (flowjoin.js, DESIGN-sfm §5.4.1): synthetic
// flights whose flow homographies are the true rotations, K R_k R_0^T K^-1 (what the frame analysis
// measures when the camera does not move).
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { reconstruct } from '../src/sfm/reconstruct.js';
import { SequentialMapper } from '../src/sfm/mapper.js';
import { flowRotation } from '../src/sfm/flowjoin.js';
import { compareModels } from '../src/sfm/compare.js';
import { mul3, transpose3 } from '../src/sfm/linalg.js';
import { assert } from './util.js';

const BASE = { numFrames: 40, maxFeatures: 1000, numGround: 8000, numBoxes: 8, numFar: 1000 };

// A straight flight with a 180 deg yaw in place halfway: the map cannot follow the turn
function turnInPlace() {
  const scene = makeSyntheticScene({ ...BASE, path: 'line', spin: { at: 0.5, frames: 12, deg: 180 } });
  const [fx, fy, cx, cy] = scene.truth.camera.params;
  const K = [fx, 0, cx, 0, fy, cy, 0, 0, 1], Ki = [1 / fx, 0, -cx / fx, 0, 1 / fy, -cy / fy, 0, 0, 1], R0 = scene.truth.images[0].R;
  scene.input.images.forEach((im, k) => {
    im.time = +im.name.slice(1, 8) / 1000;
    im.flowH = mul3(K, mul3(mul3(scene.truth.images[k].R, transpose3(R0)), Ki));
    im.flowChain = 0;
  });
  return scene;
}

Deno.test('flow join: the pieces before and after a turn in place, apart without it, one map with it', () => {
  const scene = turnInPlace();
  const run = (flowJoin) => {
    const { model, stats } = reconstruct(scene.input, { loops: false, flowJoin });  // no loop closure: the flow alone joins them
    return { stats, m: compareModels(model, scene.truth.model) };
  };
  const tag = ({ stats, m }) => `${stats.numSegments} segments, ${m.numCommon} registered, RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg`;
  const apart = run(false), joined = run(true);
  assert(apart.stats.numSegments === 2, `without the join: ${tag(apart)}`);
  assert(joined.stats.numSegments === 1 && joined.stats.flowJoin.joins.length === 1, `with it: ${tag(joined)}`);
  assert(joined.m.numCommon === scene.input.images.length && joined.m.rmsCenterRel < 0.005 && joined.m.medianRotDeg < 0.5, `with it: ${tag(joined)}`);
});

Deno.test('flow rotation: a rotation homography gives its rotation, a homography of a moving camera none', () => {
  const scene = turnInPlace(), mapper = new SequentialMapper(scene.input, {});
  mapper.camera = scene.truth.camera;  // the true intrinsics: flowRotation reads the frames' camera
  const [a, b] = [10, 30], R = mul3(scene.truth.images[b].R, transpose3(scene.truth.images[a].R));
  const rot = flowRotation(mapper, a, b);
  assert(rot?.R && rot.spread < 1e-6 && R.every((x, i) => Math.abs(x - rot.R[i]) < 1e-6), `spread ${rot?.spread}`);
  // a plane's homography seen while moving: K (R + t n^T / d) K^-1 is no rotation
  const H = scene.input.images[b].flowH, [fx, , cx] = scene.truth.camera.params;
  scene.input.images[b].flowH = mul3([1, 0, 0.3 * fx, 0, 1, 0, 0, 0, 1 + 0.3 * cx / fx], H);
  const moved = flowRotation(mapper, a, b);
  assert(moved && moved.R === null && moved.spread > 0.05, `spread ${moved?.spread}`);
});
