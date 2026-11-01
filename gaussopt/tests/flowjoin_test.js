// Pieces joined across a turn in place by the optical flow (flowjoin.js, DESIGN-sfm §5.4.1): synthetic
// flights whose flow homographies are the true rotations, K R_k R_0^T K^-1 (what the frame analysis
// measures when the camera does not move).
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { reconstruct, mapSequential } from '../src/sfm/reconstruct.js';
import { SequentialMapper } from '../src/sfm/mapper.js';
import { flowRotation, joinByFlow } from '../src/sfm/flowjoin.js';
import { umeyama } from '../src/sfm/align.js';
import { poseCenter } from '../src/sfm/camera.js';
import { compareModels } from '../src/sfm/compare.js';
import { mul3, transpose3 } from '../src/sfm/linalg.js';
import { assert } from './util.js';
import { SfmSession } from '../src/sfm/session.js';
import { tablePairSource } from './session_util.js';

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

Deno.test('flow join: a segment holding pieces on both sides of another\'s, the turn between them joined', () => {
  // the two pieces of the turn, each moved onto the truth; the last frames of the later one handed to the
  // earlier one's segment (as a loop merge of a stretch seen again would): the segments interleave in time,
  // then the later one is moved off by a similarity for the join to undo
  const scene = turnInPlace(), { mapper } = mapSequential(scene.input, {});
  const segs = [...new Set(mapper.segment.filter((s, k) => mapper.poses[k]))];
  assert(segs.length === 2, `segments ${segs}`);
  for (const seg of segs) {
    const ks = [...mapper.segment.keys()].filter(k => mapper.poses[k] && mapper.segment[k] === seg);
    mapper.transformSegment(seg, umeyama(ks.map(k => poseCenter(mapper.poses[k])), ks.map(k => poseCenter(scene.truth.images[k]))));
  }
  const [early, late] = segs.sort((p, q) => mapper.segment.indexOf(p) - mapper.segment.indexOf(q)), n = mapper.n;
  for (let k = n - 5; k < n; k++) mapper.segment[k] = early;
  mapper.transformSegment(late, { s: 2.5, R: [0, -1, 0, 1, 0, 0, 0, 0, 1], t: [3, -1, 7] });
  const stats = joinByFlow(mapper);
  const m = compareModels(mapper.model(), scene.truth.model);
  const tag = `${JSON.stringify(stats.joins)}, RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg`;
  assert(stats.joins.length === 1 && stats.joins[0].from === late && new Set(mapper.segment.filter((s, k) => mapper.poses[k])).size === 1, tag);
  assert(m.rmsCenterRel < 0.005 && m.medianRotDeg < 0.5, tag);
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

Deno.test('flow join: the keyframes it places are rotation only, not weak, and repair leaves them', async () => {
  // the turn's middle frames matched to nothing: the join places them, with no points
  const scene = turnInPlace(), blind = (k) => k >= 23 && k <= 30;
  const input = { ...scene.input, pairs: scene.input.pairs.filter(p => !blind(p.i) && !blind(p.j)) };
  const s = new SfmSession(input, { pairSource: tablePairSource(input), loops: false });
  const r = await s.reconstruct();
  assert(r.stats.flowJoin.joins.length === 1 && r.stats.flowJoin.joins[0].carried === 8, JSON.stringify(r.stats.flowJoin.joins));
  const placed = (q) => q.frames.filter(f => blind(f.index));
  const tag = (q) => placed(q).map(f => `${f.index} ${f.score} ${f.reason}`).join(', ');
  const q = s.quality();
  assert(placed(q).every(f => f.score === 'amber' && /placed by the flow.*rotation only/.test(f.reason)), tag(q));
  const before = JSON.stringify(s.mapper.poses.filter((_, k) => blind(k)));
  await s.repair();
  assert(JSON.stringify(s.mapper.poses.filter((_, k) => blind(k))) === before, `repair moved them: ${tag(s.quality())}`);
  // and a saved project keeps them so (mapper.js toState / setState)
  const st = s.mapper.toState();
  s.mapper.byFlow.fill(0);
  s.mapper.setState(st);
  assert(placed(s.quality()).every(f => f.score === 'amber'), `after reload: ${tag(s.quality())}`);
});
