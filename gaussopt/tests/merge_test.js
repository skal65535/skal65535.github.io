// loop.js mergeSegments: two segments that meet in time are merged by their pairs across the joint, not by a loop
// between them. The case of gare2 at 10 s: a weak joint (a few pairs), many loop pairs between the first segment's
// start and the second's end, that end drifted in scale; the loop's G put the second segment's middle 1/0.7 off. Then
// the frames around the joint retriangulated: a frame there left with few points sees the other side's.
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { SequentialMapper } from '../src/sfm/mapper.js';
import { mergeSegments } from '../src/sfm/loop.js';
import { joinPieces } from '../src/sfm/joinpieces.js';
import { compareModels } from '../src/sfm/compare.js';
import { poseCenter } from '../src/sfm/camera.js';
import { compose, inverse, apply, fromPose, toPose } from '../src/sfm/sim3.js';
import { makeRng } from '../src/math.js';
import { assert, ORBIT } from './util.js';

const JOINT = 20, END = 40, WEAK = JOINT - 2;  // segment 1: frames JOINT.., its end END.. (the revisit of frames 0..)

// The mapped orbit split at JOINT: a point seen on both sides becomes two, one a side (each with 2 views at least)
function split(mapper) {
  for (let k = JOINT; k < mapper.n; k++) if (mapper.poses[k]) mapper.segment[k] = 1;
  const n = mapper.points.length;
  for (let pid = 0; pid < n; pid++) {
    const P = mapper.points[pid];
    if (!P) continue;
    const late = [...P.obs].filter(([img]) => img >= JOINT);
    if (!late.length || late.length === P.obs.size) continue;
    for (const [img, kp] of late) { P.obs.delete(img); mapper.kpPoint[img][kp] = -1; }
    if (late.length >= 2) mapper.addPoint([...P.X], late);
    if (P.obs.size < 2) { for (const [img, kp] of P.obs) mapper.kpPoint[img][kp] = -1; mapper.points[pid] = null; }
  }
  mapper.nextSegmentId = 2;
}

// Frames END.. and the points only they see moved by D (a drift of segment 1's far end)
function drift(mapper, D) {
  const Dinv = inverse(D), moved = new Set();
  for (let k = END; k < mapper.n; k++) {
    mapper.poses[k] = toPose(compose(fromPose(mapper.poses[k]), Dinv));
    for (const pid of mapper.kpPoint[k]) {
      const P = mapper.points[pid];
      if (pid < 0 || moved.has(pid) || !P || [...P.obs.keys()].some(img => img < END)) continue;
      moved.add(pid);
      P.X = apply(D, P.X);
    }
  }
}

Deno.test('mergeSegments: segments meeting in time merged across their joint, not by a loop between them', async () => {
  const scene = makeSyntheticScene({ ...ORBIT });
  const mapper = new SequentialMapper(scene.input);
  await mapper.run();
  assert(mapper.nextSegmentId === 1 && mapper.poses.every(Boolean), `one segment first: ${mapper.nextSegmentId}`);
  split(mapper);
  let keep = 3;
  mapper.transformSegment(1, { s: 2, R: [0, -1, 0, 1, 0, 0, 0, 0, 1], t: [3, 0, 1] });
  const c = poseCenter(mapper.poses[END]);
  drift(mapper, { s: 0.7, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: c.map(x => 0.3 * x) });  // 0.7 about frame END's centre
  // the frame before the joint's own left with 3 observations
  mapper.kpPoint[WEAK].forEach((pid, kp) => {
    const P = pid >= 0 && mapper.points[pid];
    if (!P || --keep >= 0) return;
    P.obs.delete(WEAK); mapper.kpPoint[WEAK][kp] = -1;
    if (P.obs.size < 2) { for (const [img, b] of P.obs) mapper.kpPoint[img][b] = -1; mapper.points[pid] = null; }
  });
  // of the pairs across the joint: neighbours (a weak joint) and the revisit's (frames 0.. with END..)
  const pairs = scene.input.pairs.filter(p => (p.i < JOINT) === (p.j < JOINT) || Math.abs(p.j - p.i) <= 1 || Math.max(p.i, p.j) >= END);
  const loops = pairs.filter(p => p.i < JOINT && p.j >= END).length;
  const log = [], res = await mergeSegments(mapper, pairs, { rng: makeRng(3), log: (s) => log.push(s) });
  const model = mapper.model();
  const middle = { ...model, images: model.images.filter(im => im.index < END) };
  const m = compareModels(middle, scene.truth.model);
  const seen = (k) => mapper.kpPoint[k].filter(pid => pid >= 0 && mapper.points[pid]).length;
  const tag = `${log.join('; ')} (${loops} loop pairs); frames 0..${END - 1}: RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg; frame ${WEAK} sees ${seen(WEAK)} points`;
  assert(res.merges === 1 && new Set(mapper.segment).size === 1 && loops >= 10, tag);
  assert(m.numCommon === END && m.rmsCenterRel < 0.002 && m.medianRotDeg < 0.05, tag);
  assert(/across their joint/.test(log[0]), tag);
  // the weak frame before the joint sees the points around it again (gare2's 45, 46: 16 and 1, then turned 27 deg)
  assert(seen(WEAK) >= 100, tag);
});

// joinPieces (the mapper's option, photo sets): two segments that share no 3D point a Sim(3) could be fitted on
// (ransacSim3 off here: minInliers Infinity) joined by the global positioning of joinpieces.js, back where they were
for (const on of [false, true]) {
  Deno.test(`mergeSegments: no 3D-3D correspondences, joinPieces ${on}`, async () => {
    const scene = makeSyntheticScene({ ...ORBIT });
    const mapper = new SequentialMapper(scene.input);
    await mapper.run();
    split(mapper);
    mapper.transformSegment(1, { s: 2, R: [0, -1, 0, 1, 0, 0, 0, 0, 1], t: [3, 0, 1] });
    mapper.o.joinPieces = on;
    const log = [], res = await mergeSegments(mapper, scene.input.pairs, { rng: makeRng(3), log: (s) => log.push(s), minInliers: Infinity });
    const m = compareModels(mapper.model(), scene.truth.model);
    const tag = `${log.join('; ') || 'no merge'}; RMS ${(100 * m.rmsCenterRel).toFixed(3)}%, rot ${m.medianRotDeg.toFixed(3)} deg`;
    if (!on) { assert(res.merges === 0 && new Set(mapper.segment).size === 2, tag); return; }
    assert(res.merges === 1 && new Set(mapper.segment).size === 1 && /join pieces: 1 segments and 0 frames/.test(log.join('; ')), tag);
    assert(m.numCommon === ORBIT.numFrames && m.rmsCenterRel < 0.005 && m.medianRotDeg < 0.1, tag);
  });
}

// A join that moves the largest segment's own frames (any, at maxShift -1) is undone: the map as it was
Deno.test('joinPieces: a join that moves the largest segment undone', async () => {
  const scene = makeSyntheticScene({ ...ORBIT });
  const mapper = new SequentialMapper(scene.input);
  await mapper.run();
  split(mapper);
  mapper.transformSegment(1, { s: 2, R: [0, -1, 0, 1, 0, 0, 0, 0, 1], t: [3, 0, 1] });
  const poses = JSON.stringify(mapper.poses), segment = [...mapper.segment], points = mapper.numPoints();
  const log = [], res = await joinPieces(mapper, { rng: makeRng(3), log: (s) => log.push(s), maxShift: -1 });
  const tag = log.join('; ');
  assert(res === null && /join pieces: undone/.test(tag), tag);
  assert(JSON.stringify(mapper.poses) === poses && mapper.segment.every((s, k) => s === segment[k]) && mapper.numPoints() === points, tag);
});
