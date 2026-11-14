// The mapper's parts taken from COLMAP's incremental mapping (DESIGN-sfm §5.3), each on and off on a synthetic scene:
// the loop pairs mapped with, the next image by its visibility pyramid, the local bundle of the most covisible frames
// with its outer points held, the angular reprojection limit of new points, two-view tracks left out, the tracks
// completed and merged, points too narrow filtered, the refinement loops, fy apart from fx.
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { SequentialMapper } from '../src/sfm/mapper.js';
import { mapSequential } from '../src/sfm/reconstruct.js';
import { compareModels, checkThresholds } from '../src/sfm/compare.js';
import { poseCenter } from '../src/sfm/camera.js';
import { triangulationAngle } from '../src/sfm/geometry.js';
import { assert, ORBIT } from './util.js';

const BASE = { numFrames: 30, maxFeatures: 1000, numGround: 8000, numBoxes: 8, numFar: 1000 };
const mapped = async (scene, options) => { const m = new SequentialMapper(scene.input, options); await m.run(); return m; };
const ORB = makeSyntheticScene(BASE);
const orbit = await mapped(ORB, {});
// the mapped orbit with options (restore takes the snapshot's: set after it)
const from = (o) => { const m = new SequentialMapper(ORB.input, {}); m.restore(orbit.snapshot()); Object.assign(m.o, o); return m; };

// the orbit cut at 40% (its sequential pairs there removed), its revisits loop pairs: with them, one segment
Deno.test('COLMAP mapping: with the loop pairs, a cut orbit mapped as one segment; without, two', async () => {
  const run = async (pairTypes) => {
    const scene = makeSyntheticScene({ ...ORBIT, cut: 0.4 }), { mapper } = await mapSequential(scene.input, { pairTypes });
    const model = mapper.model(), m = compareModels(model, scene.truth.model);
    return { segs: new Set(mapper.segment.filter((_, k) => mapper.poses[k])).size, m, n: scene.input.images.length };
  };
  const loop = await run(['seq', 'loop']), seq = await run(['seq']);
  assert(loop.segs === 1 && loop.m.numCommon === loop.n && checkThresholds(loop.m, 'truth').length === 0,
         `with the loop pairs: ${loop.segs} segments, ${loop.m.numCommon} of ${loop.n}, RMS ${loop.m.rmsCenterRel}`);
  assert(seq.segs >= 2, `without: ${seq.segs} segments`);
});

// COLMAP's VisibilityPyramid: each correspondence adds 4^(l+1) for each level l (2x2 .. 64x64 cells) whose cell it is
// the first in
Deno.test('COLMAP mapping: the visibility pyramid scores the spread of the correspondences, not their count', () => {
  const m = orbit, { width: W, height: H } = m.camera, img = 0, kp = m.images[img].keypoints;
  const near = (x, y) => { let best = -1, d = Infinity; for (let k = 0; k < kp.length / 2; k++) { const e = Math.hypot(kp[2 * k] - x, kp[2 * k + 1] - y); if (e < d) { d = e; best = k; } } return best; };
  const one = 4 + 16 + 64 + 256 + 1024 + 4096;
  const corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]].map(([x, y]) => near(x * W, y * H));
  const score = (kps) => m.pyramidScore(img, new Map(kps.map(k => [k, 0])));
  assert(score([corners[0]]) === one, `one: ${score([corners[0]])}`);
  assert(score(corners) === 4 * one, `four corners: ${score(corners)}`);
  assert(score([corners[0], corners[0]]) === one, 'a keypoint twice');
  // the 4 corners against the keypoints of the fullest 64th of the image
  const cells = new Map();
  for (let k = 0; k < kp.length / 2; k++) {
    const c = Math.floor(64 * kp[2 * k] / W) + 64 * Math.floor(64 * kp[2 * k + 1] / H);
    cells.set(c, [...(cells.get(c) ?? []), k]);
  }
  const cluster = [...cells.values()].sort((a, b) => b.length - a.length)[0];
  assert(cluster.length > 4 && score(cluster) === one && score(corners) > score(cluster), `cluster of ${cluster.length}: ${score(cluster)}`);
});

// FindLocalBundle: the frame and localBAImages - 1 others, all sharing its points, not the nearest in time: frames 14 and
// 16 made to share none of 15's points, they are left out
Deno.test('COLMAP mapping: the local bundle, the frame and the most covisible frames', () => {
  const m = from({}), img = 15;
  for (const pid of m.kpPoint[img]) {
    const P = pid >= 0 && m.points[pid];
    if (P) for (const g of [14, 16]) if (P.obs.has(g)) { m.kpPoint[g][P.obs.get(g)] = -1; P.obs.delete(g); }
  }
  const B = m.localBundle(img);
  const shared = (k) => { let n = 0; for (const pid of m.kpPoint[img]) if (pid >= 0 && m.points[pid]?.obs.has(k)) n++; return n; };
  assert(B[0] === img && B.length === m.o.localBAImages && new Set(B).size === B.length, `bundle ${B}`);
  assert(B.slice(1).every(k => shared(k) > 0) && !B.includes(14) && !B.includes(16), `bundle ${B}: shared ${B.map(shared)}`);
  const small = from({ localBAImages: 3 });
  assert(small.localBundle(img).length === 3, `localBAImages 3: ${small.localBundle(img)}`);
});

// the covisible local BA holds a point seen outside the bundle (unless the new frame's, of a short track); 'window'
// moves every point of its frames
Deno.test('COLMAP mapping: the covisible local BA holds the points seen outside its bundle', async () => {
  const run = async (localBA) => {
    const m = from({ localBA });
    const img = 15, inB = new Set(m.localBundle(img));
    const outer = [];
    for (const pid of m.kpPoint[img]) {
      const P = pid >= 0 && m.points[pid];
      if (P && P.obs.size > m.o.localBAMaxTrack) outer.push([pid, P.X.slice()]);
    }
    const before = m.points.map(P => P?.X.slice());
    for (const pid of m.kpPoint[img]) if (pid >= 0 && m.points[pid]) m.points[pid].X = m.points[pid].X.map(x => x + 1e-3);  // a step to take
    for (const [pid] of outer) before[pid] = m.points[pid].X.slice();
    await m.localBA(img, localBA === 'covisible' ? { covisible: true, iterations: 10 } : { iterations: 10 });
    const heldOuter = outer.filter(([pid]) => m.points[pid] && m.points[pid].X.every((x, i) => x === before[pid][i])).length;
    return { outer: outer.length, heldOuter, inB: inB.size };
  };
  const cov = await run('covisible'), win = await run('window');
  assert(cov.outer > 10 && cov.heldOuter === cov.outer, `covisible: ${cov.heldOuter} of ${cov.outer} long tracks held`);
  assert(win.heldOuter < win.outer / 2, `window: ${win.heldOuter} of ${win.outer} held`);
});

// a new point's views within triMaxDeg (2 deg: f tan 2 deg px) instead of maxReproj (4 px)
Deno.test('COLMAP mapping: a new point kept within triMaxDeg of its views, not maxReproj', () => {
  const run = (triMaxDeg) => {
    const m = from({ triMaxDeg });
    const pid = m.kpPoint[10].findIndex(p => p >= 0 && m.points[p]?.obs.size >= 3), P = m.points[m.kpPoint[10][pid]];
    const obs = [...P.obs], [img] = obs[0];
    // X moved off along the image plane of img by about 10 px
    const C = poseCenter(m.poses[img]), d = Math.hypot(...P.X.map((x, i) => x - C[i])), R = m.poses[img].R, f = m.camera.params[0];
    const X = P.X.map((x, i) => x + R[i] * 10 * d / f);
    const err = Math.max(...obs.map(([g, k]) => m.reprojErr(g, k, X)));
    return { err, id: m.addPoint(X, obs.map(([g, k]) => [g, k]), true), lim: m.triMax(), f };
  };
  const deg = run(2), px = run(null);
  assert(deg.err > 4 && deg.err < deg.lim && Math.abs(deg.lim - deg.f * Math.tan(2 * Math.PI / 180)) < 1e-9 && deg.id >= 0, `triMaxDeg 2: ${JSON.stringify(deg)}`);
  assert(px.lim === 4 && px.id < 0, `triMaxDeg null: ${JSON.stringify(px)}`);
});

// ignore_two_view_tracks: a keypoint matched in one other frame only makes no point (the initial pair's aside)
Deno.test('COLMAP mapping: no new point from a two-view track with ignoreTwoView', async () => {
  const run = async (ignoreTwoView) => {
    const init = new Set(), m = await mapped(ORB, { ignoreTwoView, trace: (_, kind, e) => { if (kind === 'init') init.add(e.i).add(e.j); } });
    let two = 0;
    for (const P of m.points) if (P && [...P.obs].every(([g, k]) => !init.has(g) && m.matchCount(g)[k] === 1)) two++;
    return { two, m };
  };
  const on = await run(true), off = await run(false);
  assert(on.two === 0, `with it: ${on.two} points of two-view tracks`);
  assert(off.two > 0, `without: ${off.two}`);
  assert(checkThresholds(compareModels(on.m.model(), ORB.truth.model), 'truth').length === 0, 'with it, the poses');
});

// CompleteTracks and MergeTracks: an observation taken off a point put back, a point split in two fused back
Deno.test('COLMAP mapping: the tracks completed and merged', () => {
  const m = from({});
  const pids = m.points.map((P, i) => (P && P.obs.size >= 4 ? i : -1)).filter(i => i >= 0).slice(0, 40);
  const cut = pids.slice(0, 20), split = pids.slice(20);
  for (const pid of cut) { const P = m.points[pid], [g, k] = [...P.obs].at(-1); P.obs.delete(g); m.kpPoint[g][k] = -1; }
  const halves = [];
  for (const pid of split) {
    const P = m.points[pid], obs = [...P.obs], back = obs.slice(obs.length >> 1);
    for (const [g] of back) P.obs.delete(g);
    halves.push(m.addPoint(P.X.slice(), back));
  }
  const n0 = m.numPoints(), [added, merged] = m.completeAndMerge([...cut, ...split]);
  assert(added >= cut.length && merged >= split.length && m.numPoints() <= n0 - split.length, `${added} added, ${merged} merged, ${n0} -> ${m.numPoints()} points`);
  assert(cut.every(pid => m.points[pid] && m.points[pid].obs.size >= 4), 'the cut tracks back');
});

// filter_min_tri_angle: a point whose views are under minAngleDeg apart removed. The points of frame 10 put 100 times as
// far along its rays (maxReproj lifted: the angle alone decides)
Deno.test('COLMAP mapping: a point whose views are under minAngleDeg apart filtered', () => {
  const run = (filterMinAngle) => {
    const m = from({ filterMinAngle, maxReproj: Infinity }), narrow = [], C = poseCenter(m.poses[10]);
    for (const pid of m.kpPoint[10]) if (pid >= 0 && m.points[pid]) m.points[pid].X = m.points[pid].X.map((x, i) => C[i] + 100 * (x - C[i]));
    m.points.forEach((P, pid) => {
      if (!P) return;
      const C = [...P.obs.keys()].map(g => poseCenter(m.poses[g]));
      let w = 0;
      for (let a = 0; a < C.length; a++) for (let b = a + 1; b < C.length; b++) w = Math.max(w, triangulationAngle(C[a], C[b], P.X));
      if (w < m.o.minAngleDeg * Math.PI / 180) narrow.push(pid);
    });
    m.filter(m.points.keys());
    return { narrow: narrow.length, left: narrow.filter(pid => m.points[pid]).length };
  };
  const on = run(true), off = run(false);
  assert(off.narrow > 0 && off.left === off.narrow, `without: ${off.left} of ${off.narrow} narrow points left`);
  assert(on.left === 0, `with it: ${on.left} of ${on.narrow} left`);
});

// the orbit with 1.5 px of keypoint noise: its tracks change in each completion, merge and filter (none at 0.5 px)
const NOISY = makeSyntheticScene({ ...BASE, noisePx: 1.5 });

// IterativeLocalRefinement: the local BA again after its completion and merge changed >= 0.001 of the observations;
// localRefinements 1: once
Deno.test('COLMAP mapping: the local refinement repeated while the tracks change', async () => {
  const two = await mapped(NOISY, { localRefinements: 2 }), one = await mapped(NOISY, { localRefinements: 1 });
  assert(two.stats.localIts[1] > 0, `localRefinements 2: ${two.stats.localIts}`);
  assert(one.stats.localIts[1] === 0, `localRefinements 1: ${one.stats.localIts}`);
});

// IterativeGlobalRefinement: the global BA again while its completion, merge and filter change >= 0.0005
Deno.test('COLMAP mapping: the global refinement repeated while the tracks change', async () => {
  const five = await mapped(NOISY, { globalRefinements: 5 }), one = await mapped(NOISY, { globalRefinements: 1 });
  assert(five.stats.globalBAs.length > 0 && five.stats.globalBAs.some(g => g.its > 1) && five.stats.globalBAs.every(g => g.its <= 5),
         `globalRefinements 5: ${five.stats.globalBAs.map(g => g.its)}`);
  assert(one.stats.globalBAs.every(g => g.its === 1), `globalRefinements 1: ${one.stats.globalBAs.map(g => g.its)}`);
});

// COLMAP's OPENCV: fy apart from fx. A lens with fy 1% over fx: refined with 'fy', found; without, fx and fy at one ratio
Deno.test('COLMAP mapping: fy refined apart from fx', async () => {
  const scene = makeSyntheticScene({ ...BASE, model: 'OPENCV', params: [1230, 1242.3, 960, 540, -0.004, 0.002, 0, 0] });
  const run = async (refine) => (await mapped(scene, { refine, refineUnobserved: refine })).camera.params;
  const a = await run(['f', 'k1', 'fy']), b = await run(['f', 'k1']);
  assert(Math.abs(a[1] / a[0] - 1.01) < 1e-3, `with fy: fx ${a[0]}, fy ${a[1]}`);
  assert(Math.abs(b[1] / b[0] - 1.01) > 5e-3, `without: fx ${b[0]}, fy ${b[1]}`);
});
