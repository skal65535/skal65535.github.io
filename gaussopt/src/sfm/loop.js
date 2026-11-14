// loop.js — Loop closure and segment merging on a SequentialMapper state (DESIGN-sfm §5.4, S3):
//   1. verify long-range and cross-segment pairs: matched keypoints whose tracks both have 3D
//      points give 3D-3D correspondences; a RANSAC Sim(3) G (i-side map -> j-side map) is scored
//      by reprojection in both frames, so the test is scale-free
//   2. merge segments, joints first then largest support, by the pooled G of their pairs across the joint, else of all;
//      when none is left and the mapper's joinPieces is on: once, the pieces placed by their pairs (joinpieces.js)
//   3. Sim(3) pose graph per segment: covisibility edges keep the current relative transforms,
//      loop edges ask for S_j o G o S_i^-1; points move with their reference frame
//   4. fuse the tracks that a verified loop matched to each other
// stats.loops lists the loop edges with their residual after the pose graph (quality.js).
// Loop pairs come from the input (loopdetect.js candidates once matched, COLMAP's, or synthetic);
// 'user-loop' pairs (forced in the UI, session.js) are verified first.
"use strict";

import { umeyama } from './align.js';
import { ransac } from './geometry.js';
import { project } from './camera.js';
import { compose, inverse, apply, fromPose, toPose, error } from './sim3.js';
import { optimizePoseGraph } from './posegraph.js';
import { pacer } from './pace.js';
import { joinPieces } from './joinpieces.js';

const DEFAULTS = { minInliers: 30, threshold: 4, longRange: 20, iterations: 500, minCovisible: 30, loopsPerFrame: 3 };

// 3D-3D correspondences of a pair (i, j) through the current tracks: [{ i, j, a, b, pa, pb }]
function correspondences(mapper, { i, j, matches }) {
  const out = [];
  for (let m = 0; m < matches.length; m += 2) {
    const a = matches[m], b = matches[m + 1];
    const pa = mapper.kpPoint[i][a], pb = mapper.kpPoint[j][b];
    if (pa >= 0 && pb >= 0 && mapper.points[pa] && mapper.points[pb]) out.push({ i, j, a, b, pa, pb });
  }
  return out;
}

// RANSAC Sim(3) mapping i-side points onto j-side points over correspondences (possibly pooled
// from several pairs); inliers reproject within `threshold` px in both frames. Adaptive RANSAC
// (geometry.js) with local refits on the inliers.
function ransacSim3(mapper, corr, { rng, threshold, iterations, minInliers }) {
  if (corr.length < Math.max(3, minInliers)) return null;
  const inverses = new WeakMap();
  const err = (G, c) => {
    let Ginv = inverses.get(G);
    if (!Ginv) { Ginv = inverse(G); inverses.set(G, Ginv); }
    const [u, v, z] = project(mapper.cam(c.j), mapper.poses[c.j], apply(G, mapper.points[c.pa].X));
    const [u2, v2, z2] = project(mapper.cam(c.i), mapper.poses[c.i], apply(Ginv, mapper.points[c.pb].X));
    if (!(z > 0) || !(z2 > 0)) return Infinity;
    const [pu, pv] = mapper.pixel(c.j, c.b), [qu, qv] = mapper.pixel(c.i, c.a);
    return Math.max((u - pu) ** 2 + (v - pv) ** 2, (u2 - qu) ** 2 + (v2 - qv) ** 2);
  };
  const res = ransac({
    n: corr.length, sampleSize: 3, rng, threshold, maxIterations: iterations,
    fit: (idx) => {
      try {
        const G = umeyama(idx.map(k => mapper.points[corr[k].pa].X), idx.map(k => mapper.points[corr[k].pb].X));
        return G.s > 0 && isFinite(G.s) ? [G] : [];
      } catch { return []; }
    },
    error: (G, k) => err(G, corr[k]),
  });
  if (!res || res.numInliers < minInliers) return null;
  return { G: res.model, inliers: corr.filter((_, k) => res.inliers[k]) };
}

// Segment merges, joints first then largest support, by the pooled G of their pairs across the joint (else of all), with their track
// fusion; no loop edges or pose graph: a local repair joining a re-mapped stretch back (session.js redo)
// does not move the rest of the map. Returns { merges, fused, joinedFrames }. Progress (the mapper's): `stage` merges of at most
// the segments less one, a yield to the worker after each.
export async function mergeSegments(mapper, pairs, { rng, log = () => {}, stage = 'loopMerge', ...options } = {}) {
  const o = { ...DEFAULTS, ...options };
  const stats = { merges: 0, fused: 0, joinedFrames: 0 };
  const most = new Set(Array.from(mapper.segment).filter(s => s >= 0)).size - 1;
  let joinTried = false;
  for (;;) {
    mapper.o.progress?.(stage, stats.merges, most);
    await new Promise(r => setTimeout(r, 0));
    // Per pair of segments, the correspondences of all their pairs, and of those within longRange: across their joint
    // in time. The joint's G when it has one, and joints first: where two segments meet their scales agree, while a
    // loop's G brings the drift of a segment's far end to the joint (gare2: scale 0.43 from the loop, 1.49 at the
    // joint). The loop is then the pose graph's.
    const bySegPair = new Map();
    for (const p of candidates(mapper, pairs, o)) {
      const sa = mapper.segment[p.i], sb = mapper.segment[p.j];
      if (sa === sb) continue;
      const key = `${sa},${sb}`;
      if (!bySegPair.has(key)) bySegPair.set(key, { all: [], joint: [] });
      const g = bySegPair.get(key), c = correspondences(mapper, p);
      g.all.push(...c);
      if (Math.abs(p.j - p.i) <= o.longRange) g.joint.push(...c);
    }
    let best = null;
    const better = (b) => !best || b.joint > best.joint || (b.joint === best.joint && b.res.inliers.length > best.res.inliers.length);
    for (const [key, { all, joint }] of bySegPair) {
      let res = ransacSim3(mapper, joint, { rng, ...o }), atJoint = !!res;
      if (!res) res = ransacSim3(mapper, all, { rng, ...o });
      const b = res && { key, res, joint: atJoint };
      if (b && better(b)) best = b;
    }
    if (!best) {
      // no Sim(3) left: the pieces placed by their pairs, once (joinpieces.js), then the 3D points tried again
      const j = mapper.o.joinPieces && !joinTried ? await joinPieces(mapper, { rng, log }) : null;
      joinTried = true;
      if (!j) break;
      stats.merges += j.segments.length;
      stats.joinedFrames += j.frames.length;
      continue;
    }
    const [sa, sb] = best.key.split(',').map(Number);
    const size = (s) => mapper.segment.reduce((c, x) => c + (x === s), 0);
    // G maps sa's map onto sb's: move the smaller segment
    if (size(sa) <= size(sb)) { mapper.transformSegment(sa, best.res.G); mapper.relabelSegment(sa, sb); }
    else { mapper.transformSegment(sb, inverse(best.res.G)); mapper.relabelSegment(sb, sa); }
    for (const c of best.res.inliers) stats.fused += mapper.fuseTracks(c.pa, c.pb);
    const extended = best.joint ? extendAcross(mapper, best.res.inliers) : 0;
    stats.extended = (stats.extended ?? 0) + extended;
    stats.merges++;
    log(`merged segments ${sa} and ${sb}: ${best.res.inliers.length} inliers${best.joint ? `, across their joint (${extended} points made there)` : ''}, scale ${best.res.G.s.toFixed(3)}`);
    mapper.trace('merge', { a: sa, b: sb, inliers: best.res.inliers.length, scale: best.res.G.s, joint: best.joint });
  }
  return stats;
}

// After a merge across a joint: the frames around it (a window either side of the inliers' frames) retriangulated, so
// that each sees the other side's points, then their points filtered. Fused, only the inliers' tracks cross the joint:
// gare2's last keyframes before it saw 16 and 1 points, and the first window of the final sweeps turned them 27 deg.
// Returns the points made
function extendAcross(mapper, inliers) {
  let first = Infinity, last = -Infinity;
  for (const { i, j } of inliers) { first = Math.min(first, i, j); last = Math.max(last, i, j); }
  const W = mapper.o.window, lo = Math.max(0, first - W), hi = Math.min(mapper.n - 1, last + W);
  return mapper.triangulateAndFilter(Array.from({ length: hi - lo + 1 }, (_, q) => lo + q));
}

// Pairs between registered frames of different segments or more than longRange apart
function candidates(mapper, pairs, { longRange }) {
  return pairs.filter(p => mapper.poses[p.i] && mapper.poses[p.j] &&
    (mapper.segment[p.i] !== mapper.segment[p.j] || Math.abs(p.j - p.i) > longRange));
}

// Segment merges (mergeSegments), then loop edges and a pose graph per segment, then the fusion of the
// tracks the loops matched. Progress (the mapper's) through each of the four, the worker given way to as it goes.
// afterMerge(mapper, before) (optional, after merges): more verified pairs [{ i, j, type, matches }], before: each
// frame's segment before them. A revisit's frames proposed while in two segments (loopdetect.js geometricCandidates
// looks within one) have their candidates only now: autumn-garden's 20-30 s and 240-250 s, merged through other
// frames, had no loop edge between them. They are appended to `pairs`.
export async function closeLoops(mapper, pairs, { rng, log = () => {}, afterMerge = null, ...options } = {}) {
  const o = { ...DEFAULTS, ...options };
  const progress = (stage, done, total) => mapper.o.progress?.(stage, done, total), pace = pacer();
  const before = Int32Array.from(mapper.segment);
  const merged = await mergeSegments(mapper, pairs, { rng, log, ...options });
  const stats = { verified: 0, merges: merged.merges, joinedFrames: merged.joinedFrames, loopEdges: 0, fused: merged.fused, poseGraph: [], loops: [], proposed: 0 };
  if (afterMerge && (merged.merges || merged.joinedFrames)) {
    const more = await afterMerge(mapper, before);
    pairs.push(...more);
    stats.proposed = more.length;
  }

  // ── 3. loop edges + pose graph, per segment ──
  const loops = [];
  const perFrame = new Map();
  const userFirst = (a, b) => (b.type === 'user-loop') - (a.type === 'user-loop') || b.matches.length - a.matches.length;
  const within = candidates(mapper, pairs, o).filter(p => mapper.segment[p.i] === mapper.segment[p.j]).sort(userFirst);
  for (const [q, p] of within.entries()) {
    if (q % 64 === 0) { progress('loopVerify', q, within.length); await pace(); }
    const c = perFrame.get(p.i) ?? 0;
    if (c >= o.loopsPerFrame) continue;
    perFrame.set(p.i, c + 1);
    const res = ransacSim3(mapper, correspondences(mapper, p), { rng, ...o });
    stats.verified++;
    if (res) loops.push({ p, ...res });
  }
  stats.loopEdges = loops.length;
  const segs = [...new Set(Array.from(mapper.segment).filter(s => s >= 0))];
  for (const [s, seg] of segs.entries()) {
    progress('poseGraph', s, segs.length);
    await new Promise(r => setTimeout(r, 0));
    const frames = [];
    for (let k = 0; k < mapper.n; k++) if (mapper.segment[k] === seg && mapper.poses[k]) frames.push(k);
    const segLoops = loops.filter(l => mapper.segment[l.p.i] === seg);
    if (!segLoops.length) continue;
    const node = new Map(frames.map((k, idx) => [k, idx]));
    const old = frames.map(k => fromPose(mapper.poses[k]));
    const nodes = old.slice();
    const edges = [];
    // covisibility edges between frames sharing points (within the long-range distance)
    for (const [a, ia] of node) {
      const shared = new Map();
      for (const pid of mapper.kpPoint[a]) {
        if (pid < 0 || !mapper.points[pid]) continue;
        for (const img of mapper.points[pid].obs.keys()) if (img > a && img - a <= o.longRange) shared.set(img, (shared.get(img) ?? 0) + 1);
      }
      for (const [b, count] of shared) {
        if (count < o.minCovisible || !node.has(b)) continue;
        const ib = node.get(b);
        edges.push({ i: ia, j: ib, S: compose(nodes[ib], inverse(nodes[ia])), weight: 1 });
      }
    }
    for (const { p, G } of segLoops) {
      const ia = node.get(p.i), ib = node.get(p.j);
      edges.push({ i: ia, j: ib, S: compose(compose(nodes[ib], G), inverse(nodes[ia])), weight: 1 });
    }
    const res = optimizePoseGraph(nodes, edges);
    stats.poseGraph.push({ segment: seg, frames: frames.length, edges: edges.length, loops: segLoops.length, ...res });
    // what each loop still disagrees with after the optimization (a false loop stands out)
    const loopEdges = edges.slice(edges.length - segLoops.length);
    segLoops.forEach(({ p, inliers }, k) => {
      const r = error(compose(compose(loopEdges[k].S, nodes[loopEdges[k].i]), inverse(nodes[loopEdges[k].j])));
      stats.loops.push({ i: p.i, j: p.j, inliers: inliers.length, rotDeg: Math.hypot(r[0], r[1], r[2]) * 180 / Math.PI, logScale: r[6] });
    });
    log(`pose graph, segment ${seg}: ${frames.length} frames, ${edges.length} edges (${segLoops.length} loops), cost ${res.initialCost.toExponential(2)} -> ${res.cost.toExponential(2)}`);
    mapper.trace('loops', { segment: seg, pairs: segLoops.map(({ p }) => [p.i, p.j]) });  // before the pose graph moves them
    // points move with their reference frame (first observer)
    const moved = new Set();
    frames.forEach((k, idx) => {
      for (const pid of mapper.kpPoint[k]) {
        if (pid < 0 || moved.has(pid) || !mapper.points[pid]) continue;
        const ref = mapper.points[pid].obs.keys().next().value;
        if (!node.has(ref)) continue;
        moved.add(pid);
        const r = node.get(ref);
        mapper.points[pid].X = apply(inverse(nodes[r]), apply(old[r], mapper.points[pid].X));
      }
      mapper.poses[k] = toPose(nodes[idx]);
    });
    mapper.trace('posegraph', { segment: seg, frames: frames.length, edges: edges.length, loops: segLoops.length, initialCost: res.initialCost, cost: res.cost });
  }

  // ── 4. fuse the tracks matched by verified loops (re-verified after the correction) ──
  for (const [q, { p }] of loops.entries()) {
    if (q % 64 === 0) { progress('loopFuse', q, loops.length); await pace(); }
    const res = ransacSim3(mapper, correspondences(mapper, p), { rng, ...o });
    if (res) for (const c of res.inliers) stats.fused += mapper.fuseTracks(c.pa, c.pb);
  }
  log(`loops: ${stats.loopEdges}/${stats.verified} verified, ${stats.merges} merges, ${stats.fused} tracks fused`);
  return stats;
}
