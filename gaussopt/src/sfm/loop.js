// loop.js — Loop closure and segment merging on a SequentialMapper state (DESIGN-sfm §5.4, S3):
//   1. verify long-range and cross-segment pairs: matched keypoints whose tracks both have 3D
//      points give 3D-3D correspondences; a RANSAC Sim(3) G (i-side map -> j-side map) is scored
//      by reprojection in both frames, so the test is scale-free
//   2. merge segments, largest support first, by the pooled G of all pairs between them
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

// Move a whole segment by the Sim(3) G (old world -> new world)
function transformSegment(mapper, seg, G) {
  const Ginv = inverse(G);
  const moved = new Set();
  for (let k = 0; k < mapper.n; k++) {
    if (mapper.segment[k] !== seg) continue;
    mapper.poses[k] = toPose(compose(fromPose(mapper.poses[k]), Ginv));
    for (const pid of mapper.kpPoint[k]) {
      if (pid < 0 || moved.has(pid) || !mapper.points[pid]) continue;
      moved.add(pid);
      mapper.points[pid].X = apply(G, mapper.points[pid].X);
    }
  }
}

// Merge track pa into pb when they never share an image
function fuse(mapper, pa, pb) {
  if (pa === pb) return false;
  const A = mapper.points[pa], B = mapper.points[pb];
  if (!A || !B) return false;
  for (const img of A.obs.keys()) if (B.obs.has(img)) return false;
  for (const [img, kp] of A.obs) { B.obs.set(img, kp); mapper.kpPoint[img][kp] = pb; }
  mapper.points[pa] = null;
  return true;
}

// mergeOnly: segment merges (and their track fusion) only, no loop edges / pose graph: a local
// repair joining a re-mapped stretch back (session.js redo) must not move the rest of the map
export function closeLoops(mapper, pairs, { rng, log = () => {}, mergeOnly = false, ...options } = {}) {
  const o = { ...DEFAULTS, ...options };
  const stats = { verified: 0, merges: 0, loopEdges: 0, fused: 0, poseGraph: [], loops: [] };
  const registered = (p) => mapper.poses[p.i] && mapper.poses[p.j];
  const candidates = () => pairs.filter(p => registered(p) &&
    (mapper.segment[p.i] !== mapper.segment[p.j] || Math.abs(p.j - p.i) > o.longRange));

  // ── 2. segment merges ──
  for (;;) {
    const bySegPair = new Map();
    for (const p of candidates()) {
      const sa = mapper.segment[p.i], sb = mapper.segment[p.j];
      if (sa === sb) continue;
      const key = `${sa},${sb}`;
      if (!bySegPair.has(key)) bySegPair.set(key, []);
      bySegPair.get(key).push(...correspondences(mapper, p));
    }
    let best = null;
    for (const [key, corr] of bySegPair) {
      const res = ransacSim3(mapper, corr, { rng, ...o });
      if (res && (!best || res.inliers.length > best.res.inliers.length)) best = { key, res };
    }
    if (!best) break;
    const [sa, sb] = best.key.split(',').map(Number);
    const size = (s) => mapper.segment.reduce((c, x) => c + (x === s), 0);
    // G maps sa's map onto sb's: move the smaller segment
    if (size(sa) <= size(sb)) { transformSegment(mapper, sa, best.res.G); relabel(mapper, sa, sb); }
    else { transformSegment(mapper, sb, inverse(best.res.G)); relabel(mapper, sb, sa); }
    for (const c of best.res.inliers) stats.fused += fuse(mapper, c.pa, c.pb);
    stats.merges++;
    log(`merged segments ${sa} and ${sb}: ${best.res.inliers.length} inliers, scale ${best.res.G.s.toFixed(3)}`);
  }

  if (mergeOnly) return stats;

  // ── 3. loop edges + pose graph, per segment ──
  const loops = [];
  const perFrame = new Map();
  const userFirst = (a, b) => (b.type === 'user-loop') - (a.type === 'user-loop') || b.matches.length - a.matches.length;
  for (const p of candidates().filter(p => mapper.segment[p.i] === mapper.segment[p.j]).sort(userFirst)) {
    const c = perFrame.get(p.i) ?? 0;
    if (c >= o.loopsPerFrame) continue;
    perFrame.set(p.i, c + 1);
    const res = ransacSim3(mapper, correspondences(mapper, p), { rng, ...o });
    stats.verified++;
    if (res) loops.push({ p, ...res });
  }
  stats.loopEdges = loops.length;
  const segs = [...new Set(Array.from(mapper.segment).filter(s => s >= 0))];
  for (const seg of segs) {
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
  }

  // ── 4. fuse the tracks matched by verified loops (re-verified after the correction) ──
  for (const { p } of loops) {
    const res = ransacSim3(mapper, correspondences(mapper, p), { rng, ...o });
    if (res) for (const c of res.inliers) stats.fused += fuse(mapper, c.pa, c.pb);
  }
  log(`loops: ${stats.loopEdges}/${stats.verified} verified, ${stats.merges} merges, ${stats.fused} tracks fused`);
  return stats;
}

function relabel(mapper, from, to) {
  for (let k = 0; k < mapper.n; k++) if (mapper.segment[k] === from) mapper.segment[k] = to;
}

