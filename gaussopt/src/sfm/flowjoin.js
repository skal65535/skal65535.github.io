// flowjoin.js — Pieces of a video's map joined across a turn in place, by the optical flow of the
// frame analysis (DESIGN-sfm §5.4.1). When the camera turns without moving, no new point can be
// triangulated: the map cannot follow it, and the keyframes after the turn start a piece of their own
// that shares no point with the first. The flow tracks
// through every frame of the turn: a camera turning in place moves each track by one homography,
// K R K^-1, so the homographies of the analysis's steps (frames.js step, chained per keyframe by the
// worker: flowH, flowChain) give the rotation between any two keyframes of one chain.
// The join of a piece B to the piece A before it in time, from A's last keyframe a to B's first b:
//   - the turn: the homography from a to b, in the camera's own normalized coordinates, a rotation within
//     maxSpread;
//   - by the frames both pieces see: the keyframes of each piece that see the other's points (next to the
//     gap, or seeing again what the other saw, e.g. the way back after a U-turn) are posed in its world too (PnP, as the mapper registers a frame), each one then posed in both
//     worlds; the Sim(3) between the worlds fitted to them (rotation, then scale and shift from their
//     centres), and their PnP inliers become observations of the other's points (or fuse the two tracks of
//     one keypoint), so that the window BAs after it hold the joint, scale included;
//   - else by the flow: B's first camera turned from A's last by the flow's rotation, b at a's centre (the
//     flow says the camera turned in place), and B's speed matching A's: each piece's fastest over speedSpan
//     within speedSeconds of the gap (an assumption: a drone stops, turns, hovers and speeds up again, so
//     its speed next to the gap says nothing; on canyon2.mp4's stop and turn at 73 s the steps' speeds there
//     put the way back 25x off, its fastest 1.19x; on canyon3.mp4, flown back 1.5x faster, 1.14x).
// B's segment, whole, is moved into A's world (SequentialMapper.transformSegment) and joins A's; the keyframes
// between a and b get a's centre and their flow rotation. The refinement after it (reconstruct.js)
// adjusts them to their points.
"use strict";

import { dist } from '../math.js';
import { intrinsics, poseCenter, poseFromCenter } from './camera.js';
import { mul3, transpose3, mulVec3, svd, det3, projectToSO3, invert3, rotationAngle } from './linalg.js';
import { ransacPnP } from './geometry.js';
import { deg, median } from './util.js';

const DEFAULTS = {
  maxGapSeconds: 3,    // the most time between a and b
  maxSpread: 0.1,      // the homography's singular values within this of each other (a rotation: 0)
  speedSeconds: 10,    // each piece's speed over this much of it next to the gap: a drone stops, turns, hovers and speeds
                       // up again, its speed next to the gap is none of its cruise
  speedSpan: 2,        // the speed of a piece: its fastest displacement over this long (a step's own is noise)
  minShared: 3,        // keyframes posed in both worlds for a join by them
  maxSharedRes: 0.25,  // their centres' residual after the fit, at most this fraction of their spread
  log: () => {},
};

// The rotation from keyframe a's camera to b's (b = R a, in camera coordinates) by the flow: { R, spread }
// (R null when the homography is not a rotation within maxSpread), or null without a flow chain from a to b
// (or with a singular homography)
export function flowRotation(mapper, a, b, { maxSpread = DEFAULTS.maxSpread } = {}) {
  const A = mapper.images[a], B = mapper.images[b];
  if (!A.flowH || !B.flowH || A.flowChain !== B.flowChain || mapper.camOf[a] !== mapper.camOf[b]) return null;
  const [fx, fy, cx, cy] = intrinsics(mapper.cam(a));
  const K = [fx, 0, cx, 0, fy, cy, 0, 0, 1], Ki = [1 / fx, 0, -cx / fx, 0, 1 / fy, -cy / fy, 0, 0, 1];
  const Ai = new Array(9);
  if (!invert3(A.flowH, Ai)) return null;
  let M = mul3(mul3(Ki, mul3(B.flowH, Ai)), K);
  const c = Math.cbrt(det3(M));
  M = M.map(x => x / c);
  const { S } = svd(Float64Array.from(M), 3, 3), spread = S[0] / S[2] - 1;
  return { spread, R: spread <= maxSpread ? projectToSO3(M) : null };
}

// Speed (map units per second) of a piece's registered keyframes `frames` (time order): its largest displacement
// over `span` seconds (over all of them where they span less), per second (NaN where they span no time)
function speed(mapper, frames, span) {
  const time = (k) => mapper.images[k].time;
  span = Math.min(span, time(frames.at(-1)) - time(frames[0]));
  if (!(span > 0)) return NaN;
  let v = 0;
  for (let i = 0, j = 0; i < frames.length; i++) {
    while (j < frames.length && time(frames[j]) - time(frames[i]) < span) j++;
    if (j === frames.length) break;
    const a = frames[i], b = frames[j];
    v = Math.max(v, dist(poseCenter(mapper.poses[b]), poseCenter(mapper.poses[a])) / (time(b) - time(a)));
  }
  return v;
}

// The keyframes `frames` (of one segment) posed in segment `seg`'s world as well, by PnP on its points (as
// SequentialMapper.register): [{ k, pose, inliers: [[kp, pid]] }]
function resect(mapper, frames, seg) {
  const { minInliers, pnpThreshold } = mapper.o, out = [];
  for (const k of frames) {
    const corr = [...mapper.correspondences(k, seg)];
    if (corr.length < minInliers) continue;
    const res = ransacPnP(corr.map(([kp]) => mapper.ray(k, kp)), corr.map(([kp]) => mapper.pixel(k, kp)), corr.map(([, pid]) => mapper.points[pid].X),
                          mapper.cam(k), { rng: mapper.rng, threshold: pnpThreshold });
    if (!res || res.numInliers < minInliers) continue;
    const inl = corr.filter((_, i) => res.inliers[i]), pose = mapper.refinePose(k, res.model, inl);
    const inliers = inl.filter(([kp, pid]) => mapper.reprojWith(pose, k, kp, mapper.points[pid].X) <= pnpThreshold);
    if (inliers.length >= minInliers) out.push({ k, pose, inliers });
  }
  return out;
}

// The Sim(3) G from B's world to A's (X_A = s R X_B + t) by keyframes posed in both: [{ A: pose, B: pose }].
// R from their rotations (R_A = R_B R^T), refitted without those off by over 3x the median; s, t from their
// centres by least squares, refitted once without those off by over 3x the median. Returns { G, res, spread }
// (res: the centres' RMS residual, spread: their RMS spread, in A's units), or null with fewer than minShared
function sharedSim3(both, minShared) {
  if (both.length < minShared) return null;
  const Gk = both.map(({ A, B }) => mul3(transpose3(A.R), B.R));
  let R = projectToSO3(Gk.reduce((S, M) => S.map((x, i) => x + M[i])));
  const off = Gk.map(M => rotationAngle(M, R)), m = median(off);
  const keep = both.map((_, i) => i).filter(i => off[i] <= 3 * m + 1e-9);
  if (keep.length < minShared) return null;
  R = projectToSO3(keep.reduce((S, i) => S.map((x, j) => x + Gk[i][j]), [0, 0, 0, 0, 0, 0, 0, 0, 0]));
  const Y = keep.map(i => poseCenter(both[i].A)), X = keep.map(i => mulVec3(R, poseCenter(both[i].B)));
  const fit = (idx) => {
    const my = [0, 1, 2].map(d => idx.reduce((a, i) => a + Y[i][d], 0) / idx.length), mx = [0, 1, 2].map(d => idx.reduce((a, i) => a + X[i][d], 0) / idx.length);
    let num = 0, den = 0;
    for (const i of idx) for (let d = 0; d < 3; d++) { num += (Y[i][d] - my[d]) * (X[i][d] - mx[d]); den += (X[i][d] - mx[d]) ** 2; }
    const s = num / den, t = my.map((y, d) => y - s * mx[d]);
    const res = Y.map((y, i) => dist(y, X[i].map((x, d) => s * x + t[d])));
    return { s, t, res, my };
  };
  let idx = Y.map((_, i) => i), f = fit(idx);
  const mr = median(idx.map(i => f.res[i])), kept = idx.filter(i => f.res[i] <= 3 * mr + 1e-12);
  if (kept.length >= minShared && kept.length < idx.length) { idx = kept; f = fit(idx); }
  if (!(f.s > 0 && isFinite(f.s))) return null;
  const rms = (v) => Math.sqrt(v.reduce((a, x) => a + x * x, 0) / v.length);
  return { G: { s: f.s, R, t: f.t }, res: rms(idx.map(i => f.res[i])), spread: rms(idx.map(i => dist(Y[i], f.my))), n: idx.length };
}

// Joins the pieces that follow each other in time across turns in place. Returns { joins: [{ from, into,
// a, b, turnDeg, spread, scale, carried }], tried }
export function joinByFlow(mapper, options = {}) {
  const o = { ...DEFAULTS, ...options, log: options.log ?? DEFAULTS.log }, time = (k) => mapper.images[k].time, stats = { joins: [], tried: 0 };
  for (let joined = true; joined;) {
    joined = false;
    // a piece: registered keyframes following each other in time in one segment. A segment can hold
    // several, interleaved with another's once loop closure merged some
    const pieces = [];
    for (let k = 0; k < mapper.n; k++) if (mapper.poses[k]) {
      if (pieces.at(-1)?.seg === mapper.segment[k]) pieces.at(-1).frames.push(k);
      else pieces.push({ seg: mapper.segment[k], frames: [k] });
    }
    for (let i = 1; i < pieces.length && !joined; i++) {
      const A = pieces[i - 1], B = pieces[i], a = A.frames.at(-1), b = B.frames[0];
      if (!mapper.images[a].flowH || !mapper.images[b].flowH) continue;  // no flow (photos): nothing to join by
      if (time(b) - time(a) > o.maxGapSeconds) continue;  // too far apart
      stats.tried++;
      const rot = flowRotation(mapper, a, b, o);
      const tag = `flow join of keyframes ${a} and ${b} (${time(a).toFixed(2)}-${time(b).toFixed(2)} s)`;
      if (!rot?.R) { o.log(`${tag}: ${rot ? `not a turn in place (spread ${rot.spread.toFixed(3)})` : 'no flow between them'}`); continue; }
      const near = (frames, fromEnd) => frames.filter(k => Math.abs(time(k) - time(fromEnd)) <= o.speedSeconds);
      // the frames both see: A's posed in B's world, B's in A's
      const inB = resect(mapper, A.frames, B.seg), inA = resect(mapper, B.frames, A.seg);
      const both = [...inB.map(r => ({ A: mapper.poses[r.k], B: r.pose })), ...inA.map(r => ({ A: r.pose, B: mapper.poses[r.k] }))];
      const shared = sharedSim3(both, o.minShared), byShared = shared && shared.res <= o.maxSharedRes * shared.spread;
      const Ra = mapper.poses[a].R, Ca = poseCenter(mapper.poses[a]);
      let G, how;
      if (byShared) {
        G = shared.G;
        how = `by ${shared.n} keyframes posed in both (${inB.length} of A's, ${inA.length} of B's; centres within ${(100 * shared.res / shared.spread).toFixed(1)}% of their spread)`;
      } else {
        const vA = speed(mapper, near(A.frames, a), o.speedSpan), vB = speed(mapper, near(B.frames, b), o.speedSpan);
        if (!(vA > 0 && vB > 0)) { o.log(`${tag}: a piece does not move (speeds ${vA}, ${vB})`); continue; }
        const s = vA / vB, RbA = mul3(rot.R, Ra), GR = mul3(transpose3(RbA), mapper.poses[b].R), r = mulVec3(GR, poseCenter(mapper.poses[b]));
        G = { s, R: GR, t: [Ca[0] - s * r[0], Ca[1] - s * r[1], Ca[2] - s * r[2]] };
        how = `by the speeds (${vA.toPrecision(3)}, ${vB.toPrecision(3)} per s; ${shared ? `the ${shared.n} keyframes posed in both off by ${(100 * shared.res / shared.spread).toFixed(0)}% of their spread` : `${both.length} keyframes posed in both`})`;
      }
      mapper.transformSegment(B.seg, G);
      mapper.relabelSegment(B.seg, A.seg);
      // the resected frames' inliers: observations of the other piece's points, or the fusion of a keypoint's two tracks
      let observed = 0, fused = 0;
      if (byShared) for (const { k, inliers } of [...inB, ...inA]) for (const [kp, pid] of inliers) {
        const P = mapper.points[pid], own = mapper.kpPoint[k][kp];
        if (!P || own === pid || mapper.reprojErr(k, kp, P.X) > mapper.o.pnpThreshold) continue;
        if (own < 0) { if (!P.obs.has(k)) { P.obs.set(k, kp); mapper.kpPoint[k][kp] = pid; observed++; } }
        else if (mapper.fuseTracks(pid, own)) fused++;
      }
      let carried = 0;
      for (let k = a + 1; k < b; k++) {
        if (mapper.poses[k]) continue;
        const rk = flowRotation(mapper, a, k, { maxSpread: Infinity });
        if (!rk?.R) continue;
        mapper.place(k, poseFromCenter(mul3(rk.R, Ra), Ca), A.seg);
        mapper.byFlow[k] = 1;  // no points: quality.js scores it rotation only, which repair leaves
        carried++;
      }
      const turn = deg(rotationAngle(rot.R, [1, 0, 0, 0, 1, 0, 0, 0, 1]));
      o.log(`${tag}: turned ${turn.toFixed(1)} deg in place (spread ${rot.spread.toFixed(3)}), segment ${B.seg} (its piece ${b}..${B.frames.at(-1)}) ` +
            `joins segment ${A.seg} at scale ${G.s.toFixed(3)} ${how}, ${observed} observations and ${fused} tracks fused across, ${carried} keyframes between placed by the flow`);
      stats.joins.push({ from: B.seg, into: A.seg, a, b, turnDeg: turn, spread: rot.spread, scale: G.s, by: byShared ? 'shared' : 'speed', shared: both.length, observed, fused, carried });
      mapper.trace('join', stats.joins.at(-1));
      joined = true;
    }
  }
  return stats;
}
