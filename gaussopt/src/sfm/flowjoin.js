// flowjoin.js — Pieces of a video's map joined across a turn in place, by the optical flow of the
// frame analysis (DESIGN-sfm §5.4.1). When the camera turns without moving, no new point can be
// triangulated: the map cannot follow it, and the keyframes after the turn start a piece of their own
// that shares no point with the first (long-garden.mp4: a 190° yaw at 33.4-35.4 s). The flow tracks
// through every frame of the turn: a camera turning in place moves each track by one homography,
// K R K^-1, so the homographies of the analysis's steps (frames.js step, chained per keyframe by the
// worker: flowH, flowChain) give the rotation between any two keyframes of one chain.
// The join of a piece B to the piece A before it in time, from A's last keyframe a to B's first b:
//   - rotation: B's first camera turned from A's last by the flow's rotation (accepted when the
//     homography, in the camera's own normalized coordinates, is a rotation within maxSpread);
//   - centre: b at a's centre (the flow says the camera turned in place);
//   - scale: B's speed at its start matches A's at its end (an assumption: no point is seen on both sides).
// B is moved into A's world (loop.js transformSegment) and joins it; the keyframes between a and b get
// a's centre and their flow rotation. The refinement after it (reconstruct.js) adjusts them to their
// points.
"use strict";

import { intrinsics, poseCenter, poseFromCenter } from './camera.js';
import { mul3, transpose3, mulVec3, svd, det3, projectToSO3 } from './linalg.js';
import { deg } from './util.js';
import { transformSegment, relabel } from './loop.js';

const DEFAULTS = {
  maxGapSeconds: 3,    // the most time between a and b
  maxSpread: 0.05,     // the homography's singular values within this of each other (a rotation: 0)
  speedSeconds: 3,     // each piece's speed over this much of it next to the gap
  speedQuantile: 0.75, // of its per-step speeds (the steps of the turn itself barely move)
  log: () => {},
};

function inverse3(m) {
  const [a, b, c, d, e, f, g, h, i] = m, A = e * i - f * h, B = f * g - d * i, C = d * h - e * g, det = a * A + b * B + c * C;
  return [A / det, (c * h - b * i) / det, (b * f - c * e) / det, B / det, (a * i - c * g) / det, (c * d - a * f) / det,
          C / det, (b * g - a * h) / det, (a * e - b * d) / det];
}

// The rotation from keyframe a's camera to b's (b = R a, in camera coordinates) by the flow: { R, spread }
// (R null when the homography is not a rotation within maxSpread), or null without a flow chain from a to b
export function flowRotation(mapper, a, b, { maxSpread = DEFAULTS.maxSpread } = {}) {
  const A = mapper.images[a], B = mapper.images[b];
  if (!A.flowH || !B.flowH || A.flowChain !== B.flowChain || mapper.camOf[a] !== mapper.camOf[b]) return null;
  const [fx, fy, cx, cy] = intrinsics(mapper.cam(a));
  const K = [fx, 0, cx, 0, fy, cy, 0, 0, 1], Ki = [1 / fx, 0, -cx / fx, 0, 1 / fy, -cy / fy, 0, 0, 1];
  let M = mul3(mul3(Ki, mul3(B.flowH, inverse3(A.flowH))), K);
  const c = Math.cbrt(det3(M));
  M = M.map(x => x / c);
  const { S } = svd(Float64Array.from(M), 3, 3), spread = S[0] / S[2] - 1;
  return { spread, R: spread <= maxSpread ? projectToSO3(M) : null };
}

// Speed (map units per second) of a piece's registered keyframes `frames` (time order): the quantile of its
// per-step speeds
function speed(mapper, frames, q) {
  const v = [];
  for (let k = 1; k < frames.length; k++) {
    const a = frames[k - 1], b = frames[k], dt = mapper.images[b].time - mapper.images[a].time;
    const Ca = poseCenter(mapper.poses[a]), Cb = poseCenter(mapper.poses[b]);
    if (dt > 0) v.push(Math.hypot(Cb[0] - Ca[0], Cb[1] - Ca[1], Cb[2] - Ca[2]) / dt);
  }
  v.sort((x, y) => x - y);
  return v.length ? v[Math.min(v.length - 1, Math.floor(q * v.length))] : 0;
}

// Joins the pieces that follow each other in time across turns in place. Returns { joins: [{ from, into,
// a, b, turnDeg, spread, scale, carried }], tried }
export function joinByFlow(mapper, options = {}) {
  const o = { ...DEFAULTS, ...options, log: options.log ?? DEFAULTS.log }, time = (k) => mapper.images[k].time, stats = { joins: [], tried: 0 };
  for (let joined = true; joined;) {
    joined = false;
    const byTime = new Map();
    for (let k = 0; k < mapper.n; k++) if (mapper.poses[k]) {
      if (!byTime.has(mapper.segment[k])) byTime.set(mapper.segment[k], []);
      byTime.get(mapper.segment[k]).push(k);
    }
    const pieces = [...byTime].map(([seg, frames]) => ({ seg, frames })).sort((p, q) => p.frames[0] - q.frames[0]);
    for (let i = 1; i < pieces.length && !joined; i++) {
      const A = pieces[i - 1], B = pieces[i], a = A.frames.at(-1), b = B.frames[0];
      if (!mapper.images[a].flowH || !mapper.images[b].flowH) continue;  // no flow (photos): nothing to join by
      if (a > b || time(b) - time(a) > o.maxGapSeconds) continue;  // they overlap in time, or too far apart
      stats.tried++;
      const rot = flowRotation(mapper, a, b, o);
      const tag = `flow join of keyframes ${a} and ${b} (${time(a).toFixed(2)}-${time(b).toFixed(2)} s)`;
      if (!rot?.R) { o.log(`${tag}: ${rot ? `not a turn in place (spread ${rot.spread.toFixed(3)})` : 'no flow between them'}`); continue; }
      const near = (frames, fromEnd) => frames.filter(k => Math.abs(time(k) - time(fromEnd)) <= o.speedSeconds);
      const vA = speed(mapper, near(A.frames, a), o.speedQuantile), vB = speed(mapper, near(B.frames, b), o.speedQuantile);
      if (!(vA > 0 && vB > 0)) { o.log(`${tag}: a piece does not move (speeds ${vA}, ${vB})`); continue; }
      const s = vA / vB, Ra = mapper.poses[a].R, Ca = poseCenter(mapper.poses[a]);
      const RbA = mul3(rot.R, Ra), GR = mul3(transpose3(RbA), mapper.poses[b].R), r = mulVec3(GR, poseCenter(mapper.poses[b]));
      transformSegment(mapper, B.seg, { s, R: GR, t: [Ca[0] - s * r[0], Ca[1] - s * r[1], Ca[2] - s * r[2]] });
      relabel(mapper, B.seg, A.seg);
      let carried = 0;
      for (let k = a + 1; k < b; k++) {
        if (mapper.poses[k]) continue;
        const rk = flowRotation(mapper, a, k, { maxSpread: Infinity });
        if (!rk?.R) continue;
        mapper.poses[k] = poseFromCenter(mul3(rk.R, Ra), Ca);
        mapper.segment[k] = A.seg;
        mapper.stats.registered++;
        carried++;
      }
      const turn = deg(Math.acos(Math.max(-1, Math.min(1, (rot.R[0] + rot.R[4] + rot.R[8] - 1) / 2))));
      o.log(`${tag}: turned ${turn.toFixed(1)} deg in place (spread ${rot.spread.toFixed(3)}), piece ${B.seg} (${B.frames.length} keyframes) ` +
            `joins piece ${A.seg} at scale ${s.toFixed(3)} (speeds ${vA.toPrecision(3)}, ${vB.toPrecision(3)} per s), ${carried} keyframes between placed by the flow`);
      stats.joins.push({ from: B.seg, into: A.seg, a, b, turnDeg: turn, spread: rot.spread, scale: s, carried });
      joined = true;
    }
  }
  return stats;
}
