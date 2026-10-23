// quality.js — Per-keyframe quality signals (DESIGN-sfm §6.1) of a SequentialMapper state, combined
// into a score (green / amber / red) with its dominant reason:
//   - registration, and whether the frame is in the exported segment
//   - support (tracked points) and mean reprojection error
//   - parallax to the previous keyframe (rotation-only stretches)
//   - two-view agreement: the relative pose of consecutive keyframes from their own matches
//     (essential matrix) against the model's, rotation and baseline direction
//   - scale breaks: a baseline / depth step far above the segment's typical one
//   - after training: the view's validation PSNR against the others'
// Path smoothness (a pose against the time interpolation of its neighbours) is reported but not
// scored: drone footage keyframed by optical flow stops, starts and yaws in place (DESIGN-sfm R5).
// Loop edges come with their residual after the pose graph (loop.js stats.loops); on request they are
// checked on the current map: one frame relocalized on the other's points (see relocalize).
"use strict";

import { makeRng } from '../math.js';
import { sub } from '../math.js';
import { poseCenter } from './camera.js';
import { mul3, mulVec3, rotationAngle, expSO3, logSO3 } from './linalg.js';
import { twoViewPose, depthIn, ransacPnP } from './geometry.js';
import { median, deg, angleBetween, relativeRotation, hashString } from './util.js';

const DEFAULTS = {
  minSupportRel: 0.1, weakSupportRel: 0.3,  // x the median number of points per keyframe
  maxReprojPx: 2,
  minParallaxDeg: 0.05,                     // median triangulation angle to the previous keyframe
  // model vs two-view relative rotation: above max(floor, factor x the video's median disagreement)
  // (two-view precision depends on the matches: 0.03 deg median on jardin5, 0.3 on small synthetic frames)
  twoViewRotDeg: { amber: 0.3, red: 1, amberFactor: 5, redFactor: 10 },
  twoViewDirDeg: { amber: 20, red: 45 },    // baseline direction, when the parallax exceeds 1 deg
  maxHomographyRatio: 0.7, maxSpreadDeg: 0.3,  // two-view pose trusted: H explains < 70% of F's inliers, 2 runs agree
  breakFactor: 10, minBreak: 0.3,           // scale break: baseline / depth > max(minBreak, factor x median)
  loopRotDeg: 2, loopLogScale: 0.05,        // loop residual after the pose graph: suspect above
  // checked loop: suspect when a relocalized frame is this far from its pose (jardin4: at most 3 deg,
  // 6% of the depth over 447 loops; gare-st-lazare, two passes of a square: 5% for half of its loops)
  loopRelocDeg: 3, loopRelocOffset: 0.1,
  lowPsnrDb: 3,                             // after training: below the median view PSNR by this
};

const norm = (v) => Math.hypot(v[0], v[1], v[2]);
const angleDeg = (u, v) => deg(angleBetween(u, v));

// Two-view relative pose of keyframes prev -> k (geometry.js twoViewPose): depends on their
// matches and on the camera's intrinsics
function pairPose(mapper, prev, k, rng, { maxHomographyRatio, maxSpreadDeg }) {
  const e = mapper.adj[prev].find(e => e.other === k);
  if (!e || e.a.length < 100) return null;
  const [x1, x2] = mapper.pairPixels(prev, e);
  return twoViewPose(x1, x2, mapper.camera, { rng, maxHomographyRatio, maxSpreadDeg });
}

// A loop edge (i, j) on the current map: j relocalized (PnP) on the 3D points of i's keypoints that
// the pair matched, against j's pose: { rotDeg, offset (center distance / median depth) }, or null
// when there are too few such points. Scale-aware, and independent of how the loop was closed.
function relocalize(mapper, i, j, rng) {
  const e = mapper.adj[i].find(e => e.other === j);
  if (!e) return null;
  const rays = [], pixels = [], points = [];
  for (let m = 0; m < e.a.length; m++) {
    const p = mapper.kpPoint[i][e.a[m]];
    if (p < 0 || !mapper.points[p]) continue;
    rays.push(mapper.ray(j, e.b[m])); pixels.push(mapper.pixel(j, e.b[m])); points.push(mapper.points[p].X);
  }
  if (points.length < 30) return null;
  const r = ransacPnP(rays, pixels, points, mapper.camera, { rng, threshold: 4 });
  if (!r || r.numInliers < 30) return null;
  const Q = mapper.poses[j], depth = median(points.filter((_, k) => r.inliers[k]).map(X => depthIn(Q, X)));
  return { rotDeg: deg(rotationAngle(r.model.R, Q.R)), offset: norm(sub(poseCenter(r.model), poseCenter(Q))) / depth };
}

// Model vs two-view: { rotDeg, dirDeg }
function twoViewAgreement(mapper, prev, k, rel) {
  const P = mapper.poses[prev], Q = mapper.poses[k];
  const R = relativeRotation(P.R, Q.R), Rt = mulVec3(R, P.t);
  return { rotDeg: deg(rotationAngle(rel.R, R)), dirDeg: angleDeg(rel.t, sub(Q.t, Rt)) };
}

// mapper: SequentialMapper. loops: loop.js stats.loops; checkLoops: relocalize each one, both ways
// (the worse kept as loop.reloc, which then decides `suspect`). main: the exported segment (default the
// largest). times: keyframe timestamps (default: the indices). viewPsnr: optional Map(name ->
// validation PSNR) from the trainer. seed: of the two-view RANSAC, per pair (a cached result does
// not depend on what was computed before it). cache: a Map kept by the caller across updates,
// for the two-view poses (keyed by keyframe names).
// Returns { frames: [{ index, name, registered, segment, support, trackLength, reprojPx, depth,
//           parallaxDeg, twoView, breakRatio, pathDev, rotDev, psnr, score, reason }], loops, counts, main }
export function keyframeQuality(mapper, { loops = [], checkLoops = false, main = null, times = null, viewPsnr = null, seed = 1, cache = new Map(), ...options } = {}) {
  const o = { ...DEFAULTS, ...options };
  const n = mapper.n;
  const sizes = new Map();
  for (let k = 0; k < n; k++) if (mapper.poses[k]) sizes.set(mapper.segment[k], (sizes.get(mapper.segment[k]) ?? 0) + 1);
  main ??= [...sizes].sort((a, b) => b[1] - a[1])[0]?.[0] ?? -1;

  const frames = Array.from({ length: n }, (_, k) => ({
    index: k, name: mapper.images[k].name, registered: !!mapper.poses[k], segment: mapper.segment[k],
    support: 0, trackLength: 0, reprojPx: NaN, depth: NaN, parallaxDeg: NaN, twoView: null, breakRatio: NaN,
    pathDev: NaN, rotDev: NaN, psnr: viewPsnr?.get(mapper.images[k].name) ?? NaN,
  }));
  const centers = mapper.poses.map(p => p && poseCenter(p));
  for (const f of frames) {
    if (!f.registered) continue;
    let err = 0, len = 0;
    const kp = mapper.kpPoint[f.index], depths = [];
    for (let a = 0; a < kp.length; a++) {
      const P = kp[a] >= 0 ? mapper.points[kp[a]] : null;
      if (!P) continue;
      f.support++; len += P.obs.size;
      err += mapper.reprojErr(f.index, a, P.X);
      if (depths.length < 1000) depths.push(depthIn(mapper.poses[f.index], P.X));
    }
    f.trackLength = f.support ? len / f.support : 0;
    f.reprojPx = f.support ? err / f.support : NaN;
    f.depth = depths.length ? median(depths) : NaN;
  }

  // per segment, in time order
  const time = (k) => times ? times[k] : k;
  for (const seg of sizes.keys()) {
    const ks = frames.filter(f => f.registered && f.segment === seg).map(f => f.index);
    for (let m = 1; m < ks.length; m++) {
      const k = ks[m], prev = ks[m - 1], angles = [];
      const kp = mapper.kpPoint[k];
      for (let a = 0; a < kp.length && angles.length < 500; a++) {
        const P = kp[a] >= 0 ? mapper.points[kp[a]] : null;
        if (P && P.obs.has(prev)) angles.push(angleDeg(sub(P.X, centers[k]), sub(P.X, centers[prev])));
      }
      if (angles.length) frames[k].parallaxDeg = median(angles);
      const key = `${mapper.images[prev].name}|${mapper.images[k].name}`;
      if (!cache.has(key)) cache.set(key, pairPose(mapper, prev, k, makeRng(seed + hashString(key)), o));
      if (cache.get(key)) frames[k].twoView = twoViewAgreement(mapper, prev, k, cache.get(key));
      frames[k].breakRatio = norm(sub(centers[k], centers[prev])) / frames[k].depth;
    }
    const steps = ks.slice(1).map((k, m) => norm(sub(centers[k], centers[ks[m]])));
    const typical = median(steps) || 1;
    for (let m = 1; m + 1 < ks.length; m++) {
      const [a, k, c] = [ks[m - 1], ks[m], ks[m + 1]], w = (time(k) - time(a)) / (time(c) - time(a));
      frames[k].pathDev = norm(sub(centers[k], centers[a].map((x, d) => x + w * (centers[c][d] - x)))) / typical;
      const Ra = mapper.poses[a].R, rel = logSO3(relativeRotation(Ra, mapper.poses[c].R));
      frames[k].rotDev = deg(rotationAngle(mapper.poses[k].R, mul3(expSO3(rel.map(x => w * x)), Ra)));
    }
  }

  const reg = frames.filter(f => f.registered);
  const typicalSupport = reg.length ? median(reg.map(f => f.support)) : 0;
  const ratios = reg.map(f => f.breakRatio).filter(Number.isFinite);
  const breakLimit = Math.max(o.minBreak, o.breakFactor * (ratios.length ? median(ratios) : 0));
  const psnrs = frames.map(f => f.psnr).filter(Number.isFinite), typicalPsnr = psnrs.length ? median(psnrs) : NaN;
  const tvs = reg.filter(f => f.twoView).map(f => f.twoView.rotDeg), tvTypical = tvs.length >= 5 ? median(tvs) : 0;
  const rotAmber = Math.max(o.twoViewRotDeg.amber, o.twoViewRotDeg.amberFactor * tvTypical);
  const rotRed = Math.max(o.twoViewRotDeg.red, o.twoViewRotDeg.redFactor * tvTypical);
  const counts = { green: 0, amber: 0, red: 0 };
  for (const f of frames) {
    const tv = f.twoView, dir = tv && f.parallaxDeg >= 1 ? tv.dirDeg : 0;  // direction: needs parallax
    const red = !f.registered ? 'not registered'
      : f.segment !== main ? `separate map (segment ${f.segment}), not exported`
      : f.support < o.minSupportRel * typicalSupport ? `weak: ${f.support} points`
      : f.reprojPx > o.maxReprojPx ? `reprojection ${f.reprojPx.toFixed(2)} px`
      : f.breakRatio > breakLimit ? `scale break: baseline ${f.breakRatio.toFixed(2)} x depth`
      : tv && tv.rotDeg > rotRed ? `rotation disagrees with its matches by ${tv.rotDeg.toFixed(2)} deg`
      : dir > o.twoViewDirDeg.red ? `baseline direction disagrees with its matches by ${dir.toFixed(0)} deg`
      : null;
    const amber = red ? null
      : f.support < o.weakSupportRel * typicalSupport ? `few points: ${f.support}`
      : tv && tv.rotDeg > rotAmber ? `rotation off its matches by ${tv.rotDeg.toFixed(2)} deg`
      : dir > o.twoViewDirDeg.amber ? `baseline direction off its matches by ${dir.toFixed(0)} deg`
      : f.parallaxDeg < o.minParallaxDeg ? `no parallax (${f.parallaxDeg.toFixed(2)} deg): rotation only`
      : f.psnr < typicalPsnr - o.lowPsnrDb ? `trains badly: ${f.psnr.toFixed(1)} dB`
      : null;
    f.score = red ? 'red' : amber ? 'amber' : 'green';
    f.reason = red ?? amber ?? '';
    counts[f.score]++;
  }
  const rng = makeRng(seed);
  const loopsOut = loops.map(l => {
    if (!checkLoops) return { ...l, suspect: l.rotDeg > o.loopRotDeg || Math.abs(l.logScale) > o.loopLogScale };
    const both = mapper.poses[l.i] && mapper.poses[l.j] ? [relocalize(mapper, l.i, l.j, rng), relocalize(mapper, l.j, l.i, rng)].filter(Boolean) : [];
    const reloc = both.reduce((w, r) => (!w || r.offset > w.offset ? r : w), null);
    return { ...l, reloc, suspect: !!reloc && (reloc.rotDeg > o.loopRelocDeg || reloc.offset > o.loopRelocOffset) };
  });
  return { frames, loops: loopsOut, counts, main };
}
