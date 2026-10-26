// compare.js — Accuracy of a reconstruction against a reference (ground truth or COLMAP), DESIGN-sfm
// §8.2: images are matched by name, the estimate is Sim(3)-aligned onto the reference (rotation
// from the camera orientations, scale and translation from the centers), then position / rotation
// / focal errors are reported; compareLocal is the gauge-free variant. The thresholds of §8.3 live
// here too, so tests and tools share them. A model is { images: [{ name, R, t, segment? }], focal,
// numPoints? } (fromColmap, mapper.model(), synth.js truth.model).
"use strict";

import { sub } from '../math.js';
import { mul3, transpose3, mulVec3, rotationAngle, projectToSO3 } from './linalg.js';
import { poseCenter } from './camera.js';
import { apply as applySim3 } from './sim3.js';
import { mean } from './align.js';
import { median, quantile, deg, angleBetween, relativeRotation } from './util.js';

// DESIGN-sfm §8.3. `registered` is the fraction of the reference's images (so "COLMAP - 1%" is
// 0.99 too); position errors are relative to the reference path length (compareModels).
export const THRESHOLDS = {
  truth:  { registered: 0.99, rmsCenterRel: 0.003, medianRotDeg: 0.2, focalErr: 0.01 },
  colmap: { registered: 0.99, rmsCenterRel: 0.01, medianRotDeg: 0.5, focalErr: 0.02 },
};

// Dataset from colmap.js buildDataset() / loadColmap*() -> model. Each image also gets its number
// of observations and median depth (reference.js uses them to find defects of the reference).
export function fromColmap({ images, points3D }) {
  const cam = images[0]?.camera;
  const depthOf = (im) => {
    if (!points3D?.pointIndexById?.size) return NaN;
    const d = [];
    for (let o = 0; o < im.obs2D.length; o += 3) {
      const i = points3D.pointIndexById.get(im.obs2D[o + 2]);
      if (i === undefined) continue;
      const p = points3D.positions;
      d.push(im.R[6] * p[3 * i] + im.R[7] * p[3 * i + 1] + im.R[8] * p[3 * i + 2] + im.t[2]);
    }
    return median(d);
  };
  return {
    images: images.map(im => ({ name: im.name, R: im.R, t: im.t, numObs: im.obs2D ? im.obs2D.length / 3 : undefined, medianDepth: depthOf(im) })),
    focal: cam ? (cam.fx + cam.fy) / 2 : NaN,
    numPoints: points3D?.count ?? 0,
  };
}

export function compareModels(est, ref) {
  const byName = new Map(est.images.map(im => [im.name, im]));
  const refSorted = [...ref.images].sort((a, b) => a.name.localeCompare(b.name));
  const common = refSorted.filter(im => byName.has(im.name));
  const refCenters = refSorted.map(poseCenter);
  let pathLength = 0;
  for (let i = 1; i < refCenters.length; i++) pathLength += Math.hypot(...refCenters[i].map((x, k) => x - refCenters[i - 1][k]));

  const result = {
    numRef: ref.images.length, numEst: est.images.length, numCommon: common.length,
    registered: ref.images.length ? common.length / ref.images.length : 0,
    pathLength, focalRatio: est.focal / ref.focal, focalErr: Math.abs(est.focal / ref.focal - 1),
    numPoints: est.numPoints ?? 0, numPointsRef: ref.numPoints ?? 0,
  };
  if (common.length < 3) return { ...result, rmsCenterRel: Infinity, medianRotDeg: Infinity, perImage: [] };

  const src = common.map(im => poseCenter(byName.get(im.name)));
  const dst = common.map(poseCenter);
  // Rotation from the camera orientations (chordal mean of R_ref^T R_est), then scale and
  // translation from the centers: well posed for straight paths and for hovering, where an
  // alignment of the centers alone is not.
  const M = new Array(9).fill(0);
  common.forEach(im => { const A = mul3(transpose3(im.R), byName.get(im.name).R); for (let k = 0; k < 9; k++) M[k] += A[k]; });
  const A = projectToSO3(M);
  const ms = mean(src), md = mean(dst);
  let num = 0, den = 0;
  for (let i = 0; i < src.length; i++) {
    const a = mulVec3(A, src[i].map((x, k) => x - ms[k])), b = dst[i].map((x, k) => x - md[k]);
    num += a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    den += a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
  }
  const s = den > 0 ? num / den : 1, Ams = mulVec3(A, ms);
  const sim = { s, R: A, t: [0, 1, 2].map(k => md[k] - s * Ams[k]) };
  // x_ref = s A x_est + b  =>  an estimated camera seen in the reference frame has R' = R_est A^T
  const At = transpose3(sim.R);
  const perImage = common.map((im, i) => {
    const e = byName.get(im.name);
    const c = applySim3(sim, src[i]);
    return {
      name: im.name,
      centerErr: Math.hypot(c[0] - dst[i][0], c[1] - dst[i][1], c[2] - dst[i][2]),
      rotDeg: deg(rotationAngle(mul3(e.R, At), im.R)),
    };
  });
  const rms = Math.sqrt(perImage.reduce((s, p) => s + p.centerErr ** 2, 0) / perImage.length);
  const rot = perImage.map(p => p.rotDeg);
  // Relative to the path length, or to the scene depth when the camera barely moves (hovering)
  const depth = median(common.map(im => im.medianDepth).filter(d => d > 0));
  const norm = Math.max(pathLength, depth > 0 ? depth : 0) || 1;
  return {
    ...result, sim3: sim, perImage, scaleNorm: norm,
    rmsCenter: rms, rmsCenterRel: rms / norm,
    maxCenterRel: Math.max(...perImage.map(p => p.centerErr)) / norm,
    medianRotDeg: median(rot), maxRotDeg: Math.max(...rot),
  };
}

// Gauge-free local accuracy: for consecutive estimated frames (same segment, both in the reference,
// not excluded by `exclude` = reference.js exclusions()), the error of the relative rotation and of
// the baseline direction (camera frame). Unlike a global Sim(3) fit, it stays meaningful when
// either model has scale breaks or several segments. Reference baselines under `minBaselineRel` of
// the median are skipped for the direction (pure rotation has none).
export function compareLocal(est, ref, { minBaselineRel = 0.2, exclude = null } = {}) {
  const refByName = new Map(ref.images.map(im => [im.name, im]));
  const ims = est.images.filter(im => refByName.has(im.name)).sort((a, b) => a.name.localeCompare(b.name));
  const pairs = [];
  for (let k = 1; k < ims.length; k++) {
    const a = ims[k - 1], b = ims[k];
    if (a.segment !== undefined && a.segment !== b.segment) continue;
    if (exclude && (exclude.frames.has(a.name) || exclude.frames.has(b.name) || exclude.breaks.has(b.name))) continue;
    const ra = refByName.get(a.name), rb = refByName.get(b.name);
    const relE = relativeRotation(a.R, b.R), relR = relativeRotation(ra.R, rb.R);
    const dE = mulVec3(b.R, sub(poseCenter(a), poseCenter(b))), dR = mulVec3(rb.R, sub(poseCenter(ra), poseCenter(rb)));
    pairs.push({ a: a.name, b: b.name, rotDeg: deg(rotationAngle(relE, relR)), dirDeg: deg(angleBetween(dE, dR)), refBaseline: Math.hypot(...dR) });
  }
  const medBase = median(pairs.map(p => p.refBaseline));
  const dirPairs = pairs.filter(p => p.refBaseline >= minBaselineRel * medBase);
  return {
    numPairs: pairs.length,
    medianRelRotDeg: median(pairs.map(p => p.rotDeg)), p90RelRotDeg: quantile(pairs.map(p => p.rotDeg), 0.9),
    medianDirDeg: median(dirPairs.map(p => p.dirDeg)), p90DirDeg: quantile(dirPairs.map(p => p.dirDeg), 0.9),
    pairs,
  };
}

// Failed checks of `metrics` (from compareModels) against THRESHOLDS[kind]; empty when all pass
export function checkThresholds(metrics, kind) {
  const T = THRESHOLDS[kind];
  if (!T) throw new Error(`Unknown threshold set ${kind}`);
  const fails = [];
  if (metrics.registered < T.registered) fails.push(`registered ${(100 * metrics.registered).toFixed(1)}% < ${100 * T.registered}%`);
  if (!(metrics.rmsCenterRel <= T.rmsCenterRel)) fails.push(`RMS center ${(100 * metrics.rmsCenterRel).toFixed(3)}% of path > ${100 * T.rmsCenterRel}%`);
  if (!(metrics.medianRotDeg <= T.medianRotDeg)) fails.push(`median rotation ${metrics.medianRotDeg.toFixed(3)} deg > ${T.medianRotDeg}`);
  if (!(metrics.focalErr <= T.focalErr)) fails.push(`focal error ${(100 * metrics.focalErr).toFixed(2)}% > ${100 * T.focalErr}%`);
  return fails;
}

export function formatMetrics(m) {
  const pct = (x) => `${(100 * x).toFixed(3)}%`;
  return [
    `images      ${m.numCommon} common / ${m.numRef} ref / ${m.numEst} est (registered ${(100 * m.registered).toFixed(1)}%)`,
    `path length ${m.pathLength.toFixed(3)} (reference units)`,
    `centers     RMS ${pct(m.rmsCenterRel)} of the path (or scene depth when hovering), max ${pct(m.maxCenterRel)}`,
    `rotations   median ${m.medianRotDeg?.toFixed(3)} deg, max ${m.maxRotDeg?.toFixed(3)} deg`,
    `focal       ratio ${m.focalRatio.toFixed(4)} (error ${(100 * m.focalErr).toFixed(2)}%)`,
    `points      ${m.numPoints} est / ${m.numPointsRef} ref`,
    `scale       ${m.sim3 ? m.sim3.s.toPrecision(5) : '-'} (est -> ref)`,
  ].join('\n');
}
