// reference.js — Defects of a reference reconstruction (COLMAP), found from independent evidence,
// so that comparisons skip them instead of matching them (DESIGN-sfm R1):
//   - weak frames: registered from few observations
//   - scale breaks: a consecutive step whose baseline is many times the typical baseline / depth
//   - arbitrated steps: the estimate and the reference disagree on a relative rotation, and the
//     two-view rotation from that pair's own matches (when trustworthy: geometry.js twoViewPose)
//     sides with the estimate
// The reference is then cut into consistent pieces (between breaks, without weak frames), each
// compared with its own Sim(3).
"use strict";

import { rotationAngle } from './linalg.js';
import { poseCenter } from './camera.js';
import { twoViewPose } from './geometry.js';
import { compareModels, compareLocal, checkThresholds } from './compare.js';
import { median, deg, matchedPixels, relativeRotation } from './util.js';

// ref: model with images [{ name, R, t, numObs, medianDepth }] (fromColmap fills both).
// est (with its camera) / input (its pairs and keypoints) / rng enable the arbitration.
// Returns { badFrames, breaks }: Map(name -> reason), a break being keyed by its second frame.
export function referenceDefects(ref, { est = null, input = null, rng = null, minObs = 150, breakFactor = 10,
                                         minBreak = 0.3, disagreeDeg = 0.2 } = {}) {
  const ims = [...ref.images].sort((a, b) => a.name.localeCompare(b.name));
  const badFrames = new Map(), breaks = new Map();
  const obsFloor = Math.max(minObs, 0.1 * median(ims.map(im => im.numObs ?? Infinity)));
  for (const im of ims) if (im.numObs !== undefined && im.numObs < obsFloor) badFrames.set(im.name, `weak: ${im.numObs} observations`);

  const steps = [], centers = ims.map(poseCenter);
  for (let k = 1; k < ims.length; k++) {
    const b = Math.hypot(...centers[k].map((x, i) => x - centers[k - 1][i]));
    if (ims[k].medianDepth > 0) steps.push({ name: ims[k].name, bd: b / ims[k].medianDepth });
  }
  const limit = Math.max(minBreak, breakFactor * median(steps.map(s => s.bd)));
  for (const s of steps) if (s.bd > limit) breaks.set(s.name, `scale break: baseline ${s.bd.toFixed(2)} x depth`);

  if (est && input && rng) {
    const refByName = new Map(ims.map(im => [im.name, im]));
    const index = new Map(input.images.map((im, k) => [im.name, k]));
    const pairOf = new Map(input.pairs.flatMap(p => [[`${p.i},${p.j}`, p], [`${p.j},${p.i}`, p]]));
    const estIms = est.images.filter(im => refByName.has(im.name)).sort((a, b) => a.name.localeCompare(b.name));
    for (let k = 1; k < estIms.length; k++) {
      const a = estIms[k - 1], b = estIms[k];
      if (a.segment !== b.segment) continue;
      const ra = refByName.get(a.name), rb = refByName.get(b.name);
      const relE = relativeRotation(a.R, b.R), relR = relativeRotation(ra.R, rb.R);
      if (deg(rotationAngle(relE, relR)) <= disagreeDeg) continue;
      const i = index.get(a.name), j = index.get(b.name);
      const pair = pairOf.get(`${i},${j}`);
      if (!pair || pair.matches.length < 2 * 50) continue;
      const [x1, x2] = pair.i === i ? matchedPixels(input.images[i].keypoints, input.images[j].keypoints, pair.matches)
                                    : matchedPixels(input.images[j].keypoints, input.images[i].keypoints, pair.matches).reverse();
      const rel = twoViewPose(x1, x2, est.camera, { rng });
      if (!rel) continue;
      const toEst = deg(rotationAngle(rel.R, relE)), toRef = deg(rotationAngle(rel.R, relR));
      if (toEst < 0.5 * toRef) breaks.set(b.name, `two-view rotation agrees with the estimate (${toEst.toFixed(2)} deg) not the reference (${toRef.toFixed(2)} deg)`);
    }
  }
  return { badFrames, breaks };
}

// Consistent pieces of the reference: name-sorted runs between breaks, without weak frames
function consistentPieces(ref, { badFrames, breaks }) {
  const pieces = [];
  let cur = [];
  for (const im of [...ref.images].sort((a, b) => a.name.localeCompare(b.name))) {
    if (badFrames.has(im.name)) { if (cur.length) pieces.push(cur); cur = []; continue; }
    if (breaks.has(im.name) && cur.length) { pieces.push(cur); cur = []; }
    cur.push(im.name);
  }
  if (cur.length) pieces.push(cur);
  return pieces;
}

// Sim(3) comparison per consistent piece, further split by estimated segment. Pieces shorter than
// minFrames are reported as skipped. Returns [{ first, last, size, segment, metrics }].
export function comparePieces(est, ref, defects, { minFrames = 10 } = {}) {
  const estByName = new Map(est.images.map(im => [im.name, im]));
  const refByName = new Map(ref.images.map(im => [im.name, im]));
  const out = [];
  for (const piece of consistentPieces(ref, defects)) {
    const bySeg = new Map();
    for (const name of piece) {
      const e = estByName.get(name);
      if (!e) continue;
      const seg = e.segment ?? 0;
      if (!bySeg.has(seg)) bySeg.set(seg, []);
      bySeg.get(seg).push(name);
    }
    for (const [segment, names] of bySeg) {
      const row = { first: names[0], last: names.at(-1), size: names.length, segment };
      if (names.length < minFrames) { out.push({ ...row, skipped: true }); continue; }
      const metrics = compareModels({ ...est, images: names.map(n => estByName.get(n)) },
                                    { ...ref, images: names.map(n => refByName.get(n)) });
      out.push({ ...row, metrics });
    }
  }
  return out;
}

// Exclusion sets for compareLocal: names of bad frames, and names starting a break
export const exclusions = ({ badFrames, breaks }) => ({ frames: new Set(badFrames.keys()), breaks: new Set(breaks.keys()) });

// Everything a tool or test reports against a COLMAP reference: its defects (excluded), the local
// metric, a Sim(3) per consistent piece, and the failed `colmap` thresholds (local rotation over
// 0.05 deg, or any piece of at least minFrames).
export function assessAgainstReference(est, ref, { input = null, rng = null, minFrames = 10 } = {}) {
  const defects = referenceDefects(ref, { est, input, rng });
  const local = compareLocal(est, ref, { exclude: exclusions(defects) });
  const pieces = comparePieces(est, ref, defects, { minFrames });
  const fails = [];
  for (const p of pieces.filter(p => !p.skipped)) {
    for (const f of checkThresholds({ ...p.metrics, registered: 1 }, 'colmap')) fails.push(`${p.first}..${p.last}: ${f}`);
  }
  if (local.medianRelRotDeg > 0.05) fails.push(`local relative rotation ${local.medianRelRotDeg.toFixed(3)} deg > 0.05`);
  return { defects, local, pieces, fails };
}

