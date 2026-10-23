// frontend.js — Features to matched pairs (DESIGN-sfm §5.2-5.4, S1-S3 inputs): sequential window
// matching on the GPU, geometric verification of every pair (fundamental matrix RANSAC, COLMAP's
// 4 px), and loop pairs between frames that a loop detector (loopdetect.js) proposes. The result
// is the `input` shape that reconstruct.js takes (as synth.js / dump.js produce).
"use strict";

import { ransacFundamental } from './geometry.js';
import { matchedPixels } from './util.js';

export const FRONTEND_DEFAULTS = { window: 10, maxError: 4, minInliers: 15 };

// Inlier matches of a pair under a RANSAC fundamental matrix, or null (SIFT keypoints: 4 floats each)
export function verifyPair(kpA, kpB, matches, { rng, maxError = 4, minInliers = 15 }) {
  const n = matches.length / 2;
  if (n < Math.max(8, minInliers)) return null;
  const [x1, x2] = matchedPixels(kpA, kpB, matches, 4);
  const F = ransacFundamental(x1, x2, { rng, threshold: maxError });
  if (!F || F.numInliers < minInliers) return null;
  const out = new Uint32Array(2 * F.numInliers);
  let o = 0;
  for (let k = 0; k < n; k++) if (F.inliers[k]) { out[o++] = matches[2 * k]; out[o++] = matches[2 * k + 1]; }
  return { matches: out, F: F.model };
}

// features: [{ name, keypoints: Float32Array(4 n), descriptors: Uint8Array(128 n) }] in time order.
// loopCandidates: [[i, j]] (i < j) to match as well. Returns { images, pairs, stats }.
export async function matchFeatures(features, matcher, { rng, loopCandidates = [], onProgress = () => {}, ...options } = {}) {
  const o = { ...FRONTEND_DEFAULTS, ...options };
  const n = features.length;
  const todo = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j <= Math.min(n - 1, i + o.window); j++) todo.push([i, j, 'seq']);
  for (const [i, j] of loopCandidates) if (j - i > o.window) todo.push([i, j, 'loop']);
  const { pairs, stats } = await matchPairs(features, matcher, todo, { rng, onProgress, ...o });
  return { images: features.map(f => ({ name: f.name, time: f.time, keypoints: xyOnly(f.keypoints) })), pairs, stats };
}

// Match and verify given pairs [[i, j, type]]: one GPU batch per first frame, descriptors kept on
// the GPU (cached by frame index) from the first batch that uses a frame to the last. Returns
// { pairs (verified, sorted), stats }.
export async function matchPairs(features, matcher, todo, { rng, onProgress = () => {}, ...options } = {}) {
  const o = { ...FRONTEND_DEFAULTS, ...options };
  const pairs = [];
  const stats = { matched: 0, verified: 0, rawMatches: 0, inliers: 0, loopsTried: 0, loopsVerified: 0 };
  const byFirst = new Map();
  todo.forEach((t, k) => { if (!byFirst.has(t[0])) byFirst.set(t[0], []); byFirst.get(t[0]).push(k); });
  const batches = [...byFirst.values()], lastBatch = new Map();
  batches.forEach((ks, b) => { for (const k of ks) { lastBatch.set(todo[k][0], b); lastBatch.set(todo[k][1], b); } });
  matcher.clear();  // the cache keys are this call's frame indices
  let done = 0;
  for (const [batch, ks] of batches.entries()) {
    const entries = ks.map(k => [matcher.upload(features[todo[k][0]].descriptors, todo[k][0]), matcher.upload(features[todo[k][1]].descriptors, todo[k][1])]);
    const results = await matcher.matchBatch(entries, o.match ?? {});  // match: MATCH_DEFAULTS overrides (Retry loosens them)
    for (const x of new Set(ks.flatMap(k => todo[k].slice(0, 2)))) if (lastBatch.get(x) === batch) matcher.release(x);
    ks.forEach((k, b) => {
      const [i, j, type] = todo[k], raw = results[b];
      stats.matched++; stats.rawMatches += raw.length / 2;
      if (type === 'loop') stats.loopsTried++;
      const v = verifyPair(features[i].keypoints, features[j].keypoints, raw, { rng, ...o });
      if (v) {
        pairs.push({ i, j, type, config: 'UNCALIBRATED', F: v.F, matches: v.matches });
        stats.verified++; stats.inliers += v.matches.length / 2;
        if (type === 'loop') stats.loopsVerified++;
      }
    });
    done += ks.length;
    onProgress(done, todo.length);
  }
  pairs.sort((a, b) => a.i - b.i || a.j - b.j);
  return { pairs, stats };
}

// reconstruct.js / mapper.js read keypoints as [x, y] pairs
export const xyOnly = (kp4) => {
  const out = new Float32Array(kp4.length / 2);
  for (let k = 0; k < kp4.length / 4; k++) { out[2 * k] = kp4[4 * k]; out[2 * k + 1] = kp4[4 * k + 1]; }
  return out;
};
