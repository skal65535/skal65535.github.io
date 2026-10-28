// frontend.js — Features to matched pairs (DESIGN-sfm §5.2-5.4, S1-S3 inputs): sequential window
// matching on the GPU, geometric verification of every pair (fundamental matrix RANSAC, COLMAP's
// 4 px), and loop pairs between frames that a loop detector (loopdetect.js) proposes. The result
// is the `input` shape that reconstruct.js takes (as synth.js / dump.js produce).
"use strict";

import { ransacFundamental } from './geometry.js';
import { makeRng } from '../math.js';

const FRONTEND_DEFAULTS = { window: 10, maxError: 4, minInliers: 15 };

// The matched pixels of a pair, [u1, v1, u2, v2] per match (SIFT keypoints: 4 floats each)
export function packPair(kpA, kpB, matches) {
  const n = matches.length / 2, xy = new Float32Array(4 * n);
  for (let k = 0; k < n; k++) {
    const a = 4 * matches[2 * k], b = 4 * matches[2 * k + 1];
    xy[4 * k] = kpA[a]; xy[4 * k + 1] = kpA[a + 1]; xy[4 * k + 2] = kpB[b]; xy[4 * k + 3] = kpB[b + 1];
  }
  return xy;
}

// The inliers (indices of the packed matches) under a RANSAC fundamental matrix drawn from its
// own seed, { inliers, F }, or null
export function verifyPacked(xy, seed, { maxError = 4, minInliers = 15 } = {}) {
  const n = xy.length / 4;
  if (n < Math.max(8, minInliers)) return null;
  const x1 = new Array(n), x2 = new Array(n);
  for (let k = 0; k < n; k++) { x1[k] = [xy[4 * k], xy[4 * k + 1]]; x2[k] = [xy[4 * k + 2], xy[4 * k + 3]]; }
  const F = ransacFundamental(x1, x2, { rng: makeRng(seed), threshold: maxError });
  if (!F || F.numInliers < minInliers) return null;
  const inliers = new Uint32Array(F.numInliers);
  let o = 0;
  for (let k = 0; k < n; k++) if (F.inliers[k]) inliers[o++] = k;
  return { inliers, F: F.model };
}

// A pair's RANSAC seed from the call's base seed: the same whichever worker verifies it, and in
// whatever order
function pairSeed(base, i, j) {
  let h = Math.imul(base ^ i, 0x9E3779B1);
  h = Math.imul(h ^ (h >>> 16) ^ j, 0x85EBCA77);
  return (h ^ (h >>> 13)) >>> 0;
}

const verifyInline = (job) => {
  const t = performance.now(), result = verifyPacked(job.xy, job.seed, job);
  return { result, ms: performance.now() - t };
};

// The verification pool: `size` workers (verify_worker.js), made for one matchPairs call and
// ended with it. null where there is no Worker (Node) or none can start: the caller verifies
// inline, with the same results. A worker that fails hands its pairs back to inline verification.
const POOL_MIN_PAIRS = 32;  // below, the workers' start costs more than they save
const defaultVerifyWorkers = () => Math.max(1, Math.min(8, (globalThis.navigator?.hardwareConcurrency || 4) - 1));
function verifyPool(size) {
  if (size < 1 || typeof Worker === 'undefined') return null;
  let workers;
  try {
    workers = Array.from({ length: size }, () => new Worker(new URL('./verify_worker.js', import.meta.url), { type: 'module' }));
  } catch {
    return null;
  }
  const pending = new Map(), load = workers.map(() => 0), dead = workers.map(() => false);
  let next = 0;
  workers.forEach((w, k) => {
    w.onmessage = ({ data }) => {
      const p = pending.get(data.id);
      pending.delete(data.id);
      load[k]--;
      p.resolve(data);
    };
    w.onerror = (e) => {
      e.preventDefault?.();
      dead[k] = true;
      for (const [id, p] of pending) if (p.k === k) { pending.delete(id); p.resolve(verifyInline(p.job)); }
    };
  });
  return {
    size,
    verify(job) {  // job: { xy, seed, maxError, minInliers } -> Promise<{ result, ms }>
      let k = -1;
      for (let w = 0; w < size; w++) if (!dead[w] && (k < 0 || load[w] < load[k])) k = w;
      if (k < 0) return Promise.resolve(verifyInline(job));
      const id = next++;
      load[k]++;
      return new Promise((resolve) => {
        pending.set(id, { job, k, resolve });
        workers[k].postMessage({ id, ...job });
      });
    },
    terminate() { for (const w of workers) w.terminate(); },
  };
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
  return { images: features.map(f => ({ name: f.name, time: f.time, ...(f.camera !== undefined ? { camera: f.camera } : {}),
                                        ...(f.flowH ? { flowH: f.flowH, flowChain: f.flowChain } : {}), keypoints: xyOnly(f.keypoints) })), pairs, stats };
}

// Match and verify given pairs [[i, j, type]]: one GPU batch per first frame, descriptors kept on
// the GPU (cached by frame index) from the first batch that uses a frame to the last; each pair
// verified on the pool (options.workers, default defaultVerifyWorkers(); 0: inline) while the GPU
// matches the next batches, with a seed of its own (pairSeed) drawn from one draw of rng. Returns
// { pairs (verified, sorted), stats }.
export async function matchPairs(features, matcher, todo, { rng, onProgress = () => {}, ...options } = {}) {
  const o = { ...FRONTEND_DEFAULTS, ...options };
  const pairs = [];
  const stats = { matched: 0, verified: 0, rawMatches: 0, inliers: 0, loopsTried: 0, loopsVerified: 0, gpuMs: 0, verifyMs: 0, waitMs: 0, workers: 0 };
  const byFirst = new Map();
  todo.forEach((t, k) => { if (!byFirst.has(t[0])) byFirst.set(t[0], []); byFirst.get(t[0]).push(k); });
  const batches = [...byFirst.values()], lastBatch = new Map();
  batches.forEach((ks, b) => { for (const k of ks) { lastBatch.set(todo[k][0], b); lastBatch.set(todo[k][1], b); } });
  matcher.clear();  // the cache keys are this call's frame indices
  const base = (rng() * 4294967296) >>> 0;
  const pool = todo.length >= POOL_MIN_PAIRS ? verifyPool(o.workers ?? defaultVerifyWorkers()) : null;
  stats.workers = pool?.size ?? 0;
  const checks = [];
  let done = 0;
  const accept = (i, j, type, raw, { result, ms }) => {
    stats.verifyMs += ms;
    if (result) {
      const m = new Uint32Array(2 * result.inliers.length);
      result.inliers.forEach((k, q) => { m[2 * q] = raw[2 * k]; m[2 * q + 1] = raw[2 * k + 1]; });
      pairs.push({ i, j, type, F: result.F, matches: m });
      stats.verified++; stats.inliers += result.inliers.length;
      if (type === 'loop') stats.loopsVerified++;
    }
    onProgress(++done, todo.length);
  };
  try {
    for (const [batch, ks] of batches.entries()) {
      const entries = ks.map(k => [matcher.upload(features[todo[k][0]].descriptors, todo[k][0]), matcher.upload(features[todo[k][1]].descriptors, todo[k][1])]);
      const t = performance.now();
      const results = await matcher.matchBatch(entries, o.match ?? {});  // match: MATCH_DEFAULTS overrides (Retry loosens them)
      stats.gpuMs += performance.now() - t;
      for (const x of new Set(ks.flatMap(k => todo[k].slice(0, 2)))) if (lastBatch.get(x) === batch) matcher.release(x);
      ks.forEach((k, b) => {
        const [i, j, type] = todo[k], raw = results[b];
        stats.matched++; stats.rawMatches += raw.length / 2;
        if (type === 'loop') stats.loopsTried++;
        const job = { xy: packPair(features[i].keypoints, features[j].keypoints, raw), seed: pairSeed(base, i, j), maxError: o.maxError, minInliers: o.minInliers };
        if (pool) checks.push(pool.verify(job).then(r => accept(i, j, type, raw, r)));
        else accept(i, j, type, raw, verifyInline(job));
      });
    }
    const t = performance.now();
    await Promise.all(checks);
    stats.waitMs = performance.now() - t;
  } finally {
    pool?.terminate();
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
