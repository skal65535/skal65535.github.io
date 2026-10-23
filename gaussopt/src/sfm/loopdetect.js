// loopdetect.js — Loop candidates (DESIGN-sfm §5.4), matched and verified afterwards like any pair
// (frontend.js), so a false candidate only costs matching time:
//   - by appearance, before mapping: a visual vocabulary learned from the video's own descriptors
//     (k-means, nearest-centroid assignment on the GPU matcher), a TF-IDF word histogram per frame,
//     and the most similar frames outside the sequential window; no pre-trained vocabulary to ship
//   - by geometry, after sequential mapping: frames far apart in time that should see the same
//     points from a similar direction and distance (R3: twice the appearance recall on jardin6)
"use strict";

import { dot, sub } from '../math.js';
import { project, inFieldOfView, poseCenter } from './camera.js';

// Per frame i, the first `perFrame` of `scored(i)` = [[j, score]] sorted best first, as
// deduplicated [i, j] pairs (i < j) in order
function topPairs(n, scored, perFrame) {
  const out = new Set();
  for (let i = 0; i < n; i++) for (const [j] of scored(i).slice(0, perFrame)) out.add(i < j ? `${i},${j}` : `${j},${i}`);
  return [...out].map(k => k.split(',').map(Number)).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

const DEFAULTS = { words: 1024, samplesPerImage: 150, iterations: 8, perFrame: 8, minGap: 20, minScoreRatio: 0.2 };

// unit L2 x 512 -> u8 (COLMAP's descriptor scale, so the matcher's angles apply to centroids)
function toDescriptor(v, out, off) {
  let n = 0;
  for (let k = 0; k < 128; k++) n += v[k] * v[k];
  const s = 512 / (Math.sqrt(n) || 1);
  for (let k = 0; k < 128; k++) out[off + k] = Math.min(255, Math.round(v[k] * s));
}

// k-means over sampled descriptors. Returns a Uint8Array(128 words).
export async function buildVocabulary(features, matcher, { rng, ...options } = {}) {
  const o = { ...DEFAULTS, ...options };
  const samples = [];
  for (const f of features) {
    const n = f.descriptors.length / 128;
    for (let k = 0; k < Math.min(n, o.samplesPerImage); k++) {
      const r = Math.floor(rng() * n);
      samples.push(f.descriptors.subarray(128 * r, 128 * r + 128));
    }
  }
  const S = new Uint8Array(128 * samples.length);
  samples.forEach((d, i) => S.set(d, 128 * i));
  const words = Math.min(o.words, samples.length);
  const vocab = new Uint8Array(128 * words);
  for (let w = 0; w < words; w++) vocab.set(samples[Math.floor(rng() * samples.length)], 128 * w);
  const sampleBuf = matcher.upload(S);
  for (let it = 0; it < o.iterations; it++) {
    const vocabBuf = matcher.upload(vocab);
    const [near] = await matcher.nearestBatch([[sampleBuf, vocabBuf]]);
    vocabBuf.buf.destroy();
    const sum = new Float64Array(128 * words), count = new Uint32Array(words);
    for (let i = 0; i < samples.length; i++) {
      const w = near[4 * i];
      if (w >= words) continue;
      count[w]++;
      for (let k = 0; k < 128; k++) sum[128 * w + k] += S[128 * i + k];
    }
    for (let w = 0; w < words; w++) {
      if (count[w]) toDescriptor(sum.subarray(128 * w, 128 * w + 128), vocab, 128 * w);
      else vocab.set(samples[Math.floor(rng() * samples.length)], 128 * w);  // re-seed empty words
    }
  }
  sampleBuf.buf.destroy();
  return vocab;
}

// Word of every descriptor of every frame -> L2-normalized TF-IDF vectors (Float32Array(words) each)
export async function bagOfWords(features, vocab, matcher) {
  const words = vocab.length / 128;
  const vocabBuf = matcher.upload(vocab);
  const hist = [];
  for (let start = 0; start < features.length; start += 16) {
    const batch = features.slice(start, start + 16).map(f => [matcher.upload(f.descriptors), vocabBuf]);
    const near = await matcher.nearestBatch(batch);
    batch.forEach(([q], b) => {
      const h = new Float32Array(words);
      for (let i = 0; i < q.n; i++) if (near[b][4 * i] < words) h[near[b][4 * i]]++;
      hist.push(h);
      q.buf.destroy();
    });
  }
  vocabBuf.buf.destroy();
  const df = new Float32Array(words);
  for (const h of hist) for (let w = 0; w < words; w++) df[w] += h[w] > 0;
  return hist.map(h => {
    let n = 0;
    for (let w = 0; w < words; w++) { h[w] *= Math.log(hist.length / Math.max(1, df[w])); n += h[w] * h[w]; }
    const s = 1 / (Math.sqrt(n) || 1);
    for (let w = 0; w < words; w++) h[w] *= s;
    return h;
  });
}

// Loop candidates [[i, j]] (i < j, j - i >= minGap): per frame the `perFrame` most similar frames
// whose score is at least minScoreRatio x its similarity to the next frame (which always overlaps).
export function loopCandidates(vectors, options = {}) {
  const o = { ...DEFAULTS, ...options };
  const n = vectors.length, words = vectors[0]?.length ?? 0;
  const sim = (a, b) => { let s = 0; for (let w = 0; w < words; w++) s += vectors[a][w] * vectors[b][w]; return s; };
  return topPairs(n, (i) => {
    const floor = o.minScoreRatio * (sim(i, Math.min(n - 1, i + 1)) || 1);
    const scored = [];
    for (let j = 0; j < n; j++) if (Math.abs(i - j) >= o.minGap) scored.push([j, sim(i, j)]);
    return scored.filter(([, s]) => s >= floor).sort((a, b) => b[1] - a[1]);
  }, o.perFrame);
}

// Loop candidates by geometry, after sequential mapping: frames of the same segment, far apart in
// time, that should match — a sample of frame i's points lands inside frame j, seen from a
// similar direction (rays within maxRayDeg) and a similar distance (depth ratio < depthRatio),
// which is what SIFT matching needs. Drift makes this approximate; matching and verification
// absorb that. rng (required, for the point samples). Returns [[i, j]] (i < j), the `perFrame` best per frame.
export function geometricCandidates(mapper, { minGap = 20, perFrame = 8, minOverlap = 0.1, samples = 200, depthRatio = 1.5,
                                               maxRayDeg = 25, rng }) {
  if (!rng) throw new Error('geometricCandidates: rng is required (deterministic candidates)');
  const { camera } = mapper;
  const frames = [];
  for (let k = 0; k < mapper.n; k++) if (mapper.poses[k]) frames.push(k);
  const center = new Map(frames.map(k => [k, poseCenter(mapper.poses[k])]));
  const sampleOf = new Map(frames.map(k => {
    const pts = [];
    for (const pid of mapper.kpPoint[k]) if (pid >= 0 && mapper.points[pid]) pts.push(mapper.points[pid].X);
    for (let s = pts.length - 1; s > 0; s--) { const r = Math.floor(rng() * (s + 1)); [pts[s], pts[r]] = [pts[r], pts[s]]; }
    return [k, pts.slice(0, samples)];
  }));
  const cosMax = Math.cos(maxRayDeg * Math.PI / 180);
  const overlap = (i, j) => {
    const pts = sampleOf.get(i), Ci = center.get(i), Cj = center.get(j), P = mapper.poses[j];
    let good = 0;
    for (const X of pts) {
      if (!inFieldOfView(camera, P, X, 0)) continue;  // also guards against distortion folding back
      const [u, v] = project(camera, P, X);
      if (u < 0 || v < 0 || u >= camera.width || v >= camera.height) continue;
      const a = sub(X, Ci), b = sub(X, Cj);
      const la = Math.hypot(...a), lb = Math.hypot(...b);
      if (la > depthRatio * lb || lb > depthRatio * la) continue;
      if (dot(a, b) / (la * lb) < cosMax) continue;  // ray angle, by its cosine (la, lb at hand)
      good++;
    }
    return good / pts.length;
  };
  return topPairs(mapper.n, (i) => {
    if (!mapper.poses[i] || sampleOf.get(i).length < 20) return [];
    return frames.filter(j => Math.abs(i - j) >= minGap && mapper.segment[i] === mapper.segment[j])
      .map(j => [j, overlap(i, j)]).filter(([, o]) => o >= minOverlap).sort((a, b) => b[1] - a[1]);
  }, perFrame);
}
