// align.js — Similarity alignment of point sets (Umeyama 1991): dst ~ s R src + t.
// Used by loop closure (loop.js: Sim(3) between segments), P3P (geometry.js: rigid pose) and the SfM panel's
// camera update (robustUmeyama).
"use strict";

import { svd, transpose, matMul, det3, mulVec3 } from './linalg.js';
import { apply } from './sim3.js';

// The centroid of points [[x, y, z]]
export const mean = (P) => [0, 1, 2].map(k => P.reduce((s, p) => s + p[k], 0) / P.length);

// src, dst: arrays of [x, y, z] (same length >= 3). withScale = false gives a rigid fit (s = 1).
export function umeyama(src, dst, { withScale = true } = {}) {
  const n = src.length;
  if (n < 3 || dst.length !== n) throw new Error(`umeyama needs >= 3 matched points, got ${n}/${dst.length}`);
  const ms = mean(src), md = mean(dst);
  const cov = new Float64Array(9);  // (1/n) sum (d - md)(s - ms)^T
  let varS = 0;
  for (let i = 0; i < n; i++) {
    const a = [src[i][0] - ms[0], src[i][1] - ms[1], src[i][2] - ms[2]];
    const b = [dst[i][0] - md[0], dst[i][1] - md[1], dst[i][2] - md[2]];
    varS += a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) cov[r * 3 + c] += b[r] * a[c];
  }
  for (let k = 0; k < 9; k++) cov[k] /= n;
  varS /= n;
  const { U, S, V } = svd(cov, 3, 3);
  const Vt = transpose(V, 3, 3);
  const d = det3(Array.from(U)) * det3(Array.from(V)) < 0 ? -1 : 1;
  const D = [1, 0, 0, 0, 1, 0, 0, 0, d];
  const R = Array.from(matMul(matMul(U, D, 3, 3, 3), Vt, 3, 3, 3));
  const s = withScale ? (S[0] + S[1] + d * S[2]) / (varS || 1) : 1;
  const rms = mulVec3(R, ms);
  const t = [md[0] - s * rms[0], md[1] - s * rms[1], md[2] - s * rms[2]];
  return { s, R, t };
}

// umeyama refitted once without the pairs whose residual exceeds 3x the median (e.g. the cameras a repair moved)
export function robustUmeyama(src, dst) {
  let idx = src.map((_, k) => k), G = null;
  for (let pass = 0; pass < 2; pass++) {
    G = umeyama(idx.map(k => src[k]), idx.map(k => dst[k]));
    const res = src.map((p, k) => Math.hypot(...apply(G, p).map((x, d) => x - dst[k][d])));
    const med = [...res].sort((a, b) => a - b)[res.length >> 1];
    const keep = src.map((_, k) => k).filter(k => res[k] <= 3 * med + 1e-12);
    if (keep.length < 3 || keep.length === idx.length) break;
    idx = keep;
  }
  return G;
}
