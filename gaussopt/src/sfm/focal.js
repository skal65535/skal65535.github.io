// focal.js — Focal length self-calibration from fundamental matrices (DESIGN-sfm §5.3).
// For the right K, E = K^T F K is an essential matrix: two equal singular values and a zero one.
// Each pair scores a candidate focal by (s1 - s2) / (s1 + s2) (the criterion of COLMAP's
// view_graph_calibration); the focal minimizing the robust (median) score over many pairs wins.
// Assumes square pixels and the principal point at the image center.
"use strict";

import { svd } from './linalg.js';
import { essentialFromFundamental } from './geometry.js';

export function focalCost(F, f, cx, cy) {
  const E = essentialFromFundamental(F, [f, f, cx, cy]);
  const { S } = svd(Float64Array.from(E), 3, 3);
  return (S[0] - S[1]) / (S[0] + S[1] || 1);
}

function aggregate(Fs, f, cx, cy) {
  const c = Fs.map(F => focalCost(F, f, cx, cy)).sort((a, b) => a - b);
  return c[c.length >> 1];
}

// Fs: fundamental matrices (pixels). Returns { focal, cost, sharpness, curve: [[f, cost]] }.
// sharpness = min(cost(f / 1.25), cost(1.25 f)) / cost(f): ~1 when the focal is unobservable,
// e.g. for translation without rotation (a nadir mapping pass), a critical motion for F.
export function estimateFocal(Fs, width, height, { minRatio = 0.25, maxRatio = 4, steps = 160 } = {}) {
  if (Fs.length === 0) throw new Error('estimateFocal: no fundamental matrices');
  const cx = width / 2, cy = height / 2, base = Math.max(width, height);
  const curve = [];
  for (let k = 0; k <= steps; k++) {
    const f = base * minRatio * (maxRatio / minRatio) ** (k / steps);  // log-spaced
    curve.push([f, aggregate(Fs, f, cx, cy)]);
  }
  let best = 0;
  curve.forEach(([, c], k) => { if (c < curve[best][1]) best = k; });
  // golden-section refinement between the grid neighbours
  let lo = curve[Math.max(0, best - 1)][0], hi = curve[Math.min(steps, best + 1)][0];
  const g = (Math.sqrt(5) - 1) / 2;
  let a = hi - g * (hi - lo), b = lo + g * (hi - lo);
  let fa = aggregate(Fs, a, cx, cy), fb = aggregate(Fs, b, cx, cy);
  for (let it = 0; it < 40; it++) {
    if (fa < fb) { hi = b; b = a; fb = fa; a = hi - g * (hi - lo); fa = aggregate(Fs, a, cx, cy); }
    else { lo = a; a = b; fa = fb; b = lo + g * (hi - lo); fb = aggregate(Fs, b, cx, cy); }
  }
  const focal = (lo + hi) / 2, cost = aggregate(Fs, focal, cx, cy);
  const sharpness = Math.min(aggregate(Fs, focal / 1.25, cx, cy), aggregate(Fs, focal * 1.25, cx, cy)) / Math.max(cost, 1e-12);
  return { focal, cost, sharpness, curve };
}
