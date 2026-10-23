// util.js — Small helpers shared by the SfM modules: robust statistics, angles, matched keypoint
// coordinates, string hash.
"use strict";

import { dot } from '../math.js';
import { mul3, transpose3 } from './linalg.js';

export const deg = (radians) => radians * 180 / Math.PI;

// Median (mean of the two middle values for even lengths); NaN when empty
export function median(values) {
  if (!values.length) return NaN;
  const s = [...values].sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// q-quantile (0..1) by nearest rank; NaN when empty
export function quantile(values, q) {
  if (!values.length) return NaN;
  const s = [...values].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
}

// [[x1...], [x2...]]: pixel coordinates of matched keypoints (matches = [a, b, a, b, ...], or
// { a, b } index arrays as in SequentialMapper.adj); `stride` floats per keypoint (2 for [x, y],
// 4 for SIFT's [x, y, scale, angle])
export function matchedPixels(kpA, kpB, matches, stride = 2) {
  const split = !!matches.a, n = split ? matches.a.length : matches.length / 2, x1 = new Array(n), x2 = new Array(n);
  for (let k = 0; k < n; k++) {
    const a = split ? matches.a[k] : matches[2 * k], b = split ? matches.b[k] : matches[2 * k + 1];
    x1[k] = [kpA[stride * a], kpA[stride * a + 1]];
    x2[k] = [kpB[stride * b], kpB[stride * b + 1]];
  }
  return [x1, x2];
}

// FNV-1a hash of a string (32 bits)
export function hashString(str) {
  let h = 2166136261;
  for (let k = 0; k < str.length; k++) h = Math.imul(h ^ str.charCodeAt(k), 16777619);
  return h >>> 0;
}

// Angle (radians) between two 3-vectors
export function angleBetween(a, b) {
  const c = dot(a, b) / (Math.hypot(a[0], a[1], a[2]) * Math.hypot(b[0], b[1], b[2]) || 1);
  return Math.acos(Math.min(1, Math.max(-1, c)));
}

// Rotation from camera a to camera b (world -> camera R's): Rb Ra^T
export const relativeRotation = (Ra, Rb) => mul3(Rb, transpose3(Ra));
