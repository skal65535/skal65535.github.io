// frames.js — S0 keyframe selection (DESIGN-sfm §5.1), a port of video2splat.py `select`:
// sharpness = variance of the Laplacian at <= 1000 px, motion = median pyramidal Lucas-Kanade
// displacement of Shi-Tomasi corners at <= 640 px (fraction of the diagonal), then one sharpest
// frame per SPACING of cumulative motion (or per 1/keepFps s), blurry picks dropped. Works on
// 8-bit gray images from any source (browser canvas, ffmpeg in the tests).
"use strict";

import { median } from './util.js';

export const FRAME_DEFAULTS = { spacing: 0.05, keepFps: 2, blurRel: 0.35, sharpDim: 1000, flowDim: 640 };

// Size of an image scaled to fit `maxDim` (never enlarged)
export function fitSize(w, h, maxDim) {
  const s = maxDim / Math.max(w, h);
  return s < 1 ? [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))] : [w, h];
}

// Area-average resize (OpenCV INTER_AREA when shrinking): Float32Array gray
export function resizeArea(src, w, h, W, H) {
  const axis = (n, N) => {  // per output index: [first input index, weights]
    const scale = n / N, out = [];
    for (let i = 0; i < N; i++) {
      const a = i * scale, b = a + scale, first = Math.floor(a), wts = [];
      for (let k = first; k < Math.min(n, Math.ceil(b)); k++) wts.push((Math.min(b, k + 1) - Math.max(a, k)) / scale);
      out.push([first, wts]);
    }
    return out;
  };
  const ax = axis(w, W), ay = axis(h, H);
  const tmp = new Float32Array(W * h), out = new Float32Array(W * H);
  for (let y = 0; y < h; y++) {
    for (let X = 0; X < W; X++) {
      const [x0, wts] = ax[X];
      let s = 0;
      for (let k = 0; k < wts.length; k++) s += wts[k] * src[y * w + x0 + k];
      tmp[y * W + X] = s;
    }
  }
  for (let Y = 0; Y < H; Y++) {
    const [y0, wts] = ay[Y];
    for (let X = 0; X < W; X++) {
      let s = 0;
      for (let k = 0; k < wts.length; k++) s += wts[k] * tmp[(y0 + k) * W + X];
      out[Y * W + X] = s;
    }
  }
  return out;
}

// Variance of the 4-neighbour Laplacian (cv2.Laplacian, ksize 1, reflect-101 borders)
export function sharpness(g, w, h) {
  let s = 0, s2 = 0;
  for (let y = 0; y < h; y++) {
    const up = (y > 0 ? y - 1 : 1) * w, dn = (y < h - 1 ? y + 1 : h - 2) * w, row = y * w;
    for (let x = 0; x < w; x++) {
      const l = x > 0 ? x - 1 : 1, r = x < w - 1 ? x + 1 : w - 2;
      const v = g[up + x] + g[dn + x] + g[row + l] + g[row + r] - 4 * g[row + x];
      s += v; s2 += v * v;
    }
  }
  const n = w * h;
  return s2 / n - (s / n) ** 2;
}

// ── Pyramidal Lucas-Kanade ─────────────────────────────────────────────────────

// Gaussian pyramid level: 5-tap [1 4 6 4 1] / 16 blur, then every other pixel (cv2.pyrDown)
function pyrDown({ data, w, h }) {
  const W = (w + 1) >> 1, H = (h + 1) >> 1, tmp = new Float32Array(W * h), out = new Float32Array(W * H);
  const cl = (v, n) => v < 0 ? -v : v >= n ? 2 * n - 2 - v : v;
  for (let y = 0; y < h; y++) {
    for (let X = 0; X < W; X++) {
      const x = 2 * X, r = y * w;
      tmp[y * W + X] = (data[r + cl(x - 2, w)] + 4 * data[r + cl(x - 1, w)] + 6 * data[r + x] + 4 * data[r + cl(x + 1, w)] + data[r + cl(x + 2, w)]) / 16;
    }
  }
  for (let Y = 0; Y < H; Y++) {
    const y = 2 * Y;
    for (let X = 0; X < W; X++) {
      out[Y * W + X] = (tmp[cl(y - 2, h) * W + X] + 4 * tmp[cl(y - 1, h) * W + X] + 6 * tmp[y * W + X] + 4 * tmp[cl(y + 1, h) * W + X] + tmp[cl(y + 2, h) * W + X]) / 16;
    }
  }
  return { data: out, w: W, h: H };
}

const LK_WIN = 21, LK_LEVELS = 4, PAD = (LK_WIN >> 1) + 2;

// Level padded by PAD replicated pixels on each side, so LK windows can overhang the image
// (cv2.buildOpticalFlowPyramid's border), with its Scharr derivatives (/ 32):
// { w, h, S (padded stride), data, gx, gy }; pixel (x, y) is at (y + PAD) * S + x + PAD
function paddedLevel({ data, w, h }) {
  const S = w + 2 * PAD, H = h + 2 * PAD, d = new Float32Array(S * H);
  for (let y = 0; y < H; y++) {
    const sy = Math.min(h - 1, Math.max(0, y - PAD)) * w;
    for (let x = 0; x < S; x++) d[y * S + x] = data[sy + Math.min(w - 1, Math.max(0, x - PAD))];
  }
  const gx = new Float32Array(S * H), gy = new Float32Array(S * H);
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1, o = y * S + 1; x < S - 1; x++, o++) {
      const nw = d[o - S - 1], n = d[o - S], ne = d[o - S + 1], west = d[o - 1], e = d[o + 1], sw = d[o + S - 1], s = d[o + S], se = d[o + S + 1];
      gx[o] = (3 * (ne - nw + se - sw) + 10 * (e - west)) / 32;
      gy[o] = (3 * (sw - nw + se - ne) + 10 * (s - n)) / 32;
    }
  }
  return { w, h, S, data: d, gx, gy };
}

// Padded Gaussian pyramid of a gray image, LK_LEVELS + 1 levels at most
export function flowPyramid(g, w, h) {
  const plain = [{ data: g instanceof Float32Array ? g : Float32Array.from(g), w, h }];
  while (plain.length <= LK_LEVELS && plain.at(-1).w > 8 && plain.at(-1).h > 8) plain.push(pyrDown(plain.at(-1)));
  return plain.map(paddedLevel);
}

// Shi-Tomasi corners (cv2.goodFeaturesToTrack) of a pyramid's base level: min eigenvalue of the
// 3x3-summed structure tensor, >= quality x max, 3x3 local maxima, strongest first, `minDistance` apart
export function goodFeatures({ w, h, S, gx, gy }, { maxCorners = 400, quality = 0.01, minDistance = 8 } = {}) {
  // 3x3 box sums of the tensor products, separably (row sums, then column sums)
  const xx = new Float32Array(w * h), xy = new Float32Array(w * h), yy = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 1; x < w - 1; x++) {
      let a = 0, b = 0, c = 0;
      for (let o = (y + PAD) * S + x + PAD - 1, k = 0; k < 3; k++, o++) { a += gx[o] * gx[o]; b += gx[o] * gy[o]; c += gy[o] * gy[o]; }
      xx[y * w + x] = a; xy[y * w + x] = b; yy[y * w + x] = c;
    }
  }
  const eig = new Float32Array(w * h);
  let best = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const o = y * w + x;
      const a = xx[o - w] + xx[o] + xx[o + w], b = xy[o - w] + xy[o] + xy[o + w], c = yy[o - w] + yy[o] + yy[o + w];
      const e = (a + c) / 2 - Math.sqrt(((a - c) / 2) ** 2 + b * b);
      eig[o] = e;
      if (e > best) best = e;
    }
  }
  const thresh = quality * best, cand = [];
  for (let y = 2; y < h - 2; y++) {
    for (let x = 2; x < w - 2; x++) {
      const o = y * w + x, e = eig[o];
      if (e <= thresh) continue;
      if (eig[o - w - 1] > e || eig[o - w] > e || eig[o - w + 1] > e || eig[o - 1] > e || eig[o + 1] > e ||
          eig[o + w - 1] > e || eig[o + w] > e || eig[o + w + 1] > e) continue;
      cand.push([e, x, y]);
    }
  }
  cand.sort((p, q) => q[0] - p[0]);
  const cell = minDistance, gw = Math.ceil(w / cell), grid = new Map(), out = [];
  for (const [, x, y] of cand) {
    const cx = Math.floor(x / cell), cy = Math.floor(y / cell);
    let ok = true;
    for (let j = cy - 1; j <= cy + 1 && ok; j++) {
      for (let i = cx - 1; i <= cx + 1 && ok; i++) {
        for (const [px, py] of grid.get(j * gw + i) ?? []) if ((px - x) ** 2 + (py - y) ** 2 < minDistance ** 2) { ok = false; break; }
      }
    }
    if (!ok) continue;
    const key = cy * gw + cx;
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push([x, y]);
    out.push([x, y]);
    if (out.length >= maxCorners) break;
  }
  return out;
}

// Track points from flow pyramid A to flow pyramid B: [[x, y] | null] (null = lost)
export function trackLK(pyrA, pyrB, points, { iterations = 30, epsilon = 0.01, minEig = 1e-4 } = {}) {
  const r = LK_WIN >> 1, n = LK_WIN * LK_WIN;
  const wI = new Float32Array(n), wX = new Float32Array(n), wY = new Float32Array(n);
  return points.map(([px, py]) => {
    let gx = 0, gy = 0;
    for (let L = pyrA.length - 1; L >= 0; L--) {
      const A = pyrA[L], B = pyrB[L], S = A.S, s = 1 / (1 << L);
      const x = px * s, y = py * s;
      // window of A, its gradients and the structure tensor, at the subpixel position
      const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
      const w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy;
      const I = A.data, X = A.gx, Y = A.gy;
      let G11 = 0, G12 = 0, G22 = 0;
      for (let j = -r, k = 0; j <= r; j++) {
        for (let i = -r, o = (y0 + j + PAD) * S + x0 - r + PAD; i <= r; i++, k++, o++) {
          const vi = w00 * I[o] + w01 * I[o + 1] + w10 * I[o + S] + w11 * I[o + S + 1];
          const vx = w00 * X[o] + w01 * X[o + 1] + w10 * X[o + S] + w11 * X[o + S + 1];
          const vy = w00 * Y[o] + w01 * Y[o + 1] + w10 * Y[o + S] + w11 * Y[o + S + 1];
          wI[k] = vi; wX[k] = vx; wY[k] = vy;
          G11 += vx * vx; G12 += vx * vy; G22 += vy * vy;
        }
      }
      const det = G11 * G22 - G12 * G12;
      // cv2's units: raw Scharr (32 x gradient) products scaled by 2^-20, per window pixel
      const eigMin = ((G11 + G22) / 2 - Math.sqrt(((G11 - G22) / 2) ** 2 + G12 * G12)) / n / 1024;
      if (eigMin < minEig || det <= 0) {  // too flat at this scale: keep the guess (cv2: lost at level 0 only)
        if (L === 0) return null;
        gx *= 2; gy *= 2;
        continue;
      }
      const J = B.data;
      let ux = gx, uy = gy;
      for (let it = 0; it < iterations; it++) {
        const qx = x + ux, qy = y + uy;
        if (qx < r - PAD || qy < r - PAD || qx >= A.w + PAD - r - 1 || qy >= A.h + PAD - r - 1) return null;  // window inside the padding
        const q0 = Math.floor(qx), q1 = Math.floor(qy), ex = qx - q0, ey = qy - q1;
        const v00 = (1 - ex) * (1 - ey), v01 = ex * (1 - ey), v10 = (1 - ex) * ey, v11 = ex * ey;
        let b1 = 0, b2 = 0;
        for (let j = -r, k = 0; j <= r; j++) {
          for (let i = -r, o = (q1 + j + PAD) * S + q0 - r + PAD; i <= r; i++, k++, o++) {
            const d = wI[k] - (v00 * J[o] + v01 * J[o + 1] + v10 * J[o + S] + v11 * J[o + S + 1]);
            b1 += d * wX[k]; b2 += d * wY[k];
          }
        }
        const dx = (G22 * b1 - G12 * b2) / det, dy = (G11 * b2 - G12 * b1) / det;
        ux += dx; uy += dy;
        if (dx * dx + dy * dy < epsilon * epsilon) break;
      }
      gx = L > 0 ? 2 * ux : ux; gy = L > 0 ? 2 * uy : uy;
    }
    const qx = px + gx, qy = py + gy;
    return qx >= 0 && qy >= 0 && qx <= pyrA[0].w - 1 && qy <= pyrA[0].h - 1 ? [qx, qy] : null;
  });
}

// One frame's analysis: { sharpness, flow (pyramid at flowDim, for motion to the next frame) }
export function analyzeFrame(gray, w, h, o = FRAME_DEFAULTS) {
  const [sw, sh] = fitSize(w, h, o.sharpDim);
  const sharp = sw === w && sh === h ? gray : resizeArea(gray, w, h, sw, sh);
  const [fw, fh] = fitSize(sw, sh, o.flowDim);
  const small = fw === sw && fh === sh ? sharp : resizeArea(sharp, sw, sh, fw, fh);
  return { sharpness: sharpness(sharp, sw, sh), flow: flowPyramid(small, fw, fh) };
}

// Median displacement of corners tracked from a to b, as a fraction of the diagonal;
// Infinity when tracking is lost (fewer than 20 corners or tracks)
export function motion(pyrA, pyrB) {
  const pts = goodFeatures(pyrA[0]);
  if (pts.length < 20) return Infinity;
  const tracked = trackLK(pyrA, pyrB, pts);
  const d = [];
  tracked.forEach((q, k) => { if (q) d.push(Math.hypot(q[0] - pts[k][0], q[1] - pts[k][1])); });
  if (d.length < 20) return Infinity;
  return median(d) / Math.hypot(pyrA[0].w, pyrA[0].h);
}

// Keyframe choice from per-candidate sharpness and motion (motion[i]: from candidate i - 1 to i,
// motion[0] unused). times: candidate timestamps in seconds (keepFps mode).
// Returns { kept: [index], status: ['kept' | 'blurry' | 'window'], path (spacing units), threshold }
export function selectFrames({ sharpness: scores, motion: steps, times }, options = {}) {
  const o = { ...FRAME_DEFAULTS, ...options };
  const n = scores.length, bins = new Map();
  let path = null;
  const add = (k, i) => { if (!bins.has(k)) bins.set(k, []); bins.get(k).push(i); };
  if (o.spacing > 0) {
    // equal spacing along the camera path; the sharpest candidate of each segment's central
    // half is picked, so neighbouring picks stay 0.5-1.5 x spacing apart
    path = [0];
    for (let i = 1; i < n; i++) path.push(path[i - 1] + Math.min(steps[i], o.spacing) / o.spacing);
    path.forEach((t, i) => add(Math.floor(t + 0.5), i));
    for (const [k, b] of bins) {
      const mid = b.filter(i => Math.abs(path[i] - k) <= 0.25);
      if (mid.length) bins.set(k, mid);
    }
  } else {
    times.forEach((t, i) => add(Math.floor((t - times[0]) * o.keepFps + 1e-6), i));
  }
  const picks = [...bins.values()].map(b => b.reduce((best, i) => scores[i] > scores[best] ? i : best, b[0]));
  const threshold = o.blurRel * median(picks.map(i => scores[i]));
  const status = new Array(n).fill('window');
  const kept = [];
  for (const i of picks) {
    status[i] = scores[i] >= threshold ? 'kept' : 'blurry';
    if (status[i] === 'kept') kept.push(i);
  }
  return { kept, status, path, threshold };
}
