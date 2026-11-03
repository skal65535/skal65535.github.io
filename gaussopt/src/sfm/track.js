// track.js — A video's keyframe keypoints tracked through every sampled frame (DESIGN-sfm §5.2.1): longer tracks
// than the matches of SIFT descriptors give, which fall off within a few keyframes as the view turns. Each point:
// a diamond search coarse to fine (zero-mean SAD, 3 Gaussian levels), a Lucas-Kanade step at full size for its
// sub-pixel position, and a forward-backward check. At each keyframe a live track takes the SIFT keypoint within
// `snap` px, else its own position becomes a keypoint of that keyframe (no descriptor: the mapper needs positions
// only). Every two keyframes of a track up to maxGap apart get a match; mergeTrackPairs adds them to the pairs.
"use strict";

import { pyrDown } from './frames.js';
import { verifyPacked } from './frontend.js';

export const TRACK_DEFAULTS = {
  maxGap: 20,       // keyframes: the farthest two a track matches (the loop stage takes pairs farther apart, loop.js longRange)
  snap: 1.5,        // px: a SIFT keypoint this near the tracked position stands for it
  minStd: 4,        // gray levels: a flatter patch is not tracked
  lkRadius: 7,      // the Lucas-Kanade window, (2 r + 1)^2
  lkMove: 3,        // px: the most the Lucas-Kanade step may move the search's position
  fb: 1,            // px: tracked back to the frame before, the point within this of where it was
  maxJump: 5,       // px: a step off its predicted motion by this more than the frame's median is lost (occlusions, lookalikes,
                    // things that move on their own)
  maxTracks: 12000, // live tracks at most (new ones start from the SIFT keypoints no track took)
};

const LEVELS = 3, RAD = [4, 3, 3];  // the search's patch radius per level
const LDSP = [[0, -2], [1, -1], [2, 0], [1, 1], [0, 2], [-1, 1], [-2, 0], [-1, -1]], SDSP = [[0, -1], [1, 0], [0, 1], [-1, 0]];

// Bilinear samples of a level { data, w } on the (2 r + 1)^2 grid around (x, y), inside it, into out (row by row);
// one set of weights: every sample of the grid has (x, y)'s fractions. Returns their sum.
function patch(d, w, x, y, r, out) {
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0, gx = 1 - fx, gy = 1 - fy;
  let s = 0, k = 0;
  for (let dy = -r; dy <= r; dy++) for (let dx = -r, o = (y0 + dy) * w + x0 - r; dx <= r; dx++, o++) {
    const v = gy * (gx * d[o] + fx * d[o + 1]) + fy * (gx * d[o + w] + fx * d[o + w + 1]);
    out[k++] = v; s += v;
  }
  return s;
}

export class PointTracker {
  constructor(width, height, options = {}) {
    this.o = { ...TRACK_DEFAULTS, ...options };
    this.w = width; this.h = height;
    this.prev = null;  // the last frame's pyramid
    this.live = [];    // { x, y, vx, vy, visits: [[k, keypoint]] }
    this.ended = [];
    this.added = new Map();  // keyframe -> [x, y, ...]: its keypoints after the SIFT ones
    this.stats = { steps: 0, lost: { border: 0, flat: 0, lk: 0, fb: 0, jump: 0 } };
    const n = Math.max(...RAD.map(r => (2 * r + 1) ** 2), (2 * this.o.lkRadius + 1) ** 2);
    this.tmpl = new Float32Array(n); this.gx = new Float32Array(n); this.gy = new Float32Array(n); this.cur = new Float32Array(n);
    this.ext = new Float32Array((2 * this.o.lkRadius + 3) ** 2);  // the Lucas-Kanade template with a 1 px rim, for its gradients
  }

  // The next frame, in time order (gray levels, width x height)
  frame(gray) {
    const levels = [{ data: gray instanceof Float32Array ? gray : Float32Array.from(gray), w: this.w, h: this.h }];
    while (levels.length < LEVELS) levels.push(pyrDown(levels.at(-1)));
    if (this.prev) {
      // each step off its predicted motion: the frame's median of it is the camera's shake, the rest the point's own
      const found = this.live.map(t => this._step(this.prev, levels, t.x, t.y, t.x + t.vx, t.y + t.vy));
      this.stats.steps += found.length;
      const off = (r, t, c) => r[c] - t[c === 0 ? 'x' : 'y'] - t[c === 0 ? 'vx' : 'vy'];
      const med = (c) => { const a = []; found.forEach((r, i) => { if (r) a.push(off(r, this.live[i], c)); }); return a.length ? a.sort((p, q) => p - q)[a.length >> 1] : 0; };
      const mx = med(0), my = med(1), live = [];
      this.live.forEach((t, i) => {
        const r = found[i];
        if (r && Math.hypot(off(r, t, 0) - mx, off(r, t, 1) - my) > this.o.maxJump) { this.stats.lost.jump++; this.ended.push(t); return; }
        if (!r) { this.ended.push(t); return; }
        t.vx = r[0] - t.x; t.vy = r[1] - t.y; t.x = r[0]; t.y = r[1];
        live.push(t);
      });
      this.live = live;
    }
    this.prev = levels;
  }

  // The last frame is keyframe k, with SIFT keypoints kp4 (4 floats each): every live track visits it, at the
  // keypoint within `snap` px (each taken by its nearest track) or at a keypoint added there; the keypoints no
  // track took start new tracks. keep(x, y): false where no point may be (an on-screen display, overlay.js).
  keyframe(k, kp4, keep = null) {
    const n = kp4.length / 4, cell = 4, grid = new Map(), key = (gx, gy) => gx * 65536 + gy;
    for (let i = 0; i < n; i++) {
      const g = key(Math.floor(kp4[4 * i] / cell), Math.floor(kp4[4 * i + 1] / cell));
      if (!grid.has(g)) grid.set(g, []);
      grid.get(g).push(i);
    }
    const live = keep ? this.live.filter(t => keep(t.x, t.y) || (this.ended.push(t), false)) : this.live;
    const taken = new Map();  // keypoint -> [track, distance]
    for (const t of live) {
      let best = -1, bd = this.o.snap;
      const gx = Math.floor(t.x / cell), gy = Math.floor(t.y / cell);
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        for (const i of grid.get(key(gx + dx, gy + dy)) ?? []) {
          const d = Math.hypot(kp4[4 * i] - t.x, kp4[4 * i + 1] - t.y);
          if (d < bd) { bd = d; best = i; }
        }
      }
      if (best >= 0 && !(taken.get(best)?.[1] <= bd)) taken.set(best, [t, bd]);
    }
    const snapped = new Set();
    for (const [i, [t]] of taken) { t.x = kp4[4 * i]; t.y = kp4[4 * i + 1]; t.visits.push([k, i]); snapped.add(t); }
    const added = [];
    for (const t of live) if (!snapped.has(t)) { t.visits.push([k, n + added.length / 2]); added.push(t.x, t.y); }
    this.added.set(k, added);
    // the new tracks move as the median of the live ones did last
    const med = (a) => (a.length ? a.sort((p, q) => p - q)[a.length >> 1] : 0);
    const vx = med(live.map(t => t.vx)), vy = med(live.map(t => t.vy));
    this.live = live;
    for (let i = 0; i < n && this.live.length < this.o.maxTracks; i++) {
      if (!taken.has(i)) this.live.push({ x: kp4[4 * i], y: kp4[4 * i + 1], vx, vy, visits: [[k, i]] });
    }
  }

  // { added: Map keyframe -> Float32Array [x, y, ...] (its keypoints after the SIFT ones), pairs: [{ i, j, matches
  // (Uint32Array, keypoint pairs) }] of keyframes up to maxGap apart, tracks: per visit count, the tracks }
  result() {
    const all = [...this.ended, ...this.live], byPair = new Map(), lengths = [];
    for (const t of all) {
      lengths[t.visits.length] = (lengths[t.visits.length] ?? 0) + 1;
      for (let a = 0; a < t.visits.length; a++) {
        for (let b = a + 1; b < t.visits.length && t.visits[b][0] - t.visits[a][0] <= this.o.maxGap; b++) {
          const [i, ki] = t.visits[a], [j, kj] = t.visits[b], key = `${i},${j}`;
          if (!byPair.has(key)) byPair.set(key, { i, j, m: [] });
          byPair.get(key).m.push(ki, kj);
        }
      }
    }
    const pairs = [...byPair.values()].map(({ i, j, m }) => ({ i, j, matches: Uint32Array.from(m) })).sort((p, q) => p.i - q.i || p.j - q.j);
    const added = new Map([...this.added].map(([k, a]) => [k, Float32Array.from(a)]));
    return { added, pairs, tracks: Array.from(lengths, x => x ?? 0) };
  }

  // The point (x, y) of pyramid A found in pyramid B from the prediction (px, py): [x, y] or null
  _step(A, B, x, y, px, py) {
    let cx = 0, cy = 0;
    for (let l = LEVELS - 1; l >= 0; l--) {
      const s = 2 ** l, std = this._template(A[l], x / s, y / s, RAD[l]);
      if (std < 0) { this.stats.lost.border++; return null; }
      if (l === 0 && std < this.o.minStd) { this.stats.lost.flat++; return null; }
      const sx = l === LEVELS - 1 ? Math.round(px / s) : 2 * cx, sy = l === LEVELS - 1 ? Math.round(py / s) : 2 * cy;
      [cx, cy] = this._diamond(B[l], sx, sy, RAD[l], l === LEVELS - 1);
      if (l > 0) continue;
      const fwd = this._lk(A[0], B[0], x, y, cx, cy);
      if (!fwd || Math.hypot(fwd[0] - cx, fwd[1] - cy) > this.o.lkMove) { this.stats.lost.lk++; return null; }
      // back from where it was found, by the motion predicted (a jump to a lookalike does not come back)
      const back = this._lk(B[0], A[0], fwd[0], fwd[1], fwd[0] - (px - x), fwd[1] - (py - y));
      if (!back || Math.hypot(back[0] - x, back[1] - y) > this.o.fb) { this.stats.lost.fb++; return null; }
      return fwd;
    }
    return null;
  }

  // this.tmpl: the zero-mean patch of radius r of level L around (x, y); its standard deviation, -1 at the border
  _template({ data, w, h }, x, y, r) {
    if (x < r + 1 || y < r + 1 || x > w - r - 2 || y > h - r - 2) return -1;
    const n = (2 * r + 1) ** 2, t = this.tmpl, s = patch(data, w, x, y, r, t);
    let s2 = 0;
    for (let i = 0; i < n; i++) s2 += t[i] * t[i];
    const m = s / n;
    for (let i = 0; i < n; i++) t[i] -= m;
    return Math.sqrt(Math.max(0, s2 / n - m * m));
  }

  // Zero-mean SAD per pixel of this.tmpl at integer centre (cx, cy) of level L
  _cost({ data, w, h }, cx, cy, r) {
    if (cx < r || cy < r || cx > w - r - 1 || cy > h - r - 1) return Infinity;
    const n = (2 * r + 1) ** 2, t = this.tmpl;
    let s = 0;
    for (let dy = -r; dy <= r; dy++) for (let dx = -r, o = (cy + dy) * w + cx - r; dx <= r; dx++, o++) s += data[o];
    const m = s / n;
    let e = 0, k = 0;
    for (let dy = -r; dy <= r; dy++) for (let dx = -r, o = (cy + dy) * w + cx - r; dx <= r; dx++, o++) e += Math.abs(data[o] - m - t[k++]);
    return e / n;
  }

  // Diamond search from (cx, cy): the large diamond until its centre is the best (large), then the small one
  _diamond(L, cx, cy, r, large) {
    let c = this._cost(L, cx, cy, r);
    const run = (pattern, iterations) => {
      for (let it = 0; it < iterations; it++) {
        let bx = cx, by = cy, bc = c;
        for (const [dx, dy] of pattern) {
          const v = this._cost(L, cx + dx, cy + dy, r);
          if (v < bc) { bc = v; bx = cx + dx; by = cy + dy; }
        }
        if (bx === cx && by === cy) return;
        cx = bx; cy = by; c = bc;
      }
    };
    if (large) run(LDSP, 16);
    run(SDSP, 8);
    return [cx, cy, c];
  }

  // Translational Lucas-Kanade (inverse compositional, the brightness offset removed) of the point (x, y) of level
  // A, from (bx, by) in level B: [x, y] or null
  _lk(A, B, x, y, bx, by, iterations = 8) {
    const r = this.o.lkRadius, { w, h } = A, a = A.data, b = B.data, n = (2 * r + 1) ** 2;
    const inside = (u, v) => u >= r + 2 && v >= r + 2 && u <= w - r - 3 && v <= h - r - 3;
    if (!inside(x, y)) return null;
    const T = this.tmpl, gx = this.gx, gy = this.gy, E = this.ext, C = this.cur, S = 2 * r + 3;
    patch(a, w, x, y, r + 1, E);
    let k = 0, h11 = 0, h12 = 0, h22 = 0, tm = 0;
    for (let dy = -r; dy <= r; dy++) for (let dx = -r, e = (dy + r + 1) * S + 1; dx <= r; dx++, k++, e++) {
      T[k] = E[e]; tm += T[k];
      gx[k] = 0.5 * (E[e + 1] - E[e - 1]);
      gy[k] = 0.5 * (E[e + S] - E[e - S]);
      h11 += gx[k] * gx[k]; h12 += gx[k] * gy[k]; h22 += gy[k] * gy[k];
    }
    tm /= n;
    const det = h11 * h22 - h12 * h12;
    if (!(det > 1e-3 * n * n)) return null;
    for (let it = 0; it < iterations; it++) {
      if (!inside(bx, by)) return null;
      const im = patch(b, w, bx, by, r, C) / n;
      let b1 = 0, b2 = 0;
      for (k = 0; k < n; k++) {
        const e = (C[k] - im) - (T[k] - tm);
        b1 += gx[k] * e; b2 += gy[k] * e;
      }
      const ux = (h22 * b1 - h12 * b2) / det, uy = (h11 * b2 - h12 * b1) / det;
      bx -= ux; by -= uy;
      if (ux * ux + uy * uy < 1e-4) break;
    }
    return [bx, by];
  }
}

// The pairs with the tracks' matches (trackPairs: PointTracker result's, keyframe indices as `images`'): per pair,
// the tracks' matches and the SIFT ones of keypoints no track matched, verified again (frontend.js's F check);
// pairs only the tracks have are added as 'seq'. images: the mapper's ([x, y] keypoints, the added ones included).
// Returns { pairs (sorted), stats: { pairs (with tracks), added (matches more than SIFT's) } }.
export function mergeTrackPairs(pairs, images, trackPairs, { maxError = 4, minInliers = 15, seed = 1 } = {}) {
  const byKey = new Map(pairs.map(p => [`${p.i},${p.j}`, p])), stats = { pairs: 0, added: 0 };
  for (const { i, j, matches } of trackPairs) {
    const key = `${i},${j}`, old = byKey.get(key), raw = Array.from(matches), tracked = new Set();
    for (let q = 0; q < matches.length; q += 2) tracked.add(matches[q]);
    if (old) for (let q = 0; q < old.matches.length; q += 2) if (!tracked.has(old.matches[q])) raw.push(old.matches[q], old.matches[q + 1]);
    const ka = images[i].keypoints, kb = images[j].keypoints, xy = new Float32Array(2 * raw.length);
    for (let q = 0; q < raw.length / 2; q++) {
      xy[4 * q] = ka[2 * raw[2 * q]]; xy[4 * q + 1] = ka[2 * raw[2 * q] + 1];
      xy[4 * q + 2] = kb[2 * raw[2 * q + 1]]; xy[4 * q + 3] = kb[2 * raw[2 * q + 1] + 1];
    }
    const res = verifyPacked(xy, (seed * 7919 + i * 65537 + j) >>> 0, { maxError, minInliers });
    if (!res || (old && res.inliers.length <= old.matches.length / 2)) continue;  // the SIFT pair stays as it was
    const m = new Uint32Array(2 * res.inliers.length);
    res.inliers.forEach((k, q) => { m[2 * q] = raw[2 * k]; m[2 * q + 1] = raw[2 * k + 1]; });
    stats.pairs++;
    stats.added += m.length / 2 - (old ? old.matches.length / 2 : 0);
    byKey.set(key, { i, j, type: old?.type ?? 'seq', F: res.F, matches: m });
  }
  return { pairs: [...byKey.values()].sort((a, b) => a.i - b.i || a.j - b.j), stats };
}
