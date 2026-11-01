// overlay.js — A video's on-screen display (OSD: text, boxes, a reticle burnt into every frame) found from
// what does not move while the scene does (DESIGN-sfm §5.1.1): the pixels a mask, which the SfM's features and
// the trainer's loss leave out. From gray frames at a small size (a grid), spread over the video:
//   still: the pixels within STILL_TOL levels of their median over the frames in STILL_SHARE of them;
//   kept: the still regions with text or lines in them (EDGE_SHARE of their pixels on an edge of the median
//         frame: a burnt-in label is sharp there, a patch of plain sky under a pan is still too but flat);
//   then the small holes they enclose filled (digits that change inside a fixed box), closed, and a margin.
// No mask when the camera barely moves (most of the frame still: an overlay cannot be told from the scene),
// and no region larger than MAX_REGION of the frame.
"use strict";

export const OVERLAY = {
  gridDim: 640,       // the grid's long side (px; at 320 a 2 px reticle and thin text average away)
  maxFrames: 200,     // frames used, spread over the video
  stillTol: 6,        // levels from the pixel's median
  stillShare: 0.85,   // of the frames within stillTol: a still pixel
  edgeLevel: 24,      // the median frame's step to a neighbour (levels): an edge
  edgeShare: 0.15,    // of a still region's pixels on an edge: an overlay
  maxRegion: 0.25,    // of the frame: a larger still region is not an overlay
  maxStill: 0.6,      // of the frame still: the camera does not move enough, no mask
  maxHole: 0.005,     // of the frame: enclosed holes up to this size are filled (changing digits; a reticle's inside is larger)
  close: 2,           // closing radius (grid px)
  margin: 1,          // added around the mask (grid px)
};

// frames: Uint8Array gray frames of w x h (the grid). Returns { mask: Uint8Array (1: overlay) | null, w, h,
// share (of the frame masked), still (share still), regions (kept), why (when no mask) }
export function staticMask(frames, w, h, o = OVERLAY) {
  const n = frames.length, P = w * h;
  if (n < 10) return { mask: null, w, h, share: 0, still: 0, regions: 0, why: `${n} frames: too few` };
  const med = new Uint8Array(P), still = new Uint8Array(P), col = new Uint8Array(n);
  let stillCount = 0;
  for (let p = 0; p < P; p++) {
    for (let k = 0; k < n; k++) col[k] = frames[k][p];
    col.sort();
    const m = col[n >> 1];
    med[p] = m;
    let near = 0;
    for (let k = 0; k < n; k++) if (Math.abs(col[k] - m) <= o.stillTol) near++;
    if (near >= o.stillShare * n) { still[p] = 1; stillCount++; }
  }
  const stillShare = stillCount / P;
  if (stillShare > o.maxStill) return { mask: null, w, h, share: 0, still: stillShare, regions: 0, why: `${(100 * stillShare).toFixed(0)}% of the frame still: the camera does not move enough` };
  const edge = edges(med, w, h, o.edgeLevel);
  // the still regions (4-connected), kept when they hold text or lines
  const label = new Int32Array(P).fill(-1), keep = new Uint8Array(P), queue = new Int32Array(P);
  let regions = 0;
  for (let s = 0; s < P; s++) {
    if (!still[s] || label[s] >= 0) continue;
    let head = 0, tail = 0, nEdge = 0;
    queue[tail++] = s; label[s] = s;
    while (head < tail) {
      const p = queue[head++], x = p % w, y = (p - x) / w;
      nEdge += edge[p];
      for (const q of [x > 0 ? p - 1 : -1, x < w - 1 ? p + 1 : -1, y > 0 ? p - w : -1, y < h - 1 ? p + w : -1]) {
        if (q >= 0 && still[q] && label[q] < 0) { label[q] = s; queue[tail++] = q; }
      }
    }
    if (tail <= o.maxRegion * P && nEdge >= o.edgeShare * tail) {
      regions++;
      for (let i = 0; i < tail; i++) keep[queue[i]] = 1;
    }
  }
  if (!regions) return { mask: null, w, h, share: 0, still: stillShare, regions: 0, why: 'no still region with text or lines' };
  let mask = fillHoles(keep, w, h, Math.round(o.maxHole * P));
  mask = erode(dilate(mask, w, h, o.close), w, h, o.close);
  mask = dilate(mask, w, h, o.margin);
  let m = 0;
  for (let p = 0; p < P; p++) m += mask[p];
  return { mask, w, h, share: m / P, still: stillShare, regions, why: null };
}

// 1 where img differs from a 4-neighbour by at least `level` (a line 1-2 px wide is an edge across: central
// differences halve it)
function edges(img, w, h, level) {
  const out = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const p = y * w + x, v = img[p];
    if (Math.max(Math.abs(v - img[p - 1]), Math.abs(v - img[p + 1]), Math.abs(v - img[p - w]), Math.abs(v - img[p + w])) >= level) out[p] = 1;
  }
  return out;
}

// The mask with its enclosed holes (0-regions not touching the border) of at most maxArea pixels filled
function fillHoles(mask, w, h, maxArea) {
  const out = mask.slice(), seen = new Uint8Array(w * h), queue = new Int32Array(w * h);
  for (let s = 0; s < w * h; s++) {
    if (mask[s] || seen[s]) continue;
    let head = 0, tail = 0, border = false;
    queue[tail++] = s; seen[s] = 1;
    while (head < tail) {
      const p = queue[head++], x = p % w, y = (p - x) / w;
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) border = true;
      for (const q of [x > 0 ? p - 1 : -1, x < w - 1 ? p + 1 : -1, y > 0 ? p - w : -1, y < h - 1 ? p + w : -1]) {
        if (q >= 0 && !mask[q] && !seen[q]) { seen[q] = 1; queue[tail++] = q; }
      }
    }
    if (!border && tail <= maxArea) for (let i = 0; i < tail; i++) out[queue[i]] = 1;
  }
  return out;
}

// Square dilation / erosion of radius r (separable; outside the frame counts as 0 for dilation, 1 for erosion)
function dilate(m, w, h, r) { return r > 0 ? morph(m, w, h, r, 1) : m; }
function erode(m, w, h, r) { return r > 0 ? morph(m, w, h, r, 0) : m; }
function morph(m, w, h, r, hit) {
  const tmp = new Uint8Array(w * h), out = new Uint8Array(w * h), miss = 1 - hit;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = miss;
    for (let d = -r; d <= r && v === miss; d++) {
      const xx = x + d;
      v = xx < 0 || xx >= w ? miss : (m[y * w + xx] === hit ? hit : miss);
    }
    tmp[y * w + x] = v;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = miss;
    for (let d = -r; d <= r && v === miss; d++) {
      const yy = y + d;
      v = yy < 0 || yy >= h ? miss : (tmp[yy * w + x] === hit ? hit : miss);
    }
    out[y * w + x] = v;
  }
  return out;
}

// The mask grown by r of its cells (no corner tracked near it, frames_worker.js)
export const grownMask = (ov, r) => ({ w: ov.w, h: ov.h, mask: dilate(ov.mask, ov.w, ov.h, r) });

// The mask at image pixel (x, y) of a W x H image (nearest grid cell); 0 without a mask
export function maskAt(ov, x, y, W, H) {
  if (!ov?.mask) return 0;
  const gx = Math.min(ov.w - 1, Math.max(0, Math.floor(x * ov.w / W))), gy = Math.min(ov.h - 1, Math.max(0, Math.floor(y * ov.h / H)));
  return ov.mask[gy * ov.w + gx];
}

// SIFT keypoints ([x, y, scale, orientation] per keypoint, pixel centers at +0.5) of a W x H frame outside the
// mask, with their descriptors (128 bytes each): { keypoints, descriptors, dropped }
export function dropMasked(keypoints, descriptors, ov, W, H) {
  const n = keypoints.length / 4, keep = [];
  for (let i = 0; i < n; i++) if (!maskAt(ov, keypoints[4 * i] - 0.5, keypoints[4 * i + 1] - 0.5, W, H)) keep.push(i);
  if (keep.length === n) return { keypoints, descriptors, dropped: 0 };
  const kp = new Float32Array(4 * keep.length), de = new Uint8Array(128 * keep.length);
  keep.forEach((i, j) => { kp.set(keypoints.subarray(4 * i, 4 * i + 4), 4 * j); de.set(descriptors.subarray(128 * i, 128 * i + 128), 128 * j); });
  return { keypoints: kp, descriptors: de, dropped: n - keep.length };
}

// The mask for a saved analysis (JSON: base64) and back; unpackMask memoizes the bytes on the packed object
export function packMask({ mask, w, h, share, still, regions, why }) {
  let s = '';
  if (mask) for (let i = 0; i < mask.length; i += 0x8000) s += String.fromCharCode(...mask.subarray(i, i + 0x8000));
  return { w, h, share, still, regions, why, b64: mask ? btoa(s) : null };
}
export function unpackMask(p) {
  if (!p?.b64) return null;
  if (!p._mask) Object.defineProperty(p, '_mask', { value: Uint8Array.from(atob(p.b64), c => c.charCodeAt(0)), enumerable: false });
  return { w: p.w, h: p.h, mask: p._mask };
}
