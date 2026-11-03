// Points tracked through every frame (src/sfm/track.js, DESIGN-sfm §5.2.1): positions on a known motion, a change
// of content no track follows, and the tracks' matches merged into the SIFT pairs.
import { PointTracker, mergeTrackPairs, TRACK_DEFAULTS } from '../src/sfm/track.js';
import { makeRng } from '../src/math.js';
import { assert } from './util.js';

const W = 320, H = 240;

// Dense smooth random texture (overlapping blobs: few patches alike) moved by (dx, dy); inside `swap` ([x0, y0, x1,
// y1], in the image), another texture; inside `grating` (moving with the rest), a pattern of period 4 px both ways
const CELL = 24;  // the blobs by cell of the texture's own coordinates: a blob reaches 3 sigma = 21 px at most
const blobCells = new Map();
function blobs(seed) {
  if (blobCells.has(seed)) return blobCells.get(seed);
  const r = makeRng(seed), cells = new Map();
  for (let k = 0; k < 2000; k++) {
    const b = [r() * (W + 100) - 50, r() * (H + 100) - 50, 2 + 5 * r(), 120 * (r() - 0.5)], key = `${Math.floor(b[0] / CELL)},${Math.floor(b[1] / CELL)}`;
    if (!cells.has(key)) cells.set(key, []);
    cells.get(key).push(b);
  }
  blobCells.set(seed, cells);
  return cells;
}
function texture(dx, dy, seed = 3, swap = null, grating = null) {
  const a = blobs(seed), b = blobs(seed + 100), out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const inside = swap && x >= swap[0] && x < swap[2] && y >= swap[1] && y < swap[3], cells = inside ? b : a;
      const sx = x - dx, sy = y - dy, gx = Math.floor(sx / CELL), gy = Math.floor(sy / CELL);
      let v = 128;
      for (let j = gy - 1; j <= gy + 1; j++) for (let i = gx - 1; i <= gx + 1; i++) {
        for (const [cx, cy, s, amp] of cells.get(`${i},${j}`) ?? []) {
          const d2 = (sx - cx) ** 2 + (sy - cy) ** 2;
          if (d2 < 9 * s * s) v += amp * Math.exp(-d2 / (2 * s * s));
        }
      }
      if (grating && x - dx >= grating[0] && x - dx < grating[2] && y - dy >= grating[1] && y - dy < grating[3]) v = 128 + 50 * (Math.cos(Math.PI * (x - dx) / 2) + Math.cos(Math.PI * (y - dy) / 2));
      out[y * W + x] = v;
    }
  }
  return out;
}

// SIFT-like keypoints (4 floats each) on a grid of the first frame
function grid(step = 20, margin = 30) {
  const kp = [];
  for (let y = margin; y < H - margin; y += step) for (let x = margin; x < W - margin; x += step) kp.push(x + 0.37, y + 0.61, 1, 0);
  return Float32Array.from(kp);
}

// The tracker over `frames` frames moving by `v` a frame, keyframes at `keys`; the keypoints of keyframe 0 the grid,
// of the others those of the grid's points at their true place for every other one (the rest added by the tracks).
// Returns the visits' errors to the true positions, per keyframe, and the result.
function run({ frames = 13, v = [1.7, -0.9], keys = [0, 6, 12], swap = null, swapFrom = 6, grating = null, options = {} } = {}) {
  const tracker = new PointTracker(W, H, options), start = grid(), n = start.length / 4;
  const sift = new Map();
  for (let f = 0; f < frames; f++) {
    tracker.frame(texture(v[0] * f, v[1] * f, 3, f >= swapFrom ? swap : null, grating));
    const k = keys.indexOf(f);
    if (k < 0) continue;
    const kp = [];
    if (k === 0) kp.push(...start);
    else for (let i = 0; i < n; i += 2) kp.push(start[4 * i] + v[0] * f, start[4 * i + 1] + v[1] * f, 1, 0);
    sift.set(k, Float32Array.from(kp));
    tracker.keyframe(k, sift.get(k));
  }
  const res = tracker.result();
  // a match (k0 point i -> keyframe k): the grid point i at frame keys[k]
  const errors = keys.map(() => []);
  for (const { i, j, matches } of res.pairs) {
    if (i !== 0) continue;
    const kp = sift.get(j), added = res.added.get(j), nk = kp.length / 4;
    for (let q = 0; q < matches.length; q += 2) {
      const p = matches[q], t = matches[q + 1];
      const [x, y] = t < nk ? [kp[4 * t], kp[4 * t + 1]] : [added[2 * (t - nk)], added[2 * (t - nk) + 1]];
      const tx = start[4 * p] + v[0] * keys[j], ty = start[4 * p + 1] + v[1] * keys[j];
      errors[j].push({ p, e: Math.hypot(x - tx, y - ty), snapped: t < nk });
    }
  }
  return { errors, res, tracker };
}

Deno.test('tracks: every keyframe visit on a known motion within 0.3 px (median 0.1), at a SIFT keypoint or an added one', () => {
  const { errors, res } = run();
  for (const j of [1, 2]) {
    const e = errors[j];
    assert(e.length >= 60, `keyframe ${j}: ${e.length} grid points tracked`);
    // the added ones (a snapped visit is at the planted keypoint): chained from frame to frame, up to 12 steps
    const sorted = e.filter(x => !x.snapped).map(x => x.e).sort((p, q) => p - q), worst = sorted.at(-1), median = sorted[sorted.length >> 1];
    assert(worst < 0.3 && median < 0.1, `keyframe ${j}: added visits worst ${worst.toFixed(3)} px, median ${median.toFixed(3)}`);
    assert(e.some(x => x.snapped) && e.some(x => !x.snapped), `keyframe ${j}: snapped and added keypoints both`);
  }
  assert(res.pairs.some(p => p.i === 0 && p.j === 2), 'a match of keyframes 0 and 2 (12 frames apart)');
});

// The changed square's points are lost or found no farther than maxJump off (a lookalike nearer than that is the F
// check's to drop), and the others tracked, but those whose window straddles its edge after the change: an occlusion
// edge drags them a little each frame, under every step's checks
export function swapCase(options = {}) {
  const swap = [100, 60, 220, 180], v = [1.7, -0.9], keys = [0, 6, 12], swapFrom = 6, margin = 12;
  const { errors, tracker } = run({ swap, swapFrom, v, keys, options });
  const s = grid();
  const nearEdge = (p, j) => {
    for (let f = swapFrom; f <= keys[j]; f++) {
      const x = s[4 * p] + v[0] * f, y = s[4 * p + 1] + v[1] * f;
      const inX = x > swap[0] - margin && x < swap[2] + margin, inY = y > swap[1] - margin && y < swap[3] + margin;
      const deep = x > swap[0] + margin && x < swap[2] - margin && y > swap[1] + margin && y < swap[3] - margin;
      if (inX && inY && !deep) return true;
    }
    return false;
  };
  const most = options.maxJump ?? TRACK_DEFAULTS.maxJump;
  return { off: [1, 2].map(j => errors[j].filter(x => x.e > Math.min(most, 50) && !nearEdge(x.p, j))), errors, tracker };
}

Deno.test('tracks: content that changes is not followed (no visit off by more than maxJump, but at its edge)', () => {
  const { off, errors, tracker } = swapCase();
  off.forEach((o, k) => assert(!o.length, `keyframe ${k + 1}: ${o.length} visits off their place (worst ${Math.max(0, ...o.map(x => x.e)).toFixed(1)} px)`));
  assert(errors[2].length >= 30, `${errors[2].length} points still tracked to keyframe 2`);
  const { lost } = tracker.stats;
  assert(lost.jump + lost.fb > 0, `the changed square's points lost: ${JSON.stringify(lost)}`);
});

// A grating (period 4 px) the first step of a track, its motion unknown, can find a period off; tracked back by the
// motion predicted, such a point comes back a period away from where it was (the forward-backward check)
export function gratingCase(options = {}) {
  const grating = [90, 50, 230, 190], { errors, tracker } = run({ grating, options });
  return { off: [1, 2].map(j => errors[j].filter(x => x.e > 1)), errors, tracker };
}

Deno.test('tracks: on a grating, no visit a period off (the forward-backward check)', () => {
  const { off, errors } = gratingCase();
  // without the check: 31 and 21 visits off, 3.7 to 7.7 px
  off.forEach((o, k) => assert(!o.length, `keyframe ${k + 1}: ${o.length} visits a period off (${o.map(x => x.e.toFixed(1)).join(', ')} px)`));
  assert(errors[2].length >= 30, `${errors[2].length} points still tracked to keyframe 2`);
});

Deno.test('tracks merged into the pairs: a SIFT pair grows, a wrong match is dropped, a pair only tracks have is added', () => {
  // points 3-8 deep, seen by cameras at x = 0, 0.3, 0.6 turned a little about y (f 300, centre 160, 120)
  const r = makeRng(7), n = 60, X = Array.from({ length: n }, () => [8 * (r() - 0.5), 6 * (r() - 0.5), 3 + 5 * r()]);
  const view = (cx, turn) => {
    const c = Math.cos(turn), s = Math.sin(turn), out = new Float32Array(2 * n);
    X.forEach(([x, y, z], q) => {
      const px = x - cx, xr = c * px - s * z, zr = s * px + c * z;
      out[2 * q] = 160 + 300 * xr / zr; out[2 * q + 1] = 120 + 300 * y / zr;
    });
    return out;
  };
  const a = view(0, 0), b = view(0.3, 0.03), c = view(0.6, 0.06);
  const images = [{ keypoints: a }, { keypoints: b }, { keypoints: c }];
  const range = (from, to) => Array.from({ length: to - from }, (_, q) => [from + q, from + q]).flat();
  const sift = [{ i: 0, j: 1, type: 'seq', F: null, matches: Uint32Array.from(range(0, 20)) }];
  const wrong = [55, 3];  // keypoint 55 of 0 with keypoint 3 of 1: off the motion
  const tracks = [{ i: 0, j: 1, matches: Uint32Array.from([...range(20, 50), ...wrong]) }, { i: 0, j: 2, matches: Uint32Array.from(range(0, 40)) }];
  const { pairs, stats } = mergeTrackPairs(sift, images, tracks);
  const p01 = pairs.find(p => p.i === 0 && p.j === 1), p02 = pairs.find(p => p.i === 0 && p.j === 2);
  assert(p01?.matches.length / 2 === 50, `pair 0-1: ${p01?.matches.length / 2} matches (20 SIFT + 30 tracked)`);
  for (let q = 0; q < p01.matches.length; q += 2) assert(!(p01.matches[q] === wrong[0] && p01.matches[q + 1] === wrong[1]), 'the wrong match dropped');
  assert(p02?.type === 'seq' && p02.matches.length / 2 === 40, `pair 0-2 added: ${p02?.type}, ${p02?.matches.length / 2}`);
  assert(stats.pairs === 2 && stats.added === 70, JSON.stringify(stats));
});
