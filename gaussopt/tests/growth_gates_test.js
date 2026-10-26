// reconcileAndGrow's E9 growth options (octree.js): splitMinErr (an absolute gate on a splat's mean error
// R / W), growRate (at most that share of new splats, the highest R first), splitInPlane (split / clone
// offsets off the splat's thinnest axis), growScore / growThreshold (grow_by 'grad'), splitMode 'shape'.
import { LooseOctree, MOMENT_FLOATS } from '../src/octree.js';
import { SPLAT_FLOATS } from '../src/colmap.js';
import { assert, rng } from './util.js';
import { quatToMat3 } from '../src/math.js';

// 200 splats: 100 with little error, then 100 with much error R in 1.1..2.1, over Tu (R grows with i); of those, the
// first 50 have a mean error R / W of about 0.1 (W 10-20), the last 50 about 0.01 (W 100-200)
function scene() {
  const r = rng(5), n = 200, params = new Float32Array(n * SPLAT_FLOATS), moments = new Float32Array(n * MOMENT_FLOATS);
  for (let i = 0; i < n; i++) {
    params.set([r() * 4 - 2, r() * 4 - 2, r() * 4 - 2, 2, Math.log(0.05), Math.log(0.05), Math.log(0.05), 0, 1, 0, 0, 0, 0.1, 0.2, 0.3, 0],
               i * SPLAT_FLOATS);
    const m = i * MOMENT_FLOATS, R = i < 100 ? 0.01 : 1.1 + (i - 100) / 100, W = i < 150 ? 10 * R : 100 * R;
    moments[m] = W; moments[m + 1] = W * W / 20; moments[m + 2] = R;
  }
  return { params, moments, n };
}

function grow(opts) {
  const { params, moments, n } = scene();
  const oct = new LooseOctree({ maxLeafSize: 64 }).build(params, n, null);
  return oct.reconcileAndGrow(params, n, moments, null, { maxSplats: 10000, enableFloaters: false, enableMerge: false, ...opts });
}
const grown = (s) => s.split + s.cloned;

Deno.test('split_min_err: only the splats whose mean error R / W is above it grow', () => {
  const off = grow({}).stats;
  assert(grown(off) === 100, `ungated: ${grown(off)} grown, expected 100`);
  const gate = grow({ splitMinErr: 0.05 }).stats;
  assert(grown(gate) === 50 && gate.splitLowError === 50, `gate 0.05: ${grown(gate)} grown, ${gate.splitLowError} held`);
  const high = grow({ splitMinErr: 0.5 }).stats;
  assert(grown(high) === 0, `gate 0.5: ${grown(high)} grown`);
});

Deno.test('grow_rate: at most that share of new splats, the highest error first', () => {
  const res = grow({ growRate: 0.1 }), s = res.stats;
  assert(grown(s) === 20 && s.splitOverRate === 80, `rate 0.1 of 200: ${grown(s)} grown, ${s.splitOverRate} held`);
  assert(res.count === 220, `count ${res.count}, expected 220`);
  // with the error gate too: the cut is taken among the splats that pass it (the 50 with R / W ~ 0.1, R 1.1-1.6)
  const both = grow({ growRate: 0.1, splitMinErr: 0.05 }).stats;
  assert(grown(both) === 20 && both.splitLowError === 50 && both.splitOverRate === 30,
         `rate + gate: ${grown(both)} grown, ${both.splitLowError} low error, ${both.splitOverRate} over the rate`);
  // the grown ones are the highest R: a split replaces its parent by two fresh children (srcIdx -1), so the
  // rows gone are the splats that split, here 130-149, the 20 highest R of the 50 that pass the gate
  const g = grow({ growRate: 0.1, splitMinErr: 0.05 }), kept = new Set();
  for (let k = 0; k < g.count; k++) if (g.srcIdx[k] >= 0) kept.add(g.srcIdx[k]);
  const gone = [];
  for (let i = 0; i < 200; i++) if (!kept.has(i)) gone.push(i);
  assert(g.stats.split === 20 && gone.join() === Array.from({ length: 20 }, (_, k) => 130 + k).join(), `split: ${gone}`);
});

// One round splat (thinnest axis z, barely) at the origin whose error lies along z, among quiet splats away
// from it: its children, without splitInPlane, move along z; with it, they stay at z = 0
function zSpread(splitInPlane) {
  const r = rng(9), n = 40, params = new Float32Array(n * SPLAT_FLOATS), moments = new Float32Array(n * MOMENT_FLOATS);
  for (let i = 0; i < n; i++) {
    const p = i === 0 ? [0, 0, 0] : [1 + r() * 2, r() * 4 - 2, r() * 4 - 2];
    params.set([...p, 2, Math.log(0.05), Math.log(0.05), Math.log(i === 0 ? 0.04 : 0.05), 0, 1, 0, 0, 0, 0.1, 0.2, 0.3, 0], i * SPLAT_FLOATS);
    const m = i * MOMENT_FLOATS, R = i === 0 ? 1 : 0.01;
    moments[m] = 10; moments[m + 1] = 5; moments[m + 2] = R;
    if (i === 0) { moments[m + 5] = R * 0.01; moments[m + 8] = R * 0.03 * 0.03; }  // centroid +0.01 z, variance along z
  }
  const oct = new LooseOctree({ maxLeafSize: 64 }).build(params, n, null);
  const res = oct.reconcileAndGrow(params, n, moments, null, { maxSplats: 1000, enableFloaters: false, enableMerge: false, splitInPlane });
  const near = [];
  for (let k = 0; k < res.count; k++) {
    const b = k * SPLAT_FLOATS;
    if (Math.hypot(res.params[b], res.params[b + 1], res.params[b + 2]) < 0.5) near.push([res.params[b], res.params[b + 1], res.params[b + 2]]);
  }
  return { near, stats: res.stats };
}

Deno.test('split_in_plane: the children of a splat whose error lies off its plane stay on the plane', () => {
  const off = zSpread(false), on = zSpread(true);
  assert(off.stats.split === 1 && on.stats.split === 1, `splits: ${off.stats.split}, ${on.stats.split}`);
  assert(off.near.length === 2 && on.near.length === 2, `children: ${off.near.length}, ${on.near.length}`);
  const zOff = Math.max(...off.near.map(p => Math.abs(p[2]))), zOn = Math.max(...on.near.map(p => Math.abs(p[2])));
  const xyOn = Math.max(...on.near.map(p => Math.hypot(p[0], p[1])));
  assert(zOff > 0.01, `off: children only ${zOff} off the plane (expected along z)`);
  assert(zOn < 1e-6 && xyOn > 0.005, `on: children at |z| ${zOn}, ${xyOn} apart in the plane`);
});

Deno.test('grow_by grad: the splats whose score is over the threshold grow, whatever their error', () => {
  const score = new Float32Array(200);
  for (let i = 0; i < 200; i++) score[i] = i % 20 === 0 ? 1e-3 * (1 + i / 200) : 1e-6;  // 10 over (distinct), among the low-error ones too
  const s = grow({ growScore: score, growThreshold: 1e-4 }).stats;
  assert(grown(s) === 10, `score over 1e-4 on 10: ${grown(s)} grown`);
  const rated = grow({ growScore: score, growThreshold: 1e-4, growRate: 0.02 }).stats;  // room for 4
  assert(grown(rated) === 4 && rated.splitOverRate === 6, `rate 0.02: ${grown(rated)} grown, ${rated.splitOverRate} held`);
});

// 64 copies of one flat, rotated splat (scales 0.1 / 0.01 / 0.001 along its axes; opacity 0.6), all split by shape
Deno.test('split_mode shape: two halves within the splat, scales / sqrt 2, opacity 1 - sqrt(1 - o)', () => {
  const n = 64, params = new Float32Array(n * SPLAT_FLOATS), moments = new Float32Array(n * MOMENT_FLOATS);
  const q = [Math.cos(0.3), Math.sin(0.3) / Math.sqrt(2), Math.sin(0.3) / Math.sqrt(2), 0];
  for (let i = 0; i < n; i++) {
    params.set([i * 0.5, 0, 0, Math.log(0.6 / 0.4), Math.log(0.1), Math.log(0.01), Math.log(0.001), 0, ...q, 0.1, 0.2, 0.3, 0], i * SPLAT_FLOATS);
  }
  const oct = new LooseOctree({ maxLeafSize: 64 }).build(params, n, null);
  const score = new Float32Array(n).fill(1);
  const res = oct.reconcileAndGrow(params, n, moments, null, { maxSplats: 1000, enableFloaters: false, enableMerge: false,
                                                                growScore: score, growThreshold: 0.5, splitMode: 'shape' });
  assert(res.stats.split === n && res.count === 2 * n, `${res.stats.split} split, ${res.count} splats`);
  const Rm = quatToMat3(q), axes = [0, 1, 2].map(k => [Rm[k], Rm[3 + k], Rm[6 + k]]);
  const byX = new Map();
  for (let k = 0; k < res.count; k++) {
    const p = res.params.subarray(k * SPLAT_FLOATS, (k + 1) * SPLAT_FLOATS);
    const o = 1 / (1 + Math.exp(-p[3]));
    assert(Math.abs(o - (1 - Math.sqrt(0.4))) < 1e-5, `opacity ${o}`);
    assert(Math.abs(p[4] - Math.log(0.1 / Math.SQRT2)) < 1e-5 && Math.abs(p[6] - Math.log(0.001 / Math.SQRT2)) < 1e-5, `scales ${p[4]}, ${p[6]}`);
    const i = Math.round(p[0] / 0.5);  // offsets are within 0.5 x 4 sigma of 0.1: the parent's x
    (byX.has(i) ? byX.get(i) : byX.set(i, []).get(i)).push([p[0], p[1], p[2]]);
  }
  let along = 0, across = 0;
  for (const [i, [a, b]] of byX) {
    const mid = [(a[0] + b[0]) / 2 - i * 0.5, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
    assert(Math.hypot(...mid) < 1e-5, `pair ${i} not centered on the parent: ${mid}`);
    const d = [0, 1, 2].map(c => (b[c] - a[c]) / 2), on = axes.map(ax => Math.abs(ax[0] * d[0] + ax[1] * d[1] + ax[2] * d[2]));
    along += on[0]; across += on[2];
  }
  assert(along / n > 0.01 && across / n < 0.001, `mean offset along the long axis ${along / n}, the thin one ${across / n}`);
});
