// reconcileAndGrow's split_mode 'shape' (octree.js splitByShape, the default since E9): a splat whose error mass R is
// over Tu becomes two halves drawn within its own shape.
import { LooseOctree, MOMENT_FLOATS } from '../src/octree.js';
import { SPLAT_FLOATS } from '../src/colmap.js';
import { assert } from './util.js';
import { quatToMat3 } from '../src/math.js';

// 64 copies of one flat, rotated splat (scales 0.1 / 0.01 / 0.001 along its axes; opacity 0.6), all seen (W 1); the first 32
// with error mass R 1, over Tu (1.35 x the mean 0.5), split by shape, the others with none kept as they are
Deno.test('split_mode shape: two halves within the splat, scales / sqrt 2, opacity 1 - sqrt(1 - o)', () => {
  const n = 64, params = new Float32Array(n * SPLAT_FLOATS), moments = new Float32Array(n * MOMENT_FLOATS);
  const q = [Math.cos(0.3), Math.sin(0.3) / Math.sqrt(2), Math.sin(0.3) / Math.sqrt(2), 0];
  for (let i = 0; i < n; i++) {
    params.set([i * 0.5, 0, 0, Math.log(0.6 / 0.4), Math.log(0.1), Math.log(0.01), Math.log(0.001), 0, ...q, 0.1, 0.2, 0.3, 0], i * SPLAT_FLOATS);
    moments[i * MOMENT_FLOATS] = 1;
    moments[i * MOMENT_FLOATS + 2] = i < n / 2 ? 1 : 0;
  }
  const oct = new LooseOctree({ maxLeafSize: 64 }).build(params, n, null);
  const res = oct.reconcileAndGrow(params, n, moments, null, { maxSplats: 1000, enableFloaters: false, enableMerge: false, splitMode: 'shape' });
  assert(res.stats.split === n / 2 && res.count === n + n / 2, `${res.stats.split} split, ${res.count} splats`);
  const Rm = quatToMat3(q), axes = [0, 1, 2].map(k => [Rm[k], Rm[3 + k], Rm[6 + k]]);
  const byX = new Map();
  for (let k = 0; k < res.count; k++) {
    const p = res.params.subarray(k * SPLAT_FLOATS, (k + 1) * SPLAT_FLOATS);
    const i = Math.round(p[0] / 0.5);  // offsets are within 0.5 x 4 sigma of 0.1: the parent's x
    const o = 1 / (1 + Math.exp(-p[3]));
    if (i >= n / 2) {  // no error: kept as it was
      assert(Math.abs(o - 0.6) < 1e-5 && p[0] === i * 0.5, `splat ${i} without error changed: opacity ${o}, x ${p[0]}`);
      continue;
    }
    assert(Math.abs(o - (1 - Math.sqrt(0.4))) < 1e-5, `opacity ${o}`);
    assert(Math.abs(p[4] - Math.log(0.1 / Math.SQRT2)) < 1e-5 && Math.abs(p[6] - Math.log(0.001 / Math.SQRT2)) < 1e-5, `scales ${p[4]}, ${p[6]}`);
    (byX.has(i) ? byX.get(i) : byX.set(i, []).get(i)).push([p[0], p[1], p[2]]);
  }
  let along = 0, across = 0;
  for (const [i, [a, b]] of byX) {
    const mid = [(a[0] + b[0]) / 2 - i * 0.5, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
    assert(Math.hypot(...mid) < 1e-5, `pair ${i} not centered on the parent: ${mid}`);
    const d = [0, 1, 2].map(c => (b[c] - a[c]) / 2), on = axes.map(ax => Math.abs(ax[0] * d[0] + ax[1] * d[1] + ax[2] * d[2]));
    along += on[0]; across += on[2];
  }
  assert(byX.size === n / 2 && [...byX.values()].every(v => v.length === 2), `${byX.size} pairs`);
  assert(along / byX.size > 0.01 && across / byX.size < 0.001, `mean offset along the long axis ${along / byX.size}, the thin one ${across / byX.size}`);
});
