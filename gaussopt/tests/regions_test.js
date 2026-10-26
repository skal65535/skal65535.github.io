// regions.js (DESIGN.md §9 E7): fixed cells over the splat centers, and each cell's window error.
import { makeRegionGrid, regionOf, regionStats, RegionTracker } from '../src/regions.js';
import { LooseOctree, relocate } from '../src/octree.js';
import { logit } from '../src/math.js';
import { MOMENT_FLOATS } from '../src/octree.js';
import { SPLAT_FLOATS } from '../src/colmap.js';
import { assert, assertClose } from './util.js';

Deno.test('regions: the grid spans the centers, each splat in its cell, W / R / n summed per cell', () => {
  // 1000 splats on a line x = 0..9.99 (y = z = 0), and one far outlier at x = 1000
  const n = 1001, p = new Float32Array(n * SPLAT_FLOATS), m = new Float32Array(n * MOMENT_FLOATS);
  for (let i = 0; i < 1000; i++) { p[i * SPLAT_FLOATS] = i / 100; m[i * MOMENT_FLOATS] = 1; m[i * MOMENT_FLOATS + 2] = i < 500 ? 2 : 0; }
  p[1000 * SPLAT_FLOATS] = 1000;
  const g = makeRegionGrid(p, n, 4);
  assert(g.lo[0] >= 0.1 && g.lo[0] < 0.3 && g.hi[0] > 9.7 && g.hi[0] < 10, `x box ${g.lo[0]}..${g.hi[0]}: the outlier moved it`);
  assert(regionOf(g, -5, 0, 0) === 0 && regionOf(g, 1000, 0, 0) === 3 && regionOf(g, 0, 1e9, 1e9) === 4 * (3 + 4 * 3), 'border cells');
  const c = regionStats(g, p, n, m);
  let W = 0, R = 0, N = 0;
  for (let k = 0; k < c.length; k += 3) { W += c[k]; R += c[k + 1]; N += c[k + 2]; }
  assertClose([W, R, N], [1000, 1000, 1001], 1e-6, 'totals');
  // the first half (x < 5) holds the error: cells 0 and 1 (the box ~0.2..9.8, 4 cells of ~2.4)
  assert(c[1] + c[4] > 950 && c[7] + c[10] < 50, `error by cell ${[c[1], c[4], c[7], c[10]]}`);
});

Deno.test('regions: a cell is stuck when its error is above the scene\'s and falls slower; the mask follows the splats', () => {
  // 2 cells on x (res 2): splats 0..99 at x < 0, 100..199 at x > 0; the scene's error halves over the 6 windows
  const n = 200, p = new Float32Array(n * SPLAT_FLOATS);
  for (let i = 0; i < n; i++) p[i * SPLAT_FLOATS] = i < 100 ? -1 - i / 100 : 1 + i / 100;
  const win = (eLeft, eRight) => {
    const m = new Float32Array(n * MOMENT_FLOATS);
    for (let i = 0; i < n; i++) { m[i * MOMENT_FLOATS] = 1; m[i * MOMENT_FLOATS + 2] = i < 100 ? eLeft : eRight; }
    return m;
  };
  const cases = [  // [left error by window, right error], the left cell stuck?
    [[2, 2, 2, 1.9, 1.9, 1.9], [1, 1, 1, 0.3, 0.3, 0.3], 1],   // high and slow: stuck
    [[2, 2, 2, 0.6, 0.6, 0.6], [1, 1, 1, 0.5, 0.5, 0.5], 0],   // high but faster than the scene
    [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [2, 2, 2, 0.6, 0.6, 0.6], 0],  // slow but low: done
  ];
  for (const [L, R, want] of cases) {
    const t = new RegionTracker(2);
    let stuck = null;
    for (let w = 0; w < 6; w++) {
      stuck = t.update(p, n, win(L[w], R[w]));
      if (w < 5) assert(stuck === null, 'stuck before 6 windows');
    }
    const left = regionOf(t.grid, -1.5, 0, 0);
    assert(stuck[left] === want, `left ${L} right ${R}: stuck ${stuck[left]}, want ${want}`);
    const mask = t.splatMask(p, n);
    assert(mask[0] === want && mask[150] === 0, 'splat mask');
  }
});

Deno.test('regions: the repair lowers the stuck splats\' split threshold and draws more relocations there', () => {
  // the octree pass: 100 splats, R 1.2 x the mean for 0..49 (no split: Tu = 1.35 mean), 0.8 for the rest
  const n = 100, F = SPLAT_FLOATS, p = new Float32Array(n * F), m = new Float32Array(n * MOMENT_FLOATS);
  for (let i = 0; i < n; i++) {
    p.set([i * 0.1, 0, 0, 2, Math.log(0.05), Math.log(0.05), Math.log(0.05), 0, 1, 0, 0, 0, 0.1, 0.2, 0.3, 0], i * F);
    m[i * MOMENT_FLOATS] = 10; m[i * MOMENT_FLOATS + 1] = 10; m[i * MOMENT_FLOATS + 2] = i < 50 ? 1.2 : 0.8;
  }
  const mask = Uint8Array.from({ length: n }, (_, i) => i < 50 ? 1 : 0);
  const pass = (repairBoost) => {
    const q = Float32Array.from(p), oct = new LooseOctree({ maxLeafSize: 64 });
    oct.sceneRadius = 1; oct.leaves = [{}];  // reconcile only reads these; radius 1: the splats are large (split in two)
    return oct.reconcileAndGrow(q, n, m, null, { maxSplats: 1000, enableFloaters: false, enableMerge: false, repairMask: mask, repairBoost }).stats;
  };
  const a = pass(0), b = pass(0.5);
  assert(a.split + a.cloned === 0 && b.split + b.cloned === 50, `splits: no boost ${a.split + a.cloned}, boost 0.5 ${b.split + b.cloned}`);
  // relocation: 2 rows of equal opacity, row 0 masked with boost 3: drawn ~4 x as often
  let hits = 0;
  for (let t = 0; t < 400; t++) {
    const out = { n: 2, cap: 8, buf: new Float32Array(8 * F), src: Int32Array.from([0, 1, 0, 0, 0, 0, 0, 0]),
                  row(k) { return this.buf.subarray(k * F, (k + 1) * F); }, _grow() {} };
    out.buf[3] = logit(0.5); out.buf[F + 3] = logit(0.5);
    relocate(out, 1, null, Uint8Array.from([1, 0]), 3);
    hits += out.buf[3] < logit(0.5) - 1e-6 ? 1 : 0;  // row 0 drawn: its opacity went down
  }
  assert(Math.abs(hits - 320) < 4 * Math.sqrt(400 * 0.8 * 0.2), `row 0 drawn ${hits} / 400, expected ~320`);
});
