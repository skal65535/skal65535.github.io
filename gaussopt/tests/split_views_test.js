// reconcileAndGrow's split gate (octree.js, splitMinViews / visitsPerView): with N_eff = W^2 / S counting
// view visits, a splat splits only when N_eff / visitsPerView reaches splitMinViews.
import { LooseOctree, MOMENT_FLOATS } from '../src/octree.js';
import { SPLAT_FLOATS } from '../src/colmap.js';
import { assert, rng } from './util.js';

// 200 splats: 100 with little error, then 50 with much error seen from 1 view (N_eff 10 over 10 visits
// per view) and 50 seen from 3 (N_eff 30)
function scene() {
  const r = rng(3), n = 200, params = new Float32Array(n * SPLAT_FLOATS), moments = new Float32Array(n * MOMENT_FLOATS);
  for (let i = 0; i < n; i++) {
    params.set([r() * 4 - 2, r() * 4 - 2, r() * 4 - 2, 2, Math.log(0.05), Math.log(0.05), Math.log(0.05), 0, 1, 0, 0, 0, 0.1, 0.2, 0.3, 0],
               i * SPLAT_FLOATS);
    const m = i * MOMENT_FLOATS, W = 10, nEff = i < 100 ? 20 : i < 150 ? 10 : 30;
    moments[m] = W; moments[m + 1] = W * W / nEff; moments[m + 2] = i < 100 ? 0.01 : 1;
  }
  return { params, moments, n };
}

function grow(opts) {
  const { params, moments, n } = scene();
  const oct = new LooseOctree({ maxLeafSize: 64 }).build(params, n, null);
  return oct.reconcileAndGrow(params, n, moments, null,
                              { maxSplats: 10000, enableFloaters: false, enableMerge: false, visitsPerView: 10, ...opts }).stats;
}

Deno.test('split gate: off, every high-error splat grows; at 2 views, only the ones seen from 3', () => {
  const off = grow({});
  const grown = (s) => s.split + s.cloned;
  assert(grown(off) === 100, `ungated: ${grown(off)} grown, expected 100`);
  const two = grow({ splitMinViews: 2 });
  assert(grown(two) === 50 && two.splitTooFewViews === 50, `gate 2: ${grown(two)} grown, ${two.splitTooFewViews} held`);
  const four = grow({ splitMinViews: 4 });
  assert(grown(four) === 0 && four.splitTooFewViews === 100, `gate 4: ${grown(four)} grown, ${four.splitTooFewViews} held`);
  // visits per view 1 (a video: N_eff counts views): N_eff 10 and 30 both pass a gate of 4
  const video = grow({ splitMinViews: 4, visitsPerView: 1 });
  assert(grown(video) === 100, `gate 4, 1 visit per view: ${grown(video)} grown`);
});
