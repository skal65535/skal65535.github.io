// Rule 4 (floater_conflict): trainer.js conflictCut's scores from the carried opacity gradient (H, H2, U, U2 a splat)
// and reconcileAndGrow removing the splats scored over its cut, those only.
import { LooseOctree, MOMENT_FLOATS } from '../src/octree.js';
import { reconcileAndGrow } from '../src/densify.js';
import { conflictCut } from '../src/trainer.js';
import { SPLAT_FLOATS } from '../src/colmap.js';
import { assert, rng } from './util.js';

// 100 splats, help H = 1 each: 0..89 helped by 10 views of 0.1 (H2 0.1), hurt U 0.1; 90..94 helped by one view
// (H2 1), hurt 0.1; 95..99 helped by 10 views, hurt U 5 (the views in conflict)
function grads() {
  const n = 100, G = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) G.set([1, i >= 90 && i < 95 ? 1 : 0.1, i >= 95 ? 5 : 0.1, 0], 4 * i);
  return { G, n };
}

Deno.test('conflictCut: hurt over help, help from one view, no help scored 0, too small a share off', () => {
  const { G, n } = grads();
  const over = (c) => [...c.score.keys()].filter(i => c.score[i] > c.cut);
  const hurt = conflictCut(G, n, 0.05, 'hurt');
  assert(over(hurt).join() === '95,96,97,98,99', `hurt: ${over(hurt)}`);
  const single = conflictCut(G, n, 0.05, 'single');
  assert(over(single).join() === '90,91,92,93,94', `single: ${over(single)}`);
  const G0 = G.slice(); G0[4 * 99] = 0;                        // splat 99 without help: score 0, whatever its hurt
  const c0 = conflictCut(G0, n, 0.04, 'hurt');
  assert(c0.score[99] === 0 && over(c0).join() === '95,96,97,98', `no help: ${c0.score[99]}, ${over(c0)}`);
  assert(conflictCut(G, n, 0.005) === null, 'a share under one splat: off');
  assert(conflictCut(G.subarray(0, 40), n, 0.05) === null, 'gradients of another count: off');
});

// the splats in a loose spread, no camera, every other floater rule quiet: only rule 4 removes
function grow(conflict) {
  const r = rng(5), { G, n } = grads();
  const params = new Float32Array(n * SPLAT_FLOATS), moments = new Float32Array(n * MOMENT_FLOATS);
  for (let i = 0; i < n; i++) {
    params.set([r() * 4 - 2, r() * 4 - 2, r() * 4 - 2, 2, Math.log(0.05), Math.log(0.05), Math.log(0.05), 0, 1, 0, 0, 0, 0.1, 0.2, 0.3, 0],
               i * SPLAT_FLOATS);
    moments.set([10, 10, 0.01], i * MOMENT_FLOATS);
  }
  const oct = new LooseOctree({ maxLeafSize: 64 }).build(params, n, null);
  return reconcileAndGrow(oct, params, n, moments, null,
                          { maxSplats: 10000, enableSplit: false, enableMerge: false, decayOccluders: false,
                            conflict: conflict && conflictCut(G, n, 0.05, conflict) });
}

Deno.test('rule 4: the splats over the cut removed, none without it', () => {
  const off = grow(null);
  assert(off.stats.floaters === 0 && off.stats.floaterRules[3] === 0 && off.count === 100, `off: ${JSON.stringify(off.stats)}`);
  for (const by of ['hurt', 'single']) {
    const on = grow(by);
    assert(on.stats.floaters === 5 && on.stats.floaterRules[3] === 5 && on.count === 95, `${by}: ${JSON.stringify(on.stats)}, ${on.count}`);
  }
});
