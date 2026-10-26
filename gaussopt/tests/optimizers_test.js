// The stochastic optimizers' CPU steps (DESIGN.md §5.4): monte_carlo's relocation (octree.js relocate,
// 3DGS-MCMC) and sim_anneal's Metropolis test at the rebalances (anneal.js).
import { LooseOctree, relocate, relocatedOpacityScale, MOMENT_FLOATS } from '../src/octree.js';
import { annealStep, remapAnnealState } from '../src/anneal.js';
import { SPLAT_FLOATS } from '../src/colmap.js';
import { sigmoid, logit } from '../src/math.js';
import { assert, assertClose, rng } from './util.js';

// sum over N copies along a line through the center: 1 - (1 - o' g(x / s'))^N, g = exp(-x^2 / 2)
const composite = (o, s, N, x) => 1 - Math.pow(1 - o * Math.exp(-0.5 * (x / s) ** 2), N);
const integral = (f) => { let a = 0; for (let x = -20; x <= 20; x += 1e-3) a += f(x) * 1e-3; return a; };

Deno.test('relocate: N copies keep the opacity at the center and the integrated opacity (3DGS-MCMC eq. 13-14)', () => {
  for (const [o, N] of [[0.6, 2], [0.6, 5], [0.95, 3], [0.1, 4], [0.3, 1]]) {
    const [on, ds] = relocatedOpacityScale(o, N);
    assertClose(composite(on, ds, N, 0), o, 1e-9, `center o=${o} N=${N}`);
    assertClose(integral(x => composite(on, ds, N, x)), integral(x => o * Math.exp(-0.5 * x * x)), 1e-6, `integral o=${o} N=${N}`);
  }
});

// a KeptSplats-like `out` (octree.js) of rows with opacities `ops`, unit scales
function rows(ops) {
  const n = ops.length, cap = n + 8, buf = new Float32Array(cap * SPLAT_FLOATS), src = new Int32Array(cap);
  ops.forEach((o, k) => { buf[k * SPLAT_FLOATS] = k; buf[k * SPLAT_FLOATS + 3] = o > 0 ? logit(o) : -Infinity; src[k] = k; });  // logit clamps 0 to 0.001: drawable
  return {
    n, cap, buf, src,
    row(k) { return this.buf.subarray(k * SPLAT_FLOATS, (k + 1) * SPLAT_FLOATS); },
    _grow(c) { const b = new Float32Array(c * SPLAT_FLOATS), s = new Int32Array(c); b.set(this.buf); s.set(this.src); this.buf = b; this.src = s; this.cap = c; },
  };
}

Deno.test('relocate: the dead go onto live rows, drawn by opacity or by error, as copies that keep the look', () => {
  // by opacity: row 1 is the only one with any
  const out = rows([0, 0.8, 0]);
  assert(relocate(out, 3) === 3 && out.n === 6, `${out.n} rows`);
  const [on, ds] = relocatedOpacityScale(0.8, 4);
  for (const k of [1, 3, 4, 5]) {
    const r = out.row(k);
    assert(r[0] === 1, `row ${k} is not a copy of row 1 (x ${r[0]})`);
    assertClose([sigmoid(r[3]), Math.exp(r[4])], [on, ds], 1e-5, `row ${k}`);
    if (k > 2) assert(out.src[k] === -1, `copy ${k} src ${out.src[k]}`);
  }
  // by error: row 2 holds all the error mass
  const out2 = rows([0.5, 0.5, 0.5]), moments = new Float32Array(3 * MOMENT_FLOATS);
  moments[2 * MOMENT_FLOATS + 2] = 1;
  relocate(out2, 2, moments);
  assert(out2.n === 5 && out2.row(3)[0] === 2 && out2.row(4)[0] === 2, 'not onto the high-error row');
  assert(relocate(rows([0.5, 0.5]), 2, new Float32Array(2 * MOMENT_FLOATS)) === 0, 'moved with no error anywhere');
});

Deno.test('relocate as shape: each live row drawn splits into two halves within its shape (brush), once while rows remain', () => {
  const out = rows([0.8, 0.6]);
  for (let k = 0; k < 2; k++) out.buf[k * SPLAT_FLOATS + 8] = 1;  // identity rotations
  assert(relocate(out, 2, null, null, 0, 'shape') === 2 && out.n === 4, `${out.n} rows`);
  // the halves of rows 0 (x 0, o 0.8) and 1 (x 1, o 0.6): rows 2 and 3 in the order drawn, told apart by their opacity
  for (const [k, o] of [[0, 0.8], [1, 0.6]]) {
    const oh = 1 - Math.sqrt(1 - o), p = out.row(k), c = out.row([2, 3].find(r => Math.abs(sigmoid(out.row(r)[3]) - oh) < 1e-4));
    for (const r of [p, c]) assertClose([sigmoid(r[3]), Math.exp(r[4]), Math.exp(r[5]), Math.exp(r[6])], [oh, Math.SQRT1_2, Math.SQRT1_2, Math.SQRT1_2], 1e-5, `row ${k}'s half`);
    assertClose([(p[0] + c[0]) / 2, (p[1] + c[1]) / 2, (p[2] + c[2]) / 2], [k, 0, 0], 1e-5, `row ${k}'s halves off its center`);
    assert(Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]) > 1e-3, `row ${k}: the halves are copies in place`);
  }
  assert(out.src[2] === -1 && out.src[3] === -1, 'new halves inherit a src');
});

// count splats at x = i, unit scales, identity rotations; moments W = 1, R = e[i]
function splats(count) {
  const p = new Float32Array(count * SPLAT_FLOATS);
  for (let i = 0; i < count; i++) { p[i * SPLAT_FLOATS] = i; p[i * SPLAT_FLOATS + 8] = 1; }
  return p;
}
function moments(e) {
  const m = new Float32Array(e.length * MOMENT_FLOATS);
  e.forEach((r, i) => { m[i * MOMENT_FLOATS] = 1; m[i * MOMENT_FLOATS + 2] = r; });
  return m;
}

Deno.test('anneal: a jump is kept when its error dropped more than the others\', else undone (T ~ 0)', () => {
  const n = 100, p = splats(n);
  // window 1: every energy 1, no jumps before; a jump for each splat (share 1)
  const r1 = annealStep(p, n, moments(new Array(n).fill(1)), null, { temp: 0.1, share: 1 });
  assert(r1.proposed === n && r1.accepted === 0 && r1.rejected === 0, JSON.stringify(r1));
  // only splats 0..9 keep their jump; window 2: every error 10% down (the training), 0..4 20% down, 5..9 5% down
  const state = { jump: r1.state.jump.map((v, k) => (k / 3 | 0) < 10 ? v : 0), e: r1.state.e };
  const moved = Float32Array.from(p);
  const e2 = Array.from({ length: n }, (_, i) => i < 5 ? 0.8 : i < 10 ? 0.95 : 0.9);
  const r2 = annealStep(p, n, moments(e2), state, { temp: 1e-9, share: 0 });
  assert(r2.accepted === 5 && r2.rejected === 5 && r2.proposed === 0, JSON.stringify({ ...r2, state: 0 }));
  for (let i = 0; i < 10; i++) {
    const back = [0, 1, 2].map(c => moved[i * SPLAT_FLOATS + c] - state.jump[3 * i + c]);
    const now = [0, 1, 2].map(c => p[i * SPLAT_FLOATS + c]);
    assertClose(now, i < 5 ? [0, 1, 2].map(c => moved[i * SPLAT_FLOATS + c]) : back, 1e-6, `splat ${i}`);
  }
});

Deno.test('anneal: worse jumps kept with probability exp(-de / T); unseen ones undone; the state follows the splats', () => {
  const n = 2000, p = splats(n);
  const r1 = annealStep(p, n, moments(new Array(n).fill(1)), null, { temp: 0.1, share: 1 });
  // the jumped half (even i) 0.1 worse than the trend (0): kept with probability exp(-0.1 / 0.1) = 0.37
  const state = { jump: r1.state.jump.map((v, k) => (k / 3 | 0) % 2 === 0 ? v : 0), e: r1.state.e };
  const e2 = Array.from({ length: n }, (_, i) => i % 2 === 0 ? 1.1 : 1);
  let seed = 7;
  const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const r2 = annealStep(p, n, moments(e2), state, { temp: 0.1, share: 0, rand });
  // e is normalized by its mean (1.05): de = 1.1 / 1.05 - 1 - (1 / 1.05 - 1) = 0.0952
  const want = Math.exp(-0.1 / 1.05 / 0.1) * (n / 2);
  assert(Math.abs(r2.accepted - want) < 4 * Math.sqrt(want), `accepted ${r2.accepted}, expected ~${want.toFixed(0)}`);
  const unseen = annealStep(p, 2, new Float32Array(2 * MOMENT_FLOATS), r1.state, { temp: 1, share: 0 });
  assert(unseen.accepted === 0, 'no splat seen: nothing to judge');
  const m = remapAnnealState(r1.state, Int32Array.from([5, -1, 0]));
  assertClose([m.e[0], m.e[1], ...m.jump.subarray(0, 6)], [r1.state.e[5], 0, ...r1.state.jump.subarray(15, 18), 0, 0, 0], 0, 'remap');
});

// 200 splats at the cap: 50 dead (opacity 0.0025), 100 with much error (they would split), 50 quiet
function capped(seeds = 0) {
  const r = rng(5), n = 200, params = new Float32Array(n * SPLAT_FLOATS), moments = new Float32Array(n * MOMENT_FLOATS);
  for (let i = 0; i < n; i++) {
    params.set([r() * 4 - 2, r() * 4 - 2, r() * 4 - 2, i < 50 ? -6 : 2, Math.log(0.05), Math.log(0.05), Math.log(0.05), 0, 1, 0, 0, 0, 0.1, 0.2, 0.3, 0],
               i * SPLAT_FLOATS);
    const m = i * MOMENT_FLOATS;
    moments[m] = 10; moments[m + 1] = 10; moments[m + 2] = i >= 50 && i < 150 ? 1 : 0.01;
  }
  const oct = new LooseOctree({ maxLeafSize: 64 }).build(params, n, null);
  const seedSplats = seeds ? params.slice(150 * SPLAT_FLOATS, (150 + seeds) * SPLAT_FLOATS) : null;
  return oct.reconcileAndGrow(params, n, moments, null, { maxSplats: n, enableFloaters: false, enableMerge: false,
                                                          optMethod: 'monte_carlo', seedSplats });
}

Deno.test('monte_carlo at the cap: the dead are relocated, not lost, and the splits never push the count over it', () => {
  const a = capped();
  assert(a.count === 200 && a.stats.teleported === 50 && a.stats.pruned === 0, `count ${a.count}, ${JSON.stringify(a.stats)}`);
  // 30 seeds: they take 30 of the dead's places, 20 dead are copies
  const b = capped(30);
  assert(b.count === 200 && b.stats.seeded === 30 && b.stats.teleported === 20, `count ${b.count}, ${JSON.stringify(b.stats)}`);
});

Deno.test('monte_carlo: a splat out of `bounds` is dead and relocated in it (the noise has no bound); the trainer\'s box', async () => {
  const run = (bounds) => {
    const r = rng(9), n = 200, params = new Float32Array(n * SPLAT_FLOATS), moments = new Float32Array(n * MOMENT_FLOATS);
    for (let i = 0; i < n; i++) {  // opaque and quiet: only the box can kill one; 20 of them 50 units out
      params.set([(i < 20 ? 50 : 0) + r() * 4 - 2, r() * 4 - 2, r() * 4 - 2, 2, Math.log(0.05), Math.log(0.05), Math.log(0.05), 0, 1, 0, 0, 0, 0.1, 0.2, 0.3, 0],
                 i * SPLAT_FLOATS);
      moments[i * MOMENT_FLOATS] = 10; moments[i * MOMENT_FLOATS + 1] = 10; moments[i * MOMENT_FLOATS + 2] = 0.01;
    }
    const out = new LooseOctree({ maxLeafSize: 64 }).build(params, n, null).reconcileAndGrow(params, n, moments, null,
      { maxSplats: n, enableFloaters: false, enableMerge: false, enableSplit: false, optMethod: 'monte_carlo', bounds });
    let far = 0;
    for (let i = 0; i < out.count; i++) if (Math.abs(out.params[i * SPLAT_FLOATS]) > 3) far++;
    return { out, far };
  };
  const a = run({ center: [0, 0, 0], halfSize: 3 });
  assert(a.out.count === 200 && a.out.stats.teleported === 20 && a.far === 0, `count ${a.out.count}, far ${a.far}, ${JSON.stringify(a.out.stats)}`);
  const b = run(null);
  assert(b.out.count === 200 && b.out.stats.teleported === 0 && b.far === 20, `no bounds: far ${b.far}, ${JSON.stringify(b.out.stats)}`);
  // the trainer's box: 1.25 times the octree's root at the start
  const { GaussOptTrainer } = await import('../src/trainer.js');
  const p = new Float32Array(2 * SPLAT_FLOATS);
  p.set([-1, 0, 0], 0); p.set([3, 0, 0], SPLAT_FLOATS);
  const oct = new LooseOctree().build(p, 2, null);
  const t = new GaussOptTrainer(null, oct, { count: 2, params: p }, [], { sa_temp: 0 });
  assertClose([...t.sceneBox.center, t.sceneBox.halfSize], [1, 0, 0, 1.25 * oct.root.halfSize], 1e-6, 'sceneBox');
  assert(Math.abs(oct.root.halfSize - 0.505 * 4) < 1e-6, `root half ${oct.root.halfSize}`);
});

// E7, the cap lease: the budget's schedule (trainer.js splatBudget) and the consolidation's trim (octree.js)
import { splatBudget, visibilityScore, remapVisibility, endPhase, consolidating, rebalanceDue } from '../src/trainer.js';

Deno.test('cap lease: max_splats x (1 + lease) until the consolidation, then linearly back, then max_splats', () => {
  const P = { max_splats: 1000, cap_lease: 0.5, train_steps: 10000, consolidate_frac: 0.2 };
  // consolidation 8000..10000: down over 8000..9500, at 1000 from 9500
  assertClose([0, 8000, 8750, 9500, 9999, 20000].map(s => splatBudget(P, s)), [1500, 1500, 1250, 1000, 1000, 1000], 0, 'schedule');
  assert(splatBudget({ ...P, train_steps: 0 }, 0) === 1000 && splatBudget({ ...P, cap_lease: 0 }, 0) === 1000, 'no lease without a planned length');
  // the end phase (polish) needs a planned length only, the consolidation also a lease
  assert(!endPhase(P, 8000) && endPhase(P, 8001) && !endPhase({ ...P, train_steps: 0 }, 9000), 'end phase');
  assert(consolidating(P, 9000) && !consolidating({ ...P, cap_lease: 0 }, 9000) && endPhase({ ...P, cap_lease: 0 }, 9000), 'consolidating');
  // the rebalances: every 400 at the cap, every 100 while the budget falls, one at 9500 (the budget at max_splats)
  const R = { ...P, rebalance_every: 100, rebalance_every_cap: 400, opacity_reset: 0 };
  assert(!rebalanceDue(R, 7900, 7600, 1500) && rebalanceDue(R, 8000, 7600, 1500), 'at the cap: 400');
  assert(rebalanceDue(R, 8900, 8800, 1300) && !rebalanceDue(R, 8899, 8800, 1300), 'falling: 100');
  assert(rebalanceDue(R, 9500, 9450, 1000) && !rebalanceDue(R, 9600, 9500, 1000) && rebalanceDue(R, 9900, 9500, 1000), 'its end, then 400');
});

Deno.test('cap lease: over the budget the splats seen least go (or the lowest keepScore), Monte-Carlo relocates none', () => {
  // 200 splats, 50 dead; W = 10 but 1 for 50 of the live (marked in slot 15); build() reorders the splats,
  // so the moments and scores follow the marks
  for (const optMethod of ['adam', 'monte_carlo']) {
    const r = rng(9), n = 200, F = SPLAT_FLOATS, params = new Float32Array(n * F);
    for (let i = 0; i < n; i++)
      params.set([r() * 4 - 2, r() * 4 - 2, r() * 4 - 2, i < 50 ? -6 : 2, Math.log(0.05), Math.log(0.05), Math.log(0.05), 0, 1, 0, 0, 0, 0.1, 0.2, 0.3, i >= 150 ? 1 : i >= 50 && i < 80 ? 2 : 0], i * F);
    const run = (score) => {
      const p = Float32Array.from(params), oct = new LooseOctree({ maxLeafSize: 64 }).build(p, n, null);
      const moments = new Float32Array(n * MOMENT_FLOATS);
      for (let i = 0; i < n; i++) { moments[i * MOMENT_FLOATS] = p[i * F + 15] === 1 ? 1 : 10; moments[i * MOMENT_FLOATS + 1] = 10; moments[i * MOMENT_FLOATS + 2] = 0.01; }
      const keepScore = score ? Float32Array.from({ length: n }, (_, i) => p[i * F + 15] === 2 ? 1 : 5) : null;
      const a = oct.reconcileAndGrow(p, n, moments, null, { maxSplats: 120, enableFloaters: false, enableMerge: false, optMethod, keepScore });
      const marks = [0, 0, 0];
      for (let k = 0; k < a.count; k++) if (a.params[k * F + 3] > 0) marks[a.params[k * F + 15]]++;
      return { a, marks };
    };
    // 150 live, 30 over: 30 of the 50 seen least
    const { a, marks } = run(false);
    assert(a.count === 120 && a.stats.trimmed === 30 && a.stats.teleported === 0, `${optMethod}: count ${a.count}, ${JSON.stringify(a.stats)}`);
    assert(marks[0] === 70 && marks[2] === 30 && marks[1] === 20, `${optMethod}: kept by mark ${marks}`);
    // by keepScore: the 30 marked 2 go
    const b = run(true);
    assert(b.a.count === 120 && b.marks[2] === 0 && b.marks[1] === 50, `${optMethod} (keepScore): kept by mark ${b.marks}`);
  }
});

Deno.test('cap lease: the keep score is W per window over the windows a splat lived (decay 0.7), new splats start afresh', () => {
  const m = (ws) => { const a = new Float32Array(ws.length * MOMENT_FLOATS); ws.forEach((w, i) => { a[i * MOMENT_FLOATS] = w; }); return a; };
  let v = visibilityScore(null, m([4, 0]), 2);
  v = visibilityScore(v, m([0, 0]), 2);  // splat 0 unseen in window 2 (a video's window): (0.7 x 4 + 0) / 1.7
  assertClose(Array.from(v.score), [2.8 / 1.7, 0], 1e-6, 'two windows');
  v = remapVisibility(v, Int32Array.from([-1, 0]));  // splat 0 is now 1, a new splat 0
  v = visibilityScore(v, m([3, 1]), 2);
  assertClose(Array.from(v.score), [3, (0.7 * 2.8 + 1) / (0.7 * 1.7 + 1)], 1e-6, 'after the rebalance');
});
