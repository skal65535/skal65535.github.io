// rebalanceDue (trainer.js): the rebalance schedule, every K steps, every K_cap once the splat count
// is at the cap, and at each opacity reset; the stipple seeds made SEED_LEAD steps before each (seedPrepDue).
import { rebalanceDue, opacityResetDue, haltBefore, seedPrepDue, SEED_LEAD } from '../src/trainer.js';
import { assert } from './util.js';

// Steps 1..steps that rebalance; count(step) = splat count before that step
function schedule(P, steps, count) {
  const out = [];
  for (let s = 1, last = 0; s <= steps; s++) {
    if (rebalanceDue(P, s, last, count(s))) { out.push(s); last = s; }
  }
  return out;
}
const P = { rebalance_every: 100, rebalance_every_cap: 400, max_splats: 300000, opacity_reset: 3000 };
const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

Deno.test('rebalance: every K steps below the cap, every K_cap at it, and at each opacity reset', () => {
  // at the cap from step 2001 on, except 3001..3300 (the prune after the reset)
  const count = (s) => (s > 2000 && !(s > 3000 && s <= 3300)) ? 299000 : 200000;
  const got = schedule(P, 4500, count);
  const want = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
                1800, 1900, 2000, 2400, 2800, 3000, 3100, 3200, 3300, 3700, 4100, 4500];
  assert(same(got, want), `schedule ${got}`);
});

Deno.test('rebalance: K_cap = 0 is every K steps (multiples of K when K divides the reset period)', () => {
  const got = schedule({ ...P, rebalance_every_cap: 0 }, 6000, () => 300000);
  assert(same(got, Array.from({ length: 60 }, (_, i) => 100 * (i + 1))), `schedule ${got}`);
  // K = 70: the resets add their own rebalances (3000, 6000), the next one 70 steps later
  const k70 = schedule({ ...P, rebalance_every: 70, rebalance_every_cap: 0 }, 3200, () => 300000);
  assert(k70.includes(3000) && k70.includes(3070) && k70.includes(2940) && !k70.includes(3010), `K = 70: ${k70.slice(40)}`);
});

Deno.test('opacity reset: at the multiples of its period up to opacity_reset_until and half the planned length, never with monte_carlo', () => {
  // the rebalances of a 30000-step Adam run, every 100 steps: the steps whose rebalance resets the opacities
  const resets = (Q) => Array.from({ length: 300 }, (_, i) => 100 * (i + 1)).filter(s => opacityResetDue(Q, s, s - 100));
  const A = { ...P, opt_method: 'adam', opacity_reset_until: 15000, train_steps: 0 };
  assert(same(resets(A), [3000, 6000, 9000, 12000, 15000]), `open-ended: ${resets(A)}`);
  assert(same(resets({ ...A, train_steps: 10000 }), [3000]), `10000 planned: ${resets({ ...A, train_steps: 10000 })}`);
  assert(resets({ ...A, opt_method: 'monte_carlo' }).length === 0 && resets({ ...A, opacity_reset: 0 }).length === 0, 'off');
  // a rebalance window holding a multiple resets once, at its end
  assert(opacityResetDue(A, 3050, 2950) && !opacityResetDue(A, 3150, 3050), 'window over 3000');
});

// The app's training loop (app.js _frame): frames of `n` steps, each stopped early by haltBefore, then
// stop_after / val_every as after the loop there. The steps validated, those that reset, and where it paused
function trainLoop(Q, steps, n) {
  const vals = [], resets = [];
  let step = 0, lastVal = 0, paused = null;
  while (step < steps && paused === null) {
    let valRequest = false;
    for (let s = 0; s < n; s++) {
      const halt = haltBefore(Q, step, lastVal, 0);
      if (halt === 'val') valRequest = true;
      if (halt) break;
      step++;
      if (rebalanceDue(Q, step, step - 1, 0) && opacityResetDue(Q, step, step - 1)) resets.push(step);
    }
    if (Q.stop_after > 0 && step >= Q.stop_after) { paused = step; valRequest = true; }
    if (Q.val_every > 0 && step - lastVal >= Q.val_every) valRequest = true;
    if (valRequest) { vals.push(step); lastVal = step; }
  }
  return { vals, resets, paused };
}

Deno.test('validation right before each opacity reset, not just after it; stop_after pauses at its step', () => {
  const Q = { ...P, opt_method: 'adam', opacity_reset_until: 15000, train_steps: 30000, val_every: 500, stop_after: 0 };
  const { vals, resets } = trainLoop(Q, 30000, 7);
  assert(same(resets, [3000, 6000, 9000, 12000, 15000]), `resets ${resets}`);
  for (const r of resets) {
    assert(vals.includes(r - 1), `no validation right before the reset at ${r}: ${vals.filter(v => v > r - 600 && v < r + 600)}`);
  }
  // Monte-Carlo never resets: validation every 500 steps (frames of 7: at the first frame past each)
  const mc = trainLoop({ ...Q, opt_method: 'monte_carlo' }, 3000, 7);
  assert(mc.resets.length === 0 && mc.vals.every((v, i) => i === 0 || v - mc.vals[i - 1] >= 500), `monte_carlo ${mc.vals}`);
  // stop_after: paused at its step exactly, not at the end of the frame
  const s = trainLoop({ ...Q, stop_after: 1000 }, 30000, 7);
  assert(s.paused === 1000, `paused at ${s.paused}`);
});

Deno.test('stipple seeds: made once per window, SEED_LEAD steps before its rebalance; an earlier one (a reset) seeds in the page', () => {
  const Q = { ...P, enable_rebalance: true, stipple_seed: true, stipple_count: 2000, opt_method: 'adam', opacity_reset_until: 15000, train_steps: 0 };
  const count = (s) => s > 2000 ? 299000 : 200000;
  const preps = [], used = [];
  for (let s = 1, last = 0, prep; s <= 4500; s++) {
    if (rebalanceDue(Q, s, last, count(s))) { used.push(prep > last ? s - prep : null); last = s; continue; }
    if (seedPrepDue(Q, s, last, count(s), prep)) { prep = s; preps.push(s); }
  }
  const reb = schedule(Q, 4500, count);
  assert(preps.length === reb.length, `${preps.length} seed jobs for ${reb.length} rebalances`);
  // every rebalance takes the seeds made SEED_LEAD steps before it (the reset at 3000 comes at a multiple of 100 too)
  assert(used.every(d => d === SEED_LEAD), `steps between the seeds and their rebalance: ${used}`);
  // no seeds without stipple or rebalance, nor when this window has some
  assert(!seedPrepDue({ ...Q, stipple_seed: false }, 84, 0, 0) && !seedPrepDue({ ...Q, enable_rebalance: false }, 84, 0, 0) &&
         seedPrepDue(Q, 84, 0, 0) && !seedPrepDue(Q, 85, 0, 0, 84), 'off / once a window');
});
