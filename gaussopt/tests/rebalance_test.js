// rebalanceDue (trainer.js): the rebalance schedule, every K steps, every K_cap once the splat count
// is at the cap, and at each opacity reset.
import { rebalanceDue } from '../src/trainer.js';
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
