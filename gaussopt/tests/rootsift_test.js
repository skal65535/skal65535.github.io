// sift.js rootSift: each descriptor L1-normalized and square-rooted (x 512, bytes): unit L2 norm before the scale, the
// dot product the Hellinger kernel of the two histograms
import { rootSift } from '../src/sfm/sift.js';
import { SESSION_DEFAULTS, FEATURE_KEYS } from '../src/sfm/defaults.js';
import { assert, assertClose, rng } from './util.js';

Deno.test('rootSift: unit norm, the Hellinger kernel, a zero descriptor left at zero', () => {
  const r = rng(5), n = 50, d = new Uint8Array(128 * n);
  for (let i = 0; i < d.length; i++) d[i] = r() < 0.3 ? 0 : Math.floor(r() * 120);
  d.fill(0, 128 * 7, 128 * 8);  // descriptor 7: all zero
  const q = rootSift(d);
  for (let k = 0; k < n; k++) {
    if (k === 7) { assert(q.subarray(128 * k, 128 * k + 128).every(x => x === 0), 'zero descriptor'); continue; }
    let s = 0; for (let i = 0; i < 128; i++) s += (q[128 * k + i] / 512) ** 2;
    assertClose(s, 1, 0.02, `descriptor ${k}'s squared norm`);
  }
  // the dot product of two (/ 512^2) against sum sqrt(a_i b_i) / sqrt(sum a sum b)
  const dotq = (a, b) => { let s = 0; for (let i = 0; i < 128; i++) s += q[128 * a + i] * q[128 * b + i]; return s / 512 / 512; };
  const hel = (a, b) => { let s = 0, sa = 0, sb = 0; for (let i = 0; i < 128; i++) { s += Math.sqrt(d[128 * a + i] * d[128 * b + i]); sa += d[128 * a + i]; sb += d[128 * b + i]; } return s / Math.sqrt(sa * sb); };
  for (const [a, b] of [[0, 1], [2, 3], [4, 9]]) assertClose(dotq(a, b), hel(a, b), 0.02, `Hellinger ${a}, ${b}`);
  assert(SESSION_DEFAULTS.rootSift === false && FEATURE_KEYS.includes('rootSift'), 'rootSift: an option, off by default');
});
