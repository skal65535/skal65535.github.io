import { umeyama, robustUmeyama } from '../src/sfm/align.js';
import { apply as applySim3 } from '../src/sfm/sim3.js';
import { rotationAngle } from '../src/sfm/linalg.js';
import { gaussian } from '../src/sfm/synth.js';
import { assert, assertClose, rng, randomRotation } from './util.js';

Deno.test('umeyama recovers a random Sim(3) exactly without noise', () => {
  const r = rng(8);
  for (let trial = 0; trial < 50; trial++) {
    const sim = { s: 0.1 + 10 * r(), R: randomRotation(r), t: [gaussian(r) * 5, gaussian(r) * 5, gaussian(r) * 5] };
    const src = Array.from({ length: 3 + Math.floor(r() * 50) }, () => [gaussian(r), gaussian(r), gaussian(r)]);
    const est = umeyama(src, src.map(p => applySim3(sim, p)));
    assertClose(est.s, sim.s, 1e-9 * sim.s, 'scale');
    assert(rotationAngle(est.R, sim.R) < 1e-7, 'rotation');
    assertClose(est.t, sim.t, 1e-7, 'translation');
  }
});

Deno.test('umeyama is a least-squares fit under noise, and rigid when asked', () => {
  const r = rng(9);
  const sim = { s: 2, R: randomRotation(r), t: [1, 2, 3] };
  const src = Array.from({ length: 2000 }, () => [gaussian(r) * 3, gaussian(r) * 3, gaussian(r)]);
  const dst = src.map(p => applySim3(sim, p).map(x => x + 0.01 * gaussian(r)));
  const est = umeyama(src, dst);
  assertClose(est.s, 2, 2e-3, 'scale');
  assert(rotationAngle(est.R, sim.R) < 1e-3, 'rotation');
  const rigid = umeyama(src, src.map(p => applySim3({ ...sim, s: 1 }, p)), { withScale: false });
  assertClose(rigid.s, 1, 0, 'rigid scale');
  assert(rotationAngle(rigid.R, sim.R) < 1e-7, 'rigid rotation');
});

Deno.test('robustUmeyama: the fit without the pairs far off (3x the median residual), as plain umeyama on the rest', () => {
  const r = rng(10);
  const sim = { s: 1.5, R: randomRotation(r), t: [0.5, -1, 2] };
  const src = Array.from({ length: 30 }, () => [gaussian(r), gaussian(r), gaussian(r)]);
  const dst = src.map(p => applySim3(sim, p).map(x => x + 1e-3 * gaussian(r)));
  for (const k of [3, 17]) dst[k] = dst[k].map(x => x + 2);  // two cameras moved
  const plain = umeyama(src, dst), est = robustUmeyama(src, dst);
  const keep = src.map((_, k) => k).filter(k => k !== 3 && k !== 17), rest = umeyama(keep.map(k => src[k]), keep.map(k => dst[k]));
  assert(Math.abs(plain.s - sim.s) > 1e-2 || rotationAngle(plain.R, sim.R) > 1e-2 || Math.hypot(...plain.t.map((x, i) => x - sim.t[i])) > 0.1, 'the outliers move the plain fit');
  assertClose([est.s, ...est.t], [rest.s, ...rest.t], 1e-12, 'the fit without them');
  assert(rotationAngle(est.R, rest.R) < 1e-9, 'rotation');
});
