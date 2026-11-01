// The gallery's fly-by on the path (demo.js PathFlight): unsmoothed, through every camera's center at its
// orientation, and still at the center of a turn in place; smoothed, the cameras' jitter out.
import { PathFlight } from '../src/demo.js';
import { normalize } from '../src/math.js';
import { assert, lookAt } from './util.js';

// A camera at c looking along f (up -y)
const cam = (c, f) => ({ center: c, view: lookAt(c, normalize(f)).view });
const maxDiff = (a, b) => Math.max(...Array.from(a, (x, i) => Math.abs(x - b[i])));

// a straight flight with a 90° turn in place at x = 2, then on
const PATH = [cam([0, 0, 0], [1, 0, 0]), cam([1, 0, 0], [1, 0, 0.1]), cam([2, 0, 0], [1, 0, 0]), cam([2, 0, 0], [0, 0, 1]),
              cam([2, 0, 1], [0, 0, 1]), cam([2.2, 0, 2], [0.3, 0, 1])];

Deno.test('fly-by on the path: each camera\'s view at its knot time', () => {
  const f = new PathFlight();
  f.blend = 0; f.smooth = 0;
  f.start(PATH, PATH[0], 5);
  for (let k = 0; k < PATH.length; k++) {
    f.t = f.T[k]; f.dir = 1;
    f.update(0);
    const d = maxDiff(f.view(), PATH[k].view);
    assert(d < 1e-5, `camera ${k}: view off by ${d}`);
  }
});

Deno.test('fly-by on the path: still through a turn in place, no overshoot past the cameras', () => {
  const f = new PathFlight();
  f.blend = 0; f.smooth = 0;
  f.start(PATH, PATH[0], 5);
  assert(f.T[3] - f.T[2] > 0.5, `the turn's step lasts ${f.T[3] - f.T[2]} s`);  // the turn takes time though it does not move
  for (let i = 0; i <= 20; i++) {
    f.t = f.T[2] + (f.T[3] - f.T[2]) * i / 20;
    f.update(0);
    assert(Math.hypot(f.pos[0] - 2, f.pos[1], f.pos[2]) < 1e-9, `turn at ${i}/20: at ${f.pos}`);
  }
  let maxX = -Infinity;  // the straight part, x from 0 to 2: never past x = 2
  for (let t = 0; t <= f.T[2]; t += 0.01) { f.t = t; f.update(0); maxX = Math.max(maxX, f.pos[0]); }
  assert(maxX <= 2 + 1e-9, `overshoot to x = ${maxX}`);
});

Deno.test('fly-by on the path: from the pose on screen into the path, then on it, back at the end', () => {
  const f = new PathFlight(), from = cam([0, 3, 0], [0, 0, 1]);
  f.smooth = 0;
  f.start(PATH, from, 5);
  assert(maxDiff(f.view(), from.view) < 1e-5, 'starts at the pose on screen');
  for (let i = 0; i < 200; i++) f.update(0.05);  // 10 s: blended in
  const t = f.t, { c } = f._sample(t);
  assert(Math.hypot(...f.pos.map((x, i) => x - c[i])) < 1e-9, 'on the path after the blend');
  for (let i = 0; i < 20 * 60 * 10; i++) f.update(0.05);
  assert(f.t >= 0 && f.t <= f.duration, `t ${f.t} in [0, ${f.duration}]`);
});

Deno.test('fly-by on the path: smoothing takes the cameras\' jitter out, the flight stays near them', () => {
  // a straight flight whose cameras jitter (as an SfM's keyframes do): 0.03 sideways, 2 deg in yaw
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5;
  const path = Array.from({ length: 60 }, (_, k) => cam([0.1 * k, 0, 0.06 * rnd()], [1, 0, 0.07 * rnd()]));
  const run = (smooth) => {
    const f = new PathFlight();
    f.blend = 0; f.smooth = smooth;
    f.start(path, path[0], 5);
    let jerk = 0, turn = 0, off = 0, p0, p1, f0, f1;
    for (let t = 0; t < f.duration; t += 1 / 60) {  // the second differences of position and view direction, at 60 Hz
      f.t = t; f.dir = 1; f.update(0);
      const v = f.view(), p = [...f.pos], fw = [v[2], v[6], v[10]];
      if (p0) {
        jerk += Math.hypot(...p.map((x, i) => x - 2 * p1[i] + p0[i]));
        turn += Math.hypot(...fw.map((x, i) => x - 2 * f1[i] + f0[i]));
      }
      [p0, p1, f0, f1] = [p1, p, f1, fw];
      off = Math.max(off, Math.abs(p[2]));
    }
    return { jerk, turn, off };
  };
  const exact = run(0), smooth = run(0.5), tag = `exact ${JSON.stringify(exact)}, smoothed ${JSON.stringify(smooth)}`;
  assert(smooth.jerk < 0.2 * exact.jerk && smooth.turn < 0.2 * exact.turn, tag);
  assert(smooth.off < 0.03, tag);  // within the jitter of the line
});
