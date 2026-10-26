// math.js estimateWorldUp: the up that keeps the photos' horizons level, not -(mean camera +Y), which
// leans with the mean pitch; photos rotated by 90° to the set's orientation (photos.js) read by their Y
import { estimateWorldUp, uprightView, viewFromAxes, viewTransform, normalize, cross, dot } from '../src/math.js';
import { makeRng } from '../src/math.js';
import { assert } from './util.js';

const deg = (a, b) => Math.acos(Math.min(1, dot(a, b))) * 180 / Math.PI;
// A camera (R rows: x right, y down, z forward) in a world with `up`, at yaw, pitch (> 0: up) and roll, degrees;
// rotated: the photo turned 90° clockwise (worker.js photoFeature): x' = -y, y' = x
function camera(up, yaw, pitch, roll, rotated = false) {
  const r = (d) => d * Math.PI / 180, U = normalize(up), e1 = normalize(cross(U, Math.abs(U[0]) > 0.9 ? [0, 0, 1] : [1, 0, 0])), e2 = cross(U, e1);
  const h = [0, 1, 2].map(k => Math.cos(r(yaw)) * e1[k] + Math.sin(r(yaw)) * e2[k]);  // horizontal heading
  const f = normalize([0, 1, 2].map(k => Math.cos(r(pitch)) * h[k] + Math.sin(r(pitch)) * U[k]));
  let x = normalize(cross(f, U)), y = cross(f, x);  // level: x horizontal, y down
  const c = Math.cos(r(roll)), s = Math.sin(r(roll));
  [x, y] = [[0, 1, 2].map(k => c * x[k] + s * y[k]), [0, 1, 2].map(k => -s * x[k] + c * y[k])];
  if (rotated) [x, y] = [y.map(v => -v), x];
  return { R: [...x, ...y, ...f] };
}

const UP = normalize([0.2, -1, 0.3]);
const scenes = {
  // facing a room from one side, pitched down 10° on average: the old estimate leans ~10°
  'pitched down': (rng) => Array.from({ length: 30 }, () => camera(UP, 120 * rng() - 60, -10 + 8 * (rng() - 0.5), 4 * (rng() - 0.5))),
  'steep, salon-like (-15..-55°)': (rng) => Array.from({ length: 17 }, () => camera(UP, 140 * rng() - 70, -15 - 40 * rng(), 6 * (rng() - 0.5))),
  'a third rotated': (rng) => Array.from({ length: 24 }, (_, k) => camera(UP, 360 * rng(), -8 + 10 * (rng() - 0.5), 4 * (rng() - 0.5), k % 3 === 0)),
};
for (const [name, make] of Object.entries(scenes)) {
  Deno.test(`estimateWorldUp: ${name}: within 1.5° of the true up`, () => {
    const ims = make(makeRng(5)), up = estimateWorldUp(ims);
    assert(deg(up, UP) < 1.5, `${deg(up, UP).toFixed(2)}° off`);
  });
}

// every photo facing one way: up is not determined around their common horizontal axis; the roll is
// levelled and the rest is -(mean +Y)'s, which leans by the mean pitch (5.5° here)
Deno.test('estimateWorldUp: photos facing one way: their horizontal axis level, up leaning by the mean pitch', () => {
  const ims = Array.from({ length: 10 }, (_, k) => camera(UP, 0, -10 + k, 0));
  const up = estimateWorldUp(ims), x = ims[0].R.slice(0, 3);
  assert(Math.abs(dot(up, x)) < 1e-6 && Math.abs(deg(up, UP) - 5.5) < 0.1, `x.up ${dot(up, x)}, ${deg(up, UP).toFixed(2)}° off`);
});

// uprightView: a camera turned by k quarter turns (and rolled up to ±40°) is turned back by 4 - k,
// to the upright camera's view; its center stays put
Deno.test('uprightView: photos turned by 0, 1, 2, 3 quarter turns shown upright', () => {
  const rng = makeRng(9), C = [1, -2, 3];
  for (let n = 0; n < 40; n++) {
    const k = n % 4, R = camera(UP, 360 * rng(), 30 * (rng() - 0.5), 80 * (rng() - 0.5)).R;
    let x = R.slice(0, 3), y = R.slice(3, 6);
    const ref = viewFromAxes(x, y, R.slice(6, 9), C);
    for (let i = 0; i < k; i++) [x, y] = [y.map(v => -v), x];  // 90° clockwise, as camera(..., rotated)
    const { view, turns } = uprightView(viewFromAxes(x, y, R.slice(6, 9), C), UP);
    const err = Math.max(...view.map((v, i) => Math.abs(v - ref[i])));
    assert(turns === (4 - k) % 4 && err < 1e-5, `${k} turns: undone by ${turns}, ${err} off`);
    assert(Math.hypot(...viewTransform(view, C)) < 1e-5, 'the center moved');
  }
});
