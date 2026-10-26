// The view popover's Level (src/math.js tiltUp, app.js tiltView): the world up turned about the view
// direction, the orbit rebuilt at the pose on screen: the camera stays, the picture turns by the angle.
import { tiltUp, buildOrbitViewMatrix, orbitAnglesFromCenter, viewTransform, normalize, madd, dot, IDENTITY4, upFromTransform, levelTransform, parseTransform } from '../src/math.js';
import { assert } from './util.js';

// the orbit camera at the pose (center c, forward f) under `up`, as app.js _syncOrbitToPose
const orbitAt = (c, f, up, d = 3) => {
  const target = madd(c, f, d), o = orbitAnglesFromCenter(target, c, up);
  return buildOrbitViewMatrix(target, o.radius, o.theta, o.phi, up);
};
// (the views are Float32Array: 1e-3° and 1e-6 tolerances)
// the screen direction (x right, y down) from the view's center to world point p, in degrees clockwise from up
const screenAngle = (view, p) => { const q = viewTransform(view, p); return Math.atan2(q[0] / q[2], -q[1] / q[2]) * 180 / Math.PI; };

Deno.test('tiltUp: the camera stays, the picture turns clockwise by the angle', () => {
  const up = normalize([0.1, -1, 0.2]);
  const start = buildOrbitViewMatrix([0.5, 0, 1], 4, 30, 20, up), v = start.view, c = start.center;
  const f = [v[2], v[6], v[10]], p = madd(madd(c, f, 4), up, 1);  // a point above the view's center
  assert(Math.abs(screenAngle(v, p)) < 1e-4, `level start: ${screenAngle(v, p)}`);
  for (const deg of [1, -1, 5, -0.2]) {
    const u = tiltUp(up, f, deg), t = orbitAt(c, f, u), tv = t.view;
    assert(Math.hypot(...madd(t.center, c, -1)) < 1e-9, `deg ${deg}: the camera moved`);
    assert(dot([tv[2], tv[6], tv[10]], f) > 1 - 1e-6, `deg ${deg}: the camera turned`);
    const a = screenAngle(tv, p);
    assert(Math.abs(a - deg) < 1e-3, `deg ${deg}: the picture turned by ${a}°`);
  }
});

Deno.test('tiltUp: tilts add up, back and forth returns the up', () => {
  const up = [0, -1, 0], f = normalize([0.3, 0.4, 1]);
  let u = up;
  for (let k = 0; k < 5; k++) u = tiltUp(u, f, 1);
  const back = tiltUp(u, f, -5);
  assert(Math.hypot(...madd(back, up, -1)) < 1e-12, JSON.stringify(back));
  assert(Math.abs(Math.acos(dot(tiltUp(up, f, 5), up)) - Math.acos(dot(u, up))) < 1e-12, 'five 1° tilts are one 5° tilt');
});

Deno.test('levelTransform: the up of the transform follows the turns, the identity keeps the estimate', () => {
  const up0 = normalize([0.1, -1, 0.2]);
  let T = [...IDENTITY4], up = up0;
  assert(Math.hypot(...madd(upFromTransform(T, up0), up0, -1)) < 1e-12, 'identity');
  const turns = [[[0.3, 0.1, 1], 3], [[1, 0.2, 0.1], -2], [[-0.4, 0.3, 0.8], 0.2], [[0, 0, 1], 7]];
  for (const [f, deg] of turns) {
    T = levelTransform(T, f, deg);
    up = tiltUp(up, f, deg);
    const got = upFromTransform(T, up0);
    assert(Math.hypot(...madd(got, up, -1)) < 1e-12, `after ${deg}° about ${f}: ${got} vs ${up}`);
  }
  const R = [0, 1, 2].map(i => T.slice(4 * i, 4 * i + 3));  // still a rotation, no translation
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) assert(Math.abs(dot(R[i], R[j]) - (i === j ? 1 : 0)) < 1e-12, 'orthonormal');
  assert(T.slice(12).join() === '0,0,0,1' && T[3] === 0 && T[7] === 0 && T[11] === 0, JSON.stringify(T));
  const back = parseTransform(JSON.parse(JSON.stringify(T)));
  assert(back.every((v, i) => v === T[i]), 'JSON round trip');
  for (const bad of [null, [1, 2], 'x', [...IDENTITY4.slice(0, 15), NaN]]) assert(parseTransform(bad).join() === IDENTITY4.join(), `bad ${bad}`);
});
