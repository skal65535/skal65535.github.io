// The cameras refined by the splats (poses.js, DESIGN.md E19 2): the gauge takes out a common similarity of the
// increments and leaves the rest; a held-out view's increment from its neighbours' as a world turn and move; a
// snapshot's JSON gives the same poses back, a trainer made again with another scene radius too.
import { Poses, expRot } from '../src/poses.js';
import { setPose } from '../src/colmap.js';
import { assert, assertClose } from './util.js';

let seed = 5;
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const mul = (A, B) => [0, 1, 2].flatMap(i => [0, 1, 2].map(j => A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j]));
const mv = (A, v) => [0, 1, 2].map(i => A[3 * i] * v[0] + A[3 * i + 1] * v[1] + A[3 * i + 2] * v[2]);
// cameras along a curve, turned about, looking around
const cams = (n) => Array.from({ length: n }, (_, i) => {
  const R = expRot([0.3 * Math.sin(i), 0.2 * i, 0.1 * Math.cos(2 * i)]), c = [Math.cos(0.4 * i) * 3, 0.5 * Math.sin(i), i * 0.4];
  return setPose({ name: `t${String(100 * i).padStart(5, '0')}.jpg` }, R, mv(R, c).map(x => -x));
});
const state = (P, img, p) => { P.update(img, [0, 0, 0, 0, 0, 0], 0); P.views.get(img.name).p.set(p); };

Deno.test('poses: the gauge takes a common similarity out, leaves the rest', () => {
  const imgs = cams(8), P = new Poses(2), byName = new Map(imgs.map(im => [im.name, im]));
  // the increments of a world similarity p -> (1 + s) Exp(q) p + tau, to first order: phi = -R q, rho = -R dc
  const q = [0.002, -0.001, 0.003], tau = [0.01, -0.02, 0.005], s = 0.004;
  const c0 = [0, 1, 2].map(k => imgs.reduce((a, im) => a + im.center[k], 0) / imgs.length);
  for (const im of imgs) {
    const d = im.center.map((v, k) => v - c0[k]);
    const dc = [q[1] * d[2] - q[2] * d[1], q[2] * d[0] - q[0] * d[2], q[0] * d[1] - q[1] * d[0]].map((v, k) => v + tau[k] + s * d[k]);
    state(P, im, [...mv(im.R, dc).map(x => -x / P.radius), ...mv(im.R, q).map(x => -x)]);
  }
  P.center(byName);
  const left = Math.max(...[...P.views.values()].flatMap(st => [...st.p].map(Math.abs)));
  assert(left < 1e-12, `a similarity: ${left} left`);
  // increments without a common part: as they were
  const Q = new Poses(2), given = new Map();
  for (const im of imgs) { const p = Array.from({ length: 6 }, () => 0.01 * (rnd() - 0.5)); state(Q, im, p); }
  Q.center(byName);
  for (const [n, st] of Q.views) given.set(n, [...st.p]);
  Q.center(byName);
  for (const [n, st] of Q.views) st.p.forEach((x, j) => assertClose(x, given.get(n)[j], 1e-12, `${n} [${j}] centred twice`));
});

Deno.test('poses: a held-out view takes its neighbours\' turn and move, by frame number', () => {
  const imgs = cams(5), P = new Poses(1);
  const pa = [0.01, 0, -0.02, 0.003, 0, 0.001], pb = [-0.01, 0.02, 0, 0, 0.004, 0];
  state(P, imgs[0], pa); state(P, imgs[3], pb);
  P.apply(imgs);
  // imgs[1], imgs[2] (frames 100, 200, between 0 and 300): a third and two thirds of the way, as world quantities in their frames
  const world = (im, p) => [[0, 1, 2].map(i => im.sfmPose.R[i] * p[3] + im.sfmPose.R[3 + i] * p[4] + im.sfmPose.R[6 + i] * p[5]),
                             [0, 1, 2].map(i => im.sfmPose.R[i] * p[0] + im.sfmPose.R[3 + i] * p[1] + im.sfmPose.R[6 + i] * p[2])];
  const [wa, da] = world(imgs[0], pa), [wb, db] = world(imgs[3], pb);
  const check = (im, w, d, what) => {  // im's pose through the world turn w and move d (in R0^T rho terms)
    const phi = mv(im.sfmPose.R, w), rho = mv(im.sfmPose.R, d);
    const R = mul(expRot(phi), im.sfmPose.R), t = mv(expRot(phi), im.sfmPose.t).map((x, k) => x + rho[k]);
    R.forEach((x, j) => assertClose(im.R[j], x, 1e-12, `${what}: R[${j}]`));
    t.forEach((x, j) => assertClose(im.t[j], x, 1e-12, `${what}: t[${j}]`));
  };
  const lerp = (x, y, t) => x.map((v, k) => (1 - t) * v + t * y[k]);
  check(imgs[1], lerp(wa, wb, 1 / 3), lerp(da, db, 1 / 3), 'a third');
  check(imgs[2], lerp(wa, wb, 2 / 3), lerp(da, db, 2 / 3), 'two thirds');
  check(imgs[4], wb, db, 'past the end: the last one\'s');
});

Deno.test('poses: a snapshot\'s JSON gives the poses back, with another scene radius too', () => {
  const imgs = cams(4), P = new Poses(2);
  imgs.forEach((im, i) => state(P, im, [0.01 * i, -0.005, 0.002, 0.001 * i, 0, -0.002]));
  P.apply(imgs);
  const views = imgs.map(im => [...im.view]);
  const again = cams(4), Q = Poses.fromJSON(JSON.parse(JSON.stringify(P.toJSON(imgs))), again, 3.5);
  Q.apply(again);
  again.forEach((im, i) => im.view.forEach((x, j) => assertClose(x, views[i][j], 1e-6, `${im.name} view[${j}]`)));
});

Deno.test('poses: the pull to the SfM\'s holds an increment within sigma, takes it back without a gradient', () => {
  const [im, im2] = cams(2), cam = { width: 1000, height: 500, fx: 800 }, lr = 2e-4, sigma = 0.1 * Math.PI / 180;
  const P = new Poses(2);
  // a steady gradient: Adam's step lr, then the pull's (1 - lr / sigma): p = (p - lr)(1 - lr / sigma) at p = -(sigma - lr)
  for (let i = 0; i < 3000; i++) { P.update(im, [1, -2, 3, -1, 2, -3], lr, sigma); P.updateLens(cam, [1, -1, 1, 1], lr, sigma); }
  const p = P.views.get(im.name).p, L = P.lenses.get(cam).p;
  const at = sigma - lr;
  [-1, 1, -1, 1, -1, 1].forEach((sg, j) => assertClose(p[j], sg * at, 1e-3 * at, `view [${j}] held within sigma`));
  assertClose(L[0], -at, 1e-3 * at, 'log f held within sigma');
  [1, -1].forEach((sg, j) => assertClose(L[1 + j], sg * at * cam.width, 1e-3 * at * cam.width, `principal point [${j}] within sigma x the width`));
  const r2 = (500 ** 2 + 250 ** 2) / 800 ** 2;  // k1: its rate and bound over the corner's r^2 (its gradient at a fixed
  assertClose(L[3], -at / r2, 1e-3 * at / r2, 'k1 within sigma over the corner\'s r^2');  // 1 - m / (1 + k1 m) > 0)
  // no gradient (and no momentum: a fresh view): back by (1 - lr / sigma) a step
  const p0 = [0.002, -0.001, 0.003, 0.001, 0.002, -0.004], n = 100;
  state(P, im2, p0);
  for (let i = 0; i < n; i++) P.update(im2, [0, 0, 0, 0, 0, 0], lr, sigma);
  p0.forEach((x, j) => assertClose(P.views.get(im2.name).p[j], x * (1 - lr / sigma) ** n, 1e-12, `view [${j}] pulled back`));
  // without a pull (sigma 0): as before, the steady gradient carries it on
  const Q = new Poses(2);
  for (let i = 0; i < 3000; i++) Q.update(im, [1, -2, 3, -1, 2, -3], lr);
  assert(Math.abs(Q.views.get(im.name).p[3]) > 0.5, `no pull: ${Q.views.get(im.name).p[3]}`);
});
