import { compose, inverse, apply, perturb, error, fromPose, toPose } from '../src/sfm/sim3.js';
import { optimizePoseGraph } from '../src/sfm/posegraph.js';
import { expSO3, rotationAngle } from '../src/sfm/linalg.js';
import { poseFromCenter, poseCenter } from '../src/sfm/camera.js';
import { gaussian } from '../src/sfm/synth.js';
import { assert, assertClose, rng, randomRotation } from './util.js';

const randomSim3 = (r) => ({ s: Math.exp(0.5 * gaussian(r)), R: randomRotation(r), t: [gaussian(r), gaussian(r), gaussian(r)] });

Deno.test('Sim(3) compose, inverse, poses', () => {
  const r = rng(50);
  for (let k = 0; k < 50; k++) {
    const a = randomSim3(r), b = randomSim3(r), x = [gaussian(r), gaussian(r), gaussian(r)];
    assertClose(apply(compose(a, b), x), apply(a, apply(b, x)), 1e-10, 'compose');
    assertClose(apply(inverse(a), apply(a, x)), x, 1e-10, 'inverse');
    assertClose(error(compose(a, inverse(a))), [0, 0, 0, 0, 0, 0, 0], 1e-10, 'identity error');
  }
  const pose = poseFromCenter(randomRotation(r), [1, 2, 3]);
  const S = compose({ s: 2.5, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0] }, fromPose(pose));  // camera frame scaled
  assertClose(poseCenter(toPose(S)), [1, 2, 3], 1e-10, 'scaled camera frame keeps the pose');
});

// A ring of cameras; sequential edges measured from the truth with noise, initial nodes with
// accumulated drift (rotation, translation, scale); one loop edge closes the ring.
Deno.test('pose graph removes accumulated drift through a loop edge', () => {
  const r = rng(51);
  const n = 60;
  const truth = Array.from({ length: n }, (_, k) => {
    const a = 2 * Math.PI * k / n;
    return fromPose(poseFromCenter(expSO3([0, 0, a]), [10 * Math.cos(a), 10 * Math.sin(a), 0.5 * Math.sin(3 * a)]));
  });
  const rel = (i, j) => compose(truth[j], inverse(truth[i]));
  const noisy = (S, sigma) => perturb(S, Array.from({ length: 7 }, () => sigma * gaussian(r)));
  const edges = [];
  for (let k = 1; k < n; k++) edges.push({ i: k - 1, j: k, S: noisy(rel(k - 1, k), 1e-4) });
  for (let k = 2; k < n; k++) edges.push({ i: k - 2, j: k, S: noisy(rel(k - 2, k), 1e-4) });
  edges.push({ i: n - 1, j: 0, S: rel(n - 1, 0) });  // loop
  // dead reckoning with drift: every step slightly rotated and scaled
  const nodes = [truth[0]];
  for (let k = 1; k < n; k++) nodes.push(compose(perturb(rel(k - 1, k), [0, 0.003, 0, 0, 0, 0, 0.01]), nodes[k - 1]));
  const before = Math.max(...nodes.map((N, k) => Math.hypot(...poseCenter(toPose(N)).map((x, i) => x - poseCenter(toPose(truth[k]))[i]))));
  const out = optimizePoseGraph(nodes, edges);
  const after = Math.max(...nodes.map((N, k) => Math.hypot(...poseCenter(toPose(N)).map((x, i) => x - poseCenter(toPose(truth[k]))[i]))));
  assert(out.cost < 1e-3 * out.initialCost, `cost ${out.initialCost} -> ${out.cost}`);
  assert(before > 2 && after < 0.05, `max center error ${before.toFixed(2)} -> ${after.toFixed(4)} (ring radius 10)`);
  for (let k = 0; k < n; k++) assert(rotationAngle(nodes[k].R, truth[k].R) < 2e-3, `rotation ${k}`);
});
