import { compareModels, compareLocal, checkThresholds } from '../src/sfm/compare.js';
import { apply as applySim3 } from '../src/sfm/sim3.js';
import { mul3, transpose3, expSO3 } from '../src/sfm/linalg.js';
import { poseCenter, poseFromCenter } from '../src/sfm/camera.js';
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { assert, assertClose, rng, randomRotation } from './util.js';

// The same cameras expressed in another world frame: x_new = s A x + b
function transformModel(model, sim) {
  const At = transpose3(sim.R);
  return {
    ...model,
    images: model.images.map(im => poseFromCenter(mul3(im.R, At), applySim3(sim, poseCenter(im)))).map((p, i) =>
      ({ name: model.images[i].name, ...p })),
  };
}

const truth = makeSyntheticScene({ numFrames: 60, numGround: 2000, numBoxes: 2, numFar: 200 }).truth.model;

Deno.test('a model compared with itself in another Sim(3) frame has zero error', () => {
  const r = rng(10);
  const sim = { s: 0.37, R: randomRotation(r), t: [4, -2, 9] };
  const m = compareModels(transformModel(truth, sim), truth);
  assertClose(m.registered, 1, 0, 'registered');
  assert(m.rmsCenterRel < 1e-9, `centers ${m.rmsCenterRel}`);
  assert(m.medianRotDeg < 1e-6 && m.maxRotDeg < 1e-6, `rotations ${m.maxRotDeg}`);
  assertClose(m.sim3.s, 1 / 0.37, 1e-9, 'scale est -> ref');
  assert(checkThresholds(m, 'truth').length === 0, 'passes');
});

Deno.test('known perturbations are measured', () => {
  // every other camera rotated by 0.573 deg about its center: the alignment, which also uses the
  // camera axes, absorbs part of it, so expect the error between a third and the full rotation
  const rot = expSO3([0, 0.01, 0]);
  const est = { ...truth, focal: truth.focal * 1.05, images: truth.images.slice(0, 50).map((im, k) =>
    ({ name: im.name, ...poseFromCenter(k % 2 ? mul3(rot, im.R) : im.R, poseCenter(im)) })) };
  const m = compareModels(est, truth);
  assertClose(m.registered, 50 / 60, 1e-12, 'registered');
  assert(m.maxRotDeg > 0.2 && m.maxRotDeg < 0.58, `max rotation ${m.maxRotDeg}`);
  assert(m.rmsCenterRel < 2e-3, `centers barely moved by the alignment: ${m.rmsCenterRel}`);
  assertClose(m.focalErr, 0.05, 1e-12, 'focal error');
  const fails = checkThresholds(m, 'truth');
  assert(fails.length >= 2 && fails.some(f => f.startsWith('registered')) && fails.some(f => f.startsWith('focal')), `fails: ${fails}`);
});

Deno.test('compareLocal is gauge-free and finds a single bad frame', () => {
  const r = rng(11);
  const sim = { s: 3, R: randomRotation(r), t: [1, 1, 1] };
  const moved = transformModel(truth, sim);
  const clean = compareLocal(moved, truth);
  assert(clean.medianRelRotDeg < 1e-6 && clean.p90DirDeg < 1e-4, 'zero in another gauge');
  moved.images[20] = { name: moved.images[20].name, ...poseFromCenter(mul3(expSO3([0.05, 0, 0]), moved.images[20].R), poseCenter(moved.images[20])) };
  const bad = compareLocal(moved, truth).pairs.filter(p => p.rotDeg > 1);
  assert(bad.length === 2 && bad.every(p => p.a === truth.images[19].name || p.a === truth.images[20].name), `bad pairs ${bad.map(p => p.a)}`);
});
