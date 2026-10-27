// The training defaults DESIGN.md E11 took from brush: the shape limits adam_step applies (trainer.js shapeLimits ->
// TrainUniforms.limits) off, 0 for none, a set limit passed through as before; lr_pos's schedule (posLrDecay).
import { TRAIN_DEFAULTS, shapeLimits, posLrDecay } from '../src/trainer.js';
import { assert, assertClose } from './util.js';

// what the shader caps a splat's largest axis to (shaders.wgsl adam_step): min(limits.x z, limits.y), and its
// log-scale to the median's + limits.z
const capAt = ([angle, scale], z) => Math.min(angle * z, scale);

Deno.test('shape limits: none by default, so a long thin splat (brush\'s grass: 0.2 x depth, 300:1) is kept', () => {
  const R = 5, z = 2, lim = shapeLimits(TRAIN_DEFAULTS, R);
  assert(capAt(lim, z) >= 0.2 * z, `largest axis capped at ${capAt(lim, z)} (the splat's: ${0.2 * z})`);
  assert(lim[2] >= Math.log(300), `anisotropy capped at ${Math.exp(lim[2])}`);
});

Deno.test('shape limits: a set limit is applied as before (angle, scale x scene radius, log anisotropy)', () => {
  const lim = shapeLimits({ max_splat_angle: 0.03, max_splat_scale: 0.1, max_anisotropy: 10 }, 5);
  assertClose(lim[0], 0.03, 1e-12, 'angle');
  assertClose(lim[1], 0.5, 1e-12, 'scale');
  assertClose(lim[2], Math.log(10), 1e-12, 'anisotropy');
  const one = shapeLimits({ max_splat_angle: 0, max_splat_scale: 0.1, max_anisotropy: 0 }, 5);
  assert(one[0] > 1e3 && one[2] > Math.log(1e3) && Math.abs(one[1] - 0.5) < 1e-12, `one limit set: ${one}`);
});

// brush v0.3.0's lr_mean: 2e-5 down to lr_mean_end 1e-6 (x the scene extent) over the run
Deno.test('lr_pos: brush\'s schedule, 2e-5 to 1e-6 over the planned length (10k steps when open-ended)', () => {
  const lr = (P, step) => P.lr_pos * posLrDecay(P, step);
  const open = { ...TRAIN_DEFAULTS, train_steps: 0 };
  assertClose(lr(open, 0), 2e-5, 1e-12, 'start');
  assertClose(lr(open, 10000), 1e-6, 1e-12, 'open-ended, at 10k');
  assertClose(lr(open, 25000), 1e-6, 1e-12, 'flat after');
  const planned = { ...TRAIN_DEFAULTS, train_steps: 30000 };
  assert(lr(planned, 10000) > 3e-6, `planned 30k, at 10k: ${lr(planned, 10000)}`);
  assertClose(lr(planned, 30000), 1e-6, 1e-12, 'planned 30k, at its end');
});
