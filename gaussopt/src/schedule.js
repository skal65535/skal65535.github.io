// schedule.js — The training's parameters and schedule, GPU-free: the defaults of the params a step reads, the
// learning-rate decay, the shape limits, when a rebalance / opacity reset / validation / seeding comes, the splat
// budget and the consolidation's keep score (run by trainer.js)
"use strict";

import { MOMENT_FLOATS } from './octree.js';

export const SEED_LEAD = 16;  // steps between the read of the stipple seeds' view and their rebalance (_prepareSeeds)

// Defaults of the params a training step reads (the app's params start from these)
export const TRAIN_DEFAULTS = {
  downsample:      4,      // the training resolution: the first camera's size over this
  opt_method:      'monte_carlo',  // 'adam' | 'monte_carlo' (SGLD/MCMC; E6: the best on average)
  // Growth: the rebalance (octree.js reconcileAndGrow)
  rebalance_every: 100,
  rebalance_every_cap: 400,  // once the count is at max_splats (0: rebalance_every), DESIGN.md §5.3
  enable_rebalance:true,
  enable_split:    true,
  enable_merge:    true,
  remove_floaters: true,
  floater_depth:   'observed',  // near-camera carve depth: 'observed' | 'view' (octree.js nearCameraCone)
  floater_spare:   0.01,        // x sceneRadius: splats this near an SfM point are not carved (0: off; DESIGN.md §5)
  max_splats:      300000,
  cap_lease:       0,      // E7: the count may go this fraction over max_splats until the consolidation (needs train_steps)
  train_steps:     0,      // E7: the planned length of the training (0: open-ended, no lease)
  consolidate_frac: 0.2,   // E7: the last fraction of train_steps, back to max_splats (splatBudget)
  polish:          false,  // E7: Monte-Carlo's noise and L1 terms off over that last fraction (plain Adam steps)
  prune_opacity:   0.008,
  hysteresis:      0.35,
  stipple_seed:    true,   // E4: seed new splats from residual stippling
  stipple_count:   2000,
  region_log:      false,  // E7: per-rebalance error of fixed cells of the scene, in trainer.regionLog (regions.js)
  region_res:      8,      // E7: cells per axis
  region_repair:   0,      // E7: > 0: the stuck cells' splits need R > Tu / (1 + this), Monte-Carlo relocates there (1 + this) as often
  // Optimizer extras: Monte-Carlo (3DGS-MCMC: noise, L1 terms, relocation)
  mc_noise:        1e6,    // lambda_noise: SGLD noise on the nearly transparent splats' centers, x lr_pos x covariance (3DGS-MCMC: 5e5;
                           // DESIGN.md §5.4)
  mc_opacity_reg:  0.01,   // lambda_o: mean |opacity| added to the loss
  mc_scale_reg:    0.01,   // lambda_s: mean |scale| (per axis)
  mc_relocate:     'opacity',  // the live splats the dead ones move onto, drawn ~ 'opacity' or 'error'
  mc_relocate_mode: 'copy',   // ... as copies in place (3DGS-MCMC) or 'shape': each one drawn splits within its shape (brush)
  // Learning rates
  lr_pos:          0.00002,  // x the scene radius, decaying to 1e-6 x it as brush's lr_mean (x its bounds' median side there;
                             // 3DGS: 1.6e-4; DESIGN.md E11)
  lr_pos_final:    0.05,   // lr_pos decays exponentially to lr_pos * lr_pos_final ...
  lr_decay_steps:  10000,  // ... over the planned length (train_steps) or, open-ended, this many steps
  lr_scale:        0.005,
  lr_quat:         0.002,
  lr_color:        0.0025,
  lr_opacity:      0.01,   // brush's; higher floats splats under floors (DESIGN.md E9)
  ssim:            0,      // D-SSIM loss weight (0 = off: 0.8 L1 + 0.2 L2), see tile_backward
  bg_lr:           0,      // the background's learning rate (env.js, DESIGN.md E13; 0 = off: black)
  exposure_lr:     0,      // each training view's exposure, its learning rate (exposure.js, DESIGN.md E19; 0 = off)
  exposure_from:   1000,   // ... learned from this step on (the splats formed first)
  pose_lr:         0,      // each training view's pose (poses.js, DESIGN.md E19; rad a visit, the translation x the scene radius; 0 = off)
  lens_lr:         0,      // each lens's log focal (the principal point x the image's width; 0 = off)
  pose_from:       1000,   // both from this step on
  pose_prior:      0,      // their pull to the SfM's: the increments held within about this (degrees; 0 = none)
  pose_refit:      0,      // every N steps (with pose_lr), each training camera refitted with the splats frozen (0 = never)
  pose_refit_iters: 20,    // ... its steps a view
  drop_rate:       0,      // DropGaussian: share of splats skipped in each training render (0 = off)
  split_min_views: 0,      // growth splits only splats seen from at least this many views (N_eff; 0 = any)
  split_mode:      'shape', // how a splat splits: 'shape' (within its own shape, as brush; DESIGN.md E9) or 'error' (along its error moments, §5.1)
  opacity_reset:   3000,   // clamp all opacities to 0.01 every N steps (0 = off)
  opacity_reset_until: 15000,  // ... up to this step (3DGS: densify_until_iter), and half the planned length (train_steps)
  // Shape limits (applied every Adam step; 0: none). Off by default, as brush: on, they shrink the long thin
  // splats of grass and branches (DESIGN.md E11)
  max_splat_angle: 0,      // max axis scale / depth, in the training view (e.g. 0.03)
  max_splat_scale: 0,      // max axis scale, as a fraction of the scene radius (0.1)
  max_anisotropy:  0,      // max ratio of the largest axis to the median one (10)
};

// lr_pos's factor at `step`: down exponentially to lr_pos_final over the planned length (train_steps), or
// lr_decay_steps when open-ended, then flat
export const posLrDecay = (P, step) =>
  Math.pow(P.lr_pos_final, Math.min(1, step / Math.max(1, P.train_steps > 0 ? P.train_steps : P.lr_decay_steps)));

// The shape limits as TrainUniforms.limits.xyz: max scale / depth, max world scale, log(max anisotropy); 0 is none
const NO_LIMIT = 1e6;
export const shapeLimits = (P, R) => [
  P.max_splat_angle > 0 ? P.max_splat_angle : NO_LIMIT,
  (P.max_splat_scale > 0 ? P.max_splat_scale : NO_LIMIT) * R,
  Math.log(P.max_anisotropy > 0 ? Math.max(1, P.max_anisotropy) : NO_LIMIT),
];

// Whether step `step` rebalances, the last rebalance at step `last`: rebalance_every steps
// after it, rebalance_every_cap once `count` is at the splat cap (0: rebalance_every), and at
// every opacity reset. While the lease's budget falls (consolidating, until consolidationEnd): rebalance_every
// steps, and one at its end, so the count is max_splats from there
export function rebalanceDue(P, step, last, count) {
  if (consolidating(P, step)) {
    const c1 = consolidationEnd(P);
    if (last < c1 && step >= c1) return true;
    if (step < c1) return step - last >= P.rebalance_every;
  }
  const atCap = P.rebalance_every_cap > 0 && count >= 0.99 * splatBudget(P, step);
  return step - last >= (atCap ? P.rebalance_every_cap : P.rebalance_every) ||
         (P.opacity_reset > 0 && step % P.opacity_reset === 0);
}

// The opacity reset (3DGS; not with monte_carlo, whose L1 term on the opacity prunes instead) at a rebalance
// at `step`, the last one at `last`: when a multiple of opacity_reset is in (last, step], up to
// opacity_reset_until and half the planned length (train_steps): the splats need the steps after the last one
// to earn their opacity back
export function opacityResetDue(P, step, last) {
  const every = resetPeriod(P);
  if (!(every > 0)) return false;
  const m = Math.floor(step / every) * every;
  return m > last && m <= resetUntil(P);
}
const resetPeriod = (P) => P.opt_method === 'monte_carlo' ? 0 : P.opacity_reset;
const resetUntil = (P) => P.train_steps > 0 ? Math.min(P.opacity_reset_until, P.train_steps / 2) : P.opacity_reset_until;

// The first step after `step` that may reset the opacities: the next multiple of opacity_reset, which rebalances
// (rebalanceDue; while consolidating, the reset waits for the next rebalance), Infinity when none is left
export function nextOpacityReset(P, step) {
  const every = resetPeriod(P);
  if (!(every > 0)) return Infinity;
  const m = (Math.floor(step / every) + 1) * every;
  return m <= resetUntil(P) ? m : Infinity;
}

// Whether the training loop (app.js) stops before its next step, at `step`: 'stop' stop_after steps after training
// started (at `start`), 'val' right before an opacity reset when not validated at this step yet (at `lastVal`), so
// validation reads the splats before the reset turns them nearly transparent, not just after
export function haltBefore(P, step, lastVal, start) {
  if (P.stop_after > 0 && step - start >= P.stop_after) return 'stop';
  if (P.val_every > 0 && lastVal < step && nextOpacityReset(P, step) === step + 1) return 'val';
  return null;
}

// The splat budget at `step` (DESIGN.md §9 E7): max_splats, or with a planned length (train_steps > 0) and a
// lease (cap_lease > 0), max_splats x (1 + cap_lease) until the consolidation, the last consolidate_frac of the
// steps: there it falls linearly back to max_splats over the first 3/4, the rest trains at max_splats
const consolidationEnd = (P) => P.train_steps * (1 - 0.25 * P.consolidate_frac);
// Whether step `step` (no rebalance), the last rebalance at `last`, starts the stipple seeds of the next one
// (GaussOptTrainer._prepareSeeds): it is due SEED_LEAD steps on, seeds it, and this window has none yet (`prepStep`,
// the step of the last ones made, or undefined). A rebalance that comes earlier seeds in the page.
export function seedPrepDue(P, step, last, count, prepStep) {
  const ahead = step + SEED_LEAD;
  if (!P.enable_rebalance || !P.stipple_seed || !(Number(P.stipple_count) > 0) || prepStep > last) return false;
  return rebalanceDue(P, ahead, last, count) && !consolidating(P, ahead);
}

export function splatBudget(P, step) {
  const T = P.train_steps, lease = P.cap_lease;
  if (!(T > 0) || !(lease > 0)) return P.max_splats;
  const c0 = T * (1 - P.consolidate_frac), c1 = consolidationEnd(P);
  const f = step <= c0 ? 1 : step >= c1 ? 0 : (c1 - step) / (c1 - c0);
  return Math.round(P.max_splats * (1 + lease * f));
}
// The consolidation's keep score: each splat's W (visibility mass) per window, averaged over the windows it
// lived with a decay of 0.7 a window (a video's window does not visit every view). vis: { sum, norm, score } or null
export function visibilityScore(vis, moments, count) {
  const sum = new Float32Array(count), norm = new Float32Array(count), score = new Float32Array(count);
  const ok = vis?.sum.length === count;
  for (let i = 0; i < count; i++) {
    sum[i] = (ok ? 0.7 * vis.sum[i] : 0) + moments[i * MOMENT_FLOATS];
    norm[i] = (ok ? 0.7 * vis.norm[i] : 0) + 1;
    score[i] = sum[i] / norm[i];
  }
  return { sum, norm, score };
}
// through a rebalance: src[i] = the pre-rebalance index of splat i (-1: new, no history)
export function remapVisibility(vis, src) {
  const sum = new Float32Array(src.length), norm = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) if (src[i] >= 0) { sum[i] = vis.sum[src[i]]; norm[i] = vis.norm[src[i]]; }
  return { sum, norm, score: null };
}
// the last consolidate_frac of a planned length: the lease's consolidation, and with polish Monte-Carlo's noise and L1 terms off
export const endPhase = (P, step) => P.train_steps > 0 && step > P.train_steps * (1 - P.consolidate_frac);
export const consolidating = (P, step) => P.cap_lease > 0 && endPhase(P, step);
