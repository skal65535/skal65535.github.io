// anneal.js — Simulated annealing of the splat centers (opt_method 'sim_anneal', DESIGN.md §5.4), at the
// rebalances: a Metropolis test of last window's jumps, then new jumps. Pure CPU (no DOM, no GPU).
//
// The energy of splat i over a window (the steps since the last rebalance, every training view it was seen
// from) is its mean footprint error e_i = R_i / W_i (the moments, octree.js), over the mean of the visible
// splats' e. A jump made at the last rebalance is judged on de_i = e_i(now) - e_i(then), less the median
// de of the visible splats that did not jump (the training lowers every e meanwhile): kept if de <= 0,
// else with probability exp(-de / T), else undone. Comparing errors of the same splat in different views
// (one view per step) instead measures the views, not the jump.
"use strict";

import { SPLAT_FLOATS } from './colmap.js';
import { MOMENT_FLOATS } from './octree.js';
import { quatToMat3, randn } from './math.js';

// state: { jump: Float32Array(3 count) (0: none), e: Float32Array(count) (the energy when it jumped) } or null.
// Changes params (centers) in place; returns { state, accepted, rejected, proposed }.
// temp: the temperature T; share: the share of the visible splats given a new jump; size: the jump's
// standard deviation in the splat's own axes (scales), times T.
export function annealStep(params, count, moments, state, { temp, share = 0.25, size = 1, rand = Math.random } = {}) {
  const e = new Float32Array(count), seen = new Uint8Array(count);
  let sum = 0, n = 0;
  for (let i = 0; i < count; i++) {
    const W = moments[i * MOMENT_FLOATS], R = moments[i * MOMENT_FLOATS + 2];
    if (!(W > 1e-3)) continue;
    e[i] = R / W; seen[i] = 1; sum += e[i]; n++;
  }
  const out = { state: { jump: new Float32Array(3 * count), e: new Float32Array(count) }, accepted: 0, rejected: 0, proposed: 0 };
  if (n === 0) return out;
  const mean = sum / n;
  for (let i = 0; i < count; i++) e[i] /= mean;

  // 1. last window's jumps: kept or undone
  if (state) {
    const jumped = (i) => state.jump[3 * i] !== 0 || state.jump[3 * i + 1] !== 0 || state.jump[3 * i + 2] !== 0;
    const still = [];
    for (let i = 0; i < count; i++) if (seen[i] && !jumped(i) && state.e[i] > 0) still.push(e[i] - state.e[i]);
    still.sort((a, b) => a - b);
    const trend = still.length ? still[still.length >> 1] : 0;
    for (let i = 0; i < count; i++) {
      if (!jumped(i)) continue;
      const de = seen[i] ? e[i] - state.e[i] - trend : Infinity;  // unseen since: no evidence, undone
      if (de <= 0 || rand() < Math.exp(-de / Math.max(1e-6, temp))) { out.accepted++; continue; }
      for (let c = 0; c < 3; c++) params[i * SPLAT_FLOATS + c] -= state.jump[3 * i + c];
      out.rejected++;
    }
  }
  // 2. new jumps, along the splat's axes; every visible splat's energy recorded (the next trend)
  for (let i = 0; i < count; i++) {
    if (!seen[i]) continue;
    out.state.e[i] = e[i];
    if (!(rand() < share)) continue;
    const b = i * SPLAT_FLOATS, Rm = quatToMat3([params[b + 8], params[b + 9], params[b + 10], params[b + 11]]);
    const j = [0, 1, 2].map(k => randn() * Math.exp(params[b + 4 + k]) * size * temp);
    for (let c = 0; c < 3; c++) {
      const d = Rm[3 * c] * j[0] + Rm[3 * c + 1] * j[1] + Rm[3 * c + 2] * j[2];  // R j (Rm row-major)
      params[b + c] += d;
      out.state.jump[3 * i + c] = d;
    }
    out.proposed++;
  }
  return out;
}

// The state carried through a rebalance: src[i] = the pre-rebalance index of new splat i (-1: new)
export function remapAnnealState(state, src) {
  if (!state) return null;
  const n = src.length, jump = new Float32Array(3 * n), e = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const k = src[i];
    if (k < 0) continue;
    e[i] = state.e[k];
    for (let c = 0; c < 3; c++) jump[3 * i + c] = state.jump[3 * k + c];
  }
  return { jump, e };
}
