// util.js — Minimal assertions and helpers for `deno test` (no network dependencies).
"use strict";

import { makeRng } from '../src/math.js';
import { gaussian } from '../src/sfm/synth.js';
import { expSO3 } from '../src/sfm/linalg.js';
import { loadDump, denoReader } from '../src/sfm/dump.js';
import { exists, colmapModelDir, assessAgainstColmap } from '../tools/common.js';

export function assert(cond, msg = 'assertion failed') {
  if (!cond) throw new Error(msg);
}

export function assertClose(actual, expected, tol, msg = '') {
  const a = Array.isArray(actual) || ArrayBuffer.isView(actual) ? Array.from(actual) : [actual];
  const e = Array.isArray(expected) || ArrayBuffer.isView(expected) ? Array.from(expected) : [expected];
  assert(a.length === e.length, `${msg} length ${a.length} != ${e.length}`);
  for (let i = 0; i < a.length; i++) {
    if (!(Math.abs(a[i] - e[i]) <= tol)) throw new Error(`${msg} [${i}] ${a[i]} != ${e[i]} (tol ${tol})`);
  }
}

// Environment (undefined without --allow-env), files, and the GPU, for the tests that need them
export const env = (k) => { try { return Deno.env.get(k); } catch { return undefined; } };
export { exists };
export const hasGPU = !!globalThis.navigator?.gpu && !!(await navigator.gpu.requestAdapter().catch(() => null));

// Slow tests (real videos, large datasets) run with SLOW=1 (deno task test:slow)
export const SLOW = env('SLOW') === '1';

// The jardin5 COLMAP dump (tools/colmap_dump.py) and model used by the slow tests (SFM_REF_MODEL
// overrides the model: a sparse/0 directory or a gs/ dataset)
export const DUMP = env('SFM_DUMP');
export const REF_MODEL = env('SFM_REF_MODEL') ?? `${env('HOME')}/gskal/brush/jardin5/out/sparse_sift/0`;
export const HAS_REF = !!colmapModelDir(REF_MODEL);
export const loadInput = () => loadDump(denoReader(DUMP));
export const assessColmap = (model, input) => assessAgainstColmap(model, REF_MODEL, input);

// The synthetic orbit of the reconstruction and session tests
export const ORBIT = { numFrames: 50, maxFeatures: 1200, numGround: 8000, numBoxes: 8, numFar: 1000, turns: 1.3, noisePx: 1.0 };

export const rng = (seed) => makeRng(seed);
export const randomRotation = (r) => expSO3([0, 1, 2].map(() => gaussian(r)));
export const randomMatrix = (r, n) => Float64Array.from({ length: n }, () => gaussian(r));
