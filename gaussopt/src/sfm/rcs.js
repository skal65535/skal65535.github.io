// rcs.js — The reduced camera system of a bundle adjustment (ba.js, ba_gpu.js): symmetric, by blocks (a pose's 6
// unknowns, a camera model's intrinsics), the lower blocks (a >= b) only filled. Up to denseMax unknowns it is one dense
// matrix solved by Cholesky, as a window's always was. Past it, the blocks are kept apart and the system solved by
// conjugate gradients preconditioned by its diagonal blocks (Ceres's ITERATIVE_SCHUR with a block Jacobi
// preconditioner, COLMAP's solver past ~1000 images): a pose couples only with the poses that see its points, so a
// segment's thousands of poses cost about their blocks per iteration, not the cube of the unknowns (the dense Cholesky
// of autumn-garden's 1094 frames, 6564 unknowns, ~1e11 operations a solve, took its global BA to 40 min).
"use strict";

import { choleskyFactor, choleskyFactorSolve, choleskySolve } from './linalg.js';

export const DENSE_MAX = 1200;  // unknowns (200 poses) solved dense
// the conjugate gradients' stop: the residual's norm over the right side's. A damped step needs no exact solve (Ceres's
// inexact Newton, its eta 0.1): autumn-garden's first 450 keyframes, its global BAs' solves 70.3 -> 32.7 s at 1e-2 from
// 1e-8 (148408 -> 40535 iterations, 6 solves stopped at maxIterations -> 0), the poses apart only at its segments' joints;
// a synthetic 1000-frame orbit 2138 -> 1977 s, RMS 0.0027% and rot 0.067 deg both
export const CG_TOL = 1e-2;

export class ReducedSystem {
  // sizes: the unknowns of each block in order; denseMax: the most unknowns solved dense; tol, maxIterations: the
  // conjugate gradients' stop (the residual's norm relative to the right side's), and their most
  constructor(sizes, { denseMax = DENSE_MAX, tol = CG_TOL, maxIterations = 2000 } = {}) {
    this.nB = sizes.length;
    this.size = Int32Array.from(sizes);
    this.off = new Int32Array(this.nB + 1);
    for (let a = 0; a < this.nB; a++) this.off[a + 1] = this.off[a] + this.size[a];
    this.m = this.off[this.nB];
    this.dense = this.m <= denseMax;
    Object.assign(this, { tol, maxIterations });
    this.S = this.dense ? new Float64Array(this.m * this.m) : null;
    this.blocks = this.dense ? null : new Map();  // a * nB + b -> Float64Array(size a x size b), row-major
    this.iterations = 0;  // the last sparse solve's
    this.converged = true;  // and whether it reached tol (false: stopped at maxIterations, its x the last iterate)
  }

  // Block (a, b), a >= b: [array, base, stride], its entry (i, j) at array[base + i * stride + j] (a diagonal block: j <= i)
  at(a, b) {
    if (this.dense) return [this.S, this.off[a] * this.m + this.off[b], this.m];
    const key = a * this.nB + b;
    let B = this.blocks.get(key);
    if (!B) { B = new Float64Array(this.size[a] * this.size[b]); this.blocks.set(key, B); }
    return [B, 0, this.size[b]];
  }

  // The diagonal, its unknowns in order
  diagonal() {
    const d = new Float64Array(this.m);
    for (let a = 0; a < this.nB; a++) {
      const [A, base, stride] = this.at(a, a);
      for (let i = 0; i < this.size[a]; i++) d[this.off[a] + i] = A[base + i * stride + i];
    }
    return d;
  }

  // The diagonal += D
  addDiagonal(D) {
    for (let a = 0; a < this.nB; a++) {
      const [A, base, stride] = this.at(a, a);
      for (let i = 0; i < this.size[a]; i++) A[base + i * stride + i] += D[this.off[a] + i];
    }
  }

  // The sum of the entries held (a NaN or an infinity among them shows)
  sum() {
    let s = 0;
    if (this.dense) { for (const v of this.S) s += v; return s; }
    for (const B of this.blocks.values()) for (const v of B) s += v;
    return s;
  }

  // x with S x = b, null when it fails (S not positive definite)
  solve(b) {
    if (!this.m) return new Float64Array(0);
    if (this.dense) return choleskySolve(this.S, b, this.m);
    return this._cg(b);
  }

  _cg(b) {
    const { nB, size, off, m } = this;
    // the blocks as lists; the diagonal ones made whole (their upper triangle from the lower), each factored for the
    // preconditioner
    const ba = [], bb = [], arr = [];
    for (const [key, B] of this.blocks) { ba.push(Math.floor(key / nB)); bb.push(key % nB); arr.push(B); }
    const diag = new Array(nB).fill(null), L = new Array(nB).fill(null);
    for (let k = 0; k < arr.length; k++) {
      if (ba[k] !== bb[k]) continue;
      const n = size[ba[k]], B = arr[k];
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) B[i * n + j] = B[j * n + i];
      diag[ba[k]] = B;
    }
    for (let a = 0; a < nB; a++) {
      const n = size[a], F = new Float64Array(n * n);
      if (diag[a] && choleskyFactor(diag[a], 0, F, 0, n)) L[a] = F;
      else return null;  // a block without its own curvature: the system is not positive definite
    }
    const mul = (x, y) => {
      y.fill(0);
      for (let k = 0; k < arr.length; k++) {
        const a = ba[k], c = bb[k], B = arr[k], na = size[a], nc = size[c], oa = off[a], oc = off[c];
        if (a === c) {
          for (let i = 0; i < na; i++) { let s = 0; for (let j = 0; j < na; j++) s += B[i * na + j] * x[oa + j]; y[oa + i] += s; }
          continue;
        }
        for (let i = 0; i < na; i++) {
          let s = 0;
          const xi = x[oa + i];
          for (let j = 0; j < nc; j++) { const v = B[i * nc + j]; s += v * x[oc + j]; y[oc + j] += v * xi; }
          y[oa + i] += s;
        }
      }
    };
    const precond = (r, z) => { for (let a = 0; a < nB; a++) choleskyFactorSolve(L[a], 0, size[a], r, off[a], z, off[a]); };
    const dot = (u, v) => { let s = 0; for (let i = 0; i < m; i++) s += u[i] * v[i]; return s; };
    const x = new Float64Array(m), r = Float64Array.from(b), z = new Float64Array(m), q = new Float64Array(m);
    const nb = Math.sqrt(dot(b, b));
    this.iterations = 0;
    this.converged = true;
    if (!(nb > 0)) return x;
    precond(r, z);
    const p = Float64Array.from(z);
    let rz = dot(r, z);
    this.converged = false;
    for (let it = 0; it < this.maxIterations; it++) {
      this.iterations = it + 1;
      mul(p, q);
      const pq = dot(p, q);
      if (!(pq > 0)) return it ? x : null;  // a direction of no curvature: not positive definite
      const alpha = rz / pq;
      for (let i = 0; i < m; i++) { x[i] += alpha * p[i]; r[i] -= alpha * q[i]; }
      if (Math.sqrt(dot(r, r)) <= this.tol * nb) { this.converged = true; break; }
      precond(r, z);
      const rz2 = dot(r, z), beta = rz2 / rz;
      rz = rz2;
      for (let i = 0; i < m; i++) p[i] = z[i] + beta * p[i];
    }
    return x;
  }
}
