// posegraph.js — Sim(3) pose graph (DESIGN-sfm §5.4): nodes are world -> camera Sim(3) transforms
// S_i, an edge (i, j, S_ji) says camera j = S_ji o camera i, with residual
// error(S_ji o S_i o S_j^-1). Levenberg-Marquardt with numeric 7x14 Jacobians and a sparse block
// Hessian solved by block-Jacobi preconditioned conjugate gradients (hundreds of nodes, thousands
// of edges). Removes the low-frequency error (scale drift, loop tension) that window BAs cannot.
"use strict";

import { compose, inverse, perturb, error } from './sim3.js';
import { choleskySolve } from './linalg.js';

const MAX_ITERATIONS = 30, CG_ITERATIONS = 300;

// nodes: [Sim3] (updated in place; node 0 fixes the gauge), edges: [{ i, j, S (S_ji), weight }].
// Returns { initialCost, cost, iterations }.
export function optimizePoseGraph(nodes, edges) {
  const n = nodes.length;
  const free = new Int32Array(n).fill(-1);
  let nf = 0;
  for (let i = 1; i < n; i++) free[i] = nf++;
  const residual = (Si, Sj, e) => error(compose(compose(e.S, Si), inverse(Sj)));
  const cost = (N) => edges.reduce((c, e) => c + (e.weight ?? 1) * residual(N[e.i], N[e.j], e).reduce((s, x) => s + x * x, 0), 0);

  let current = cost(nodes);
  const initialCost = current;
  let lambda = 1e-4, it = 0;
  const h = 1e-7;
  for (; it < MAX_ITERATIONS; it++) {
    // Linearize: per edge, r (7) and J_i, J_j (7 x 7 each, numeric, left perturbations)
    const diag = new Map(), off = new Map(), g = new Float64Array(7 * nf);
    const addBlock = (map, key, A, B, w) => {
      let M = map.get(key);
      if (!M) { M = new Float64Array(49); map.set(key, M); }
      for (let r = 0; r < 7; r++) for (let c = 0; c < 7; c++) {
        let s = 0;
        for (let k = 0; k < 7; k++) s += A[k * 7 + r] * B[k * 7 + c];
        M[r * 7 + c] += w * s;
      }
    };
    for (const e of edges) {
      const w = e.weight ?? 1;
      const r = residual(nodes[e.i], nodes[e.j], e);
      const J = {};
      for (const v of [e.i, e.j]) {
        if (free[v] < 0) continue;
        const Jv = new Float64Array(49);  // row-major 7 (residual) x 7 (parameter)
        const at = (d) => { const Sv = perturb(nodes[v], d); return residual(v === e.i ? Sv : nodes[e.i], v === e.j ? Sv : nodes[e.j], e); };
        for (let p = 0; p < 7; p++) {
          const d = new Array(7).fill(0);
          d[p] = h; const rp = at(d);
          d[p] = -h; const rm = at(d);
          for (let k = 0; k < 7; k++) Jv[k * 7 + p] = (rp[k] - rm[k]) / (2 * h);
        }
        J[v] = Jv;
        for (let p = 0; p < 7; p++) {
          let s = 0;
          for (let k = 0; k < 7; k++) s += Jv[k * 7 + p] * r[k];
          g[7 * free[v] + p] += w * s;
        }
      }
      for (const a of [e.i, e.j]) {
        if (!J[a]) continue;
        addBlock(diag, free[a], J[a], J[a], w);
        for (const b of [e.i, e.j]) if (b !== a && J[b]) addBlock(off, `${free[a]},${free[b]}`, J[a], J[b], w);
      }
    }
    const offList = [...off.entries()].map(([k, M]) => { const [a, b] = k.split(',').map(Number); return { a, b, M }; });

    let accepted = false;
    for (let tries = 0; tries < 8 && !accepted; tries++) {
      // Damped block diagonal and its inverse (preconditioner)
      const D = new Map(), Pinv = new Map();
      for (const [a, M] of diag) {
        const Md = Float64Array.from(M);
        for (let k = 0; k < 7; k++) Md[k * 8] += lambda * Math.max(M[k * 8], 1e-9);
        D.set(a, Md);
        const inv = new Float64Array(49);
        for (let c = 0; c < 7; c++) {
          const ec = new Float64Array(7); ec[c] = 1;
          const col = choleskySolve(Md, ec, 7);
          if (col) for (let r = 0; r < 7; r++) inv[r * 7 + c] = col[r];
        }
        Pinv.set(a, inv);
      }
      const matvec = (x) => {
        const y = new Float64Array(7 * nf);
        for (const [a, M] of D) for (let r = 0; r < 7; r++) { let s = 0; for (let c = 0; c < 7; c++) s += M[r * 7 + c] * x[7 * a + c]; y[7 * a + r] += s; }
        for (const { a, b, M } of offList) for (let r = 0; r < 7; r++) { let s = 0; for (let c = 0; c < 7; c++) s += M[r * 7 + c] * x[7 * b + c]; y[7 * a + r] += s; }
        return y;
      };
      const precond = (x) => {
        const y = new Float64Array(7 * nf);
        for (const [a, M] of Pinv) for (let r = 0; r < 7; r++) { let s = 0; for (let c = 0; c < 7; c++) s += M[r * 7 + c] * x[7 * a + c]; y[7 * a + r] = s; }
        return y;
      };
      const dx = pcg(matvec, precond, g.map(v => -v), CG_ITERATIONS);
      const cand = nodes.map((N, v) => free[v] < 0 ? N : perturb(N, Array.from(dx.subarray(7 * free[v], 7 * free[v] + 7))));
      const c = cost(cand);
      if (c < current) {
        const rel = (current - c) / Math.max(current, 1e-300);
        for (let v = 0; v < n; v++) nodes[v] = cand[v];
        current = c;
        lambda = Math.max(1e-10, lambda / 3);
        accepted = true;
        if (rel < 1e-9) return { initialCost, cost: current, iterations: it + 1 };
      } else lambda *= 10;
    }
    if (!accepted) break;
  }
  return { initialCost, cost: current, iterations: it };
}

// Preconditioned conjugate gradients for A x = b
function pcg(matvec, precond, b, maxIterations, tol = 1e-10) {
  const x = new Float64Array(b.length);
  let r = Float64Array.from(b);
  let z = precond(r), p = Float64Array.from(z);
  let rz = dot(r, z);
  const b2 = dot(b, b);
  for (let k = 0; k < maxIterations && dot(r, r) > tol * tol * b2; k++) {
    const Ap = matvec(p);
    const alpha = rz / (dot(p, Ap) || 1e-300);
    for (let i = 0; i < x.length; i++) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
    z = precond(r);
    const rzNew = dot(r, z);
    const beta = rzNew / (rz || 1e-300);
    for (let i = 0; i < p.length; i++) p[i] = z[i] + beta * p[i];
    rz = rzNew;
  }
  return x;
}

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
