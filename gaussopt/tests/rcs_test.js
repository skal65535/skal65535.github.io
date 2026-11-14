// The reduced camera system (src/sfm/rcs.js): its blocks solved by preconditioned conjugate gradients past denseMax,
// the same solution as the dense Cholesky.
import { ReducedSystem } from '../src/sfm/rcs.js';
import { assert, rng } from './util.js';

// A system shaped as a video's: poses (6) coupled to their next `band` in time and by a few long links (revisits),
// a camera model's intrinsics (2) coupled to every pose; J^T J of random rows, so positive definite. Filled into
// `sys` through at(), lower blocks only. Returns its dense full matrix for checking.
function fill(sys, r, { poses, band = 8, links = 20 }) {
  const m = sys.m, full = new Float64Array(m * m), blk = (k) => k < poses ? k : poses;
  const pairs = [];
  for (let a = 0; a < poses; a++) for (let b = Math.max(0, a - band); b < a; b++) pairs.push([a, b]);
  for (let l = 0; l < links; l++) { const a = Math.floor(r() * poses), b = Math.floor(r() * poses); if (a !== b) pairs.push([a, b]); }
  const rows = [];
  for (const [a, b] of pairs) for (let n = 0; n < 4; n++) rows.push([a, b]);  // a few rows of each coupling
  for (let a = 0; a < poses; a++) for (let n = 0; n < 8; n++) rows.push([a, a]);
  for (const [a, b] of rows) {
    // a row touching poses a, b and the intrinsics
    const idx = [], val = [];
    for (const k of new Set([a, b, poses])) for (let i = 0; i < sys.size[blk(k)]; i++) { idx.push(sys.off[blk(k)] + i); val.push(2 * r() - 1); }
    for (let x = 0; x < idx.length; x++) for (let y = 0; y < idx.length; y++) full[idx[x] * m + idx[y]] += val[x] * val[y];
  }
  for (let i = 0; i < m; i++) full[i * m + i] += 1e-3;
  const blockOf = (i) => { let a = 0; while (sys.off[a + 1] <= i) a++; return a; };
  for (let i = 0; i < m; i++) for (let j = 0; j <= i; j++) {
    if (full[i * m + j] === 0) continue;
    const a = blockOf(i), b = blockOf(j), [A, base, stride] = sys.at(a, b);
    A[base + (i - sys.off[a]) * stride + (j - sys.off[b])] = full[i * m + j];
  }
  return full;
}

Deno.test('reduced system: conjugate gradients on the blocks solve as the dense Cholesky does', () => {
  for (const [poses, seed] of [[60, 1], [150, 2]]) {
    const sizes = [...new Array(poses).fill(6), 2];
    const r1 = rng(seed), r2 = rng(seed);
    const dense = new ReducedSystem(sizes, { denseMax: Infinity }), sparse = new ReducedSystem(sizes, { denseMax: 0, tol: 1e-8 });  // solved to tol
    const full = fill(dense, r1, { poses }); fill(sparse, r2, { poses });
    assert(dense.dense && !sparse.dense && sparse.blocks.size < (poses + 1) * (poses + 2) / 4, `${sparse.blocks.size} blocks`);
    const rb = rng(seed + 10), b = Float64Array.from({ length: dense.m }, () => 2 * rb() - 1);
    const xd = dense.solve(b), xs = sparse.solve(b);
    let err = 0, nx = 0, res = 0;
    for (let i = 0; i < dense.m; i++) {
      err = Math.max(err, Math.abs(xs[i] - xd[i])); nx = Math.max(nx, Math.abs(xd[i]));
      let s = 0; for (let j = 0; j < dense.m; j++) s += full[i * dense.m + j] * xs[j];
      res = Math.max(res, Math.abs(s - b[i]));
    }
    assert(err <= 1e-6 * nx && res < 1e-6, `${poses} poses: sparse vs dense ${err} (|x| ${nx}), residual ${res}, ${sparse.iterations} iterations`);
    // the diagonal read and added the same both ways
    const d = dense.diagonal(), ds = sparse.diagonal();
    assert(d.every((v, i) => v === ds[i] && v === full[i * dense.m + i]), 'the diagonal');
  }
});

Deno.test('reduced system: a solve stopped at maxIterations says so (converged false), one that reaches tol does not', () => {
  const sizes = [...new Array(60).fill(6), 2];
  for (const [maxIterations, want] of [[3, false], [2000, true]]) {
    const s = new ReducedSystem(sizes, { denseMax: 0, maxIterations });
    fill(s, rng(5), { poses: 60 });
    const b = Float64Array.from({ length: s.m }, (_, i) => Math.sin(i));
    assert(s.solve(b) && s.converged === want, `maxIterations ${maxIterations}: ${s.iterations} iterations, converged ${s.converged}`);
  }
});

Deno.test('reduced system: a block without curvature fails the sparse solve, as the dense one', () => {
  const sizes = [6, 6, 6];
  for (const denseMax of [0, Infinity]) {
    const s = new ReducedSystem(sizes, { denseMax });
    for (const a of [0, 2]) { const [A, base, stride] = s.at(a, a); for (let i = 0; i < 6; i++) A[base + i * stride + i] = 1; }
    assert(s.solve(new Float64Array(18).fill(1)) === null, `denseMax ${denseMax}: solved`);
  }
});
