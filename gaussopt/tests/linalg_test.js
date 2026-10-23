import { svd, nullVector, choleskySolve, matMul, transpose, expSO3, logSO3, rotationAngle,
         projectToSO3, mul3, transpose3, det3 } from '../src/sfm/linalg.js';
import { assert, assertClose, rng, randomMatrix, randomRotation } from './util.js';

const eye = (n) => Array.from({ length: n * n }, (_, i) => (i % (n + 1) === 0 ? 1 : 0));

Deno.test('svd reconstructs tall, square and wide matrices', () => {
  const r = rng(1);
  for (const [m, n] of [[3, 3], [9, 4], [8, 9], [2, 5]]) {
    for (let trial = 0; trial < 20; trial++) {
      const A = randomMatrix(r, m * n);
      const { U, S, V } = svd(A, m, n);
      for (let k = 1; k < n; k++) assert(S[k - 1] >= S[k], 'singular values descending');
      const US = Float64Array.from(U, (u, idx) => u * S[idx % n]);
      assertClose(matMul(US, transpose(V, n, n), m, n, n), A, 1e-10, `A = U S V^T (${m}x${n})`);
      assertClose(matMul(transpose(V, n, n), V, n, n, n), eye(n), 1e-10, 'V orthonormal');
    }
  }
});

Deno.test('nullVector finds the null space of a wide system', () => {
  const r = rng(2);
  const x = randomMatrix(r, 9);
  const nx = Math.hypot(...x);
  const A = new Float64Array(8 * 9);
  for (let i = 0; i < 8; i++) {  // rows orthogonal to x
    const row = randomMatrix(r, 9);
    const d = row.reduce((s, v, k) => s + v * x[k], 0) / (nx * nx);
    for (let k = 0; k < 9; k++) A[i * 9 + k] = row[k] - d * x[k];
  }
  const v = nullVector(A, 8, 9);
  const sign = Math.sign(v.reduce((s, vi, k) => s + vi * x[k], 0));
  assertClose(v.map(vi => vi * sign), Array.from(x, xi => xi / nx), 1e-9, 'null vector');
});

Deno.test('choleskySolve solves SPD systems and rejects indefinite ones', () => {
  const r = rng(4);
  const n = 6;
  const B = randomMatrix(r, n * n);
  const A = matMul(B, transpose(B, n, n), n, n, n);
  for (let i = 0; i < n; i++) A[i * n + i] += 0.1;
  const x = randomMatrix(r, n);
  const b = matMul(A, x, n, n, 1);
  assertClose(choleskySolve(A, b, n), x, 1e-9, 'solution');
  assert(choleskySolve([1, 2, 2, 1], [1, 1], 2) === null, 'indefinite rejected');
});

Deno.test('SO(3) exp / log round trip, including near 0 and near pi', () => {
  const r = rng(5);
  const cases = [[0, 0, 0], [1e-9, 0, 0], [0, 0, Math.PI - 1e-7], [Math.PI / Math.sqrt(2), -Math.PI / Math.sqrt(2), 0]];
  for (let i = 0; i < 200; i++) {
    const w = [r() - 0.5, r() - 0.5, r() - 0.5];
    const n = Math.hypot(...w), a = r() * Math.PI * 0.999;
    cases.push(w.map(x => x / n * a));
  }
  for (const w of cases) {
    const R = expSO3(w);
    assertClose(mul3(R, transpose3(R)), eye(3), 1e-12, 'orthonormal');
    assertClose(det3(R), 1, 1e-12, 'det');
    assertClose(expSO3(logSO3(R)), R, 1e-6, `exp(log(R)) for |w| = ${Math.hypot(...w)}`);
  }
});

Deno.test('rotationAngle and projectToSO3', () => {
  const r = rng(6);
  const R = randomRotation(r);
  assertClose(rotationAngle(mul3(expSO3([0, 0, 0.3]), R), R), 0.3, 1e-12, 'relative angle');
  const noisy = R.map(x => x + 1e-3 * (r() - 0.5));
  const P = projectToSO3(noisy);
  assertClose(det3(P), 1, 1e-12, 'det');
  assert(rotationAngle(P, R) < 2e-3, 'close to the original');
  const reflection = R.map((x, i) => (i % 3 === 2 ? -x : x));  // det = -1 input
  assertClose(det3(projectToSO3(reflection)), 1, 1e-12, 'reflection mapped to a rotation');
});
