// linalg.js — Small dense linear algebra for the SfM solvers (plain JS, f64, no dependencies):
//   - m x n matrices as flat row-major arrays with explicit dimensions
//   - one-sided Jacobi SVD (full V, so null spaces of wide systems are available)
//   - Cholesky solve
//   - 3x3 helpers, SO(3) exp / log maps, projection onto SO(3)
// Conventions follow math.js: 3x3 matrices are flat row-major, vectors are plain arrays.
"use strict";

// ── General m x n ────────────────────────────────────────────────────────────

export function matMul(A, B, m, k, n) {  // (m x k) * (k x n)
  const C = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i * k + p];
      if (a === 0) continue;
      for (let j = 0; j < n; j++) C[i * n + j] += a * B[p * n + j];
    }
  }
  return C;
}

export function transpose(A, m, n) {
  const T = new Float64Array(m * n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) T[j * m + i] = A[i * n + j];
  return T;
}

// A = U diag(S) V^T, S descending. U is m x n (thin), V is n x n (full).
// Wide inputs (m < n) are padded with zero rows, so V still spans the null space.
export function svd(A, m, n, { maxSweeps = 60, eps = 1e-15 } = {}) {
  const rows = Math.max(m, n);
  const U = new Float64Array(rows * n);
  U.set(A.subarray ? A.subarray(0, m * n) : A.slice(0, m * n));
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let rotated = false;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        let alpha = 0, beta = 0, gamma = 0;
        for (let i = 0; i < rows; i++) {
          const up = U[i * n + p], uq = U[i * n + q];
          alpha += up * up; beta += uq * uq; gamma += up * uq;
        }
        if (gamma === 0 || Math.abs(gamma) <= eps * Math.sqrt(alpha * beta)) continue;
        rotated = true;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t), s = c * t;
        for (let i = 0; i < rows; i++) {
          const up = U[i * n + p], uq = U[i * n + q];
          U[i * n + p] = c * up - s * uq;
          U[i * n + q] = s * up + c * uq;
        }
        for (let i = 0; i < n; i++) {
          const vp = V[i * n + p], vq = V[i * n + q];
          V[i * n + p] = c * vp - s * vq;
          V[i * n + q] = s * vp + c * vq;
        }
      }
    }
    if (!rotated) break;
  }

  const S = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let s2 = 0;
    for (let i = 0; i < rows; i++) s2 += U[i * n + j] ** 2;
    S[j] = Math.sqrt(s2);
  }
  const order = [...S.keys()].sort((a, b) => S[b] - S[a]);
  const Us = new Float64Array(m * n), Vs = new Float64Array(n * n), Ss = new Float64Array(n);
  order.forEach((src, dst) => {
    Ss[dst] = S[src];
    const inv = S[src] > 0 ? 1 / S[src] : 0;
    for (let i = 0; i < m; i++) Us[i * n + dst] = U[i * n + src] * inv;
    for (let i = 0; i < n; i++) Vs[i * n + dst] = V[i * n + src];
  });
  return { U: Us, S: Ss, V: Vs };
}

// Unit vector x minimizing |A x| (last right singular vector)
export function nullVector(A, m, n) {
  const { V } = svd(A, m, n);
  return Array.from({ length: n }, (_, i) => V[i * n + n - 1]);
}

// Solve A x = b for symmetric positive-definite n x n A. Returns null if A is not SPD.
export function choleskySolve(A, b, n) {
  const L = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    let d = A[j * n + j];
    for (let k = 0; k < j; k++) d -= L[j * n + k] ** 2;
    if (!(d > 0)) return null;
    const ljj = Math.sqrt(d);
    L[j * n + j] = ljj;
    for (let i = j + 1; i < n; i++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      L[i * n + j] = s / ljj;
    }
  }
  const x = Float64Array.from(b);
  for (let i = 0; i < n; i++) {  // L y = b
    for (let k = 0; k < i; k++) x[i] -= L[i * n + k] * x[k];
    x[i] /= L[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {  // L^T x = y
    for (let k = i + 1; k < n; k++) x[i] -= L[k * n + i] * x[k];
    x[i] /= L[i * n + i];
  }
  return x;
}

// ── 3x3 ──────────────────────────────────────────────────────────────────────

export const mul3 = (A, B) => Array.from(matMul(A, B, 3, 3, 3));
export const transpose3 = (A) => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
export const mulVec3 = (A, v) => [
  A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
  A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
  A[6] * v[0] + A[7] * v[1] + A[8] * v[2],
];
export const det3 = (A) =>
  A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
const skew3 = ([x, y, z]) => [0, -z, y, z, 0, -x, -y, x, 0];

// Axis-angle vector w (|w| = angle) -> rotation matrix (Rodrigues)
export function expSO3(w) {
  const th = Math.hypot(w[0], w[1], w[2]);
  const K = skew3(w);
  const K2 = mul3(K, K);
  // sin(th)/th and (1 - cos(th))/th^2, with Taylor expansions near 0
  const a = th < 1e-6 ? 1 - th * th / 6 : Math.sin(th) / th;
  const b = th < 1e-6 ? 0.5 - th * th / 24 : (1 - Math.cos(th)) / (th * th);
  return Array.from({ length: 9 }, (_, i) => (i % 4 === 0 ? 1 : 0) + a * K[i] + b * K2[i]);
}

// Rotation matrix -> axis-angle vector (inverse of expSO3, angle in [0, pi])
export function logSO3(R) {
  const c = Math.min(1, Math.max(-1, (R[0] + R[4] + R[8] - 1) / 2));
  const th = Math.acos(c);
  const v = [R[7] - R[5], R[2] - R[6], R[3] - R[1]];
  if (th < 1e-6) return [v[0] / 2, v[1] / 2, v[2] / 2];
  if (Math.PI - th > 1e-4) {
    const k = th / (2 * Math.sin(th));
    return [v[0] * k, v[1] * k, v[2] * k];
  }
  // Near pi: axis from the largest diagonal entry of (R + I) / 2 = a a^T
  const d = [R[0], R[4], R[8]];
  const i = d[0] >= d[1] && d[0] >= d[2] ? 0 : d[1] >= d[2] ? 1 : 2;
  const col = [0, 1, 2].map(r => (R[r * 3 + i] + (r === i ? 1 : 0)) / 2);
  const n = Math.hypot(...col) || 1;
  let axis = col.map(x => x / n);
  if (axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2] < 0) axis = axis.map(x => -x);
  return axis.map(x => x * th);
}

// Angle (radians) of the relative rotation Ra Rb^T. atan2(sin, cos) keeps full precision near 0,
// where acos((trace - 1) / 2) loses half the digits.
export function rotationAngle(Ra, Rb) {
  const M = mul3(Ra, transpose3(Rb));
  const s = Math.hypot(M[7] - M[5], M[2] - M[6], M[3] - M[1]) / 2;
  return Math.atan2(s, (M[0] + M[4] + M[8] - 1) / 2);
}

// Closest rotation matrix (Frobenius norm) to a 3x3 matrix
export function projectToSO3(M) {
  const { U, V } = svd(M, 3, 3);
  const R = Array.from(matMul(U, transpose(V, 3, 3), 3, 3, 3));
  if (det3(R) >= 0) return R;
  const U2 = Float64Array.from(U);
  for (let i = 0; i < 3; i++) U2[i * 3 + 2] = -U2[i * 3 + 2];
  return Array.from(matMul(U2, transpose(V, 3, 3), 3, 3, 3));
}
