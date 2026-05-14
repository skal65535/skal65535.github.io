// matching/geometry.js
// Structure-from-motion bits that opencv.js's stripped WASM build doesn't expose:
// normalized 8-point F estimation + RANSAC, E = K^T F K, E decomposition,
// linear DLT triangulation, and chirality (cheirality) test.
//
// Matrices are flat row-major Float64Array (9 = 3×3, 12 = 3×4, 16 = 4×4).
// Points are plain {x, y} in pixel coords.

// ── Linear algebra ─────────────────────────────────────────────────────

function m3mul(A, B) {
  const C = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += A[i*3+k] * B[k*3+j];
      C[i*3+j] = s;
    }
  return C;
}
function m3T(A) {
  return new Float64Array([A[0],A[3],A[6], A[1],A[4],A[7], A[2],A[5],A[8]]);
}
function m3det(A) {
  return A[0]*(A[4]*A[8]-A[5]*A[7])
       - A[1]*(A[3]*A[8]-A[5]*A[6])
       + A[2]*(A[3]*A[7]-A[4]*A[6]);
}

// Jacobi eigendecomposition for symmetric n×n. V's columns are eigenvectors.
// Quadratically convergent — 60 sweeps is plenty for n ≤ 9.
function jacobiEigen(Ain, n, maxSweeps = 60) {
  const A = new Float64Array(Ain);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i*n+i] = 1;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let maxOff = 0, p = 0, q = 1;
    for (let i = 0; i < n; i++)
      for (let j = i+1; j < n; j++) {
        const a = Math.abs(A[i*n+j]);
        if (a > maxOff) { maxOff = a; p = i; q = j; }
      }
    if (maxOff < 1e-14) break;
    const app = A[p*n+p], aqq = A[q*n+q], apq = A[p*n+q];
    const theta = (aqq - app) / (2 * apq);
    const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(1 + theta*theta));
    const c = 1 / Math.sqrt(1 + t*t);
    const s = t * c;
    A[p*n+p] = app - t*apq;
    A[q*n+q] = aqq + t*apq;
    A[p*n+q] = A[q*n+p] = 0;
    for (let i = 0; i < n; i++) {
      if (i !== p && i !== q) {
        const aip = A[i*n+p], aiq = A[i*n+q];
        A[i*n+p] = A[p*n+i] = c*aip - s*aiq;
        A[i*n+q] = A[q*n+i] = s*aip + c*aiq;
      }
      const vip = V[i*n+p], viq = V[i*n+q];
      V[i*n+p] = c*vip - s*viq;
      V[i*n+q] = s*vip + c*viq;
    }
  }
  const evals = new Float64Array(n);
  for (let i = 0; i < n; i++) evals[i] = A[i*n+i];
  return { evals, V };
}

// Right null vector of N×9 → smallest eigvec of A^T A.
function nullspace9(rows) {
  const AtA = new Float64Array(81);
  for (const r of rows)
    for (let i = 0; i < 9; i++)
      for (let j = 0; j < 9; j++)
        AtA[i*9+j] += r[i] * r[j];
  const { evals, V } = jacobiEigen(AtA, 9);
  let m = 0;
  for (let i = 1; i < 9; i++) if (evals[i] < evals[m]) m = i;
  const v = new Float64Array(9);
  for (let i = 0; i < 9; i++) v[i] = V[i*9+m];
  return v;
}

// 3×3 SVD: M = U · diag(sigma) · V^T, sigma descending.
function svd3(M) {
  const MtM = m3mul(m3T(M), M);
  const e = jacobiEigen(MtM, 3);
  const order = [0,1,2].sort((a,b) => e.evals[b] - e.evals[a]);
  const sigma = order.map(i => Math.sqrt(Math.max(0, e.evals[i])));
  const V = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      V[i*3+j] = e.V[i*3+order[j]];
  // U = M · V · Σ^-1
  const U = new Float64Array(9);
  for (let j = 0; j < 3; j++) {
    if (sigma[j] < 1e-12) continue;
    for (let i = 0; i < 3; i++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += M[i*3+k] * V[k*3+j];
      U[i*3+j] = s / sigma[j];
    }
  }
  return { U, sigma, V };
}

// ── 8-point algorithm with Hartley normalization ─────────────────────────

function normalizePts(pts) {
  let mx = 0, my = 0;
  for (const p of pts) { mx += p.x; my += p.y; }
  mx /= pts.length; my /= pts.length;
  let mean = 0;
  for (const p of pts) mean += Math.hypot(p.x - mx, p.y - my);
  mean /= pts.length;
  const s = Math.SQRT2 / Math.max(mean, 1e-9);
  const T = new Float64Array([s,0,-s*mx,  0,s,-s*my,  0,0,1]);
  const q = pts.map(p => ({ x: s*(p.x-mx), y: s*(p.y-my) }));
  return { T, q };
}

function eightPoint(p1, p2) {
  const { T: T1, q: q1 } = normalizePts(p1);
  const { T: T2, q: q2 } = normalizePts(p2);
  const rows = new Array(q1.length);
  for (let i = 0; i < q1.length; i++) {
    const a = q1[i], b = q2[i];
    rows[i] = [a.x*b.x, a.y*b.x, b.x, a.x*b.y, a.y*b.y, b.y, a.x, a.y, 1];
  }
  const f = nullspace9(rows);
  let F = new Float64Array(f);
  // Enforce rank 2.
  const { U, sigma, V } = svd3(F);
  sigma[2] = 0;
  const UD = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) UD[i*3+j] = U[i*3+j] * sigma[j];
  F = m3mul(UD, m3T(V));
  // Undo normalization.
  return m3mul(m3mul(m3T(T2), F), T1);
}

// Sampson² distance for one match.
function sampson(F, p1, p2) {
  const f0 = F[0]*p1.x + F[1]*p1.y + F[2];
  const f1 = F[3]*p1.x + F[4]*p1.y + F[5];
  const f2 = F[6]*p1.x + F[7]*p1.y + F[8];
  const g0 = F[0]*p2.x + F[3]*p2.y + F[6];
  const g1 = F[1]*p2.x + F[4]*p2.y + F[7];
  const num = p2.x*f0 + p2.y*f1 + f2;
  return (num * num) / (f0*f0 + f1*f1 + g0*g0 + g1*g1 + 1e-12);
}

function ransacF(p1, p2, iters = 1000, thresh = 1.5) {
  const N = p1.length;
  if (N < 8) return null;
  const thr2 = thresh * thresh;
  let bestF = null, bestMask = null, bestCount = 0;
  const seen = new Uint8Array(N);
  for (let it = 0; it < iters; it++) {
    seen.fill(0);
    const idx = [];
    while (idx.length < 8) {
      const k = (Math.random() * N) | 0;
      if (!seen[k]) { seen[k] = 1; idx.push(k); }
    }
    let F;
    try { F = eightPoint(idx.map(i => p1[i]), idx.map(i => p2[i])); }
    catch { continue; }
    if (!F || !isFinite(F[0])) continue;
    let cnt = 0;
    const mask = new Uint8Array(N);
    for (let i = 0; i < N; i++)
      if (sampson(F, p1[i], p2[i]) < thr2) { cnt++; mask[i] = 1; }
    if (cnt > bestCount) { bestCount = cnt; bestF = F; bestMask = mask; }
  }
  if (!bestF || bestCount < 8) return null;
  // Refit on inliers.
  const in1 = [], in2 = [];
  for (let i = 0; i < N; i++) if (bestMask[i]) { in1.push(p1[i]); in2.push(p2[i]); }
  try {
    const Fref = eightPoint(in1, in2);
    if (Fref && isFinite(Fref[0])) bestF = Fref;
  } catch {}
  return { F: bestF, mask: bestMask, inliers: bestCount };
}

// ── E decomposition + chirality ─────────────────────────────────────────

function decomposeE(E) {
  const { U, V } = svd3(E);
  // E has rank 2 → sigma_3 = 0 → svd3 leaves U[:,2] and V[:,2] as zero (we
  // can't compute them via M·V/sigma). Recover them as the cross product of
  // the first two columns — that's the orthonormal completion. Without this
  // patch, t = U[:,2] comes out as the zero vector and all cameras collapse
  // to the origin.
  U[2] = U[3]*U[7] - U[6]*U[4];
  U[5] = U[6]*U[1] - U[0]*U[7];
  U[8] = U[0]*U[4] - U[3]*U[1];
  V[2] = V[3]*V[7] - V[6]*V[4];
  V[5] = V[6]*V[1] - V[0]*V[7];
  V[8] = V[0]*V[4] - V[3]*V[1];
  // Force det(U) = det(V) = +1 so R candidates are proper rotations.
  if (m3det(U) < 0) for (let i = 0; i < 3; i++) U[i*3+2] = -U[i*3+2];
  if (m3det(V) < 0) for (let i = 0; i < 3; i++) V[i*3+2] = -V[i*3+2];
  const W  = new Float64Array([0,-1,0,  1,0,0,  0,0,1]);
  const Wt = m3T(W);
  const Vt = m3T(V);
  const R1 = m3mul(m3mul(U, W),  Vt);
  const R2 = m3mul(m3mul(U, Wt), Vt);
  const t  = new Float64Array([ U[2],  U[5],  U[8]]);
  const tn = new Float64Array([-U[2], -U[5], -U[8]]);
  return [{R:R1,t}, {R:R1,t:tn}, {R:R2,t}, {R:R2,t:tn}];
}

// Linear DLT triangulation: 4D homogeneous X.
function triangulate(P1, P2, p1, p2) {
  const A = [
    [p1.x*P1[8]-P1[0],  p1.x*P1[9]-P1[1],  p1.x*P1[10]-P1[ 2],  p1.x*P1[11]-P1[ 3]],
    [p1.y*P1[8]-P1[4],  p1.y*P1[9]-P1[5],  p1.y*P1[10]-P1[ 6],  p1.y*P1[11]-P1[ 7]],
    [p2.x*P2[8]-P2[0],  p2.x*P2[9]-P2[1],  p2.x*P2[10]-P2[ 2],  p2.x*P2[11]-P2[ 3]],
    [p2.y*P2[8]-P2[4],  p2.y*P2[9]-P2[5],  p2.y*P2[10]-P2[ 6],  p2.y*P2[11]-P2[ 7]],
  ];
  const AtA = new Float64Array(16);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += A[k][i] * A[k][j];
      AtA[i*4+j] = s;
    }
  const { evals, V } = jacobiEigen(AtA, 4);
  let m = 0;
  for (let i = 1; i < 4; i++) if (evals[i] < evals[m]) m = i;
  return [V[0*4+m], V[1*4+m], V[2*4+m], V[3*4+m]];
}

function buildP1(K) {
  return new Float64Array([
    K[0], K[1], K[2], 0,
    K[3], K[4], K[5], 0,
    K[6], K[7], K[8], 0,
  ]);
}
function buildP2(K, R, t) {
  const P2 = new Float64Array(12);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += K[i*3+k] * R[k*3+j];
      P2[i*4+j] = s;
    }
    let st = 0;
    for (let k = 0; k < 3; k++) st += K[i*3+k] * t[k];
    P2[i*4+3] = st;
  }
  return P2;
}

function pickPose(cands, K, p1s, p2s) {
  const P1 = buildP1(K);
  let best = cands[0], bestCount = -1;
  const N = Math.min(p1s.length, 50);
  for (const c of cands) {
    const P2 = buildP2(K, c.R, c.t);
    let cnt = 0;
    for (let i = 0; i < N; i++) {
      const X = triangulate(P1, P2, p1s[i], p2s[i]);
      const w = X[3];
      if (Math.abs(w) < 1e-12) continue;
      const x0 = X[0]/w, y0 = X[1]/w, z0 = X[2]/w;
      const z2 = c.R[6]*x0 + c.R[7]*y0 + c.R[8]*z0 + c.t[2];
      if (z0 > 0 && z2 > 0) cnt++;
    }
    if (cnt > bestCount) { bestCount = cnt; best = c; }
  }
  return best;
}


// ── Public API ──────────────────────────────────────────────────────────
//
// recoverPose: parallel point arrays + image size → pose + RANSAC mask + K.
//   Triangulation is *not* done here so the caller can apply a scale fix
//   between pairs before triangulating.
export function recoverPose(p1, p2, width, height, opts = {}) {
  if (p1.length < 8 || p2.length < 8) return null;
  const { ransacThresh = 1.5, ransacIters = 1000 } = opts;
  const res = ransacF(p1, p2, ransacIters, ransacThresh);
  if (!res) return null;

  const f = Math.max(width, height);
  const K = new Float64Array([f,0,width/2,  0,f,height/2,  0,0,1]);
  const E = m3mul(m3mul(m3T(K), res.F), K);
  const cands = decomposeE(E);

  const in1 = [], in2 = [];
  for (let i = 0; i < p1.length; i++) if (res.mask[i]) { in1.push(p1[i]); in2.push(p2[i]); }
  const pose = pickPose(cands, K, in1, in2);
  return { R: pose.R, t: pose.t, mask: res.mask, K, inliers: res.inliers };
}

// Per-input triangulation in cam_i's frame. Returns one [x, y, z] per input
// pair, or null for points behind either camera / numerically degenerate.
// Caller is responsible for inlier filtering (pass only inliers in / handle
// nulls out).
export function triangulate2View(K, R, t, p1s, p2s) {
  const P1 = buildP1(K);
  const P2 = buildP2(K, R, t);
  const out = new Array(p1s.length);
  for (let i = 0; i < p1s.length; i++) {
    const X = triangulate(P1, P2, p1s[i], p2s[i]);
    const w = X[3];
    if (Math.abs(w) < 1e-12) { out[i] = null; continue; }
    const x = X[0]/w, y = X[1]/w, z = X[2]/w;
    if (!isFinite(x) || !isFinite(y) || !isFinite(z)) { out[i] = null; continue; }
    const z2 = R[6]*x + R[7]*y + R[8]*z + t[2];
    out[i] = (z > 0 && z2 > 0) ? [x, y, z] : null;
  }
  return out;
}
