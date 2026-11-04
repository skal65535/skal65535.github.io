// ba_gpu.wgsl — The GPU passes of ba_gpu.js's bundle adjustment: per observation its linearization and the candidate's
// cost, per point its block inverse, its Schur rows and its back-substitution, and the reduced camera system's terms
// summed by chunks (a workgroup each, no atomics). A block of the reduced system is a free pose's (6) or a refined
// lens's intrinsics (ni <= 3, padded to 6); an observation has two halves, its pose's and its lens's Jacobian rows.

struct Params {
  loss: f32, lambda: f32, nObs: u32, nPts: u32,
  ni: u32, k0: u32, k1: u32, k2: u32,  // the refined intrinsics: [f, k1, k2] indices
  fixPoints: u32, _a: u32, _b: u32, _c: u32,
}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> lens: array<vec4f>;      // 2 per lens: fx fy cx cy, k1 k2 p1 p2
@group(0) @binding(2) var<storage, read> poses: array<vec4f>;     // 4 per pose: rows [R | t], (lens, -, -, -)
@group(0) @binding(3) var<storage, read> pts: array<vec4f>;
@group(0) @binding(4) var<storage, read> obs: array<vec4u>;       // pose, point, u, v (f32 bits)
// per observation (36): r (2), w, -, pose rows (2 x 6), intrinsics rows (2 x 6, ni used), Jp rows (2 x 3), -
@group(0) @binding(5) var<storage, read_write> J: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;     // per observation, per half: L^-1 W^T (3 x 6, by columns)
@group(0) @binding(7) var<storage, read> ptStart: array<u32>;
@group(0) @binding(8) var<storage, read> ptObs: array<u32>;
@group(0) @binding(9) var<storage, read> halfBlock: array<i32>;  // per half: its block, -1 if none
@group(0) @binding(10) var<storage, read_write> Vi: array<f32>;   // per point (16): L (6, V = L L^T), L^-1 gp (3), ok, -
@group(0) @binding(11) var<storage, read> chunks: array<vec2u>;  // [start, end) in terms
@group(0) @binding(12) var<storage, read> terms: array<vec2u>;   // half pairs (a, b)
@group(0) @binding(13) var<storage, read_write> S: array<f32>;    // per chunk (42): sum of its terms (6 x 6), the J^T w J diagonal (6)
@group(0) @binding(14) var<storage, read> gChunks: array<vec2u>;
@group(0) @binding(15) var<storage, read> gTerms: array<u32>;    // halves
@group(0) @binding(16) var<storage, read_write> G: array<f32>;    // per chunk (6)
@group(0) @binding(17) var<storage, read> dX: array<f32>;        // the step, 6 per block
@group(0) @binding(18) var<storage, read_write> ptsCand: array<vec4f>;
@group(0) @binding(19) var<storage, read> posesCand: array<vec4f>;
@group(0) @binding(20) var<storage, read> lensCand: array<vec4f>;
@group(0) @binding(21) var<storage, read_write> C: array<f32>;    // per 256 observations: cost, squares, count, -
@group(0) @binding(22) var<storage, read_write> PP: array<f32>;   // per 256 points: their gp^T V^-1 gp = |L^-1 gp|^2

fn apply(R0: vec4f, R1: vec4f, R2: vec4f, X: vec3f) -> vec3f { return vec3f(dot(R0.xyz, X) + R0.w, dot(R1.xyz, X) + R1.w, dot(R2.xyz, X) + R2.w); }
fn distort(k: vec4f, x: f32, y: f32) -> vec2f {
  let rr = x * x + y * y; let rad = 1.0 + k.x * rr + k.y * rr * rr;
  return vec2f(x * rad + 2.0 * k.z * x * y + k.w * (rr + 2.0 * x * x), y * rad + 2.0 * k.w * x * y + k.z * (rr + 2.0 * y * y));
}

// a half's Jacobian row (0, 1) entry i, its W = w Jh^T Jp row i, its L^-1 W^T column i
fn jh(o: u32, h: u32, row: u32, i: u32) -> f32 { return J[36u * o + 4u + 12u * h + 6u * row + i]; }
fn jp(o: u32, row: u32) -> vec3f { let k = 36u * o + 28u + 3u * row; return vec3f(J[k], J[k + 1u], J[k + 2u]); }
fn wRow(o: u32, h: u32, i: u32) -> vec3f { return J[36u * o + 2u] * (jh(o, h, 0u, i) * jp(o, 0u) + jh(o, h, 1u, i) * jp(o, 1u)); }
fn yRow(o: u32, h: u32, i: u32) -> vec3f { let k = 36u * o + 18u * h + 3u * i; return vec3f(Y[k], Y[k + 1u], Y[k + 2u]); }

@compute @workgroup_size(64) fn linearize(@builtin(global_invocation_id) gid: vec3u) {
  let o = gid.x;
  if (o >= P.nObs) { return; }
  let ob = obs[o]; let b = 36u * o; let l = bitcast<u32>(poses[4u * ob.x + 3u].x);
  let q = apply(poses[4u * ob.x], poses[4u * ob.x + 1u], poses[4u * ob.x + 2u], pts[ob.y].xyz);
  if (!(q.z > 1e-9)) { J[b + 2u] = 0.0; return; }
  let f = lens[2u * l]; let k = lens[2u * l + 1u];
  let iz = 1.0 / q.z; let x = q.x * iz; let y = q.y * iz; let rr = x * x + y * y;
  let rad = 1.0 + k.x * rr + k.y * rr * rr; let dr = 2.0 * k.x + 4.0 * k.y * rr;
  let d = distort(k, x, y);
  let a00 = f.x * (rad + x * dr * x + 2.0 * k.z * y + 6.0 * k.w * x); let a01 = f.x * (x * dr * y + 2.0 * k.z * x + 2.0 * k.w * y);
  let a10 = f.y * (y * dr * x + 2.0 * k.w * y + 2.0 * k.z * x); let a11 = f.y * (rad + y * dr * y + 2.0 * k.w * x + 6.0 * k.z * y);
  let d0 = vec3f(a00 * iz, a01 * iz, -(a00 * x + a01 * y) * iz); let d1 = vec3f(a10 * iz, a11 * iz, -(a10 * x + a11 * y) * iz);
  let r = vec2f(f.x * d.x + f.z, f.y * d.y + f.w) - bitcast<vec2f>(ob.zw);
  let s = dot(r, r);
  J[b] = r.x; J[b + 1u] = r.y; J[b + 2u] = select(P.loss / sqrt(s), 1.0, s <= P.loss * P.loss);
  let c0 = cross(q, d0); let c1 = cross(q, d1);
  let R0 = poses[4u * ob.x].xyz; let R1 = poses[4u * ob.x + 1u].xyz; let R2 = poses[4u * ob.x + 2u].xyz;
  let p0 = d0.x * R0 + d0.y * R1 + d0.z * R2; let p1 = d1.x * R0 + d1.y * R1 + d1.z * R2;
  // d(uv)/d[f, k1, k2] (fy keeps its ratio to fx)
  var dk = array<f32, 6>(d.x, f.x * x * rr, f.x * x * rr * rr, (f.y / f.x) * d.y, f.y * y * rr, f.y * y * rr * rr);
  var ki = array<u32, 3>(P.k0, P.k1, P.k2);
  for (var i = 0u; i < 3u; i++) {
    J[b + 4u + i] = c0[i]; J[b + 7u + i] = d0[i]; J[b + 10u + i] = c1[i]; J[b + 13u + i] = d1[i];
    J[b + 28u + i] = p0[i]; J[b + 31u + i] = p1[i];
  }
  for (var i = 0u; i < 6u; i++) {
    let used = i < P.ni;
    J[b + 16u + i] = select(0.0, dk[ki[min(i, 2u)]], used); J[b + 22u + i] = select(0.0, dk[3u + ki[min(i, 2u)]], used);
  }
}

// 3 x 3 SPD A's Cholesky factor (rows: l00; l10 l11; l20 l21 l22), false when a pivot is not positive
fn chol3(A: mat3x3f, L: ptr<function, array<f32, 6>>) -> bool {
  let l00 = sqrt(A[0][0]); let l10 = A[0][1] / l00; let l20 = A[0][2] / l00;
  let d1 = A[1][1] - l10 * l10; let l11 = sqrt(d1); let l21 = (A[1][2] - l20 * l10) / l11;
  let d2 = A[2][2] - l20 * l20 - l21 * l21;
  *L = array<f32, 6>(l00, l10, l11, l20, l21, sqrt(d2));
  return A[0][0] > 0.0 && d1 > 0.0 && d2 > 0.0;
}
fn lsolve(L: array<f32, 6>, b: vec3f) -> vec3f {  // L^-1 b
  let y0 = b.x / L[0]; let y1 = (b.y - L[1] * y0) / L[2];
  return vec3f(y0, y1, (b.z - L[3] * y0 - L[4] * y1) / L[5]);
}
fn ltsolve(L: array<f32, 6>, b: vec3f) -> vec3f {  // L^-T b
  let x2 = b.z / L[5]; let x1 = (b.y - L[4] * x2) / L[2];
  return vec3f((b.x - L[1] * x1 - L[3] * x2) / L[0], x1, x2);
}
fn factorOf(p: u32) -> array<f32, 6> { let v = 16u * p; return array<f32, 6>(Vi[v], Vi[v + 1u], Vi[v + 2u], Vi[v + 3u], Vi[v + 4u], Vi[v + 5u]); }

// per point: V = sum w Jp^T Jp damped, its Cholesky factor L (none when V is not positive or the points fixed: the point
// left as it is), gp, L^-1 gp, and L^-1 W^T of its halves (the Schur terms Gram products of them: V^-1 never formed, its
// f32 adjugate garbage for a far point's near-singular V)
@compute @workgroup_size(64) fn points(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.x;
  if (p >= P.nPts) { return; }
  var V = mat3x3f(); var g = vec3f();
  for (var k = ptStart[p]; k < ptStart[p + 1u]; k++) {
    let o = ptObs[k]; let w = J[36u * o + 2u];
    if (w == 0.0) { continue; }
    let j0 = jp(o, 0u); let j1 = jp(o, 1u);
    V += w * (mat3x3f(j0 * j0.x, j0 * j0.y, j0 * j0.z) + mat3x3f(j1 * j1.x, j1 * j1.y, j1 * j1.z));
    g += w * (j0 * J[36u * o] + j1 * J[36u * o + 1u]);
  }
  var D = mat3x3f();
  for (var i = 0; i < 3; i++) { D[i][i] = P.lambda * max(V[i][i], 1e-9); }
  V += D;
  var Lp: array<f32, 6>;
  let ok = chol3(V, &Lp) && P.fixPoints == 0u;
  let gy = select(vec3f(), lsolve(Lp, g), ok);
  for (var i = 0u; i < 6u; i++) { Vi[16u * p + i] = Lp[i]; }
  for (var i = 0u; i < 3u; i++) { Vi[16u * p + 6u + i] = gy[i]; }
  Vi[16u * p + 9u] = select(0.0, 1.0, ok);
  for (var k = ptStart[p]; k < ptStart[p + 1u]; k++) {
    let o = ptObs[k];
    for (var h = 0u; h < 2u; h++) {
      if (halfBlock[2u * o + h] < 0) { continue; }
      for (var i = 0u; i < 6u; i++) {
        let y = select(vec3f(), lsolve(Lp, wRow(o, h, i)), ok); let at = 36u * o + 18u * h + 3u * i;
        Y[at] = y.x; Y[at + 1u] = y.y; Y[at + 2u] = y.z;
      }
    }
  }
}

// per chunk of half pairs (a, b), each of one point: the reduced system's block (a's, b's) term, J_a^T w J_b when they
// are one observation's, minus (L^-1 W_a^T)^T L^-1 W_b^T (= W_a V^-1 W_b^T); and when a is b, the J^T w J diagonal (its
// damping's)
var<workgroup> sh: array<f32, 2688>;  // 64 x 42
@compute @workgroup_size(64) fn system(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let c = chunks[wid.x];
  var acc: array<f32, 42>;
  for (var k = c.x + lid; k < c.y; k += 64u) {
    let t = terms[k]; let oa = t.x >> 1u; let ha = t.x & 1u; let ob = t.y >> 1u; let hb = t.y & 1u;
    let same = oa == ob; let w = J[36u * oa + 2u];
    for (var i = 0u; i < 6u; i++) {
      let yi = yRow(oa, ha, i); let a0 = jh(oa, ha, 0u, i); let a1 = jh(oa, ha, 1u, i);
      for (var j = 0u; j < 6u; j++) {
        var v = -dot(yi, yRow(ob, hb, j));
        if (same) { v += w * (a0 * jh(ob, hb, 0u, j) + a1 * jh(ob, hb, 1u, j)); }
        acc[6u * i + j] += v;
      }
      if (t.x == t.y) { acc[36u + i] += w * (jh(oa, ha, 0u, i) * jh(oa, ha, 0u, i) + jh(oa, ha, 1u, i) * jh(oa, ha, 1u, i)); }
    }
  }
  for (var i = 0u; i < 42u; i++) { sh[42u * lid + i] = acc[i]; }
  workgroupBarrier();
  if (lid < 42u) { var s = 0.0; for (var k = 0u; k < 64u; k++) { s += sh[42u * k + lid]; } S[42u * wid.x + lid] = s; }
}

// per chunk of halves: their block's gradient, sum of J_h^T w r - (L^-1 W_h^T)^T L^-1 gp (= W_h V^-1 gp)
@compute @workgroup_size(64) fn gradient(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let c = gChunks[wid.x];
  var acc: array<f32, 6>;
  for (var k = c.x + lid; k < c.y; k += 64u) {
    let hh = gTerms[k]; let o = hh >> 1u; let h = hh & 1u; let w = J[36u * o + 2u];
    let p = obs[o].y; let gy = vec3f(Vi[16u * p + 6u], Vi[16u * p + 7u], Vi[16u * p + 8u]);
    for (var i = 0u; i < 6u; i++) { acc[i] += w * (jh(o, h, 0u, i) * J[36u * o] + jh(o, h, 1u, i) * J[36u * o + 1u]) - dot(yRow(o, h, i), gy); }
  }
  for (var i = 0u; i < 6u; i++) { sh[6u * lid + i] = acc[i]; }
  workgroupBarrier();
  if (lid < 6u) { var s = 0.0; for (var k = 0u; k < 64u; k++) { s += sh[6u * k + lid]; } G[6u * wid.x + lid] = s; }
}

// per point: dX = -V^-1 (gp + W^T dx) = -L^-T (L^-1 gp + L^-1 W^T dx), the candidate X + dX (X itself without L)
@compute @workgroup_size(64) fn back(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.x;
  if (p >= P.nPts) { return; }
  let v = 16u * p;
  if (Vi[v + 9u] == 0.0) { ptsCand[p] = pts[p]; return; }
  var rhs = vec3f(Vi[v + 6u], Vi[v + 7u], Vi[v + 8u]);
  for (var k = ptStart[p]; k < ptStart[p + 1u]; k++) {
    let o = ptObs[k];
    for (var h = 0u; h < 2u; h++) {
      let blk = halfBlock[2u * o + h];
      if (blk < 0) { continue; }
      for (var i = 0u; i < 6u; i++) { rhs += yRow(o, h, i) * dX[6u * u32(blk) + i]; }
    }
  }
  ptsCand[p] = vec4f(pts[p].xyz - ltsolve(factorOf(p), rhs), 0.0);
}

// per observation at the candidate: the Huber cost (behind the camera: rho(1e6)) and its squared residual, by 256
var<workgroup> shc: array<vec4f, 256>;
fn rho(s: f32) -> f32 { return select(2.0 * P.loss * sqrt(s) - P.loss * P.loss, s, s <= P.loss * P.loss); }
@compute @workgroup_size(256) fn cost(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wid: vec3u,
                                      @builtin(local_invocation_index) lid: u32) {
  let o = gid.x;
  var e = vec4f();
  if (o < P.nObs) {
    let ob = obs[o]; let l = bitcast<u32>(posesCand[4u * ob.x + 3u].x);
    let q = apply(posesCand[4u * ob.x], posesCand[4u * ob.x + 1u], posesCand[4u * ob.x + 2u], ptsCand[ob.y].xyz);
    if (q.z > 1e-9) {
      let f = lensCand[2u * l]; let d = distort(lensCand[2u * l + 1u], q.x / q.z, q.y / q.z);
      let r = vec2f(f.x * d.x + f.z, f.y * d.y + f.w) - bitcast<vec2f>(ob.zw); let s = dot(r, r);
      e = vec4f(rho(s), s, 1.0, 0.0);
    } else { e = vec4f(rho(1e6), 0.0, 0.0, 0.0); }
  }
  shc[lid] = e;
  for (var n = 128u; n > 0u; n >>= 1u) {
    workgroupBarrier();
    if (lid < n) { shc[lid] += shc[lid + n]; }
  }
  if (lid == 0u) { C[4u * wid.x] = shc[0].x; C[4u * wid.x + 1u] = shc[0].y; C[4u * wid.x + 2u] = shc[0].z; }
}

// per 256 points: the sum of their gp^T V^-1 gp, the decrease the points alone predict
@compute @workgroup_size(256) fn pointPred(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wid: vec3u,
                                           @builtin(local_invocation_index) lid: u32) {
  var y = vec3f();
  if (gid.x < P.nPts) { y = vec3f(Vi[16u * gid.x + 6u], Vi[16u * gid.x + 7u], Vi[16u * gid.x + 8u]); }
  shc[lid] = vec4f(dot(y, y), 0.0, 0.0, 0.0);
  for (var n = 128u; n > 0u; n >>= 1u) {
    workgroupBarrier();
    if (lid < n) { shc[lid] += shc[lid + n]; }
  }
  if (lid == 0u) { PP[wid.x] = shc[0].x; }
}
