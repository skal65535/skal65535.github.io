////////////////////////////////////////////////////////////////////////////////
// Matrix utils for 3D

function normalize([x, y, z]) {
  const d = 1. / Math.hypot(x, y, z);
  return [x * d, y * d, z * d];
}
function cross([Ax, Ay, Az], [Bx, By, Bz]) {
  return [Ay * Bz - Az * By, Az * Bx - Ax * Bz, Ax * By - Ay * Bx];
}
function dot([Ax, Ay, Az], [Bx, By, Bz]) {
  return Ax * Bx + Ay * By + Az * Bz;
}

function id4([scale_x = 1., scale_y = 1., scale_z = 1.]) {
  return new Float32Array([
     scale_x, 0., 0., 0.,
     0., scale_y, 0., 0.,
     0., 0., scale_z, 0.,
     0., 0., 0.,      1.]);
}

function look_at([p_x, p_y, p_z],  // position relative to target
                 [t_x, t_y, t_z],  // target
                 [up_x, up_y, up_z]) {
  const [f_x, f_y, f_z] = normalize([p_x, p_y, p_z]);
  const [r_x, r_y, r_z] = normalize(cross([up_x, up_y, up_z], [f_x, f_y, f_z]));
  const [d_x, d_y, d_z] = cross([f_x, f_y, f_z], [r_x, r_y, r_z]);
  p_x += t_x;
  p_y += t_y;
  p_z += t_z;
  return new Float32Array([
      r_x, d_x, f_x, 0.,
      r_y, d_y, f_y, 0.,
      r_z, d_z, f_z, 0.,
      -dot([r_x, r_y, r_z], [p_x, p_y, p_z]),
      -dot([d_x, d_y, d_z], [p_x, p_y, p_z]),
      -dot([f_x, f_y, f_z], [p_x, p_y, p_z]), 1.]);
}

function perspective(fx, fy, znear, zfar) {
  const A = znear / (zfar - znear);
  const B = zfar * A;
  return new Float32Array([
     -fx,  0., 0.,  0.,
      0., -fy, 0.,  0.,
      0.,  0.,  A, -1.,
      0.,  0.,  B,  0.]);
}

function ortho(fx, fy, znear, zfar) {
  const A = 1. / (zfar - znear);
  const B = znear * A;
  return new Float32Array([
     -fx,  0., 0.,  0.,
      0., -fy, 0.,  0.,
      0.,  0.,  B,  0.,
      0.,  0.,  A,  1.]);
}

function rotation([ax, ay, az], angle) {
  [ax, ay, az] = normalize([ax, ay, az]);
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const [sx, sy, sz] = [ax * s, ay * s, az * s];
  const [cx, cy, cz] = [ax * (1. - c), ay * (1. - c), az * (1. - c)];
  return new Float32Array([
    cx * ax +  c, cx * ay - sz, cx * az + sy, 0.,
    cy * ax + sz, cy * ay +  c, cy * az - sx, 0.,
    cz * ax - sy, cz * ay + sx, cz * az +  c, 0.,
    0., 0., 0., 1.]);
}

function multiply4(A, B) {  // 4x4 multiply A.B
  let AB = new Float32Array(16);
  for (let j = 0; j < 4; ++j) {
    for (let i = 0; i < 4; ++i) {
      let s = 0.;
      for (let k = 0; k < 4; ++k) s += B[j * 4 + k] * A[i + k * 4];
      AB[i + j * 4] = s;
    }
  }
  return AB;
}

function multiply(A, v) {  // matrix-vector multiply: A.v
  let Av = new Float32Array(4);
  for (let i = 0; i < 4; ++i) {
    let s = 0.;
    for (let j = 0; j < 4; ++j) s += A[i * 4 + j] * v[j];
    Av[i] = s;
  }
  return Av;
}

// taken from wgpu-matrix
// https://github.com/greggman/wgpu-matrix/blob/main/src/mat4-impl.ts
function inverse(A) {
  const dst = new Float32Array(16);

  const m00 = A[0 * 4 + 0];
  const m01 = A[0 * 4 + 1];
  const m02 = A[0 * 4 + 2];
  const m03 = A[0 * 4 + 3];
  const m10 = A[1 * 4 + 0];
  const m11 = A[1 * 4 + 1];
  const m12 = A[1 * 4 + 2];
  const m13 = A[1 * 4 + 3];
  const m20 = A[2 * 4 + 0];
  const m21 = A[2 * 4 + 1];
  const m22 = A[2 * 4 + 2];
  const m23 = A[2 * 4 + 3];
  const m30 = A[3 * 4 + 0];
  const m31 = A[3 * 4 + 1];
  const m32 = A[3 * 4 + 2];
  const m33 = A[3 * 4 + 3];
  const tmp0  = m22 * m33;
  const tmp1  = m32 * m23;
  const tmp2  = m12 * m33;
  const tmp3  = m32 * m13;
  const tmp4  = m12 * m23;
  const tmp5  = m22 * m13;
  const tmp6  = m02 * m33;
  const tmp7  = m32 * m03;
  const tmp8  = m02 * m23;
  const tmp9  = m22 * m03;
  const tmp10 = m02 * m13;
  const tmp11 = m12 * m03;
  const tmp12 = m20 * m31;
  const tmp13 = m30 * m21;
  const tmp14 = m10 * m31;
  const tmp15 = m30 * m11;
  const tmp16 = m10 * m21;
  const tmp17 = m20 * m11;
  const tmp18 = m00 * m31;
  const tmp19 = m30 * m01;
  const tmp20 = m00 * m21;
  const tmp21 = m20 * m01;
  const tmp22 = m00 * m11;
  const tmp23 = m10 * m01;

  const t0 = (tmp0 * m11 + tmp3 * m21 + tmp4 * m31) -
      (tmp1 * m11 + tmp2 * m21 + tmp5 * m31);
  const t1 = (tmp1 * m01 + tmp6 * m21 + tmp9 * m31) -
      (tmp0 * m01 + tmp7 * m21 + tmp8 * m31);
  const t2 = (tmp2 * m01 + tmp7 * m11 + tmp10 * m31) -
      (tmp3 * m01 + tmp6 * m11 + tmp11 * m31);
  const t3 = (tmp5 * m01 + tmp8 * m11 + tmp11 * m21) -
      (tmp4 * m01 + tmp9 * m11 + tmp10 * m21);

  const d = 1 / (m00 * t0 + m10 * t1 + m20 * t2 + m30 * t3);

  dst[ 0] = d * t0;
  dst[ 1] = d * t1;
  dst[ 2] = d * t2;
  dst[ 3] = d * t3;
  dst[ 4] = d * ((tmp1 * m10 + tmp2 * m20 + tmp5 * m30) -
          (tmp0 * m10 + tmp3 * m20 + tmp4 * m30));
  dst[ 5] = d * ((tmp0 * m00 + tmp7 * m20 + tmp8 * m30) -
          (tmp1 * m00 + tmp6 * m20 + tmp9 * m30));
  dst[ 6] = d * ((tmp3 * m00 + tmp6 * m10 + tmp11 * m30) -
          (tmp2 * m00 + tmp7 * m10 + tmp10 * m30));
  dst[ 7] = d * ((tmp4 * m00 + tmp9 * m10 + tmp10 * m20) -
          (tmp5 * m00 + tmp8 * m10 + tmp11 * m20));
  dst[ 8] = d * ((tmp12 * m13 + tmp15 * m23 + tmp16 * m33) -
          (tmp13 * m13 + tmp14 * m23 + tmp17 * m33));
  dst[ 9] = d * ((tmp13 * m03 + tmp18 * m23 + tmp21 * m33) -
          (tmp12 * m03 + tmp19 * m23 + tmp20 * m33));
  dst[10] = d * ((tmp14 * m03 + tmp19 * m13 + tmp22 * m33) -
          (tmp15 * m03 + tmp18 * m13 + tmp23 * m33));
  dst[11] = d * ((tmp17 * m03 + tmp20 * m13 + tmp23 * m23) -
          (tmp16 * m03 + tmp21 * m13 + tmp22 * m23));
  dst[12] = d * ((tmp14 * m22 + tmp17 * m32 + tmp13 * m12) -
          (tmp16 * m32 + tmp12 * m12 + tmp15 * m22));
  dst[13] = d * ((tmp20 * m32 + tmp12 * m02 + tmp19 * m22) -
          (tmp18 * m22 + tmp21 * m32 + tmp13 * m02));
  dst[14] = d * ((tmp18 * m12 + tmp23 * m32 + tmp15 * m02) -
          (tmp22 * m32 + tmp14 * m02 + tmp19 * m12));
  dst[15] = d * ((tmp22 * m22 + tmp16 * m02 + tmp21 * m12) -
          (tmp20 * m12 + tmp23 * m22 + tmp17 * m02));
  return dst;
}
