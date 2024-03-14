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
      for (let k = 0; k < 4; ++k) s += A[j * 4 + k] * B[i + k * 4];
      AB[i + j * 4] = s;
    }
  }
  return AB;
}

