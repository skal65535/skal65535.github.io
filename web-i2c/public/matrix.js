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

function look_at_NED(pos, target, up) {
  const front = normalize([target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]]);
  const right = normalize(cross(up, front));
  const down = cross(front, right);
  return [pos, front, right, down];
}

function look_at(pos, target, up) {
  const [_pos, f, r, d] = look_at_NED(pos, target, up);
  return new Float32Array([
      r[0], r[1], r[2], -dot(r, pos),
      d[0], d[1], d[2], -dot(d, pos),
      f[0], f[1], f[2], -dot(f, pos),
      0., 0., 0., 1.]);
}

function perspective(fx, fy, znear, zfar) {
  const A = 1. / (zfar - znear);
  const B = znear * A;
  return new Float32Array([
      fx,  0., 0.,  0.,
      0., fy,  0.,  0.,
      0.,  0.,  A,  -B,
      0.,  0., 1.,  0.]);
}

function ortho(fx, fy, znear, zfar) {
  const A = zfar / (zfar - znear);
  const B = znear * A;
  return new Float32Array([
      fx,  0., 0.,  0.,
      0.,  fy, 0.,  0.,
      0.,  0.,  A,  -B,
      0.,  0., 0., -100000.]);
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

function multiply_AB(A, B) {  // 4x4 multiply A.B
  let AB = new Float32Array(4 * 4);
  for (let j = 0; j < 4; ++j) {
    for (let i = 0; i < 4; ++i) {
      let s = 0.;
      for (let k = 0; k < 4; ++k) s += A[j * 4 + k] * B[i + 4 * k];
      AB[i + j * 4] = s;
    }
  }
  return AB;
}

function multiply_Av3(A, v) {  // matrix-vector3 multiply: A.v
  let Av = new Float32Array(4);
  for (let i = 0; i < 4; ++i) {
    let s = A[i * 4 + 3];  // v[3] = 1.
    for (let j = 0; j < 3; ++j) s += A[i * 4 + j] * v[j];
    Av[i] = s;
  }
  return Av;
}

// returns A + scale * B
function FMA(A, scale, B) {
  return [A[0] + scale * B[0], A[1] + scale * B[1], A[2] + scale * B[2]];
}
