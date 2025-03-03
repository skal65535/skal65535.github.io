// Product: filtering measurements

"use strict;"

function fmod360(v) {
  return v * 180. / Math.PI; //  ((v * 180. / Math.PI + 360.) % 360.);
}

class Filter {
  constructor(use_ahrs) {
    this.reset();
  }
  reset() {
    this.q = new Float32Array([1., 0., 0., 0]);
    this.t = performance.now();
    this.rpy = [0., 0., 0];
    this.smooth = 0.99;
    this.initialized = false;
  }

  mix(a, b) { return this.smooth * a + (1. - this.smooth) * b; }
  apply_smoothing(rpy) {
    if (!this.initialized) {
      this.initialized = true;
      this.rpy = rpy;
    } else {
      this.rpy[0] = this.mix(this.rpy[0], rpy[0]);
      this.rpy[1] = this.mix(this.rpy[1], rpy[1]);
      this.rpy[2] = this.mix(this.rpy[2], rpy[2]);
    }
    return this.rpy;
  }

  update([roll, pitch, yaw], t) {
     const dt = 0.005;//(t - this.t) * 0.5 * 0.01;
     this.t = t;
     if (dt < 0.001) return false;
     const rpy = this.apply_smoothing([roll, pitch, yaw]);
     const gx = dt * rpy[0], gy = dt * rpy[1], gz = dt * rpy[2];
     const qw = this.q[0], qx = this.q[1], qy = this.q[2], qz = this.q[3];
     this.q[0] += -qx * gx - qy * gy - qz * gz;
     this.q[1] +=  qw * gx - qz * gy + qy * gz;
     this.q[2] +=  qz * gx + qw * gy - qx * gz;
     this.q[3] += -qy * gx + qx * gy + qw * gz;
     this.normalize();
     return true;
  }
  normalize() {
    const qw = this.q[0], qx = this.q[1], qy = this.q[2], qz = this.q[3];
    let n = qw * qw + qx * qx + qy * qy + qz * qz;
    if (n != 0.) {
      n = 1. / Math.sqrt(n);
      this.q[0] = qw * n;
      this.q[1] = qx * n;
      this.q[2] = qy * n;
      this.q[3] = qz * n;
    }
  }
  // convert quaternion to Discrete Cosine Matrix (=rotation matrix)
  // https://en.wikipedia.org/wiki/Rotation_matrix#Spin_group
  to_dcm() {
    const qw = this.q[0], qx = this.q[1], qy = this.q[2], qz = this.q[3];
    const xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const xy = qy * qx, yz = qz * qy, xz = qx * qz;
    const wx = qw * qx, wy = qw * qy, wz = qw * qz;
    const s = 2. / Math.sqrt(xx + yy + zz + qw * qw);  // just in case...
    return [[1. - s * (yy + zz),      s * (xy - wz),      s * (xz + wy), 0.],
            [     s * (xy + wz), 1. - s * (xx + zz),      s * (yz - wx), 0.],
            [     s * (xz - wy),      s * (yz + wx), 1. - s * (xx + yy), 0.],
            [0., 0., 0., 1.]];
  }    
  get_rpy() {  // return [roll, pitch, yaw] in degrees
    const dcm = this.to_dcm();
    // Tait-Bryan (z-axis down) -> "zyx" sequence
    const roll  = fmod360(Math.atan2(dcm[2][1], dcm[2][2]));
    const pitch = fmod360(Math.asin(-dcm[2][0]));
    const yaw   = fmod360(Math.atan2(dcm[1][0], dcm[0][0]));
    return [roll, pitch, yaw];
  }
};
