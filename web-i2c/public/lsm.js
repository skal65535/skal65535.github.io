// Product: https://www.st.com/content/st_com/en/products/mems-and-sensors/inemo-inertial-modules/lsm6dsox.html

"use strict";

class LSM6DSOX {
  constructor(I2C) {
    this.I2C = I2C;
    this.accel_scale = 1.,
    this.accel_bias = [0., 0., 0.];
    this.accel = [0., 0., 0.];
    this.gyro_scale = 1.,
    this.gyro_bias = [0., 0., 0.];
    this.gyro = [0., 0., 0.];
    this.temperature = 25.;

    // control registers
    this.CTRL1_XL = 0x10;    // Accel: bit2-3: full-scale, bit 4-7: ODR
    this.CTRL2_G = 0x11;     // Gyro:  bit1-3: full_scale, bit 4-7: ODR
    this.CTRL3_C = 0x12;
    this.CTRL5_C = 0x14;     // bit7: ultra-low power, bit0-3:self-tests
    this.CTRL7_G = 0x16;
    this.CTRL8_XL = 0x17;
    this.STATUS_REG = 0X1e;
    this.ADDRESS = 0x6a;    // I2C: default LSMxxx device address
    this.WHO_AM_I = 0x0f;

    // accel / gyro
    this.ACCEL_OUT = 0x28;   // 0x28 -> 0x2d : linear accel
    this.GYRO_OUT = 0x22;    // 0x22 -> 0x27 : angular rate
    this.TEMPERATURE_OUT = 0x20;   // 0x20 / 0x21
  }

  async write(reg, v) {
    return await this.I2C.write(this.ADDRESS, reg, v);
  }
  async read(reg) {
    return await this.I2C.read_byte(this.ADDRESS, reg);
  }

  async init() {
    if (!await this.I2C.is_writable(this.ADDRESS)) return false;
    const id = await this.read(this.WHO_AM_I);
    if (id != 0x6c && id != 0x69) return false;

    // accel : 104Hz, 4G, bypass, low-pass filtering
    if (!await this.write(this.CTRL1_XL, 0x4a)) return false
    // gyro : 104Hz, 2000dps
    if (!await this.write(this.CTRL2_G, 0x4c)) return false;
    // gyro is 'high-performance', 16MHz bandwidth
    if (!await this.write(this.CTRL7_G, 0x00)) return false;
    // ODR config: ODR/4
    if (!await this.write(this.CTRL8_XL, 0x09)) return false;

    if (!await this.set_gyro_scale("250DPS")) return false;
    if (!await this.set_accel_scale("2G")) return false;
    return true;
  }

  async close() {
    this.I2C = null;
  }

  async set_gyro_scale(scale) {
    let v = await this.read(this.CTRL2_G);
    v &= ~(7 << 1);  // clear bit 1-3
    let dps = 0.;    // degree per seconds
    switch (scale) {
      case "150DPS":  v |= 1 << 1; dps =  125.; break;
      case "250DPS":  v |= 0 << 1; dps =  250.; break;
      case "500DPS":  v |= 2 << 1; dps =  500.; break;
      case "1000DPS": v |= 4 << 1; dps = 1000.; break;
      default:
      case "2000DPS": v |= 6 << 1; dps = 2000.; break;
    }
    if (!await this.write(this.CTRL2_G, v)) return false;
    this.gyro_scale = dps * 35. / 1000000.;
    return true;
  }

  async set_accel_scale(scale) {
    let v = await this.read(this.CTRL1_XL);
    v &= ~(7 << 1);  // clear bit 1-3
    let g = 16.;
    switch (scale) {
      case "2G":  v |= 0 << 2; g =  2.; break;  // 2g
      case "4G":  v |= 2 << 2; g =  4.; break;  // 4g
      case "8G":  v |= 3 << 2; g =  8.; break;  // 8g
      default:
      case "16G": v |= 1 << 2; g = 16.; break;  // 16g
    }
    if (!await this.write(this.CTRL1_XL, v)) return false;
    this.accel_scale = g / 32768.0;
    return true;
  }

  async get_measurement() {
    // read all GYRO / ACCEL / T bytes in one call
    const status_bits = await this.read(this.STATUS_REG);
    if (status_bits) {
      const data = await this.I2C.read(this.ADDRESS, this.GYRO_OUT, 6 + 6 + 2);
      if (data != null) {
        if (status_bits & 1) {
          this.accel = get_3f_le(data, 6, this.accel_scale, this.accel_bias);
        }
        if (status_bits & 2) {
          this.gyro = get_3f_le(data, 0, this.gyro_scale, this.gyro_bias);
        }
        if (status_bits & 4) {
          this.temperature = le_16s(data, 12) / 256.0 + 25.0;
        }
      }
    }
    return [this.accel, this.gyro, this.temperature];
  }
}
