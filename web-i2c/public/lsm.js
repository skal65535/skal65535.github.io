// Product: https://www.st.com/content/st_com/en/products/mems-and-sensors/inemo-inertial-modules/lsm6dsox.html

"use strict";

// control registers
const CTRL1_XL = 0x10;    // Accel: bit2-3: full-scale, bit 4-7: ODR
const CTRL2_G = 0x11;     // Gyro:  bit1-3: full_scale, bit 4-7: ODR
const CTRL3_C = 0x12;
const CTRL5_C = 0x14;     // bit7: ultra-low power, bit0-3:self-tests
const CTRL7_G = 0x16;
const CTRL8_XL = 0x17;
const STATUS_REG = 0X1e;

const LSM_ADDRESS = 0x6a;    // I2C: default LSMxxx device address
const LSM_WHO_AM_I = 0x0f;

// accel / gyro
const ACCEL_OUT = 0x28;   // 0x28 -> 0x2d : linear accel
const GYRO_OUT = 0x22;    // 0x22 -> 0x27 : angular rate
const TEMPERATURE_OUT = 0x20;   // 0x20 / 0x21

class LSM {
  constructor(I2C) {
    this.I2C = I2C;
    this.accel_scale = 1.,
    this.accel_bias = [0., 0., 0.];
    this.accel = [0., 0., 0.];
    this.gyro_scale = 1.,
    this.gyro_bias = [0., 0., 0.];
    this.gyro = [0., 0., 0.];
    this.temperature = 25.;
  }

  async init() {
    if (!this.I2C.is_writable(LSM_ADDRESS)) return false;
    const id = await this.I2C.read_byte(LSM_ADDRESS, LSM_WHO_AM_I);
    if (id != 0x6c && id != 0x69) return false;

    // accel : 104Hz, 4G, bypass, low-pass filtering
    if (!await this.I2C.write(LSM_ADDRESS, CTRL1_XL, 0x4a)) return false
    // gyro : 104Hz, 2000dps
    if (!await this.I2C.write(LSM_ADDRESS, CTRL2_G, 0x4c)) return false;
    // gyro is 'high-performance', 16MHz bandwidth
    if (!await this.I2C.write(LSM_ADDRESS, CTRL7_G, 0x00)) return false;
    // ODR config: ODR/4
    if (!await this.I2C.write(LSM_ADDRESS, CTRL8_XL, 0x09)) return false;;

    if (!await this.set_gyro_scale("250DPS")) return false;;
    if (!await this.set_accel_scale("2G")) return false;
    return true;
  }

  async close() {
    this.I2C = null;
  }

  async set_gyro_scale(scale) {
    let v = await this.I2C.read_byte(LSM_ADDRESS, CTRL2_G);
    v &= ~(7 << 1);  // clear bit 1-3
    let dps = 0.;   // degree per seconds
    switch (scale) {
      case "150DPS":  v |= 1 << 1; dps =  125.; break;
      case "250DPS":  v |= 0 << 1; dps =  250.; break;
      case "500DPS":  v |= 2 << 1; dps =  500.; break;
      case "1000DPS": v |= 4 << 1; dps = 1000.; break;
      default:
      case "2000DPS": v |= 6 << 1; dps = 2000.; break;
    }
    if (!await this.I2C.write(LSM_ADDRESS, CTRL2_G, v)) return false;
    this.gyro_scale = dps * 35. / 1000000.;
    return true;
  }

  async set_accel_scale(scale) {
    let v = await this.I2C.read_byte(LSM_ADDRESS, CTRL1_XL);
    v &= ~(7 << 1);  // clear bit 1-3
    let g = 16.;
    switch (scale) {
      case "2G":  v |= 0 << 2; g =  2.; break;  // 2g
      case "4G":  v |= 2 << 2; g =  4.; break;  // 4g
      case "8G":  v |= 3 << 2; g =  8.; break;  // 8g
      default:
      case "16G": v |= 1 << 2; g = 16.; break;  // 16g
    }
    if (!await this.I2C.write(LSM_ADDRESS, CTRL1_XL, v)) return false;
    this.accel_scale = g / 32768.0;
    return true;
  }

  async get_3f_le(reg, scale, bias) {
    const data = await this.I2C.read(LSM_ADDRESS, reg, 6);
    if (data == null) return null;
    return [scale * le_16s(data, 0) - bias[0],
            scale * le_16s(data, 2) - bias[1],
            scale * le_16s(data, 4) - bias[2]];
  }

  async get_measurement() {
    const status_bit = await this.I2C.read_byte(LSM_ADDRESS, STATUS_REG);
    if (status_bit & 1) {
      this.accel = await this.get_3f_le(ACCEL_OUT, this.accel_scale, this.accel_bias);
    }
    if (status_bit & 2) {
      this.gyro = await this.get_3f_le(GYRO_OUT, this.gyro_scale, this.gyro_bias);
    }
    if (status_bit & 4) {
      const data = await this.I2C.read(LSM_ADDRESS, TEMPERATURE_OUT, 2);
      this.temperature = le_16s(data, 0) / 256.0 + 25.0;
    }
    return [this.accel, this.gyro, this.temperature];
  }
}