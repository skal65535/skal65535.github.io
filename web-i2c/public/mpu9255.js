// Product: https://stanford.edu/class/ee267/misc/MPU-9255-Datasheet.pdf
// https://stanford.edu/class/ee267/misc/MPU-9255-Register-Map.pdf

"use strict";

class MPU9255 {
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
    this.ADDRESS = 0x68;    // I2C: default MPUxxx device address
    this.WHO_AM_I = 0x75;    // self-identify: 0x71, 0x73, 0x70
    this.SMPLRT_DIV = 0x19;    // sample rate = 1000 / (1 + value) Hz
    this.CONFIG = 0x1a;
    this.GYRO_CONFIG = 0x1b;
    this.ACCEL_CONFIG1 = 0x1c;
    this.ACCEL_CONFIG2 = 0x1d;
    this.FIFO_EN = 0x23;
    this.I2C_MST_CTRL = 0x24;
    this.INT_PIN_CFG = 0x37;
    this.INT_ENABLE = 0x38;
    this.ACCEL_OUT = 0x3b;         // 0x3b->0x40 : XOUT / YOUT / ZOUT
    // this.TEMPERATURE_OUT = 0x41;   // 0x41/0x42: OUT_H/L
    // this.GYRO_OUT = 0x43;          // 0x43->0x48 : XOUT / YOUT / ZOUT
    this.USER_CTRL = 0x6a;    // DMP: bit 7: enable, bit 3: reset
    this.PWR_MGMT_1 = 0x6b;
    this.PWR_MGMT_2 = 0x6c;
    this.FIFO_COUNT = 0x72;
    this.FIFO_R_W = 0x74;

    this.filter = new Filter(false);
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
    if (id != 0x71 && id != 0x73 && id != 0x70) return false;

    await this.write(this.PWR_MGMT_1, 0x80);  // reset device (bit 7)
    sleep(100);
    await this.write(this.PWR_MGMT_1, 0x00);  // enable all sensors
    sleep(100);
    await this.write(this.PWR_MGMT_1, 0x01);  // auto-select clock source
    sleep(100);
    await this.write(this.CONFIG, 3);  // DLPF: 41Hz
    await this.write(this.SMPLRT_DIV, 4);  // Sample rate: 200Hz

    let c = await this.read(this.GYRO_CONFIG)
    c = (c & 0x04) | 0x00;   // fchoice = 3
    await this.write(this.GYRO_CONFIG, c);

    let d1 = await this.read(this.ACCEL_CONFIG1)
    d1 = (d1 & 0xf8);
    await this.write(this.ACCEL_CONFIG1, d1);

    let d2 = await this.read(this.ACCEL_CONFIG2)
    d2 = (d2 & 0xf0) | (0x01 << 3) | 0x03;  // fchoice + DLPF 41Hz (3)
    await this.write(this.ACCEL_CONFIG2, d2);

    await this.write(this.INT_PIN_CFG, 0x22);// BYPASS ENABLE, LATCH_INT_EN
    await this.write(this.INT_ENABLE, 0x01); // bit0: Enable data ready
    sleep(100);

    // set default full scale range for gyro and accel
    this.set_gyro_scale("250DPS");
    this.set_accel_scale("8G");

    return true;
  }

  async close() {
    this.I2C = null;
  }

  async set_gyro_scale(scale) {
    let v = await this.read(this.GYRO_CONFIG);
    v &= ~(3 << 3);  // clear bit 3 and 4
    let dps = 0.;   // degree per seconds
    switch (scale) {
      case "250DPS":  v |= 0 << 3; dps =  250.; break;
      case "500DPS":  v |= 1 << 3; dps =  500.; break;
      case "1000DPS": v |= 2 << 3; dps = 1000.; break;
      case "2000DPS": v |= 3 << 3; dps = 2000.; break;
    }
    await this.write(this.GYRO_CONFIG, v);
    this.gyro_scale = dps / 32768. * Math.PI / 180;  // scale in rad/s, not degree/s
    return true;
  }

  async set_accel_scale(scale) {
    let v = await this.read(this.ACCEL_CONFIG1);
    v &= ~(3 << 3);  // clear bit 3 and 4
    let g = 16.;
    switch (scale) {
      case "2G":  v |= 0 << 2; g =  2.; break;  // 2g
      case "4G":  v |= 1 << 2; g =  4.; break;  // 4g
      case "8G":  v |= 2 << 2; g =  8.; break;  // 8g
      default:
      case "16G": v |= 3 << 2; g = 16.; break;  // 16g
    }
    await this.write(this.ACCEL_CONFIG1, v);
    this.accel_scale = g / 32768.0 * 9.80665;  // include G in the scale
    return true;
  }

  reset() {
    this.filter.reset();
    this.gyro_bias = [0., 0., 0.];
    this.accel_bias = [0., 0., 0.];
  }

  async get_measurement() {  // read all ACCEL / T / GYRO at once
    const data = await this.I2C.read(this.ADDRESS, this.ACCEL_OUT, 6 + 2 + 6);
    if (data != null) {
      this.accel = get_3f_be(data, 0, this.accel_scale, this.accel_bias);
      const d_gyro = get_3f_be(data, 8, this.gyro_scale, this.gyro_bias);
      this.temperature = be_16s(data, 6) / 333.87 + 21.0;  // MPU9250 doc, paragraph 3.4.2

      if (!this.filter.update(d_gyro, performance.now())) return;
      this.gyro = this.filter.get_rpy();  // in degrees
    }
    return [this.accel, this.gyro, this.temperature];
  }
}
