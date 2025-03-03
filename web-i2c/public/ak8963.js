// Product: https://stanford.edu/class/ee267/misc/MPU-9255-Datasheet.pdf

"use strict";

class AK8963 {
  constructor(I2C) {
    this.I2C = I2C;
    this.mag_scale = 1.;
    this.bias = [0., 0., 0.];  // in 16b precision
    this.scales = [1., 1., 1.]; 
    this.mag_asa_f = [1., 1., 1.]; // axis sensity adjustment factors
    this.mag = [0., 0., 0.];

    // control registers
    this.ADDRESS = 0x0c;
    this.WHO_AM_I = 0x00;
    this.MAG_ST1 = 0x02;    // data ready status bit 0
    this.MAG_OUT = 0x03;    // 0x03 -> 0x08 : XOUT_L/H YOUT_L/H ZOUT_L/H
    this.MAG_ST2 = 0x09;    // bit3: data overflow, bit2: error stat
    this.MAG_CNTL = 0x0a;   // bit 0-3: MAG_MODE, bit 4: ODR
    this.MAG_CNTL2 = 0x0b;
    this.MAG_ASTC = 0x0c;   // self-test
    this.MAG_ASA = 0x10;    // fuse ROM axis sensitivity adjustment (0x10->0x12)
    this.MAG_MODE = 0x06;   // continuous read @ 100Hz
  }

  async write(reg, v) {
    return await this.I2C.write(this.ADDRESS, reg, v);
  }
  async read(reg) {
    return await this.I2C.read_byte(this.ADDRESS, reg);
  }
  async read_bytes(reg, len) {
    await this.I2C.write(0x25, this.ADDRESS);
    await this.I2C.write(0x26, reg);
    await this.I2C.write(0x27, 0x80 | len);
    sleep(1);
    const data = await this.I2C.read(0x49, len);
    return data;
  }
    
  async power_down() {
    await this.write(this.MAG_CNTL, 0x00);  // power down
    sleep(100);
  }

  async init() {
    if (!await this.I2C.is_writable(this.ADDRESS)) return false;
    const id = await this.read(this.WHO_AM_I);
    if (id != 0x48) return false;
//    if (!await this.set_mag_scale("250DPS")) return false;

    await this.power_down();
    await this.write(this.MAG_CNTL, 0x01);  // RESET
    await this.write(this.MAG_CNTL, 0x0f);  // enter fuse ROM access mode
    sleep(100);
    const data = await this.I2C.read(this.ADDRESS, this.MAG_ASA, 3);
    if (data == null) return false;
    this.mag_asa_f[0] = 1. + (data[0] - 128.) / 256.;
    this.mag_asa_f[1] = 1. + (data[1] - 128.) / 256.;
    this.mag_asa_f[2] = 1. + (data[2] - 128.) / 256.;
    await this.power_down();

    // bit 0-3: MAG_MODE -> 0x02=8Hz, 0x06=100Hz continuous read
    // bit4: 0 = M14BITS, 1 = M16BITS
    const use_14b = false;
    await this.write(this.MAG_CNTL, (use_14b ? 0x00 : 0x10) | this.MAG_MODE);
    sleep(50);
    // 1 milliGauss = 10 micro-Tesla.
    this.mag_scale = 10. * 4912. / (use_14b ? 8190. : 32760.);

//    const data2 = await this.I2C.read(this.ADDRESS, this.MAG_OUT, 7);
//    console.log(data2);

    return true;
  }

  async get_measurement() {
    const is_continuous_read = (this.MAG_MODE == 0x02 || this.MAG_MODE == 0x04 || this.MAG_MODE == 0x06);
    let st1 = 0x00;
    if (this.MAG_MODE == 0x01 || this.MAG_MODE == 0x08 || is_continuous_read) {
      st1 = await this.read(this.MAG_ST1);
      // DRDY: data not ready, needed for single-measurement and self-test
      if (this.MAG_MODE == 0x01 || this.MAG_MODE == 0x08) {
        if (!(st1 & 1)) return this.mag;
      }
      if (is_continuous_read) {
//        if (st1 & 2) return this.mag;   // DOR: Data overrun
      }
    }
    // this also reads ST2 as buf[6] at the end:
    const data = await this.I2C.read(this.ADDRESS, this.MAG_OUT, 7);
    if (data == null) return this.mag;
    const st2 = data[6];
    if (st2 & 8) console.log(`Overflow! st2=0x${hex(st2, 2)}`);
    if (st2 & 8) return this.mag;  // HOFL: measurement overflow
    for (let i in [0, 1, 2]) {
      const v = le_16s(data, 2 * i) * this.mag_scale * this.mag_asa_f[i];
      this.mag[i] = (v  - this.bias[i]) * this.scales[i];
    }
    return this.mag;
  }

  async close() {
    await this.power_down();
    this.I2C = null;
    return true;
  }
}
