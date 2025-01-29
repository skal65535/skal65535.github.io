"use strict";

////////////////////////////////////////////////////////////////////////////////
// "MCP2221 USB-I2C/UART Combo"

const FLASH_MANUFACTURER   = 0x02;
const FLASH_PRODUCT        = 0x03;
const FLASH_SERIAL         = 0x04;
const FLASH_FACTORY_SERIAL = 0x05;

const STATUS_SET    = 0x10;   // Table 3-1 / 3-2
const READ_FLASH    = 0xB0;   // Table 3-3
const I2C_WRITE     = 0x90;   // Table 3-20 / 3-21
const I2C_WRITE_RPT = 0x92;   // Table 3-22 / 3-23
const I2C_READ      = 0x91;   // launch the 'read' command
const I2C_READ_RPT  = 0x93;   // read w/ repeated start. Table 3-28 / 3-29
const I2C_READ_GET  = 0x40;   // retrieve the read data once command is completed

const SET_SRAM  = 0x60;       // table 3-36 / 3-37
const GET_SRAM  = 0x61;       // table 3-38 / 3-39
const RESET     = 0x70;       // table 3-40

function to_string(data) {
  if (data[3] != 3) return "error";   // error
  const len = data[2];
  return new TextDecoder("utf-16").decode(data.slice(4, len));
}

function stringify(obj) {
  let str = new Array();
  for (const p in obj) str.push(` ${p} : ${obj[p]}`);
  return str;
}

function le_16s(data, off) {
  const tmp = new ArrayBuffer(2);
  const a = new DataView(tmp);
  a.setUint8(0, data[off + 1]);
  a.setUint8(1, data[off + 0]);
  return a.getInt16(0);
}

function be_16s(data, off) {
  const tmp = new ArrayBuffer(2);
  const a = new DataView(tmp);
  a.setUint8(0, data[off + 0]);
  a.setUint8(1, data[off + 1]);
  return a.getInt16(0);
}

////////////////////////////////////////////////////////////////////////////////

class I2C_Device {
  constructor() {
    this.device = null;
    this.MCP2221_filter = [{ 'vendorId': 1240, 'productId': 221 }];
    this.report = new Uint8Array(64);
    this.reportId = 0;  // always
    this.gyro_scale = 0.;
    this.gyro_bias = [0., 0., 0.];
    this.accel_scale = 0.;
    this.accel_bias = [0., 0., 0.];
  }

  // prepare the promise that will wait for an inputreport event
  // https://stackoverflow.com/questions/73137521/wait-for-webhid-response-to-be-ready
  make_response() {
    return new Promise((resolve) => {
          this.device.addEventListener('inputreport', resolve, { once: true });
      });
  }

  async sleep(ms) {
    await new Promise(r => setTimeout(r, ms));
  }

  async send_flash_command(desc) {
    return to_string(await this.send_command(READ_FLASH, desc));
  }

  async finish_command() {
    await this.device.sendReport(this.reportId, this.report);
    const { reportId, data } = await this.make_response();
    return new Uint8Array(data.buffer);
  }

  init_report(cmd, array) {
    this.report.fill(0);
    this.report[0] = cmd;
    for (const [slot, v] of array) this.report[slot] = v;
  }

  async send_command(cmd, desc) {
    this.init_report(cmd, [[1, desc]]);
    return await this.finish_command();
  }

  async reset_device() {
    this.init_report(RESET, [[1, 0xab], [2, 0xcd], [3, 0xef]]);
    await this.device.sendReport(this.reportId, this.report);  // no answer expected!
    this.sleep(1000);
  }

  async set_clock(divider, duty) {
    if (divider == 0 || divider > 7) divider = 4;   // 3MHz
    const v2 = 0x80 | (duty & 0x18) | (divider & 7);  // table 3-37const
    this.init_report(SET_SRAM, [[2, v2]]);
    return await this.finish_command();
  }

  async set_divider(divider) {
    this.init_report(STATUS_SET, [[3, 0x20], [4, divider]]);   // Table 3-1
    return await this.finish_command();
  }

  async is_connected() {
    const data = await this.send_command(STATUS_SET, 0);
    return (data[0] == STATUS_SET);
  }

  async is_writable(slave) {
    if (slave >= 128) return false;
    this.init_report(I2C_WRITE, [[3, slave * 2]]);
    const data = await this.finish_command();
    //if (data[1] != 0) return false;
    //console.log("addr: " + slave + "  -> ", data);
    const state = await this.get_state();
    if (state != 0) {
      await this.cancel();
      return false;
    }
    return true;
  }

  async get_state() {
    const data = await this.send_command(STATUS_SET, 0);
    return data[8];
  }

  async cancel() {
    this.init_report(STATUS_SET, [[2, 0x10]]);
    await this.finish_command();
  }

  async wait_state(expected, nb_try) {
    while (nb_try-- > 0) {
      const state = await this.get_state();
      if (state == expected) return true;
      this.sleep(50);
    }
    return false;
  }

  async read(slave, reg, len) {
    if (reg >= 128 || len >= 60) return null;
    await this.cancel();

    // Table 3-26:
    this.init_report(I2C_WRITE, [[1, 1], [3, slave * 2], [4, reg]]);
    await this.finish_command();
    this.init_report(I2C_READ_RPT, [[1, len], [3, slave * 2 + 1]]);
    await this.finish_command();
    if (!await this.wait_state(85, 10)) return null;

    // Table 3-30 / 3-31:
    const data = await this.send_command(I2C_READ_GET, 0);
    if (data[3] != len) return null;
    return new Uint8Array(data.slice(4));
  }

  async read_byte(slave, reg) {
    let data = await this.read(slave, reg, 1);
    if (data == null) return 0x00;
    return data[0];
  }

  async write(slave, reg, data) {
    if (reg >= 128) return false;
    this.init_report(I2C_WRITE, [[1, 2], [3, slave * 2], [4, reg], [5, data]]);
    await this.finish_command();
    return true;
  }

////////////////////////////////////////////////////////////////////////////////

  async connect() {
    if (this.device == null) {
      /*
      this.device = await navigator.hid.requestDevice({ filters: this.MCP2221_filter, });
      console.log(device);
      */
      const device_list = await navigator.hid.getDevices();
      let devices = await navigator.hid.requestDevice({ filters: this.MCP2221_filter, });
      this.device = devices[0];
      if (this.device == null) return false;

      this.flash = null;
      this.sram = null;
      this.info = null;
      this.response = new Promise((resolve) => {
        this.device.addEventListener('inputreport', resolve, { once: true });
        resolve(true);
      });
    }

    await this.device.open()
      .then(() => console.log("HID Device opened"))
      .then(async () => {
        await this.set_clock(5 /* 1.5 MHz */, 0x18 /* duty < 75%*/);
        await this.set_divider(26);
/*
        console.log("COLLECTIONS:");
        for (let r of this.device.collections) {
          // https://usb.org/sites/default/files/hut1_5.pdf page 18
          // 0xff00 = vendor-defined
          console.log(" USAGE PAGE: ", "0x" + r.usagePage.toString(16));
          console.log(" USAGE: ", r.usage);
        }
*/
      })
      .then(async () => {
        this.flash = {};
        this.flash.manufacturer = await this.send_flash_command(FLASH_MANUFACTURER);
        this.flash.product = await this.send_flash_command(FLASH_PRODUCT);
        this.flash.serial = await this.send_flash_command(FLASH_SERIAL);
        this.flash.factory_serial = await this.send_flash_command(FLASH_FACTORY_SERIAL);
      }).then(async () => {
        this.info = {};
        const data = await this.send_command(STATUS_SET, 0);
        this.info.hardware = `${String.fromCharCode(data[46])}${String.fromCharCode(data[47])}`;  //  'A', '6'
        this.info.firmware = `${String.fromCharCode(data[48])}${String.fromCharCode(data[49])}`;  //  '1', '2'
      }).then(async () => {
        this.sram = {};
        const data = await this.send_command(GET_SRAM, 0);
        this.sram.vid = le_16s(data,  8);
        this.sram.pid  = le_16s(data, 10);
        this.sram.self_powered  = (data[12] & 0x40) ? true : false;
        this.sram.remote_wakeup = (data[12] & 0x20) ? true : false;
        this.sram.milliamps = data[13] * 2;
        this.sram.divider   = data[5] & 0x07;
        this.sram.duty      = data[5] & 0x18;
      }
    );
  }

  get_flash_status() {
    let str = new Array();
    str.push("======== FLASH =======");
    str.push.apply(str, stringify(this.flash));
    str.push.apply(str, stringify(this.info));
    str.push.apply(str, stringify(this.sram));
    return str;
  }

  async get_status(short) {
    let str = new Array();
    const r = await this.send_command(STATUS_SET, 0);
    if (!short) str.push("======== STATE =======");
    str.push(` [ 1] success = 0x${r[1].toString(16)} (success = 0x00)`);
    str.push(` [ 2] cancel state = 0x${r[2].toString(16)} (!= 0x00 ?)`);
    if (!short) {
      str.push(` [ 3] speed state = 0x${r[3].toString(16)}`);
      if (r[4]) str.push(` [ 4] new speed divider = ${r[4]}`);
      str.push(` [ 8] chip comm state = ${r[8]}`);
    }
    str.push(` [ 9] length requested = ${le_16s(r, 9)}`);
    str.push(` [11] length transfered = ${le_16s(r, 11)}`);
    if (!short) str.push(` [14] speed divider = ${r[14]}`);
    str.push(` [16] slave address = 0x${le_16s(r, 16).toString(16)}`);
    str.push(` [20] ACK = ${!(r[20] & 0x40)}`);
    str.push(` [22-23] pin values: SCL=${r[22]} SDA=${r[23]}`);
    str.push(` [25] pending value? ${r[25]}`);

    return str;
  }

  async disconnect() {
    this.device.close();
    this.device = null;
  }
}

////////////////////////////////////////////////////////////////////////////////
