// Product: histogram of recorded gyro

"use strict;"

class HistoView {
  constructor(canvas, name) {
    this.W = canvas.width;
    this.H = canvas.height;
    this.ctx = canvas.getContext("2d");
    this.data = new Float32Array(3 * this.W);
    this.data.fill(0);
    this.pos = 0;
    this.name = name;
  }

  add(gyro) {
    this.data[3 * this.pos + 0] = gyro[0];
    this.data[3 * this.pos + 1] = gyro[1];
    this.data[3 * this.pos + 2] = gyro[2];
    this.pos = (this.pos + 1) % this.W;
  }

  draw_lines(offset, color, scale_h) {
    this.ctx.strokeStyle = color;
    this.ctx.beginPath();
    for (let x = 0; x < this.W; ++x) {
      const pos = (x + this.pos) % this.W;
      const y = scale_h * this.data[3 * pos + offset] + this.H / 2.;
      if (x == 0) this.ctx.moveTo(x, y);
      else this.ctx.lineTo(x, y);
    }
    this.ctx.stroke();
  }
  render(imu) {
    this.add(imu.filter.get_rpy());
    
    this.ctx.fillStyle = '#000';
    this.ctx.fillRect(0, 0, this.W, this.H);

    const scale_h = .3 * this.H / 2.;
    this.draw_lines(0, '#ff0', scale_h);
    this.draw_lines(1, '#0ff', scale_h);
    this.draw_lines(2, '#f0f', scale_h);

    this.ctx.strokeStyle = '#fff';
    this.ctx.strokeText(this.name, 5, 15);
  }
}

