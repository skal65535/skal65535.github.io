//
// YUV <-> RGB functions
//
////////////////////////////////////////////////////////////////////////////////

function clamp(v, m, M) { return Math.max(m, Math.min(M, v)); }
function clamp8(v) { return Math.max(0, Math.min(255, v)); }

const GAMMA = 1.2;
const GAMMA_INV = 1. / GAMMA;
function to_linear(x) { return 255. * Math.pow(x / 255., GAMMA); }
function to_gamma(x) { return 255. * Math.pow(x / 255., GAMMA_INV); }

function clamp_coord(x, y, w, h) {
  x = clamp(x, 0, w - 1);
  y = clamp(y, 0, h - 1);
  return 4 * (x + y * w);
}

////////////////////////////////////////////////////////////////////////////////
// YUV -> RGB

function to_rgb(y, u, v) {  // BT.601
  const R = (19077 * y             + 26149 * v - (14234 << 8) + (1 << 13)) >> 14;
  const G = (19077 * y -  6419 * u - 13320 * v +  (8708 << 8)            ) >> 14;
  const B = (19077 * y + 33050 * u             - (17685 << 8) + (1 << 13)) >> 14;
  return new Uint8ClampedArray([R, G, B]);
}

function filter_9331(a, b, c, d) {
  return (9 * a + 3 * b + 3 * c + d) / 16;
}

function filter_rgbW(a, b, c, d, W) {
  let v = filter_9331(a, b, c, d);
  return clamp8(v + W);  // clipping is important!!
}

function get_rgb(Yo, Uo, Vo, X, Y, W, H) {
  X = clamp(X, 0, W - 1);
  Y = clamp(Y, 0, H - 1);

  // luma is full size
  const luma = Yo[X + Y * W];

  // downsize x2 for U/V planes
  const dX = (X & 1) ? 1 : -1;
  const dY = (Y & 1) ? 1 : -1;
  X = X >>> 1;
  Y = Y >>> 1;
  W = (W + 1) >>> 1;
  H = (H + 1) >>> 1;
  const OFF = X + Y * W;
  const u0 = Uo[OFF], v0 = Vo[OFF];  // nearest samples
  if (!params.fancy_upscaler) {
    return to_rgb(luma, u0, v0);
  }
  // compute the 'far' X/Y coordinates (Xf,Yf):
  const Xf = clamp(X + dX, 0, W - 1);
  const Yf = clamp(Y + dY, 0, H - 1);
  // get nearby samples
  const u1 = Uo[Xf + Y * W], u2 = Uo[X + Yf * W], u3 = Uo[Xf + Yf * W];
  const v1 = Vo[Xf + Y * W], v2 = Vo[X + Yf * W], v3 = Vo[Xf + Yf * W];
  // and apply the fancy upscaler:
  const u = filter_9331(u0, u1, u2, u3);
  const v = filter_9331(v0, v1, v2, v3);
  return to_rgb(luma, u, v);
}

// mean squared error
function mse(err) { return err[0] * err[0] + err[1] * err[1] + err[2] * err[2]; }
// for visual display, emphasis on small errors
function map_err(v) { return clamp8(Math.floor(Math.pow(Math.abs(v) / 255., .4) * 255.)); }

function convert_to_rgb(Y, U, V, RGB, ERR) {
  let total_err = 0.
  for (let y = 0; y < RGB.height; ++y) {
    for (let x = 0; x < RGB.width; ++x) {
      const rgb = get_rgb(Y, U, V, x, y, RGB.width, RGB.height);
      const off = 4 * (x + y * params.W);
      RGB.data[off + 0] = rgb[0];
      RGB.data[off + 1] = rgb[1];
      RGB.data[off + 2] = rgb[2];
      RGB.data[off + 3] = 255;

      const err = get_err(x, y, rgb);
      total_err += mse(err);

      ERR.data[off + 0] = map_err(err[0]);
      ERR.data[off + 1] = map_err(err[1]);
      ERR.data[off + 2] = map_err(err[2]);
      ERR.data[off + 3] = 255;
    }
  }
  return total_err / (RGB.width * RGB.height);
}

////////////////////////////////////////////////////////////////////////////////

function get_err(x, y, rgb) {
  const off = 4 * (x + y * params.W);
  const er = params.RGB.data[off + 0] - rgb[0];
  const eg = params.RGB.data[off + 1] - rgb[1];
  const eb = params.RGB.data[off + 2] - rgb[2];
  return new Float32Array([er, eg, eb]);
}

////////////////////////////////////////////////////////////////////////////////
// RGB -> YUV (the real stuff!)

function to_y(r, g, b) {
  return (+16839 * r + 33059 * g + 6420 * b + (16 << 16) + (1 << 15)) >> 16; // in [16, 235] range
}

function to_uv(r, g, b) {
  const U = ( -9719 * r - 19081 * g + 28800 * b + (128 << 16) + (1 << 15)) >> 16;
  const V = (+28800 * r - 24116 * g -  4684 * b + (128 << 16) + (1 << 15)) >> 16;
  return [U, V];  // in [16,240] range
}

function to_yuv(r, g, b) {
  const Y = to_y(r, g, b);
  const [U, V] = to_uv(r, g, b);
  return new Uint8ClampedArray([Y, U, V]);
}

function get_yuv(RGB, x, y, w, h) {
  const off = clamp_coord(x, y, w, h);
  return to_yuv(RGB[off + 0], RGB[off + 1], RGB[off + 2]);
}

function get_y(RGB, x, y, w, h) {
  const off = clamp_coord(x, y, w, h);
  return to_y(RGB[off + 0], RGB[off + 1], RGB[off + 2]);
}

function get_uv(RGB, x, y, w, h) {
  const off = clamp_coord(x, y, w, h);
  return to_uv(RGB[off + 0], RGB[off + 1], RGB[off + 2]);
}

function convert_to_yuv_fast(RGB, Yo, Uo, Vo, delta) {
  for (let y = 0; y < params.H; ++y) {
    for (let x = 0; x < params.W; ++x) {
      Yo[x + y * params.W] = get_y(RGB, x, y, params.W, params.H);
    }
  }
  for (let y = 0; y < params.h; ++y) {
    for (let x = 0; x < params.w; ++x) {
      let all_U = 0., all_V = 0., W = 0.;
      for (let Y = -delta + 1; Y <= delta; ++Y) {
        for (let X = -delta + 1; X <= delta; ++X) {
          const yuv = get_yuv(RGB, 2 * x + X, 2 * y + Y, params.W, params.H);
          const w = 1.;
          all_U += w * yuv[1];
          all_V += w * yuv[2];
          W += w;
        }
      }
      const off = x + y * params.w;
      const inv_W = 1. / W;
      Uo[off] = Math.floor(all_U * inv_W);
      Vo[off] = Math.floor(all_V * inv_W);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Optimisation:

// Update four Y,U,V values at x,y (which must be even)
//   uv0 x  x  uv1
//    x  Y0 Y1  x
//    x  Y2 Y3  x
//   uv2 x  x  uv3
// The floating-point BT.601 conversion is:
//   R = 1.164 * (Y-16) + 1.596 * (V-128)
//   G = 1.164 * (Y-16) - 0.813 * (V-128) - 0.392 * (U-128)
//   B = 1.164 * (Y-16)                   + 2.017 * (U-128)
// which can be written a t[RGB] = Ky.Y + Ku.U + Kv.V + B
//  U and V values are interpolated at Y co-locations using
//  the [9,3,3,1]/16 filter of fancy_upscaler.

function update_yuv(Y, U, V, RGB, x, y, W, H) {
  const xx = x >> 1;
  const yy = y >> 1;
  const w = (W + 1) >>> 1;
  const h = (H + 1) >>> 1;
  const dx = clamp(x + 1, 0, W - 1) - x;
  const dy = (clamp(y + 1, 0, H - 1) - y) * W;
  const off0 = x + y * W;
  const off1 = off0 + dx;
  const off2 = off0 + dy;
  const off3 = off2 + dx;
  const ys = new Float32Array([Y[off0] - 16., Y[off1] - 16., Y[off2] - 16., Y[off3] - 16.]);

  const dxx = clamp(xx + 1, 0, w - 1) - xx;
  const dyy = (clamp(yy + 1, 0, h - 1) - yy) * w;
  const uv_off0 = xx + yy * w;
  const uv_off1 = uv_off0 + dxx;
  const uv_off2 = uv_off0 + dyy;
  const uv_off3 = uv_off2 + dxx;
  const u0 = U[uv_off0] - 128.;
  const u1 = U[uv_off1] - 128.;
  const u2 = U[uv_off2] - 128.;
  const u3 = U[uv_off3] - 128.;
  const v0 = V[uv_off0] - 128.;
  const v1 = V[uv_off1] - 128.;
  const v2 = V[uv_off2] - 128.;
  const v3 = V[uv_off3] - 128.;

  const us = new Float32Array([
               filter_9331(u0, u1, u2, u3),
               filter_9331(u1, u0, u3, u2),
               filter_9331(u2, u3, u0, u1),
               filter_9331(u3, u2, u1, u0),]);
  const vs = new Float32Array([
               filter_9331(v0, v1, v2, v3),
               filter_9331(v1, v0, v3, v2),
               filter_9331(v2, v3, v0, v1),
               filter_9331(v3, v2, v1, v0),]);
  let R = new Float32Array([
               RGB[4 * off0 + 0],
               RGB[4 * off1 + 0],
               RGB[4 * off2 + 0],
               RGB[4 * off3 + 0],]);
  let G = new Float32Array([
               RGB[4 * off0 + 1],
               RGB[4 * off1 + 1],
               RGB[4 * off2 + 1],
               RGB[4 * off3 + 1],]);
  let B = new Float32Array([
               RGB[4 * off0 + 2],
               RGB[4 * off1 + 2],
               RGB[4 * off2 + 2],
               RGB[4 * off3 + 2],]);

  for (i = 0; i < 4; ++i) {
    R[i] = to_linear(R[i]);
    G[i] = to_linear(G[i]);
    B[i] = to_linear(B[i]);
  }
  let dY = new Float32Array([0., 0., 0., 0.]);
  let dU = new Float32Array([0., 0., 0., 0.]);
  let dV = new Float32Array([0., 0., 0., 0.]);
  for (let i = 0; i < 4; ++i) {
    const r = to_linear(1.164 * ys[i] + 1.596 * vs[i]);
    const g = to_linear(1.164 * ys[i] - 0.813 * vs[i] - 0.392 * us[i]);
    const b = to_linear(1.164 * ys[i]                 + 2.017 * us[i]);
    if (r >= 0. && r <= 255.) {
      dY[i] += (r - R[i]) * 1.164;
      dV[i] += (r - R[i]) * 1.596;
    }
    if (g >= 0. && g <= 255.) {
      dY[i] += (g - G[i]) * 1.164;
      dV[i] += (g - G[i]) * (-0.813);
      dU[i] += (g - G[i]) * (-0.392);
    }
    if (b >= 0. && b <= 255.) {
      dY[i] += (b - B[i]) * 1.164;
      dU[i] += (b - B[i]) * 2.017;
    }
  }

  const lambda = -.05;
  Y[off0] = Math.floor(clamp8(Y[off0] + lambda * dY[0]));
  Y[off1] = Math.floor(clamp8(Y[off1] + lambda * dY[1]));
  Y[off2] = Math.floor(clamp8(Y[off2] + lambda * dY[2]));
  Y[off3] = Math.floor(clamp8(Y[off3] + lambda * dY[3]));

  U[uv_off0] = Math.floor(clamp8(U[uv_off0] + lambda * dU[0]));
  U[uv_off1] = Math.floor(clamp8(U[uv_off1] + lambda * dU[1]));
  U[uv_off2] = Math.floor(clamp8(U[uv_off2] + lambda * dU[2]));
  U[uv_off3] = Math.floor(clamp8(U[uv_off3] + lambda * dU[3]));

  V[uv_off0] = Math.floor(clamp8(V[uv_off0] + lambda * dV[0]));
  V[uv_off1] = Math.floor(clamp8(V[uv_off1] + lambda * dV[1]));
  V[uv_off2] = Math.floor(clamp8(V[uv_off2] + lambda * dV[2]));
  V[uv_off3] = Math.floor(clamp8(V[uv_off3] + lambda * dV[3]));
}

function converge_sharp(Y, U, V, RGB, W, H) {
  for (let y = 1; y + 2 < H; y += 2) {
    for (let x = 1; x + 2 < W; x += 2) {
      update_yuv(Y, U, V, RGB, x, y, W, H);
    }
  }
}

function convert_to_yuv_sharp(RGB, Y, U, V, W, H, iters) {
  convert_to_yuv_fast(RGB, Y, U, V, params.delta);  // initial values
  for (let iter = 0; iter < iters; ++iter) {
    converge_sharp(Y, U, V, RGB, W, H);
  }
}

////////////////////////////////////////////////////////////////////////////////
// original Sharp-YUV algo

function to_gray(r, g, b) {
  return (13933 * r + 46871 * g + 4732 * b + (1 << 15)) >> 16;
}

class WRGB {
  constructor(W, H, with_tmp_buffer = true) {
    this.W = W;
    this.H = H;
    this.Wr = (W + 1) & ~1;  // rounded up
    this.Hr = (H + 1) & ~1;
    this.w = this.Wr >>> 1;
    this.h = this.Hr >>> 1;
    this.dRGB = new Int16Array(3 * this.w * this.h);
    this.Wl = new Int16Array(this.Wr * this.Hr);
    if (with_tmp_buffer) {
      // tmp buffers for transient RGB values (imported or computed)
      // In there, we store R, then G, then B sequentially, not interleaved.
      this.rgb1 = new Int16Array(3 * this.Wr);
      this.rgb2 = new Int16Array(3 * this.Wr);
      this.scratch = new Int16Array(Math.max(this.w * 3, this.Wr));
    }
  }

  import_rgb_row(RGB, dst) {
    for (let i = 0; i < this.W; ++i) {
      dst[i + 0 * this.Wr] = RGB[4 * i + 0];
      dst[i + 1 * this.Wr] = RGB[4 * i + 1];
      dst[i + 2 * this.Wr] = RGB[4 * i + 2];
    }
    if (this.W & 1) {  // replicate last pixel
      dst[this.W + 0 * this.Wr] = dst[this.W - 1 + 0 * this.Wr];
      dst[this.W + 1 * this.Wr] = dst[this.W - 1 + 1 * this.Wr];
      dst[this.W + 2 * this.Wr] = dst[this.W - 1 + 2 * this.Wr];
    }
  }

  rgb_to_Wl(rgb, dst) {
    for (let i = 0; i < this.Wr; ++i) {
      const r = to_linear(rgb[i + 0 * this.Wr]);
      const g = to_linear(rgb[i + 1 * this.Wr]);
      const b = to_linear(rgb[i + 2 * this.Wr]);
      dst[i] = to_gamma(to_gray(r, g, b));
    }
  }

  scale_down(rgb1, rgb2, off) {
    const a = to_linear(rgb1[off + 0]), b = to_linear(rgb1[off + 1]);
    const c = to_linear(rgb2[off + 0]), d = to_linear(rgb2[off + 1]);
    return to_gamma((a + b + c + d + 2) >> 2);
  }

  rgb_to_dRGB(rgb1, rgb2, dst) {
    for (let i = 0; i < this.w; ++i) {
      const r = this.scale_down(rgb1, rgb2, 2 * i + 0 * this.Wr);
      const g = this.scale_down(rgb1, rgb2, 2 * i + 1 * this.Wr);
      const b = this.scale_down(rgb1, rgb2, 2 * i + 2 * this.Wr);
      const wl = to_gray(r, g, b);
      dst[i + 0 * this.w] = r - wl;
      dst[i + 1 * this.w] = g - wl;
      dst[i + 2 * this.w] = b - wl;
    }
  }

  interpolate_two_rows(j) {
    const jj = j >> 1;
    const jm = Math.max(jj - 1, 0);
    const jM = Math.min(jj + 1, this.h - 1)
    let off = j * this.Wr;
    for (let N = 0; N < 3; ++N) {
      const drgb1 = this.dRGB.subarray((3 * jm + N) * this.w);
      const drgb2 = this.dRGB.subarray((3 * jj + N) * this.w);
      const drgb3 = this.dRGB.subarray((3 * jM + N) * this.w);
      const Wl1 = this.Wl.subarray(off + 0 * this.Wr);
      const Wl2 = this.Wl.subarray(off + 1 * this.Wr);
      const rgb1 = this.rgb1.subarray(N * this.Wr);
      const rgb2 = this.rgb2.subarray(N * this.Wr);
      rgb1[0] = filter_rgbW(drgb1[0], drgb1[0], drgb2[0], drgb2[0], Wl1[0]);
      rgb2[0] = filter_rgbW(drgb2[0], drgb2[0], drgb3[0], drgb3[0], Wl2[0]);
      for (let i = 1; i < this.w; ++i) {
        rgb1[2 * i - 1] = filter_rgbW(drgb2[i - 1], drgb2[i + 0], drgb1[i - 1], drgb1[i + 0], Wl1[2 * i - 1]);
        rgb1[2 * i + 0] = filter_rgbW(drgb2[i + 0], drgb2[i - 1], drgb1[i + 0], drgb1[i - 1], Wl1[2 * i + 0]);
        rgb2[2 * i - 1] = filter_rgbW(drgb2[i - 1], drgb2[i + 0], drgb3[i - 1], drgb3[i + 0], Wl2[2 * i - 1]);
        rgb2[2 * i + 0] = filter_rgbW(drgb2[i + 0], drgb2[i - 1], drgb3[i + 0], drgb3[i - 1], Wl2[2 * i + 0]);
      }
      if (!(this.W & 1)) {
        rgb1[this.W - 1] = filter_rgbW(drgb2[this.w - 1], drgb2[this.w - 1], drgb1[this.w - 1], drgb1[this.w - 1], Wl1[this.W - 1]);
        rgb2[this.W - 1] = filter_rgbW(drgb2[this.w - 1], drgb2[this.w - 1], drgb3[this.w - 1], drgb3[this.w - 1], Wl2[this.W - 1]);
      }
    }
  }

  converge_Wl(target, j, tmp_Wl) {
    let sum_diff = 0.;
    const off = j * this.Wr;
    for (let i = 0; i < this.Wr; ++i) {
      const diff = target.Wl[off + i] - tmp_Wl[i];
      sum_diff += Math.abs(diff);
      this.Wl[off + i] = clamp8(this.Wl[off + i] + diff);
    }
    return sum_diff;
  }
  converge_dRGB(target, j, tmp_dRGB) {
    const off = 3 * j * this.w;
    for (let i = 0; i < 3 * this.w; ++i) {
      const diff = target.dRGB[off + i] - tmp_dRGB[i];
      this.dRGB[off + i] = this.dRGB[off + i] + diff;
    }
  }

  import(RGB) {
    for (let j = 0; j < this.H; j += 2) {
      const J = Math.min(j + 1, this.H);
      this.import_rgb_row(RGB.subarray(j * 4 * this.W), this.rgb1);
      this.import_rgb_row(RGB.subarray(J * 4 * this.W), this.rgb2);
      // import Wl
      this.rgb_to_Wl(this.rgb1, this.Wl.subarray((j + 0) * this.Wr));
      this.rgb_to_Wl(this.rgb2, this.Wl.subarray((j + 1) * this.Wr));
      // import delta R/G/B
      this.rgb_to_dRGB(this.rgb1, this.rgb2, this.dRGB.subarray((j >> 1) * 3 * this.w));
    }
  }

  export_to(Y, U, V) {
    for (let j = 0; j < this.H; ++j) {
      for (let i = 0; i < this.W; ++i) {
        const wl = this.Wl[j * this.Wr + i];
        const off = (i >>> 1) + (j >>> 1) * 3 * this.w;
        const r = this.dRGB[off + 0 * this.w] + wl;
        const g = this.dRGB[off + 1 * this.w] + wl;
        const b = this.dRGB[off + 2 * this.w] + wl;
        Y[j * this.W + i] = to_y(r, g, b);  // clamp to [16, 240] ?
      }
    }
    for (let j = 0; j < this.h; ++j) {
      for (let i = 0; i < this.w; ++i) {
        const off = i + j * 3 * this.w;
        const r = this.dRGB[off + 0 * this.w];
        const g = this.dRGB[off + 1 * this.w];
        const b = this.dRGB[off + 2 * this.w];
        const uv = to_uv(r, g, b);
        U[i + j * this.w] = uv[0];  // clamp to [16, 235] ?
        V[i + j * this.w] = uv[1];
      }
    }
  }

  copy_from(src) {
    this.dRGB.set(src.dRGB);
    this.Wl.set(src.Wl);
  }
};

function one_iteration(target, cur) {
  let diff = 0.;
  for (let j = 0; j < cur.Hr; j += 2) {
    cur.interpolate_two_rows(j);  // -> cur.rgb1[] and cur.rgb2[] filled
    cur.rgb_to_Wl(cur.rgb1, cur.scratch);
    diff += cur.converge_Wl(target, j + 0, cur.scratch);
    cur.rgb_to_Wl(cur.rgb2, cur.scratch);
    diff += cur.converge_Wl(target, j + 1, cur.scratch);

    cur.rgb_to_dRGB(cur.rgb1, cur.rgb2, cur.scratch);
    cur.converge_dRGB(target, j >> 1, cur.scratch);
  }
  return diff / (cur.Wr * cur.Hr);
}

function convert_to_yuv_sharp_ref(RGB, Y, U, V, W, H, iters) {
  let cur = new WRGB(W, H, true);
  cur.import(RGB);
  let target = new WRGB(W, H, false);
  target.copy_from(cur);
  let prev_diff = 1e38;
  for (let iter = 0; iter < iters; ++iter) {
    const diff = one_iteration(target, cur);
    console.log(`#${iter}: ${diff}`);
    if (diff < .4 || diff > prev_diff) break;
    prev_diff = diff;
  }
  cur.export_to(Y, U, V);
}

////////////////////////////////////////////////////////////////////////////////
