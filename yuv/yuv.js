//
// YUV <-> RGB functions
//
////////////////////////////////////////////////////////////////////////////////

function clamp(v, m, M) { return Math.max(m, Math.min(M, v)); }
function clamp8(v) { return Math.max(0, Math.min(255, v)); }

const GAMMA = 1.2;
const GAMMA_INV = 1. / GAMMA;
function to_linear(x) { return 255. * Math.pow(x / 255., GAMMA); }
function from_linear(x) { return 255. * Math.pow(x / 255., GAMMA_INV); }

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
function map_err(v) { return clamp8(Math.floor(Math.pow(v / 255., .4) * 255.)); }

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
