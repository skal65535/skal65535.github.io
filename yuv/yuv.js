//
// YUV <-> RGB functions
//

////////////////////////////////////////////////////////////////////////////////

function clamp(v, m, M) { return Math.max(m, Math.min(M, v)); }
function clamp8(v) { return Math.max(0, Math.min(255, v)); }

const GAMMA = 2.2;
const GAMMA_INV = 1. / GAMMA;
function to_linear(x) { return Math.floor(255. * Math.pow(x / 255., GAMMA)); }
function from_linear(x) { return Math.floor(255. * Math.pow(x / 255., GAMMA_INV)); }

function clamp_coord(x, y, w, h) {
  x = clamp(x, 0, w - 1);
  y = clamp(y, 0, h - 1);
  return 4 * (x + y * w);
}

////////////////////////////////////////////////////////////////////////////////
// YUV -> RGB

function to_rgb(y, u, v) {  // BT.601
  const R = (19077 * y             + 26149 * v - (14234 << 8)) >> 14;
  const G = (19077 * y -  6419 * u - 13320 * v +  (8708 << 8)) >> 14;
  const B = (19077 * y + 33050 * u             - (17685 << 8)) >> 14;
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
      total_err += err;
      // for visual display, emphasis on small errors
      const v_err = clamp8(Math.floor(Math.pow(err / 255., .4) * 255.));
      ERR.data[off + 0] = v_err;
      ERR.data[off + 1] = v_err;
      ERR.data[off + 2] = v_err;
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
  return er * er + eg * eg + eb * eb;
}

////////////////////////////////////////////////////////////////////////////////
// RGB -> YUV (the real stuff!)

function to_y(r, g, b) {
  return (+16839 * r + 33059 * g + 6420 * b + (16 << 16) + (1 << 15)) >> 16; // in [16, 235] range
}

function to_uv(r, g, b) {
  const U = ( -9719 * r - 19081 * g + 28800 * b + (128 << 16) + (1 << 16)) >> 16;
  const V = (+28800 * r - 24116 * g -  4684 * b + (128 << 16) + (1 << 16)) >> 16;
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
