// metrics.js — Image quality metrics of the validation pass, on rgba8 readbacks (row pitch `bpr`,
// RGB channels only): PSNR, SSIM, and PSNR at 1/4 resolution.
"use strict";

const psnrOf = (mse) => (mse < 1e-10) ? 99.9 : -10 * Math.log10(mse);

// PSNR over the RGB channels of two rgba8 images
export function rgbPsnr(a, b, W, H, bpr) {
  let sumSq = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0, o = y * bpr; x < W; x++, o += 4) {
      for (let c = 0; c < 3; c++) {
        const d = (a[o + c] - b[o + c]) / 255;
        sumSq += d * d;
      }
    }
  }
  return psnrOf(sumSq / (W * H * 3));
}

// PSNR of the two images averaged over f x f blocks (the partial last row / column of blocks is
// dropped): coarse agreement (colors, shapes, exposure), blind to fine detail and to sub-pixel
// misalignment such as small pose errors.
export function rgbPsnrDown(a, b, W, H, bpr, f = 4) {
  const w = Math.floor(W / f), h = Math.floor(H / f);
  if (w === 0 || h === 0) return rgbPsnr(a, b, W, H, bpr);
  let sumSq = 0;
  for (let by = 0; by < h; by++) {
    for (let bx = 0; bx < w; bx++) {
      for (let c = 0; c < 3; c++) {
        let d = 0;
        for (let y = by * f; y < by * f + f; y++) {
          for (let x = bx * f, o = y * bpr + x * 4 + c; x < bx * f + f; x++, o += 4) d += a[o] - b[o];
        }
        d /= 255 * f * f;
        sumSq += d * d;
      }
    }
  }
  return psnrOf(sumSq / (w * h * 3));
}

// SSIM as in the 3DGS code (utils/loss_utils.py): per channel, 11x11 Gaussian window (sigma 1.5),
// zero padding at the borders, C1 = 0.01^2, C2 = 0.03^2 on [0, 1] values; mean over pixels and
// channels. Separable blur of the five moment maps, about 50 ms for 480 x 270.
const SSIM_R = 5;
const SSIM_W = (() => {
  const w = Array.from({ length: 2 * SSIM_R + 1 }, (_, k) => Math.exp(-((k - SSIM_R) ** 2) / (2 * 1.5 * 1.5)));
  const s = w.reduce((x, y) => x + y, 0);
  return Float32Array.from(w, x => x / s);
})();

export function rgbSsim(a, b, W, H, bpr) {
  const n = W * H, x = new Float32Array(n), y = new Float32Array(n);
  let sum = 0;
  for (let c = 0; c < 3; c++) {
    for (let j = 0; j < H; j++) {
      for (let i = 0, o = j * bpr + c; i < W; i++, o += 4) { x[j * W + i] = a[o] / 255; y[j * W + i] = b[o] / 255; }
    }
    sum += ssimSum(x, y, W, H);
  }
  return sum / (3 * n);
}

// The SSIM map of one channel (x, y: W x H values in [0, 1]) summed over its pixels
export function ssimSum(x, y, W, H) {
  const C1 = 0.01 ** 2, C2 = 0.03 ** 2, n = W * H;
  const [hx, hy, hxx, hyy, hxy] = Array.from({ length: 5 }, () => new Float32Array(n));  // horizontal pass of the moments
  let sum = 0;
  for (let j = 0; j < H; j++) {
    const row = j * W;
    for (let i = 0; i < W; i++) {
      let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
      for (let k = Math.max(-SSIM_R, -i); k <= Math.min(SSIM_R, W - 1 - i); k++) {
        const g = SSIM_W[k + SSIM_R], u = x[row + i + k], v = y[row + i + k];
        sx += g * u; sy += g * v; sxx += g * u * u; syy += g * v * v; sxy += g * u * v;
      }
      hx[row + i] = sx; hy[row + i] = sy; hxx[row + i] = sxx; hyy[row + i] = syy; hxy[row + i] = sxy;
    }
  }
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      let mx = 0, my = 0, exx = 0, eyy = 0, exy = 0;
      for (let k = Math.max(-SSIM_R, -j); k <= Math.min(SSIM_R, H - 1 - j); k++) {
        const g = SSIM_W[k + SSIM_R], p = (j + k) * W + i;
        mx += g * hx[p]; my += g * hy[p]; exx += g * hxx[p]; eyy += g * hyy[p]; exy += g * hxy[p];
      }
      const vx = exx - mx * mx, vy = eyy - my * my, cxy = exy - mx * my;
      sum += ((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx * mx + my * my + C1) * (vx + vy + C2));
    }
  }
  return sum;
}
