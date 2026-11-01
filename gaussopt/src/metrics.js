// metrics.js — Image quality metrics of the validation pass, on rgba8 readbacks (row pitch `bpr`,
// RGB channels only): PSNR, SSIM (rgbSsim: the reference of the GPU's, ssimMean), PSNR at 1/4
// resolution, and PSNR by part of the image.
"use strict";

const psnrOf = (mse) => (mse < 1e-10) ? 99.9 : -10 * Math.log10(mse);

// PSNR over the RGB channels of two rgba8 images
export function rgbPsnr(a, b, W, H, bpr, { masked = false } = {}) {
  let sumSq = 0, n = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0, o = y * bpr; x < W; x++, o += 4) {
      if (masked && a[o + 3] === 0) continue;  // masked: a's alpha 0 leaves the pixel out (a video's on-screen display)
      for (let c = 0; c < 3; c++) {
        const d = (a[o + c] - b[o + c]) / 255;
        sumSq += d * d;
      }
      n++;
    }
  }
  return psnrOf(sumSq / (Math.max(1, n) * 3));
}

// PSNR after b's best gain and offset per channel (least squares: a ~ g b + o, the result kept in [0, 255]): a held-out
// view's score without its exposure, which no training step learned (trainer.js exposure, DESIGN.md E19)
export function rgbPsnrAffine(a, b, W, H, bpr, { masked = false } = {}) {
  const S = Array.from({ length: 3 }, () => [0, 0, 0, 0]);  // per channel: sum b, sum b^2, sum a, sum a b
  let n = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0, o = y * bpr; x < W; x++, o += 4) {
      if (masked && a[o + 3] === 0) continue;  // as rgbPsnr
      for (let c = 0; c < 3; c++) { const u = b[o + c], v = a[o + c], s = S[c]; s[0] += u; s[1] += u * u; s[2] += v; s[3] += u * v; }
      n++;
    }
  }
  const fit = S.map(([sb, sbb, sa, sab]) => {
    const vb = n * sbb - sb * sb, g = vb > 1e-9 ? (n * sab - sb * sa) / vb : 1;
    return [g, (sa - g * sb) / Math.max(1, n)];
  });
  let sumSq = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0, o = y * bpr; x < W; x++, o += 4) {
      if (masked && a[o + 3] === 0) continue;
      for (let c = 0; c < 3; c++) {
        const d = (a[o + c] - Math.min(255, Math.max(0, fit[c][0] * b[o + c] + fit[c][1]))) / 255;
        sumSq += d * d;
      }
    }
  }
  return { psnr: psnrOf(sumSq / (Math.max(1, n) * 3)), gain: fit.map(f => f[0]), offset: fit.map(f => f[1] / 255) };
}

// PSNR over parts of the image (null for an empty part): covered / uncovered, by the render's own final
// transmittance (pix: tile_forward's pix_state, 4 floats per pixel, T_final first; covered: T_final < tMax,
// as meanDepthSpread), with coverage the covered fraction of the pixels; centre / border, the middle half
// of the width and height (a quarter of the pixels) and the rest. Covered alone can be gamed by a model
// that leaves the hard parts empty: read it with coverage.
export const COVERED_T = 0.25;
export function rgbPsnrRegions(a, b, W, H, bpr, pix, tMax = COVERED_T, { masked = false } = {}) {
  const sq = [0, 0, 0, 0], n = [0, 0, 0, 0];  // covered, uncovered, centre, border
  const x0 = Math.floor(W / 4), x1 = W - x0, y0 = Math.floor(H / 4), y1 = H - y0;
  for (let y = 0; y < H; y++) {
    for (let x = 0, o = y * bpr; x < W; x++, o += 4) {
      if (masked && a[o + 3] === 0) continue;  // as rgbPsnr
      let s = 0;
      for (let c = 0; c < 3; c++) {
        const d = (a[o + c] - b[o + c]) / 255;
        s += d * d;
      }
      const cov = pix[4 * (y * W + x)] < tMax ? 0 : 1, mid = x >= x0 && x < x1 && y >= y0 && y < y1 ? 2 : 3;
      sq[cov] += s; n[cov]++;
      sq[mid] += s; n[mid]++;
    }
  }
  const p = (k) => n[k] ? psnrOf(sq[k] / (3 * n[k])) : null;
  return { covered: p(0), uncovered: p(1), coverage: n[0] / Math.max(1, n[0] + n[1]), centre: p(2), border: p(3) };
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
// channels. Separable blur of the five moment maps; too slow for the held-out views, which take the
// GPU's (ssimMean): this is its reference (ssim.e2e.js).
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

// SSIM from the GPU's sums (shaders.wgsl ssim_map, the same window and constants as rgbSsim) over a
// W x H image: S summed over the RGB channels and the pixels of each workgroup; the mean over pixels and channels
export function ssimMean(sums, W, H) {
  let sum = 0;
  for (let k = 0; k < sums.length; k++) sum += sums[k];
  return sum / (3 * W * H);
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
