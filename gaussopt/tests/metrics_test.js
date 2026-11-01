// Validation metrics (metrics.js) against an independent numpy computation (2D 11x11 Gaussian
// convolution with zero padding for SSIM), and the held-out view splits (colmap.js).
import { rgbPsnr, rgbPsnrDown, rgbSsim, rgbPsnrRegions, rgbPsnrAffine } from '../src/metrics.js';
import { splitValidation } from '../src/colmap.js';
import { assert, assertClose } from './util.js';

// W x H rgba8 image with a 4-byte row padding; the same integer pattern as the numpy reference
const W = 37, H = 23, BPR = W * 4 + 4;
function pattern(seed) {
  const v = new Float64Array(W * H * 3);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < 3; c++) {
    v[(y * W + x) * 3 + c] = (((x * 7 + y * 13 + c * 29 + seed) * (x + 3 * y + seed) + x * x) % 256) / 255;
  }
  return v;
}
function rgba(v) {
  const out = new Uint8Array(BPR * H).fill(77);  // alpha and padding must not count
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < 3; c++) {
    out[y * BPR + x * 4 + c] = Math.round(v[(y * W + x) * 3 + c] * 255);
  }
  return out;
}
const p1 = pattern(1), p5 = pattern(5);
const a = rgba(p1), b = rgba(p1.map((x, i) => Math.min(1, Math.max(0, 0.8 * x + 0.2 * p5[i]))));

Deno.test('metrics: PSNR, 1/4-resolution PSNR and SSIM match numpy', () => {
  assertClose(rgbPsnr(a, b, W, H, BPR), 21.744836, 1e-4, 'psnr');
  assertClose(rgbPsnrDown(a, b, W, H, BPR, 4), 34.576905, 1e-4, 'psnr4');
  assertClose(rgbSsim(a, b, W, H, BPR), 0.95927626, 1e-5, 'ssim');
  assertClose(rgbSsim(a, a, W, H, BPR), 1, 1e-6, 'ssim of an image with itself');
  assert(rgbPsnr(a, a, W, H, BPR) === 99.9, 'identical images');
});

Deno.test('metrics: a one-pixel shift of fine detail costs PSNR, not 1/4-resolution PSNR', () => {
  const stripes = (s) => rgba(Float64Array.from({ length: W * H * 3 }, (_, i) => ((Math.floor(i / 3) % W + s) % 2)));
  const [s0, s1] = [stripes(0), stripes(1)];
  assert(rgbPsnr(s0, s1, W, H, BPR) < 1, 'full resolution: every pixel wrong');
  assert(rgbPsnrDown(s0, s1, W, H, BPR, 4) > 90, 'blocks of 4 average the stripes out');
});

Deno.test('metrics: PSNR by part, covered / uncovered by T_final, centre / border', () => {
  // the render off by 10 levels where uncovered (T_final 0.9, columns x < 10) and by 20 on the border's rows y < 5;
  // exact elsewhere: each part's PSNR is that of its own error alone
  const pix = new Float32Array(W * H * 4), b = Uint8Array.from(a);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    pix[4 * (y * W + x)] = x < 10 ? 0.9 : 0.1;
    for (let c = 0; c < 3; c++) {
      const o = y * BPR + x * 4 + c;
      if (x < 10) b[o] = a[o] < 128 ? a[o] + 10 : a[o] - 10;
      else if (y < 5) b[o] = a[o] < 128 ? a[o] + 20 : a[o] - 20;
    }
  }
  const r = rgbPsnrRegions(a, b, W, H, BPR, pix), dB = (k, frac) => -10 * Math.log10(frac * (k / 255) ** 2);
  assertClose(r.coverage, 27 / 37, 1e-12, 'coverage: 27 of 37 columns');
  assertClose(r.uncovered, dB(10, 1), 1e-9, 'uncovered: 10 levels everywhere');
  assertClose(r.covered, dB(20, 5 / 23), 1e-9, 'covered: 20 levels on 5 of its 23 rows');
  // centre x 9..27, y 5..17: 1 of its 19 columns uncovered (x = 9), none of the rows y < 5
  assertClose(r.centre, dB(10, 1 / 19), 1e-9, 'centre');
  const all = rgbPsnr(a, b, W, H, BPR), mse = (p, n) => n * 10 ** (-p / 10);
  assertClose(mse(r.border, W * H - 19 * 13) + mse(r.centre, 19 * 13), mse(all, W * H), 1e-9, 'centre + border = the whole image');
  const none = rgbPsnrRegions(a, a, W, H, BPR, new Float32Array(W * H * 4));
  assert(none.uncovered === null && none.coverage === 1 && none.covered === 99.9, 'all covered: no uncovered part');
});

Deno.test('splitValidation: every8 by index or name hash; block is the middle tenth in name order; none', () => {
  const images = Array.from({ length: 40 }, (_, i) => ({ name: `f_${String(1000 - 7 * i).padStart(6, '0')}.jpg` }));
  splitValidation(images);
  assert(images.filter(im => im.isValidation).length === 5 && images[4].isValidation && images[12].isValidation);
  splitValidation(images, 'block');
  const held = images.filter(im => im.isValidation).map(im => im.name).sort();
  const sorted = images.map(im => im.name).sort();
  assert(held.length === 4 && held.join() === sorted.slice(18, 22).join(), `block: ${held}`);
  splitValidation(images.slice(0, 7), 'block');
  assert(!images.slice(0, 7).some(im => im.isValidation), 'fewer than 8 views: none held out');
  splitValidation(images, 'every8');
  splitValidation(images, 'none');
  assert(!images.some(im => im.isValidation), "'none' (photo sets): every view trains");
});

Deno.test('metrics: PSNR after the best gain and offset per channel (rgbPsnrAffine), the masked pixels left out', () => {
  const G = [0.8, 0.9, 0.7], O = [0.1, 0.05, 0.2];
  const lit = rgba(p1.map((x, i) => G[i % 3] * x + O[i % 3]));
  const fit = rgbPsnrAffine(a, lit, W, H, BPR);
  assert(rgbPsnr(a, lit, W, H, BPR) < 25, 'plain: the exposure costs');
  assert(fit.psnr > 45, `fitted: ${fit.psnr.toFixed(2)} dB (the rounding only)`);
  fit.gain.forEach((g, c) => assertClose(g, 1 / G[c], 0.01, `gain ${c}`));
  fit.offset.forEach((o, c) => assertClose(o, -O[c] / G[c], 0.005, `offset ${c}`));
  const m = a.slice(), l2 = lit.slice();
  m[3] = 0; l2[0] = 255 - l2[0];  // one pixel masked (alpha 0), its render wrong
  assert(rgbPsnrAffine(m, l2, W, H, BPR).psnr < fit.psnr - 3, 'not masked: it counts');
  assertClose(rgbPsnrAffine(m, l2, W, H, BPR, { masked: true }).psnr, fit.psnr, 0.5, 'masked: left out');
});
