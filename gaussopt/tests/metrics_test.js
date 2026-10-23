// Validation metrics (metrics.js) against an independent numpy computation (2D 11x11 Gaussian
// convolution with zero padding for SSIM), and the held-out view splits (colmap.js).
import { rgbPsnr, rgbPsnrDown, rgbSsim } from '../src/metrics.js';
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

Deno.test('splitValidation: every8 by index or name hash; block is the middle tenth in name order', () => {
  const images = Array.from({ length: 40 }, (_, i) => ({ name: `f_${String(1000 - 7 * i).padStart(6, '0')}.jpg` }));
  splitValidation(images);
  assert(images.filter(im => im.isValidation).length === 5 && images[4].isValidation && images[12].isValidation);
  splitValidation(images, 'block');
  const held = images.filter(im => im.isValidation).map(im => im.name).sort();
  const sorted = images.map(im => im.name).sort();
  assert(held.length === 4 && held.join() === sorted.slice(18, 22).join(), `block: ${held}`);
  splitValidation(images.slice(0, 7), 'block');
  assert(!images.slice(0, 7).some(im => im.isValidation), 'fewer than 8 views: none held out');
});
