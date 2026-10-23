// Undistortion for the hand-over (undistort.wgsl) against a CPU reference built on camera.js,
// point colors, and (slow) against tools/undistort.py's output (cv2.undistort) on a jardin5 frame.
import { GpuUndistort, PointColors } from '../src/sfm/undistort.js';
import { distort } from '../src/sfm/camera.js';
import { makeRng } from '../src/math.js';
import { assert, SLOW, env, exists, hasGPU } from './util.js';

const camera = (w, h, f, dist = [0, 0, 0, 0]) => ({ model: 'OPENCV', width: w, height: h, params: [f, 1.02 * f, w / 2 + 3, h / 2 - 2, ...dist] });
const pinholeOf = (cam) => ({ model: 'PINHOLE', width: cam.width, height: cam.height, params: cam.params.slice(0, 4) });

function image(w, h, seed) {
  const r = makeRng(seed), blobs = Array.from({ length: 80 }, () => [r() * w, r() * h, 3 + 10 * r(), [0, 1, 2].map(() => 255 * r())]);
  const out = new Uint8Array(4 * w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const c = [40, 80, 120];
      for (const [cx, cy, s, rgb] of blobs) {
        const a = Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s));
        for (let k = 0; k < 3; k++) c[k] = (1 - a) * c[k] + a * rgb[k];
      }
      out.set([...c.map(Math.round), 255], 4 * (y * w + x));
    }
  }
  return out;
}

// cv2.undistort semantics on the CPU: pinhole pixel -> distorted source pixel, bilinear, black outside
function reference(rgba, cam, pin) {
  const { width: w, height: h } = cam, [fx, fy, cx, cy] = pin.params, [Fx, Fy, Cx, Cy] = cam.params, out = new Float64Array(4 * w * h);
  const at = (x, y, k) => x < 0 || y < 0 || x >= w || y >= h ? 0 : rgba[4 * (y * w + x) + k];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [xd, yd] = distort(cam, (x + 0.5 - cx) / fx, (y + 0.5 - cy) / fy);
      const u = Fx * xd + Cx - 0.5, v = Fy * yd + Cy - 0.5, x0 = Math.floor(u), y0 = Math.floor(v), a = u - x0, b = v - y0;
      for (let k = 0; k < 4; k++) {
        out[4 * (y * w + x) + k] = (1 - b) * ((1 - a) * at(x0, y0, k) + a * at(x0 + 1, y0, k)) + b * ((1 - a) * at(x0, y0 + 1, k) + a * at(x0 + 1, y0 + 1, k));
      }
    }
  }
  return out;
}

Deno.test({ name: 'GPU undistortion equals the CPU reference; identity without distortion', ignore: !hasGPU, fn: async () => {
  const w = 320, h = 200, src = image(w, h, 4), gpu = await GpuUndistort.create();
  const flat = camera(w, h, 280);
  const same = await gpu.run(src, flat, pinholeOf(flat));
  assert(same.every((v, k) => v === src[k]), 'identity');
  const cam = camera(w, h, 280, [0.08, -0.02, 0.001, -0.002]);
  const out = await gpu.run(src, cam, pinholeOf(cam)), ref = reference(src, cam, pinholeOf(cam));
  let worst = 0;
  for (let k = 0; k < out.length; k++) worst = Math.max(worst, Math.abs(out[k] - ref[k]));
  assert(worst <= 1.01, `max difference ${worst.toFixed(2)} levels`);
  const corner = 4 * 0;
  assert(out[corner + 3] < 255 && ref[corner + 3] < 255, 'pincushion distortion (k1 > 0) maps the corners outside the source (black)');
  gpu.destroy();
} });

Deno.test('point colors: per-channel median of the observations, the pixel containing each', () => {
  const w = 4, h = 2, cam = { width: w, height: h };
  const px = (x, y) => [x + 0.2, y + 0.2];  // off-center: rounding x - 0.5 would take the pixel to the left / above
  const points = [{ id: 7, track: [[1, 0], [2, 0], [3, 0]], rgb: [0, 0, 0] }, { id: 9, track: [[1, 1], [2, 1]], rgb: [0, 0, 0] },
                  { id: 11, track: [[3, 4]], rgb: [0, 0, 0] }];
  const colors = new PointColors({ camera: cam, points });
  const img = (vals) => Uint8Array.from(vals.flatMap(v => [v, 255 - v, 7, 255]));
  colors.add({ points2D: [[...px(1, 0), 7], [...px(3, 1), 9]] }, img([0, 10, 0, 0, 0, 0, 0, 100]));
  colors.add({ points2D: [[...px(1, 0), 7], [...px(3, 1), 9]] }, img([0, 30, 0, 0, 0, 0, 0, 51]));
  colors.add({ points2D: [[...px(1, 0), 7], [w + 3, -2, 11]] }, img([0, 20, 0, 40, 0, 0, 0, 0]));  // outside: clamped to (3, 0)
  colors.finish();
  assert(points[0].rgb.join() === '20,235,7', `median of 3: ${points[0].rgb}`);
  assert(points[1].rgb.join() === '75,179,7', `median of 2, truncated: ${points[1].rgb}`);
  assert(points[2].rgb.join() === '40,215,7', `clamped: ${points[2].rgb}`);
});

const GS = env('SFM_GS_OURS') ?? `${env('HOME')}/gskal/brush/jardin5/out/gs_ours`;
const FRAMES = env('SFM_FRAMES') ?? `${env('HOME')}/gskal/brush/jardin5/out/images`;

const decodeRGBA = async (path) => (await new Deno.Command('ffmpeg', { args: ['-v', 'quiet', '-i', path, '-f', 'rawvideo', '-pix_fmt', 'rgba', '-'], stdout: 'piped' }).output()).stdout;

Deno.test({ name: 'GPU undistortion vs tools/undistort.py on a jardin5 frame (SFM_GS_OURS)', ignore: !SLOW || !hasGPU || !exists(`${GS}/undistort.json`), fn: async () => {
  const meta = JSON.parse(await Deno.readTextFile(`${GS}/undistort.json`));
  const name = meta.images[0], cam = meta.distorted;
  const src = await decodeRGBA(`${FRAMES}/${name}`), theirs = await decodeRGBA(`${GS}/images/${name}`);
  assert(src.length === 4 * cam.width * cam.height && theirs.length === src.length, 'frame sizes');
  const gpu = await GpuUndistort.create();
  const ours = await gpu.run(src, cam, meta.pinhole);
  const mad = (a) => {
    let sum = 0, n = 0;
    for (let k = 0; k < a.length; k += 4) for (let c = 0; c < 3; c++, n++) sum += Math.abs(a[k + c] - theirs[k + c]);
    return sum / n;
  };
  // both sides also differ by one JPEG re-encoding (q 95); the frame as is shows what a wrong remap costs
  const [same, raw] = [mad(ours), mad(src)];
  console.log(`  ${name}: mean absolute difference ${same.toFixed(2)} levels (not remapped: ${raw.toFixed(2)})`);
  assert(same < 2 && raw > 4 * same, 'same remap');
  gpu.destroy();
} });
