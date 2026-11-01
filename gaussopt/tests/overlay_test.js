// The on-screen display's mask (src/sfm/overlay.js staticMask) on a synthetic video: a textured scene panning under
// a fixed label bar (text strokes on black) whose digits change, a reticle (a square outline) and a plain sky band
// above a fixed horizon (still under the pan, but without text or lines). The bar, its changing digits and the
// reticle's outline are masked; the reticle's inside, the sky and the scene are not. A camera that does not move
// gives no mask. dropMasked keeps the keypoints outside the mask with their descriptors.
import { staticMask, dropMasked, maskAt } from '../src/sfm/overlay.js';
import { assert } from './util.js';

const W = 160, H = 120;
const rand = (s) => () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);

// a texture wider than the frame, panned by `shift` px a frame
function video({ frames = 60, shift = 3, noise = 2 } = {}) {
  const r = rand(7), TW = W + frames * shift + 8, tex = new Uint8Array(TW * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < TW; x++) tex[y * TW + x] = 60 + Math.floor(140 * r());
  const out = [];
  for (let k = 0; k < frames; k++) {
    const f = new Uint8Array(W * H), nz = rand(100 + k);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let v = y < 20 ? 180 : tex[y * TW + x + k * shift];                                // the sky band, then the scene
      if (y >= 96 && y < 106 && x >= 4 && x < 100) {                                     // the label bar
        v = 0;
        if (x % 4 === 0 && y >= 98 && y < 104) v = 255;                                  // its text strokes
        if (x >= 80 && x < 92 && y >= 98 && y < 104) v = ((x + k) % 3 === 0) ? 255 : 0;  // its digits, changing
      }
      if (x >= 70 && x < 90 && y >= 40 && y < 60 && (x < 72 || x >= 88 || y < 42 || y >= 58)) v = 90;  // the reticle
      f[y * W + x] = Math.max(0, Math.min(255, v + Math.round((nz() - 0.5) * 2 * noise)));
    }
    out.push(f);
  }
  return out;
}

const at = (r, x, y) => r.mask[y * W + x];

Deno.test('overlay: the label bar, its changing digits and the reticle masked; the sky, the scene and the reticle\'s inside not', () => {
  const r = staticMask(video(), W, H);
  assert(r.mask, `no mask: ${r.why}`);
  assert(at(r, 10, 100) && at(r, 50, 97) && at(r, 98, 104), 'the bar not masked');
  assert(at(r, 84, 100) && at(r, 90, 101), 'the changing digits inside the bar not masked');
  assert(at(r, 70, 50) && at(r, 80, 41) && at(r, 89, 55), 'the reticle\'s outline not masked');
  assert(!at(r, 80, 50), 'the reticle\'s inside masked');
  let sky = 0, scene = 0;
  for (let x = 0; x < W; x++) for (let y = 0; y < 18; y++) sky += at(r, x, y);
  for (let y = 22; y < 90; y++) for (let x = 0; x < 60; x++) scene += at(r, x, y);
  assert(sky === 0, `${sky} pixels of the plain sky masked`);
  assert(scene === 0, `${scene} pixels of the panning scene masked`);
});

Deno.test('overlay: a camera that does not move gives no mask', () => {
  const r = staticMask(video({ shift: 0 }), W, H);
  assert(r.mask === null && /does not move/.test(r.why), `a still camera: ${r.why ?? `${(100 * r.share).toFixed(1)}% masked`}`);
});

Deno.test('overlay: dropMasked keeps the keypoints outside the mask with their descriptors', () => {
  const r = staticMask(video(), W, H);
  const kp = Float32Array.from([10.5, 100.5, 2, 0, 30.5, 60.5, 2, 0, 80.5, 50.5, 2, 0, 71.5, 50.5, 2, 0]);  // bar, scene, reticle inside, outline
  const de = new Uint8Array(128 * 4).map((_, i) => Math.floor(i / 128) + 1);
  const out = dropMasked(kp, de, r, W, H);
  assert(out.dropped === 2, `dropped ${out.dropped}, want 2`);
  assert(out.keypoints[0] === 30.5 && out.keypoints[4] === 80.5, `kept ${Array.from(out.keypoints)}`);
  assert(out.descriptors[0] === 2 && out.descriptors[128] === 3, 'the descriptors not those of the kept keypoints');
  assert(maskAt(r, 2 * 10, 2 * 100, 2 * W, 2 * H) === 1, 'maskAt at twice the size');
});
