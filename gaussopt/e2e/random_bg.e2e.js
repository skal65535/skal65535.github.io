// random_bg.e2e.js — A training step's random background (random_bg, trainer.js _stepImpl: one color behind the splats
// in tile_forward and tile_backward, not learned). The synthetic room (photo_runs/tools/synth_room.py) trained 300 steps,
// then its splats held (adam_step skipped) and D-SSIM off:
//   - forward: a training step's render with background c is the black one's plus T c (T the pixel's transmittance,
//     pix_state.x), within the 8-bit rounding;
//   - under a learned background (bg_lr > 0, here too slow to move it off black), random_bg leaves it: the render is the
//     black one;
//   - backward: the photo made the black-background render and the background white: a pixel the splats leave
//     transparent shows white over its photo, so nearly every splat seen wants to be more opaque (its opacity gradient
//     dL/dop < 0); a backward pass still behind black would see the splats' own colors over black (the photo) and want
//     them less opaque.
//   cd e2e && PORT=8791 node random_bg.e2e.js   (env: MESH_E2E_ROOM, the synthetic room's folder; see harness.js)
import { fail, appUrl, run, room, appReady } from './harness.js';

const ROOM = room();

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await appReady(page);
  const r = await page.evaluate(async () => {
    const app = window.app, tr = app.trainer, P = app.params;
    const { train } = await import('./e2e/page_util.js');
    P.random_bg = false;  // the 300 steps behind black, as the scene the checks below expect
    await train(app, 300);
    Object.assign(P, { enable_rebalance: false, ssim: 0, random_bg: false });
    const v = tr.trainViews[0], W = tr.trainW, H = tr.trainH, bpr = tr.rowPitch, gtTex = await tr.ensureGtTexture(v);
    const pass = tr._pass, random = Math.random;
    tr._pass = (enc, pipe, ...rest) => pipe === tr.pipeAdam ? null : pass.call(tr, enc, pipe, ...rest);  // the splats held
    // one training step on view v: its render, transmittance and splats' gradients (background c, null: random_bg off)
    const stepOn = async (c) => {
      P.random_bg = c !== null;
      if (c !== null) Math.random = () => c;
      tr.device.queue.writeBuffer(tr.grads2dBuf, 0, new Float32Array(tr.count * 12));
      // the step and its readback under the step lock: the page's own frames draw into hatTex too
      const [hat, pix, g] = await tr._withStepLock(async () => {
        try { await tr._stepImpl(v); } finally { Math.random = random; }
        return tr._readback('random_bg', [{ tex: tr.hatTex }, { buf: tr.pixStateBuf, bytes: W * H * 16 },
                                          { buf: tr.grads2dBuf, bytes: tr.count * 48 }]);
      });
      return { hat: hat.slice(), pix: pix.slice(), g: g.slice() };
    };
    const black = await stepOn(null), grey = await stepOn(0.6);
    P.bg_lr = 1e-12;
    const learned = await stepOn(0.6);
    P.bg_lr = 0;
    let learnedOff = 0;
    for (let i = 0; i < black.hat.length; i++) if ((i & 3) !== 3) learnedOff = Math.max(learnedOff, Math.abs(learned.hat[i] - black.hat[i]));
    let worst = 0, seen = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const T = black.pix[4 * (y * W + x)];
      if (T > 0.01) seen++;
      for (let ch = 0; ch < 3; ch++) {
        const o = y * bpr + 4 * x + ch, want = Math.min(255, black.hat[o] + 255 * 0.6 * T);
        worst = Math.max(worst, Math.abs(grey.hat[o] - want));
      }
    }
    // backward: the photo the black render, the background white
    tr.device.queue.writeTexture({ texture: gtTex }, black.hat, { bytesPerRow: bpr }, [W, H]);
    const white = await stepOn(0.9999), off = await stepOn(null);
    let more = 0, less = 0, offMax = 0;
    for (let i = 0; i < tr.count; i++) {
      const d = white.g[i * 12 + 3];
      if (d < -1e-12) more++; else if (d > 1e-12) less++;
      offMax = Math.max(offMax, Math.abs(off.g[i * 12 + 3]));
    }
    tr._pass = pass;
    return { worst, seen, pixels: W * H, more, less, offMax, learnedOff };
  });
  if (!(r.seen > 0.01 * r.pixels)) fail(`only ${r.seen} of ${r.pixels} pixels show the background: nothing to test`);
  if (!(r.worst <= 1.5)) fail(`forward: background 0.6 off the black render + T c by ${r.worst} / 255`);
  if (!(r.learnedOff <= 1)) fail(`a learned background (bg_lr > 0): the render off the black one by ${r.learnedOff} / 255`);
  if (!(r.more > 0 && r.more >= 9 * r.less)) fail(`backward, white background: ${r.more} splats want more opacity, ${r.less} less`);
  if (!(r.offMax < 1e-6)) fail(`backward, black background on its own render: an opacity gradient of ${r.offMax}`);
  return `forward: the render + T x 0.6 within ${r.worst} / 255 (${r.seen} of ${r.pixels} pixels show the background), ` +
         `the black one under a learned background (within ${r.learnedOff}); ` +
         `backward, white background: ${r.more} splats want more opacity, ${r.less} less; black on its own render: |g| <= ${r.offMax.toExponential(1)}`;
}, { serverArgs: [ROOM] });
