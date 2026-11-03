// robust.e2e.js — The robust loss (robust_loss = tau, tile_backward): a pixel's gradient and its error in the moments
// weighted 1 / (1 + (e / tau)^2), e its mean absolute error. The synthetic room (photo_runs/tools/synth_room.py)
// trained 300 steps; a view's photo made its own render off by +-d in every channel (the sign keeping it in range):
// e = d at every pixel, so the splats' 2D gradients (grads_2d 0..8) and their error (the moments' R) are
// 1 / (1 + (d / tau)^2) times those without it, their view weight (grads_2d 9, the moments' W) the same. d = 5 / 255: 0.96 at tau 0.1,
// d = 100 / 255: 0.06.
//   cd e2e && PORT=8791 node robust.e2e.js   (env: MESH_E2E_ROOM, the synthetic room's folder; see harness.js)
import { fail, appUrl, run, room, appReady } from './harness.js';

const ROOM = room(), TAU = 0.1;

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await appReady(page);
  const r = await page.evaluate(async (TAU) => {
    const app = window.app, tr = app.trainer, P = app.params;
    const { train } = await import('./e2e/page_util.js');
    await train(app, 300);
    Object.assign(P, { opt_method: 'adam', enable_rebalance: false, drop_rate: 0 });
    const v = tr.trainViews[0], img = tr.images[v], k = tr.intrinsics(img), W = tr.trainW, H = tr.trainH, bpr = tr.rowPitch;
    const gtTex = await tr.ensureGtTexture(v), { hat } = await tr.readView(v);
    const pass = tr._pass;
    tr._pass = (enc, pipe, ...rest) => pipe === tr.pipeAdam ? null : pass.call(tr, enc, pipe, ...rest);  // adam_step clears them
    const grads = async (d, tau) => {
      const y = hat.slice();
      for (let r = 0; r < H; r++) for (let x = 0, o = r * bpr; x < W; x++, o += 4) {
        for (let c = 0; c < 3; c++) y[o + c] = hat[o + c] + d <= 255 ? hat[o + c] + d : hat[o + c] - d;
        y[o + 3] = 255;
      }
      tr.device.queue.writeTexture({ texture: gtTex }, y, { bytesPerRow: bpr }, [W, H]);
      tr.device.queue.writeBuffer(tr.grads2dBuf, 0, new Float32Array(tr.count * 12));
      tr.device.queue.writeBuffer(tr.momentsBuf, 0, new Float32Array(tr.count * 12));
      tr._writeUniforms(img, k, [0, 0, 0, 0, 0, 0, 0, 1, 0.03, 1, Math.log(10), 0, 0, 0, 0, tau]);
      tr._runPasses(gtTex);
      const [g, m] = await tr._readback('robust_g2', [{ buf: tr.grads2dBuf, bytes: tr.count * 48 }, { buf: tr.momentsBuf, bytes: tr.count * 48 }]);
      return { g: g.slice(), m: m.slice() };
    };
    // the ratio of a's slots to b's, least squares over the splats
    const ratio = (a, b, slots) => {
      let ab = 0, bb = 0;
      for (let i = 0; i < tr.count; i++) for (const j of slots) { ab += a[i * 12 + j] * b[i * 12 + j]; bb += b[i * 12 + j] ** 2; }
      return ab / bb;
    };
    const out = [];
    for (const d of [5, 100]) {
      const g0 = await grads(d, 0), g1 = await grads(d, TAU);
      out.push({ d, want: 1 / (1 + (d / 255 / TAU) ** 2), grad: ratio(g1.g, g0.g, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
                 view: ratio(g1.g, g0.g, [9]), err: ratio(g1.m, g0.m, [2]) });
    }
    tr._pass = pass;
    return out;
  }, TAU);
  for (const x of r) {
    const near = (a) => Math.abs(a - x.want) <= 0.01 * Math.max(1, 10 * x.want);  // false for NaN
    if (!near(x.grad) || !near(x.err))
      fail(`d ${x.d}: the gradients x ${x.grad.toFixed(4)}, the error x ${x.err.toFixed(4)}, want ${x.want.toFixed(4)}`);
    if (!(Math.abs(x.view - 1) <= 1e-4)) fail(`d ${x.d}: the view weight x ${x.view}`);
  }
  return r.map(x => `d ${x.d}/255: gradients x ${x.grad.toFixed(4)}, error x ${x.err.toFixed(4)}, view weight x ${x.view.toFixed(4)} (want ${x.want.toFixed(4)})`).join('; ');
}, { serverArgs: [ROOM] });
