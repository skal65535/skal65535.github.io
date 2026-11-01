// exposure.e2e.js — Each training view's exposure learned on the GPU (trainer.js exposure, exposure.js, DESIGN.md E19):
// the synthetic room (photo_runs/tools/synth_room.py) trained 1000 steps, then its splats frozen (lrs 0, no rebalance,
// no opacity reset). Each training view's ground truth replaced by its own render through a known exposure (a log-gain
// within +-0.1 and an offset within +-0.02 a channel, their means 0: the gauge): learned from there (at 0.02, the last
// third at 0.004), the gains come back within 0.5% and the offsets within 0.005; no held-out view gets one; the splats'
// gradient takes the gain (0.8: 0.64 times that of a gain of 1 against the photo over 0.8, the loss linear there; 0.8
// without it); a snapshot gives them back. The photos as they were would not do: the splats leave a residual, and the
// loss (smooth-L1 and L2) is not scale-invariant, so a brighter photo weighs it otherwise and moves its best gain. The
// gradient cut (no sum into loss_acc), or its sign wrong, leaves them off.
//   cd e2e && PORT=8791 node exposure.e2e.js   (env: MESH_E2E_ROOM, the synthetic room's folder; see harness.js)
import { fail, appUrl, run, room, appReady } from './harness.js';

const ROOM = room();

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await appReady(page);
  const r = await page.evaluate(async () => {
    const app = window.app, tr = app.trainer, P = app.params;
    const { train: trainOf, photoIsRender } = await import('./e2e/page_util.js'), train = (n) => trainOf(app, n);
    await train(1000);
    Object.assign(P, { opt_method: 'adam', lr_pos: 0, lr_scale: 0, lr_quat: 0, lr_color: 0, lr_opacity: 0, opacity_reset: 0,
                       enable_rebalance: false, drop_rate: 0, exposure_lr: 0.02, exposure_from: 0 });
    // the known exposures (exposure.js: c' = g (c - 1/2) + 1/2 + o), a fixed pattern, in the gauge
    const views = tr.trainViews, steps = 150 * views.length;
    const lg = views.map((_, i) => [0, 1, 2].map(c => 0.1 * Math.sin(1.7 * i + 2.3 * c + 0.5)));
    const off = views.map((_, i) => [0, 1, 2].map(c => 0.02 * Math.cos(2.9 * i + 1.1 * c)));
    for (const t of [lg, off]) for (let c = 0; c < 3; c++) { const m = t.reduce((s, x) => s + x[c], 0) / t.length; for (const x of t) x[c] -= m; }
    const W = tr.trainW, H = tr.trainH, bpr = tr.rowPitch;
    await photoIsRender(tr, views, (hat, i) => {
      const gt = hat.slice();
      for (let y = 0; y < H; y++) for (let x = 0, o = y * bpr; x < W; x++, o += 4) {
        for (let c = 0; c < 3; c++) gt[o + c] = Math.max(0, Math.min(255, Math.round(255 * (Math.exp(lg[i][c]) * (hat[o + c] / 255 - 0.5) + 0.5 + off[i][c]))));
        gt[o + 3] = 255;
      }
      return gt;
    });
    await train(steps * 2 / 3);
    P.exposure_lr = 0.004;  // the rate's decay (posLrDecay) to 5% comes at the planned length, far past these steps
    await train(steps / 3);
    let worst = 0, worstOff = 0;
    for (const [i, v] of views.entries()) {
      const s = tr.exposure.views.get(tr.images[v].name);
      for (let c = 0; c < 3; c++) {
        worst = Math.max(worst, Math.abs(Math.exp(s.p[c] - lg[i][c]) - 1));
        worstOff = Math.max(worstOff, Math.abs(s.p[3 + c] - off[i][c]));
      }
    }
    const names = [...tr.exposure.views.keys()];
    // the splats' gradient through an exposure: view v at gain 0.8 against a photo y, and at gain 1 against y / 0.8
    // (its render plus +-5 / 255: the loss linear there, smooth-L1's knee at 0.04), the 2D gradients (grads_2d,
    // adam_step skipped: it clears them) 0.8^2 = 0.64 times the latter's (the chain's factor g twice; 0.8 without it)
    const { Exposure } = await import('./src/exposure.js');
    const v = views[0], img = tr.images[v], k = tr.intrinsics(img), G = 0.8;
    tr.exposure = new Exposure();
    const out = { gtTex: await tr.ensureGtTexture(v) }, { hat: c } = await tr.readView(v);
    const yB = c.slice(), yA = c.slice();
    for (let o = 0; o < c.length; o += 4) for (let ch = 0; ch < 3; ch++) {
      yB[o + ch] = Math.max(0, Math.min(255, c[o + ch] + (((o >> 2) * 7 + ch * 3) % 2 ? 5 : -5)));
      yA[o + ch] = Math.round(G * yB[o + ch]);
    }
    const pass = tr._pass;
    tr._pass = (enc, pipe, ...rest) => pipe === tr.pipeAdam ? null : pass.call(tr, enc, pipe, ...rest);
    const grads = async (y, expo) => {
      tr.exposure = Exposure.fromJSON(expo);
      tr.device.queue.writeTexture({ texture: out.gtTex }, y, { bytesPerRow: bpr }, [W, H]);
      tr.device.queue.writeBuffer(tr.grads2dBuf, 0, new Float32Array(tr.count * 12));
      tr._writeUniforms(img, k, [0, 0, 0, 0, 0, 0, 0, 1, 0.03, 1, Math.log(10), 0, 0, 0, 0, 0]);
      tr._runPasses(out.gtTex);
      return (await tr._readback('expo_g2', [{ buf: tr.grads2dBuf, bytes: tr.count * 48 }]))[0].slice();
    };
    const gB = await grads(yB, {}), gA = await grads(yA, { [img.name]: [G, G, G, 0, 0, 0] });
    tr._pass = pass;
    let ab = 0, bb = 0;
    for (let i = 0; i < tr.count; i++) for (const j of [0, 1, 2, 3, 4, 5]) { ab += gA[i * 12 + j] * gB[i * 12 + j]; bb += gB[i * 12 + j] ** 2; }
    const chain = ab / bb;
    tr.exposure = Exposure.fromJSON(Object.fromEntries(names.map(n => [n, [1, 1, 1, 0, 0, 0]])));
    // a snapshot: the exposures restored with it (exposure.js toJSON / fromJSON)
    const before = names.map(n => tr.exposure.uniforms(n)), { buf } = await app._encodeSnapshot('manual');
    await app.restoreSnapshot(buf, 'e2e.gosnap', app.colmap, ++app.loadGen);
    const back = names.map(n => app.trainer.exposure.uniforms(n));
    const snapErr = Math.max(...before.flatMap((u, i) => u.map((x, j) => Math.abs(x - back[i][j]))));
    return { worst, worstOff, views: views.length, steps, learned: names.length, snapErr, chain, fresh: app.trainer !== tr,
             heldOut: tr.valViews.filter(v => names.includes(tr.images[v].name)).length };
  });
  if (r.learned !== r.views || r.heldOut) fail(`${r.learned} exposures learned for ${r.views} training views, ${r.heldOut} held out`);
  if (!(Math.abs(r.chain - 0.64) < 0.03)) fail(`the splats' gradient at gain 0.8: ${r.chain.toFixed(3)} times that of gain 1 (0.64 expected)`);
  if (!(r.fresh && r.snapErr < 1e-6)) fail(`the exposures from a snapshot: off by ${r.snapErr} (a new trainer: ${r.fresh})`);
  if (!(r.worst < 0.005 && r.worstOff < 0.005)) fail(`the known gains recovered within ${(100 * r.worst).toFixed(2)}%, the offsets within ${r.worstOff.toFixed(4)}`);
  return `${r.views} training views, ${r.steps} steps on frozen splats, each view's photo its render through a known exposure ` +
         `(gains +-10%, offsets +-0.02): the gains recovered within ${(100 * r.worst).toFixed(2)}%, the offsets within ${r.worstOff.toFixed(4)}; none held out; ` +
         `the splats' gradient at gain 0.8 ${r.chain.toFixed(3)} times that at 1 (0.64); restored from a snapshot`;
}, { serverArgs: [ROOM] });
