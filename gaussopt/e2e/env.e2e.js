// env.e2e.js — The background behind the splats (src/env.js, DESIGN.md E13) learned in the page: the synthetic room
// (photo_runs/tools/synth_room.py) served as a dataset, every splat made nearly transparent (opacity 0.0045: each pixel
// then is nearly the background alone) and frozen (lrs 0, Adam, no rebalance, no opacity reset). With bg_lr 0, 100 steps change nothing
// (the background stays black: the held-out PSNR as before, to 0.01 dB); with bg_lr 0.05, 300 steps learn it from the
// training views: the held-out views (other directions of the same room) gain over 3 dB, the background no longer black.
// A wrong sign in its gradient would lower them instead, a background not drawn would leave them as they were.
// Then, reloaded, 600 steps of the default training with bg_lr 0.05: the room is closed, every direction ends on a wall,
// so the background is seen where the splats do not cover yet (it learns the walls' colors there) or through nearly opaque
// splats: few of its texels saturated (max - min over 100 / 255, colors no photo has; under 0.3%, which needs it taught
// only by the pixels it dominates, T_final over 0.5, tile_backward, and its rate scaled by how much it is seen, env_adam).
//   cd e2e && node env.e2e.js   (env: MESH_E2E_ROOM, the synthetic room's folder; see harness.js)
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { fail, appUrl, run } from './harness.js';

const ROOM = process.env.MESH_E2E_ROOM ?? `${process.env.HOME}/gskal/brush/synth/room10`;
if (!existsSync(join(ROOM, 'scene_sdf.json'))) throw new Error(`no synthetic room in ${ROOM} (set MESH_E2E_ROOM)`);

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await page.waitForFunction(() => !!window.app?.trainer && window.app.gtAvailable, null, { timeout: 120000 });
  const r = await page.evaluate(async () => {
    const app = window.app, tr = app.trainer, P = app.params;
    Object.assign(P, { opt_method: 'adam', lr_pos: 0, lr_scale: 0, lr_quat: 0, lr_color: 0, lr_opacity: 0, opacity_reset: 0,
                       enable_rebalance: false, bg_lr: 0 });
    for (let i = 0; i < tr.count; i++) tr.cpuParams[i * 16 + 3] = -5.4;  // opacity 0.0045, just over 1 / 255 (under it the octree culls them all)
    tr._uploadSplats();
    const train = async (n) => {  // n steps (null when they do not come within 120 s)
      P.stop_after = tr.stepCount + n;
      app._setTraining(true);
      for (const t0 = Date.now(); tr.stepCount < P.stop_after || P.training; await new Promise(res => setTimeout(res, 100)))
        if (Date.now() - t0 > 120000) throw new Error(`stuck at step ${tr.stepCount} of ${P.stop_after}, training ${P.training}: ${app.logText().split('\n').slice(-5).join(' | ')}`);
    };
    const val = async () => (await tr.evaluateValidation())?.psnr;
    const lit = (rgb) => rgb.filter(x => x > 8).length / rgb.length;
    const v0 = await val();
    await train(100);
    const v1 = await val(), lit1 = lit(await tr.readEnv());
    P.bg_lr = 0.05;
    await train(300);
    const v2 = await val(), lit2 = lit(await tr.readEnv());
    return { v0, v1, v2, lit1, lit2, views: tr.valViews.length };
  });
  if (!(r.views > 0)) fail('no held-out views');
  if (!(Math.abs(r.v1 - r.v0) < 0.01 && r.lit1 === 0)) fail(`bg_lr 0: held-out ${r.v0.toFixed(2)} -> ${r.v1.toFixed(2)} dB, ${(100 * r.lit1).toFixed(1)}% of the background lit`);
  // lit: a part only (the room's cameras see a part of the directions)
  if (!(r.v2 > r.v1 + 3 && r.lit2 > 0.05)) fail(`bg_lr 0.05: held-out ${r.v1.toFixed(2)} -> ${r.v2.toFixed(2)} dB, ${(100 * r.lit2).toFixed(1)}% of the background lit`);
  // the default training with the background on: few saturated texels
  await page.goto(appUrl('?dataset=/dataset&ds=2&bg_lr=0.05'));
  await page.waitForFunction(() => !!window.app?.trainer && window.app.gtAvailable, null, { timeout: 120000 });
  const saturated = await page.evaluate(async () => {
    const app = window.app, tr = app.trainer;
    app.params.stop_after = 600;
    app._setTraining(true);
    for (const t0 = Date.now(); tr.stepCount < 600 || app.params.training; await new Promise(res => setTimeout(res, 100)))
      if (Date.now() - t0 > 240000) throw new Error(`stuck at step ${tr.stepCount}`);
    const rgb = await tr.readEnv();
    let sat = 0;
    for (let t = 0; t < rgb.length; t += 3) if (Math.max(rgb[t], rgb[t + 1], rgb[t + 2]) - Math.min(rgb[t], rgb[t + 1], rgb[t + 2]) > 100) sat++;
    return sat / (rgb.length / 3);
  });
  if (!(saturated < 0.003)) fail(`the room trained with bg_lr 0.05: ${(100 * saturated).toFixed(2)}% of the background's texels saturated`);
  return `the room trained 600 steps with bg_lr 0.05: ${(100 * saturated).toFixed(2)}% of the background's texels saturated; transparent splats, ${r.views} held-out views: ${r.v0.toFixed(2)} dB; bg_lr 0, 100 steps: ${r.v1.toFixed(2)} dB (background black); ` +
         `bg_lr 0.05, 300 steps: ${r.v2.toFixed(2)} dB, ${(100 * r.lit2).toFixed(0)}% of the background's values over 8 / 255`;
}, { serverArgs: [ROOM] });
