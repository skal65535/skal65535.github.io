// dome.e2e.js — A dome in place of the learned background (trainer.setDome, photo_runs/tools/dome.py): a cube map
// larger than env.js's ENV_N, read by the tile passes at its own size and never learned. The synthetic room
// (photo_runs/tools/synth_room.py), its splats nearly transparent and frozen as in env.e2e.js. The dome: 64 texels a
// face, each the color of its direction (d / 2 + 1 / 2, made here from env.js's layout, not from env_taps). A training
// view rendered with it and with a black dome: their difference over T_final (pix_state.x) is the dome's color at each
// pixel's ray (R^T K^-1 p), within 0.03 on average; read at ENV_N texels a face instead, or with the faces mislaid,
// it is not. Then 50 steps with bg_lr 0.05: the dome's texture read back the same, bit for bit (the learned background's
// update would copy its 32 x 32 a face over the dome's first texels).
//   cd e2e && node dome.e2e.js   (env: MESH_E2E_ROOM, the synthetic room's folder; see harness.js)
import { fail, appUrl, run, room, appReady } from './harness.js';

const ROOM = room();

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await appReady(page);
  const r = await page.evaluate(async () => {
    const app = window.app, tr = app.trainer, P = app.params, n = 64;
    Object.assign(P, { opt_method: 'adam', lr_pos: 0, lr_scale: 0, lr_quat: 0, lr_color: 0, lr_opacity: 0, opacity_reset: 0,
                       enable_rebalance: false, bg_lr: 0 });
    for (let i = 0; i < tr.count; i++) tr.cpuParams[i * 16 + 3] = -5.4;  // nearly transparent (env.e2e.js)
    tr._uploadSplats();
    // texel (face, row, col)'s direction: faces +x -x +y -y +z -z, u along the columns, v along the rows (env.js)
    const dir = (f, v, u) => [[1, -v, -u], [-1, -v, u], [u, 1, v], [u, -1, -v], [u, -v, 1], [-u, -v, -1]][f];
    const dome = new Uint8Array(18 * n * n);
    for (let f = 0; f < 6; f++) for (let row = 0; row < n; row++) for (let col = 0; col < n; col++) {
      const d = dir(f, (row + 0.5) / n * 2 - 1, (col + 0.5) / n * 2 - 1), l = Math.hypot(...d);
      for (let c = 0; c < 3; c++) dome[3 * ((f * n + row) * n + col) + c] = Math.round(255 * (d[c] / l / 2 + 0.5));
    }
    const view = 0, img = tr.images[view], k = tr.intrinsics(img), W = tr.trainW, H = tr.trainH;
    const render = async () => {
      const { pix, hat } = await tr.readView(view, ['pix', 'hat']);
      return { pix, rgb: hat };
    };
    tr.setDome(new Uint8Array(18 * n * n), n);
    const black = await render();
    tr.setDome(dome, n);
    const lit = await render();
    let err = 0, cnt = 0, lo = Infinity, hi = -Infinity;
    for (let y = 2; y < H; y += 7) for (let x = 3; x < W; x += 7) {
      const p = y * W + x, T = lit.pix[4 * p];
      if (T < 0.5) continue;
      const ray = [(x + 0.5 - k.cx) / k.fx, (y + 0.5 - k.cy) / k.fy, 1], R = img.R;
      const d = [0, 1, 2].map(i => R[i] * ray[0] + R[3 + i] * ray[1] + R[6 + i] * ray[2]), l = Math.hypot(...d);
      for (let c = 0; c < 3; c++) {
        const o = y * tr.rowPitch + 4 * x + c, want = d[c] / l / 2 + 0.5;
        err += Math.abs((lit.rgb[o] - black.rgb[o]) / 255 / T - want); cnt++;
        lo = Math.min(lo, want); hi = Math.max(hi, want);
      }
    }
    const readDome = async () => {  // the dome's texture, rgba32float, n x 6n
      const buf = tr.device.createBuffer({ size: n * 16 * 6 * n, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = tr.device.createCommandEncoder();
      enc.copyTextureToBuffer({ texture: tr.envTex }, { buffer: buf, bytesPerRow: n * 16 }, [n, 6 * n]);
      tr.device.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const f = new Float32Array(buf.getMappedRange().slice(0));
      buf.destroy();
      return f;
    };
    const dome0 = await readDome(), step0 = tr.stepCount;
    P.bg_lr = 0.05;
    P.stop_after = tr.stepCount + 50;
    app.setTraining(true);
    for (const t0 = Date.now(); tr.stepCount < P.stop_after || P.training; await new Promise(res => setTimeout(res, 100)))
      if (Date.now() - t0 > 120000) throw new Error(`stuck at step ${tr.stepCount}`);
    const dome1 = await readDome();
    let moved = 0;
    for (let i = 0; i < dome1.length; i++) if (dome1[i] !== dome0[i]) moved++;
    return { err: err / Math.max(cnt, 1), cnt, lo, hi, moved, steps: tr.stepCount - step0 };
  });
  if (!(r.cnt > 1000 && r.hi - r.lo > 0.3)) fail(`too few or too alike pixels: ${r.cnt}, colors ${r.lo.toFixed(2)}-${r.hi.toFixed(2)}`);
  if (!(r.err < 0.03)) fail(`the dome at each pixel's ray: mean error ${r.err.toFixed(3)} over ${r.cnt} samples`);
  if (!(r.steps >= 50 && r.moved === 0)) fail(`${r.steps} steps with bg_lr 0.05: ${r.moved} of the dome's values changed`);
  return `the dome (64 texels a face) at the pixels' rays: mean error ${r.err.toFixed(4)} over ${r.cnt} samples (colors ${r.lo.toFixed(2)}-${r.hi.toFixed(2)}); after ${r.steps} steps at bg_lr 0.05 none of its values changed`;
}, { serverArgs: [ROOM] });
