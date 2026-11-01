// heldout_fit.e2e.js — The held-out views scored with their camera fitted to their photo, the splats frozen
// (trainer.evaluateHeldOutFitted, DESIGN.md E19's held-out protocol): the synthetic room trained 1000 steps, its
// held-out photo replaced by its render (its pose then exact), then its camera pushed off by 0.5 degree and 1% of the
// scene's radius: the plain PSNR falls, the fitted one comes back within 1 dB of the exact pose's (capped at 50 dB: the
// same image is 99.9, 8-bit rounding keeps a fit under it); the splats, their Adam state and moments the same bit
// for bit after it, the view's pose put back.
//   cd e2e && PORT=8791 node heldout_fit.e2e.js   (env: MESH_E2E_ROOM, default ROOM below; see harness.js)
import { fail, appUrl, run, room, appReady } from './harness.js';

const ROOM = room();

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await appReady(page);
  const r = await page.evaluate(async () => {
    const app = window.app, tr = app.trainer, P = app.params;
    const { train, photoIsRender, splatState, changedCount } = await import('./e2e/page_util.js');
    await train(app, 1000);
    const { setPose } = await import('./src/colmap.js');
    const { expRot } = await import('./src/poses.js');
    await photoIsRender(tr, tr.valViews);
    const exact = (await tr.evaluateValidation()).psnr;
    // pushed off: 0.5 degree about (1, 2, -1), 1% of the radius along (-1, 0.5, 1), in the camera's frame
    const n = (v) => { const l = Math.hypot(...v); return v.map(x => x / l); }, rad = tr.octree.sceneRadius;
    const poses = tr.valViews.map(v => { const im = tr.images[v]; return { R: im.R.slice(), t: im.t.slice() }; });
    for (const v of tr.valViews) {
      const im = tr.images[v], E = expRot(n([1, 2, -1]).map(x => x * 0.5 * Math.PI / 180)), d = n([-1, 0.5, 1]).map(x => 0.01 * rad * x);
      const R = [0, 1, 2].flatMap(i => [0, 1, 2].map(j => E[3 * i] * im.R[j] + E[3 * i + 1] * im.R[3 + j] + E[3 * i + 2] * im.R[6 + j]));
      const t = [0, 1, 2].map(i => E[3 * i] * im.t[0] + E[3 * i + 1] * im.t[1] + E[3 * i + 2] * im.t[2] + d[i]);
      setPose(im, R, t);
    }
    const off = (await tr.evaluateValidation()).psnr;
    const state = () => splatState(tr, true);
    const s0 = await state(), views0 = tr.valViews.map(v => tr.images[v].view.slice());
    const fit = await tr.evaluateHeldOutFitted(60);
    const s1 = await state();
    const changed = changedCount(s0, s1);
    const poseBack = Math.max(...tr.valViews.map((v, i) => Math.max(...tr.images[v].view.map((x, j) => Math.abs(x - views0[i][j])))));
    return { exact, off, fit: fit.psnr, perView: fit.perView, changed, poseBack, views: tr.valViews.length };
  });
  if (!(r.views > 0)) return fail('no held-out view');
  if (!(r.off < r.exact - 5)) fail(`pushed off: ${r.exact.toFixed(2)} -> ${r.off.toFixed(2)} dB only`);
  if (!(r.fit > Math.min(r.exact, 50) - 1)) fail(`fitted: ${r.fit.toFixed(2)} dB, the exact pose ${r.exact.toFixed(2)}, pushed off ${r.off.toFixed(2)}`);
  if (r.changed.some(c => c)) fail(`the splats / Adam m / Adam v / moments changed by the fit: ${r.changed.join(' / ')} values`);
  if (r.poseBack !== 0) fail(`the held-out poses not put back: ${r.poseBack}`);
  return `${r.views} held-out view, its photo its render: ${r.exact.toFixed(2)} dB at its pose, ${r.off.toFixed(2)} pushed off by 0.5 degree ` +
         `and 1% of the radius, ${r.fit.toFixed(2)} fitted (60 steps; it turned ${r.perView[0][2].toFixed(3)} degree, moved ${(100 * r.perView[0][3]).toFixed(3)}%); ` +
         `the splats, Adam's state and the moments unchanged, the pose put back`;
}, { serverArgs: [ROOM] });
