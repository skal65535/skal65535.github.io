// pose_refit.e2e.js — The training views' cameras refitted with the splats frozen (trainer._refitCameras, pose_refit;
// DESIGN.md E19 2): the synthetic room trained 1000 steps, each training view's photo its own render (its pose then
// exact), every training camera pushed off by 0.5 degree and 1% of the radius (the gauge removed, as pose.e2e.js), then
// one refit (60 steps a view): the turns and moves left under a fifth of those given; the splats, their Adam state the
// same bit for bit; the exposures not kept (exposure_lr 0). Then the schedule: pose_refit 30, pose_from 30 steps on: a
// refit logged 30 and 60 steps after pose_from (the first after the poses had pose_refit steps).
//   cd e2e && PORT=8791 node pose_refit.e2e.js   (env: MESH_E2E_ROOM, the synthetic room's dataset; see harness.js)
import { fail, appUrl, run, room, appReady } from './harness.js';

const ROOM = room();

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await appReady(page);
  const r = await page.evaluate(async () => {
    const app = window.app, tr = app.trainer, P = app.params;
    const { train: trainOf, photoIsRender, splatState, changedCount } = await import('./e2e/page_util.js'), train = (n) => trainOf(app, n);
    await train(1000);
    Object.assign(P, { opt_method: 'adam', lr_pos: 0, lr_scale: 0, lr_quat: 0, lr_color: 0, lr_opacity: 0, opacity_reset: 0,
                       enable_rebalance: false, drop_rate: 0, pose_lr: 1e-9, pose_from: 0 });
    const views = tr.trainViews;
    await photoIsRender(tr, views);
    const truth = tr.images.map(im => ({ R: im.R.slice(), center: im.center.slice() }));
    let seed = 7;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const unit = () => { const v = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5], n = Math.hypot(...v); return v.map(x => x / n); };
    for (const v of views) {
      tr.poses.update(tr.images[v], [0, 0, 0, 0, 0, 0], 0);
      tr.poses.views.get(tr.images[v].name).p.set([...unit().map(x => 0.01 * x), ...unit().map(x => 0.5 * Math.PI / 180 * x)]);
    }
    tr.poses.center(new Map(tr.images.map(im => [im.name, im])));
    tr.poses.apply(tr.images);
    const R = tr.octree.sceneRadius;
    const errors = () => views.map(v => {
      const im = tr.images[v], T = truth[v];
      let t3 = 0;
      for (let i = 0; i < 9; i++) t3 += im.R[i] * T.R[i];
      return { deg: Math.acos(Math.min(1, (t3 - 1) / 2)) * 180 / Math.PI, move: Math.hypot(...im.center.map((x, k) => x - T.center[k])) / R };
    });
    const mean = (a, k) => a.reduce((s, e) => s + e[k], 0) / a.length;
    const state = () => splatState(tr);
    const e0 = errors(), s0 = await state();
    const info = await tr._withStepLock(() => tr._refitCameras(60));
    const e1 = errors(), s1 = await state();
    const changed = changedCount(s0, s1);
    // the schedule
    Object.assign(P, { pose_refit: 30, pose_refit_iters: 2, pose_from: tr.stepCount + 30 });
    const logged = [];
    const log0 = app.log.bind(app);
    app.log = (msg, ...a) => { if (/^Cameras refitted/.test(msg)) logged.push(msg); return log0(msg, ...a); };
    const from = P.pose_from + 30;
    await train(91);
    return { views: views.length, deg0: mean(e0, 'deg'), deg1: mean(e1, 'deg'), move0: mean(e0, 'move'), move1: mean(e1, 'move'), info, changed,
             expos: tr.exposure.views.size, from, logged };
  });
  if (!(r.deg1 < r.deg0 / 5 && r.move1 < r.move0 / 5)) fail(`refitted: ${r.deg0.toFixed(3)} -> ${r.deg1.toFixed(4)} degree, ${(100 * r.move0).toFixed(3)} -> ${(100 * r.move1).toFixed(4)}% of the radius`);
  if (r.changed.some(c => c)) fail(`the splats / Adam m / Adam v changed by the refit: ${r.changed.join(' / ')} values`);
  if (r.expos) fail(`${r.expos} exposures kept with exposure_lr 0`);
  const at = r.logged.map(m => Number(m.match(/@ step (\d+)/)[1]));
  if (at.join() !== [r.from, r.from + 30].join()) fail(`refits logged at steps ${at.join(', ')}, expected ${r.from}, ${r.from + 30}`);
  return `${r.views} training views pushed off by ${r.deg0.toFixed(3)} degree and ${(100 * r.move0).toFixed(3)}% of the radius (mean), one refit ` +
         `(60 steps a view, ${(r.info.ms / 1000).toFixed(1)} s): ${r.deg1.toFixed(4)} degree and ${(100 * r.move1).toFixed(4)}%; the splats and their Adam state ` +
         `unchanged, no exposure kept; the schedule's refits at steps ${at.join(', ')}`;
}, { serverArgs: [ROOM] });
