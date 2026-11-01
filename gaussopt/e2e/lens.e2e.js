// lens.e2e.js — The lens refined by the splats (poses.js, trainer.js, DESIGN.md E19 3): the synthetic room trained 1000
// steps, its splats frozen, each training view's ground truth its own render (as pose.e2e.js); the lens then pushed off
// (its focal 1% longer, its principal point by (+2, -2) px, a residual distortion k1 0.03: shaders.wgsl distort_jac)
// and learned back (lens_lr 1e-3, the last third 2e-4; the poses not refined): the principal point comes back within a
// fifth of its push, the focal and k1 within a twentieth (k1 is learned with the focal through log f, the two trade:
// poses.js updateLens's a).
//   cd e2e && PORT=8791 node lens.e2e.js   (env: MESH_E2E_ROOM, the synthetic room's folder; see harness.js)
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
                       enable_rebalance: false, drop_rate: 0, lens_lr: 1e-3, pose_from: 0 });
    const views = tr.trainViews, steps = 150 * views.length;
    await photoIsRender(tr, views);
    const cams = [...new Set(views.map(v => tr.images[v].camera))];
    const truth = cams.map(c => ({ fx: c.fx, cx: c.cx, cy: c.cy }));
    for (const c of cams) {  // the state's a: log f + log(1 + k1 m), m the mean r^2 (poses.js updateLens)
      const m = (c.width ** 2 + c.height ** 2) / (12 * c.fx ** 2);
      tr.poses.updateLens(c, [0, 0, 0, 0], 0);
      tr.poses.lenses.get(c).p.set([Math.log(1.01 * (1 + 0.03 * m)), 2, -2, 0.03]);
    }
    tr.poses.apply(tr.images);
    const err = () => cams.map((c, i) => ({ f: Math.abs(c.fx / truth[i].fx - 1), c: Math.hypot(c.cx - truth[i].cx, c.cy - truth[i].cy), k: Math.abs(c.k1Residual ?? 0) }));
    const e0 = err();
    await train(steps * 2 / 3);
    P.lens_lr = 2e-4;
    await train(steps / 3);
    const e1 = err();
    return { lenses: cams.length, steps, f0: Math.max(...e0.map(e => e.f)), f1: Math.max(...e1.map(e => e.f)),
             c0: Math.max(...e0.map(e => e.c)), c1: Math.max(...e1.map(e => e.c)), k0: Math.max(...e0.map(e => e.k)), k1: Math.max(...e1.map(e => e.k)),
             poses: tr.poses.views.size };
  });
  if (r.poses) fail(`${r.poses} poses refined with pose_lr 0`);
  if (!(r.f1 < r.f0 / 20 && r.c1 < r.c0 / 5 && r.k1 < r.k0 / 20)) fail(`the lens learned back: focal ${(100 * r.f0).toFixed(3)} -> ${(100 * r.f1).toFixed(3)}%, principal point ${r.c0.toFixed(3)} -> ${r.c1.toFixed(3)} px, k1 ${r.k0} -> ${r.k1.toFixed(5)}`);
  return `${r.lenses} lens, ${r.steps} steps on frozen splats, each photo its render: the focal ${(100 * r.f0).toFixed(2)}% off -> ${(100 * r.f1).toFixed(3)}%, ` +
         `the principal point ${r.c0.toFixed(2)} px off -> ${r.c1.toFixed(3)} px, k1 ${r.k0} -> ${r.k1.toFixed(5)}`;
}, { serverArgs: [ROOM] });
