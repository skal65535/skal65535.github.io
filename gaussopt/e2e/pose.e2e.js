// pose.e2e.js — Each training view's pose refined by the splats (poses.js, trainer.js, DESIGN.md E19 2): the synthetic
// room (photo_runs/tools/synth_room.py) trained 1000 steps, then its splats frozen (lrs 0, no rebalance, no opacity
// reset), each training view's ground truth replaced by its own render (the poses then exact). Every training camera
// then pushed off by a known increment (a turn of 0.5 degree and a move of 1% of the scene's radius about a random
// axis, without their common part: the gauge), and the poses learned back (pose_lr 2e-3, the last third 4e-4): the
// turns and the moves left come under a fifth of those given; the held-out views take their neighbours' (none of
// them refined); a snapshot gives the poses back.
//   cd e2e && PORT=8791 node pose.e2e.js   (env: MESH_E2E_ROOM, default ~/gskal/brush/synth/room10; see harness.js)
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
    const train = async (n) => {  // n steps more
      P.stop_after = n;
      app._setTraining(true);
      const end = tr.stepCount + n;
      for (const t0 = Date.now(); tr.stepCount < end || P.training; await new Promise(res => setTimeout(res, 100)))
        if (Date.now() - t0 > 300000) throw new Error(`stuck at step ${tr.stepCount} of ${end}, training ${P.training}`);
      await tr._drainLossRead();
    };
    await train(1000);
    Object.assign(P, { opt_method: 'adam', lr_pos: 0, lr_scale: 0, lr_quat: 0, lr_color: 0, lr_opacity: 0, opacity_reset: 0,
                       enable_rebalance: false, drop_rate: 0, pose_lr: 2e-3, pose_from: 0 });
    const views = tr.trainViews, steps = 150 * views.length, W = tr.trainW, H = tr.trainH, bpr = tr.rowPitch;
    for (const v of views) {  // each training photo its render at its pose
      tr._lastHatKey = '';
      const out = await tr._renderViewForward(v), [hat] = await tr._readback('pose_gt', [{ tex: out.hatTex }]);
      tr.device.queue.writeTexture({ texture: out.gtTex }, hat, { bytesPerRow: bpr }, [W, H]);
    }
    const truth = tr.images.map(im => ({ R: im.R.slice(), center: im.center.slice() }));
    // the increments: 0.5 degree about a random axis, 1% of the radius along another, the gauge removed
    let seed = 3;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const unit = () => { const v = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5], n = Math.hypot(...v); return v.map(x => x / n); };
    for (const v of views) {
      tr.poses.update(tr.images[v], [0, 0, 0, 0, 0, 0], 0);  // a state for it
      tr.poses.views.get(tr.images[v].name).p.set([...unit().map(x => 0.01 * x), ...unit().map(x => 0.5 * Math.PI / 180 * x)]);
    }
    const byName = new Map(tr.images.map(im => [im.name, im]));
    tr.poses.center(byName);
    tr.poses.apply(tr.images);
    // the error of each training view's pose: its turn from the truth (degrees), its centre's move (x the radius)
    const R = tr.octree.sceneRadius;
    const errors = () => views.map(v => {
      const im = tr.images[v], T = truth[v];
      let tr3 = 0;
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) tr3 += (i === j ? 1 : 0) * [0, 1, 2].reduce((a, k) => a + im.R[3 * i + k] * T.R[3 * j + k], 0);
      return { deg: Math.acos(Math.min(1, (tr3 - 1) / 2)) * 180 / Math.PI, move: Math.hypot(...im.center.map((x, k) => x - T.center[k])) / R };
    });
    const mean = (a, k) => a.reduce((s, e) => s + e[k], 0) / a.length;
    const e0 = errors();
    tr._invalidateCaches();
    await train(steps * 2 / 3);
    P.pose_lr = 4e-4;
    await train(steps / 3);
    const e1 = errors();
    const names = new Set(tr.poses.views.keys());
    const heldOut = tr.valViews.length, heldOutMoved = tr.valViews.filter(v => !names.has(tr.images[v].name) && tr.images[v].sfmPose).length;
    // a snapshot: the poses restored with it
    const before = views.map(v => tr.images[v].view.slice()), { buf } = await app._encodeSnapshot('manual');
    await app.restoreSnapshot(buf, 'e2e.gosnap', app.colmap, ++app.loadGen);
    const tr2 = app.trainer;
    const snapErr = Math.max(...views.map((v, i) => Math.max(...tr2.images[v].view.map((x, j) => Math.abs(x - before[i][j])))));
    return { views: views.length, steps, deg0: mean(e0, 'deg'), deg1: mean(e1, 'deg'), move0: mean(e0, 'move'), move1: mean(e1, 'move'),
             learned: names.size, heldOut, heldOutMoved, snapErr, fresh: tr2 !== tr };
  });
  if (r.learned !== r.views) fail(`${r.learned} poses learned for ${r.views} training views`);
  if (!(r.deg1 < r.deg0 / 5 && r.move1 < r.move0 / 5)) fail(`the poses learned back: ${r.deg0.toFixed(3)} -> ${r.deg1.toFixed(3)} degrees, ${(100 * r.move0).toFixed(3)} -> ${(100 * r.move1).toFixed(3)}% of the radius`);
  if (r.heldOut && r.heldOutMoved !== r.heldOut) fail(`${r.heldOutMoved} of the ${r.heldOut} held-out views given their neighbours' increment`);
  if (!(r.fresh && r.snapErr < 1e-5)) fail(`the poses from a snapshot: off by ${r.snapErr} (a new trainer: ${r.fresh})`);
  return `${r.views} training views, ${r.steps} steps on frozen splats, each photo its render: pushed off by ${r.deg0.toFixed(3)} degrees and ` +
         `${(100 * r.move0).toFixed(3)}% of the radius (mean), learned back to ${r.deg1.toFixed(4)} degrees and ${(100 * r.move1).toFixed(4)}%; ` +
         `${r.heldOutMoved} held-out views given their neighbours'; restored from a snapshot`;
}, { serverArgs: [ROOM] });
