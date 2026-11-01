// octree_depth.e2e.js — The octree boxes against the splats' depth (render.js renderScene, wireframe_render): the
// synthetic room (photo_runs/tools/synth_room.py) trained 600 steps, the view a training camera's. The boxes drawn
// through the splats (octree_hidden 1) change some of the pixels; at the default (0.15) the boxes behind the walls are
// faint: clearly fewer changed (those in front of the walls and the objects still show; with the splats' depth not
// drawn, as many as drawn through).
//   cd e2e && node octree_depth.e2e.js   (env: MESH_E2E_ROOM, default ROOM below; see harness.js)
import { fail, appUrl, run, room, appReady } from './harness.js';

const ROOM = room();

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await appReady(page);
  await page.evaluate(async () => {
    const app = window.app;
    await (await import('./e2e/page_util.js')).train(app, 600, 240000);
    Object.assign(app.params, { show_cameras: false, show_path: false, show_splits: false });
  });
  // the canvas's pixels (rgba) with the given params, after a few frames
  const pixels = async (set) => {
    await page.evaluate((set) => { Object.assign(window.app.params, set); window.app._refreshWireframes(); }, set);
    await page.waitForTimeout(1500);
    const png = (await page.locator('#main-canvas').screenshot()).toString('base64');
    return page.evaluate(async (png) => Array.from(await (await import('./e2e/page_util.js')).pngPixels(png)), png);
  };
  const changed = (a, b) => {  // share of the pixels whose color moved by over 24 levels
    let n = 0;
    for (let i = 0; i < a.length; i += 4) if (Math.max(Math.abs(a[i] - b[i]), Math.abs(a[i + 1] - b[i + 1]), Math.abs(a[i + 2] - b[i + 2])) > 24) n++;
    return n / (a.length / 4);
  };
  const leaves = await page.evaluate(() => window.app.octree.leaves.length);
  const off = await pixels({ show_octree: false });
  const through = changed(off, await pixels({ show_octree: true, octree_hidden: 1 }));
  const faint = changed(off, await pixels({ show_octree: true, octree_hidden: 0.15 }));
  if (!(through > 0.02)) fail(`the boxes drawn through change ${(100 * through).toFixed(2)}% of the pixels (${leaves} leaves)`);
  if (!(faint < 0.8 * through && faint > 0)) fail(`behind the splats faint: ${(100 * faint).toFixed(2)}% of the pixels changed, drawn through ${(100 * through).toFixed(2)}%`);
  return `${leaves} leaves: drawn through ${(100 * through).toFixed(2)}% of the pixels changed, faint behind the splats ${(100 * faint).toFixed(2)}%`;
}, { serverArgs: [ROOM] });
