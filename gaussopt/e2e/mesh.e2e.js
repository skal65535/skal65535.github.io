// mesh.e2e.js — "Make mesh" and the mesh view (DESIGN.md §9 E5) in the page: the synthetic room
// (photo_runs/tools/synth_room.py) served as a dataset, 600 training steps, then the Train step's
// "Make mesh →": the page moves to the Export step (its mesh settings there, done with the mesh's summary once saved), the mesh
// (src/mesh.js on the trainer's forward renders, simplified in mesh_worker.js, textured from the photos
// by mesh_texture.js) reaches the save path as a .glb with a JPEG texture, whose vertices are nearly all
// inside the room (once turned back from glTF's +Y up to the app's world up), and is drawn in the 3D view
// from the first camera: 'mesh only' textured covers most of it, in color, closer to that camera's
// photo than with its vertex colors (the splat renders after 600 steps are blurry); shaded, all gray; 'mesh +
// splats' shows some of the mesh (the splats behind its surface hidden; drawn unhidden and opaque, they
// would cover nearly all of it); 'wireframe + splats' hides them too (a quad in front of the camera as
// the mesh: its diagonal alone drawn, no fill, no splat); without the mesh, the splats are colored.
// The save itself is captured (it would write into the dataset's splats_gaussopt/).
//   cd e2e && node mesh.e2e.js   (env: MESH_E2E_ROOM, default ~/gskal/brush/synth/room10; see harness.js)
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { APP, fail, appUrl, run } from './harness.js';

const { readGlb, rotationToY } = await import(join(APP, 'src', 'glb.js'));
const ROOM = process.env.MESH_E2E_ROOM ?? `${process.env.HOME}/gskal/brush/synth/room10`;
if (!existsSync(join(ROOM, 'scene_sdf.json'))) throw new Error(`no synthetic room in ${ROOM} (set MESH_E2E_ROOM)`);
const geo = JSON.parse(readFileSync(join(ROOM, 'scene_sdf.json'), 'utf8'));

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await page.waitForFunction(() => !!window.app?.trainer && window.app.gtAvailable, null, { timeout: 120000 });
  // 600 steps: at 300, with lr_opacity 0.01 (d576675), the splats under the floor have not faded yet and 60-122 of
  // ~51k vertices are 0.18-0.24 below it (0-6 at 0.05); at 600, 3-9 of ~65k
  await page.evaluate(() => { window.app.params.stop_after = 600; window.app._setTraining(true); });
  await page.waitForFunction(() => window.app.trainer.stepCount >= 600 && !window.app.params.training, null, { timeout: 300000 });
  const r = await page.evaluate(async () => {
    const app = window.app;
    const saved = new Promise(res => { app._saveOrDownload = async (bytes, name) => res({ bytes: Array.from(bytes), name }); });
    const show = app.showMesh;
    app.showMesh = function (m) { window.__shown = m; return show.call(this, m); };
    app.shell.show('train');
    app.shell.refresh();
    const make = [...document.querySelectorAll('#phase-foot button')].find(b => b.textContent === 'Make mesh →');
    if (!make || make.disabled) return { noButton: [...document.querySelectorAll('#phase-foot button')].map(b => b.textContent) };
    make.click();
    const phase = app.shell.current, settings = [...document.querySelectorAll('#export-mesh .skal-folder-title')].map(t => t.textContent);
    const got = await Promise.race([saved, new Promise(res => setTimeout(() => res(null), 120000))]);
    for (let t = 0; t < 50 && !app.meshState?.faces; t++) await new Promise(res => setTimeout(res, 100));  // saveMesh returns after the save
    app.shell.refresh();
    const step = document.querySelector('.step[data-phase="export"]');
    return got && { ...got, up: app.worldUp, view: app.params.mesh_view, log: app.logText().split('\n').filter(l => /Mesh/.test(l)), phase, settings,
                    exportStep: { cls: step.className, sum: step.querySelector('.step-sum').textContent } };
  });
  if (r?.noButton) fail(`no enabled "Make mesh →" in the Train step: ${JSON.stringify(r.noButton)}`);
  if (!r) throw new Error('nothing saved 120 s after Make mesh');
  if (r.phase !== 'export') fail(`Make mesh → shows the ${r.phase} step, not export`);
  if (!r.settings.includes('Mesh')) fail(`no Mesh settings in the Export step: ${JSON.stringify(r.settings)}`);
  if (!(/\bdone\b/.test(r.exportStep.cls) && /^\d+k faces · \d+² · [\d.]+ MB$/.test(r.exportStep.sum))) fail(`Export step after the mesh: ${JSON.stringify(r.exportStep)}`);
  if (!/^gaussopt_\d{5}\.mesh\.glb$/.test(r.name)) fail(`saved as ${r.name}`);
  const mesh = readGlb(Uint8Array.from(r.bytes).buffer, { gltfFrame: true }), nv = mesh.positions.length / 3, nf = mesh.faces.length / 3;
  if (!(nv > 1000 && nf > 1000)) fail(`${nv} vertices, ${nf} faces`);
  // the vertices inside the room, give or take 6%: all but a few (splats behind the walls), once turned
  // back by R^T (R takes the app's world up to glTF's +Y)
  const R = rotationToY(r.up);
  let out = 0;
  for (let v = 0; v < nv; v++) {
    const q = [0, 1, 2].map(k => mesh.positions[3 * v + k]), p = [0, 1, 2].map(c => R[c] * q[0] + R[3 + c] * q[1] + R[6 + c] * q[2]);
    for (let k = 0; k < 3; k++) {
      const [lo, hi] = [geo.room[0][k], geo.room[1][k]], pad = 0.06 * (hi - lo);
      if (!(p[k] > lo - pad && p[k] < hi + pad)) { out++; break; }
    }
  }
  if (out > 0.001 * nv) fail(`${out} of ${nv} vertices outside the room`);
  if (!r.log.some(l => /Mesh simplified to \d+ faces/.test(l))) fail(`no simplification log line: ${r.log.join(' | ')}`);
  if (!r.log.some(l => /Mesh textured: \d+ charts/.test(l))) fail(`no texture log line: ${r.log.join(' | ')}`);
  const tex = readGlb(Uint8Array.from(r.bytes).buffer);
  if (!(tex.texture?.mimeType === 'image/jpeg' && tex.texture.bytes[0] === 0xff && tex.texture.bytes[1] === 0xd8)) fail('no JPEG texture in the .glb');
  if (!(tex.uvs?.length === 2 * nv && tex.uvs.every(u => u >= 0 && u <= 1))) fail('uvs missing or outside [0, 1]');
  if (r.view !== 1) fail(`mesh_view ${r.view} after Make mesh (1: mesh + splats)`);

  // the view in 'mesh only' from the first camera: the share of canvas pixels drawn (alpha > 0), read in
  // the task that drew them (a WebGPU canvas is not readable once presented)
  const drawn = () => page.evaluate(async () => {
    const app = window.app, rs = app.renderer.renderScene;
    window.__cov = null;
    app.renderer.renderScene = function (...a) {
      rs.apply(this, a);
      const c = document.getElementById('main-canvas'), cv = new OffscreenCanvas(c.width, c.height), cx = cv.getContext('2d');
      cx.drawImage(c, 0, 0);
      const px = cx.getImageData(0, 0, c.width, c.height).data;
      // the photo of the camera the view is snapped to, at the canvas pixel (fitIntrinsics: scaled to fit, centered)
      const g = window.__gt, s = Math.min(c.width / g.cam.width, c.height / g.cam.height);
      const ox = 0.5 * (c.width - g.cam.width * s), oy = 0.5 * (c.height - g.cam.height * s);
      let n = 0, gray = 0, se = 0, ns = 0;
      for (let i = 0; i < px.length; i += 4) if (px[i + 3] > 0) {
        n++;
        if (Math.max(px[i], px[i + 1], px[i + 2]) - Math.min(px[i], px[i + 1], px[i + 2]) <= 2) gray++;
        const x = (i / 4) % c.width, y = Math.floor(i / 4 / c.width);
        const u = Math.floor(((x + 0.5 - ox) / s) * g.W / g.cam.width), v = Math.floor(((y + 0.5 - oy) / s) * g.H / g.cam.height);
        if (u < 0 || v < 0 || u >= g.W || v >= g.H) continue;
        for (let k = 0; k < 3; k++) { const d = px[i + k] - g.px[v * g.pitch + 4 * u + k]; se += d * d; }
        ns += 3;
      }
      window.__cov = { drawn: n / (px.length / 4), gray: gray / Math.max(1, n), psnr: 10 * Math.log10(255 * 255 * ns / Math.max(se, 1)) };
    };
    try {
      app.drawGen++;
      for (let t = 0; t < 50 && window.__cov === null; t++) await new Promise(res => setTimeout(res, 100));
      return window.__cov;
    } finally { app.renderer.renderScene = rs; }
  });
  await page.evaluate(() => {
    const app = window.app;
    Object.assign(app.params, { mesh_view: 2, mesh_shaded: true, show_cameras: false, show_octree: false, show_splits: false });
    app._refreshWireframes();
    app.flyToCamera(0);
  });
  await page.evaluate(async () => {  // camera 0's photo, at training size
    const tr = window.app.trainer, [px] = await tr._readback('e2e_gt', [{ tex: await tr.ensureGtTexture(0) }]);
    window.__gt = { px, pitch: tr.rowPitch, W: tr.trainW, H: tr.trainH, cam: tr.images[0].camera };
  });
  await new Promise(res => setTimeout(res, 2000));
  const withMesh = await drawn();
  await page.evaluate(() => { window.app.params.mesh_shaded = false; });
  const textured = await drawn();
  await page.evaluate(() => { const { image, ...m } = window.__shown; window.app.renderer.setMesh(m); });  // the same mesh, vertex colors
  const colored = await drawn();
  await page.evaluate(() => { window.app.renderer.setMesh(window.__shown); window.app.params.mesh_shaded = true; });
  await page.evaluate(() => { window.app.params.mesh_view = 1; });
  const over = await drawn();
  // 'wireframe + splats' with a quad 0.1 in front of the camera, wider than the view, as the mesh: all
  // the splats behind it hidden, its diagonal alone drawn (no fill)
  await page.evaluate(() => {
    const app = window.app, V = app.lastView.view, d = 0.1;  // column-major world to view: p = R^T (v - t)
    const world = (...v) => [0, 1, 2].map(j => [0, 1, 2].reduce((s, i) => s + V[4 * j + i] * (v[i] - V[12 + i]), 0));
    const positions = Float32Array.from([[-1, -1], [1, -1], [1, 1], [-1, 1]].flatMap(([x, y]) => world(4 * d * x, 4 * d * y, d)));
    app.renderer.setMesh({ positions, faces: Uint32Array.from([0, 1, 2, 0, 2, 3]) });
    app.params.mesh_view = 3;
  });
  const wall = await drawn();
  await page.evaluate(() => { window.app.renderer.setMesh(window.__shown); window.app.params.mesh_view = 1; });
  await page.evaluate(() => window.app.renderer.setMesh(null));
  const without = await drawn();
  if (!(withMesh.drawn > 0.3 && withMesh.gray > 0.99)) fail(`mesh only, shaded: ${(100 * withMesh.drawn).toFixed(1)}% of the canvas drawn, ${(100 * withMesh.gray).toFixed(1)}% of it gray`);
  if (!(textured.drawn > 0.3 && textured.gray < 0.5)) fail(`mesh only, textured: ${(100 * textured.drawn).toFixed(1)}% of the canvas drawn, ${(100 * textured.gray).toFixed(1)}% of it gray`);
  // measured: +1.5 to +2.8 dB (3 runs) with the centers' depth, +0.9 to +1.5 (6 runs) with the plane depth (the mesh
  // closer to the surface: the vertex colors better too), +2.3 to +3.3 (3 runs) at 600 steps; 13 dB below with the uvs read upside down
  if (!(textured.psnr > colored.psnr + 0.5)) fail(`textured not closer to the photo than vertex colors: ${textured.psnr.toFixed(2)} vs ${colored.psnr.toFixed(2)} dB`);
  // measured 9.3-9.7% with the centers' depth, 3.8-3.9% with the plane depth (tile_forward: the mesh on the splats'
  // own surface, fewer of them behind it), 6.6-6.7% at 600 steps; 0.7% with the splats drawn unhidden (most are on or just in front of it)
  if (!(over.gray > 0.02)) fail(`mesh + splats, shaded: ${(100 * over.gray).toFixed(1)}% gray (the mesh)`);
  if (!(wall.drawn < 0.02)) fail(`wireframe + splats behind a quad: ${(100 * wall.drawn).toFixed(1)}% of the canvas drawn (splats not hidden, or the quad filled)`);
  if (!(without.gray < 0.5)) fail(`mesh removed (the splats): ${(100 * without.gray).toFixed(1)}% gray`);
  return `${r.name}: ${nv} vertices, ${nf} faces, ${(r.bytes.length / 1e6).toFixed(2)} MB; mesh only (shaded) covers ${(100 * withMesh.drawn).toFixed(0)}% of the view, ` +
         `textured ${(100 * textured.drawn).toFixed(0)}%, ${textured.psnr.toFixed(2)} dB to the photo vs ${colored.psnr.toFixed(2)} with vertex colors; ` +
         `${(100 * withMesh.gray).toFixed(1)}% gray; mesh + splats ${(100 * over.gray).toFixed(1)}% gray; ` +
         `wireframe behind a quad ${(100 * wall.drawn).toFixed(1)}% drawn; the splats ${(100 * without.gray).toFixed(1)}% gray`;
}, { serverArgs: [ROOM] });
