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
//   cd e2e && node mesh.e2e.js   (env: MESH_E2E_ROOM, the synthetic room's directory; see harness.js)
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { APP, fail, appUrl, run, room, appReady } from './harness.js';

const { readGlb, rotationToY } = await import(join(APP, 'src', 'glb.js'));
const ROOM = room();
const geo = JSON.parse(readFileSync(join(ROOM, 'scene_sdf.json'), 'utf8'));

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await appReady(page);
  // 600 steps: by then the splats under the floor have faded (at 300, too many vertices below it)
  await page.evaluate(async () => (await import('./e2e/page_util.js')).train(window.app, 600));
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
    return got && { ...got, up: app.worldUp, view: app.params.mesh_view, shownVoxel: window.__shown?.voxel, log: app.logText().split('\n').filter(l => /Mesh/.test(l)), phase, settings,
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
  // the grid's voxel, as logged (3 digits), in the .glb and on the mesh shown
  const voxel = Number(r.log.find(l => /^Mesh: /.test(l))?.match(/voxel ([\d.e-]+)/)?.[1]);
  if (!(voxel > 0 && Math.abs(mesh.voxel - voxel) < 5e-3 * voxel && r.shownVoxel === mesh.voxel)) fail(`voxel: logged ${voxel}, .glb ${mesh.voxel}, shown ${r.shownVoxel}`);
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
  await page.evaluate(() => { Object.assign(window.app.params, { mesh_view: 1, mesh_margin: 0 }); });
  const over = await drawn();
  await page.evaluate(() => { window.app.params.mesh_margin = 2; });  // the splats up to 2 voxels behind the mesh drawn: less of it shows
  const overMargin = await drawn();
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
  // failing with the uvs read upside down
  if (!(textured.psnr > colored.psnr + 0.5)) fail(`textured not closer to the photo than vertex colors: ${textured.psnr.toFixed(2)} vs ${colored.psnr.toFixed(2)} dB`);
  // failing with the splats drawn unhidden (most are on or just in front of the mesh)
  if (!(over.gray > 0.02)) fail(`mesh + splats, shaded: ${(100 * over.gray).toFixed(1)}% gray (the mesh)`);
  // failing with the pre-pass at the mesh's own depth
  if (!(overMargin.gray < 0.85 * over.gray)) fail(`mesh + splats, margin 2 voxels: ${(100 * overMargin.gray).toFixed(1)}% gray against ${(100 * over.gray).toFixed(1)}% at 0`);
  if (!(wall.drawn < 0.02)) fail(`wireframe + splats behind a quad: ${(100 * wall.drawn).toFixed(1)}% of the canvas drawn (splats not hidden, or the quad filled)`);
  if (!(without.gray < 0.5)) fail(`mesh removed (the splats): ${(100 * without.gray).toFixed(1)}% gray`);

  // mesh_kind 'behind the splats' (mesh_grow.js): the same Make mesh, the solid grown from the splats and carved by
  // the cameras' rays; its vertices behind the room's surfaces (in the walls, the boxes, the sphere: their signed
  // distance to the free space, in voxels, below 0 at the median and over 2 for few), and over the splats from camera 0
  // less of it shows than of the surface mesh (it hides fewer of them), both at margin 0
  const b = await page.evaluate(async () => {
    const app = window.app;
    const saved = new Promise(res => { app._saveOrDownload = async (bytes, name) => res({ bytes: Array.from(bytes), name }); });
    app.params.mesh_kind = 'behind';
    app.saveMesh();
    const got = await Promise.race([saved, new Promise(res => setTimeout(() => res(null), 240000))]);
    for (let t = 0; t < 50 && app.meshState?.running; t++) await new Promise(res => setTimeout(res, 100));
    return got && { ...got, log: app.logText().split('\n').filter(l => /^Mesh: .*behind the splats/.test(l)), shownVoxel: window.__shown?.voxel };
  });
  if (!b) throw new Error('nothing saved 240 s after Make mesh (behind the splats)');
  const bm = readGlb(Uint8Array.from(b.bytes).buffer, { gltfFrame: true }), bnv = bm.positions.length / 3;
  const bVoxel = Number(b.log.at(-1)?.match(/voxel ([\d.e-]+)/)?.[1]);
  if (!(bVoxel > 0 && Math.abs(bm.voxel - bVoxel) < 5e-3 * bVoxel && b.shownVoxel === bm.voxel)) fail(`behind: voxel logged ${bVoxel}, .glb ${bm.voxel}, shown ${b.shownVoxel} (${b.log.join(' | ')})`);
  if (!(bnv > 1000)) fail(`behind: ${bnv} vertices`);
  const free = (p) => {  // signed distance to the free space (> 0 in it): inside the room, outside the boxes and the sphere
    let d = Math.min(...[0, 1, 2].map(k => Math.min(p[k] - geo.room[0][k], geo.room[1][k] - p[k])));
    for (const [lo, hi] of geo.boxes) {
      const o = [0, 1, 2].map(k => Math.max(lo[k] - p[k], 0, p[k] - hi[k])), inside = Math.min(...[0, 1, 2].map(k => Math.min(p[k] - lo[k], hi[k] - p[k])));
      d = Math.min(d, Math.hypot(...o) > 0 ? Math.hypot(...o) : -inside);
    }
    const [c, rad] = geo.sphere;
    return Math.min(d, Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]) - rad);
  };
  const freeOf = (m) => {  // the vertices' distances to the free space in voxels: [median, the share over 2 voxels in it]
    const n = m.positions.length / 3, d = [];
    for (let v = 0; v < n; v++) {
      const q = [0, 1, 2].map(k => m.positions[3 * v + k]);
      d.push(free([0, 1, 2].map(c => R[c] * q[0] + R[3 + c] * q[1] + R[6 + c] * q[2])) / m.voxel);
    }
    d.sort((x, y) => x - y);
    return [d[n >> 1], d.filter(x => x > 2).length / n];
  };
  const [med, front] = freeOf(bm), [sMed, sFront] = freeOf(mesh);
  if (!(med < -1 && med < sMed - 1 && front < 0.15)) fail(`behind: free-space distance median ${med.toFixed(2)} voxels, ${(100 * front).toFixed(1)}% over 2 voxels in front; the surface mesh ${sMed.toFixed(2)}, ${(100 * sFront).toFixed(1)}%`);
  await page.evaluate(() => { Object.assign(window.app.params, { mesh_view: 1, mesh_margin: 0, mesh_shaded: true }); window.app.flyToCamera(0); });
  await new Promise(res => setTimeout(res, 2000));
  const overBehind = await drawn();
  if (!(overBehind.gray < over.gray)) fail(`mesh + splats, margin 0: behind the splats ${(100 * overBehind.gray).toFixed(1)}% gray, not below the surface mesh's ${(100 * over.gray).toFixed(1)}%`);
  return `${r.name}: ${nv} vertices, ${nf} faces, ${(r.bytes.length / 1e6).toFixed(2)} MB; mesh only (shaded) covers ${(100 * withMesh.drawn).toFixed(0)}% of the view, ` +
         `textured ${(100 * textured.drawn).toFixed(0)}%, ${textured.psnr.toFixed(2)} dB to the photo vs ${colored.psnr.toFixed(2)} with vertex colors; ` +
         `${(100 * withMesh.gray).toFixed(1)}% gray; mesh + splats ${(100 * over.gray).toFixed(1)}% gray, ${(100 * overMargin.gray).toFixed(1)}% with a margin of 2 voxels; ` +
         `wireframe behind a quad ${(100 * wall.drawn).toFixed(1)}% drawn; the splats ${(100 * without.gray).toFixed(1)}% gray; ` +
         `behind the splats: ${bnv} vertices, free-space distance median ${med.toFixed(2)} voxels (${(100 * front).toFixed(1)}% over 2 in front; the surface ` +
         `mesh ${sMed.toFixed(2)}, ${(100 * sFront).toFixed(1)}%), ` +
         `mesh + splats at margin 0 ${(100 * overBehind.gray).toFixed(1)}% gray against ${(100 * over.gray).toFixed(1)}%`;
}, { serverArgs: [ROOM] });
