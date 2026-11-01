// scene_glb.e2e.js — The scene's .glb (src/scene_glb.js) through the page: the synthetic room (photo_runs/tools/synth_room.py)
// trained 300 steps with the background on (bg_lr), leveled by 10 degrees (the Level transform), saved by Save scene
// (app.saveSceneGlb; the bytes caught instead of written); the page reloaded, the file opened as a picked file would
// be (app.loadGlbFile): the same splats (positions exact), octree leaves, held-out PSNR (to 0.02 dB: the rotations,
// scales and opacities go through their linear forms), the leveling and the background. Save photos (.glb): every photo
// of the dataset, by name, byte for byte as served.
//   cd e2e && node scene_glb.e2e.js   (env: MESH_E2E_ROOM, the synthetic room's folder; see harness.js)
import { fail, appUrl, run, room, appReady } from './harness.js';

const ROOM = room();

// the app's save caught: { name: base64 }
const catchSaves = (page) => page.evaluate(() => {
  window.__saved = {};
  window.app._saveOrDownload = async (bytes, name) => {
    let s = '';
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    window.__saved[name] = btoa(s);
  };
});

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2&bg_lr=0.05'));
  await appReady(page);
  await catchSaves(page);
  const before = await page.evaluate(async () => {
    const app = window.app, tr = app.trainer;
    await (await import('./e2e/page_util.js')).train(app, 300, 240000);
    app.tiltView(10);
    await app.saveSceneGlb();
    await app.savePhotosGlb();
    await tr.syncSplatsFromGpu();
    const env = await tr.readEnv();
    return { count: tr.count, pos: Array.from(tr.cpuParams.filter((_, i) => i % 16 < 3)), leaves: app.octree.leaves.length,
             psnr: (await tr.evaluateValidation()).psnr, transform: [...app.sceneTransform], env: Array.from(env), saved: Object.keys(window.__saved),
             photos: app.colmap.images.map(im => [im.name, im.imageUrl]) };
  });
  const scene = before.saved.find(n => n.endsWith('.scene.glb'));
  if (!scene || !before.saved.includes('gaussopt_photos.glb')) fail(`saved ${before.saved.join(', ')}`);
  const files = await page.evaluate(() => window.__saved);

  // the photos' .glb: every photo as served
  const photosOk = await page.evaluate(async ({ b64, photos }) => {
    const { readPhotosGlb } = await import('/gaussopt/src/scene_glb.js');
    const r = readPhotosGlb(Uint8Array.from(atob(b64), c => c.charCodeAt(0)).buffer);
    let same = 0;
    for (const [name, url] of photos) {
      const p = r.photos.find(q => q.name === name), want = new Uint8Array(await (await fetch(url)).arrayBuffer());
      if (p && p.bytes.length === want.length && p.bytes.every((v, i) => v === want[i])) same++;
    }
    return { same, n: r.photos.length };
  }, { b64: files['gaussopt_photos.glb'], photos: before.photos });
  if (photosOk.same !== before.photos.length || photosOk.n !== before.photos.length) fail(`photos .glb: ${photosOk.same} of ${before.photos.length} as served (${photosOk.n} in it)`);

  // a fresh page: the scene opened as a picked file
  await page.goto(appUrl('?dataset=/dataset&ds=2&bg_lr=0.05'));
  await appReady(page);
  const after = await page.evaluate(async ({ b64, name }) => {
    const app = window.app;
    await app.loadGlbFile(new File([Uint8Array.from(atob(b64), c => c.charCodeAt(0))], name));
    const tr = app.trainer;
    await tr.syncSplatsFromGpu();
    return { count: tr.count, pos: Array.from(tr.cpuParams.filter((_, i) => i % 16 < 3)), leaves: app.octree.leaves.length,
             psnr: (await tr.evaluateValidation()).psnr, transform: [...app.sceneTransform], env: Array.from(await tr.readEnv()) };
  }, { b64: files[scene], name: scene });
  const bad = [];
  if (after.count !== before.count) bad.push(`${after.count} splats, saved ${before.count}`);
  else if (after.pos.some((v, i) => v !== before.pos[i])) bad.push('positions differ');
  if (after.leaves !== before.leaves) bad.push(`${after.leaves} leaves, saved ${before.leaves}`);
  if (!(Math.abs(after.psnr - before.psnr) < 0.02)) bad.push(`held out ${after.psnr.toFixed(3)} dB, saved ${before.psnr.toFixed(3)}`);
  if (after.transform.some((v, i) => Math.abs(v - before.transform[i]) > 1e-12) || before.transform[0] === 1) bad.push(`leveling ${after.transform.slice(0, 3)} vs ${before.transform.slice(0, 3)}`);
  if (after.env.some((v, i) => v !== before.env[i])) bad.push('background differs');
  if (bad.length) fail(bad.join('; '));
  return `${scene} (${(files[scene].length * 0.75 / 1e6).toFixed(2)} MB): ${after.count} splats, ${after.leaves} leaves, held out ${before.psnr.toFixed(3)} -> ${after.psnr.toFixed(3)} dB, ` +
         `the leveling and the background back; gaussopt_photos.glb: ${photosOk.n} photos as served`;
}, { serverArgs: [ROOM] });
