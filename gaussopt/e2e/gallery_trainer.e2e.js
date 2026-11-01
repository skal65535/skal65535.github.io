// gallery_trainer.e2e.js — The gallery's viewer (gallery/index.html) draws a scene as the trainer does: the same splats,
// the same camera, the viewer's render against the trainer's (its tile rasterizer, trainer._renderViewForward) by PSNR.
// The cameras: a training view's own (VIEW) and that view turned 8 degrees about the point at its render's median depth.
// The trainer: the photos (PHOTOS) mapped again (the SfM is deterministic: the scene's cameras), the scene's scene.glb
// loaded as a checkpoint (app.loadSceneGlb), the view's pose replaced; the viewer: its page with the camera and the
// field of view set through a hook patched into the served index.html. On a local gallery scene (GALLERY_SCENE,
// default pavilion-cams-off, the synthetic Barcelona Pavilion's SfM run; gallery/scenes: the scenes' folder) and its
// photos (PHOTOS), at its training size (ds 2). Without the trainer's 0.3 px^2 low-pass in the viewer: 25.84 / 26.02 dB;
// with it 29.71 / 29.41 (pavilion-cams-off, p001).
//   cd e2e && node gallery_trainer.e2e.js
import { spawn } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readdirSync, writeFileSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { APP, HERE, NAME, PORT, SAFARI, appUrl, browser, fail, server, urlOf } from './harness.js';

const SCENE = process.env.GALLERY_SCENE ?? 'pavilion-cams-off';
const PHOTOS = process.env.PHOTOS ?? `${process.env.HOME}/gskal/brush/synth/bl_pabellon_p200/photos`;
const VIEW = process.env.VIEW ?? 'p001_', QUERY = process.env.QUERY ?? 'ds=2&max_splats=1000000&stipple=0';
const MIN_DB = [28, 28];  // the view's own camera, turned

const glb = join(dirname(APP), 'gallery', 'scenes', SCENE, 'scene.glb');
if (!existsSync(glb) || !existsSync(PHOTOS)) {
  console.log(`SKIP in ${NAME}: no ${glb} or no ${PHOTOS} (GALLERY_SCENE, PHOTOS)`);
  process.exit(0);
}
const dir = join(HERE, '.cache', `photos_${basename(resolve(PHOTOS, '..'))}`);
mkdirSync(dir, { recursive: true });
const names = readdirSync(PHOTOS).filter(n => /\.jpe?g$/i.test(n)).sort();
for (const n of names) if (!existsSync(join(dir, n))) copyFileSync(join(PHOTOS, n), join(dir, n));

// the camera turned by `deg` about its own down axis, around the point `depth` ahead (R row-major world -> camera)
const mul = (A, B) => [0, 1, 2].flatMap(i => [0, 1, 2].map(j => A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j]));
const mv = (A, x) => [0, 1, 2].map(i => A[3 * i] * x[0] + A[3 * i + 1] * x[1] + A[3 * i + 2] * x[2]);
const tr3 = (A) => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
function turned(R, t, deg, depth) {
  const a = deg * Math.PI / 180, Ry = [Math.cos(a), 0, Math.sin(a), 0, 1, 0, -Math.sin(a), 0, Math.cos(a)];
  const c = mv(tr3(R), t).map(x => -x), P = c.map((x, i) => x + depth * R[6 + i]);
  const c2 = mv(mul(tr3(R), mul(Ry, R)), c.map((x, i) => x - P[i])).map((x, i) => x + P[i]);
  const R2 = mul(tr3(Ry), R);
  return { R: R2, t: mv(R2, c2).map(x => -x), center: c2 };
}
const viewOf = ({ R, t }) => [R[0], R[3], R[6], 0, R[1], R[4], R[7], 0, R[2], R[5], R[8], 0, t[0], t[1], t[2], 1];

// RGB PSNR of two same-sized PNG screenshots / renders, decoded in the page
const psnr = (page, a, b) => page.evaluate(async ([a, b]) => {
  const px = async (s) => {
    const im = await createImageBitmap(await (await fetch(`data:image/png;base64,${s}`)).blob());
    const c = new OffscreenCanvas(im.width, im.height), x = c.getContext('2d');
    x.drawImage(im, 0, 0);
    return x.getImageData(0, 0, im.width, im.height).data;
  };
  const [p, q] = [await px(a), await px(b)];
  if (p.length !== q.length) return NaN;
  let e = 0;
  for (let i = 0; i < p.length; i += 4) for (let c = 0; c < 3; c++) e += (p[i + c] - q[i + c]) ** 2;
  return 10 * Math.log10(255 * 255 / (e / (3 * p.length / 4)));
}, [a, b]);

if (SAFARI) {  // the viewer's hook is a Playwright route
  console.log(`SKIP in ${NAME}: Chrome only`);
  process.exit(0);
}
const srv = await server([]);
// the gallery served statically on PORT + 1, as it is (serve.py does not follow gallery/scenes' link out of its root)
const GPORT = PORT + 1, gsrv = spawn('python3', ['-m', 'http.server', String(GPORT), '--bind', '127.0.0.1'], { cwd: dirname(APP), stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1500));
const { page, close, errors } = await browser();
const t0 = Date.now();
try {
  const summary = await (async () => {
  copyFileSync(glb, join(HERE, '.cache', 'gallery_trainer.glb'));
  await page.goto(appUrl(`?photos=${names.map(n => encodeURIComponent(urlOf(join(dir, n)))).join(',')}&${QUERY}`));
  await page.waitForFunction(() => window.app?.sfm?.state.status === 'ready', null, { timeout: 120000 });
  await page.evaluate(() => window.app.sfm.run());
  await page.waitForFunction(() => /^(done|error)/.test(window.app.sfm.state.status), null, { timeout: 600000, polling: 1000 });
  // the hand-over's trainer, on all the cameras exported (as train.mjs)
  await page.waitForFunction(() => window.app.colmap?.images?.length === window.app.sfm.result?.summary?.exported && !!window.app.trainer && window.app.gtAvailable, null, { timeout: 300000 });
  await page.evaluate(async (url) => window.app.loadSceneGlb(await (await fetch(url)).arrayBuffer(), 'scene.glb'), urlOf(join(HERE, '.cache', 'gallery_trainer.glb')));
  await page.waitForFunction(() => window.app.gtAvailable, null, { timeout: 300000 });
  // the trainer's renders: the view's own camera, then turned; its median depth the pivot
  const T = await page.evaluate(async (view) => {
    const tr = window.app.trainer, v = tr.images.findIndex(im => im.name.startsWith(view));
    if (v < 0) return { error: `no view ${view}` };
    const W = tr.trainW, H = tr.trainH, img = tr.images[v];
    // a read-back render (rows rowPitch bytes apart) as PNG, base64
    window.__png = async (px) => {
      const bpr = tr.rowPitch, c = new OffscreenCanvas(W, H), x = c.getContext('2d'), im = x.createImageData(W, H);
      for (let y = 0; y < H; y++) for (let i = 0; i < W; i++) for (let k = 0; k < 4; k++) im.data[4 * (y * W + i) + k] = k === 3 ? 255 : px[y * bpr + 4 * i + k];
      x.putImageData(im, 0, 0);
      const b = new Uint8Array(await (await c.convertToBlob({ type: 'image/png' })).arrayBuffer());
      let s = ''; for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode(...b.subarray(i, i + 0x8000));
      return btoa(s);
    };
    // the median depth of its render, by the view's own pose
    tr._lastHatKey = ''; tr._invalidateCaches();
    await tr._renderViewForward(v);
    const [pix] = await tr._readback('gallery_trainer', [{ buf: tr.pixStateBuf, bytes: W * H * 16 }]);
    const z = []; for (let i = 0; i < pix.length; i += 16) if (pix[i] < 0.5) z.push(pix[i + 1]);
    z.sort((a, b) => a - b);
    return { v, W, H, fx: tr._intrinsics(img).fx, R: Array.from(img.R), t: Array.from(img.t), depth: z[z.length >> 1] };
  }, VIEW);
  if (T.error) return fail(T.error);
  const cams = [{ R: T.R, t: T.t, center: mv(tr3(T.R), T.t).map(x => -x) }, turned(T.R, T.t, 8, T.depth)];
  // each camera's render, the view's pose set to it (colmap.js setPose) and put back. Its own pose too: rendered
  // without setPose after the scene's load, its lower half came back black (2026-11-01, not looked into)
  const trainer = await page.evaluate(async ({ v, cams }) => {
    const tr = window.app.trainer, img = tr.images[v], out = [];
    const { setPose } = await import(new URL('src/colmap.js', location.href).href);
    const R0 = Array.from(img.R), t0 = Array.from(img.t);
    try {
      for (const { R, t } of cams) {
        setPose(img, R, t);
        tr._lastHatKey = ''; tr._invalidateCaches();
        const r = await tr._renderViewForward(v);
        const [hat] = await tr._readback('gallery_trainer', [{ tex: r.hatTex }]);
        out.push(await window.__png(hat));
      }
    } finally { setPose(img, R0, t0); tr._invalidateCaches(); }
    return out;
  }, { v: T.v, cams });

  // the viewer, in the same page: the canvas alone at the trainer's size, its camera and horizontal field of view ours
  await page.setViewportSize({ width: T.W, height: T.H });
  await page.route('**/gallery/index.html*', async (route) => {
    const r = await route.fetch(); let s = await r.text();
    const a = 'function current_camera(dt) {', b = '  render.fx = 1. / Math.tan(params.fov * Math.PI / 360.);';
    if (s.split(a).length !== 2 || s.split(b).length !== 2) throw new Error('gallery/index.html: the hook points moved');
    s = s.replace(a, `${a}\n  if (window.__cam) return window.__cam;`).replace(b, `  if (window.__fov) params.fov = window.__fov;\n${b}`);
    await route.fulfill({ response: r, body: s, headers: { ...r.headers(), 'content-type': 'text/html' } });
  });
  await page.goto(`http://localhost:${GPORT}/gallery/index.html?scene=${SCENE}&mode=orbit&canvas-only&cameras-hidden`);
  await page.waitForFunction(() => document.getElementById('scene-info')?.textContent.length > 0, null, { timeout: 120000 });
  const fov = 2 * Math.atan(T.W / (2 * T.fx)) * 180 / Math.PI, got = [];
  for (const [k, c] of cams.entries()) {
    await page.evaluate(({ view, center, fov }) => { window.__cam = { view, center }; window.__fov = fov; window.dispatchEvent(new Event('resize')); }, { view: viewOf(c), center: c.center, fov });
    // drawn: every splat sent (scene-info comes before), then a frame the same as the last ('idle')
    await page.waitForTimeout(1000);
    await page.waitForFunction(() => /^idle/.test(document.getElementById('fps')?.textContent ?? ''), null, { timeout: 120000 });
    const shot = (await page.locator('canvas').first().screenshot()).toString('base64');
    got.push(await psnr(page, shot, trainer[k]));
    if (process.env.KEEP_PNG) for (const [n, b] of [['viewer', shot], ['trainer', trainer[k]]]) writeFileSync(join(HERE, '.cache', `gallery_trainer_${n}_${k}.png`), Buffer.from(b, 'base64'));
  }
  const label = ['its own camera', 'turned 8 degrees'];
  got.forEach((db, k) => { if (!(db >= MIN_DB[k])) fail(`${VIEW} ${label[k]}: the viewer's render ${db.toFixed(2)} dB from the trainer's (at least ${MIN_DB[k]})`); });
  return `${SCENE} ${VIEW}: the viewer against the trainer ${got.map((db, k) => `${db.toFixed(2)} dB ${label[k]}`).join(', ')}`;
  })();
  const serious = await errors();
  if (serious.length) fail(`${serious.length} page errors:\n  ${serious.slice(0, 10).join('\n  ')}`);
  if (!process.exitCode) console.log(`PASS in ${NAME} (${((Date.now() - t0) / 1000).toFixed(0)} s): ${summary}`);
} catch (e) {
  fail(e.message);
} finally {
  await close();
  srv.kill();
  gsrv.kill();
}
