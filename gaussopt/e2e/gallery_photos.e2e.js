// gallery_photos.e2e.js — The gallery's 'photos in frustums' (gallery/index.html): a toggle, off by default, whose
// click draws the photos of the nearest cameras in their frustums (the canvas changes) and whose second click takes
// them away (the canvas as before); photos=N in the URL starts it on. On a scene with photos.glb (GALLERY_SCENE, default
// canyon3; gallery/scenes: the scenes' folder), served statically as the gallery is (serve.py does not serve the scenes' link).
// The photos read by their byte ranges (scene_glb.js openPhotosGlb; photos.glb served with Range here, from the file, as
// the gallery's server does; python's http.server ignores Range): the header, then one range per photo drawn, far less
// than the file; after a reload none of them asked again (Cache Storage). Without Range (the whole file): drawn too.
//   cd e2e && node gallery_photos.e2e.js
import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { APP, NAME, PORT, browser, fail } from './harness.js';

const SCENE = process.env.GALLERY_SCENE ?? 'canyon3';
const url = (q = '') => `http://localhost:${PORT}/gallery/index.html?scene=${SCENE}&mode=orbit${q}`;

const srv = spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1'], { cwd: dirname(APP), stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1500));
const { page, close, errors } = await browser();
const t0 = Date.now();
// photos.glb with Range, from the file: the ranges asked ([start, end]) kept
const glb = readFileSync(join(dirname(APP), 'gallery', 'scenes', SCENE, 'photos.glb')), ranges = [];
const ranged = (route) => {
  const m = /^bytes=(\d+)-(\d+)$/.exec(route.request().headers().range ?? '');
  if (!m) return route.fulfill({ status: 200, body: glb, headers: { 'Content-Type': 'model/gltf-binary' } });
  const a = +m[1], b = Math.min(+m[2], glb.length - 1);
  ranges.push([a, b]);
  return route.fulfill({ status: 206, body: glb.subarray(a, b + 1), headers: { 'Content-Type': 'model/gltf-binary',
    'Content-Range': `bytes ${a}-${b}/${glb.length}`, 'Accept-Ranges': 'bytes' } });
};
try {
  const summary = await (async () => {
  // the middle of the canvas as shown (a PNG; the corners hold the status labels, which change by themselves)
  const pixels = async () => {
    await page.waitForTimeout(3000);
    const r = await page.locator('#main-canvas').boundingBox();
    return (await page.screenshot({ clip: { x: r.x + r.width / 4, y: r.y + r.height / 4, width: r.width / 2, height: r.height / 2 } })).toString('base64');
  };
  const changed = (a, b) => { const x = Buffer.from(a, 'base64'), y = Buffer.from(b, 'base64'); return x.length !== y.length || !x.equals(y); };
  const toggle = () => page.locator('.skal-row', { hasText: 'photos in frustums' }).locator('input[type=checkbox]');
  const shown = () => page.waitForFunction(() => document.getElementById('scene-info')?.textContent.length > 0, null, { timeout: 120000 });
  // without Range: the whole file, the photos drawn from it
  await page.goto(url());
  await shown();
  await page.evaluate(() => caches.delete('gaussopt-photos'));
  const plain0 = await pixels();
  await toggle().click();
  if (!changed(plain0, await pixels())) fail('without Range (the whole photos.glb): the photos not drawn');
  // with Range
  await page.route('**/photos.glb', ranged);
  await page.goto(url());
  await page.waitForFunction(() => document.getElementById('scene-info')?.textContent.length > 0, null, { timeout: 120000 });
  if (await toggle().count() !== 1) return fail("no 'photos in frustums' checkbox");
  if (await toggle().isChecked()) fail('photos in frustums on by default');
  const off = await pixels();
  await toggle().click();
  const on = await pixels();
  if (!changed(off, on)) fail('photos in frustums turned on: the view did not change');
  const first = ranges.filter(([a]) => a > 0), read = ranges.reduce((s, [a, b]) => s + b - a + 1, 0);
  if (!first.length || !ranges.some(([a]) => a === 0)) fail(`photos.glb by its ranges: ${ranges.length} asked (${first.length} photos)`);
  if (!(read < 0.5 * glb.length)) fail(`photos.glb by its ranges: ${read} bytes read of ${glb.length}`);
  ranges.length = 0;
  await toggle().click();
  const off2 = await pixels();
  if (changed(off, off2)) fail('photos in frustums turned off again: the view is not as before');
  // a reload: the same photos from Cache Storage
  await page.goto(url('&photos=8'));
  await shown();
  if (!(await toggle().isChecked())) fail('photos=8 in the URL: photos in frustums off');
  if (!changed(off, await pixels())) fail('after a reload with photos=8: the photos not drawn');
  const again = ranges.filter(([a]) => a > 0 && first.some(([c]) => c === a));
  if (again.length) fail(`after a reload, ${again.length} of the ${first.length} photos read again (not from Cache Storage)`);
  const cached = await page.evaluate(async () => (await (await caches.open('gaussopt-photos')).keys()).length);
  return `${SCENE}: photos in frustums off by default, on and off by its checkbox (the view changes and comes back), on with photos=8; ` +
         `photos.glb (${(glb.length / 1e6).toFixed(1)} MB): ${first.length} photos in ${(read / 1e3).toFixed(0)} kB by their ranges, ` +
         `none again after a reload (${cached} in Cache Storage); drawn from the whole file without Range`;
  })();
  const serious = await errors();
  if (serious.length) fail(`${serious.length} page errors:\n  ${serious.slice(0, 10).join('\n  ')}`);
  if (!process.exitCode) console.log(`PASS in ${NAME} (${((Date.now() - t0) / 1000).toFixed(0)} s): ${summary}`);
} catch (e) {
  fail(e.message);
} finally {
  await close();
  srv.kill();
}
