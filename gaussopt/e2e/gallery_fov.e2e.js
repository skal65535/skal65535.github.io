// gallery_fov.e2e.js — The gallery's fov (gallery/index.html, the 'camera' folder): in 'fly-by' on the path and in
// 'photo camera' the view takes the photos' field of view while "photos' fov" is on; a fov set with the slider turns
// it off and holds, the flight going on; turned on again, the photos' fov comes back. On GALLERY_SCENE (default SCENE
// below; gallery/scenes: the scenes' folder), served statically as the gallery is.
//   cd e2e && node gallery_fov.e2e.js
import { dirname } from 'node:path';
import { APP, fail, galleryShown, PORT, run } from './harness.js';

const SCENE = process.env.GALLERY_SCENE ?? 'canyon3', HELD = 35;
const url = (mode) => `http://localhost:${PORT}/gallery/index.html?scene=${SCENE}&mode=${encodeURIComponent(mode)}`;

await run(async (page) => {
  const row = (name) => page.locator('.skal-row').filter({ has: page.locator('.skal-row-label', { hasText: new RegExp(`^${name}$`) }) });
  const fov = async () => Number(await row('fov').locator('.skal-readout').textContent());
  const photos = () => row("photos' fov").locator('input[type=checkbox]');
  const frames = (n) => page.evaluate((n) => new Promise(r => { const f = (k) => k ? requestAnimationFrame(() => f(k - 1)) : r(); f(n); }), n);
  const out = [];
  for (const mode of ['fly-by', 'photo camera']) {
    await page.goto(url(mode));
    await galleryShown(page);
    if (!(await photos().isChecked())) fail(`${mode}: "photos' fov" off by default`);
    const own = await fov();
    if (Math.abs(own - HELD) < 2) fail(`${mode}: the photos' fov is ${own}, too close to the test's ${HELD}`);
    // the slider, as a drag sets it
    await row('fov').locator('input[type=range]').evaluate((el, v) => { el.value = v; el.dispatchEvent(new Event('input')); }, HELD);
    await frames(60);
    const held = await fov();
    if (held !== HELD) fail(`${mode}: the fov set to ${HELD} is ${held} 60 frames later`);
    if (await photos().isChecked()) fail(`${mode}: "photos' fov" still on after the slider`);
    await photos().evaluate(el => el.click());  // in the folder, collapsed
    await frames(10);
    const back = await fov();
    if (Math.abs(back - HELD) < 1) fail(`${mode}: "photos' fov" on again, the fov stays ${back}`);
    out.push(`${mode}: ${own.toFixed(0)} from the photos, ${HELD} held, ${back.toFixed(0)} back`);
  }
  return `${SCENE}: ${out.join('; ')}`;
}, { static: dirname(APP) });
