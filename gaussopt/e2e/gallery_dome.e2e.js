// gallery_dome.e2e.js — The gallery's 'dome only' (gallery/index.html): the scene's background (scene.glb env: a dome, or
// the learned one) drawn alone, without the splats nor the mesh, to see what the dome took. Off by default; its click
// changes the view and a second one brings it back; on, the 'background (dome)' toggle does not hide it, nor does the
// mesh menu draw over it; dome-only in the URL starts it on. On a scene with a background (GALLERY_SCENE;
// gallery/scenes: the scenes' folder), served statically as the gallery is.
//   cd e2e && node gallery_dome.e2e.js
import { dirname } from 'node:path';
import { APP, PORT, changed, fail, galleryShown, pixels, run } from './harness.js';

const SCENE = process.env.GALLERY_SCENE ?? 'canyon3-bg';
const url = (q = '') => `http://localhost:${PORT}/gallery/index.html?scene=${SCENE}&mode=orbit${q}`;

await run(async (page) => {
  const row = (name) => page.locator('.skal-row').filter({ has: page.locator('.skal-row-label', { hasText: new RegExp(`^${name.replace(/[()]/g, '\\$&')}$`) }) });
  const toggle = (name) => row(name).locator('input[type=checkbox]');
  await page.goto(url());
  await galleryShown(page);
  if (await toggle('dome only').count() !== 1) return fail("no 'dome only' checkbox");
  if (await toggle('dome only').isChecked()) fail('dome only on by default');
  if (await toggle('dome only').isDisabled()) fail(`${SCENE}: 'dome only' disabled on a scene with a background`);
  const off = await pixels(page);
  await toggle('dome only').click();
  const on = await pixels(page);
  if (!changed(off, on)) fail('dome only turned on: the view did not change (the splats still drawn)');
  await toggle('background (dome)').click();
  if (changed(on, await pixels(page))) fail("dome only: 'background (dome)' off hid it");
  await toggle('background (dome)').click();
  await row('mesh').locator('select').selectOption('mesh only');
  if (changed(on, await pixels(page))) fail("dome only: the mesh menu's 'mesh only' drew over it");
  await row('mesh').locator('select').selectOption('splats only');
  await toggle('dome only').click();
  if (changed(off, await pixels(page))) fail('dome only turned off again: the view is not as before');
  await page.goto(url('&dome-only'));
  await galleryShown(page);
  if (!(await toggle('dome only').isChecked())) fail('dome-only in the URL: dome only off');
  if (changed(on, await pixels(page))) fail('after a reload with dome-only: not the background alone');
  return `${SCENE}: 'dome only' off by default, on and off by its checkbox (the view changes and comes back), ` +
         `kept with 'background (dome)' off and with the mesh menu's 'mesh only', on with dome-only`;
}, { static: dirname(APP) });
