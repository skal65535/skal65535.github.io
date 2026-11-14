// gallery_missing.e2e.js — The gallery on a scene listed whose scene.glb is not there yet (a run still in its SfM): the page
// says it is not made yet (a notice on the side canvas, the scene's line), nothing thrown, and the wheel over the canvas
// then zooms nothing, without an error (ZoomBy read the view, none yet); another scene picked then loads, the notice
// gone. Served statically as the gallery is.
//   cd e2e && node gallery_missing.e2e.js
import { dirname } from 'node:path';
import { APP, PORT, fail, run } from './harness.js';

const SCENE = '__no_such_scene__', OTHER = process.env.GALLERY_SCENE ?? 'canyon3';
// the expected one: the browser's own line for the 404 (the viewer's error for it would be another)
const expected = (e) => /status of 404/.test(e);

await run(async (page) => {
  // the scene listed (as a run's entry is from its start), its folder not there
  await page.route('**/gallery/scenes/index.json', async (route) => {
    const r = await route.fetch(), list = await r.json();
    await route.fulfill({ response: r, json: [...list, { id: SCENE, title: 'not made yet', photos: false, about: '' }] });
  });
  await page.goto(`http://localhost:${PORT}/gallery/index.html?scene=${SCENE}&mode=orbit`);
  await page.waitForFunction(() => document.querySelector('#main-canvas'));
  await page.waitForTimeout(1500);  // main() gives up on the scene
  const info = await page.locator('#scene-info').textContent();
  if (!/not made yet/.test(info)) fail(`the scene's line: '${info}'`);
  if (!(await page.locator('#side-canvas').isVisible())) fail('no notice on the side canvas');
  const box = await page.locator('#main-canvas').boundingBox();
  await page.mouse.move(box.x + 0.5 * box.width, box.y + 0.5 * box.height);
  for (const dy of [-100, -100, 100, -300]) await page.mouse.wheel(0, dy);
  await page.waitForTimeout(300);
  // another scene picked: loaded, the notice gone
  const row = page.locator('.skal-row').filter({ has: page.locator('.skal-row-label', { hasText: /^scene$/ }) });
  await row.locator('select').selectOption(JSON.stringify(OTHER));  // the panel's options: their values in JSON
  await page.waitForFunction((id) => document.getElementById('scene-info')?.textContent.length > 0 && !/not made yet/.test(document.getElementById('scene-info').textContent), OTHER, { timeout: 60000 });
  if (await page.locator('#side-canvas').isVisible()) fail(`${OTHER} loaded, the notice still shown`);
  return `the scene's line '${info.trim()}', the notice shown, the wheel on it without an error; ${OTHER} picked: loaded, the notice gone`;
}, { static: dirname(APP), ignored: expected });
