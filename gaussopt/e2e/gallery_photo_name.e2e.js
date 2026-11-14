// gallery_photo_name.e2e.js — The gallery's 'photo camera' (gallery/index.html): its slider's label names the photo (as
// the trainer and the runs' logs do), and the URL's photo= takes its index along the path or that name, with or without
// its extension; moving the slider writes its index into the URL. On a local gallery scene (GALLERY_SCENE, default
// jardin5-1m-rbg; gallery/scenes: the scenes' folder), its names read here from its scene.glb, served statically as the gallery is.
//   cd e2e && node gallery_photo_name.e2e.js
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { APP, PORT, fail, run } from './harness.js';
import { readSceneGlb } from '../src/scene_glb.js';

const SCENE = process.env.GALLERY_SCENE ?? 'jardin5-1m-rbg';
const bytes = readFileSync(join(dirname(APP), 'gallery', 'scenes', SCENE, 'scene.glb'));
const sc = readSceneGlb(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const stem = (k) => sc.cameras[sc.path[k]].name.replace(/\.[a-z0-9]+$/i, '');

await run(async (page) => {
  const row = page.locator('.skal-row').filter({ has: page.locator('.skal-row-label', { hasText: /^photo camera/ }) });
  const shown = async () => ({ label: (await row.locator('.skal-row-label').textContent()).trim(), value: Number(await row.locator('input').last().inputValue()) });
  const open = async (photo) => {
    await page.goto(`http://localhost:${PORT}/gallery/index.html?scene=${SCENE}&mode=photo%20camera&photo=${encodeURIComponent(photo)}`);
    await page.waitForFunction(() => document.getElementById('scene-info')?.textContent.length > 0, null, { timeout: 60000 });
    await page.waitForTimeout(500);
    return shown();
  };
  const want = (k) => `photo camera · ${stem(k)}`;
  const a = await open('5');
  if (a.value !== 5 || a.label !== want(5)) fail(`photo=5: ${JSON.stringify(a)}, want 5 and '${want(5)}'`);
  const b = await open(`${stem(7)}.jpg`);
  if (b.value !== 7 || b.label !== want(7)) fail(`photo=${stem(7)}.jpg: ${JSON.stringify(b)}, want 7 and '${want(7)}'`);
  const input = row.locator('input').last();
  await input.fill('9'); await input.press('Enter');
  await page.waitForTimeout(500);
  const c = await shown(), url = new URL(page.url());
  if (c.label !== want(9) || url.searchParams.get('photo') !== '9') fail(`the slider at 9: ${JSON.stringify(c)}, the URL's photo=${url.searchParams.get('photo')}`);
  return `${SCENE}: photo=5 -> '${a.label}'; photo=${stem(7)}.jpg -> 7, '${b.label}'; the slider at 9 -> '${c.label}', photo=9 in the URL`;
}, { static: dirname(APP) });
