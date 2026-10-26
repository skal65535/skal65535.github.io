// photos_add.e2e.js — Add photos to a mapped photo set (DESIGN-sfm §13.3 P4): the set opened
// without its last ADD photos (capture order), mapped, then those added through the panel
// (matched against every photo, registered, pieces regrown), the cameras sent to the trainer, and
// undone.
//   node photos_add.e2e.js   (env: SFM_E2E_PHOTOS = a folder of JPEGs, default ~/gskal/brush/cuisine/images;
//                             SFM_E2E_ADD (default 4), CHROME, PORT, BROWSER)
import { copyFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { basename, join } from 'node:path';
import { HERE, fail, urlOf, appUrl, run } from './harness.js';

const SOURCE = process.env.SFM_E2E_PHOTOS ?? `${process.env.HOME}/gskal/brush/cuisine/images`;
const ADD = Number(process.env.SFM_E2E_ADD ?? 4);

if (!existsSync(SOURCE)) throw new Error(`no photos in ${SOURCE} (set SFM_E2E_PHOTOS)`);
const dir = join(HERE, '.cache', `photos_${basename(SOURCE === `${process.env.HOME}/gskal/brush/cuisine/images` ? 'cuisine' : SOURCE)}`);
mkdirSync(dir, { recursive: true });
const names = readdirSync(SOURCE).filter(n => /\.jpe?g$/i.test(n)).sort();  // Pixel names: capture order
for (const n of names) if (!existsSync(join(dir, n))) copyFileSync(join(SOURCE, n), join(dir, n));
const urls = names.map(n => urlOf(join(dir, n))), first = urls.slice(0, -ADD), later = urls.slice(-ADD);

await run(async (page) => {
  await page.goto(appUrl(`?photos=${first.map(encodeURIComponent).join(',')}`));
  await page.waitForFunction(() => window.app.sfm?.state.status === 'ready', null, { timeout: 60000 });
  await page.evaluate(() => window.app.sfm.run());
  let status = await page.waitForFunction(() => { const s = window.app.sfm.state.status; return /^(done|error)/.test(s) ? s : null; },
    null, { timeout: 15 * 60000, polling: 500 }).then(h => h.jsonValue());
  if (!status.startsWith('done')) throw new Error(status);
  const before = await page.evaluate(() => window.app.sfm.result.summary);
  console.log(`${first.length} photos: ${JSON.stringify(before)}`);
  if (!await page.evaluate(() => !window.app.sfm.addPhotosBtn._row_el.hidden)) fail('no Add photos button for a photo set');

  // add the later photos as the picker gives them (Files)
  await page.evaluate(async (later) => {
    const files = await Promise.all(later.map(async u => new File([await (await fetch(u)).blob()], u.split('/').pop(), { type: 'image/jpeg' })));
    window.app.sfm.addPhotos(files);
  }, later);
  status = await page.waitForFunction(() => { const s = window.app.sfm.state.status; return /addPhotos done|error|failed/.test(s) ? s : null; },
    null, { timeout: 15 * 60000, polling: 500 }).then(h => h.jsonValue());
  console.log(`add: ${status}`);
  if (!/addPhotos done/.test(status)) throw new Error(status);
  const q = await page.evaluate(() => {
    const q = window.app.sfm.quality;
    return { keyframes: q.frames.length, registered: q.frames.filter(f => f.registered).length, coverage: q.coverage, photos: window.app.sfm.photos.length };
  });
  console.log(`after add: ${JSON.stringify(q)}`);
  if (q.keyframes !== names.length || q.photos !== names.length) fail(`${q.keyframes} keyframes, ${q.photos} photos, not ${names.length}`);
  const sfmLog = await page.evaluate(() => window.app.logText('sfm'));
  console.log(sfmLog.split('\n').filter(l => /add photos|insert|regrown|to shoot/.test(l)).map(l => `  ${l}`).join('\n'));

  // undo: back to the first photos
  await page.evaluate(() => window.app.sfm.undo());
  await page.waitForFunction((n) => window.app.sfm.ready && window.app.sfm.quality.frames.length === n, first.length, { timeout: 60000 });
  const undone = await page.evaluate(() => window.app.sfm.photos.length);
  if (undone !== first.length) fail(`undo: ${undone} photos in the panel, not ${first.length}`);
  return `${first.length} photos: ${before.exported} cameras; + ${ADD}: ${q.coverage.inMain} in the main piece of ${q.keyframes}; undo back to ${undone}`;
}, { onFail: (page) => page.evaluate(() => window.app?.logText?.('sfm')) });
