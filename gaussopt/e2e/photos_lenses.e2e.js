// photos_lenses.e2e.js — A photo set mixing lenses, zoom and orientations (DESIGN-sfm §13.3 P8): a
// camera model per lens and zoom, portraits rotated; then the label filter "only the ultrawide
// photos", mapped again from the cached features; then the filter undone.
//   node photos_lenses.e2e.js   (env: SFM_E2E_PHOTOS = a folder of JPEGs, default ~/gskal/brush/out_scene1/images;
//                                SFM_E2E_MIN_ALL (default 13, COLMAP's count with a camera per lens and zoom),
//                                SFM_E2E_LENS, SFM_E2E_MIN_LENS (default the ultrawide, 7), CHROME, PORT, BROWSER)
import { copyFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { HERE, fail, urlOf, appUrl, run } from './harness.js';

const SOURCE = process.env.SFM_E2E_PHOTOS ?? `${process.env.HOME}/gskal/brush/out_scene1/images`;
const MIN_ALL = Number(process.env.SFM_E2E_MIN_ALL ?? 13);
const LENS = process.env.SFM_E2E_LENS ?? 'Pixel 10 back camera 1.854mm f/2.2';
const MIN_LENS = Number(process.env.SFM_E2E_MIN_LENS ?? 7);

if (!existsSync(SOURCE)) throw new Error(`no photos in ${SOURCE} (set SFM_E2E_PHOTOS)`);
const dir = join(HERE, '.cache', 'photos_lenses');
mkdirSync(dir, { recursive: true });
const names = readdirSync(SOURCE).filter(n => /\.jpe?g$/i.test(n)).sort();
for (const n of names) if (!existsSync(join(dir, n))) copyFileSync(join(SOURCE, n), join(dir, n));
const urls = names.map(n => urlOf(join(dir, n)));

const mapped = (page) => page.waitForFunction(() => { const s = window.app.sfm.state.status; return /^(done|error|nothing to train)/.test(s) ? s : null; },
  null, { timeout: 15 * 60000, polling: 500 }).then(h => h.jsonValue());
const trained = (page, n) => page.waitForFunction((n) => window.app.colmap?.images?.length === n && !!window.app.trainer, n, { timeout: 120000 });

await run(async (page) => {
  await page.goto(appUrl(`?photos=${urls.map(encodeURIComponent).join(',')}`));
  await page.waitForFunction(() => window.app.sfm?.state.status === 'ready', null, { timeout: 60000 });
  await page.evaluate(() => window.app.sfm.run());
  let status = await mapped(page);
  if (!status.startsWith('done')) throw new Error(status);
  const all = await page.evaluate(() => window.app.sfm.result.summary);
  await trained(page, all.exported);
  const cams = await page.evaluate(() => window.app.colmap.cameras.size);
  console.log(`every lens: ${all.exported} cameras to the trainer on ${cams} camera models, ${all.registered}/${all.keyframes} registered`);
  if (all.exported < MIN_ALL) fail(`${all.exported} cameras with every lens, < ${MIN_ALL}`);
  if (cams < 2) fail(`${cams} camera model(s): a camera per lens expected`);

  await page.evaluate((lens) => window.app.sfm.setPhotoFilter({ lens: [lens] }), LENS);
  await page.waitForFunction(() => /Run to map them/.test(window.app.sfm.state.status), null, { timeout: 60000 });
  await page.evaluate(() => window.app.sfm.run());
  status = await mapped(page);
  if (!status.startsWith('done')) throw new Error(status);
  const one = await page.evaluate(() => window.app.sfm.result.summary);
  const log = await page.evaluate(() => window.app.logText('sfm'));
  const cached = log.match(/features: \d+ photos at \d+ x \d+ \((\d+) extracted, (\d+) cached\)/g)?.at(-1);
  console.log(`only ${LENS}: ${one.exported} cameras, ${one.registered}/${one.keyframes} registered; ${cached}`);
  if (one.exported < MIN_LENS || one.keyframes !== MIN_LENS) fail(`only ${LENS}: ${one.exported}/${one.keyframes}, not ${MIN_LENS}/${MIN_LENS}`);
  if (!/\(0 extracted/.test(cached ?? '')) fail(`the filtered photos were extracted again: ${cached}`);

  // undo the filter: the map of every lens again, with its cameras; then nothing left to undo
  await page.evaluate(() => window.app.sfm.undo());
  await page.waitForFunction((n) => window.app.sfm.ready && window.app.sfm.result?.summary.keyframes === n, all.keyframes, { timeout: 120000 });
  const back = await page.evaluate(() => ({ summary: window.app.sfm.result.summary, source: window.app.sfm.state.source, changes: window.app.sfm.filterChanges }));
  console.log(`undo: ${back.summary.exported} cameras of ${back.summary.keyframes} (${back.source}), filter changes left ${back.changes}`);
  if (back.summary.exported !== all.exported || back.changes !== 0) fail(`undo: ${JSON.stringify(back)}`);
  await page.evaluate(() => window.app.sfm.undo());
  await page.waitForFunction(() => window.app.sfm.state.status === 'nothing to undo', null, { timeout: 60000 });
  return `every lens ${all.exported} cameras on ${cams} models; only one lens ${one.exported}/${one.keyframes}; undo back to ${back.summary.exported}`;
}, { onFail: (page) => page.evaluate(() => window.app?.logText?.('sfm')) });
