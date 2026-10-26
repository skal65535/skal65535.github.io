// photos.e2e.js — A photo set end to end in the page (DESIGN-sfm §13, P1): the photos are copied
// into .cache/ (served) and opened with ?photos=<urls> (fetched as Files, as a drop gives them); the worker plans them
// (EXIF, labels, rotation, one camera), shows the Photos phase and its contact sheet (P5), extracts SIFT
// on "Find cameras", maps every pair, hands the cameras to the
// trainer with the shooting advice shown, and training on every photo (no hold-out) fits them.
//   node photos.e2e.js   (env: SFM_E2E_PHOTOS = a folder of JPEGs, default ~/gskal/brush/cuisine/images;
//                         SFM_E2E_MIN_REGISTERED, CHROME, PORT, BROWSER)
import { copyFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { basename, join } from 'node:path';
import { HERE, NAME, fail, urlOf, appUrl, run } from './harness.js';

const SOURCE = process.env.SFM_E2E_PHOTOS ?? `${process.env.HOME}/gskal/brush/cuisine/images`;
const MIN_REGISTERED = Number(process.env.SFM_E2E_MIN_REGISTERED ?? 12);  // COLMAP's count on cuisine (DESIGN-sfm §13.2)
const TRAIN_STEPS = 1000;
// a smoke floor: cuisine's training views scored 13.4-16.9 dB after ~1000 steps with 3 photos held out
// (photo_runs/logs/train__cuisine_*); a flat image of the mean colour scores 13.1-13.4 dB on those 3
const MIN_TRAIN_PSNR = 12;

if (!existsSync(SOURCE)) throw new Error(`no photos in ${SOURCE} (set SFM_E2E_PHOTOS)`);
const dir = join(HERE, '.cache', `photos_${basename(SOURCE === `${process.env.HOME}/gskal/brush/cuisine/images` ? 'cuisine' : SOURCE)}`);
mkdirSync(dir, { recursive: true });
const names = readdirSync(SOURCE).filter(n => /\.jpe?g$/i.test(n)).sort();
for (const n of names) if (!existsSync(join(dir, n))) copyFileSync(join(SOURCE, n), join(dir, n));
const urls = names.map(n => urlOf(join(dir, n)));

await run(async (page) => {
  const t0 = Date.now();
  await page.goto(appUrl(`?photos=${urls.map(encodeURIComponent).join(',')}`));
  if (!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter())) throw new Error(`no WebGPU adapter in ${NAME}`);
  await page.waitForFunction(() => !!window.app?.sfm, null, { timeout: 60000 });
  await page.waitForFunction(() => window.app.sfm?.state.status === 'ready', null, { timeout: 60000 });
  const source = await page.evaluate(() => window.app.sfm.state.source);
  console.log(`source: ${source}`);
  // the Photos phase (DESIGN-sfm §13.3 P5): its label, no Keyframes step, no timeline tab, the contact
  // sheet of every photo with its labels, and a "Find cameras" button that runs
  await page.waitForFunction((n) => document.querySelectorAll('#frames-sheet figure').length === n, urls.length, { timeout: 120000 });
  const ui = await page.evaluate(() => ({
    title: document.getElementById('phase-title').textContent,
    steps: [...document.querySelectorAll('#go-stepper .step')].filter(b => getComputedStyle(b).display !== 'none').map(b => b.querySelector('.step-label').textContent),
    timeline: getComputedStyle(document.querySelector('.dock-tabs button[data-tab="timeline"]')).display !== 'none',
    sheet: document.getElementById('frames-sheet').classList.contains('shown'),
    caption: document.querySelector('#frames-sheet figcaption')?.textContent,
    find: [...document.querySelectorAll('#phase-foot button')].find(b => /Find cameras/.test(b.textContent))?.disabled ?? null,
  }));
  console.log(`Photos phase: ${JSON.stringify(ui)}`);
  if (ui.title !== 'Photos' || ui.steps.join() !== 'Project,Photos,Cameras,Train,Export' || ui.timeline || !ui.sheet) fail(`Photos phase: ${JSON.stringify(ui)}`);
  if (!/^p001 · 24 mm · portrait · 0 s$/.test(ui.caption ?? '')) fail(`photo caption: ${ui.caption}`);
  if (ui.find !== false) fail(`Find cameras button: ${ui.find === null ? 'missing' : 'disabled'}`);
  await page.evaluate(() => [...document.querySelectorAll('#phase-foot button')].find(b => /Find cameras/.test(b.textContent)).click());
  const status = await page.waitForFunction(() => {
    const s = window.app.sfm.state.status;
    return /^(done|error)/.test(s) ? s : null;
  }, null, { timeout: 15 * 60000, polling: 1000 }).then(h => h.jsonValue());
  const sfmLog = await page.evaluate(() => window.app.logText('sfm'));
  console.log(sfmLog.split('\n').map(l => `  ${l}`).join('\n'));
  if (!status.startsWith('done')) throw new Error(status);
  const summary = await page.evaluate(() => window.app.sfm.result.summary);
  console.log(`SfM: ${JSON.stringify(summary)} (${((Date.now() - t0) / 1000).toFixed(0)} s wall)`);
  if (summary.exported < MIN_REGISTERED) fail(`${summary.exported} cameras to the trainer, < ${MIN_REGISTERED}`);

  await page.waitForFunction((n) => window.app.colmap?.images?.length === n && !!window.app.trainer, summary.exported, { timeout: 120000 });
  const names = await page.evaluate(() => window.app.colmap.images.map(im => im.name));
  if (!names.every(n => /^p\d+_PXL_.*\.jpg$/.test(n))) fail(`photo names: ${names.slice(0, 3).join(', ')}`);
  // the camera path (P7): cuisine's capture order goes back and forth, the shortest path is offered;
  // 'shortest' reorders the path, 'capture' restores name order
  const path = await page.evaluate(() => {
    const app = window.app, c = app.colmap, offer = app.logText().match(/camera path: the shortest is ([\d.]+)x shorter[^\n]*/)?.[0] ?? null;
    app.params.path_order = 'shortest'; app._applyPathOrder();
    const shortest = [...c.path];
    app.params.path_order = 'capture'; app._applyPathOrder();
    return { offer, shortest, capture: [...c.path], n: c.images.length };
  });
  console.log(`camera path: ${path.offer}; shortest ${path.shortest.join(' ')}`);
  if (!path.offer) fail('no shortest-path offer for cuisine');
  if (path.shortest.length !== path.n || new Set(path.shortest).size !== path.n || path.shortest.join() === path.capture.join()) fail(`shortest path: ${path.shortest}`);
  if (path.capture.join() !== [...Array(path.n).keys()].join()) fail(`capture order: ${path.capture}`);
  // no held-out photo (DESIGN-sfm §13.3 P6): every view trains; scored on the training views
  if (await page.evaluate(() => window.app.colmap.images.some(im => im.isValidation))) fail('a photo set holds out views');
  const advice = await page.evaluate(() => { const el = window.app.sfm.adviceEl; return el.hidden ? null : el.innerText; });
  if (!advice) fail('no shooting advice in the Cameras phase');
  console.log(`advice:\n${advice.split('\n').map(l => `  ${l}`).join('\n')}`);
  await page.waitForFunction(() => window.app.gtAvailable, null, { timeout: 120000 });
  await page.evaluate((n) => { window.app.params.stop_after = n; window.app._setTraining(true); }, TRAIN_STEPS);
  await page.waitForFunction((n) => window.app.trainer.stepCount >= n && !window.app.params.training, TRAIN_STEPS, { timeout: 15 * 60000, polling: 1000 });
  const psnr = await page.evaluate(async () => {
    const tr = window.app.trainer, keep = tr.valViews;
    tr.valViews = tr.images.map((im, i) => i);
    try { const r = await tr.evaluateValidation(); tr.valHistory.pop(); return r.psnr; } finally { tr.valViews = keep; }
  });
  console.log(`training: ${TRAIN_STEPS} steps, PSNR over the training views ${psnr.toFixed(2)} dB`);
  if (!(psnr > MIN_TRAIN_PSNR)) fail(`training views at ${psnr.toFixed(2)} dB, <= ${MIN_TRAIN_PSNR}`);
  return `${summary.exported} cameras of ${summary.keyframes} photos, focal ${summary.focal.toFixed(0)} px, training views ${psnr.toFixed(2)} dB @ ${TRAIN_STEPS}`;
}, { onFail: (page) => page.evaluate(() => window.app?.logText?.('sfm')) });
