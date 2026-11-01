// overlay.e2e.js — A video's on-screen display masked (src/sfm/overlay.js, DESIGN-sfm §5.1.1) through the page: a thermal
// drone video with burnt-in text bars, a version line, a reticle and a caption (OVERLAY_E2E_VIDEO, default
// SRC below) mapped and trained 300 steps. The analysis finds the mask (a share of the frame in 8-15%, the
// log says so), the overlay's keypoints are dropped, every camera the trainer gets has its mask, and its ground truths
// have alpha 0 there (tile_backward leaves those pixels out). With sfm_overlay=0: no mask anywhere.
//   cd e2e && node overlay.e2e.js
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { basename, join } from 'node:path';
import { HERE, fail, urlOf, appUrl, run } from './harness.js';

const SRC = process.env.OVERLAY_E2E_VIDEO ?? `${process.env.HOME}/test-parking.mov`;
if (!existsSync(SRC)) throw new Error(`no video ${SRC} (set OVERLAY_E2E_VIDEO)`);
const dir = join(HERE, '.cache', 'videos'), video = join(dir, basename(SRC));
mkdirSync(dir, { recursive: true });
if (!existsSync(video)) copyFileSync(SRC, video);

const pass = (page, query) => page.evaluate(async () => {
  const app = window.app, sleep = (ms) => new Promise(r => setTimeout(r, ms));
  while (app.sfm?.state.status !== 'ready') await sleep(200);
  app.sfm.run(null, true);
  for (const t0 = Date.now(); !(app.colmap?.fromSfm && app.trainer && app.gtAvailable); await sleep(500))
    if (Date.now() - t0 > 600000 || /^error/.test(app.sfm.state.status)) throw new Error(`no cameras: ${app.sfm.state.status}`);
  const tr = app.trainer, P = app.params;
  P.stop_after = 300; app.trainStartStep = tr.stepCount; app.setTraining(true);
  for (const t0 = Date.now(); tr.stepCount < 300 || P.training; await sleep(200)) if (Date.now() - t0 > 300000) throw new Error(`stuck at step ${tr.stepCount}`);
  const cams = [...new Set(app.colmap.images.map(im => im.camera))], masks = cams.map(c => c.mask);
  const share = masks[0] ? masks[0].data.reduce((a, b) => a + b, 0) / masks[0].data.length : 0;
  // a training view's ground truth: the share of its pixels with alpha 0
  const v = tr.trainViews[0], gt = await tr.ensureGtTexture(v);
  const [px] = await tr._readback('gt_alpha', [{ tex: gt }]);
  let zero = 0;
  for (let y = 0; y < tr.trainH; y++) for (let x = 0; x < tr.trainW; x++) zero += px[y * tr.rowPitch + 4 * x + 3] === 0;
  const log = app.logText('sfm');
  return { cams: cams.length, masked: masks.filter(Boolean).length, share, alpha0: zero / (tr.trainW * tr.trainH), psnr: tr.lastPsnr,
           found: log.split('\n').find(l => /overlay: /.test(l) && /masked|none/.test(l)) ?? null,
           dropped: log.split('\n').find(l => /keypoints on it dropped/.test(l)) ?? null };
});

let on;
await run(async (page) => {
  await page.goto(appUrl(`?video=${encodeURIComponent(urlOf(video))}`));
  await page.waitForFunction(() => !!window.app?.sfm, null, { timeout: 60000 });
  on = await pass(page);
  if (!(on.share > 0.08 && on.share < 0.15)) fail(`the mask: ${(100 * on.share).toFixed(1)}% of the frame (want 8-15%); ${on.found}`);
  if (!/masked/.test(on.found ?? '')) fail(`the log: ${on.found}`);
  if (!on.dropped) fail('no keypoints dropped on the overlay');
  if (on.masked !== on.cams) fail(`${on.masked} of ${on.cams} cameras with a mask`);
  if (!(Math.abs(on.alpha0 - on.share) < 0.03)) fail(`a training view's ground truth: ${(100 * on.alpha0).toFixed(1)}% alpha 0, the mask ${(100 * on.share).toFixed(1)}%`);
  return `${on.found?.replace(/^.*overlay: /, '')}; ${on.dropped?.replace(/^.*overlay: /, '')}; ${on.cams} camera(s) masked, a ground truth ${(100 * on.alpha0).toFixed(1)}% alpha 0; 300 steps: ${on.psnr.toFixed(2)} dB`;
});
await run(async (page) => {
  await page.goto(appUrl(`?video=${encodeURIComponent(urlOf(video))}&sfm_overlay=0`));
  await page.waitForFunction(() => !!window.app?.sfm, null, { timeout: 60000 });
  const off = await pass(page);
  if (off.masked || off.alpha0 > 0 || off.dropped) fail(`sfm_overlay=0: ${off.masked} cameras masked, ${(100 * off.alpha0).toFixed(1)}% alpha 0, ${off.dropped}`);
  return `sfm_overlay=0: no mask, no keypoint dropped; 300 steps: ${off.psnr.toFixed(2)} dB`;
});
