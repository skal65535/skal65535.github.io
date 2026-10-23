// sfm.e2e.js — Browser end-to-end smoke test of the in-browser SfM (DESIGN-sfm §8.1, M4):
// headless Chrome with WebGPU opens the page with ?video=<clip>, the Video → Cameras panel runs
// S0..S5 in its worker, hands the model to the trainer, and training reaches a validation PSNR;
// then a repair and its undo (M5), repair all and its undo, densify + a camera update of the running trainer (splats
// kept), and a reload restoring the session (OPFS).
//   cd e2e && npm install && npm test   (env: SFM_E2E_VIDEO, SFM_E2E_SECONDS, SFM_E2E_DUMP=dir, CHROME, PORT)
//   BROWSER=safari npm test: the same in Safari through safaridriver (safari.js; a window, not headless)
// The clip is the first SFM_E2E_SECONDS (default 12) of jardin5.mp4, cut with ffmpeg into .cache/.
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { APP, SECONDS, PORT, NAME, fail, clip, urlOf, server, browser } from './harness.js';

const VAL_STEPS = 500;
// Validation PSNR after VAL_STEPS: the default 12 s clip (35 views) reaches 17.0-17.5 dB; on long
// videos 500 steps are about one pass over the views, so only a sanity floor applies
const MIN_PSNR = Number(process.env.SFM_E2E_MIN_PSNR ?? (SECONDS === 12 ? 16 : 12));

const video = clip();
const srv = await server();
const { page, close, errors } = await browser();
const t0 = Date.now();
try {
  await page.goto(`http://localhost:${PORT}${urlOf(join(APP, 'index.html'))}?video=${encodeURIComponent(urlOf(video))}&val=${VAL_STEPS}`);
  if (!await page.evaluate(async () => !!await navigator.gpu?.requestAdapter())) throw new Error(`no WebGPU adapter in ${NAME}`);

  // the panel opens from ?video=, then runs on demand
  await page.waitForFunction(() => !!window.app?.sfm, null, { timeout: 60000 });
  await page.waitForFunction(() => window.app.sfm?.state.status === 'ready', null, { timeout: 60000 });
  await page.evaluate(() => {  // the stages the status line went through, so none runs silently
    window.seenStages = new Set();
    setInterval(() => window.seenStages.add(window.app.sfm.state.status.replace(/[ …].*$/, '')), 50);
    window.app.sfm.run();
  });
  const status = await page.waitForFunction(() => {
    const s = window.app.sfm.state.status;
    return /^(done|error)/.test(s) ? s : null;
  }, null, { timeout: 15 * 60000, polling: 1000 }).then(h => h.jsonValue());
  const sfmLog = await page.evaluate(() => window.app.logText('sfm'));
  console.log(sfmLog.split('\n').map(l => `  ${l}`).join('\n'));
  if (!status.startsWith('done')) throw new Error(status);
  const stages = await page.evaluate(() => [...window.seenStages]);
  console.log(`status stages: ${stages.join(', ')}`);
  for (const s of ['matching', 'mapping:', 'refining', 'undistorting'])
    if (!stages.includes(s)) throw new Error(`the status line never showed ${s}`);
  const summary = await page.evaluate(() => window.app.sfm.result.summary);
  if (process.env.SFM_E2E_DUMP) {  // selection + model, for comparisons with video2splat.py / tools/run_frames.js
    const dump = await page.evaluate(() => ({ selection: window.app.sfm.frameAnalysis, names: window.app.sfm.result.images.map(i => i.name),
                                              sparse: Object.fromEntries(Object.entries(window.app.sfm.result.sparse).map(([k, b]) => [k, Array.from(new Uint8Array(b))])) }));
    mkdirSync(process.env.SFM_E2E_DUMP, { recursive: true });
    writeFileSync(join(process.env.SFM_E2E_DUMP, 'selection.json'), JSON.stringify({ ...dump.selection, names: dump.names }));
    mkdirSync(join(process.env.SFM_E2E_DUMP, 'sparse'), { recursive: true });
    for (const [k, bytes] of Object.entries(dump.sparse)) writeFileSync(join(process.env.SFM_E2E_DUMP, 'sparse', k), Uint8Array.from(bytes));
  }
  console.log(`SfM: ${JSON.stringify(summary)} (${((Date.now() - t0) / 1000).toFixed(0)} s wall)`);
  if (summary.registered < 0.95 * summary.keyframes) fail(`only ${summary.registered}/${summary.keyframes} keyframes registered`);

  // hand-over (autoSend) -> the trainer has the cameras; train to the first validation
  await page.waitForFunction((n) => window.app.colmap?.images?.length === n && !!window.app.trainer, summary.exported, { timeout: 120000 });
  await page.evaluate(() => window.app._setTraining(true));
  const val = await page.waitForFunction(() => {
    const m = document.getElementById('log-body').innerText.match(/Val PSNR @ step (\d+): ([\d.]+) dB/);
    return m ? { step: Number(m[1]), psnr: Number(m[2]) } : null;
  }, null, { timeout: 15 * 60000, polling: 2000 }).then(h => h.jsonValue());
  console.log(`training: validation PSNR ${val.psnr.toFixed(2)} dB at step ${val.step}`);
  if (!(val.psnr >= MIN_PSNR)) fail(`validation PSNR ${val.psnr} dB < ${MIN_PSNR} dB`);
  // repairs (DESIGN-sfm §6.3): retry a range and undo it (same scores again), the same for repair
  // all, then densify a range
  // and send the larger model to the trainer
  const sfm = (fn, arg) => page.evaluate(fn, arg);
  const settled = (re) => page.waitForFunction((re) => !window.app.sfm.busy && new RegExp(re).test(window.app.sfm.state.status),
                                               re, { timeout: 10 * 60000, polling: 500 });
  const counts = () => sfm(() => JSON.stringify(window.app.sfm.quality.counts));
  // validation PSNRs change the scores: stop training, then wait until no validation is pending and
  // the worker has answered every PSNR sent to it (it handles messages in order)
  await sfm(() => {
    const s = window.app.sfm, n = window.__psnr = { sent: 0, got: 0 }, viewPsnr = s.viewPsnr.bind(s);
    s.viewPsnr = (v) => { if (s.worker && s.quality && v?.length) n.sent++; viewPsnr(v); };
    s.worker.addEventListener('message', ({ data }) => { if (data.type === 'quality' && !data.action) n.got++; });
    window.app._setTraining(false);
  });
  await page.waitForFunction(() => {
    const a = window.app, n = window.__psnr;
    const last = [...document.getElementById('log-body').innerText.matchAll(/Val PSNR @ step (\d+)/g)].at(-1);
    return !a.valRequest && Number(last?.[1]) >= a.lastValStep && n.got === n.sent;
  }, null, { timeout: 120000, polling: 200 });
  const initial = await counts(), kf = await sfm(() => window.app.sfm.quality.frames.length);
  await sfm(() => { const s = window.app.sfm; s._select(s.quality.frames.slice(8, 14).map(f => f.name)); s.act('retry', [s.selected]); });
  await settled('^retry done');
  const afterRetry = await counts();
  await sfm(() => window.app.sfm.undo());
  await settled('^undo done');
  const afterUndo = await counts();
  console.log(`repairs: scores ${initial} -> retry ${afterRetry} -> undo ${afterUndo}`);
  if (afterUndo !== initial) fail(`undo did not restore the scores: ${afterUndo} vs ${initial}`);
  // repair all: only kept retries, so never more red; undone as one action
  await sfm(() => window.app.sfm.act('repair', []));
  await settled('^repair done');
  const afterRepair = await counts();
  await sfm(() => window.app.sfm.undo());
  await settled('^undo done');
  console.log(`repair all: scores ${initial} -> ${afterRepair} -> undo ${await counts()}`);
  if (JSON.parse(afterRepair).red > JSON.parse(initial).red) fail(`repair all added red keyframes: ${afterRepair} vs ${initial}`);
  if (await counts() !== initial) fail(`undo of repair all did not restore the scores: ${await counts()} vs ${initial}`);
  await sfm(() => { const s = window.app.sfm; s._select(s.quality.frames.slice(8, 12).map(f => f.name)); s.act('densify', [s.selected]); });
  await settled('^densify done');
  const added = (await sfm(() => window.app.sfm.quality.frames.length)) - kf;
  // the repaired model into the running trainer: splats and step count kept, training goes on
  const before = await sfm(() => ({ step: window.app.trainer.stepCount, splats: window.app.trainer.count }));
  await sfm(() => window.app.sfm.requestSend('update'));
  await page.waitForFunction((n) => window.app.trainer?.images?.length >= n && /splats kept/.test(window.app.sfm.state.model),
                             summary.exported + added, { timeout: 120000 });
  const upd = await sfm(() => ({ ...window.app.sfm.lastUpdate, step: window.app.trainer.stepCount, splats: window.app.trainer.count,
                                 views: window.app.trainer.images.length }));
  console.log(`densify: ${added} keyframes added; update: ${upd.kept} kept, ${upd.added} added, cameras moved ` +
              `${(100 * upd.moved).toFixed(3)}% of the scene (max ${(100 * upd.maxMoved).toFixed(2)}%), step ${upd.step}, ${upd.splats} splats`);
  if (added < 2) fail(`densify added ${added} keyframes`);
  if (upd.step !== before.step || upd.splats !== before.splats || upd.added !== added) fail(`update did not keep the training: ${JSON.stringify(upd)} vs ${JSON.stringify(before)}`);
  if (!(upd.moved < 0.01)) fail(`unchanged cameras moved by ${upd.moved} of the scene radius`);
  await sfm(() => window.app._setTraining(true));
  const val2 = await page.waitForFunction((after) => {
    const all = [...document.getElementById('log-body').innerText.matchAll(/Val PSNR @ step (\d+): ([\d.]+) dB/g)];
    const m = all.at(-1);
    return m && Number(m[1]) > after ? { step: Number(m[1]), psnr: Number(m[2]) } : null;
  }, before.step, { timeout: 15 * 60000, polling: 2000 }).then(h => h.jsonValue());
  await sfm(() => window.app._setTraining(false));
  console.log(`after the update: validation PSNR ${val2.psnr.toFixed(2)} dB at step ${val2.step} (was ${val.psnr.toFixed(2)} at ${val.step})`);
  if (val2.psnr < val.psnr - 1) fail(`validation PSNR fell from ${val.psnr} to ${val2.psnr} dB after the update`);

  // persistence: the same video after a reload restores the session from OPFS, without a run
  await page.reload();
  await page.waitForFunction(() => window.app.sfm?.restored, null, { timeout: 120000 });
  const restored = await sfm(() => ({ frames: window.app.sfm.quality.frames.length, counts: JSON.stringify(window.app.sfm.quality.counts) }));
  console.log(`reload: restored ${restored.frames} keyframes, scores ${restored.counts}`);
  if (restored.frames !== kf + added) fail(`restored ${restored.frames} keyframes, expected ${kf + added}`);
  await page.waitForFunction(() => window.app.colmap?.fromSfm && window.app.sfm.state.model.startsWith('the current'), null, { timeout: 120000 });

  const serious = await errors();
  if (serious.length) fail(`${serious.length} page errors:\n  ${serious.slice(0, 10).join('\n  ')}`);
  if (!process.exitCode) console.log(`PASS in ${NAME} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
} catch (e) {
  fail(e.message);
  const panel = await page.evaluate(() => `${window.app.sfm?.state.status}\n${window.app.logText('sfm')}`).catch(() => null);
  if (panel) console.error(`panel status and log:\n${panel.split('\n').slice(-15).join('\n')}`);
} finally {
  await close();
  srv.kill();
}
