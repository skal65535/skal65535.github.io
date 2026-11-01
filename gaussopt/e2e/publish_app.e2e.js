// publish_app.e2e.js — The app (the editor) in the published gallery (gallery/publish.sh): the gallery published to a
// fresh folder with one scene (GALLERY_SCENE, default SCENE below, from gallery/scenes), served statically as the VPS
// serves it (no serve.py): gaussopt/index.html starts (window.app, its WebGPU device, the Project step shown), every
// file it asks for is there (no 404), no serve.py found (no snapshots to it); a video (harness.js clip: SFM_E2E_VIDEO's
// first seconds, served next to it) mapped in the page (its workers) and 20 steps trained; the gallery page next to
// it still shows its scene.
//   cd e2e && node publish_app.e2e.js
import { execFileSync, spawn } from 'node:child_process';
import { mkdtempSync, mkdirSync, copyFileSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { APP, NAME, PORT, browser, clip, fail } from './harness.js';

const SCENE = process.env.GALLERY_SCENE ?? 'canyon3-bg';
const WEB = dirname(APP), scenes = join(WEB, 'gallery', 'scenes');
const tmp = mkdtempSync(join(tmpdir(), 'publish_app-')), src = join(tmp, 'scenes'), out = join(tmp, 'site');
mkdirSync(join(src, SCENE), { recursive: true });
mkdirSync(out);
copyFileSync(join(scenes, SCENE, 'scene.glb'), join(src, SCENE, 'scene.glb'));
writeFileSync(join(src, 'index.json'), JSON.stringify([{ id: SCENE, title: SCENE, about: 'publish_app.e2e.js' }]));
execFileSync('zsh', [join(WEB, 'gallery', 'publish.sh'), src, out], { stdio: 'ignore' });
copyFileSync(clip(), join(out, 'clip.mp4'));
if (!existsSync(join(out, 'gaussopt', 'index.html'))) { fail('publish.sh: no gaussopt/index.html'); process.exit(); }

const srv = spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1'], { cwd: out, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1500));
const { page, close, errors } = await browser();
// serve.py's lists, asked for to find it (app.js _probeServer, the project list): not there on a static server
const probe = (url) => /\/(datasets|projects)\.json$|\/favicon\.ico$/.test(url);
const t0 = Date.now(), missing = [];
page.on('response', (r) => { if (r.status() === 404 && !probe(r.url())) missing.push(r.url()); });
try {
  const summary = await (async () => {
    await page.goto(`http://localhost:${PORT}/gaussopt/index.html`);
    await page.waitForFunction(() => window.app?.gpu?.device && window.app.serverOk !== null, null, { timeout: 60000 });
    const a = await page.evaluate(() => ({ server: window.app.serverOk, phase: window.app.shell?.current, title: document.title }));
    if (a.server) fail(`a serve.py found on a static server: ${JSON.stringify(a)}`);
    if (a.phase !== 'project') fail(`the app's first step: ${a.phase}`);
    // a video: mapped in the page, then trained
    await page.goto(`http://localhost:${PORT}/gaussopt/index.html?video=/clip.mp4&stop_after=20`);
    await page.waitForFunction(() => window.app?.sfm?.state.status === 'ready', null, { timeout: 60000 });
    await page.evaluate(() => window.app.sfm.run());
    const status = await page.waitForFunction(() => { const s = window.app.sfm.state.status; return /^(done|error)/.test(s) ? s : null; },
                                              null, { timeout: 15 * 60000, polling: 1000 }).then(h => h.jsonValue());
    if (!status.startsWith('done')) return fail(`the video's mapping: ${status}`);
    const sfm = await page.evaluate(() => window.app.sfm.result.summary);
    await page.waitForFunction((n) => window.app.colmap?.images?.length === n && !!window.app.trainer, sfm.exported, { timeout: 120000 });
    await page.evaluate(() => window.app._setTraining(true));
    await page.waitForFunction(() => window.app.trainer.stepCount >= 20 && !window.app.params.training, null, { timeout: 300000 });
    const steps = await page.evaluate(() => window.app.trainer.stepCount);
    await page.goto(`http://localhost:${PORT}/index.html?scene=${SCENE}&mode=orbit`);
    await page.waitForFunction(() => document.getElementById('scene-info')?.textContent.length > 0, null, { timeout: 120000 });
    if (missing.length) fail(`${missing.length} files not published:\n  ${missing.slice(0, 10).join('\n  ')}`);
    return `published to ${out}: the app (${a.title}) starts on its Project step, no serve.py, every file there; the clip ` +
           `mapped (${sfm.registered} of ${sfm.keyframes} keyframes) and ${steps} steps trained; the gallery shows ${SCENE}`;
  })();
  const serious = (await errors()).filter(e => !(e.includes('404') && probe(e.split(' ').at(-1))));
  if (serious.length) fail(`${serious.length} page errors:\n  ${serious.slice(0, 10).join('\n  ')}`);
  if (!process.exitCode) console.log(`PASS in ${NAME} (${((Date.now() - t0) / 1000).toFixed(0)} s): ${summary}`);
} catch (e) {
  fail(e.message);
} finally {
  await close();
  srv.kill();
}
