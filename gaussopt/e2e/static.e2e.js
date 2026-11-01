// static.e2e.js — The app served statically (python's http.server, as GitHub Pages or the VPS serve it: no serve.py):
// the default dataset is under serve.py's /datasets mounts, so the page shows the synthetic scene without asking for it
// (no request under /datasets/: they were 4 errors in the console, 404s). The probes for serve.py (datasets.json,
// projects.json) still answer 404: they are how the page finds it is not there.
//   cd e2e && node static.e2e.js
import { spawn } from 'node:child_process';
import { dirname } from 'node:path';
import { APP, NAME, PORT, appUrl, browser, fail } from './harness.js';

const srv = spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1'], { cwd: dirname(APP), stdio: 'ignore' });
for (let i = 0; i < 50 && !await fetch(`http://localhost:${PORT}/`).then(r => r.ok, () => false); i++) await new Promise(r => setTimeout(r, 200));
const { page, close, errors } = await browser();
const t0 = Date.now();
try {
  const asked = [];
  page.on('request', r => { if (new URL(r.url()).pathname.startsWith('/datasets/')) asked.push(r.url()); });
  await page.goto(appUrl());
  await page.waitForFunction(() => window.app?.trainer?.count > 0, null, { timeout: 60000 });
  const log = await page.evaluate(() => window.app.logText());
  if (asked.length) fail(`served statically: ${asked.length} requests under /datasets/ (${asked.join(', ')})`);
  if (!/No dataset server: showing a synthetic scene/.test(log)) fail('served statically: no note of the synthetic scene in the log');
  const errs = (await errors()).filter(e => !/datasets\.json|projects\.json/.test(e));
  if (errs.length) fail(`page errors besides serve.py's probes:\n  ${errs.join('\n  ')}`);
  if (!process.exitCode) console.log(`PASS in ${NAME} (${((Date.now() - t0) / 1000).toFixed(0)} s): served statically, the synthetic scene shown, nothing asked under /datasets/`);
} catch (e) {
  fail(e.message);
} finally {
  await close();
  srv.kill();
}
