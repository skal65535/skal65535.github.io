// static.e2e.js — The app served statically (python's http.server, as GitHub Pages or the VPS serve it: no serve.py):
// the default dataset is under serve.py's /datasets mounts, so the page shows the synthetic scene without asking for it
// (no request under /datasets/: each would be a 404 in the console). The probes for serve.py (datasets.json,
// projects.json) still answer 404: they are how the page finds it is not there.
//   cd e2e && node static.e2e.js
import { dirname } from 'node:path';
import { APP, appUrl, fail, run } from './harness.js';

await run(async (page) => {
  const asked = [];
  page.on('request', r => { if (new URL(r.url()).pathname.startsWith('/datasets/')) asked.push(r.url()); });
  await page.goto(appUrl());
  await page.waitForFunction(() => window.app?.trainer?.count > 0, null, { timeout: 60000 });
  const log = await page.evaluate(() => window.app.logText());
  if (asked.length) fail(`served statically: ${asked.length} requests under /datasets/ (${asked.join(', ')})`);
  if (!/No dataset server: showing a synthetic scene/.test(log)) fail('served statically: no note of the synthetic scene in the log');
  return 'served statically, the synthetic scene shown, nothing asked under /datasets/';
}, { static: dirname(APP), ignored: e => /datasets\.json|projects\.json/.test(e) });
