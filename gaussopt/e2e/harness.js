// harness.js — What the e2e tests share: the video clip, the synthetic room, a serve.py of this checkout or a static
// server, the browser (headless Chrome through Playwright, or Safari through safaridriver with BROWSER=safari) and
// its page errors.
//   env: SFM_E2E_VIDEO, SFM_E2E_SECONDS, MESH_E2E_ROOM, CHROME, PORT, BROWSER, VERBOSE
import { chromium } from 'playwright-core';
import { launchSafari } from './safari.js';
import { spawn, execFileSync } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import { basename, dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

export const HERE = dirname(fileURLToPath(import.meta.url));
export const APP = dirname(HERE);
export const SOURCE = process.env.SFM_E2E_VIDEO ?? `${process.env.HOME}/gskal/brush/jardin5/jardin5.mp4`;
export const SECONDS = Number(process.env.SFM_E2E_SECONDS ?? 12);
export const PORT = Number(process.env.PORT ?? 8765);
export const SAFARI = process.env.BROWSER === 'safari';
export const NAME = SAFARI ? 'Safari' : 'headless Chrome';
const CHROME = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

export const fail = (msg) => { console.error(`FAIL: ${msg}`); process.exitCode = 1; };

// The first SECONDS of SOURCE, cut with ffmpeg into .cache/ once
export function clip() {
  const dir = join(HERE, '.cache'), out = join(dir, `${basename(SOURCE).replace(/\.[^.]*$/, '')}_${SECONDS}s.mp4`);
  if (existsSync(out)) return out;
  if (!existsSync(SOURCE)) throw new Error(`no source video ${SOURCE} (set SFM_E2E_VIDEO)`);
  mkdirSync(dir, { recursive: true });
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', SOURCE, '-t', String(SECONDS), '-c', 'copy', '-an', out]);
  return out;
}

// The synthetic room's dataset (MESH_E2E_ROOM), checked
export function room() {
  const dir = process.env.MESH_E2E_ROOM ?? `${process.env.HOME}/gskal/brush/synth/room10`;
  if (!existsSync(join(dir, 'scene_sdf.json'))) throw new Error(`no synthetic room in ${dir} (set MESH_E2E_ROOM)`);
  return dir;
}

// The URL path of a file under the app's parent (what serve.py serves)
export const urlOf = (p) => '/' + relative(dirname(APP), p);

// This checkout's serve.py on PORT, ready once it says it listens (it binds before printing): a
// stale server on PORT makes it exit instead
export async function server(args = []) {
  const proc = spawn('python3', [join(APP, 'serve.py'), '--port', String(PORT), ...args], { stdio: ['ignore', 'pipe', 'pipe'] });
  let out = '', log = '';  // both drained (a full pipe would block serve.py), the stderr tail kept for errors
  proc.stdout.on('data', d => { out = (out + d).slice(-2000); });
  proc.stderr.on('data', d => { log = (log + d).slice(-2000); });
  const deadline = Date.now() + 10000;
  while (Date.now() < deadline) {
    if (proc.exitCode !== null) throw new Error(`serve.py exited (${proc.exitCode}):\n${log.trim()}`);
    if (out.includes('listening on')) return proc;
    await new Promise(r => setTimeout(r, 200));
  }
  proc.kill();
  throw new Error(`serve.py did not start on port ${PORT}`);
}

// python's http.server of `cwd` on `port` (as GitHub Pages or the VPS serve the files), ready once it answers
export async function staticServer(cwd, port = PORT) {
  const proc = spawn('python3', ['-m', 'http.server', String(port), '--bind', '127.0.0.1'], { cwd, stdio: 'ignore' });
  const deadline = Date.now() + 10000;
  while (Date.now() < deadline) {
    if (proc.exitCode !== null) throw new Error(`http.server exited (${proc.exitCode}) on port ${port}`);
    if (await fetch(`http://127.0.0.1:${port}/`).then(r => r.ok, () => false)) return proc;
    await new Promise(r => setTimeout(r, 200));
  }
  proc.kill();
  throw new Error(`http.server did not start on port ${port}`);
}

// The app's page URL, `query` ('?...') appended
export const appUrl = (query = '') => `http://localhost:${PORT}${urlOf(join(APP, 'index.html'))}${query}`;

// The app ready: its trainer made and the photos readable
export const appReady = (page, timeout = 120000) =>
  page.waitForFunction(() => !!window.app?.trainer && window.app.gtAvailable, null, { timeout });

// The gallery's page (gallery/index.html) showing its scene: the scene's info line filled
export const galleryShown = (page, timeout = 120000) =>
  page.waitForFunction(() => document.getElementById('scene-info')?.textContent.length > 0, null, { timeout });

// The middle of the canvas as shown, `settle` ms on (a PNG; the corners hold the status labels, which change by
// themselves); `changed` compares two of them byte for byte
export async function pixels(page, settle = 3000) {
  await page.waitForTimeout(settle);
  const r = await page.locator('#main-canvas').boundingBox();
  return page.screenshot({ clip: { x: r.x + r.width / 4, y: r.y + r.height / 4, width: r.width / 2, height: r.height / 2 } });
}
export const changed = (a, b) => a.length !== b.length || !a.equals(b);

// One test: serve.py (serverArgs) on PORT and a browser around body(page), then a FAIL for page errors or PASS, with
// body's return value as a summary. With `static` (a folder), python's http.server serves it on staticPort, in place of
// serve.py when that is PORT. After a failure, onFail(page) returns what to log (or null); cleanup() runs last.
// pageOptions: the page's (browser()); ignored(e): the page errors not counted.
export async function run(body, { serverArgs = [], static: dir = null, staticPort = PORT, ignored = () => false,
                                  onFail = null, cleanup = null, pageOptions = {} } = {}) {
  const srvs = [];
  try {
    if (dir === null || staticPort !== PORT) srvs.push(await server(serverArgs));
    if (dir !== null) srvs.push(await staticServer(dir, staticPort));
  } catch (e) {
    for (const p of srvs) p.kill();
    throw e;
  }
  const { page, close, errors } = await browser(pageOptions);
  const t0 = Date.now();
  try {
    const summary = await body(page);
    const serious = (await errors()).filter(e => !ignored(e));
    if (serious.length) fail(`${serious.length} page errors:\n  ${serious.slice(0, 10).join('\n  ')}`);
    if (!process.exitCode) console.log(`PASS in ${NAME} (${((Date.now() - t0) / 1000).toFixed(0)} s)${summary ? `: ${summary}` : ''}`);
  } catch (e) {
    fail(e.message);
    const log = await onFail?.(page).catch(() => null);
    if (log) console.error(log);
  } finally {
    await close();
    for (const p of srvs) p.kill();
    cleanup?.();
  }
}

// { page, close, errors(): the page errors so far, favicon misses aside }; pageOptions: Playwright's newPage options
// over the default viewport (e.g. a phone: viewport, isMobile, hasTouch; Chrome only)
export async function browser(pageOptions = {}) {
  if (SAFARI) {  // the errors are kept in the page: banked before each navigation
    const { page, close } = await launchSafari({ width: 1400, height: 1000 });
    const seen = [];
    for (const k of ['goto', 'reload']) {
      const go = page[k].bind(page);
      page[k] = async (...a) => { seen.push(...await page.errors()); return go(...a); };
    }
    return { page, close, errors: async () => [...seen, ...await page.errors()].filter(e => !/favicon/.test(e)) };
  }
  const b = await chromium.launch({ executablePath: CHROME, headless: true, args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU'] });
  const page = await b.newPage({ viewport: { width: 1400, height: 1000 }, acceptDownloads: true, ...pageOptions });
  const seen = [];
  page.on('pageerror', e => seen.push(`page: ${e.message}`));
  page.on('console', m => {
    if (m.type() === 'error') seen.push(`console: ${m.text()} ${m.location()?.url ?? ''}`);  // a failed load names its URL only there
    if (process.env.VERBOSE) console.log(`  [${m.type()}] ${m.text()}`);
  });
  return { page, close: () => b.close(), errors: async () => seen.filter(e => !/favicon/.test(e)) };
}
