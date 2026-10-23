// safari.js — The few Playwright page calls the e2e tests use, on Safari through safaridriver
// (W3C WebDriver). Safari has no headless mode: the test runs in a separate automation window.
// One-time setup: Safari > Settings > Developer > "Allow remote automation" (or
// `safaridriver --enable`).
// safaridriver does not forward console messages, so errors are caught in the page after each
// load (console.error, uncaught errors and rejections) and read back by errors(); errors
// before that hook is in place are missed.
import { spawn } from 'node:child_process';

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// Installed after each navigation: collects errors into window.__e2eErrors
const HOOK = `if (!window.__e2eErrors) {
  window.__e2eErrors = [];
  const push = (s) => window.__e2eErrors.push(s);
  const ce = console.error.bind(console);
  console.error = (...a) => { push('console: ' + a.map(String).join(' ')); ce(...a); };
  addEventListener('error', e => push('page: ' + (e.message || e.target?.src || e.type)), true);
  addEventListener('unhandledrejection', e => push('page: ' + (e.reason?.message ?? e.reason)));
}`;

export async function launchSafari({ port = 4447, width = 1400, height = 1000 } = {}) {
  const proc = spawn('safaridriver', ['-p', String(port)], { stdio: ['ignore', 'ignore', 'pipe'] });
  let log = '';
  proc.stderr.on('data', d => { log = (log + d).slice(-2000); });
  const base = `http://127.0.0.1:${port}`;
  const call = async (method, path, body) => {
    const r = await fetch(base + path, { method, headers: { 'Content-Type': 'application/json' }, body: body && JSON.stringify(body) });
    const j = await r.json();
    if (j.value?.error) throw new Error(`safaridriver ${path}: ${j.value.error}: ${j.value.message}`);
    return j.value;
  };
  let session = null;
  for (let k = 0; k < 50 && !session; k++) {  // wait for the driver to listen
    try { session = await call('POST', '/session', { capabilities: { alwaysMatch: { browserName: 'safari' } } }); }
    catch (e) { if (/remote automation|session not created/.test(e.message)) { proc.kill(); throw e; } await sleep(200); }
  }
  if (!session) { proc.kill(); throw new Error(`safaridriver did not start on port ${port}: ${log}`); }
  const sid = `/session/${session.sessionId}`;
  await call('POST', `${sid}/window/rect`, { width, height });
  await call('POST', `${sid}/timeouts`, { script: 300000, pageLoad: 120000 });

  // fn(arg) in the page; a returned promise is awaited, its value comes back as JSON
  const evaluate = async (fn, arg) => {
    const src = fn.toString();
    const r = await call('POST', `${sid}/execute/async`, {
      script: `const done = arguments[arguments.length - 1];
               Promise.resolve().then(() => (${src})(arguments[0])).then(v => done({ v: v === undefined ? null : v }), e => done({ e: String(e?.message ?? e) }));`,
      args: [arg === undefined ? null : arg],
    });
    if (r?.e) throw new Error(r.e);
    return r?.v;
  };
  const hook = () => call('POST', `${sid}/execute/sync`, { script: HOOK, args: [] });
  const page = {
    async goto(url) { await call('POST', `${sid}/url`, { url }); await hook(); },
    async reload() { await call('POST', `${sid}/refresh`, {}); await hook(); },
    evaluate,
    async waitForFunction(fn, arg, { timeout = 30000, polling = 100 } = {}) {
      const deadline = Date.now() + timeout;
      let last = null;
      for (;;) {  // as Playwright: a throwing predicate (e.g. the page still loading) is retried
        const v = await evaluate(fn, arg).catch(e => { last = e; return null; });
        if (v) return { jsonValue: async () => v };
        if (Date.now() > deadline) throw new Error(`waitForFunction timed out after ${timeout} ms${last ? ` (${last.message})` : ''}: ${fn.toString().slice(0, 120)}`);
        await sleep(typeof polling === 'number' ? polling : 100);
      }
    },
    async waitForSelector(sel, opts) {
      return page.waitForFunction((s) => !!document.querySelector(s), sel, opts);
    },
    // fn(element) for the first match of sel
    $eval: (sel, fn) => evaluate(new Function('s', `return (${fn.toString()})(document.querySelector(s));`), sel),
    errors: () => evaluate(() => window.__e2eErrors ?? []).catch(() => []),
  };
  return {
    page,
    async close() { await call('DELETE', sid).catch(() => {}); proc.kill(); },
  };
}
