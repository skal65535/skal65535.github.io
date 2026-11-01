// controls.e2e.js — common/controls.js listen(): a row that is out of the page for a while (not yet attached, or
// moved) shows obj[key] again once back (its polling not stopped for good at the first frame it is out: a status line
// frozen at its last value); a row removed by clear() stops polling.
//   cd e2e && node controls.e2e.js
import { fail, appUrl, run } from './harness.js';

await run(async (page) => {
  await page.goto(appUrl());
  await page.waitForFunction(() => !!window.app?.shell, null, { timeout: 60000 });
  const r = await page.evaluate(async () => {
    const { mkGui } = await import('/common/controls.js');
    const frames = (n) => new Promise(res => { const f = () => (--n > 0 ? requestAnimationFrame(f) : res()); requestAnimationFrame(f); });
    const shown = (c) => c._row_el.querySelector('input, select')?.value ?? c._row_el.textContent;
    const host = document.createElement('div');
    document.body.append(host);
    const g = mkGui(host, { draggable: false }), obj = { status: 'first' };
    const c = g.add(obj, 'status').listen().disable();
    await frames(3);
    const a = shown(c);
    host.remove();  // out of the page a few frames (moved)
    await frames(5);
    document.body.append(host);
    obj.status = 'second';
    await new Promise(res => setTimeout(res, 1200));
    const b = shown(c);
    g.clear();
    return { a, b, removed: c._removed === true };
  });
  if (!r.a.includes('first')) fail(`listened row: shows '${r.a}', want 'first'`);
  if (!r.b.includes('second')) fail(`listened row back in the page: shows '${r.b}', want 'second' (its polling stopped while it was out)`);
  if (!r.removed) fail('clear(): the row not marked removed (its polling would go on)');
  return `a listened row out of the page and back shows the new value ('${r.b}'); clear() ends its polling`;
});
