// dirty.e2e.js — Unsaved training (app.js dirty): a dataset just loaded is clean; training steps make it dirty (the
// tab's title starts with ●) and leaving the page then asks (its beforeunload handler cancels the event: the browser's
// prompt); a saved .ply makes it clean again, and leaving does not ask. On the synthetic room (MESH_E2E_ROOM, as mesh.e2e.js) served by serve.py.
//   cd e2e && node dirty.e2e.js
import { fail, appUrl, run, room, appReady } from './harness.js';

const ROOM = room();

await run(async (page) => {
  // the page's beforeunload handler cancels the event when it would ask (headless Chrome closes without the prompt)
  const leaveAsks = () => page.evaluate(() => !window.dispatchEvent(new Event('beforeunload', { cancelable: true })));
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await appReady(page);
  const clean = await page.evaluate(() => ({ dirty: window.app.dirty, title: document.title }));
  if (clean.dirty || clean.title.startsWith('●')) fail(`just loaded: dirty ${clean.dirty}, title '${clean.title}'`);
  await page.evaluate(async () => {
    await (await import('./e2e/page_util.js')).train(window.app, 20);
    await new Promise(r => setTimeout(r, 500));  // a redraw: the title
  });
  const trained = await page.evaluate(() => ({ dirty: window.app.dirty, title: document.title }));
  if (!trained.dirty || !trained.title.startsWith('●')) fail(`20 steps trained: dirty ${trained.dirty}, title '${trained.title}'`);
  if (!await leaveAsks()) fail('20 steps trained, not saved: leaving the page did not ask');
  await page.evaluate(async () => { await window.app.savePlyCheckpoint(); await new Promise(r => setTimeout(r, 500)); });
  const saved = await page.evaluate(() => ({ dirty: window.app.dirty, title: document.title }));
  if (saved.dirty || saved.title.startsWith('●')) fail(`.ply saved: dirty ${saved.dirty}, title '${saved.title}'`);
  if (await leaveAsks()) fail('.ply saved: leaving the page still asked');
  return 'clean when loaded; 20 steps: dirty, ● in the title, leaving asks; .ply saved: clean, leaving does not ask';
}, { serverArgs: [ROOM] });
