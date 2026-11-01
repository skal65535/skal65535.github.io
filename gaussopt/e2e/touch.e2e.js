// touch.e2e.js — The app on a phone (Chrome's emulation: 390 x 844 CSS px, 2x, touch; real touch events through the
// DevTools protocol) on the synthetic room (MESH_E2E_ROOM, as mesh.e2e.js). A finger dragged 80 px up the view turns
// the camera 80 px worth of the vertical field of view down (drone.js look: the scene follows the finger) and does not
// scroll the page (common/main.css: the canvas's touch-action none; without it the page scrolls and the drag is cut
// short); two fingers spread move it forward along its view. The status is one line along the view's bottom, and the
// charts are drawn at the screen's pixels.
//   cd e2e && node touch.e2e.js
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { fail, appUrl, run } from './harness.js';

const ROOM = process.env.MESH_E2E_ROOM ?? `${process.env.HOME}/gskal/brush/synth/room10`;
if (!existsSync(join(ROOM, 'scene_sdf.json'))) throw new Error(`no synthetic room in ${ROOM} (set MESH_E2E_ROOM)`);

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await page.waitForFunction(() => !!window.app?.trainer && window.app.gtAvailable, null, { timeout: 120000 });
  await page.evaluate(() => {  // the charts' tab (a click on the open tab folds the dock)
    if (!document.querySelector('.dock-pane[data-tab="charts"]')?.classList.contains('shown')) document.querySelector('.dock-tabs button[data-tab="charts"]')?.click();
  });
  await page.waitForTimeout(1000);
  const cdp = await page.context().newCDPSession(page);
  const touch = (type, pts) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: pts.map(([x, y], id) => ({ x, y, id })) });
  const r = await page.evaluate(() => { const b = document.getElementById('main-canvas').getBoundingClientRect(); return { x: b.x, y: b.y, w: b.width, h: b.height }; });
  const cx = r.x + r.w / 2, cy = r.y + r.h / 2;
  const state = () => page.evaluate(() => {
    const d = window.app.drone, v = window.app.lastView.view;
    return { gimbal: d.gimbal, live: window.app.droneLive, pos: [...d.pos], f: [v[2], v[6], v[10]], scroll: document.scrollingElement.scrollTop, fov: window.app.params.fov };
  });
  // one finger, 80 px up in 8 moves
  await touch('touchStart', [[cx, cy + 40]]);
  await touch('touchMove', [[cx, cy + 39]]);  // the first move takes the stick (the drone starts from the pose on screen)
  const s0 = await state();
  for (let i = 1; i <= 8; i++) { await touch('touchMove', [[cx, cy + 39 - 10 * i]]); await page.waitForTimeout(30); }
  await touch('touchEnd', []);
  await page.waitForTimeout(300);
  const s1 = await state();
  const want = 80 * (s0.fov * Math.PI / 180) / r.h, got = s1.gimbal - s0.gimbal;
  if (s1.scroll !== s0.scroll) fail(`a finger dragged on the view scrolled the page: ${s0.scroll} -> ${s1.scroll} px`);
  if (!(s0.live && Math.abs(got - want) < 0.1 * want)) fail(`a finger 80 px up: the camera tilted ${(got * 180 / Math.PI).toFixed(1)} deg, want ${(want * 180 / Math.PI).toFixed(1)} (down; drone live ${s0.live})`);
  // two fingers spread from 60 to 180 px apart
  await touch('touchStart', [[cx - 30, cy], [cx + 30, cy]]);
  for (let i = 1; i <= 6; i++) { await touch('touchMove', [[cx - 30 - 10 * i, cy], [cx + 30 + 10 * i, cy]]); await page.waitForTimeout(30); }
  await touch('touchEnd', []);
  await page.waitForTimeout(300);
  const s2 = await state();
  const fwd = s2.pos.reduce((a, p, k) => a + (p - s1.pos[k]) * s1.f[k], 0);
  if (!(fwd > 0)) fail(`two fingers spread: moved ${fwd.toFixed(4)} along the view, want forward`);
  // the browser's own touch scroll gesture (through the compositor, as a finger: touch-action decides who takes it)
  const before = await state();
  await cdp.send('Input.synthesizeScrollGesture', { x: Math.round(cx), y: Math.round(cy), yDistance: -150, gestureSourceType: 'touch', speed: 800 });
  await page.waitForTimeout(300);
  const after = await state();
  if (after.scroll !== before.scroll) fail(`a touch scroll gesture on the view scrolled the page: ${before.scroll} -> ${after.scroll} px`);
  const ui = await page.evaluate(() => {
    const info = document.getElementById('info').getBoundingClientRect(), view = document.getElementById('main-canvas').getBoundingClientRect();
    const c = document.getElementById('side-canvas');
    return { infoH: info.height, infoBottom: view.bottom - info.bottom, chart: [c.width, c.clientWidth, devicePixelRatio] };
  });
  if (!(ui.infoH < 24 && ui.infoBottom >= 0 && ui.infoBottom < 20)) fail(`the status: ${ui.infoH.toFixed(0)} px tall, ${ui.infoBottom.toFixed(0)} px above the view's bottom`);
  if (!(ui.chart[1] > 0 && ui.chart[0] === Math.round(ui.chart[1] * ui.chart[2]))) fail(`the charts: ${ui.chart[0]} px drawn for ${ui.chart[1]} CSS px at ${ui.chart[2]}x`);
  return `a finger 80 px up: tilted ${(got * 180 / Math.PI).toFixed(1)} deg (want ${(want * 180 / Math.PI).toFixed(1)}), the page not scrolled; ` +
         `two fingers spread: ${fwd.toFixed(3)} forward; the status ${ui.infoH.toFixed(0)} px tall; the charts ${ui.chart[0]} px for ${ui.chart[1]} CSS px`;
}, { serverArgs: [ROOM], pageOptions: { viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true } });
