// gallery_orbit.e2e.js — The gallery's orbit (gallery/index.html): a drag released off the canvas ends there; the wheel
// zooms by its delta (a trackpad's many small ones as much as a mouse notch of the same sum); zooming in goes on
// through the target; a touch drag turns by its own move (not from where the last touch ended); two fingers pinch
// without turning; a double click re-centers on the 3D point under it, the camera kept where it is; the arrows fly the
// drone after a click in the panel (its checkbox, its slider focused) and leave the slider as it was. On GALLERY_SCENE
// (default SCENE below; gallery/scenes: the scenes' folder), served statically as the gallery is; the viewer's state
// read through a hook patched into the served page.
//   cd e2e && node gallery_orbit.e2e.js
import { dirname } from 'node:path';
import { APP, fail, galleryShown, PORT, run } from './harness.js';

const SCENE = process.env.GALLERY_SCENE ?? 'canyon3';

// a world point's CSS pixel in the canvas under an OpenCV view (column-major) and the state's fx, fy (as PanBy)
const project = ({ view: v, fx, fy }, p, box) => {
  const c = [0, 1, 2].map(k => v[k] * p[0] + v[4 + k] * p[1] + v[8 + k] * p[2] + v[12 + k]);
  return [0.5 * box.width * (1 + fx * c[0] / c[2]), 0.5 * box.height * (1 + fy * c[1] / c[2])];
};

await run(async (page) => {
  await page.route('**/gallery/viewer.js*', async (route) => {
    const r = await route.fetch(); let s = await r.text();
    const a = 'function current_camera(dt) {';
    if (s.split(a).length !== 2) throw new Error('gallery/viewer.js: the hook point moved');
    s = s.replace(a, `${a}\n  window.__g ??= { params, render };`);
    await route.fulfill({ response: r, body: s, headers: { ...r.headers(), 'content-type': 'text/javascript' } });
  });
  await page.goto(`http://localhost:${PORT}/gallery/index.html?scene=${SCENE}&mode=orbit&cameras-hidden`);
  await galleryShown(page);
  await page.waitForFunction(() => window.__g?.render.cv_view);
  const box = await page.locator('#main-canvas').boundingBox();
  const cx = box.x + 0.5 * box.width, cy = box.y + 0.5 * box.height;
  const frames = (n) => page.evaluate((n) => new Promise(r => { const f = (k) => k ? requestAnimationFrame(() => f(k - 1)) : r(); f(n); }), n);
  const state = () => page.evaluate(() => {
    const { params: p, render: r } = window.__g;
    return { theta: p.theta, phi: p.phi, radius: p.radius, target: [...p.target], center: [...r.cam_center], view: [...r.cv_view], fx: r.fx, fy: r.fy, mode: p.mode };
  });
  const reset = () => page.evaluate(() => Object.assign(window.__g.params, { theta: -91, phi: 30, radius: 2.5, target: [0, 0, 0] })).then(() => frames(2));
  const wheel = async (dy, n) => { await page.mouse.move(cx, cy); for (let i = 0; i < n; i++) await page.mouse.wheel(0, dy); await frames(2); };
  const out = [];

  // a drag released below the canvas, then the mouse over it again, no button down: no more turning
  await reset();
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 40, cy, { steps: 4 });
  await page.mouse.move(cx + 40, box.y + box.height + 40, { steps: 4 });
  await page.mouse.up();
  const released = await state();
  await page.mouse.move(cx - 200, cy, { steps: 8 });
  await frames(2);
  const after = await state();
  if (after.theta !== released.theta || after.phi !== released.phi) {
    fail(`released off the canvas, the mouse over it turns the view: theta ${released.theta.toFixed(1)} -> ${after.theta.toFixed(1)}`);
  }
  // the grab: dragged right, the point between the camera and the target goes right on the screen
  await reset();
  const s0 = await state();
  const mid = s0.target.map((t, i) => 0.5 * (t + s0.center[i]));
  const side = () => page.evaluate((m) => { const v = window.__g.render.cv_view; return v[0] * m[0] + v[4] * m[1] + v[8] * m[2] + v[12]; }, mid);
  const x0 = await side();
  await page.mouse.move(cx, cy); await page.mouse.down(); await page.mouse.move(cx + 60, cy, { steps: 6 }); await page.mouse.up();
  await frames(2);
  const turned = await state(), x1 = await side();
  out.push(`a 60 px drag: theta ${s0.theta.toFixed(1)} -> ${turned.theta.toFixed(1)}`);
  if (!(x1 > x0)) fail(`dragged right, the point before the target goes left (${x0.toFixed(3)} -> ${x1.toFixed(3)})`);

  // the wheel: 20 deltas of 5 px zoom as one of 100 px
  await reset(); await wheel(100, 1);
  const notch = (await state()).radius / 2.5;
  await reset(); await wheel(5, 20);
  const pad = (await state()).radius / 2.5;
  out.push(`zoom out: x${notch.toFixed(3)} by a 100 px notch, x${pad.toFixed(3)} by 20 of 5 px`);
  if (!(notch > 1.02) || Math.abs(Math.log(pad / notch)) > 0.1 * Math.log(notch)) fail(`the wheel: x${notch.toFixed(3)} by one 100 px delta, x${pad.toFixed(3)} by 20 of 5 px`);

  // zooming in goes on through the target
  await reset();
  const z0 = await state();
  const fwd = z0.target.map((t, i) => (t - z0.center[i]) / z0.radius);
  await wheel(-100, 120);
  const z1 = await state();
  const went = z1.center.reduce((s, c, i) => s + (c - z0.center[i]) * fwd[i], 0);
  out.push(`zoomed in: the camera ${went.toFixed(2)} forward (the target ${z0.radius.toFixed(2)} away)`);
  if (!(went > 1.2 * z0.radius)) fail(`zoomed in 120 notches, the camera went ${went.toFixed(3)} forward, the target was ${z0.radius.toFixed(3)} away`);

  // touch, through the browser's touch input: a drag ended at the left, a second one on the right turns by its own move
  const cdp = await page.context().newCDPSession(page);
  const touch = (type, pts) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: pts.map(([x, y], id) => ({ x, y, id })) });
  await reset();
  await touch('touchStart', [[cx - 300, cy]]); await touch('touchMove', [[cx - 290, cy]]); await touch('touchEnd', []);
  await frames(2);
  const t0 = await state();
  await touch('touchStart', [[cx + 300, cy]]); await touch('touchMove', [[cx + 302, cy]]); await touch('touchEnd', []);
  await frames(2);
  const t1 = await state();
  out.push(`touch: a 2 px drag turns ${Math.abs(t1.theta - t0.theta).toFixed(2)} deg`);
  if (Math.abs(t1.theta - t0.theta) > 2) fail(`a 2 px touch drag turned ${(t1.theta - t0.theta).toFixed(1)} deg (from where the last touch ended)`);
  // a pinch apart: closer, not turned
  await touch('touchStart', [[cx - 50, cy], [cx + 50, cy]]);
  for (let k = 1; k <= 5; k++) await touch('touchMove', [[cx - 50 - 20 * k, cy], [cx + 50 + 20 * k, cy]]);
  await touch('touchEnd', []);
  await frames(2);
  const t2 = await state();
  out.push(`a pinch: the radius x${(t2.radius / t1.radius).toFixed(2)}, theta ${(t2.theta - t1.theta).toFixed(2)}`);
  if (!(t2.radius < 0.7 * t1.radius)) fail(`a pinch apart from 100 to 300 px: the radius x${(t2.radius / t1.radius).toFixed(2)}`);
  if (Math.abs(t2.theta - t1.theta) > 0.5 || Math.abs(t2.phi - t1.phi) > 0.5) fail(`a pinch turned the view (theta ${(t2.theta - t1.theta).toFixed(1)}, phi ${(t2.phi - t1.phi).toFixed(1)})`);

  // a double click on a 3D point: the target on it, the camera where it was
  await reset();
  const pick = await page.evaluate(({ w, h }) => {
    const { render: r } = window.__g, v = r.cv_view, P = r.scene.points.positions;
    let best = null;
    for (let i = 0; i < P.length; i += 3) {
      const c = [0, 1, 2].map(k => v[k] * P[i] + v[4 + k] * P[i + 1] + v[8 + k] * P[i + 2] + v[12 + k]);
      if (c[2] <= 0) continue;
      const x = 0.5 * w * (1 + r.fx * c[0] / c[2]), y = 0.5 * h * (1 + r.fy * c[1] / c[2]);
      const off = Math.hypot(x - 0.3 * w, y - 0.6 * h);
      if (!best || off < best.off) best = { x, y, off, p: [P[i], P[i + 1], P[i + 2]] };
    }
    return best;
  }, { w: box.width, h: box.height });
  if (!pick) return fail(`${SCENE}: no 3D point in the view`);
  const d0 = await state();
  await page.mouse.dblclick(box.x + pick.x, box.y + pick.y);
  await frames(2);
  const d1 = await state();
  const moved = Math.hypot(...d1.center.map((c, i) => c - d0.center[i]));
  const near = Math.hypot(...d1.target.map((t, i) => t - d0.target[i]));
  // where the new target was on the screen before the click (the frontmost point in reach, not always the nearest)
  const was = project(d0, d1.target, box), now = project(d1, d1.target, box);
  out.push(`a double click: the target ${near.toFixed(2)} from the last, ${Math.hypot(was[0] - pick.x, was[1] - pick.y).toFixed(1)} px from the click, the camera moved ${moved.toFixed(4)}`);
  if (near < 1e-6) fail('a double click on a 3D point: the target is not moved');
  else {
    if (Math.hypot(was[0] - pick.x, was[1] - pick.y) > 12) fail(`a double click: the new target was ${was.map(a => a.toFixed(0))} on the screen, the click at ${pick.x.toFixed(0)},${pick.y.toFixed(0)}`);
    if (Math.hypot(now[0] - 0.5 * box.width, now[1] - 0.5 * box.height) > 2) fail(`after a double click the target is not at the view's center (${now.map(a => a.toFixed(0))})`);
  }
  if (moved > 1e-3 * d0.radius) fail(`a double click moved the camera by ${moved.toFixed(4)}`);

  // the keys after a click in the panel: a checkbox, then a slider, focused; an arrow flies the drone, the slider kept
  const row = (name) => page.locator('.skal-row').filter({ has: page.locator('.skal-row-label', { hasText: new RegExp(`^${name}$`) }) });
  await row('3D points').locator('input[type=checkbox]').click();
  await page.keyboard.press('ArrowUp');
  await frames(2);
  if ((await state()).mode !== 'drone') fail(`a click on '3D points', then an arrow: the view in ${(await state()).mode}, not the drone`);
  await page.keyboard.press('f');  // orbit again
  const speed = row('fly-by speed').locator('input[type=range]');
  await speed.focus();
  const v0 = await speed.inputValue();
  await page.keyboard.press('ArrowLeft');
  await frames(2);
  const k1 = await state(), v1 = await speed.inputValue();
  out.push(`the keys: an arrow after the panel's checkbox and slider, ${k1.mode}`);
  if (k1.mode !== 'drone' || v1 !== v0) fail(`the 'fly-by speed' slider focused, an arrow: ${k1.mode}, the slider ${v0} -> ${v1}`);
  return `${SCENE}: ${out.join('; ')}`;
}, { static: dirname(APP) });
