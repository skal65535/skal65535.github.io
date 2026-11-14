// view_switch.e2e.js — The view's toolbar (src/ui/view_switch.js): 3D, frames and SfM one at a time (SfM over a
// shown frames sheet hid it under, still lit); 1 / 2 / 3 and Escape back to 3D from either; frames greyed and not
// chosen without keyframes, left when emptied; the photo and view buttons in 3D only, the view's settings closed
// on leaving it; the photo button labelled with the selected camera, a click keeping the photo, a hold showing it
// while held; the segment still as the others come and go and the photo's label changes.
// On the synthetic room (MESH_E2E_ROOM, as dirty.e2e.js) served by serve.py.
//   cd e2e && node view_switch.e2e.js
import { fail, appUrl, run, room, appReady } from './harness.js';

const ROOM = room();

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await appReady(page);
  const state = () => page.evaluate(() => {
    const $ = (id) => document.getElementById(id), app = window.app;
    return {
      mode: app.viewSwitch.mode,
      lit: ['BtnView3d', 'BtnFrames', 'BtnTheatre'].filter(id => $(id).classList.contains('on')).join(),
      frames: $('frames-sheet').classList.contains('shown'), sfm: $('sfm-theatre').classList.contains('shown'),
      framesOff: $('BtnFrames').disabled, photo: $('BtnTarget').hidden ? 'hidden' : $('BtnTarget').textContent,
      view: !$('BtnView').hidden, popover: $('view-popover').classList.contains('shown'), held: app.holdTarget,
      segX: Math.round(document.querySelector('#view-toolbar .seg').getBoundingClientRect().left),
    };
  });
  const click = (id) => page.click(`#${id}`);
  const key = (k) => page.keyboard.press(k);
  const want = async (what, w) => {
    const s = await state();
    const bad = Object.entries(w).filter(([k, v]) => s[k] !== v);
    if (bad.length) fail(`${what}: ${bad.map(([k, v]) => `${k} ${JSON.stringify(s[k])}, want ${JSON.stringify(v)}`).join('; ')}`);
  };

  await page.waitForFunction(() => /#\d/.test(document.getElementById('BtnTarget').textContent), null, { timeout: 10000 });
  const { segX } = await state();
  await want('loaded', { segX, mode: '3d', lit: 'BtnView3d', frames: false, sfm: false, framesOff: true, photo: '◎ photo #0', view: true });
  await page.click('#BtnFrames', { force: true });  // greyed: no click
  await key('2');
  await want('frames without keyframes', { mode: '3d', frames: false });

  await page.evaluate(() => window.app.frames.set(['a', 'b'].map((name, i) => ({ name, time: i, blob: new Blob([]) }))));
  await click('BtnView');
  await want('view settings open', { popover: true });
  await click('BtnFrames');
  await want('frames', { segX, mode: 'frames', lit: 'BtnFrames', frames: true, sfm: false, photo: 'hidden', view: false, popover: false });
  await click('BtnTheatre');
  await want('SfM after frames', { mode: 'sfm', lit: 'BtnTheatre', frames: false, sfm: true, photo: 'hidden' });
  await key('Escape');
  await want('Escape in SfM', { mode: '3d', lit: 'BtnView3d', frames: false, sfm: false, photo: '◎ photo #0', view: true });
  await key('2');
  await want('2', { mode: 'frames', frames: true });
  await key('Escape');
  await want('Escape in frames', { mode: '3d', frames: false });
  await key('3');
  await key('1');
  await want('3 then 1', { mode: '3d', sfm: false });
  await key('2');
  await page.evaluate(() => window.app.frames.set([]));
  for (const label of ['◎ photo', '◎ photo #1234']) {
    await page.evaluate((l) => { document.getElementById('BtnTarget').textContent = l; }, label);
    await want(`photo labelled '${label}'`, { segX });
  }
  await want('frames emptied', { mode: '3d', frames: false, framesOff: true });

  // the photo: a click keeps it, another one ends it; held, while held
  await page.evaluate(() => { window.app.params.cam_idx = 1; window.app.drawGen++; });
  await page.waitForFunction(() => document.getElementById('BtnTarget').textContent === '◎ photo #1', null, { timeout: 10000 });
  await click('BtnTarget');
  await want('photo clicked', { held: true });
  await click('BtnTarget');
  await want('photo clicked again', { held: false });
  const box = await page.locator('#BtnTarget').boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await new Promise(r => setTimeout(r, 500));
  await want('photo held', { held: true });
  await page.mouse.up();
  await want('photo released', { held: false });
  return 'one of 3D / frames / SfM at a time, 1 2 3 and Escape, frames off without keyframes, the photo by click and by hold';
}, { serverArgs: [ROOM] });
