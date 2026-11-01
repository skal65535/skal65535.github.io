// watchdog.e2e.js — The frame loop's watchdog (app.js _watchFrames): a frame held up over 3 s (here the drawing,
// made to wait 6 s) is logged with what it waits on, the step buttons are refreshed meanwhile (the frame loop refreshes
// them: stalled, a finished step's buttons would stay as they were), and its end is logged.
// A step button's click is logged with the step. On the synthetic room (MESH_E2E_ROOM, as mesh.e2e.js).
//   cd e2e && node watchdog.e2e.js
import { fail, appUrl, run, room, appReady } from './harness.js';

const ROOM = room();

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await appReady(page);
  const r = await page.evaluate(async () => {
    const app = window.app, draw = app._draw.bind(app), sleep = (ms) => new Promise(res => setTimeout(res, ms));
    let hold = true, refreshes = 0;
    app._draw = async (dt) => { if (hold) { hold = false; await sleep(6000); } return draw(dt); };
    const refresh = app.shell.refresh.bind(app.shell);
    app.shell.refresh = () => { refreshes++; return refresh(); };
    app.drawGen++;  // a frame that draws
    await sleep(1000);
    const r0 = refreshes;  // the stalled frame's own refresh comes after the drawing
    await sleep(4000);
    const during = refreshes - r0, log = app.logText();
    await sleep(3000);
    const button = [...document.querySelectorAll('#phase-foot button')].find(b => /Step/.test(b.textContent));
    button?.click();
    await sleep(300);
    return { during, waiting: /frame loop: waiting on 'the drawing'/.test(log), again: /frame loop: going again/.test(app.logText()),
             button: button?.textContent ?? null, clicked: app.logText().split('\n').find(l => /button: Step/.test(l)) ?? null };
  });
  if (!r.waiting) fail("a frame held 6 s in its drawing: no 'frame loop: waiting on' in the log");
  if (!(r.during >= 2)) fail(`a frame held 6 s: the step buttons refreshed ${r.during} times meanwhile (want every second)`);
  if (!r.again) fail("the held frame done: no 'frame loop: going again' in the log");
  if (!r.clicked) fail(`a click on '${r.button}': not in the log`);
  return `a frame held 6 s: logged waiting on the drawing, the buttons refreshed ${r.during} times meanwhile, its end logged; the click logged ('${r.clicked.replace(/^.*button/, 'button')}')`;
}, { serverArgs: [ROOM] });
