// stepper.e2e.js — The stepper (ui/phases.js, ui/shell.js) on a video's run to its keyframes, in a project:
//   - before: Video done, Keyframes next, Train waiting ("Needs cameras", its footer leading to Cameras first);
//   - while running: the Keyframes step running, its progress bar drawn;
//   - after: Keyframes done with "saved HH:MM", the notice saying the keyframes are in the project (cache/features.gosfm),
//     the inspector's saved line;
//   - a keyframe option changed: ✎ Options changed with what changed, Revert the options and Find keyframes again in the
//     footer, the cameras untouched; reverted: the option back, the step done again.
//   cd e2e && node stepper.e2e.js   (SFM_E2E_VIDEO, SFM_E2E_SECONDS: the clip, harness.js)
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fail, clip, urlOf, appUrl, run } from './harness.js';

const video = clip(), projects = mkdtempSync(join(tmpdir(), 'gaussopt-stepper-'));
const step = (p) => {
  const b = document.querySelector(`#go-stepper .step[data-phase="${p}"]`);
  return { cls: b.className.replace(/^step | current$/g, ''), chip: b.querySelector('.step-chip').textContent, sum: b.querySelector('.step-sum').textContent,
           saved: b.querySelector('.step-saved').textContent, bar: getComputedStyle(b.querySelector('.step-bar')).display !== 'none' };
};
const foot = () => [...document.querySelectorAll('#phase-foot button')].map(b => `${b.className === 'primary' ? '*' : ''}${b.textContent}${b.disabled ? ' (off)' : ''}`).join(' | ');

await run(async (page) => {
  await page.goto(appUrl(`?video=${encodeURIComponent(urlOf(video))}&sfm_autoSend=0`));
  await page.waitForFunction(() => window.app?.sfm?.state.status === 'ready', null, { timeout: 120000, polling: 500 });
  await page.evaluate(async () => {
    const { ServerStore } = await import('./src/project/store.js');
    await window.app.projectPanel.attach(new ServerStore('stepper'));
  });
  const before = await page.evaluate(({ step, foot }) => {
    const app = window.app, s = eval(`(${step})`), f = eval(`(${foot})`);
    app.shell.show('train');
    const train = { ...s('train'), foot: f(), note: document.getElementById('phase-note').textContent };
    app.shell.show('keyframes');
    return { video: s('video'), keyframes: s('keyframes'), train };
  }, { step: step.toString(), foot: foot.toString() });
  console.log(JSON.stringify(before));
  if (before.video.cls !== 'done' || before.keyframes.cls !== 'next' || before.keyframes.chip !== 'Next') fail(`before the run: ${JSON.stringify(before)}`);
  if (before.train.cls !== 'locked' || before.train.chip !== 'Needs cameras' || !before.train.foot.startsWith('*Go to Cameras')) fail(`Train before the cameras: ${JSON.stringify(before.train)}`);

  await page.evaluate((step) => {
    const s = eval(`(${step})`);
    window.seen = { running: 0, bar: 0 };
    window.poll = setInterval(() => { const k = s('keyframes'); if (k.cls.startsWith('running')) window.seen.running++; if (k.bar) window.seen.bar++; }, 50);
    window.app.sfm.run('keyframes');
  }, step.toString());
  await page.waitForFunction(() => window.app.sfm.running, null, { timeout: 60000 });
  await page.waitForFunction(() => !window.app.sfm.running, null, { timeout: 15 * 60000, polling: 500 });
  await page.waitForFunction(() => !!window.app.saved.keyframes, null, { timeout: 120000, polling: 500 });
  const after = await page.evaluate(({ step, foot }) => {
    clearInterval(window.poll);
    const app = window.app, s = eval(`(${step})`), f = eval(`(${foot})`), o = app.sfm.o;
    app.shell.show('keyframes');
    app.shell.refresh();
    const done = { ...s('keyframes'), toast: document.getElementById('phase-toast').textContent, savedLine: document.getElementById('phase-saved').textContent };
    const spacing = o.spacing;
    o.spacing = +(spacing * 0.8).toFixed(4);
    app._optionChanged(o, 'spacing', o.spacing);  // as its control does
    app.shell.refresh();
    const changed = { ...s('keyframes'), foot: f(), note: document.getElementById('phase-note').textContent, cameras: s('cameras') };
    [...document.querySelectorAll('#phase-foot button')].find(b => b.textContent === 'Revert the options').click();
    const reverted = { ...s('keyframes'), spacing: o.spacing, was: spacing };
    return { seen: window.seen, done, changed, reverted, errors: document.getElementById('error-line').textContent };
  }, { step: step.toString(), foot: foot.toString() });
  console.log(JSON.stringify(after));
  const { done, changed, reverted } = after;
  if (!(after.seen.running > 0 && after.seen.bar > 0)) fail(`while running: ${JSON.stringify(after.seen)}`);
  if (done.cls !== 'done' || !/^saved \d\d:\d\d$/.test(done.saved)) fail(`Keyframes after the run: ${JSON.stringify(done)}`);
  if (!done.toast.startsWith('Keyframes saved in the project (cache/features.gosfm)') || !/^Saved \d\d:\d\d in the project \(cache\/features\.gosfm\)/.test(done.savedLine))
    fail(`the notice: '${done.toast}', the line: '${done.savedLine}'`);
  if (changed.cls !== 'changed' || changed.chip !== 'Options changed' || !changed.sum.startsWith(`spacing ${reverted.was} → `) ||
      changed.foot !== 'Revert the options | *Find keyframes again' || !/An option changed/.test(changed.note)) fail(`a keyframe option changed: ${JSON.stringify(changed)}`);
  if (changed.cameras.cls === 'stale') fail(`the cameras out of date before a run: ${JSON.stringify(changed.cameras)}`);
  if (reverted.spacing !== reverted.was || reverted.cls !== 'done') fail(`reverted: ${JSON.stringify(reverted)}`);
  if (after.errors) fail(`page error: ${after.errors}`);
  return `Keyframes: next, running (bar drawn ${after.seen.bar} times), done and ${done.saved} with its notice; spacing changed: '${changed.sum}', reverted`;
}, { serverArgs: ['--projects', projects] });
