// theatre.e2e.js — The SfM theatre through the page (src/ui/theatre.js, src/sfm/trace.js):
//   - a video mapped with the trace on streams its events to the page as it runs (some arrive while keyframes are
//     still being placed); the replay at its last event places the keyframes the SfM registered, its stages in
//     order; shown, it draws (its canvas not the background alone);
//   - shown, it takes the keys (space plays, Escape closes) and none reaches the 3D view's flight; going back over
//     a renumbering counts the keyframes of then; a frame that throws does not stop the drawing;
//   - in a project, cache/trace.bin is the theatre's events, stamped with the model; reopened, it is replayed and
//     records on (a repair after it, written back); an unreadable one, or another model's, is not replayed and the
//     session restores all the same;
//   - mapped again with the trace off, the theatre holds no event (not the run before's).
//   cd e2e && node theatre.e2e.js   (SFM_E2E_VIDEO, SFM_E2E_SECONDS: the clip, harness.js; THEATRE_SHOT: a screenshot there)
import { mkdtempSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fail, clip, urlOf, appUrl, run } from './harness.js';
import { decodeTrace, encodeTrace } from '../src/sfm/trace.js';

const video = clip(), PROJECT = 'e2e-theatre';
const projects = mkdtempSync(join(tmpdir(), 'gaussopt-theatre-')), traceFile = join(projects, PROJECT, 'cache', 'trace.bin');
const readTrace = () => { const b = readFileSync(traceFile); return decodeTrace(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };

await run(async (page) => {
  const wait = (fn, arg, timeout = 15 * 60000) => page.waitForFunction(fn, arg, { timeout, polling: 500 });
  const mapRun = async () => {
    await page.evaluate(() => {
      window.seenWhileRunning = 0;  // the theatre's events while the status still said it was placing keyframes
      const t = setInterval(() => { if (/^done/.test(window.app.sfm.state.status)) clearInterval(t); else if (/^mapping: registered/.test(window.app.sfm.state.status)) window.seenWhileRunning = window.app.theatre.events.length; }, 50);
      window.app.sfm.run();
    });
    await page.waitForFunction(() => !/^done/.test(window.app.sfm.state.status), null, { timeout: 60000 });
    const status = await wait(() => { const s = window.app.sfm.state.status; return /^(done|error)/.test(s) ? s : null; }).then(h => h.jsonValue());
    if (!status.startsWith('done')) throw new Error(status);
  };
  const attach = () => page.evaluate(async (name) => {
    const { ServerStore } = await import('./src/project/store.js');
    await window.app.projectPanel.attach(new ServerStore(name));
    return window.app.project?.where ?? null;
  }, PROJECT);
  const reopen = async () => {
    await page.goto(appUrl('?dataset=[synth]'));
    await wait(() => !!(window.app?.trainer && window.app.projectPanel));
    await page.evaluate(() => window.app.setTraining(false));
    await attach();
    await wait(() => !!window.app.sfm?.quality && !window.app.sfm.busy, null, 5 * 60000);
    return page.evaluate(() => ({ events: window.app.theatre.events.length, status: window.app.sfm.state.status,
                                  log: window.app.logText('sfm').split('\n').filter(l => /trace\.bin/.test(l)) }));
  };

  await page.goto(appUrl(`?video=${encodeURIComponent(urlOf(video))}&sfm_autoSend=0`));
  await wait(() => window.app?.sfm?.state.status === 'ready');
  await mapRun();
  const r = await page.evaluate(async () => {
    const app = window.app, th = app.theatre, ev = th.events;
    th.toggle(true);
    th.seek(ev.length - 1);
    let posed = 0;
    for (const s of th.state.segments) if (s >= 0) posed++;
    await new Promise(res => setTimeout(res, 1500));  // a few frames drawn
    const g = th.canvas.getContext('2d'), { width: W, height: H } = th.canvas, px = g.getImageData(0, 0, W, H).data;
    let lit = 0;
    for (let i = 0; i < px.length; i += 4) if (px[i] + px[i + 1] + px[i + 2] > 3 * 40) lit++;
    return { events: ev.length, whileRunning: window.seenWhileRunning, stages: ev.filter(e => e.kind === 'stage').map(e => e.info.name),
             last: ev.at(-1)?.kind, posed, registered: app.sfm.result.summary.registered, lit, pixels: W * H };
  });
  console.log(JSON.stringify(r));
  if (!(r.whileRunning > 0 && r.whileRunning < r.events)) fail(`${r.whileRunning} events while mapping, ${r.events} in all: not streamed`);
  const want = ['map', 'loops', 'flowJoin', 'refine', 'retriangulate', 'lateMerge'];
  if (r.stages.slice(0, want.length).join() !== want.join()) fail(`stages ${r.stages}`);
  if (r.last !== 'end' || r.posed !== r.registered) fail(`replayed: ${r.posed} keyframes placed, the SfM ${r.registered}; last event ${r.last}`);
  if (!(r.lit > 500)) fail(`the theatre drew ${r.lit} lit pixels of ${r.pixels}`);
  if (process.env.THEATRE_SHOT) await page.screenshot({ path: process.env.THEATRE_SHOT });

  // the keys, a renumbering gone back over, a frame that throws
  const k = await page.evaluate(async () => {
    const app = window.app, th = app.theatre, key = (k) => window.dispatchEvent(new KeyboardEvent('keydown', { key: k }));
    const playing = th.playing;
    key(' ');
    const out = { droneKeys: app.drone.keys.size, toggled: th.playing !== playing };
    key(' '); key('Escape');
    out.closed = !th.shown;
    // 300 events: keyframes a, b until event 280, then a, x, b (renumbered); back to 260, past the copy at 256
    const I = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], ev = (k, images, frames, reset) => ({
      kind: 'ba', t: k, info: { free: [], observers: 0, points: 0 }, frames: Int32Array.from(frames), segments: Int32Array.from(frames, () => 0),
      poses: Float32Array.from(frames.flatMap(() => I)), ...(images ? { images: images.map(name => ({ name, time: 0 })) } : {}), ...(reset ? { reset: true } : {}) });
    const evs = Array.from({ length: 300 }, (_, k) => k === 0 ? ev(k, ['a', 'b'], [0, 1]) : k === 280 ? ev(k, ['a', 'x', 'b'], [0, 1, 2], true) : ev(k, null, []));
    th.toggle(true);
    th.load(evs);
    th.playing = false;
    th.seek(299);
    const total = () => /\/(\d+) keyframes/.exec(th.countsEl.textContent)?.[1];
    th._counts(); out.after = total();
    th.seek(260); th._counts(); out.back = total();
    // a frame that throws: the next ones still drawn
    const draw = th._draw.bind(th);
    let frames = 0, thrown = false;
    th._draw = (dt) => { frames++; if (!thrown) { thrown = true; throw new Error('a bad frame'); } draw(dt); };
    await new Promise(res => setTimeout(res, 800));
    out.framesAfterThrow = frames;
    th._draw = draw;
    th.toggle(false);
    return out;
  });
  console.log(`keys and seeking: ${JSON.stringify(k)}`);
  if (k.droneKeys !== 0 || !k.toggled || !k.closed) fail(`keys: ${JSON.stringify(k)}`);
  if (k.after !== '3' || k.back !== '2') fail(`keyframes counted ${k.after} at the end, ${k.back} back before the renumbering`);
  if (!(k.framesAfterThrow > 5)) fail(`${k.framesAfterThrow} frames after one threw`);

  // the project: the trace beside the model, replayed when reopened, recording on
  const events = r.events;
  console.log(`project: ${await attach()}`);
  await wait(() => !window.app.sfm.busy);
  if (!existsSync(traceFile)) fail(`no ${traceFile}`);
  const saved = readTrace();
  if (saved.events.length !== events || typeof saved.meta.model !== 'string') fail(`trace.bin: ${saved.events.length} events of ${events}, model ${saved.meta.model}`);
  const re = await reopen();
  console.log(`reopened: ${JSON.stringify(re)}`);
  if (re.events !== events) fail(`reopened: ${re.events} events replayed of ${events}`);
  await page.evaluate(() => { const s = window.app.sfm; s._select(s.quality.frames.slice(5, 10).map(f => f.name)); s.act('retry', [s.selected]); });
  await wait(() => !window.app.sfm.busy && /retry done|error/.test(window.app.sfm.state.status));
  const more = await page.evaluate(() => {  // the retry's stage, then its mapper's own events (window BAs, ...)
    const ev = window.app.theatre.events, at = ev.findLastIndex(e => e.kind === 'stage');
    return { events: ev.length, stage: ev[at]?.info.name, own: ev.slice(at + 1).filter(e => !['stage', 'end'].includes(e.kind)).length };
  });
  const rewritten = readTrace();
  console.log(`a retry after: ${JSON.stringify(more)}, trace.bin ${rewritten.events.length} events`);
  if (!(more.events > events) || more.stage !== 'retry' || !(more.own > 0)) fail(`the retry not recorded: ${JSON.stringify(more)}`);
  if (rewritten.events.length !== more.events || rewritten.meta.model === saved.meta.model) fail(`trace.bin after the retry: ${rewritten.events.length} events, model ${rewritten.meta.model}`);
  // unreadable, then another model's: not replayed, the session restored
  writeFileSync(traceFile, Buffer.from('not a trace at all'));
  const bad = await reopen();
  console.log(`an unreadable trace.bin: ${JSON.stringify(bad)}`);
  if (bad.events !== 0 || !/restored/.test(bad.status) || !bad.log.some(l => /unreadable/.test(l))) fail(`unreadable trace.bin: ${JSON.stringify(bad)}`);
  writeFileSync(traceFile, Buffer.from(encodeTrace(rewritten.events, { model: 'another model' })));
  const other = await reopen();
  console.log(`another model's trace.bin: ${JSON.stringify(other)}`);
  if (other.events !== 0 || !/restored/.test(other.status) || !other.log.some(l => /another model/.test(l))) fail(`another model's trace.bin: ${JSON.stringify(other)}`);

  // mapped again, the trace off: the theatre holds none of the run before's events
  await page.goto(appUrl(`?video=${encodeURIComponent(urlOf(video))}&sfm_autoSend=0`));
  await wait(() => window.app?.sfm?.state.status === 'ready');
  await mapRun();
  const before = await page.evaluate(() => window.app.theatre.events.length);
  await page.evaluate(() => { window.app.sfm.o.trace = false; });
  await mapRun();
  const off = await page.evaluate(() => window.app.theatre.events.length);
  console.log(`mapped again with the trace off: ${off} events (${before} before)`);
  if (!(before > 0) || off !== 0) fail(`trace off: ${off} events, ${before} before`);
  if (!process.exitCode) console.log('theatre.e2e: ok');
}, { serverArgs: ['--projects', projects], ignored: (e) => /a bad frame/.test(e) });  // the frame thrown on purpose
