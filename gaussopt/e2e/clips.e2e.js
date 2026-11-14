// clips.e2e.js — Two videos of one place in one project (DESIGN-sfm §14): two cuts of the e2e video (SOURCE's first
// 2 x CUT s) opened together (?video=a,b), a serve.py project attached before the run (as train.mjs's), mapped and sent
// to the trainer. Then:
//   - the second video's keyframes after the first's by its offset, both in the map;
//   - the project has the features container (written in pieces) with both videos and each keyframe's video, the
//     model, each video's sharpness file;
//   - the page reloaded with the same two videos and the project attached again: the session restored, as many
//     keyframes, none of the videos' own state lost (no "not the open videos").
//   cd e2e && node clips.e2e.js
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { appUrl, fail, HERE, run, SOURCE, urlOf } from './harness.js';

const CUT = Number(process.env.CLIPS_SECONDS ?? 6), PROJECT = 'two-videos';
const cut = (k) => {
  const dir = join(HERE, '.cache'), out = join(dir, `clips_${k}_${CUT}s.mp4`);
  if (existsSync(out)) return out;
  mkdirSync(dir, { recursive: true });
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-ss', String(k * CUT), '-i', SOURCE, '-t', String(CUT), '-c', 'copy', '-an', out]);
  return out;
};
const videos = [cut(0), cut(1)], projects = mkdtempSync(join(tmpdir(), 'gaussopt-clips-'));
// a .gosfm's header JSON (project.js: magic, u32 length, JSON)
const header = (file) => { const b = readFileSync(file); return JSON.parse(b.subarray(12, 12 + b.readUInt32LE(8)).toString()); };

await run(async (page) => {
  const wait = (fn, arg, timeout = 900000) => page.waitForFunction(fn, arg, { timeout, polling: 500 });
  const url = appUrl(`?video=${videos.map(v => encodeURIComponent(urlOf(v))).join(',')}&val=0`);
  const attach = () => page.evaluate(async (name) => {
    const { ServerStore } = await import('./src/project/store.js');
    await window.app.projectPanel.attach(new ServerStore(name));
  }, PROJECT);
  await page.goto(url);
  await wait(() => window.app?.sfm?.state.status === 'ready');
  const info = await page.evaluate(() => window.app.sfm.videoInfo.videos);
  // the second's first frame 10 s after the first's last (worker.js CLIP_GAP), whatever its own times (a cut by ffmpeg -c
  // copy starts before 0: its frames from the keyframe before the cut)
  if (info?.length !== 2 || Math.abs(info[1].start - info[0].end - 10) > 1e-6) fail(`the videos: ${JSON.stringify(info)}`);
  const gap = (info[0].end + info[1].start) / 2;
  await attach();
  await page.evaluate(() => window.app.sfm.run(null, true));
  await wait(() => /^(done|error|nothing)/.test(window.app.sfm.state.status));
  const status = await page.evaluate(() => window.app.sfm.state.status);
  if (!status.startsWith('done')) return fail(`SfM: ${status}`);
  await wait(() => window.app.colmap?.fromSfm && window.app.sfm.quality);
  const frames = await page.evaluate(() => window.app.sfm.quality.frames.map(f => ({ time: f.time, registered: f.registered })));
  const second = frames.filter(f => f.time > gap), registered = frames.filter(f => f.registered).length;
  if (!second.length || second.length === frames.length) fail(`keyframes: ${second.length} of ${frames.length} after ${gap} s`);
  if (!second.some(f => f.registered) || !frames.filter(f => f.time < gap).some(f => f.registered)) fail('a video with no keyframe in the map');

  // the project: both videos' files, the containers
  const dir = join(projects, PROJECT), cache = join(dir, 'cache');
  await wait(() => !window.app.sfm.busy);
  for (const f of ['sharpness.csv', 'sharpness-1.csv', 'cache/features.gosfm', 'cache/model.gosfm']) if (!existsSync(join(dir, f))) fail(`no ${f} in the project (${readdirSync(dir)})`);
  if (existsSync(join(cache, 'features.gosfm'))) {
    const h = header(join(cache, 'features.gosfm')).json;
    const own = h.features.frames.filter(f => f.clip === 1).length;
    if (h.videos?.length !== 2 || h.analyses?.length !== 2 || own !== second.length) fail(`features.gosfm: ${h.videos?.length} videos, ${h.analyses?.length} analyses, ${own} keyframes of the second (the map ${second.length})`);
    if ((statSync(join(cache, 'features.gosfm')).mode & 0o777) !== 0o600) fail('features.gosfm: not 0600');
  }

  // reopened with the same videos: the session back, each video's own
  await page.goto(url);
  await wait(() => window.app?.sfm?.state.status === 'ready' && !!window.app.projectPanel);
  await attach();
  await wait(() => window.app.sfm.quality?.frames.length > 0);
  const again = await page.evaluate(() => ({ frames: window.app.sfm.quality.frames.length, registered: window.app.sfm.quality.frames.filter(f => f.registered).length,
                                             log: window.app.logText('sfm') }));
  if (again.frames !== frames.length || again.registered !== registered) fail(`reopened: ${again.frames} keyframes (${again.registered} registered), ${frames.length} (${registered}) before`);
  if (/not the open videos|features are not saved/.test(again.log)) fail(`reopened: ${again.log.split('\n').filter(l => /not the open videos|not saved/.test(l)).join(' / ')}`);
  return `${frames.length} keyframes (${second.length} of the second video), ${registered} registered; reopened with the same`;
}, {
  serverArgs: ['--projects', projects],
  onFail: async (page) => `log:\n${(await page.evaluate(() => window.app.logText('sfm'))).split('\n').slice(-25).join('\n')}`,
  cleanup: () => rmSync(projects, { recursive: true, force: true }),
});
