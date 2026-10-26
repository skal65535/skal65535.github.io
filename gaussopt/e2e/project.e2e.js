// project.e2e.js — Project folders end to end (DESIGN.md §3.6): a clip through the SfM to 300
// training steps, then a new project on serve.py, filled with what the page has; the files on
// disk; a reload without the video that reopens it (session, gs/ dataset, snapshot, settings, the
// scene's transform after Level turns, also in a snapshot) and repairs;
// the same project into a folder handle (the browser's private storage gives the same
// FileSystemDirectoryHandle a picked folder does), and (Chrome) its .zip download.
//   cd e2e && node project.e2e.js   (env: see harness.js; BROWSER=safari: the same in Safari, no .zip)
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SAFARI, fail, clip, urlOf, appUrl, run } from './harness.js';

const STEPS = 300, PROJECT = 'e2e-project';
const projects = mkdtempSync(join(tmpdir(), 'gaussopt-projects-'));
const walk = (d, pre = '') => readdirSync(d).flatMap(n => statSync(join(d, n)).isDirectory() ? walk(join(d, n), `${pre}${n}/`) : [`${pre}${n}`]);
const shape = (files) => files.map(f => f.replace(/^(images|gs\/images)\/.*/, '$1/*.jpg')).filter((f, i, a) => a.indexOf(f) === i).sort();

const video = clip();
await run(async (page) => {
  const wait = (fn, arg, timeout = 600000) => page.waitForFunction(fn, arg, { timeout, polling: 500 });
  const attach = (kind, name) => page.evaluate(async ([kind, name]) => {
    const { ServerStore, DirStore } = await import('./src/project/store.js');
    const store = kind === 'server' ? new ServerStore(name)
      : new DirStore(await (await navigator.storage.getDirectory()).getDirectoryHandle(name, { create: true }));
    await window.app.projectPanel.attach(store);
    return window.app.project?.where ?? null;
  }, [kind, name]);
  await page.goto(appUrl(`?video=${encodeURIComponent(urlOf(video))}&val=0`));
  await wait(() => window.app?.sfm?.state.status === 'ready');
  await page.evaluate(() => window.app.sfm.run(null, true));
  await wait(() => window.app.trainer?.count > 0 && window.app.colmap?.fromSfm && window.app.gtAvailable);  // the photos are readable
  await page.evaluate(() => window.app._setTraining(true));
  await wait((n) => window.app.trainer.stepCount >= n, STEPS);
  await page.evaluate(() => window.app._setTraining(false));
  await new Promise(r => setTimeout(r, 2000));  // the frame in flight finishes its steps
  const before = await page.evaluate(() => ({ step: window.app.trainer.stepCount, cams: window.app.colmap.images.length,
                                              keyframes: window.app.sfm.quality.frames.length }));

  // a new project on serve.py, filled with the session, its outputs and a snapshot
  const where = await attach('server', PROJECT);
  await wait(() => !window.app.sfm.busy);
  const dir = join(projects, PROJECT), files = walk(dir);
  const n = (re) => files.filter(f => re.test(f)).length;
  console.log(`${where}: ${files.length} files (${n(/^images\//)} keyframes, ${n(/^gs\/images\//)} undistorted)`);
  const want = ['cache/features.gosfm', 'cache/model.gosfm', 'gs/images/*.jpg', 'gs/sparse/cameras.bin', 'gs/sparse/images.bin',
                'gs/sparse/points3D.bin', 'images/*.jpg', 'project.json', 'sharpness.csv', 'sparse/0/cameras.bin',
                'sparse/0/images.bin', 'sparse/0/points3D.bin', `splats/gaussopt_snap_${String(before.step).padStart(5, '0')}.gosnap`];
  if (JSON.stringify(shape(files)) !== JSON.stringify(want)) fail(`project files: ${JSON.stringify(shape(files))}`);
  const served = want.map(f => f.replace(/_\d{5}\.gosnap$/, '.gosnap'));  // later snapshots have other steps
  if (n(/^images\//) !== before.keyframes || n(/^gs\/images\//) !== before.cams) fail(`${n(/^images\//)} keyframes, ${n(/^gs\/images\//)} undistorted`);
  const m = JSON.parse(readFileSync(join(dir, 'project.json'), 'utf8'));
  if (m.source?.name !== video.split('/').pop() || m.phases.keyframes?.count !== before.keyframes || m.phases.cameras?.exported !== before.cams ||
      m.phases.train?.step !== before.step) fail(`project.json: ${JSON.stringify(m)}`);
  // the distorted model opens in COLMAP, when it is installed
  const analyzed = spawnSync(`${process.env.HOME}/gskal/colmap/build/src/colmap/exe/colmap`, ['model_analyzer', '--path', join(dir, 'sparse/0')], { encoding: 'utf8' });
  if (!analyzed.error) {
    const registered = /Registered images: (\d+)/.exec(analyzed.stdout + analyzed.stderr)?.[1];
    console.log(`sparse/0: colmap model_analyzer finds ${registered} registered images`);
    if (Number(registered) !== before.cams) fail(`colmap model_analyzer on sparse/0: ${(analyzed.stdout + analyzed.stderr).split('\n').slice(-3).join(' ')}`);
  }

  // settings: a click on a control goes into project.json (after a pause), and comes back on reopening;
  // so does the scene's transform, after three Level turns ([ / ])
  const identity = JSON.parse(readFileSync(join(dir, 'project.json'), 'utf8')).transform;
  if (JSON.stringify(identity) !== '[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]') fail(`a new project's transform: ${JSON.stringify(identity)}`);
  const leveled = await page.evaluate(async () => {
    const click = (guis, key) => guis.flatMap(g => g.controllersRecursive()).find(c => c.property === key)._cb_el.click();
    click(window.app.guis, 'remove_floaters');
    click(window.app.sfm.guis, 'loops');
    for (let k = 0; k < 3; k++) window.dispatchEvent(new KeyboardEvent('keydown', { key: ']' }));
    return { T: window.app.sceneTransform, up: window.app.worldUp, est: window.app.upEstimate };
  });
  const turned = Math.acos(leveled.up.reduce((s, v, k) => s + v * leveled.est[k], 0)) * 180 / Math.PI;
  if (!(turned > 1 && turned <= 3.001)) fail(`3 Level turns: the up ${turned.toFixed(2)}° from the cameras' up`);
  const saved = await (async () => {
    for (let t = 0; t < 50; t++) {
      const m = JSON.parse(readFileSync(join(dir, 'project.json'), 'utf8')), st = m.settings;
      if (st?.app?.remove_floaters === false && st?.sfm?.loops === false && JSON.stringify(m.transform) === JSON.stringify(leveled.T)) return true;
      await new Promise(r => setTimeout(r, 200));
    }
    return false;
  })();
  if (!saved) { const m = JSON.parse(readFileSync(join(dir, 'project.json'), 'utf8')); fail(`project.json settings: ${JSON.stringify(m.settings)}, transform ${JSON.stringify(m.transform)}`); }

  // a reload without the video reopens the project: session (repairs), gs/ dataset, snapshot, settings
  await page.goto(appUrl(`?dataset=[synth]`));
  await wait(() => !!(window.app?.trainer && window.app.projectPanel));
  await attach('server', PROJECT);
  await wait(() => window.app.colmap?.fromSfm && window.app.sfm?.quality && window.app.trainer?.stepCount > 0);
  const after = await page.evaluate(() => ({ step: window.app.trainer.stepCount, cams: window.app.colmap.images.length,
                                             keyframes: window.app.sfm.quality.frames.length }));
  console.log(`reopened: step ${after.step}, ${after.cams} cameras, ${after.keyframes} keyframes`);
  if (JSON.stringify(after) !== JSON.stringify(before)) fail(`reopened ${JSON.stringify(after)}, expected ${JSON.stringify(before)}`);
  const restored = await page.evaluate(() => ({ floaters: window.app.params.remove_floaters, loops: window.app.sfm.o.loops, T: window.app.sceneTransform, up: window.app.worldUp }));
  console.log(`reopened settings: ${JSON.stringify({ floaters: restored.floaters, loops: restored.loops })}, leveled ${turned.toFixed(2)}°`);
  if (restored.floaters !== false || restored.loops !== false) fail(`settings not restored: ${JSON.stringify(restored)}`);
  if (JSON.stringify(restored.T) !== JSON.stringify(leveled.T) || Math.hypot(...restored.up.map((v, k) => v - leveled.up[k])) > 1e-6) {
    fail(`transform not restored: ${JSON.stringify(restored.T)} up ${restored.up}, saved ${JSON.stringify(leveled.T)} up ${leveled.up}`);
  }
  // the leveling in the snapshot too: one taken now, the level reset, the snapshot restored
  const snapT = await page.evaluate(async () => {
    const app = window.app, { buf } = await app._encodeSnapshot('manual');
    const { decodeSnapshot } = await import('./src/snapshot.js');
    const header = decodeSnapshot(buf).header.transform;
    app.resetLevel();
    const reset = [...app.sceneTransform];
    await app.restoreSnapshot(buf, 'e2e.gosnap', app.colmap, ++app.loadGen);
    return { header, reset, restored: app.sceneTransform };
  });
  if (JSON.stringify(snapT.header) !== JSON.stringify(leveled.T) || JSON.stringify(snapT.reset) === JSON.stringify(leveled.T) ||
      JSON.stringify(snapT.restored) !== JSON.stringify(leveled.T)) fail(`snapshot transform: ${JSON.stringify(snapT)}, saved ${JSON.stringify(leveled.T)}`);
  await page.evaluate(() => { const s = window.app.sfm; s._select(s.quality.frames.slice(5, 10).map(f => f.name)); s.act('retry', [s.selected]); });
  await wait(() => !window.app.sfm.busy && /retry done|error/.test(window.app.sfm.state.status));
  const repair = await page.evaluate(() => window.app.sfm.state.status);
  console.log(`repair without the video: ${repair}`);
  if (!/retry done/.test(repair)) fail(repair);

  // the same project into a folder handle (not in Safari: it cannot post a handle to a worker,
  // DataCloneError, and has no folder picker anyway; its project folders are serve.py's or a .zip)
  if (!SAFARI) {
    await page.evaluate((name) => navigator.storage.getDirectory().then(d => d.removeEntry(name, { recursive: true })).catch(() => {}), PROJECT);
    await attach('folder', PROJECT);
    await wait(() => !window.app.sfm.busy);
    const listed = await page.evaluate(async () => {
      const walk = async (dir) => (await Promise.all((await window.app.project.list(dir)).map(e =>
        e.kind === 'directory' ? walk(`${dir}${e.name}/`) : [`${dir}${e.name}`]))).flat();
      return walk('');
    });
    const local = shape(listed).map(f => f.replace(/_\d{5}\.gosnap$/, '.gosnap'));
    console.log(`folder handle: ${listed.length} files`);
    if (JSON.stringify(local) !== JSON.stringify(served)) fail(`folder files: ${JSON.stringify(local)}`);
    await page.evaluate((name) => navigator.storage.getDirectory().then(d => d.removeEntry(name, { recursive: true })).catch(() => {}), PROJECT);
  }

  if (!SAFARI) {  // the .zip download (Safari through safaridriver: no download events)
    const download = page.waitForEvent('download', { timeout: 300000 });
    await page.evaluate(() => window.app.projectPanel.downloadZip());
    const d = await download, path = join(projects, 'download.zip');
    await d.saveAs(path);
    const names = execFileSync('python3', ['-I', '-c', 'import sys, zipfile; z = zipfile.ZipFile(sys.argv[1]); assert z.testzip() is None; print("\\n".join(z.namelist()))', path]).toString().trim().split('\n');
    console.log(`zip: ${d.suggestedFilename()}, ${names.length} files, ${(statSync(path).size / 1e6).toFixed(0)} MB`);
    if (JSON.stringify(shape(names).map(f => f.replace(/_\d{5}\.gosnap$/, '.gosnap'))) !== JSON.stringify(served)) fail(`zip files: ${JSON.stringify(shape(names))}`);
  }
}, {
  serverArgs: ['--projects', projects],
  onFail: async (page) => { const log = await page.evaluate(() => window.app.logText()); return `log:\n${log.split('\n').slice(-15).join('\n')}`; },
  cleanup: () => rmSync(projects, { recursive: true, force: true }),
});
