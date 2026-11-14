// snapshot_glb.e2e.js — Snapshots as scene .glb files (snapshot.js) in the page:
//   - a snapshot after training is a scene .glb with its training state: restored, the splats, Adam's m and v and the
//     moments are on the GPU bit for bit as they were, the step count too; dropped as a file, it resumes the same way;
//   - an older .gosnap of the same state restores the same, and so does the .glb gosnapToSceneGlb makes of it;
//   - a scene saved to view (no training state) loads its splats, the optimizer anew;
//   - with SNAPSHOT_E2E_PROJECT (a project folder whose latest snapshot is a .gosnap), a clone of it converted by
//     tools/gosnap2glb.js reopens from its .scene.glb at the step and splat count the project says.
// On the synthetic room (MESH_E2E_ROOM, as dirty.e2e.js) served by serve.py.
//   cd e2e && node snapshot_glb.e2e.js   (SNAPSHOT_E2E_PROJECT: optional, e.g. ~/gskal/brush/photo_runs/projects/alpine-lake-01-cpuba)
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';
import { HERE, fail, appUrl, run, room, appReady } from './harness.js';

const ROOM = room(), SOURCE = process.env.SNAPSHOT_E2E_PROJECT ?? null;
const projects = mkdtempSync(join(tmpdir(), 'gaussopt-snapglb-'));
let converted = null;
if (SOURCE) {
  const name = basename(SOURCE), dir = join(projects, name);
  execFileSync('cp', ['-c', '-R', SOURCE, dir]);  // a clone (APFS): the project itself untouched
  const out = execFileSync('deno', ['run', '--allow-read', '--allow-write', join(HERE, '..', 'tools', 'gosnap2glb.js'), '--project', dir]).toString();
  console.log(out.trim());
  converted = { name, train: JSON.parse(readFileSync(join(dir, 'project.json'), 'utf8')).phases.train };
  if (!converted.train.latest.endsWith('.scene.glb') || !existsSync(join(dir, converted.train.latest))) fail(`converted latest ${converted.train.latest}`);
}

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await appReady(page);
  const r = await page.evaluate(async () => {
    const app = window.app, util = await import('./e2e/page_util.js'), snap = await import('./src/snapshot.js'), sg = await import('./src/scene_glb.js');
    await util.train(app, 40);
    const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
    const state = async () => ({ step: app.trainer.stepCount, count: app.trainer.count, gpu: await util.splatState(app.trainer, true) });
    const sameState = (a, b) => a.step === b.step && a.count === b.count && a.gpu.every((x, k) => same(x, b.gpu[k]));
    const before = await state(), { buf, name } = await app._encodeSnapshot('manual');
    const out = { name, kind: snap.snapshotKind(buf), mb: +(buf.byteLength / 1e6).toFixed(1), step: before.step };
    // trained on, then back to the snapshot
    await util.train(app, 10);
    await app.restoreSnapshot(buf, name, app.colmap, ++app.loadGen);
    out.restored = sameState(await state(), before);
    // dropped as a file
    await util.train(app, 10);
    await app.loadGlbFile(new File([buf], name));
    out.dropped = sameState(await state(), before);
    // an older .gosnap of it, and that .gosnap converted
    const s = snap.snapshotOfScene(sg.readSceneGlb(buf));
    const gosnap = snap.encodeSnapshot({ ...s.header, octree: { ...s.header.octree, leafIndices: undefined } }, s.blobs);
    await util.train(app, 10);
    await app.restoreSnapshot(gosnap, 'e2e.gosnap', app.colmap, ++app.loadGen);
    out.gosnap = sameState(await state(), before);
    const conv = snap.gosnapToSceneGlb(gosnap);
    await util.train(app, 10);
    await app.loadGlbFile(new File([conv], 'converted.scene.glb'));
    out.converted = sameState(await state(), before);
    // a scene to view: its splats, a new optimizer
    app._saveOrDownload = async (bytes) => { out.scene = bytes; };  // the file kept here, not downloaded
    await app.saveSceneGlb();
    delete app._saveOrDownload;
    await app.loadGlbFile(new File([out.scene], 'view.scene.glb'));
    const v = await state();
    out.view = { step: v.step, count: v.count, splatsSame: same(v.gpu[0], before.gpu[0]), adamZero: v.gpu[1].every(x => x === 0) };
    delete out.scene;
    out.errors = document.getElementById('error-line').textContent;
    return out;
  });
  console.log(JSON.stringify(r));
  if (r.kind !== 'glb' || !r.name.endsWith('.scene.glb')) fail(`the snapshot: ${r.name}, kind ${r.kind}`);
  if (!r.restored) fail('restored from the .glb: not the state it was taken at');
  if (!r.dropped) fail('dropped as a file: not the state it was taken at');
  if (!r.gosnap) fail('the .gosnap of the same state: not restored to it');
  if (!r.converted) fail('the .gosnap converted to a .glb: not restored to it');
  if (r.view.step !== 0 || !r.view.adamZero) fail(`a scene to view: ${JSON.stringify(r.view)}`);
  if (r.errors) fail(`page error: ${r.errors}`);

  let proj = '';
  if (converted) {
    await page.goto(appUrl('?dataset=[synth]'));
    await page.waitForFunction(() => !!(window.app?.trainer && window.app.projectPanel), null, { timeout: 120000 });
    await page.evaluate(async (name) => {
      const { ServerStore } = await import('./src/project/store.js');
      await window.app.projectPanel.attach(new ServerStore(name));
    }, converted.name);
    await page.waitForFunction(() => window.app.colmap?.fromSfm && window.app.trainer?.stepCount > 0, null, { timeout: 600000, polling: 500 });
    const p = await page.evaluate(() => ({ step: window.app.trainer.stepCount, count: window.app.trainer.count,
                                           log: window.app.logText().split('\n').filter(l => /Restoring snapshot|Resumed snapshot/.test(l)) }));
    console.log(`${converted.name} reopened: ${JSON.stringify(p)}`);
    if (p.step !== converted.train.step || p.count !== converted.train.splats || !p.log.some(l => l.includes(converted.train.latest.split('/').pop())))
      fail(`${converted.name}: step ${p.step}, ${p.count} splats from ${p.log}; the project says ${JSON.stringify(converted.train)}`);
    proj = `; ${converted.name} converted, reopened at step ${p.step} with ${p.count} splats`;
  }
  return `a ${r.mb} MB snapshot .glb at step ${r.step}, restored, dropped, from a .gosnap and converted: the same state; a scene to view: new optimizer${proj}`;
}, { serverArgs: [ROOM, '--projects', projects] });
