// Phase states of the stepper (src/ui/phases.js), derived from what is loaded and running.
import { phaseStates, advanceTo, phaseLabel, phaseShown, canMesh, nextPhase, followRunning, phaseActions, phaseNote, optionsChanged, savedLine,
         savedNotice } from '../src/ui/phases.js';
import { assert } from './util.js';

const states = (s) => Object.fromEntries(Object.entries(phaseStates({ label: '', video: null, sfm: null, colmap: null, trainer: null, mesh: null, ...s }))
  .map(([k, v]) => [k, v.state]));
const input = (s) => ({ label: '', video: null, sfm: null, colmap: null, trainer: null, mesh: null, ...s });
const quality = (version = 0) => ({ keyframes: 35, registered: 34, problems: 2, version });

Deno.test('phases: nothing loaded, a COLMAP dataset, a video through the stages', () => {
  // nothing: the project the step to do, the video can be opened as well; the rest waits, Dome and Mesh optional
  assert(JSON.stringify(states({})) === JSON.stringify(
    { project: 'next', video: 'ready', keyframes: 'locked', cameras: 'locked', dome: 'optional', train: 'locked', mesh: 'optional', export: 'locked' }),
    JSON.stringify(states({})));
  const none = phaseStates(input({}));
  assert(none.keyframes.chip === 'Needs video' && none.train.chip === 'Needs cameras' && none.export.chip === 'Needs train' &&
         none.dome.chip === 'Optional · off' && none.mesh.chip === 'Optional' && none.mesh.summary === 'after training', JSON.stringify(none));

  const colmap = states({ label: '/dataset', colmap: { cams: 152, fromSfm: false }, trainer: { step: 0, training: false, psnr: 0 } });
  assert(colmap.video === 'skipped' && colmap.keyframes === 'skipped' && colmap.cameras === 'done' && colmap.train === 'next', JSON.stringify(colmap));

  const video = { name: 'a.mp4' }, sfm = (x) => ({ running: null, busy: false, keyframes: null, quality: null, sentVersion: null, ...x });
  assert(states({ video, sfm: sfm({ running: 'keyframes' }) }).keyframes === 'running');
  const kf = states({ video, sfm: sfm({ keyframes: 35, running: 'cameras' }) });
  assert(kf.keyframes === 'done' && kf.cameras === 'running' && kf.train === 'locked');
  // a run going: nothing after it is next; its progress on the step
  const at = phaseStates(input({ video, sfm: sfm({ keyframes: 35, running: 'cameras', progress: { done: 12, total: 48 } }) }));
  assert(at.cameras.progress === 0.25 && !Object.values(at).some(o => o.state === 'next'), JSON.stringify(at));
  const sent = { video, sfm: sfm({ keyframes: 35, quality: quality(0), sentVersion: 0 }), colmap: { cams: 34, fromSfm: true },
                 trainer: { step: 1200, training: true, psnr: 17.2 } };
  assert(states(sent).cameras === 'done' && states(sent).train === 'running' && states(sent).export === 'ready');
  assert(phaseStates({ ...sent, trainer: { ...sent.trainer, steps: 4800 } }).train.progress === 0.25);
  const s = phaseStates(sent);
  assert(s.cameras.summary === '34/35 ⚠2' && s.train.summary === 'step 1200 · 17.2 dB', JSON.stringify(s));
  // a repair after the hand-over: the trainer's cameras are older than the model
  const repaired = states({ ...sent, sfm: sfm({ keyframes: 35, quality: quality(1), sentVersion: 0 }), trainer: { ...sent.trainer, training: false } });
  assert(repaired.train === 'stale');
  assert(/Update training cameras/.test(phaseNote('train', input({ ...sent, sfm: sfm({ keyframes: 35, quality: quality(1), sentVersion: 0 }), trainer: { ...sent.trainer, training: false } }))));
});

Deno.test('phases: a finished step moves forward only', () => {
  assert(advanceTo('video', 'keyframes') === 'keyframes');
  assert(advanceTo('cameras', 'train') === 'train');
  assert(advanceTo('export', 'train') === null && advanceTo('train', 'train') === null);
});

// A photo set (DESIGN-sfm §13.3 P5): Photos in place of Video, no Keyframes step
Deno.test('phases: a photo set has a Photos phase and no Keyframes', () => {
  const sfm = { running: null, busy: false, keyframes: 18, quality: { keyframes: 18, registered: 18, problems: 0, version: 0 }, sentVersion: 0 };
  const s = phaseStates({ label: '', video: { name: '18 photos', photos: 18 }, sfm, colmap: { cams: 12, fromSfm: true }, trainer: null });
  assert(s.video.state === 'done' && s.video.summary === '18 photos' && s.keyframes.state === 'skipped', JSON.stringify(s));
  assert(phaseLabel('video', { photos: true }) === 'Photos' && phaseLabel('video') === 'Video' && phaseLabel('cameras', { photos: true }) === 'Cameras');
  assert(!phaseShown('keyframes', { photos: true }) && phaseShown('keyframes') && phaseShown('video', { photos: true }));
});

// The Mesh step follows the mesh (app.js saveMesh: app.meshState); Train's "Make mesh →" needs trained splats
Deno.test('phases: the mesh drives the Mesh step', () => {
  const base = { label: '/dataset', video: null, sfm: null, colmap: { cams: 152, fromSfm: false }, mesh: null };
  const at = (step, mesh = null) => phaseStates({ ...base, trainer: { step, training: false, psnr: 20 }, mesh });
  assert(at(0).mesh.state === 'optional' && !canMesh({ ...base, trainer: { step: 0 } }) && !canMesh({ ...base, trainer: null }));
  assert(canMesh({ ...base, trainer: { step: 10 } }));
  const running = { running: true, step: 5000 };
  assert(at(5000, running).mesh.state === 'running' && !canMesh({ ...base, trainer: { step: 5000 }, mesh: running }));
  const made = { running: false, step: 5000, faces: 308123, texture: 2048, mb: 8.78 };
  const done = at(5000, made).mesh;
  assert(done.state === 'done' && done.summary === '308k faces · 2048² · 8.8 MB', JSON.stringify(done));
  assert(at(5000, { ...made, texture: 0 }).mesh.summary === '308k faces · 8.8 MB');
  // trained on after the mesh, or new splats since: it is out of date
  const stale = at(6000, made).mesh;
  assert(stale.state === 'stale' && stale.summary === 'mesh @ step 5000', JSON.stringify(stale));
  assert(at(200, made).mesh.state === 'stale');
  assert(at(5000, { running: false, step: 5000, error: 'x' }).mesh.state === 'optional');
});

// The footer of each phase (phaseActions): its next move is the primary button; a phase whose work
// is done offers the next phase instead of a re-run
const labels = (phase, s) => phaseActions(phase, input(s)).map(a => `${a.primary ? '*' : ''}${a.label}${a.enabled ? '' : ' (off)'}`).join(' | ');
const sfmOf = (x) => ({ running: null, busy: false, keyframes: null, quality: null, sentVersion: null, canRun: true, ready: false, canUpdate: false, ...x });

Deno.test('phases: the next phase skips what has nothing to do', () => {
  const colmap = input({ label: '/dataset', colmap: { cams: 152, fromSfm: false }, trainer: { step: 0 } });
  assert(nextPhase('project', colmap) === 'train', nextPhase('project', colmap));  // no video, keyframes, nor cameras to find
  const video = input({ video: { name: 'a.mp4' }, sfm: sfmOf({}) });
  assert(nextPhase('project', video) === 'video' && nextPhase('video', video) === 'keyframes' && nextPhase('export', video) === null);
  const photos = input({ video: { name: '18 photos', photos: 18 }, sfm: sfmOf({}) });
  assert(nextPhase('video', photos) === 'cameras', nextPhase('video', photos));
});

Deno.test('phases: a run that starts ahead is followed, once', () => {
  const video = { name: 'a.mp4' };
  const at = (x) => phaseStates(input({ video, sfm: sfmOf(x) }));
  const idle = at({}), kf = at({ running: 'keyframes' }), cams = at({ keyframes: 35, running: 'cameras' });
  assert(followRunning(idle, kf, 'video') === 'keyframes');
  assert(followRunning(kf, cams, 'keyframes') === 'cameras');
  assert(followRunning(cams, cams, 'video') === null, 'only on the change: a user gone back stays');
  assert(followRunning(idle, cams, 'train') === null, 'never backwards');
  assert(followRunning(null, kf, 'project') === 'keyframes');
});

Deno.test('phases: each footer leads to the next step', () => {
  assert(labels('project', {}) === '', labels('project', {}));
  assert(labels('project', { label: '/dataset', colmap: { cams: 152, fromSfm: false } }) === '*Next: Train →');
  const video = { name: 'a.mp4' };
  assert(labels('video', { video, sfm: sfmOf({}) }) === 'Run all, then train | *Find keyframes →');
  assert(labels('video', { video, sfm: sfmOf({ canRun: false }) }) === 'Run all, then train (off) | *Find keyframes → (off)');
  assert(labels('video', { video, sfm: sfmOf({ running: 'keyframes' }) }) === '■ Stop');
  assert(labels('video', { video, sfm: sfmOf({ busy: true }) }) === '■ Stop (off)');
  assert(labels('video', { video, sfm: sfmOf({ keyframes: 35 }) }) === 'Find keyframes again | *Next: Keyframes →');
  assert(labels('keyframes', { video, sfm: sfmOf({ keyframes: 35 }) }) === '*Find cameras →');
  // the Cameras step before any camera is found (keyframes only, or nothing run yet): it finds them
  assert(labels('cameras', { video, sfm: sfmOf({ keyframes: 35 }) }) === '*Find cameras →', labels('cameras', { video, sfm: sfmOf({ keyframes: 35 }) }));
  assert(labels('cameras', { video, sfm: sfmOf({}) }) === '*Find cameras →');
  const q = { keyframes: 35, registered: 34, problems: 2, version: 3 };
  const found = { video, sfm: sfmOf({ keyframes: 35, quality: q, ready: true }) };
  assert(labels('keyframes', found) === 'Find cameras again | *Next: Cameras →', labels('keyframes', found));
  assert(labels('cameras', found) === '*Train on these cameras →', labels('cameras', found));
  // the trainer has these cameras: on to Train; a repair since: update them, or start over
  const sent = { ...found, sfm: { ...found.sfm, sentVersion: 3, canUpdate: true }, colmap: { cams: 34, fromSfm: true }, trainer: { step: 0, gt: true } };
  assert(labels('cameras', sent) === 'Restart training | *Next: Train →', labels('cameras', sent));
  const repaired = { ...sent, sfm: { ...sent.sfm, sentVersion: 2 } };
  assert(labels('cameras', repaired) === 'Update training cameras | *Train on these cameras →', labels('cameras', repaired));
  // a photo set: its photos are the keyframes
  const photos = { video: { name: '18 photos', photos: 18 }, sfm: sfmOf({}) };
  assert(labels('video', photos) === 'Run all, then train | *Find cameras →');
  assert(labels('video', { ...photos, sfm: sfmOf({ keyframes: 18, quality: q }) }) === 'Find cameras again | *Next: Cameras →');
});

Deno.test('phases: Train starts, pauses, then leads to Export, the mesh on the way', () => {
  const colmap = { cams: 152, fromSfm: false };
  assert(labels('train', { colmap }) === 'Step ×1 (off) | *▶ Train (off) | Make mesh → (off)', labels('train', { colmap }));
  assert(labels('train', { colmap, trainer: { step: 0, gt: true } }) === 'Step ×1 | *▶ Train | Make mesh → (off)');
  assert(labels('train', { colmap, trainer: { step: 900, gt: true, training: true } }) === 'Step ×1 (off) | *⏸ Pause | Make mesh → | Next: Export →');
  assert(labels('train', { colmap, trainer: { step: 900, gt: true } }) === 'Step ×1 | ▶ Resume | Make mesh → | *Next: Export →');
  const trainer = { step: 900, gt: true };
  assert(labels('mesh', { colmap, trainer }) === '*Make mesh + scene (.glb)');
  assert(labels('mesh', { colmap, trainer, mesh: { running: true, step: 900 } }) === '*Make mesh + scene (.glb) (off)');
  assert(labels('mesh', { colmap, trainer, mesh: { running: false, step: 900, faces: 1000, mb: 1 } }) === 'Make mesh again | *Next: Export →',
         labels('mesh', { colmap, trainer, mesh: { running: false, step: 900, faces: 1000, mb: 1 } }));
  // the mesh is optional: Train's next step is Export
  assert(nextPhase('train', input({ colmap, trainer })) === 'export');
  assert(labels('export', { colmap, trainer }) === '*Save scene (.glb)');
  assert(labels('export', { colmap, trainer, saved: { export: { at: 0, file: 'splats/gaussopt_00900.scene.glb', where: 'project', step: 900 } } }) === 'Save scene again');
});

// The Dome step (src/dome.js, ui/dome_step.js): off by default (optional), auto: before the training
Deno.test('phases: the Dome step', () => {
  const colmap = { cams: 152, fromSfm: false }, trainer = { step: 0, gt: true };
  const dome = (x) => ({ mode: 'auto', running: false, result: null, ...x });
  const st = (x) => phaseStates(input({ label: '/dataset', colmap, trainer, dome: x })).dome;
  assert(st(null).state === 'optional' && st(dome({ mode: 'off' })).chip === 'Optional · off');
  const early = phaseStates(input({ dome: dome() })).dome;
  assert(early.state === 'optional' && early.summary === 'needs cameras', JSON.stringify(early));
  assert(st(dome()).state === 'next' && st(dome({ running: true })).state === 'running');
  const found = st(dome({ result: { worth: true, farShare: 0.138 } }));
  assert(found.state === 'done' && found.summary === 'sky 14%', JSON.stringify(found));
  assert(st(dome({ result: { worth: true, farShare: 0.138, fill: 'panorama' } })).summary === 'sky 14%, panorama');  // its fill
  assert(st(dome({ result: { worth: false, farShare: 0.0018 } })).summary === 'no dome (sky 0.2%)');
  assert(st(dome({ result: { worth: true, farShare: null } })).summary === 'from the scene');
  // auto: Dome comes between Cameras (or the project) and Train; off: it is passed over
  assert(nextPhase('project', input({ label: '/dataset', colmap, trainer, dome: dome() })) === 'dome');
  assert(nextPhase('project', input({ label: '/dataset', colmap, trainer, dome: dome({ mode: 'off' }) })) === 'train');
  const at = (x) => labels('dome', { label: '/dataset', colmap, trainer, dome: dome(x) });
  assert(at({}) === '*Find the sky | Next: Train →', at({}));
  assert(at({ running: true }) === '*Find the sky (off) | Next: Train →', at({ running: true }));
  assert(at({ result: { worth: true, farShare: 0.1 } }) === 'No dome | Find it again | *Next: Train →');
  assert(at({ result: { worth: false, farShare: 0.001 } }) === 'Find it again | *Next: Train →');
  // not while training: the dome is made from the photos before the splats
  assert(labels('dome', { label: '/dataset', colmap, trainer: { step: 10, gt: true, training: true }, dome: dome() }) === '*Find the sky (off) | Next: Train →');
});

// What a step waits for, and the step to do now: one at a time, none while a run goes
Deno.test('phases: needs, next and ready', () => {
  const video = { name: 'a.mp4' };
  const s = phaseStates(input({ label: 'p', video, sfm: sfmOf({}) }));
  assert(s.video.state === 'done' && s.keyframes.state === 'next' && s.cameras.state === 'ready' && s.train.state === 'locked' &&
         s.train.needs === 'cameras' && s.export.chip === 'Needs train', JSON.stringify(s));
  // locked: its footer leads to what it waits for first, its own buttons not primary
  assert(labels('train', { label: 'p', video, sfm: sfmOf({}) }) === '*Go to Cameras | Step ×1 (off) | ▶ Train (off) | Make mesh → (off)');
  assert(phaseNote('train', input({ video, sfm: sfmOf({}) })) === 'Cameras first: this step is made from it.');
  // a photo set: its cameras wait for the photos (Photos), the Keyframes step is not there
  const photos = phaseStates(input({ video: { name: '18 photos', photos: 18 }, sfm: sfmOf({}) }));
  assert(photos.cameras.state === 'next' && phaseStates(input({})).cameras.chip === 'Needs video', JSON.stringify(photos.cameras));
  assert(phaseStates(input({ video: { name: '18 photos', photos: 18 } })).train.chip === 'Needs cameras');
  // trained and paused: Export next (the mesh optional), nothing saved yet
  const colmap = { cams: 152, fromSfm: false }, t = phaseStates(input({ label: '/d', colmap, trainer: { step: 900, gt: true } }));
  assert(t.train.chip === 'Paused' && t.mesh.state === 'optional' && t.export.state === 'next' && t.export.summary === 'nothing saved yet', JSON.stringify(t));
});

// A done step whose options changed since it ran: ✎ Options changed, run again or revert; nothing after it out of date
Deno.test('phases: options changed', () => {
  assert(optionsChanged({ spacing: 0.05, minRate: 4 }, { spacing: 0.04, minRate: 4 }, ['spacing', 'minRate']) === 'spacing 0.05 → 0.04');
  assert(optionsChanged({ a: 1, b: 2, c: 3 }, { a: 2, b: 3, c: 4 }, ['a', 'b', 'c']) === 'a 1 → 2, b 2 → 3 +1');
  assert(optionsChanged({ a: 1 }, { a: 1, b: 5 }, ['a', 'b']) === '', 'a key the run did not record is not compared');
  assert(optionsChanged(null, { a: 1 }, ['a']) === '' && optionsChanged({ x: 0.1 + 0.2 }, { x: 0.4 }, ['x']) === 'x 0.3 → 0.4');
  const video = { name: 'a.mp4' }, q = { keyframes: 35, registered: 34, problems: 2, version: 3 };
  const base = { video, sfm: sfmOf({ keyframes: 35, quality: q, ready: true, sentVersion: 3 }), colmap: { cams: 34, fromSfm: true }, trainer: { step: 900, gt: true } };
  const ch = { ...base, changed: { keyframes: 'spacing 0.05 → 0.04' } }, s = phaseStates(input(ch));
  assert(s.keyframes.state === 'changed' && s.keyframes.chip === 'Options changed' && s.keyframes.summary === 'spacing 0.05 → 0.04', JSON.stringify(s.keyframes));
  assert(s.cameras.state === 'done' && s.train.state === 'done', 'not run again: nothing after it changes');
  assert(labels('keyframes', ch) === 'Revert the options | *Find keyframes again', labels('keyframes', ch));
  assert(/spacing 0\.05 → 0\.04/.test(phaseNote('keyframes', input(ch))));
  assert(labels('cameras', { ...base, changed: { cameras: 'window 10 → 12' } }) === 'Revert the options | *Find cameras again');
  assert(labels('keyframes', { ...ch, sfm: { ...ch.sfm, busy: true } }) === 'Revert the options | *Find keyframes again (off)');
  // the mesh's: from the same splats; a mesh not made has nothing changed
  const mesh = { running: false, step: 900, faces: 1000, mb: 1 };
  assert(phaseStates(input({ ...base, mesh, changed: { mesh: 'mesh_res 256 → 384' } })).mesh.state === 'changed');
  assert(labels('mesh', { ...base, mesh, changed: { mesh: 'mesh_res 256 → 384' } }) === 'Revert the options | *Make mesh again');
  assert(phaseStates(input({ ...base, changed: { mesh: 'x' } })).mesh.state === 'optional');
  // done with options: how to redo it with others
  assert(/change them below/.test(phaseNote('cameras', input(base))));
});

// The saves: Export out of date once the splats trained on; the notices say what can be recovered
Deno.test('phases: saved', () => {
  const colmap = { cams: 152, fromSfm: false }, ex = { at: 0, file: 'splats/gaussopt_00900.scene.glb', where: 'project', step: 900 };
  const at = (step) => phaseStates(input({ colmap, trainer: { step, gt: true }, saved: { export: ex } })).export;
  assert(at(900).state === 'done' && at(900).chip === 'Saved' && at(900).summary === 'splats/gaussopt_00900.scene.glb @ 900');
  assert(at(1200).state === 'stale' && phaseNote('export', input({ colmap, trainer: { step: 1200 }, saved: { export: ex } })) ===
         'The last save is of step 900; the splats are at step 1200.');
  const snap = { at: 0, file: 'splats/gaussopt_snap_12000.scene.glb', where: 'project', step: 12000 };
  assert(savedNotice('train', snap) === 'Snapshot @ step 12000 saved in the project (splats/gaussopt_snap_12000.scene.glb): training resumes from it after a reload or a crash.',
         savedNotice('train', snap));
  assert(savedNotice('keyframes', { file: 'cache/features.gosfm', where: 'browser' }) ===
         'Keyframes saved in this browser (cache/features.gosfm): reopening the same video goes on from here.');
  assert(savedLine('mesh', { file: 'splats/gaussopt_00900.mesh.glb', where: 'download' }, '12:00') === 'Saved 12:00 as a download (splats/gaussopt_00900.mesh.glb).');
});
