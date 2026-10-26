// Phase states of the stepper (src/ui/phases.js), derived from what is loaded and running.
import { phaseStates, advanceTo, phaseLabel, phaseShown, canMesh, nextPhase, followRunning, phaseActions } from '../src/ui/phases.js';
import { assert } from './util.js';

const states = (s) => Object.fromEntries(Object.entries(phaseStates({ label: '', video: null, sfm: null, colmap: null, trainer: null, mesh: null, ...s }))
  .map(([k, v]) => [k, v.state]));
const quality = (version = 0) => ({ keyframes: 35, registered: 34, problems: 2, version });

Deno.test('phases: nothing loaded, a COLMAP dataset, a video through the stages', () => {
  assert(JSON.stringify(states({})) === JSON.stringify(
    { project: 'todo', video: 'todo', keyframes: 'todo', cameras: 'todo', train: 'todo', export: 'todo' }));

  const colmap = states({ label: '/dataset', colmap: { cams: 152, fromSfm: false }, trainer: { step: 0, training: false, psnr: 0 } });
  assert(colmap.video === 'skipped' && colmap.keyframes === 'skipped' && colmap.cameras === 'done' && colmap.train === 'todo', JSON.stringify(colmap));

  const video = { name: 'a.mp4' }, sfm = (x) => ({ running: null, busy: false, keyframes: null, quality: null, sentVersion: null, ...x });
  assert(states({ video, sfm: sfm({ running: 'keyframes' }) }).keyframes === 'running');
  const kf = states({ video, sfm: sfm({ keyframes: 35, running: 'cameras' }) });
  assert(kf.keyframes === 'done' && kf.cameras === 'running' && kf.train === 'todo');
  const sent = { video, sfm: sfm({ keyframes: 35, quality: quality(0), sentVersion: 0 }), colmap: { cams: 34, fromSfm: true },
                 trainer: { step: 1200, training: true, psnr: 17.2 } };
  assert(states(sent).cameras === 'done' && states(sent).train === 'running');
  const s = phaseStates(sent);
  assert(s.cameras.summary === '34/35 ⚠2' && s.train.summary === 'step 1200 · 17.2 dB', JSON.stringify(s));
  // a repair after the hand-over: the trainer's cameras are older than the model
  const repaired = states({ ...sent, sfm: sfm({ keyframes: 35, quality: quality(1), sentVersion: 0 }), trainer: { ...sent.trainer, training: false } });
  assert(repaired.train === 'stale');
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

// The Export step follows the mesh (app.js saveMesh: app.meshState); Train's "Make mesh →" needs trained splats
Deno.test('phases: the mesh drives the Export step', () => {
  const base = { label: '/dataset', video: null, sfm: null, colmap: { cams: 152, fromSfm: false }, mesh: null };
  const at = (step, mesh = null) => phaseStates({ ...base, trainer: { step, training: false, psnr: 20 }, mesh });
  assert(at(0).export.state === 'todo' && !canMesh({ ...base, trainer: { step: 0 } }) && !canMesh({ ...base, trainer: null }));
  assert(canMesh({ ...base, trainer: { step: 10 } }));
  const running = { running: true, step: 5000 };
  assert(at(5000, running).export.state === 'running' && !canMesh({ ...base, trainer: { step: 5000 }, mesh: running }));
  const made = { running: false, step: 5000, faces: 308123, texture: 2048, mb: 8.78 };
  const done = at(5000, made).export;
  assert(done.state === 'done' && done.summary === '308k faces · 2048² · 8.8 MB', JSON.stringify(done));
  assert(at(5000, { ...made, texture: 0 }).export.summary === '308k faces · 8.8 MB');
  // trained on after the mesh, or new splats since: it is out of date
  const stale = at(6000, made).export;
  assert(stale.state === 'stale' && stale.summary === 'mesh @ step 5000', JSON.stringify(stale));
  assert(at(200, made).export.state === 'stale');
  assert(at(5000, { running: false, step: 5000, error: 'x' }).export.state === 'todo');
});

// The footer of each phase (phaseActions): its next move is the primary button; a phase whose work
// is done offers the next phase instead of a re-run
const input = (s) => ({ label: '', video: null, sfm: null, colmap: null, trainer: null, mesh: null, ...s });
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

Deno.test('phases: Train starts, pauses, then leads to the mesh', () => {
  const colmap = { cams: 152, fromSfm: false };
  assert(labels('train', { colmap }) === 'Step ×1 (off) | *▶ Train (off) | Make mesh → (off)', labels('train', { colmap }));
  assert(labels('train', { colmap, trainer: { step: 0, gt: true } }) === 'Step ×1 | *▶ Train | Make mesh → (off)');
  assert(labels('train', { colmap, trainer: { step: 900, gt: true, training: true } }) === 'Step ×1 (off) | *⏸ Pause | Make mesh →');
  assert(labels('train', { colmap, trainer: { step: 900, gt: true } }) === 'Step ×1 | ▶ Resume | *Make mesh →');
  const trainer = { step: 900, gt: true };
  assert(labels('export', { colmap, trainer }) === '*Make mesh (.glb)');
  assert(labels('export', { colmap, trainer, mesh: { running: true, step: 900 } }) === '*Make mesh (.glb) (off)');
  assert(labels('export', { colmap, trainer, mesh: { running: false, step: 900, faces: 1000, mb: 1 } }) === '*Make mesh again');
});
