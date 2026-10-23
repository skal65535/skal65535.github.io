// Phase states of the stepper (src/ui/phases.js), derived from what is loaded and running.
import { phaseStates, advanceTo } from '../src/ui/phases.js';
import { assert } from './util.js';

const states = (s) => Object.fromEntries(Object.entries(phaseStates({ label: '', video: null, sfm: null, colmap: null, trainer: null, ...s }))
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
