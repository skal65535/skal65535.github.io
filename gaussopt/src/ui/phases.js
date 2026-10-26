// phases.js — The phases of a project (Project · Video · Keyframes · Cameras · Train · Export; a photo
// set: Project · Photos · Cameras · Train · Export) and
// their states, derived from what is loaded and running, never stored: the stepper of shell.js
// shows them. DOM-free (tests/phases_test.js).
"use strict";

export const PHASES = ['project', 'video', 'keyframes', 'cameras', 'train', 'export'];
const PHASE_LABELS = { project: 'Project', video: 'Video', keyframes: 'Keyframes', cameras: 'Cameras', train: 'Train', export: 'Export' };

// A photo set (DESIGN-sfm §13.3 P5): one Photos phase in place of Video and Keyframes (the photos are
// the keyframes); the stepper hides Keyframes
export const phaseLabel = (phase, { photos = false } = {}) => (photos && phase === 'video' ? 'Photos' : PHASE_LABELS[phase]);
export const phaseShown = (phase, { photos = false } = {}) => !(photos && phase === 'keyframes');
export const STATE_ICONS = { todo: '○', running: '●', done: '✓', stale: '⟳', skipped: '⤼' };

// s: {
//   label:   what is loaded, for the Project step ('' when nothing)
//   video:   { name, projectOnly, photos? } | null  the open video, or the one a .gosfm was made from; photos:
//            the number of photos of an open photo set
//   sfm:     { running: 'keyframes' | 'cameras' | null, busy, keyframes: n | null,
//              quality: { keyframes, registered, problems, version } | null, sentVersion: n | null } | null
//   colmap:  { cams, fromSfm } | null           the trainer's cameras
//   trainer: { step, training, psnr, gt? } | null
//   mesh:    { running, step, faces?, texture?, mb?, error? } | null  the last Make mesh (app.js meshState)
// }
// Returns { [phase]: { state: 'todo' | 'running' | 'done' | 'stale' | 'skipped', summary } }.
export function phaseStates(s) {
  const sfm = s.sfm, q = sfm?.quality ?? null;
  const external = !!s.colmap && !s.colmap.fromSfm;  // cameras from COLMAP (or a checkpoint's dataset)
  const out = {};
  out.project = (s.label || s.video || s.colmap) ? { state: 'done', summary: s.label } : { state: 'todo', summary: '' };

  if (s.video?.photos) out.video = { state: 'done', summary: `${s.video.photos} photos` };
  else if (s.video) out.video = { state: 'done', summary: s.video.projectOnly ? `${s.video.name} (not open)` : s.video.name };
  else out.video = external ? { state: 'skipped', summary: 'cameras given' } : { state: 'todo', summary: '' };

  if (s.video?.photos) out.keyframes = { state: 'skipped', summary: '' };  // the photos are the keyframes
  else if (sfm?.running === 'keyframes') out.keyframes = { state: 'running', summary: '' };
  else if (sfm?.keyframes) out.keyframes = { state: 'done', summary: `${sfm.keyframes}` };
  else out.keyframes = external ? { state: 'skipped', summary: '' } : { state: 'todo', summary: '' };

  if (sfm?.running === 'cameras' || (sfm?.busy && q)) {
    out.cameras = { state: 'running', summary: q ? `${q.registered}/${q.keyframes}` : '' };
  } else if (q) {
    out.cameras = { state: 'done', summary: `${q.registered}/${q.keyframes}${q.problems ? ` ⚠${q.problems}` : ''}` };
  } else if (external) {
    out.cameras = { state: 'done', summary: `${s.colmap.cams} given` };
  } else {
    out.cameras = { state: 'todo', summary: '' };
  }

  const tr = s.trainer;
  const trainSummary = tr && tr.step > 0 ? `step ${tr.step}${tr.psnr > 0 ? ` · ${tr.psnr.toFixed(1)} dB` : ''}` : '';
  if (!s.colmap) out.train = { state: 'todo', summary: tr ? trainSummary : 'needs cameras' };
  else if (tr?.training) out.train = { state: 'running', summary: trainSummary };
  else if (s.colmap.fromSfm && q && sfm.sentVersion !== null && sfm.sentVersion < q.version) {
    out.train = { state: 'stale', summary: 'cameras changed' };
  } else {
    out.train = { state: tr && tr.step > 0 ? 'done' : 'todo', summary: trainSummary || 'ready' };
  }

  const m = s.mesh;
  if (m?.running) out.export = { state: 'running', summary: 'meshing…' };
  else if (m?.faces) {
    out.export = !tr || tr.step !== m.step ?  // trained on, or new splats, since
      { state: 'stale', summary: `mesh @ step ${m.step}` }
      : { state: 'done', summary: `${Math.round(m.faces / 1000)}k faces${m.texture ? ` · ${m.texture}²` : ''} · ${m.mb.toFixed(1)} MB` };
  } else out.export = { state: 'todo', summary: '' };
  return out;
}

// Train's "Make mesh →": trained splats, and no mesh being made
export const canMesh = (s) => !!s.trainer && s.trainer.step > 0 && !s.mesh?.running;

// When a step finishes with `target` next: the phase to show, or null to stay. Only forward: a
// user who already moved on to a later phase stays there.
export function advanceTo(current, target) {
  return PHASES.indexOf(current) < PHASES.indexOf(target) ? target : null;
}

// The phase after `phase` that has something to do: shown, not skipped, and not the Cameras of a
// COLMAP input (nothing to do there); null after Export
export function nextPhase(phase, s, states = phaseStates(s)) {
  const photos = !!s.video?.photos, external = !!s.colmap && !s.colmap.fromSfm;
  for (const p of PHASES.slice(PHASES.indexOf(phase) + 1)) {
    if (!phaseShown(p, { photos }) || states[p].state === 'skipped' || (external && p === 'cameras')) continue;
    return p;
  }
  return null;
}

// A phase that started running ahead of the shown one (a "Run all" going from keyframes to
// cameras to training): the phase to follow it to, or null. Only on the change, so a user who goes
// back to an earlier phase while it runs stays there.
export function followRunning(prev, states, current) {
  for (const p of PHASES.slice(PHASES.indexOf(current) + 1)) {
    if (states[p].state === 'running' && prev?.[p]?.state !== 'running') return p;
  }
  return null;
}

// The buttons pinned under a phase, first to last (the primary one is the step's next move):
// { id, label, enabled, primary?, title? }. shell.js runs them by id: stop, runAll, keyframes,
// cameras, update, send, step, train, mesh, or go:<phase> to show that phase.
// s: phaseStates' input, plus sfm.{ canRun, ready, canUpdate } and trainer.gt (the photos loaded)
export function phaseActions(phase, s, states = phaseStates(s)) {
  const photos = !!s.video?.photos, sfm = s.sfm, tr = s.trainer;
  const next = nextPhase(phase, s, states);
  const go = (primary = true) => next ? [{ id: `go:${next}`, label: `Next: ${phaseLabel(next, { photos })} →`, enabled: true, primary }] : [];
  const done = (p) => states[p].state === 'done';
  if (phase === 'project') return done('project') ? go() : [];
  if (phase === 'train') {
    const trained = !!tr && tr.step > 0;
    return [
      { id: 'step', label: 'Step ×1', enabled: !!tr?.gt && !tr.training, title: 'one training step, then pause' },
      { id: 'train', label: tr?.training ? '⏸ Pause' : trained ? '▶ Resume' : '▶ Train', enabled: !!tr?.gt,
        primary: !trained || tr.training, title: 'start / pause training (the step count and splats are kept)' },
      { id: 'mesh', label: 'Make mesh →', enabled: canMesh(s), primary: trained && !tr.training,
        title: 'a textured mesh (.glb) of the splats, in the Export step' },
    ];
  }
  if (phase === 'export') {
    return [{ id: 'mesh', label: s.mesh?.faces ? 'Make mesh again' : 'Make mesh (.glb)', enabled: canMesh(s), primary: true,
              title: 'a triangle mesh (.glb) from the rendered depth of every camera, simplified, textured from the photos (Mesh > texture); also shown in the 3D view (⚙ view > Mesh)' }];
  }
  // video, keyframes, cameras: the SfM's steps (nothing to run without a video or photos)
  if (!sfm) return done('project') ? go() : [];
  if (sfm.running || sfm.busy) return [{ id: 'stop', label: '■ Stop', enabled: !!sfm.running, title: 'interrupts the video work (the session is lost)' }];
  const canRun = !!sfm.canRun;
  if (phase === 'video') {
    const ran = photos ? done('cameras') : done('keyframes');
    const find = photos ? { id: 'cameras', label: 'Find cameras' } : { id: 'keyframes', label: 'Find keyframes' };
    if (ran) return [{ ...find, label: `${find.label} again`, enabled: canRun, title: 'run it again: the cameras found are replaced (the stages whose options did not change are reused)' }, ...go()];
    return [
      { id: 'runAll', label: 'Run all, then train', enabled: canRun, title: photos ? 'cameras, then training' : 'keyframes, cameras, then training' },
      { ...find, label: `${find.label} →`, enabled: canRun, primary: true },
    ];
  }
  if (phase === 'keyframes') {
    if (done('cameras')) return [{ id: 'cameras', label: 'Find cameras again', enabled: canRun, title: 'run it again: the cameras found are replaced (the stages whose options did not change are reused)' }, ...go()];
    return [{ id: 'cameras', label: 'Find cameras →', enabled: canRun, primary: true }];
  }
  // cameras: the trainer has these cameras (its splats are from them) => on to Train
  const update = sfm.canUpdate ? [{ id: 'update', label: 'Update training cameras', enabled: true,
                                    title: 'the repaired cameras into the running training, splats kept' }] : [];
  const sent = !!s.colmap?.fromSfm && !!sfm.quality && sfm.sentVersion === sfm.quality.version;
  if (sent) return [{ id: 'send', label: 'Restart training', enabled: !!sfm.ready, title: 'new splats from these cameras and points' }, ...go()];
  return [...update, { id: 'send', label: 'Train on these cameras →', enabled: !!sfm.ready, primary: true,
                       title: 'new splats from these cameras and points' }];
}
