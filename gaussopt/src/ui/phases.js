// phases.js — The phases of a project (Project · Video · Keyframes · Cameras · Train · Export) and
// their states, derived from what is loaded and running, never stored: the stepper of shell.js
// shows them. DOM-free (tests/phases_test.js).
"use strict";

export const PHASES = ['project', 'video', 'keyframes', 'cameras', 'train', 'export'];
export const PHASE_LABELS = { project: 'Project', video: 'Video', keyframes: 'Keyframes', cameras: 'Cameras', train: 'Train', export: 'Export' };
export const STATE_ICONS = { todo: '○', running: '●', done: '✓', stale: '⟳', skipped: '⤼' };

// s: {
//   label:   what is loaded, for the Project step ('' when nothing)
//   video:   { name, projectOnly } | null      the open video, or the one a .gosfm was made from
//   sfm:     { running: 'keyframes' | 'cameras' | null, busy, keyframes: n | null,
//              quality: { keyframes, registered, problems, version } | null, sentVersion: n | null } | null
//   colmap:  { cams, fromSfm } | null           the trainer's cameras
//   trainer: { step, training, psnr } | null
// }
// Returns { [phase]: { state: 'todo' | 'running' | 'done' | 'stale' | 'skipped', summary } }.
export function phaseStates(s) {
  const sfm = s.sfm, q = sfm?.quality ?? null;
  const external = !!s.colmap && !s.colmap.fromSfm;  // cameras from COLMAP (or a checkpoint's dataset)
  const out = {};
  out.project = (s.label || s.video || s.colmap) ? { state: 'done', summary: s.label } : { state: 'todo', summary: '' };

  if (s.video) out.video = { state: 'done', summary: s.video.projectOnly ? `${s.video.name} (not open)` : s.video.name };
  else out.video = external ? { state: 'skipped', summary: 'COLMAP input' } : { state: 'todo', summary: '' };

  if (sfm?.running === 'keyframes') out.keyframes = { state: 'running', summary: '' };
  else if (sfm?.keyframes) out.keyframes = { state: 'done', summary: `${sfm.keyframes}` };
  else out.keyframes = external ? { state: 'skipped', summary: '' } : { state: 'todo', summary: '' };

  if (sfm?.running === 'cameras' || (sfm?.busy && q)) {
    out.cameras = { state: 'running', summary: q ? `${q.registered}/${q.keyframes}` : '' };
  } else if (q) {
    out.cameras = { state: 'done', summary: `${q.registered}/${q.keyframes}${q.problems ? ` ⚠${q.problems}` : ''}` };
  } else if (external) {
    out.cameras = { state: 'done', summary: `${s.colmap.cams} (COLMAP)` };
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

  out.export = { state: 'todo', summary: '' };
  return out;
}

// When a step finishes with `target` next: the phase to show, or null to stay. Only forward: a
// user who already moved on to a later phase stays there.
export function advanceTo(current, target) {
  return PHASES.indexOf(current) < PHASES.indexOf(target) ? target : null;
}
