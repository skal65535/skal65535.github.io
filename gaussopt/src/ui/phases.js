// phases.js — The phases of a project (Project · Video · Keyframes · Cameras · Dome · Train · Mesh · Export; a
// photo set: Project · Photos · Cameras · Dome · Train · Mesh · Export) and their states, derived from what is
// loaded and running, never stored: the stepper of shell.js shows them. DOM-free (tests/phases_test.js).
"use strict";

export const PHASES = ['project', 'video', 'keyframes', 'cameras', 'dome', 'train', 'mesh', 'export'];
const PHASE_LABELS = { project: 'Project', video: 'Video', keyframes: 'Keyframes', cameras: 'Cameras', dome: 'Dome', train: 'Train', mesh: 'Mesh', export: 'Export' };

// A photo set (DESIGN-sfm §13.3 P5): one Photos phase in place of Video and Keyframes (the photos are
// the keyframes); the stepper hides Keyframes
export const phaseLabel = (phase, { photos = false } = {}) => (photos && phase === 'video' ? 'Photos' : PHASE_LABELS[phase]);
export const phaseShown = (phase, { photos = false } = {}) => !(photos && phase === 'keyframes');
// on the step's circle (the others: its number)
export const STATE_ICONS = { running: '●', done: '✓', stale: '⟳', changed: '✎', skipped: '⤼' };

// What each phase is, the inspector's first sentence
const WHAT = {
  project: 'Where the work is kept: the video, the keyframes, the cameras and the splats, saved as they are made.',
  video: 'The video the scene is made from.',
  keyframes: 'Sharp frames picked along the video, their SIFT features, and the points tracked between them.',
  cameras: 'Where each keyframe was shot (the SfM) and the sparse points; the repairs are made here.',
  dome: 'A sky dome behind the splats, for scenes with much sky.',
  train: 'The splats fitted to the photos, from the cameras and their points.',
  mesh: "A textured triangle mesh from the splats' rendered depth (optional).",
  export: 'Files out: the scene (.glb), the splats (.ply), the photos (.glb), a snapshot to go on training from.',
};
export const phaseWhat = (phase, { photos = false } = {}) => (photos && phase === 'video' ? 'The photos the scene is made from.' : WHAT[phase]);

// A phase waits for the one before it: its first state is "Needs <that one>" (locked) until it is done. Cameras: the
// video's (Find cameras finds the keyframes first when they are not)
const NEEDS = { keyframes: 'video', cameras: 'video', dome: 'cameras', train: 'cameras', mesh: 'train', export: 'train' };
const OPTIONAL = new Set(['dome', 'mesh']);  // never the next step: skipping them is fine
const CHIPS = { done: 'Done', running: 'Running', next: 'Next', ready: 'Ready', stale: 'Out of date', changed: 'Options changed',
                optional: 'Optional', skipped: 'Not needed' };

// s: {
//   label:   what is loaded, for the Project step ('' when nothing)
//   video:   { name, projectOnly, photos? } | null  the open video, or the one a .gosfm was made from; photos:
//            the number of photos of an open photo set
//   sfm:     { running: 'keyframes' | 'cameras' | null, busy, keyframes: n | null,
//              quality: { keyframes, registered, problems, version } | null, sentVersion: n | null } | null
//   colmap:  { cams, fromSfm } | null           the trainer's cameras
//   trainer: { step, training, psnr, gt? } | null
//   dome:    { mode: 'auto' | 'off', running, result: { worth, farShare, fill (its name, null: a scene's) } | null } | null  the Dome step (ui/dome_step.js state)
//   mesh:    { running, step, faces?, texture?, mb?, error? } | null  the last Make mesh (app.js meshState)
//   changed: { keyframes?, cameras?, mesh? }   a done step's options changed since its run: what changed ('spacing 0.05 → 0.04')
//   saved:   { [phase]: { at, file, where: 'project' | 'browser' | 'server' | 'download', step? } }  what was saved last (app.js saved)
// }
// sfm.progress: { done, total } of the running stage; trainer.steps: the planned length (0: open-ended)
// Returns { [phase]: { state, chip, summary, progress?, needs? } }, state one of
//   done, running, stale (made from something that changed since), changed (its options changed since it ran),
//   next (the step to do now), ready (can be done), locked (needs: the phase it waits for), optional, skipped.
export function phaseStates(s) {
  const out = rawStates(s), trained = !!s.trainer && s.trainer.step > 0;
  const met = (p) => p === 'train' ? trained : p === 'cameras' ? !!s.colmap : ['done', 'skipped', 'stale'].includes(out[p].state);
  for (const p of ['keyframes', 'cameras', 'mesh']) if (s.changed?.[p] && out[p].state === 'done') out[p] = { state: 'changed', summary: s.changed[p] };
  const running = PHASES.some(p => out[p].state === 'running');
  let next = running;  // a run going: nothing is next but what follows it
  for (const p of PHASES) {
    const o = out[p];
    if (o.state === 'skipped' && OPTIONAL.has(p)) { o.state = 'optional'; o.chip = 'Optional · off'; }
    if (o.state !== 'todo') continue;
    const need = NEEDS[p];
    if (OPTIONAL.has(p) && (p === 'mesh' || !met(need))) o.state = 'optional';
    else if (need && !met(need)) Object.assign(o, { state: 'locked', needs: need, summary: '' });
    else { o.state = next ? 'ready' : 'next'; next = true; }
  }
  for (const p of PHASES) {
    const o = out[p];
    o.chip ??= o.state === 'locked' ? `Needs ${phaseLabel(o.needs, { photos: !!s.video?.photos }).toLowerCase()}` : CHIPS[o.state];
  }
  if (out.train.state === 'done' && trained) out.train.chip = 'Paused';
  if (out.export.state === 'done') out.export.chip = 'Saved';
  const frac = (d, t) => (t > 0 ? Math.min(1, d / t) : null);
  const pr = s.sfm?.progress;
  for (const p of ['keyframes', 'cameras']) if (out[p].state === 'running' && pr) out[p].progress = frac(pr.done, pr.total);
  if (out.train.state === 'running') out.train.progress = frac(s.trainer.step, s.trainer.steps ?? 0);
  return out;
}

function rawStates(s) {
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

  const d = s.dome, pct = (x) => `${(100 * x).toFixed(x < 0.1 ? 1 : 0)}%`;
  if (d?.running) out.dome = { state: 'running', summary: 'finding the sky…' };
  else if (d?.result) {
    const f = d.result.farShare;  // null: a scene's dome
    out.dome = { state: 'done', summary: f === null ? 'from the scene' : d.result.worth ? `sky ${pct(f)}${d.result.fill ? `, ${d.result.fill}` : ''}` : `no dome (sky ${pct(f)})` };
  }
  else if (!d || d.mode === 'off') out.dome = { state: 'skipped', summary: 'off' };  // the default (params.js dome)
  else if (!s.colmap) out.dome = { state: 'todo', summary: 'needs cameras' };
  else out.dome = { state: 'todo', summary: 'before training' };

  const tr = s.trainer;
  const trainSummary = tr && tr.step > 0 ? `step ${tr.step}${tr.psnr > 0 ? ` · ${tr.psnr.toFixed(1)} dB` : ''}` : '';
  if (!s.colmap) out.train = { state: 'todo', summary: tr ? trainSummary : 'needs cameras' };
  else if (tr?.training) out.train = { state: 'running', summary: trainSummary };
  else if (s.colmap.fromSfm && q && sfm.sentVersion !== null && sfm.sentVersion < q.version) {
    out.train = { state: 'stale', summary: 'cameras changed' };
  } else {
    out.train = { state: tr && tr.step > 0 ? 'done' : 'todo', summary: trainSummary || 'ready' };
  }

  const m = s.mesh, trained = !!tr && tr.step > 0;
  if (m?.running) out.mesh = { state: 'running', summary: 'meshing…' };
  else if (m?.faces) {
    out.mesh = !tr || tr.step !== m.step ?  // trained on, or new splats, since
      { state: 'stale', summary: `mesh @ step ${m.step}` }
      : { state: 'done', summary: `${Math.round(m.faces / 1000)}k faces${m.texture ? ` · ${m.texture}²` : ''} · ${m.mb.toFixed(1)} MB` };
  } else out.mesh = { state: 'todo', summary: trained ? '' : 'after training' };

  // Export: the last scene or splats saved, of these splats or older ones
  const ex = s.saved?.export;
  if (!trained || !ex) out.export = { state: 'todo', summary: trained ? 'nothing saved yet' : '' };
  else out.export = ex.step === tr.step ? { state: 'done', summary: `${ex.file} @ ${ex.step}` } : { state: 'stale', summary: `${ex.file} @ ${ex.step}` };
  return out;
}

// The inspector's note on a phase in its state ('' none): why it is out of date, what an option change does, how to
// redo a step with other options
export function phaseNote(phase, s, states = phaseStates(s)) {
  const o = states[phase], photos = !!s.video?.photos, L = (p) => phaseLabel(p, { photos });
  const tr = s.trainer;
  if (o.state === 'locked') return `${L(o.needs)} first: this step is made from it.`;
  if (o.state === 'changed') return `An option changed since the last run (${o.summary}): what is shown is still that run's. Run it again to apply it, ` +
                                    'or revert the options; nothing after it changes until then.';
  if (o.state === 'stale') {
    if (phase === 'train') return 'Trained on the cameras from before their last repair: Update training cameras (in Cameras) keeps the splats.';
    if (phase === 'mesh') return tr && s.mesh && tr.step > s.mesh.step ? `Trained ${tr.step - s.mesh.step} steps further since this mesh was made.` : 'Made from other splats.';
    if (phase === 'export') return `The last save is of step ${s.saved.export.step}; the splats are at step ${tr.step}.`;
  }
  if (o.state === 'done' && ['keyframes', 'cameras', 'dome', 'mesh'].includes(phase))
    return 'To redo it with other options: change them below (the step shows ✎ Options changed), then run it again; until then this result stays.';
  if (o.state === 'optional') return phase === 'dome' && o.chip === 'Optional · off' ? 'Off: set dome to auto below to find one.' : 'Optional: skipping it is fine.';
  return '';
}

// What a save of a phase makes recoverable, the stepper's notice and the inspector's line
const RECOVERS = {
  keyframes: { project: 'a reload or a crash goes on from here, not from the video', browser: 'reopening the same video goes on from here' },
  cameras: { project: 'a reload goes on from these cameras and their repairs', browser: 'reopening the same video goes on from these cameras' },
  train: { project: 'training resumes from it after a reload or a crash', server: 'Start from lists it after a reload' },
};
const WHERE = { project: 'in the project', browser: 'in this browser', server: 'by serve.py', download: 'as a download' };
export function savedLine(phase, saved, time) {
  const r = RECOVERS[phase]?.[saved.where];
  return `Saved ${time} ${WHERE[saved.where] ?? ''} (${saved.file})${r ? `: ${r}` : ''}.`;
}
export function savedNotice(phase, saved) {
  const what = phase === 'train' ? `Snapshot @ step ${saved.step}` : phase === 'export' ? `${saved.file}` : `${phaseLabel(phase)}`;
  const r = RECOVERS[phase]?.[saved.where];
  return `${what} saved ${WHERE[saved.where] ?? ''}${phase === 'export' ? '' : ` (${saved.file})`}${r ? `: ${r}` : ''}.`;
}

// What changed between the options a step ran with (`ran`, null: no run known) and `now`, over `keys` ('' nothing):
// 'spacing 0.05 → 0.04', two at most then '+n'. A key `ran` lacks (an older project's record) is not compared.
export function optionsChanged(ran, now, keys) {
  if (!ran) return '';
  const f = (v) => (typeof v === 'number' ? +v.toPrecision(4) : v);
  const diff = keys.filter(k => k in ran && ran[k] !== now[k]).map(k => `${k} ${f(ran[k])} → ${f(now[k])}`);
  return diff.length > 2 ? `${diff.slice(0, 2).join(', ')} +${diff.length - 2}` : diff.join(', ');
}

// Train's "Make mesh →": trained splats, and no mesh being made
export const canMesh = (s) => !!s.trainer && s.trainer.step > 0 && !s.mesh?.running;

// When a step finishes with `target` next: the phase to show, or null to stay. Only forward: a
// user who already moved on to a later phase stays there.
export function advanceTo(current, target) {
  return PHASES.indexOf(current) < PHASES.indexOf(target) ? target : null;
}

// The phase after `phase` that has something to do: shown, not skipped nor optional, and not the Cameras of a
// COLMAP input (nothing to do there); null after Export
export function nextPhase(phase, s, states = phaseStates(s)) {
  const photos = !!s.video?.photos, external = !!s.colmap && !s.colmap.fromSfm;
  for (const p of PHASES.slice(PHASES.indexOf(phase) + 1)) {
    if (!phaseShown(p, { photos }) || ['skipped', 'optional'].includes(states[p].state) || (external && p === 'cameras')) continue;
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
// cameras, update, send, step, train, mesh, scene, dome, domeOff, revert:<phase> (its last run's options back), or
// go:<phase> to show that phase.
// s: phaseStates' input, plus sfm.{ canRun, ready, canUpdate } and trainer.gt (the photos loaded)
export function phaseActions(phase, s, states = phaseStates(s)) {
  const o = states[phase], acts = stepActions(phase, s, states);
  // locked: to the step it waits for first; options changed: run it again, or back to the last run's options
  if (o.state === 'locked') return [{ id: `go:${o.needs}`, label: `Go to ${phaseLabel(o.needs, { photos: !!s.video?.photos })}`, enabled: true, primary: true },
                                    ...acts.map(({ primary, ...a }) => a)];
  if (o.state === 'changed') {
    const again = { keyframes: { id: 'keyframes', label: 'Find keyframes again' }, cameras: { id: 'cameras', label: 'Find cameras again' },
                    mesh: { id: 'mesh', label: 'Make mesh again' } }[phase];
    const can = phase === 'mesh' ? canMesh(s) : !!s.sfm?.canRun && !s.sfm.running && !s.sfm.busy;
    return [{ id: `revert:${phase}`, label: 'Revert the options', enabled: true, title: "back to the last run's options: nothing to run again" },
            { ...again, enabled: can, primary: true, title: 'with the options as they are now' }];
  }
  return acts;
}

function stepActions(phase, s, states) {
  const photos = !!s.video?.photos, sfm = s.sfm, tr = s.trainer;
  const next = nextPhase(phase, s, states);
  const go = (primary = true) => next ? [{ id: `go:${next}`, label: `Next: ${phaseLabel(next, { photos })} →`, enabled: true, primary }] : [];
  const done = (p) => states[p].state === 'done';
  if (phase === 'project') return done('project') ? go() : [];
  if (phase === 'dome') {
    const d = s.dome, ready = !!tr?.gt && !d?.running && !tr?.training;
    return [
      ...(d?.result?.worth ? [{ id: 'domeOff', label: 'No dome', enabled: ready, title: 'train without it (the learned background of bg_lr, if any)' }] : []),
      { id: 'dome', label: d?.result ? 'Find it again' : 'Find the sky', enabled: ready, primary: !d?.result,
        title: 'the directions where the photos agree (no parallax): a dome behind the splats when they cover enough of the photos' },
      ...go(!!d?.result || d?.mode === 'off'),
    ];
  }
  if (phase === 'train') {
    const trained = !!tr && tr.step > 0;
    return [
      { id: 'step', label: 'Step ×1', enabled: !!tr?.gt && !tr.training, title: 'one training step, then pause' },
      { id: 'train', label: tr?.training ? '⏸ Pause' : trained ? '▶ Resume' : '▶ Train', enabled: !!tr?.gt,
        primary: !trained || tr.training, title: 'start / pause training (the step count and splats are kept)' },
      { id: 'mesh', label: 'Make mesh →', enabled: canMesh(s), title: 'a textured mesh of the splats (.mesh.glb), then the scene with it, in the Mesh step' },
      ...(trained ? [{ id: 'go:export', label: 'Next: Export →', enabled: true, primary: !tr.training }] : []),
    ];
  }
  if (phase === 'mesh') {
    return [{ id: 'mesh', label: s.mesh?.faces ? 'Make mesh again' : 'Make mesh + scene (.glb)', enabled: canMesh(s), primary: !s.mesh?.faces,
              title: 'a triangle mesh (.mesh.glb) from the rendered depth of every camera, simplified, textured from the photos (Mesh > texture), ' +
                     'then the scene with it (.scene.glb: the splats, the cameras, the mesh; as Save scene); also shown in the 3D view (⚙ view > Mesh)' },
            ...(s.mesh?.faces ? go() : [])];
  }
  if (phase === 'export') {
    return [{ id: 'scene', label: states.export.state === 'done' ? 'Save scene again' : 'Save scene (.glb)', enabled: !!tr && tr.step > 0, primary: states.export.state !== 'done',
              title: 'the splats, the cameras, the mesh on show, the background, the leveling and the settings, in one .glb' }];
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
  // cameras: none found yet (keyframes only, or nothing run) => find them here too
  if (!sfm.quality) return [{ id: 'cameras', label: 'Find cameras →', enabled: canRun, primary: true }];
  // the trainer has these cameras (its splats are from them) => on to Train
  const update = sfm.canUpdate ? [{ id: 'update', label: 'Update training cameras', enabled: true,
                                    title: 'the repaired cameras into the running training, splats kept' }] : [];
  const sent = !!s.colmap?.fromSfm && !!sfm.quality && sfm.sentVersion === sfm.quality.version;
  if (sent) return [{ id: 'send', label: 'Restart training', enabled: !!sfm.ready, title: 'new splats from these cameras and points' }, ...go()];
  return [...update, { id: 'send', label: 'Train on these cameras →', enabled: !!sfm.ready, primary: true,
                       title: 'new splats from these cameras and points' }];
}
