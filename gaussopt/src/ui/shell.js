// shell.js — The page layout (index.html, gaussopt.css): the phase stepper (each step's state, what it waits
// for, its progress, when it was saved), the inspector that says what the current phase is and why it is in its state,
// shows its controls with its next-step buttons pinned at the bottom, and a notice when a phase is saved, the dock tabs
// (timeline / charts / log), the view toolbar and help, and the 3D canvas fitted to the view.
// refresh() re-derives the phase states (phases.js) from the app; the DOM changes only when they do.
"use strict";

import { PHASES, STATE_ICONS, phaseStates, advanceTo, phaseLabel, phaseShown, phaseActions, followRunning, phaseWhat, phaseNote, savedLine,
         savedNotice } from './phases.js';

const DOCK_TAB = { project: 'log', video: 'timeline', keyframes: 'timeline', cameras: 'timeline', dome: 'log', train: 'charts', mesh: 'log', export: 'log' };
const TOAST_MS = 8000;
const hhmm = (t) => new Date(t).toTimeString().slice(0, 5);
// a photo set has no timeline (DESIGN-sfm §13.3 P5): its contact sheet is the view of the Photos phase
const DOCK_TAB_PHOTOS = { ...DOCK_TAB, video: 'log', keyframes: 'log', cameras: 'log' };
const DOCK_FOLDED_KEY = 'gaussopt.dockFolded';  // this browser's choice
const CANVAS_LONG_SIDE = 960; // backing store: the view's shape, this on its long side

export class Shell {
  constructor(app) {
    this.app = app;
    const $ = (id) => document.getElementById(id);
    this.stepper = $('go-stepper');
    this.title = $('phase-title');
    this.summary = $('phase-summary');
    this.foot = $('phase-foot');
    this.label = $('project-label');
    this.chip = $('phase-chip');
    this.position = $('phase-position');
    this.what = $('phase-what');
    this.note = $('phase-note');
    this.savedEl = $('phase-saved');
    this.toastEl = $('phase-toast');
    this.sections = Object.fromEntries([...document.querySelectorAll('#phase-body > section')].map(s => [s.dataset.phase, s]));
    this.steps = {};
    PHASES.forEach((p) => {
      const b = document.createElement('button');
      b.className = 'step todo';
      b.dataset.phase = p;
      b.innerHTML = `<span class="step-icon"></span><span class="step-label">${phaseLabel(p)}</span><span class="step-chip"></span>` +
                    '<span class="step-sum"></span><span class="step-saved"></span><span class="step-bar"><span></span></span>';
      b.addEventListener('click', () => this.show(p));
      this.stepper.append(b);
      this.steps[p] = b;
    });
    this.tabs = [...document.querySelectorAll('.dock-tabs button')];
    // a click on the open tab folds the dock to its tabs (the view gets the height), any tab unfolds it
    this.main = $('main-area');
    for (const t of this.tabs) t.addEventListener('click', () => {
      const fold = t.classList.contains('on') && !this.main.classList.contains('dock-folded');
      this.main.classList.toggle('dock-folded', fold);
      try { localStorage.setItem(DOCK_FOLDED_KEY, fold ? '1' : ''); } catch { /* private window: not remembered */ }
      this.setDock(t.dataset.tab);
    });
    try { this.main.classList.toggle('dock-folded', !!localStorage.getItem(DOCK_FOLDED_KEY)); } catch { /* idem */ }
    const help = $('help-overlay'), pop = $('view-popover'), viewBtn = $('BtnView');
    $('BtnHelp').addEventListener('click', () => help.classList.toggle('shown'));
    viewBtn.addEventListener('click', () => viewBtn.classList.toggle('on', pop.classList.toggle('shown')));
    this.viewHost = pop;
    this.statesKey = '';
    this.footKey = '';
    this.current = null;
    this.states = null;  // the last phaseStates, to follow a run that starts ahead (followRunning)
    this._fitCanvas();
    new ResizeObserver(() => this._fitCanvas()).observe($('go-view'));
    this.show('project');
  }

  // The host element of a phase's controls (index.html <section data-phase>)
  section(phase) { return this.sections[phase]; }

  show(phase) {
    if (this.current === phase) return;
    this.current = phase;
    for (const [p, s] of Object.entries(this.sections)) s.classList.toggle('shown', p === phase);
    for (const [p, b] of Object.entries(this.steps)) b.classList.toggle('current', p === phase);
    const photos = this.photos;
    this.title.textContent = phaseLabel(phase, { photos });
    this.setDock((photos ? DOCK_TAB_PHOTOS : DOCK_TAB)[phase]);
    // the keyframes are what the Keyframes phase is about (a photo set: the Photos phase); Train, Mesh
    // and Export are about the 3D view
    const vs = this.app.viewSwitch;
    if ((phase === 'keyframes' || (photos && phase === 'video')) && this.app.frames.order.length) vs.set('frames');
    if (['train', 'mesh', 'export'].includes(phase) && vs.mode === 'frames') vs.set('3d');
    this.statesKey = this.footKey = '';
    this.refresh();
  }

  // A step finished and `phase` comes next (phases.js advanceTo)
  advance(phase) {
    const to = advanceTo(this.current, phase);
    if (to) this.show(to);
  }

  setDock(tab) {
    for (const t of this.tabs) t.classList.toggle('on', t.dataset.tab === tab);
    for (const p of document.querySelectorAll('.dock-pane')) p.classList.toggle('shown', p.dataset.tab === tab);
  }

  // An open photo set (DESIGN-sfm §13.3 P5): the phases are labelled and shown for it
  get photos() { return !!this.app.sfm?.photos; }

  // Stepper states, the inspector's summary and buttons, the project label
  refresh() {
    const input = this._input(), states = phaseStates(input), photos = this.photos;
    const follow = this.states && followRunning(this.states, states, this.current);
    this.states = states;
    if (follow) return this.show(follow);
    const key = JSON.stringify([states, photos, input.saved, this.current]);
    if (key !== this.statesKey) {
      this.statesKey = key;
      let n = 0, k = 0;  // the shown steps are numbered
      for (const p of PHASES) {
        const b = this.steps[p], { state, chip, summary, progress } = states[p], label = phaseLabel(p, { photos });
        const shown = phaseShown(p, { photos }), saved = input.saved[p];  // style, not `hidden`: .step sets its display
        if (shown) n++;
        if (p === this.current) k = n;
        b.className = `step ${state}${p === this.current ? ' current' : ''}`;
        b.style.display = shown ? '' : 'none';
        b.querySelector('.step-label').textContent = label;
        b.querySelector('.step-icon').textContent = STATE_ICONS[state] ?? (state === 'locked' ? '' : n);
        b.querySelector('.step-chip').textContent = chip;
        b.querySelector('.step-chip').className = `step-chip ${state}`;
        // the Project's summary is the header's project label
        b.querySelector('.step-sum').textContent = p === 'project' ? '' : summary;
        b.querySelector('.step-saved').textContent = saved ? `saved ${hhmm(saved.at)}` : '';
        b.querySelector('.step-bar').style.display = progress == null ? 'none' : '';
        b.querySelector('.step-bar > span').style.width = `${Math.round(100 * (progress ?? 0))}%`;
        b.title = `${label}: ${chip}${summary ? ` (${summary})` : ''}${saved ? `; ${savedLine(p, saved, hhmm(saved.at))}` : ''}`;
        b.setAttribute('aria-label', `${label}: ${chip}`);
      }
      this.title.textContent = phaseLabel(this.current, { photos });
      for (const t of this.tabs) if (t.dataset.tab === 'timeline') t.style.display = photos ? 'none' : '';
      // the phase may have been shown before the photo set was known: off the timeline then
      if (photos && this.tabs.some(t => t.dataset.tab === 'timeline' && t.classList.contains('on'))) this.setDock(DOCK_TAB_PHOTOS[this.current]);
      const cur = states[this.current], saved = input.saved[this.current];
      this.summary.textContent = cur.summary;
      this.chip.textContent = cur.chip;
      this.chip.className = `step-chip ${cur.state}`;
      this.position.textContent = `step ${k} of ${n}`;
      this.what.textContent = phaseWhat(this.current, { photos });
      const note = phaseNote(this.current, input, states);
      this.note.textContent = note;
      this.note.className = cur.state;
      this.note.hidden = !note;
      this.savedEl.textContent = saved ? savedLine(this.current, saved, hhmm(saved.at)) : '';
      this.savedEl.hidden = !saved;
      this.label.textContent = states.project.summary;
    }
    const actions = phaseActions(this.current, input, states);
    const footKey = JSON.stringify(actions);
    if (footKey === this.footKey) return;
    this.footKey = footKey;
    this.foot.replaceChildren(...actions.map(a => {
      const b = document.createElement('button');
      b.textContent = a.label;
      b.title = a.title ?? '';
      b.disabled = !a.enabled;
      if (a.primary) b.className = 'primary';
      b.addEventListener('click', () => {
        const sfm = this.app.sfm;  // in the log: which button, in what state (a click that starts nothing shows there)
        this.app.log(`button: ${a.label} (${this.current}${sfm ? `; video: ${sfm.running ? 'running' : sfm.busy ? 'busy' : 'idle'}, ${sfm.state.status}` : ''})`);
        this._run(a.id); this.footKey = ''; this.refresh();
      });
      return b;
    }));
  }

  // A phase saved (app.js stageSaved): its notice over the inspector for a while
  toast(phase, saved) {
    const el = this.toastEl;
    el.textContent = savedNotice(phase, saved);
    el.hidden = false;
    clearTimeout(this.toastTimer);
    this.toastTimer = setTimeout(() => { el.hidden = true; }, TOAST_MS);
    this.refresh();
  }

  _input() {
    const app = this.app, sfm = app.sfm, tr = app.trainer, colmap = app.colmap;
    const q = sfm?.quality;
    return {
      label: app.projectLabel(),
      video: sfm?.videoName ? { name: sfm.videoName, projectOnly: !sfm.file && !sfm.photos, ...(sfm.photos ? { photos: sfm.videoInfo?.photos ?? sfm.photos.length } : {}) } : null,
      sfm: sfm ? {
        running: sfm.running ? (sfm.keyframeCount === null ? 'keyframes' : 'cameras') : null,
        progress: sfm.running ? sfm.progress : null,
        busy: sfm.busy,
        keyframes: sfm.keyframeCount,
        quality: q ? { keyframes: q.frames.length, registered: q.frames.filter(f => f.registered).length,
                       problems: q.counts.amber + q.counts.red, version: q.version } : null,
        sentVersion: sfm.sent?.version ?? null,
        canRun: !!(sfm.file || sfm.photos), ready: sfm.ready, canUpdate: sfm.canUpdate,
      } : null,
      colmap: colmap ? { cams: colmap.images.length, fromSfm: !!colmap.fromSfm } : null,
      trainer: tr ? { step: tr.stepCount, training: app.params.training, psnr: tr.lastPsnr, gt: !!app.gtAvailable, steps: app.params.train_steps } : null,
      mesh: app.meshState,
      dome: app.domeStep.state,
      changed: { ...sfm?.changedOptions(), ...(app.meshOptionsChanged() ? { mesh: app.meshOptionsChanged() } : {}) },
      saved: app.saved,
    };
  }

  // A footer button (phases.js phaseActions ids)
  _run(id) {
    const app = this.app, sfm = app.sfm;
    if (id.startsWith('go:')) return this.show(id.slice(3));
    if (id === 'revert:mesh') return app.revertMeshOptions();
    if (id.startsWith('revert:')) return sfm.revertOptions(id.slice(7));
    switch (id) {
      case 'stop': return sfm.stop();
      case 'runAll': return sfm.run(null, true);
      case 'keyframes': return sfm.run('keyframes');
      case 'cameras': return sfm.run();
      case 'update': return sfm.requestSend('update');
      case 'send': return sfm.requestSend('restart');
      case 'train': return app.setTraining(!app.params.training);
      case 'step': return app.stepOnce();
      case 'mesh': this.show('mesh'); return app.saveMesh();
      case 'scene': return app.saveSceneGlb();
      case 'dome': return app.domeStep.build();
      case 'domeOff': return app.domeStep.set(null);
    }
  }

  // The photos' aspect: the stacked layout's view takes it (a portrait video: a portrait view)
  shapeCanvas({ width, height }) {
    this.aspect = width / height;
    this._fitCanvas();
  }

  // The canvas fills the view (the 3D view has no bars; the 2D compare modes letterbox the photo
  // inside it, render.js), its backing store CANVAS_LONG_SIDE on the long side
  _fitCanvas() {
    const view = document.getElementById('go-view'), canvas = this.app.canvas;
    const narrow = window.matchMedia('(max-width: 1000px)').matches;
    // narrow (stacked) layout: the view's height follows the canvas, the photos' shape, capped
    const w = view.clientWidth, h = narrow ? Math.min(w / (this.aspect ?? 16 / 9), 0.8 * window.innerHeight) : view.clientHeight;
    if (!(w > 0 && h > 0)) return;
    const s = CANVAS_LONG_SIDE / Math.max(w, h), bw = Math.round(w * s), bh = Math.round(h * s);
    if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
    canvas.style.width = `${Math.floor(w)}px`;
    canvas.style.height = `${Math.floor(h)}px`;
  }
}
