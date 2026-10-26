// shell.js — The page layout (index.html, gaussopt.css): the phase stepper, the inspector that shows
// the current phase's controls with its next-step buttons pinned at the bottom, the dock tabs
// (timeline / charts / log), the view toolbar and help, and the 3D canvas fitted to the view.
// refresh() re-derives the phase states (phases.js) from the app; the DOM changes only when they do.
"use strict";

import { PHASES, STATE_ICONS, phaseStates, advanceTo, phaseLabel, phaseShown, phaseActions, followRunning } from './phases.js';

const DOCK_TAB = { project: 'log', video: 'timeline', keyframes: 'timeline', cameras: 'timeline', train: 'charts', export: 'log' };
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
    this.sections = Object.fromEntries([...document.querySelectorAll('#phase-body > section')].map(s => [s.dataset.phase, s]));
    this.steps = {};
    PHASES.forEach((p) => {
      const b = document.createElement('button');
      b.className = 'step todo';
      b.dataset.phase = p;
      b.innerHTML = `<span class="step-icon"></span><span class="step-label">${phaseLabel(p)}</span><span class="step-sum"></span>`;
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
    // the keyframes are what the Keyframes phase is about (a photo set: the Photos phase); Train and
    // Export are about the 3D view
    if ((phase === 'keyframes' || (photos && phase === 'video')) && this.app.frames.order.length) this.app.frames.toggle(true);
    if (phase === 'train' || phase === 'export') this.app.frames.toggle(false);
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
    const key = JSON.stringify([states, photos]);
    if (key !== this.statesKey) {
      this.statesKey = key;
      let n = 0;  // the shown steps are numbered
      for (const p of PHASES) {
        const b = this.steps[p], { state, summary } = states[p], label = phaseLabel(p, { photos });
        const shown = phaseShown(p, { photos });  // style, not `hidden`: .step sets its display
        if (shown) n++;
        b.className = `step ${state}${p === this.current ? ' current' : ''}`;
        b.style.display = shown ? '' : 'none';
        b.querySelector('.step-label').textContent = label;
        b.querySelector('.step-icon').textContent = state === 'todo' ? n : STATE_ICONS[state];
        // the Project's summary is the header's project label
        b.querySelector('.step-sum').textContent = p === 'project' ? '' : summary;
        b.title = `${label}: ${state}${summary ? ` (${summary})` : ''}`;
      }
      this.title.textContent = phaseLabel(this.current, { photos });
      for (const t of this.tabs) if (t.dataset.tab === 'timeline') t.style.display = photos ? 'none' : '';
      // the phase may have been shown before the photo set was known: off the timeline then
      if (photos && this.tabs.some(t => t.dataset.tab === 'timeline' && t.classList.contains('on'))) this.setDock(DOCK_TAB_PHOTOS[this.current]);
      this.summary.textContent = states[this.current].summary;
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
      b.addEventListener('click', () => { this._run(a.id); this.footKey = ''; this.refresh(); });
      return b;
    }));
  }

  _input() {
    const app = this.app, sfm = app.sfm, tr = app.trainer, colmap = app.colmap;
    const q = sfm?.quality;
    return {
      label: app.projectLabel(),
      video: sfm?.videoName ? { name: sfm.videoName, projectOnly: !sfm.file && !sfm.photos, ...(sfm.photos ? { photos: sfm.videoInfo?.photos ?? sfm.photos.length } : {}) } : null,
      sfm: sfm ? {
        running: sfm.running ? (sfm.keyframeCount === null ? 'keyframes' : 'cameras') : null,
        busy: sfm.busy,
        keyframes: sfm.keyframeCount,
        quality: q ? { keyframes: q.frames.length, registered: q.frames.filter(f => f.registered).length,
                       problems: q.counts.amber + q.counts.red, version: q.version } : null,
        sentVersion: sfm.sent?.version ?? null,
        canRun: !!(sfm.file || sfm.photos), ready: sfm.ready, canUpdate: sfm.canUpdate,
      } : null,
      colmap: colmap ? { cams: colmap.images.length, fromSfm: !!colmap.fromSfm } : null,
      trainer: tr ? { step: tr.stepCount, training: app.params.training, psnr: tr.lastPsnr, gt: !!app.gtAvailable } : null,
      mesh: app.meshState,
    };
  }

  // A footer button (phases.js phaseActions ids)
  _run(id) {
    const app = this.app, sfm = app.sfm;
    if (id.startsWith('go:')) return this.show(id.slice(3));
    switch (id) {
      case 'stop': return sfm.stop();
      case 'runAll': return sfm.run(null, true);
      case 'keyframes': return sfm.run('keyframes');
      case 'cameras': return sfm.run();
      case 'update': return sfm.requestSend('update');
      case 'send': return sfm.requestSend('restart');
      case 'train': return app._setTraining(!app.params.training);
      case 'step': return app.stepOnce();
      case 'mesh': this.show('export'); return app.saveMesh();
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
