// shell.js — The page layout (index.html, gaussopt.css): the phase stepper, the inspector that shows
// the current phase's controls with its next-step buttons pinned at the bottom, the dock tabs
// (timeline / charts / log), the view toolbar and help, and the 3D canvas fitted to the view.
// refresh() re-derives the phase states (phases.js) from the app; the DOM changes only when they do.
"use strict";

import { PHASES, STATE_ICONS, phaseStates, advanceTo, phaseLabel, phaseShown } from './phases.js';

const DOCK_TAB = { project: 'log', video: 'timeline', keyframes: 'timeline', cameras: 'timeline', train: 'charts', export: 'log' };
// a photo set has no timeline (DESIGN-sfm §13.3 P5): its contact sheet is the view of the Photos phase
const DOCK_TAB_PHOTOS = { ...DOCK_TAB, video: 'log', keyframes: 'log', cameras: 'log' };
const CANVAS_MAX_SCALE = 2;  // CSS scales the canvas's backing store up to 2x
const DOCK_FOLDED_KEY = 'gaussopt.dockFolded';  // this browser's choice
const CANVAS_LONG_SIDE = 960; // backing store: the dataset's aspect, this on its long side

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
    PHASES.forEach((p, i) => {
      if (i) this.stepper.append(Object.assign(document.createElement('span'), { className: 'sep' }));
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
    const states = phaseStates(this._input()), photos = this.photos;
    const key = JSON.stringify([states, photos]);
    if (key !== this.statesKey) {
      this.statesKey = key;
      for (const p of PHASES) {
        const b = this.steps[p], { state, summary } = states[p], label = phaseLabel(p, { photos });
        b.className = `step ${state}${p === this.current ? ' current' : ''}`;
        const shown = phaseShown(p, { photos });  // style, not `hidden`: .step sets its display
        b.style.display = shown ? '' : 'none';
        if (b.previousElementSibling?.classList.contains('sep')) b.previousElementSibling.style.display = b.style.display;
        b.querySelector('.step-label').textContent = label;
        b.querySelector('.step-icon').textContent = STATE_ICONS[state];
        b.querySelector('.step-sum').textContent = summary;
        b.title = `${label}: ${state}${summary ? ` (${summary})` : ''}`;
      }
      this.title.textContent = phaseLabel(this.current, { photos });
      for (const t of this.tabs) if (t.dataset.tab === 'timeline') t.style.display = photos ? 'none' : '';
      // the phase may have been shown before the photo set was known: off the timeline then
      if (photos && this.tabs.some(t => t.dataset.tab === 'timeline' && t.classList.contains('on'))) this.setDock(DOCK_TAB_PHOTOS[this.current]);
      this.summary.textContent = states[this.current].summary;
      this.label.textContent = states.project.summary;
    }
    const actions = this._actions(this.current);
    const footKey = JSON.stringify(actions.map(a => [a.label, a.enabled, a.primary]));
    if (footKey === this.footKey) return;
    this.footKey = footKey;
    this.foot.replaceChildren(...actions.map(a => {
      const b = document.createElement('button');
      b.textContent = a.label;
      b.title = a.title ?? '';
      b.disabled = !a.enabled;
      if (a.primary) b.className = 'primary';
      b.addEventListener('click', () => { a.run(); this.footKey = ''; this.refresh(); });
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
      } : null,
      colmap: colmap ? { cams: colmap.images.length, fromSfm: !!colmap.fromSfm } : null,
      trainer: tr ? { step: tr.stepCount, training: app.params.training, psnr: tr.lastPsnr } : null,
    };
  }

  // Next-step buttons of a phase: { label, run, enabled, primary, title }
  _actions(phase) {
    const sfm = this.app.sfm;
    if (!sfm || !['video', 'keyframes', 'cameras'].includes(phase)) return [];
    if (sfm.running || sfm.busy) return [{ label: '■ Stop', run: () => sfm.stop(), enabled: sfm.running, title: 'interrupts the video work (the session is lost)' }];
    const canRun = !!(sfm.file || sfm.photos);
    if (phase === 'video' && sfm.photos) {  // the Photos phase: the photos are the keyframes
      return [
        { label: 'Run all, then train', run: () => sfm.run(null, true), enabled: canRun, title: 'cameras, then training' },
        { label: 'Find cameras →', run: () => sfm.run(), enabled: canRun, primary: true },
      ];
    }
    if (phase === 'video') {
      return [
        { label: 'Run all, then train', run: () => sfm.run(null, true), enabled: canRun, title: 'keyframes, cameras, then training' },
        { label: 'Find keyframes →', run: () => sfm.run('keyframes'), enabled: canRun, primary: true },
      ];
    }
    if (phase === 'keyframes') return [{ label: 'Find cameras →', run: () => sfm.run(), enabled: canRun, primary: true }];
    return [
      ...(sfm.canUpdate ? [{ label: 'Update training cameras', run: () => sfm.requestSend('update'), enabled: true,
                             title: 'the repaired cameras into the running training, splats kept' }] : []),
      { label: 'Train on these cameras →', run: () => sfm.requestSend('restart'), enabled: sfm.ready, primary: true,
        title: 'new splats from these cameras and points' },
    ];
  }

  // The canvas takes the photos' aspect (a portrait video: a portrait view), so the GT | render
  // comparison, which fills the canvas, is not stretched
  shapeCanvas({ width, height }) {
    const canvas = this.app.canvas, s = CANVAS_LONG_SIDE / Math.max(width, height);
    const w = Math.round(width * s), h = Math.round(height * s);
    if (canvas.width === w && canvas.height === h) return;
    canvas.width = w;
    canvas.height = h;
    this._fitCanvas();
  }

  // The canvas CSS size: the largest box of the canvas's aspect in the view, at most CANVAS_MAX_SCALE
  _fitCanvas() {
    const view = document.getElementById('go-view'), canvas = this.app.canvas;
    const narrow = window.matchMedia('(max-width: 1000px)').matches;
    // narrow (stacked) layout: the view's height follows the canvas, so a portrait one is capped
    const w = view.clientWidth, h = narrow ? 0.8 * window.innerHeight : view.clientHeight;
    const s = Math.min(w / canvas.width, h / canvas.height, CANVAS_MAX_SCALE);
    if (!(s > 0)) return;
    canvas.style.width = `${Math.floor(canvas.width * s)}px`;
    canvas.style.height = `${Math.floor(canvas.height * s)}px`;
  }
}
