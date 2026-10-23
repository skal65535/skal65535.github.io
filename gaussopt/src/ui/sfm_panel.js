// sfm_panel.js — The "Video → Cameras" controls (DESIGN-sfm §6, §7), loaded on first use by
// src/app.js: they fill the Video, Keyframes and Cameras phases of the inspector (shell.js, whose
// footer runs them), drive src/sfm/worker.js, and draw the dock's frame timeline: sharpness per
// sampled frame, keyframes colored by quality, loop edges.
// A keyframe range selected on the strip (drag; click for one; or a frustum in the 3D view) is
// what the repair actions apply to; undo reverts the last one. The result goes to the trainer or
// out as a COLMAP gs/ folder. Options can be preset from the URL by name with an `sfm_` prefix
// (sfm_spacing=0.03, sfm_maxFeatures=8192, ...).
"use strict";

import { mkGui } from '../../../common/controls.js';
import { loadColmapFromMemory, setPose } from '../colmap.js';
import { umeyama } from '../sfm/align.js';
import { compose, inverse, apply, fromPose, toPose } from '../sfm/sim3.js';
import { rotationAngle } from '../sfm/linalg.js';
import { SESSION_DEFAULTS } from '../sfm/defaults.js';

// progress stages (worker.js, pipeline.js, mapper.js, reconstruct.js); total 0: no count
const STAGES = { analyze: 'analyzing frames', features: 'keyframes + SIFT', match: 'matching', register: 'mapping: registered',
                 loopMatch: 'matching loop candidates', loops: 'closing loops', refine: 'refining windows',
                 retriangulate: 'retriangulating', lateMerge: 'merging segments', repair: 'repairing stretches', undistort: 'undistorting' };
const SCORE_CSS = { green: '#4ade80', amber: '#fbbf24', red: '#f87171' };
const SCORE_ABGR = { green: 0xff80de4a, amber: 0xff24bffb, red: 0xff7171f8 }, SELECTED_ABGR = 0xffffffff;

const DEFAULTS = { ...SESSION_DEFAULTS, autoSend: true, fixFocal: true };

// Tooltips of the video, keyframes and cameras controls (mkGui tips: by key, or by folder name)
const TIPS = {
  source: 'the video the keyframes and cameras come from',
  pick: 'open another video file (the current session is replaced)',
  status: 'what this step is doing, or its last result',
  'Frames (S0)': 'which video frames become keyframes: sampled, scored for sharpness, picked by motion',
  sampleFps: 'frames per second of the video looked at (sharpness and motion are measured on these)',
  spacing: 'a new keyframe once the image content moved this much, as a fraction of the image diagonal (smaller: more keyframes)',
  blurRel: 'a picked frame less sharp than this fraction of the picks\' median sharpness is dropped (0: keep all)',
  maxDim: 'keyframes are scaled down to this long side (photos for training and SIFT)',
  model: 'which cameras the trainer has: the current model, or an older one (before repairs)',
  autoSend: 'start training with the cameras as soon as they are found',
  autoRepair: 'once the cameras are found, run Repair all (one undo reverts it)',
  'Repair (selection on the timeline)': 'fix bad stretches of the camera path: select keyframes on the Timeline (drag, ' +
                                        'or click one), then an action. Each action can be undone',
  selection: 'the keyframes the actions apply to, and the worst problem among them',
  'Next problem →': 'select the next stretch of amber / red keyframes (with one good keyframe either side)',
  'Repair all (retry each, keep what helps)': 'Retry on every amber / red stretch (not rotation-only frames, not separate ' +
    'maps), each kept only when the scores improve, undone otherwise. No selection needed; one undo reverts it all',
  'Retry (re-match, re-register)': 'match the selection again with a wider window and a looser ratio test, then register ' +
                                   'it again in the map. For few points, weak links',
  'Densify (more keyframes)': 'add keyframes from the video between the selected ones, then register them. For fast ' +
                              'motion or blur gaps (needs the video)',
  'Drop keyframes': 'remove the selected keyframes (blurry, moving objects, flares) and re-adjust around them',
  'Refine (window BA)': 'bundle adjustment over the selection and the keyframes that see the same points. For jitter',
  'Redo (re-map, merge back)': 'map the selection again as a new piece and merge it back. For a broken stretch',
  'Force loop (first ↔ last)': 'the first and last selected keyframes see the same place: match them and close a loop. For missed loops',
  'Reject loops of the selection': 'drop the loop edges that touch the selection (Timeline arcs), then re-run loop closure. ' +
                                   'For false loops (repeated façades)',
  'Set focal (field below)': 'set the focal (Camera & mapping: focal, fix it) and run the reconstruction again',
  '↶ Undo': 'revert the last action',
  'Features & matching (S1–S2)': 'SIFT features and which keyframe pairs are matched',
  maxFeatures: 'most SIFT features per keyframe',
  window: 'each keyframe is matched to the next W ones (also the size of the bundle adjustment windows)',
  loops: 'look for loop closures: keyframes far apart in time that see the same place',
  'Camera & mapping (S2–S4)': 'intrinsics and the mapper',
  focal: "focal length in pixels, or 'auto' to estimate it",
  fixFocal: 'Set focal keeps it fixed (off: refined from that value)',
  hfovDeg: 'horizontal field of view of the camera, if known (0: estimated); sets the starting focal',
  sweeps: 'refinement passes of bundle adjustment over the whole path after mapping',
  'Cameras (video)': 'export the cameras found from the video',
  exp: 'write a COLMAP folder (sparse/0 + undistorted images) for other trainers',
  save: 'the whole SfM session as one .gosfm file, to reopen and repair later',
};

// Sim(3) mapping src points onto dst (Umeyama), refitted without the pairs whose residual
// exceeds 3x the median (cameras a repair moved)
function robustSim3(src, dst) {
  let idx = src.map((_, k) => k), G = null;
  for (let pass = 0; pass < 2; pass++) {
    G = umeyama(idx.map(k => src[k]), idx.map(k => dst[k]));
    const res = src.map((p, k) => Math.hypot(...apply(G, p).map((x, d) => x - dst[k][d])));
    const med = [...res].sort((a, b) => a - b)[res.length >> 1];
    const keep = src.map((_, k) => k).filter(k => res[k] <= 3 * med + 1e-12);
    if (keep.length < 3 || keep.length === idx.length) break;
    idx = keep;
  }
  return G;
}

// The options of DEFAULTS in `o` (a project's saved settings), of the same types
const known = (o = {}) => Object.fromEntries(Object.entries(o).filter(([k, v]) => k in DEFAULTS && typeof v === typeof DEFAULTS[k]));

function urlOptions() {
  const args = new URLSearchParams(location.search), out = {};
  for (const [k, v] of Object.entries(DEFAULTS)) {
    const a = args.get(`sfm_${k}`);
    if (a === null) continue;
    out[k] = typeof v === 'number' ? Number(a) : typeof v === 'boolean' ? a !== '0' && a !== 'false' : a;
  }
  return out;
}

export class SfmPanel {
  constructor(app) {
    this.app = app;
    this.o = { ...DEFAULTS, ...known(app.project?.manifest.settings?.sfm), ...urlOptions() };
    this.state = { source: '(no video)', status: 'idle', selection: '(none)', model: '' };
    this.worker = null;
    this.file = null;           // the open video (null for a .gosfm alone)
    this.videoName = null;      // the open video's name, or the one a .gosfm was made from
    this.keyframeCount = null;  // keyframes of the last selection + features (worker 'keyframes')
    this.result = null;     // last { sparse, images, summary } from the worker
    this.quality = null;    // last per-keyframe scores
    this.selected = [];     // selected keyframe names (time order)
    this.frameAnalysis = null;  // last S0 'selection' message: sampled frames, sharpness, keyframe picks
    this.busy = false;      // a request (repair, send, save) is in the worker
    this.running = false;   // a run is in the worker (runUntil: 'keyframes', or null for all)
    this.runUntil = null;
    this.pending = null;    // { resolve, reject } of exportFolder waiting for a fresh result
    this.storeDesc = app.project?.store.describe() ?? null;  // the project folder the worker writes into, null: none
    this.flushing = null;   // resolve() of flush(), waiting for 'flushed'
    this.videoInfo = null;  // the worker's last 'video' info
    this.restoring = false; // a project's session is being restored: no automatic send
    this._build();
  }

  _build() {
    const shell = this.app.shell, o = this.o, s = this.state;
    this.guis = [];
    const host = (phase) => {
      const sec = shell.section(phase);
      sec.replaceChildren();
      const div = document.createElement('div');
      div.className = 'gui-host';
      sec.append(div);
      const g = mkGui(div, { draggable: false, tips: TIPS });
      this.guis.push(g);
      return g;
    };

    const vid = host('video');
    vid.add(s, 'source').name('video').listen().disable();
    vid.add({ pick: () => this._pickFile() }, 'pick').name('Choose another video…');
    vid.add(s, 'status').name('status').listen().disable();

    const kf = host('keyframes');
    kf.add(s, 'status').name('status').listen().disable();
    const fr = kf.addFolder('Frames (S0)');
    fr.add(o, 'sampleFps', 1, 60, 1).name('sample fps');
    fr.add(o, 'spacing', 0.01, 0.2, 0.005).name('spacing (× diagonal)');
    fr.add(o, 'blurRel', 0, 1, 0.05).name('blur reject (× median)');
    fr.add(o, 'maxDim', [960, 1280, 1920, 3840]).name('max size (px)');

    const cams = host('cameras');
    cams.add(s, 'status').name('status').listen().disable();
    cams.add(s, 'model').name('training has').listen().disable();
    cams.add(o, 'autoSend').name('train when found');
    // DESIGN-sfm §6.3
    const fix = cams.addFolder('Repair (selection on the timeline)');
    fix.add(s, 'selection').name('selected').listen().disable();
    const act = (label, fn) => fix.add({ [label]: fn }, label).name(label);
    this.repairBtns = [
      act('Next problem →', () => this.selectNextProblem()),
      act('Repair all (retry each, keep what helps)', () => this.act('repair', [])),
      act('Retry (re-match, re-register)', () => this.act('retry', [this.selected])),
      act('Densify (more keyframes)', () => this.act('densify', [this.selected])),
      act('Drop keyframes', () => this.act('drop', [this.selected])),
      act('Refine (window BA)', () => this.act('refine', [this.selected, { intrinsics: false }])),
      act('Redo (re-map, merge back)', () => this.act('redo', [this.selected])),
      act('Force loop (first ↔ last)', () => this.act('forceLoop', [this.selected[0], this.selected.at(-1)])),
      act('Reject loops of the selection', () => this.act('rejectLoops', [this._loopsOfSelection()])),
      act('Set focal (field below)', () => this.act('setFocal', [this._focal(), { fix: this.o.fixFocal }])),
      act('↶ Undo', () => this.undo()),
    ];
    const fm = cams.addFolder('Features & matching (S1–S2)', { collapsed: true });
    fm.add(o, 'maxFeatures', [1024, 2048, 4096, 8192, 16384]).name('max features');
    fm.add(o, 'window', 3, 30, 1).name('window W');
    fm.add(o, 'loops').name('loop closure');
    const cam = cams.addFolder('Camera & mapping (S2–S4)', { collapsed: true });
    cam.add(o, 'focal').name("focal (px or 'auto')");
    cam.add(o, 'fixFocal').name('fix it (Set focal)');
    cam.add(o, 'hfovDeg', 0, 120, 1).name('HFOV ° (0 = estimate)');
    cam.add(o, 'sweeps', 0, 4, 1).name('refinement sweeps');
    cam.add(o, 'autoRepair').name('repair all when found');

    const exp = mkGui(document.getElementById('export-sfm'), { draggable: false, tips: TIPS }).addFolder('Cameras (video)');
    this.exportBtn = exp.add({ exp: () => this.exportFolder() }, 'exp').name('Export COLMAP gs/ folder…');
    this.saveBtn = exp.add({ save: () => this.saveProject() }, 'save').name('Save session (.gosfm)');
    this._enable();

    this.strip = document.getElementById('sfm-strip');
    this._stripEvents();
  }

  get ready() { return !!this.quality && !this.busy && !this.running; }

  // Update the running trainer's cameras (splats kept) instead of starting over
  get canUpdate() { return this.ready && !!this.app.colmap?.fromSfm && !!this.app.trainer; }

  _enable() {
    const ready = this.ready;
    for (const b of this.repairBtns) b.enable(ready);
    this.exportBtn.enable(!!this.result && !this.busy && !this.running);
    this.saveBtn.enable(ready);
    this.app.shell.refresh();
  }

  open(file = null) {
    if (file) this.load(file);
  }

  _pickFile() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'video/*,.mp4,.mov,.m4v,.gosfm';
    input.onchange = () => input.files[0] && this.load(input.files[0]);
    input.click();
  }

  _log(text, level = 'info') { this.app.log(text, level || 'info', 'sfm'); }

  // reopen: a new worker (after Stop) opens the current video again
  _worker(reopen = true) {
    if (this.worker) return this.worker;
    this.worker = new Worker(new URL('../sfm/worker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = ({ data }) => this._message(data);
    this.worker.onerror = (e) => { this._fail(e.message || 'worker failed to start'); };
    if (this.storeDesc) this.worker.postMessage({ type: 'store', store: this.storeDesc });
    if (reopen && this.file) this.worker.postMessage({ type: 'open', file: this.file });
    return this.worker;
  }

  load(file) {
    if (this.running || this.busy) { this._log(`${file.name}: stop the current work first`, 'warn'); return; }
    this.file = file;
    this.videoName = file.name;
    this.result = this.quality = this.frameAnalysis = this.duration = this.keyframeCount = null;
    this.selected = [];
    this.state.source = file.name;
    this.state.status = 'opening…';
    this._enable();
    const project = /\.gosfm$/i.test(file.name);  // a saved session: repairs, no re-run without the video
    if (project) this.file = null;
    (this.worker ?? this._worker(false)).postMessage({ type: project ? 'project' : 'open', file });
  }

  // ── project folder (src/project/, src/ui/project_panel.js) ──────────────────────

  // The folder the worker writes the keyframes, models and session into (null: none)
  setStore(desc) {
    this.storeDesc = desc;
    this.worker?.postMessage({ type: 'store', store: desc });
  }

  // Everything the session has, into the store just set; resolves to the files of a memory store
  flush() {
    if (!this.worker || this.running || this.busy) return Promise.resolve(null);
    this.busy = true;
    this._enable();
    return new Promise((resolve) => { this.flushing = resolve; this.worker.postMessage({ type: 'flush' }); });
  }

  // A project's saved session, without its video (repairs; re-runs need the video)
  restoreFromStore() {
    this.restoring = true;
    this.file = null;
    this.state.status = 'restoring the project…';
    this._worker(false).postMessage({ type: 'restoreStore' });
  }

  // The trainer has the cameras of model `version` (a reopened project)
  markSent(version) {
    this.sent = { version };
    this.state.model = 'the current model (from the project)';
  }

  sourceInfo() {
    const v = this.videoInfo ?? {}, f = this.file;
    return { name: v.name ?? this.videoName, size: f?.size ?? null, lastModified: f?.lastModified ?? null,
             duration: v.duration ?? null, width: v.width ?? null, height: v.height ?? null, fps: v.fps ?? null };
  }

  // The Keyframes phase of the manifest (null before the keyframes exist)
  keyframesRecord() {
    const { sampleFps, spacing, blurRel, maxDim, maxFeatures } = this.o;
    return this.keyframeCount ? { phases: { keyframes: { count: this.keyframeCount, options: { sampleFps, spacing, blurRel, maxDim, maxFeatures } } } } : null;
  }

  // The project's manifest (no project: nothing)
  // The options the controls set, for project.json (app.js settings())
  settings() { return known(this.o); }

  applySettings(s) {
    const url = urlOptions();
    for (const [k, v] of Object.entries(known(s))) if (!(k in url)) this.o[k] = v;
    for (const g of this.guis) g.updateDisplay();
  }

  _record(patch) {
    if (patch) this.app.project?.update(patch).catch(e => this._log(`project.json: ${e.message}`, 'warn'));
  }

  saveProject() {
    if (!this.ready) return;
    this.busy = true;
    this.state.status = 'packing the project…';
    this._enable();
    this.worker.postMessage({ type: 'save' });
  }

  // until: 'keyframes' stops after the keyframes and their features; null runs to the cameras,
  // then into the trainer when `send`
  run(until = null, send = this.o.autoSend) {
    if (this.running || this.busy) return;
    if (!this.file) { this._pickFile(); return; }
    this.running = true;
    this.runUntil = until;
    this.quality = null;
    this.keyframeCount = null;
    if (this.app.params.training) this._log('training paused while the video is processed (one GPU)', 'warn');
    this.app._setTraining(false);  // one GPU: SfM first (DESIGN-sfm §5.7)
    this._enable();
    this.sendMode = null;
    this.autoSend = !until && send;
    this._worker().postMessage({ type: 'run', options: { ...this.o, focal: this._focal(), until } });
    this.state.status = 'running…';
  }

  _focal() {
    const focal = Number(this.o.focal) > 0 ? Number(this.o.focal) : 'auto';
    if (focal === 'auto' && this.o.focal !== 'auto') { this.o.focal = 'auto'; this._log("focal: not a positive number, using 'auto'", 'warn'); }
    return focal;
  }

  // Terminating is the only way to interrupt a stage; the next run re-opens the video (the
  // session is lost)
  stop() {
    if (!this.worker) return;
    this.worker.terminate();
    this.worker = null;
    this.quality = null;
    this._abandon();
    this._done('stopped', 'warn');
  }

  // Requests in flight are lost (stop, worker error): nothing waits for their result
  _abandon() {
    this.busy = false;
    this.flushing?.(null);
    this.flushing = null;
    this.restoring = false;
    this.sendMode = null;
    this.autoSend = false;
    this.pending?.reject(new DOMException('no result from the worker', 'AbortError'));  // the cause is reported already
    this.pending = null;
  }

  _done(status, level) {
    this.running = false;
    this.state.status = status;
    if (level) this._log(status, level);
    this._enable();
  }

  _fail(message) {
    this._abandon();
    this._done(`error: ${message}`, 'error');
    this.app.showError(`Video → Cameras: ${message}`);
  }

  // ── repairs ──────────────────────────────────────────────────────────────────

  act(op, args) {
    if (!this.ready) return;
    if (op !== 'setFocal' && op !== 'repair' && !this.selected.length) { this._log('select keyframes on the strip first', 'warn'); return; }
    if (op === 'rejectLoops' && !args[0].length) { this._log('no loop edge touches the selection', 'warn'); return; }
    this.busy = true;
    this.state.status = `${op}…`;
    this._enable();
    this.worker.postMessage({ type: 'action', op, args });
  }

  undo() {
    if (!this.ready) return;
    this.busy = true;
    this._enable();
    this.worker.postMessage({ type: 'undo' });
  }

  // mode: 'restart' (new splats from the points) | 'update' (the running trainer's cameras)
  requestSend(mode = 'restart') {
    if (!this.ready) return;
    this.busy = true;
    this.sendMode = mode;
    this.state.status = 'undistorting…';
    this._enable();
    this.worker.postMessage({ type: 'send' });
  }

  _loopsOfSelection() {
    const sel = new Set(this.selected);
    return (this.quality?.loops ?? []).filter(l => sel.has(l.a) || sel.has(l.b)).map(l => [l.a, l.b]);
  }

  // The next run of keyframes that are not green, after the current selection (wraps around)
  selectNextProblem() {
    const frames = this.quality?.frames ?? [];
    const start = this.selected.length ? frames.findIndex(f => f.name === this.selected.at(-1)) + 1 : 0;
    for (let n = 0; n < frames.length; n++) {
      const k = (start + n) % frames.length;
      if (frames[k].score === 'green') continue;
      let e = k;
      while (e + 1 < frames.length && frames[e + 1].score !== 'green') e++;
      this._select(frames.slice(Math.max(0, k - 1), Math.min(frames.length, e + 2)).map(f => f.name));  // with one green either side
      return;
    }
    this._log('no keyframe is amber or red', 'ok');
  }

  _select(names) {
    this.selected = names;
    const frames = new Map((this.quality?.frames ?? []).map(f => [f.name, f]));
    const worst = names.map(n => frames.get(n)).filter(Boolean)
      .sort((a, b) => ['red', 'amber', 'green'].indexOf(a.score) - ['red', 'amber', 'green'].indexOf(b.score))[0];
    this.state.selection = names.length ? `${names.length}: ${names[0]}${names.length > 1 ? ` … ${names.at(-1)}` : ''}` +
      (worst && worst.score !== 'green' ? ` (${worst.reason})` : '') : '(none)';
    this.app.frames.setScores(frames);
    this.app.frames.setSelected(names);
    this._drawStrip();
    this._paint3D();
  }

  // From the contact sheet (src/ui/frames.js): before the cameras exist, it only marks the frames
  pickFrames(names) { this._select(names); }

  // From the 3D view (app.openThumbnail)
  picked(name) {
    if (this.quality?.frames.some(f => f.name === name)) this._select([name]);
  }

  // Validation PSNR per view, from the trainer: "trains badly" scores (DESIGN-sfm §6.1)
  viewPsnr(perView) {
    if (this.worker && this.quality && perView?.length) this.worker.postMessage({ type: 'psnr', values: perView });
  }

  // Frustums of the trainer's cameras colored by score, the selection in white
  _paint3D() {
    const images = this.app.colmap?.fromSfm ? this.app.colmap.images : null;
    if (!images || !this.quality) return;
    const score = new Map(this.quality.frames.map(f => [f.name, f.score])), sel = new Set(this.selected);
    for (const im of images) im.wireColor = sel.has(im.name) ? SELECTED_ABGR : SCORE_ABGR[score.get(im.name)] ?? null;
    this.app._refreshWireframes();
  }

  // ── worker messages ──────────────────────────────────────────────────────────

  _message(m) {
    if (m.type === 'video') {  // a video, or the one a dropped project was made from (info: { project, name, duration })
      const v = m.info;
      this.duration = v.duration;
      this.videoName = v.name;
      this.videoInfo = v;
      this.state.status = 'ready';
      if (v.project) {
        this.state.source = `${v.project} — of ${v.name}, ${v.duration.toFixed(1)} s (repairs only)`;
        this._log(`${v.project}: a project of ${v.name}; densify and Run all need the video`);
        return;
      }
      this.state.source = `${v.name} — ${v.width}×${v.height}, ${v.fps.toFixed(2)} fps, ${v.duration.toFixed(1)} s`;
      if (this.app.project) this._record({ source: this.sourceInfo() });
      this._log(`${v.name}: ${v.codec}, ${v.width}×${v.height}${v.rotation ? ` (rotated ${v.rotation}°)` : ''}, ` +
                `${v.frames} frames, ${v.fps.toFixed(2)} fps, ${v.duration.toFixed(1)} s`);
    } else if (m.type === 'log') {
      this._log(m.text, m.level || 'info');
    } else if (m.type === 'progress') {
      this.state.status = `${STAGES[m.stage] ?? m.stage}${m.total ? ` ${m.done}/${m.total}` : '…'}`;
    } else if (m.type === 'selection') {
      this.frameAnalysis = m;
      this._drawStrip();
    } else if (m.type === 'keyframes') {  // selection + features done (or restored)
      this.keyframeCount = m.count;
      if (!this.restoring) this._record(this.keyframesRecord());
      this.app.shell.refresh();
    } else if (m.type === 'thumbs') {
      this.app.frames.set(m.frames, m.aspect);
      if (this.app.shell.current === 'keyframes') this.app.frames.toggle(true);
      this._select(this.selected);
    } else if (m.type === 'quality') {
      this.quality = m.quality;
      const names = new Set(m.quality.frames.map(f => f.name));
      this.selected = this.selected.filter(n => names.has(n));
      const c = m.quality.counts;
      if (m.action) {
        this.busy = false;
        this.state.status = `${m.action.op} done: ${c.green} green, ${c.amber} amber, ${c.red} red`;
        if (this.app.colmap?.fromSfm) this.state.model = 'an older model: Update trainer cameras';
      }
      this._select(this.selected);
      this._enable();
    } else if (m.type === 'result') {
      this.result = m;
      const s = m.summary;
      this.busy = !!this.flushing;  // a flush goes on until 'flushed'
      this._done(`done: ${s.exported} cameras, ${s.points} points`);
      if (!this.sendMode && !this.autoSend && !this.flushing) this.app.shell.advance('cameras');
      this._record({ phases: { cameras: { version: s.version, keyframes: s.keyframes, registered: s.registered, exported: s.exported,
                                          focal: s.focal, points: s.points, window: this.o.window, loops: this.o.loops } } });
      if (this.sendMode === 'update') this.update();
      else if (this.sendMode || this.autoSend) this.send();
      this.sendMode = null;
      this.autoSend = false;
      this.pending?.resolve(m);
      this.pending = null;
      this._log(`Video → Cameras: ${s.exported} cameras (${s.registered}/${s.keyframes} keyframes registered), ` +
                `${s.points} points, focal ${s.focal.toFixed(1)} px`, 'ok');
    } else if (m.type === 'restored') {
      this.restored = true;
      this.state.status = 'restored the saved session';
      if (this.o.autoSend && !this.restoring) this.requestSend();
      this.restoring = false;
    } else if (m.type === 'flushed') {
      this.busy = false;
      this._enable();
      this.flushing?.(m.files);
      this.flushing = null;
    } else if (m.type === 'idle') {
      this.busy = false;
      if (this.running) {  // a run that stopped early (until)
        this._done(m.status);
        if (this.runUntil === 'keyframes') this.app.shell.advance('keyframes');
      }
      this.state.status = m.status;
      this._enable();
    } else if (m.type === 'project') {
      this.busy = false;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(m.blob);
      a.download = m.name;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 60000);
      this.state.status = `saved ${m.name} (${(m.blob.size / 1e6).toFixed(0)} MB)`;
      this._enable();
    } else if (m.type === 'error') {
      this._fail(m.message);
    }
  }

  // ── the strip ────────────────────────────────────────────────────────────────

  _x(t) { return (t / (this.duration || 1)) * this.strip.width; }
  _t(x) { return (x / this.strip.width) * (this.duration || 1); }

  _stripEvents() {
    const c = this.strip, at = (e) => (e.offsetX / c.clientWidth) * c.width;
    let down = null;
    c.addEventListener('pointerdown', (e) => { down = at(e); c.setPointerCapture(e.pointerId); });
    c.addEventListener('pointermove', (e) => {
      const x = at(e);
      if (down !== null) { this.drag = [Math.min(down, x), Math.max(down, x)]; this._drawStrip(); }
      const f = this._nearest(x);
      const loops = f ? this.quality.loops.filter(l => l.a === f.name || l.b === f.name) : [];
      c.dataset.tip = f ? `${f.name}: ${f.score}${f.reason ? ` — ${f.reason}` : ''}` +
                    (loops.length ? `\n${loops.length} loop(s), ${loops.filter(l => l.suspect).length} suspect` : '') : '';
    });
    c.addEventListener('pointerup', (e) => {
      if (down === null) return;
      const x = at(e), [a, b] = [Math.min(down, x), Math.max(down, x)];
      down = null; this.drag = null;
      const frames = this.quality?.frames ?? [];
      if (b - a < 3) { const f = this._nearest(x); this._select(f ? [f.name] : []); return; }
      this._select(frames.filter(f => this._x(f.time) >= a && this._x(f.time) <= b).map(f => f.name));
    });
  }

  _nearest(x) {
    let best = null;
    for (const f of this.quality?.frames ?? []) if (!best || Math.abs(this._x(f.time) - x) < Math.abs(this._x(best.time) - x)) best = f;
    return best && Math.abs(this._x(best.time) - x) < 6 ? best : null;
  }

  _drawStrip() {
    const c = this.strip, ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    document.getElementById('strip-hint').hidden = !!this.frameAnalysis;
    if (this.frameAnalysis) {  // sampled frames: sharpness, keyframe picks bright, blurry picks red
      const { sharpness, status, times } = this.frameAnalysis, max = sharpness.reduce((m, v) => Math.max(m, v), 0);
      sharpness.forEach((v, i) => {
        const x = this._x(times[i]), h = Math.max(1, 30 * v / max);
        ctx.fillStyle = status[i] === 'kept' ? 'rgba(148, 163, 184, 0.9)' : status[i] === 'blurry' ? '#f87171' : 'rgba(148, 163, 184, 0.3)';
        ctx.fillRect(x, 32 - h, Math.max(1, c.width / sharpness.length), h);
      });
    }
    if (!this.quality) return;
    const sel = new Set(this.selected), main = this.quality.main;
    for (const f of this.quality.frames) {  // keyframes by score; other segments dimmed
      const x = this._x(f.time);
      ctx.globalAlpha = f.segment === main ? 1 : 0.45;
      ctx.fillStyle = f.registered ? SCORE_CSS[f.score] : '#64748b';
      ctx.fillRect(x - 1, 38, 3, 20);
      ctx.globalAlpha = 1;
      if (sel.has(f.name)) { ctx.fillStyle = '#ffffff'; ctx.fillRect(x - 1, 60, 3, 4); }
    }
    if (sel.size) {  // the selection: a band over its time range
      const xs = this.quality.frames.filter(f => sel.has(f.name)).map(f => this._x(f.time));
      ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.fillRect(Math.min(...xs) - 3, 0, Math.max(...xs) - Math.min(...xs) + 6, c.height);
    }
    // loop edges as arcs under the keyframes, higher for a longer span: verified ones blue, suspect
    // ones (residual after the pose graph) red, on top; with a selection, the others are dimmed
    const time = new Map(this.quality.frames.map(f => [f.name, f.time]));
    const arcs = this.quality.loops.map(l => ({ ...l, xa: this._x(time.get(l.a)), xb: this._x(time.get(l.b)), hit: sel.has(l.a) || sel.has(l.b) }))
      .sort((p, q) => (p.hit - q.hit) || (p.suspect - q.suspect));
    for (const l of arcs) {
      const h = 6 + (c.height - 72) * Math.sqrt(Math.abs(l.xb - l.xa) / c.width);
      ctx.globalAlpha = l.hit ? 1 : sel.size ? 0.06 : 0.3;
      ctx.strokeStyle = l.suspect ? '#f87171' : '#60a5fa';
      ctx.lineWidth = l.hit ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(l.xa, 66);
      ctx.quadraticCurveTo((l.xa + l.xb) / 2, 66 + 2 * h, l.xb, 66);  // the curve peaks at half its control point
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
    if (this.drag) { ctx.fillStyle = 'rgba(255, 255, 255, 0.15)'; ctx.fillRect(this.drag[0], 0, this.drag[1] - this.drag[0], c.height); }
  }

  // ── output ───────────────────────────────────────────────────────────────────

  async send() {
    if (!this.result) return;
    const app = this.app;
    try {
      if (await app.loadColmap(loadColmapFromMemory(this.result.sparse, this.result.images, { valSplit: app.params.val_split }), { label: 'Video → Cameras' })) {
        this.sent = this.result.summary;
        this.state.model = `the current model (${this.result.summary.exported} cameras)`;
        this._record({ phases: { train: { fromCameras: this.sent.version } } });
        this._paint3D();
      }
    } catch (e) {
      app.showError(`Send to trainer failed: ${e.message}`);
    }
    this._enable();
  }

  // The repaired model's cameras into the running trainer, splats kept: the new model is aligned
  // onto the old cameras (Sim(3) on the keyframes both have, refitted without the ones that moved),
  // so the cameras land in the frame the splats were trained in. How far they moved is reported;
  // after a large change (focal, loops), new splats may train better.
  async update() {
    const app = this.app, old = app.colmap;
    if (!this.result || !old?.fromSfm || !app.trainer) return this.send();
    try {
      const ds = loadColmapFromMemory(this.result.sparse, this.result.images, { valSplit: app.params.val_split });
      const before = new Map(old.images.map(im => [im.name, im]));
      const common = ds.images.filter(im => before.has(im.name));
      if (common.length < 3) throw new Error(`only ${common.length} keyframes in common with the trainer's model`);
      const G = robustSim3(common.map(im => im.center), common.map(im => before.get(im.name).center));
      for (const im of ds.images) {
        const pose = toPose(compose(fromPose(im), inverse(G)));
        setPose(im, pose.R, pose.t);
        im.zMin *= G.s; im.zMax *= G.s;
      }
      const P = ds.points3D.positions;
      for (let k = 0; k < P.length; k += 3) P.set(apply(G, [P[k], P[k + 1], P[k + 2]]), k);
      const radius = app.octree.sceneRadius, moved = common.map(im => Math.hypot(...im.center.map((x, d) => x - before.get(im.name).center[d])) / radius);
      const turned = common.map(im => rotationAngle(im.R, before.get(im.name).R) * 180 / Math.PI);
      const keepTextures = JSON.stringify(this.result.summary.distorted) === JSON.stringify(this.sent?.distorted);
      const r = await app.swapCameras(ds, { keepTextures });
      const q = (v, p) => [...v].sort((a, b) => a - b)[Math.min(v.length - 1, Math.floor(p * v.length))];
      const msg = `cameras updated, splats kept: ${r.kept} kept, ${r.added} added, ${r.dropped} dropped; moved by ` +
        `${(100 * q(moved, 0.5)).toFixed(2)}% of the scene radius (max ${(100 * q(moved, 1)).toFixed(1)}%), turned ` +
        `${q(turned, 0.5).toFixed(2)} deg (max ${q(turned, 1).toFixed(1)})${keepTextures ? '' : '; photos reloaded (new undistortion)'}`;
      const large = q(moved, 0.9) > 0.02 || q(turned, 0.9) > 1;
      this._log(`Video → Cameras: ${msg}`, large ? 'warn' : 'ok');
      if (large) this._log('a large part of the map moved: if training degrades, Send to trainer (new splats)', 'warn');
      this.sent = this.result.summary;
      this._record({ phases: { train: { fromCameras: this.sent.version } } });
      this.lastUpdate = { ...r, moved: q(moved, 0.5), maxMoved: q(moved, 1), turned: q(turned, 0.5), keepTextures };
      this.state.model = `the current model (${this.result.summary.exported} cameras, splats kept)`;
      this._paint3D();
    } catch (e) {
      app.showError(`Update of the trainer's cameras failed: ${e.message}`);
    }
    this._enable();
  }

  // gs/ folder: sparse/{cameras,images,points3D}.bin + images/*.jpg (File System Access API) of
  // the current model (asked from the worker again after repairs; the picker needs the click)
  async exportFolder() {
    if (!this.result || this.busy || this.running) return;
    if (!window.showDirectoryPicker) { this.app.showError('Export needs the File System Access API (Chrome / Edge)'); return; }
    try {
      const root = await window.showDirectoryPicker({ mode: 'readwrite' });
      if (this.busy || this.running) return;  // started while the picker was open
      if (this.quality && this.quality.version !== this.result.summary.version) {
        this.busy = true;
        this._enable();
        await new Promise((resolve, reject) => { this.pending = { resolve, reject }; this.worker.postMessage({ type: 'send' }); });
      }
      const write = async (dir, name, data) => {
        const w = await (await dir.getFileHandle(name, { create: true })).createWritable();
        await w.write(data);
        await w.close();
      };
      const sparse = await root.getDirectoryHandle('sparse', { create: true });
      for (const [name, buf] of Object.entries(this.result.sparse)) await write(sparse, name, buf);
      const images = await root.getDirectoryHandle('images', { create: true });
      for (const { name, blob } of this.result.images) await write(images, name, blob);
      this._log(`exported ${this.result.images.length} images + sparse/ to ${root.name}/`, 'ok');
    } catch (e) {
      if (e.name !== 'AbortError') this.app.showError(`Export failed: ${e.message}`);
    }
  }
}
