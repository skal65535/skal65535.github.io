// Gauss-Opt app: GUI (its params: params.js), dataset / checkpoint / snapshot loading, camera
// modes (orbit, drone, COLMAP snap, spline fly-to), training loop triggers and the action log.

import { initGPU } from '../../common/webgpu.js';
import { mkGui, onAnyChange } from '../../common/controls.js';
import {
  SPLAT_FLOATS, rgbToSh0,
  loadColmapFromUrl, loadColmapFromFiles,
  parsePlyBuffer, parseSplatBuffer, exportPlyBuffer,
} from './colmap.js';
import { LooseOctree } from './octree.js';
import {
  GaussOptRenderer, drawTelemetryPanel, createCameraSplineAnimation, evaluateCameraSpline,
} from './render.js';
import { readGlb, writeGlb } from './glb.js';
import { encodeJpeg, decodeImage } from './mesh_image.js';
import { clamp, dot, sub, madd, logit, buildOrbitViewMatrix, estimateWorldUp, orbitAnglesFromCenter, fitIntrinsics } from './math.js';
import { GaussOptTrainer } from './trainer.js';
import {
  VISU_MODES, COLOR_MODES, COMPARE_MODES, FRUSTUM_IMAGES, MESH_VIEWS, MESH_TEXTURES, OPT_METHODS, TIPS, DATASETS, NOT_A_URL, POINTS3D,
  SNAPSHOT_PARAMS, PROJECT_PARAMS, defaultParams, applyUrlParams, urlNamesOf, urlAfterChange, URL_FLAGS,
} from './params.js';
import { Drone } from './drone.js';
import { DemoFlight } from './demo.js';
import { shortestPath, pathStats } from './path.js';
import { encodeSnapshot, decodeSnapshot, isSnapshotName, SNAPSHOT_EXT } from './snapshot.js';
import { meshFromTrainer, meshFromPly, textureFromTrainer } from './mesh.js';
import { LogPanel } from './ui/log.js';
import { Shell } from './ui/shell.js';
import { FramesSheet } from './ui/frames.js';
import { ProjectPanel } from './ui/project_panel.js';
import { downloadBlob } from './ui/download.js';
import { FramePacer } from './pace.js';

// CPU time budget per frame for training steps (FramePacer sets their count, steps_per_frame
// bounds it)
const TRAIN_BUDGET_MS = 20;
// A still view is redrawn at least this often (ms), for the changes the redraw key misses (App._draw)
const REDRAW_MS = 1000;
// the camera path (P7, path.js): the turn weight, measured on cuisine, salon, minerals, out_scene1
// (photo_runs/logs/pathweights.log: at 2, no step turns more than capture order's largest), the
// offer's ratio, and the most cameras the shortest path is computed for
const PATH_ROT_WEIGHT = 2, PATH_OFFER = 1.25, PATH_MAX = 100;
const pathViews = (images) => images.map(im => ({ center: im.center, forward: im.forward }));

const SETTINGS_DELAY_MS = 800;  // after the last control change, the settings go to project.json

const pad5 = n => String(n).padStart(5, '0');

const mb   = buf => (buf.byteLength / 1048576).toFixed(1);
// Checkpoint entry name: availablePlys holds strings for URL datasets, {name, file} for dropped folders
const plyName = p => (typeof p === 'string' ? p : p.name);

// fetch(url), throwing on an HTTP error
async function fetchOk(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText || 'Not Found'} (${url})`);
  return resp;
}

// url as a File named after its last path component, dated by the server, not now: the saved
// session of a video or photo is found by name, size and date
async function fetchFile(url, fallbackName, type) {
  const resp = await fetchOk(url);
  const lastModified = Date.parse(resp.headers.get('last-modified') ?? '') || 0;
  return new File([await resp.blob()], url.split('/').pop() || fallbackName, { type, lastModified });
}

function createSyntheticScene(count = 256) {
  const params = new Float32Array(count * SPLAT_FLOATS);
  for (let i = 0; i < count; i++) {
    const b = i * SPLAT_FLOATS;
    const a = (i / count) * Math.PI * 2;
    const r = 0.6 + 0.3 * Math.sin(i * 1.7);
    params[b + 0]  = Math.cos(a) * r;
    params[b + 1]  = Math.sin(a * 2) * 0.4;
    params[b + 2]  = Math.sin(a) * r;
    params[b + 3]  = logit(0.75);
    params[b + 4]  = params[b + 5] = params[b + 6] = -2.8;  // log-scale
    params[b + 8]  = 1.0;                                   // identity quat
    params[b + 12] = rgbToSh0(0.5 + 0.45 * Math.cos(a));
    params[b + 13] = rgbToSh0(0.5 + 0.45 * Math.sin(a));
    params[b + 14] = rgbToSh0(0.6);
  }
  return { count, params, octreeData: null };
}

export class GaussOptApp {
  constructor() {
    this.canvas     = document.getElementById('main-canvas');
    this.sideCanvas = document.getElementById('side-canvas');
    this.statusEl   = document.getElementById('status-line');
    this.errorEl    = document.getElementById('error-line');
    this.logPanel   = new LogPanel(document.getElementById('log-body'), document.getElementById('log-filter'));
    this.trainBtn   = document.getElementById('BtnTrain');

    this.params = defaultParams();
    this.urlKeys = applyUrlParams(this.params);
    this.autoTrain = this.params.training;  // &train: start training once the first model is loaded
    this.trainStartStep = 0;  // stepCount when training last started (stop_after)

    this.gpu         = null;
    this.renderer    = null;
    this.octree      = null;
    this.trainer     = null;
    this.colmap      = null;
    this.retired     = [];    // replaced trainers, destroyed at the next frame boundary
    this.loadGen     = 0;     // bumped by every load: stale completions are dropped
    this.loadedPly   = POINTS3D;  // resume_ply of the model in the trainer
    this.plyFromFile = false; // model loaded from a local file (no URL form)
    this.gtAvailable = false; // training photos of the loaded dataset readable
    this.serverOk    = null;  // serve.py reachable (POST /save), see _probeServer()
    this.gpuFrames   = [];    // training frames submitted, not yet done on the GPU: { done }
    this.pacer       = new FramePacer();  // steps per training frame, from their GPU time
    this._loggedStats = null; // octree.lastStats already logged by _logRebalance()
    this.shell       = null;  // layout and phases (src/ui/shell.js)
    this.frames      = null;  // keyframe contact sheet (src/ui/frames.js)
    this.project     = null;  // the project folder (src/project/project.js), null: none
    this.projectPanel = null; // its controls (src/ui/project_panel.js)
    this.guis        = [];    // control groups (project, train, view), see _setupGui()
    this.datasetFolder = null;
    this.servedDatasets = null;  // mounts listed by serve.py's datasets.json (null: no server)
    this.plySelectFolder = null;
    this.camSlider   = null;
    this.snapCtrls   = [];
    this.sfm         = null;  // SfmPanel, loaded on first use
    this.holdTarget  = false;
    this.lastStepOut = null;
    this.expSummary  = '';    // last 'ok' / 'error' log line, shown on the side canvas
    this.mouse       = { down: false, x: 0, y: 0, sx: 0, sy: 0 };
    this.worldUp     = [0, -1, 0]; // orbit up axis, estimated from the COLMAP cameras
    this.pickedCam   = -1;    // camera shown in #frustum-thumb (-1 = none)
    this.flyAnim     = null;  // active spline fly-to animation
    this.orbitAnim   = null;  // Space re-center animation, see _recenterOrbit()
    this.lastView    = null;  // { view, center, fx, fy, cx, cy } of the last 3D frame (for picking / fly-from)
    this.thumbUrl    = null;  // object URL backing #thumb-img for dropped folders
    this.frustumLoad = false; // sequential frustum texture loader running
    this.frustumFail = new Set(); // cam indices whose image failed to load (don't retry every frame)
    this.snapRequest = null;  // 'manual' | 'auto': snapshot to write at the next frame boundary
    this.valRequest  = false; // validation pass requested for the next frame boundary
    this.lastValStep = 0;
    this.lastSnapStep = 0;
    this.drone       = new Drone();
    this.orbitKeys   = new Set();  // arrows held in orbit mode (pan)
    this.droneLive   = false;  // drone state initialized from the current view
    this.demo        = new DemoFlight();
    this.demoLive    = false;  // demo flight started on the current images
    this.lastFrameT  = performance.now();
    // Redraw only when what is drawn may have changed (_draw: its inputs as a key), so a paused, still
    // view leaves the GPU alone: drawGen counts the changes the key doesn't see (a control, an input
    // event, the wireframes); a redraw at least every REDRAW_MS catches the rest (e.g. a texture loaded)
    this.drawGen     = 0;
    this.lastDrawKey = '';
    this.lastDrawT   = 0;
    for (const ev of ['pointerdown', 'pointerup', 'wheel', 'keydown', 'keyup', 'resize']) {
      window.addEventListener(ev, () => { this.drawGen++; }, { capture: true, passive: true });
    }
  }

  async init() {
    // &nosg: force the no-subgroup shader path (A/B tests)
    const features = new URLSearchParams(location.search).has('nosg') ? [] : ['subgroups'];
    this.gpu = await initGPU(this.canvas, { alphaMode: 'premultiplied', features });
    this.log(`WebGPU subgroups: ${this.gpu.device.features.has('subgroups') ? 'on' : 'off'}`);
    this.renderer = await GaussOptRenderer.create(this.gpu.device, this.gpu.context, this.gpu.format);
    this.frames = new FramesSheet(document.getElementById('frames-sheet'), document.getElementById('BtnFrames'),
                                  names => this.sfm?.pickFrames(names));
    this.shell = new Shell(this);
    this._setupEvents();
    this._setupGui();
    // any control (the SfM panel's too): the settings into the project, once the user pauses
    for (const ev of ['input', 'change']) document.addEventListener(ev, (e) => {
      if (e.target.closest?.('.skal-gui')) this._settingsChanged();
    });
    await this._probeServer();
    // ?video= or ?photos= (comma-separated URLs) alone: the cameras come from the panel, no default dataset to load
    const q = new URLSearchParams(location.search), video = q.get('video'), photos = q.get('photos');
    if ((!video && !photos) || q.has('dataset')) await this.loadDatasetUrl(this.params.dataset_url, this.params.resume_ply);
    requestAnimationFrame(() => this.frame());
    if (video) this.openVideoPanel(video);
    else if (photos) this.openPhotoUrls(photos.split(',').filter(Boolean));
  }

  // Controls in three places: the Project phase (example datasets), the Train phase (checkpoint,
  // optimizer, growth, evaluation, snapshots) and the view popover (visualization, camera).
  _setupGui() {
    const P = this.params;
    const gui = mkGui(document.getElementById('gui-container'), { draggable: false, tips: TIPS });
    const proj = mkGui(document.getElementById('project-gui'), { draggable: false, tips: TIPS });
    const view = mkGui(this.shell.viewHost, { draggable: false, tips: TIPS });
    this.guis = [proj, gui, view];
    // a param given in the URL follows its control (the SfM panel's as sfm_<name>): the URL keeps
    // reloading the page as shown
    onAnyChange((obj, key, value) => {
      this.drawGen++;
      const names = obj === this.params ? urlNamesOf(key) : obj === this.sfm?.o ? [`sfm_${key}`] : [];
      const q = names.length ? urlAfterChange(location.search, names, value, URL_FLAGS) : null;
      if (q !== null && q !== location.search) history.replaceState(null, '', `${location.pathname}${q}${location.hash}`);
    });
    const wire = () => this._refreshWireframes();

    this.projectPanel = new ProjectPanel(this, document.getElementById('project-folder'));
    this.guis.push(this.projectPanel.gui);
    this.datasetFolder = proj.addFolder('Example datasets');
    this._rebuildDatasetSelector();

    this.plySelectFolder = gui.addFolder('Checkpoint / Resume');
    this._rebuildPlySelector([]);

    const optF = gui.addFolder('Optimizer (Adam / MC / SA)');
    optF.add(P, 'opt_method', OPT_METHODS).name('method');
    optF.add(P, 'mc_noise', 0, 2e6, 1e4).name('MC Langevin λ');
    optF.add(P, 'mc_opacity_reg', 0, 0.1, 0.001).name('MC opacity L1');
    optF.add(P, 'mc_scale_reg', 0, 0.1, 0.001).name('MC scale L1');
    optF.add(P, 'mc_relocate', ['opacity', 'error']).name('MC relocate by');
    optF.add(P, 'mc_relocate_mode', ['copy', 'shape']).name('MC relocate as');
    optF.add(P, 'sa_temp', 0.01, 1.0, 0.01).name('SA temp T₀');
    optF.add(P, 'sa_cooling', 0.80, 0.999, 0.005).name('SA cooling γ');
    optF.add(P, 'sa_reheat').name('SA re-heat @ K');
    optF.add(P, 'sa_share', 0, 1, 0.05).name('SA jump share');
    optF.add(P, 'sa_size', 0, 5, 0.1).name('SA jump size');
    optF.add(P, 'ssim', 0, 1, 0.05).name('D-SSIM weight');
    optF.add(P, 'drop_rate', 0, 0.5, 0.05).name('DropGaussian rate');
    optF.add(P, 'split_min_views', 0, 5, 1).name('split: min views');
    optF.add(P, 'split_min_err', 0, 0.2, 0.005).name('E9: split min error');
    optF.add(P, 'grow_rate', 0, 0.2, 0.01).name('E9: growth rate');
    optF.add(P, 'split_in_plane').name('E9: split in plane');
    optF.add(P, 'grow_by', ['error', 'grad']).name('E9: grow by');
    optF.add(P, 'grow_grad', 0, 0.002, 0.00001).name('E9: grow grad threshold');
    optF.add(P, 'split_mode', ['error', 'shape']).name('E9: split mode');
    optF.add(P, 'mesh_snap', 0, 1, 0.05).name('E5: mesh snap');
    optF.add(P, 'mesh_snap_every', 100, 5000, 100).name('E5: snap every');
    optF.add(P, 'mesh_snap_views', 1, 10, 1).name('E5: snap min views');

    const visF = view.addFolder('Visualization');
    visF.add(P, 'visu_mode', VISU_MODES).name('splat shape');
    visF.add(P, 'color_mode', COLOR_MODES).name('color mode');
    visF.add(P, 'compare_mode', COMPARE_MODES).name('view mode');
    visF.add(P, 'split_x', 0.0, 1.0, 0.01).name('split wipe X');
    visF.add(P, 'haze', 0.2, 3.0, 0.05).name('scale multiplier');
    visF.add(P, 'show_octree').name('3D octree boxes').onChange(wire);
    visF.add(P, 'show_loose').name('show loose (1+2β)L').onChange(wire);
    visF.add(P, 'show_cameras').name('COLMAP frustums').onChange(wire);
    visF.add(P, 'show_path').name('camera path').onChange(wire);
    visF.add(P, 'show_splits').name('WLBG split axes').onChange(wire);
    visF.add(P, 'frustum_images', FRUSTUM_IMAGES).name('frustum images');
    visF.add(P, 'frustum_scale', 0.2, 4.0, 0.05).name('frustum size').onChange(wire);
    visF.add(P, 'mesh_view', MESH_VIEWS).name('mesh');
    visF.add(P, 'mesh_shaded').name('mesh shaded');
    visF.add(P, 'mesh_res', 64, 512, 32).name('mesh grid');
    visF.add(P, 'mesh_simplify', 0, 4, 0.25).name('mesh simplify (voxels)');
    visF.add(P, 'mesh_texture', MESH_TEXTURES).name('mesh texture');

    const camF = view.addFolder('Camera', { collapsed: true });
    camF.add(P, 'snap_to_colmap').name('snap to COLMAP cam').listen().onChange(v => {
      if (!v) this._syncOrbitToLastView();
      this.flyAnim = null;
      wire();
    });
    this.camSlider = camF.add(P, 'cam_idx', 0, 15, 1).name('COLMAP cam #').listen().onChange(() => {
      P.snap_to_colmap = true;
      this.flyAnim = null;
      if (this.pickedCam >= 0) this.openThumbnail(P.cam_idx);
      wire();
    });
    camF.add(P, 'path_order', ['capture', 'shortest']).name('camera path order').onChange(() => { this._applyPathOrder(); wire(); });
    camF.add(P, 'camera_mode', ['orbit', 'drone', 'demo']).name('free camera (F, G)').listen()
      .onChange(v => this._setCameraMode(v));
    camF.add(P, 'demo_speed', 0.01, 0.3, 0.01).name('demo speed (R/s)');
    camF.add(P, 'demo_wander', 0, 2, 0.05).name('demo wander');
    camF.add(P, 'fov', 20, 120, 1).name('free-view FoV');
    camF.add(P, 'radius', 0.2, 30, 0.1).name('orbit radius');

    const octF = gui.addFolder('Octree & WLBG Growth', { collapsed: true });
    octF.add(P, 'max_leaf_size', [64, 128, 256]).name('leaf N_max').onChange(v => {
      if (!this.octree || !this.trainer) return;
      this.octree.maxLeafSize = v;
      this.octree.build(this.trainer.cpuParams, this.trainer.count, this.colmap?.images);
      this.trainer._uploadSplats();
      wire();
    });
    octF.add(P, 'beta', 0.0, 0.4, 0.05).name('loose overlap β').onChange(v => {
      if (!this.octree) return;
      this.octree.beta = v;
      if (this.trainer) this.trainer.sortGen++;
      wire();
    });
    octF.add(P, 'rebalance_every', 20, 500, 10).name('rebalance K');
    octF.add(P, 'rebalance_every_cap', 0, 2000, 50).name('rebalance K at the cap');
    octF.add(P, 'enable_rebalance').name('WLBG grow/prune');
    octF.add(P, 'remove_floaters').name('octree floater kill');
    octF.add(P, 'floater_depth', ['observed', 'view']).name('near-camera carve depth');
    octF.add(P, 'floater_spare', 0, 0.05, 0.001).name('spare near SfM points');
    octF.add(P, 'enable_split').name('error-cov split');
    octF.add(P, 'enable_merge').name('overlap merge');
    octF.add(P, 'max_splats', 10000, 2000000, 10000).name('max splats');
    octF.add(P, 'train_steps', 0, 100000, 1000).name('E7: planned steps');
    octF.add(P, 'cap_lease', 0, 1, 0.05).name('E7: cap lease');
    octF.add(P, 'consolidate_frac', 0.05, 0.5, 0.05).name('E7: consolidation');
    octF.add(P, 'region_repair', [0, 0.5, 1, 2]).name('E7: stuck-region repair');
    octF.add(P, 'polish').name('E7: polish (Adam) at the end');
    octF.add(P, 'max_splat_angle', 0.005, 0.2, 0.005).name('max splat angle');
    octF.add(P, 'max_anisotropy', 2, 50, 1).name('max anisotropy');
    octF.add(P, 'opacity_reset', [0, 1000, 2000, 3000, 5000]).name('opacity reset every');
    octF.add(P, 'stipple_seed').name('E4: stipple seeding');
    octF.add(P, 'stipple_count', [500, 1000, 2000, 5000, 10000]).name('stipple seeds / K');

    const evalF = gui.addFolder('Evaluation', { collapsed: true });
    evalF.add(P, 'val_every', [0, 250, 500, 1000, 2500]).name('val PSNR every');
    evalF.add({ val: () => { this.valRequest = true; } }, 'val').name('evaluate val now');

    const snapF = gui.addFolder('Snapshots (Export: save one now)', { collapsed: true });
    this.snapCtrls = [snapF.add(P, 'autosnap_every', [0, 250, 500, 1000, 2500, 5000]).name('auto-snap every')];
  }

  // Example datasets that can load here, plus what is loaded when it is none of them
  _rebuildDatasetSelector() {
    const P = this.params, opts = {};
    for (const [label, url] of Object.entries(DATASETS)) {
      if (!url.startsWith('/datasets/') || this.servedDatasets?.includes(url)) opts[label] = url;
    }
    if (this.servedDatasets?.includes('/dataset')) opts['/dataset (serve.py argument)'] = '/dataset';
    if (!Object.values(opts).includes(P.dataset_url)) {
      opts[P.dataset_url === NOT_A_URL ? '(none: video or local folder)' : P.dataset_url] = P.dataset_url;
    }
    this.datasetFolder.clear();
    this.datasetFolder.add(P, 'dataset_url', opts).name('dataset').onChange(v => {
      if (v === NOT_A_URL) return;
      P.resume_ply = POINTS3D;
      this.loadDatasetUrl(v, POINTS3D);
    });
  }

  // What is loaded, for the header and the Project step
  projectLabel() {
    const p = this.project ? `project ${this.project.name}: ` : '';
    return p + this._loadedLabel();
  }

  _loadedLabel() {
    const c = this.colmap, kind = this.sfm?.photos ? '' : 'video ';  // a photo set's name says "N photos"
    if (c?.fromSfm) return `${kind}${this.sfm?.videoName ?? ''} (${c.images.length} cameras)`;
    if (c?.baseUrl) return `${c.baseUrl} (${c.images.length} cameras)`;
    if (c) return `local folder (${c.images.length} cameras)`;
    if (this.sfm?.videoName) return `${kind}${this.sfm.videoName}`;
    return this.trainer ? (this.params.dataset_url === '[synth]' ? 'synthetic scene' : 'splats only') : '';
  }

  // Snapshots are written via serve.py's POST /save: disable them when it isn't reachable. Its
  // datasets.json also lists the datasets it mounts (the example selector).
  async _probeServer() {
    let ok = false;
    try {
      const resp = await fetch('/datasets.json', { cache: 'no-store' });
      const j = resp.ok ? await resp.json() : null;
      ok = !!j?.datasets;
      if (ok) this.servedDatasets = [...Object.keys(j.datasets), ...(j.default ? [j.default] : [])];
    } catch {}
    this._setSnapshotsEnabled(ok);
    this._rebuildDatasetSelector();
  }

  // Snapshots go into the project's splats/, else to serve.py (POST /save)
  get snapshotsOk() { return !!this.project || !!this.serverOk; }

  _setSnapshotsEnabled(ok) {
    if (this.serverOk !== ok) {
      this.log(ok ? 'serve.py reachable: snapshots enabled'
                  : 'serve.py not reachable: snapshots disabled (saves fall back to downloads)', ok ? 'info' : 'warn');
    }
    this.serverOk = ok;
    ok = this.snapshotsOk;
    const btn = document.getElementById('BtnSnapshot');
    btn.disabled = !ok;
    btn.title = ok ? 'full training state (splats + Adam + moments + octree), resumable'
                   : 'needs a project folder (Project) or serve.py (POST /save)';
    if (!ok) {
      this.params.autosnap_every = 0;
      this.snapRequest = null;
    }
    for (const c of this.snapCtrls) c.enable(ok);
    this._refreshGui();
  }

  // { app: PROJECT_PARAMS values, sfm: the SfM panel's options }, as project.json keeps them
  settings() {
    const out = { app: Object.fromEntries(PROJECT_PARAMS.map(k => [k, this.params[k]])) };
    if (this.sfm) out.sfm = this.sfm.settings();
    return out;
  }

  // A project's saved settings (settings()); parameters given in the URL win. The SfM panel takes
  // its own when it is created (sfm_panel.js)
  applySettings(s) {
    for (const [k, v] of Object.entries(s?.app ?? {})) {
      if (PROJECT_PARAMS.includes(k) && !this.urlKeys.has(k) && typeof v === typeof this.params[k]) this.params[k] = v;
    }
    if (s?.sfm) this.sfm?.applySettings(s.sfm);
    this._refreshGui();
    this._refreshWireframes();
  }

  _settingsChanged() {
    if (!this.project) return;
    clearTimeout(this.settingsTimer);
    this.settingsTimer = setTimeout(() => {
      const s = this.settings(), text = JSON.stringify(s);
      if (text === this.savedSettings) return;
      this.savedSettings = text;
      this.project?.update({ settings: s }).catch(e => this.log(`project.json: ${e.message}`, 'warn'));
    }, SETTINGS_DELAY_MS);
  }

  // Push externally-changed params (e.g. restored from a snapshot) back into the GUI
  _refreshGui() {
    for (const g of this.guis) g.updateDisplay();
  }

  _setTraining(on) {
    if (on && !this.gtAvailable) {
      this.log('Training needs the dataset images (none loaded)', 'warn');
      on = false;
    }
    if (on && !this.params.training && this.trainer) this.trainStartStep = this.trainer.stepCount;
    this.params.training = on;
    this.trainBtn.textContent = on ? '⏸ Pause' : '▶ Train';
  }

  _rebuildPlySelector(plyList) {
    this.plySelectFolder.clear();
    const opts = { '[init from points3D.bin]': POINTS3D };
    for (const name of plyList.map(plyName)) opts[name] = name;
    this.plySelectFolder.add(this.params, 'resume_ply', opts).name('checkpoint')
      .onChange(v => this.resumeFromSelectedPly(v));
  }

  _setupEvents() {
    const $ = id => document.getElementById(id);
    const cv = this.canvas;
    cv.addEventListener('pointerdown', e => {
      this.mouse.down = true;
      this.mouse.x = this.mouse.sx = e.clientX;
      this.mouse.y = this.mouse.sy = e.clientY;
    });
    // Click (< 5 px drag) on a frustum opens its thumbnail. Runs before the window handler
    // clears mouse.down, so drags that started off-canvas are ignored.
    cv.addEventListener('pointerup', e => {
      if (!this.mouse.down || Math.hypot(e.clientX - this.mouse.sx, e.clientY - this.mouse.sy) >= 5) return;
      const hit = this._pickFrustumAt(e);
      if (hit >= 0) this.openThumbnail(hit);
    });
    window.addEventListener('pointerup', () => { this.mouse.down = false; this.drone.stick = [0, 0]; });
    // hold target: captured, so releasing off the button still ends the hold
    const tgt = $('BtnTarget');
    tgt.addEventListener('pointerdown', e => { tgt.setPointerCapture(e.pointerId); this.setHoldTarget(true); });
    for (const ev of ['pointerup', 'pointercancel', 'lostpointercapture']) tgt.addEventListener(ev, () => this.setHoldTarget(false));
    cv.addEventListener('dblclick', e => {
      const hit = this._pickFrustumAt(e);
      if (hit >= 0) this.flyToCamera(hit);
    });
    cv.addEventListener('pointermove', e => {
      if (!this.mouse.down) return;
      const dx = e.clientX - this.mouse.x;
      const dy = e.clientY - this.mouse.y;
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
      if (this.params.compare_mode === 1) {
        const rect = cv.getBoundingClientRect();
        this.params.split_x = clamp((e.clientX - rect.left) / rect.width, 0, 1);
      } else if (this.params.camera_mode === 'drone' || this.params.camera_mode === 'demo') {
        // stick = drag offset from the press point, full deflection at 120 px
        this._takeStick();
        let sx = (e.clientX - this.mouse.sx) / 120, sy = (this.mouse.sy - e.clientY) / 120;
        const l = Math.hypot(sx, sy);
        if (l > 1) { sx /= l; sy /= l; }
        this.drone.stick = [sx, sy];
      } else if (dx !== 0 || dy !== 0) {
        this._leaveColmapView();
        this.params.theta += dx * 0.35;
        this.params.phi = clamp(this.params.phi - dy * 0.35, -89, 89);
        this.orbitAnim = null;
      }
    });
    cv.addEventListener('wheel', e => {
      e.preventDefault();
      if (this.params.camera_mode === 'drone' || this.params.camera_mode === 'demo') {  // wheel = gimbal tilt
        this._takeStick();
        this.drone.tilt(e.deltaY > 0 ? 0.05 : -0.05);
        return;
      }
      this._leaveColmapView();
      this.orbitAnim = null;
      this.params.radius *= (e.deltaY > 0) ? 1.07 : 1 / 1.07;
    }, { passive: false });

    // Keys: H = help, Escape = close the help and the view popover, F = orbit / drone, G = orbit /
    // demo, drone flight keys (in demo they take over), orbit Space / arrows; ignored while typing in the GUI
    const droneKeys = new Set(['arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' ', 'e', 'q']);
    const keyName = e => e.key.toLowerCase();
    const help = $('help-overlay');
    window.addEventListener('keydown', e => {
      if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target?.tagName) || e.ctrlKey || e.metaKey || e.altKey) return;
      const key = keyName(e);
      if (key === 'h' || key === '?') { help.classList.toggle('shown'); return; }
      if (key === 'escape') {
        help.classList.remove('shown');
        $('view-popover').classList.remove('shown');
        $('BtnView').classList.remove('on');
        return;
      }
      if (key === 'f') {
        this._setCameraMode(this.params.camera_mode === 'drone' ? 'orbit' : 'drone');
        return;
      }
      if (key === 'g') {
        this._setCameraMode(this.params.camera_mode === 'demo' ? 'orbit' : 'demo');
        return;
      }
      if (this.params.camera_mode !== 'drone' && this.params.camera_mode !== 'demo') {
        if (key === ' ') { e.preventDefault(); this._recenterOrbit(); }
        if (key.startsWith('arrow')) {  // pan, see _panOrbit()
          e.preventDefault();
          this._leaveColmapView();
          this.orbitAnim = null;
          this.orbitKeys.add(key);
        }
        return;
      }
      if (!droneKeys.has(key)) return;
      e.preventDefault();
      this._takeStick();
      this.drone.keys.add(key);
    });
    window.addEventListener('keyup', e => { this.drone.keys.delete(keyName(e)); this.orbitKeys.delete(keyName(e)); });
    window.addEventListener('blur', () => { this.drone.keys.clear(); this.orbitKeys.clear(); this.drone.stick = [0, 0]; });

    $('thumb-close').addEventListener('click', () => this.closeThumbnail());
    $('thumb-fly').addEventListener('click', () => { if (this.pickedCam >= 0) this.flyToCamera(this.pickedCam); });

    this.trainBtn.addEventListener('click', () => this._setTraining(!this.params.training));
    $('BtnStep').addEventListener('click', async () => {
      if (!this.trainer || !this.gtAvailable) return;
      this.lastStepOut = await this.trainer.step(this.params.snap_to_colmap ? this.params.cam_idx : null);
      this._refreshWireframes();
    });
    $('BtnExportPly').addEventListener('click', () => this.savePlyCheckpoint());
    $('BtnExportOctree').addEventListener('click', () => this.saveOctreeJson());
    $('BtnExportMesh').addEventListener('click', () => this.saveMesh());
    $('BtnSnapshot').addEventListener('click', () => this.requestSnapshot());
    $('log-clear').addEventListener('click', () => this.logPanel.clear());

    $('FolderElmt').addEventListener('change', e => {
      if (e.target.files.length > 0) this.loadDroppedFolder(e.target.files);
    });
    $('PlyElmt').addEventListener('change', e => {
      const f = e.target.files[0];
      if (f) /\.(glb|mesh\.ply)$/i.test(f.name) ? this.loadMeshFile(f) : this.loadPlyOrOctreeFile(f);
    });
    $('BtnVideo').addEventListener('click', () => this.openVideoPanel());

    const stop = e => { e.preventDefault(); e.stopPropagation(); };
    document.addEventListener('dragenter', stop);
    document.addEventListener('dragover', stop);
    document.addEventListener('drop', e => {
      stop(e);
      const files = e.dataTransfer.files;
      if (!files || files.length === 0) return;
      const n = files[0].name;
      if (files.length === 1 && (files[0].type.startsWith('video/') || /\.(mp4|mov|m4v|gosfm)$/i.test(n))) {
        this.openVideoPanel(files[0]);
      } else if (files.length > 1 && [...files].every(f => /^image\/jpeg$/.test(f.type) || /\.jpe?g$/i.test(f.name))) {
        this.openVideoPanel([...files]);  // photos, not a COLMAP folder (DESIGN-sfm §13)
      } else if (files.length === 1 && /\.(glb|mesh\.ply)$/i.test(n)) {
        this.loadMeshFile(files[0]);
      } else if (files.length === 1 && (/\.(ply|splat|json)$/i.test(n) || isSnapshotName(n))) {
        this.loadPlyOrOctreeFile(files[0]);
      } else {
        this.loadDroppedFolder(files);
      }
    });
  }

  // Free camera: 'orbit' (drag = orbit, wheel = zoom), 'drone' (src/drone.js) or 'demo' (a slow
  // flight along the capture path, src/demo.js). Switching keeps the current pose.
  _setCameraMode(mode) {
    if (mode === 'drone') {
      this.droneLive = false;
      this._droneTakeover();
      this.log('Drone mode: drag = pitch/roll, ↑/↓ = forward/back, ←/→ = yaw, E/Q = climb/descend, Space = re-center, wheel = tilt, F = orbit, H = help');
    } else if (mode === 'demo') {
      this.demoLive = false;  // starts from the pose on screen at the next frame
      this.flyAnim = null;
      this.params.snap_to_colmap = false;
      this.log('Demo mode: a slow flight along the capture path; drag, wheel or a flight key takes over (drone), G = orbit');
    } else if (this.params.camera_mode !== 'orbit' && !this.params.snap_to_colmap && !this.flyAnim) {
      this._syncOrbitToLastView();
    }
    this.params.camera_mode = mode;
    this.drone.keys.clear();
    this.drone.stick = [0, 0];
  }

  // Drone input in drone or demo mode: the user takes the stick (demo -> drone)
  _takeStick() {
    if (this.params.camera_mode === 'demo') this._setCameraMode('drone');
    this._droneTakeover();
  }

  // Drone input takes over from a snapped / in-flight view, starting from the pose on screen
  _droneTakeover() {
    if (this.droneLive && !this.flyAnim && !this.params.snap_to_colmap) return;
    if (this.lastView) this.drone.resetFromView(this.lastView.view, this.lastView.center, this.worldUp);
    this.flyAnim = null;
    this.params.snap_to_colmap = false;
    this.droneLive = !!this.lastView;
    this._refreshWireframes();
  }

  // Space in orbit mode: ease the orbit back to the scene center, default distance and
  // elevation (heading kept), over 0.6 s. Mouse orbit / zoom cancels it.
  _recenterOrbit() {
    if (!this.octree) return;
    this._leaveColmapView();
    const P = this.params;
    this.orbitAnim = {
      t0: performance.now(), dur: 600,
      from: { target: [...P.target], radius: P.radius, phi: P.phi },
      to:   { target: [...this.octree.sceneCenter], radius: this.octree.sceneRadius * 1.4, phi: 15 },
    };
  }

  // Arrows in orbit mode: pan the orbit target (and camera) along the screen's right axis
  // (Left / Right) and world up (Up / Down), at half the orbit radius per second.
  _panOrbit(dt) {
    const k = this.orbitKeys, P = this.params;
    if (k.size === 0 || !this.lastView) return;
    const v = this.lastView.view, right = [v[0], v[4], v[8]], U = this.worldUp;
    const sx = (k.has('arrowright') ? 1 : 0) - (k.has('arrowleft') ? 1 : 0);
    const sy = (k.has('arrowup') ? 1 : 0) - (k.has('arrowdown') ? 1 : 0);
    const d = 0.5 * P.radius * dt;
    P.target = P.target.map((t, i) => t + d * (sx * right[i] + sy * U[i]));
  }

  _stepOrbitAnim() {
    const a = this.orbitAnim, P = this.params;
    if (!a) return;
    const t = Math.min(1, (performance.now() - a.t0) / a.dur), s = t * t * (3 - 2 * t);
    P.target = a.from.target.map((v, k) => v + (a.to.target[k] - v) * s);
    P.radius = a.from.radius * Math.pow(a.to.radius / a.from.radius, s);  // geometric: even zoom speed
    P.phi    = a.from.phi + (a.to.phi - a.from.phi) * s;
    if (t >= 1) this.orbitAnim = null;
  }

  // Orbit input takes over from a snapped / in-flight view. Orbit params are only kept up to
  // date in free view, so re-derive them from the pose on screen to avoid jumping back.
  _leaveColmapView() {
    if (!this.flyAnim && !this.params.snap_to_colmap) return;
    this._syncOrbitToLastView();
    this.flyAnim = null;
    this.params.snap_to_colmap = false;
    this._refreshWireframes();
  }

  _syncOrbitToLastView() {
    if (!this.lastView) return;
    const { view, center, fy } = this.lastView;
    this._syncOrbitToPose(center, [view[2], view[6], view[10]], fy);
  }

  // Orbit params (target, radius, theta, phi, fov) for a camera at `center` looking along
  // `forward` with focal `fyPx` (canvas pixels). Roll is dropped: the orbit camera is level.
  _syncOrbitToPose(c, f, fyPx) {
    const sc = this.octree.sceneCenter;
    const d = Math.max(0.1 * this.octree.sceneRadius, dot(sub(sc, c), f));
    const target = madd(c, f, d);
    const orb = orbitAnglesFromCenter(target, c, this.worldUp);
    this.params.target = target;
    this.params.radius = orb.radius;
    this.params.phi    = clamp(orb.phi, -89, 89);
    this.params.theta  = orb.theta;
    this.params.fov    = clamp(2 * Math.atan((0.5 * this.canvas.height) / fyPx) * 180 / Math.PI, 20, 120);
  }

  // Pinhole intrinsics of the free (orbit / drone) camera, from params.fov (vertical)
  _freeIntrinsics(W, H) {
    const fy = (0.5 * H) / Math.tan((this.params.fov * Math.PI) / 360);
    return { fx: fy, fy, cx: 0.5 * W, cy: 0.5 * H };
  }

  _pickFrustumAt(e) {
    const imgs = this.colmap?.images;
    if (!this.lastView || !imgs || imgs.length === 0) return -1;
    if (!this.params.show_cameras || this.params.compare_mode !== 0 || this.holdTarget) return -1;
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return -1;
    const px = (e.clientX - rect.left) * (this.canvas.width  / rect.width);
    const py = (e.clientY - rect.top)  * (this.canvas.height / rect.height);
    const { view, fx, fy, cx, cy } = this.lastView;
    const hit = this.renderer.pickFrustum(px, py, view, fx, fy, cx, cy);
    return (hit >= 0 && hit < imgs.length) ? hit : -1;
  }

  // Camera highlighted in green (picked thumbnail > snapped view)
  _activeCamIdx() {
    if (this.pickedCam >= 0) return this.pickedCam;
    return this.params.snap_to_colmap ? this.params.cam_idx : -1;
  }

  _revokeThumbUrl() {
    if (this.thumbUrl) { URL.revokeObjectURL(this.thumbUrl); this.thumbUrl = null; }
  }

  openThumbnail(idx) {
    const im = this.colmap?.images?.[idx];
    if (!im) return;
    this.pickedCam = idx;
    this.params.cam_idx = idx;
    this._revokeThumbUrl();
    if (im.fileBlob) this.thumbUrl = URL.createObjectURL(im.fileBlob);
    document.getElementById('thumb-img').src = this.thumbUrl || im.imageUrl || '';
    const title = document.getElementById('thumb-title');
    title.textContent = `#${idx} ${im.name}`;
    title.title = im.name;
    document.getElementById('thumb-res').textContent = `${im.camera.width}×${im.camera.height}`;
    const tag = document.getElementById('thumb-tag');
    tag.textContent = im.isValidation ? 'Val' : 'Train';
    tag.className = `thumb-tag ${im.isValidation ? 'val' : 'train'}`;
    document.getElementById('frustum-thumb').style.display = 'block';
    this._refreshWireframes();
    if (this.colmap?.fromSfm) this.sfm?.picked(im.name);  // selects that keyframe in the panel
  }

  closeThumbnail() {
    this.pickedCam = -1;
    document.getElementById('frustum-thumb').style.display = 'none';
    document.getElementById('thumb-img').removeAttribute('src');
    this._revokeThumbUrl();
    this._refreshWireframes();
  }

  flyToCamera(idx) {
    const im = this.colmap?.images?.[idx];
    if (!im || !this.lastView || !this.octree) return;
    this.params.cam_idx = idx;
    this.params.snap_to_colmap = false;
    const { view, center, fx, fy, cx, cy } = this.lastView;
    this.flyAnim = createCameraSplineAnimation({
      fromView: view, fromCenter: center,
      fromFx: fx, fromFy: fy, fromCx: cx, fromCy: cy,
      toImg: im, W: this.canvas.width, H: this.canvas.height,
      sceneCenter: this.octree.sceneCenter,
    });
    this.flyAnim.camIdx = idx;
    this._refreshWireframes();
  }

  // End of fly-to: snap exactly into the COLMAP view, with orbit params matching it
  // so the next drag continues orbiting from (approximately) that pose.
  _finishFly(idx) {
    this.flyAnim = null;
    const im = this.colmap?.images?.[idx];
    if (!im) return;
    this._syncOrbitToPose(im.center, im.forward, fitIntrinsics(im.camera, this.canvas.width, this.canvas.height).fy);
    this.params.cam_idx = idx;
    this.params.snap_to_colmap = true;
    this._refreshWireframes();
  }

  // [camIdx, GPUTexture] pairs for in-frustum images; missing textures are loaded in the background
  _frustumTextures(imgs) {
    const mode = this.params.frustum_images;
    if (mode === 'off' || !this.params.show_cameras || imgs.length === 0) return null;
    let want;
    if (mode === 'all') {
      want = imgs.map((_, i) => i);
    } else {
      const a = this._activeCamIdx();
      want = [a >= 0 ? a : Math.min(this.params.cam_idx | 0, imgs.length - 1)];
    }
    // Don't draw a camera's own image plane while looking through it
    const inside = this.params.snap_to_colmap ? this.params.cam_idx : -1;
    const out = [], missing = [];
    for (const i of want) {
      if (i === inside || this.frustumFail.has(i)) continue;
      const tex = this.trainer.gtTextures.get(i);
      if (tex) out.push([i, tex]);
      else missing.push(i);
    }
    if (missing.length > 0 && !this.frustumLoad) this._loadFrustumTextures(missing);
    return out;
  }

  async _loadFrustumTextures(idxs) {
    this.frustumLoad = true;
    const trainer = this.trainer;
    try {
      for (const i of idxs) {
        if (this.trainer !== trainer || this.params.frustum_images === 'off') break;
        const ok = await trainer.ensureGtTexture(i).then(t => !!t, () => false);
        if (!ok && this.trainer === trainer) this.frustumFail.add(i);
      }
    } finally {
      this.frustumLoad = false;
    }
  }

  setHoldTarget(v) {
    this.holdTarget = v;
    if (v && this.colmap?.images?.length > 0) this.trainer?.ensureGtTexture(this.params.cam_idx);
  }

  // Append to the log (src/ui/log.js). level: 'info' | 'ok' | 'warn' | 'error'; source: 'app' |
  // 'sfm' | 'train'. 'ok' / 'error' results also become the telemetry summary line on the side canvas.
  log(msg, level = 'info', source = 'app') {
    (level === 'error' ? console.warn : console.log)(`[Gauss-Opt ${source}]`, msg);
    if (level === 'ok' || level === 'error') this.expSummary = (level === 'error') ? `ERROR: ${msg}` : msg;
    this.logPanel.add(msg, level, source);
  }

  // The logged messages of one source ('app', 'sfm', 'train'; all when null), one per line
  logText(source = null) { return this.logPanel.text(source); }

  showError(msg) {
    this.errorEl.textContent = `⚠ ${msg}`;
    this.errorEl.style.display = 'block';
    this.log(msg, 'error');
  }

  clearError() {
    this.errorEl.textContent = '';
    this.errorEl.style.display = 'none';
  }

  _onProgress = msg => { this.statusEl.textContent = msg; this.log(msg); };

  // Loads (dataset, folder, SfM model, checkpoint) may overlap: each takes a generation number
  // (++loadGen) and gives up after any await once a newer load started. The new model becomes
  // visible (colmap, octree, trainer) all at once, in _initOctreeAndTrainer().
  _stale(gen) { return gen !== this.loadGen; }

  async loadDatasetUrl(url, plyRel = POINTS3D) {
    const gen = ++this.loadGen;
    try {
      this.clearError();
      this._setTraining(false);
      if (url === '[synth]') {
        const splats = createSyntheticScene();
        if (await this._initOctreeAndTrainer(splats, null, gen)) {
          this.statusEl.textContent = `Loaded synthetic scene (${splats.count} splats)`;
        }
        return;
      }
      this._onProgress(`Loading COLMAP dataset from ${url}...`);
      const colmap = await loadColmapFromUrl(url, { onProgress: this._onProgress, valSplit: this.params.val_split });
      if (this._stale(gen)) return;
      if (plyRel && plyRel !== POINTS3D) {
        await this.resumeFromSelectedPly(plyRel, colmap, gen);
      } else {
        await this._initFromPoints3D(colmap, gen);
      }
    } catch (e) {
      if (this._stale(gen)) return;
      if (!this.serverOk && url.startsWith('/dataset')) {
        // static hosting (e.g. GitHub Pages): the /datasets mounts only exist under serve.py
        this.log(`No dataset server: showing a synthetic scene. Use "load gs/ folder", or run gaussopt/serve.py`, 'warn');
      } else {
        this.showError(`Dataset load failed (${url}): ${e.message}`);
      }
      await this._initOctreeAndTrainer(createSyntheticScene(), null, gen);
    }
  }

  // The "Video → Cameras" panel (src/ui/sfm_panel.js, DESIGN-sfm §7), loaded on first use.
  // file: File | URL string (?video=) | [File] (photos) | null (choose in the panel)
  async openVideoPanel(file = null) {
    try {
      const { SfmPanel } = await import('./ui/sfm_panel.js');
      this.sfm ??= new SfmPanel(this);
      this.shell.show('video');
      if (typeof file === 'string') {
        this._onProgress(`Fetching video ${file}...`);
        file = await fetchFile(file, 'video.mp4');
      }
      this.sfm.open(file);
    } catch (e) {
      this.showError(`Video panel: ${e.message}`);
    }
  }

  // ?photos=: fetched as Files, then opened as a photo set (DESIGN-sfm §13)
  async openPhotoUrls(urls) {
    try {
      this._onProgress(`Fetching ${urls.length} photos...`);
      const files = await Promise.all(urls.map(u => fetchFile(u, 'photo.jpg', 'image/jpeg')));
      await this.openVideoPanel(files);
    } catch (e) {
      this.showError(`Photos: ${e.message}`);
    }
  }

  async loadDroppedFolder(fileList) {
    const gen = ++this.loadGen;
    try {
      this.clearError();
      const colmap = await loadColmapFromFiles(fileList, { onProgress: this._onProgress, valSplit: this.params.val_split });
      if (!this._stale(gen)) await this._initFromPoints3D(colmap, gen);
    } catch (e) {
      if (!this._stale(gen)) this.showError(`Folder load error: ${e.message}`);
    }
  }

  // A parsed dataset (loadColmapFromMemory / ...Files / ...Url) into the trainer, splats initialized
  // from its points3D (src/ui/sfm_panel.js "Send to trainer"). label: prefix of the result log line.
  // Resolves true once installed, false if a newer load superseded it; throws on failure.
  async loadColmap(colmap, { label = '' } = {}) {
    const gen = ++this.loadGen;
    this.clearError();
    return this._initFromPoints3D(colmap, gen, label);
  }

  // A reopened project's dataset (src/ui/project_panel.js), from its snapshot `buf` (named `name`)
  // when there is one, else from its points
  async loadProjectModel(colmap, buf, name) {
    const gen = ++this.loadGen;
    this.clearError();
    if (buf) await this.restoreSnapshot(buf, name, colmap, gen);
    else await this._initFromPoints3D(colmap, gen, 'Project');
  }

  // Reflect the loaded dataset / checkpoint in the URL (?dataset=...&ply=...) so the page can be
  // reloaded or shared as-is. Other params are kept. Dropped folders / files have no URL form,
  // so those drop the params; a failed dataset load leaves the URL as typed.
  _syncUrl() {
    let dataset = null, ply = null;
    if (this.colmap?.baseUrl) {
      dataset = this.colmap.baseUrl;
      if (!this.plyFromFile && this.params.resume_ply !== POINTS3D) ply = this.params.resume_ply;
    } else if (this.colmap || this.params.dataset_url === '[synth]') {
      dataset = this.colmap ? null : '[synth]';
    } else {
      return;
    }
    const enc = v => encodeURIComponent(v).replace(/%2F/g, '/');
    const rest = location.search.slice(1).split('&')
      .filter(kv => kv && !/^(dataset|ply)(=|$)/.test(kv));
    const parts = [];
    if (dataset) parts.push(`dataset=${enc(dataset)}`);
    if (ply) parts.push(`ply=${enc(ply)}`);
    const q = [...parts, ...rest].join('&');
    history.replaceState(null, '', `${location.pathname}${q ? '?' + q : ''}${location.hash}`);
  }

  _newOctree() {
    return new LooseOctree({ maxLeafSize: this.params.max_leaf_size, beta: this.params.beta });
  }

  async _initFromPoints3D(colmap = this.colmap, gen = ++this.loadGen, label = '') {
    label = label ? `${label}: ` : '';
    const octree = this._newOctree();
    const splats = octree.initSplatsFromPoints3D(colmap.points3D, colmap.images, { aniso: this.params.aniso_init });
    if (!await this._initOctreeAndTrainer(splats, colmap, gen, { octree, ply: POINTS3D })) return false;
    if (colmap.points3D.fallback) {
      this.log(`${label}No sparse/points3D.bin: init from ${splats.count} random points in the camera frustums`, 'warn');
      return true;
    }
    const rep = colmap.stats.meanReprojError.toFixed(2);
    this.log(`${label}COLMAP verified: ${colmap.images.length} cams, ${splats.count} pts, mean reproj err = ${rep} px`, 'ok');
    return true;
  }

  // Parse a .ply / .splat buffer and restore its embedded octree, if any
  _parseSplats(buf, name, images) {
    const splats = name.toLowerCase().endsWith('.splat') ? parseSplatBuffer(buf) : parsePlyBuffer(buf);
    let octree = null;
    if (splats.octreeData) {
      const t0 = performance.now();
      octree = LooseOctree.fromSerializedNodes(splats.octreeData.meta, splats.octreeData.nodes);
      if (images.length > 0) octree.updateCameraAndNeighborStats(images);
      this.log(`Restored embedded octree (${octree.leaves.length} leaves) in ${(performance.now() - t0).toFixed(2)} ms`);
    }
    return { splats, octree };
  }

  // Load a checkpoint (.ply / .splat / .gosnap) of `colmap` (default: the current dataset), by
  // relative path. On failure the previous model stays, unless it was of another dataset.
  async resumeFromSelectedPly(plyRel, colmap = this.colmap, gen = ++this.loadGen) {
    if (!colmap) return;
    this.clearError();
    if (plyRel === POINTS3D) {
      await this._initFromPoints3D(colmap, gen);
      return;
    }
    try {
      const found = colmap.availablePlys.find(p => p.file && plyName(p) === plyRel);
      if (found) {  // dropped folder
        await this.loadPlyOrOctreeFile(found.file, colmap, gen, plyRel);
        return;
      }
      const rel = plyRel.replace(/^\//, '');
      const known = colmap.availablePlys.map(plyName);
      if (known.length > 0 && !known.includes(rel)) {
        throw new Error(`File "${rel}" not found in dataset (available: ${known.join(', ')})`);
      }
      const url = `${colmap.baseUrl}/${rel}`;
      this._onProgress(`Fetching checkpoint ${url}...`);
      const buf = await (await fetchOk(url)).arrayBuffer();
      if (this._stale(gen)) return;
      if (isSnapshotName(rel)) {
        await this.restoreSnapshot(buf, rel, colmap, gen, { ply: plyRel });
        return;
      }
      const { splats, octree } = this._parseSplats(buf, rel, colmap.images);
      if (!await this._initOctreeAndTrainer(splats, colmap, gen, { octree, ply: plyRel })) return;
      const octMsg = octree ? ` [pre-built octree: ${octree.leaves.length} leaves]` : '';
      this.log(`Resumed ${plyRel} (${splats.count} splats, ${colmap.images.length} cams)${octMsg}`, 'ok');
    } catch (e) {
      if (this._stale(gen)) return;
      let msg = 'previous model kept';
      if (!this.trainer || colmap !== this.colmap) {
        msg = 'using points3D.bin';
        await this._initFromPoints3D(colmap, gen).catch(e2 => { msg = `points3D.bin init failed too: ${e2.message}`; });
      } else {
        this.params.resume_ply = this.loadedPly;
      }
      this._refreshGui();
      this.showError(`Resume failed for "${plyRel}": ${e.message} (${msg})`);
    }
  }

  // Local file: .json (standalone octree for the current splats), .gosnap, .ply or .splat.
  // plyRel: its name in the checkpoint selector (dropped folders), null for a picked / dropped file.
  // A mesh file (Save mesh's .glb, or a .mesh.ply) shown in the 3D view; not a load (no loadGen)
  async loadMeshFile(file) {
    try {
      const buf = await file.arrayBuffer(), mesh = /\.glb$/i.test(file.name) ? readGlb(buf) : meshFromPly(buf);
      if (mesh.texture) mesh.image = await decodeImage(mesh.texture);
      this.showMesh(mesh);
      this.log(`Mesh ${file.name}: ${mesh.positions.length / 3} vertices, ${mesh.faces.length / 3} faces (Visualization > mesh)`, 'ok');
    } catch (e) {
      this.showError(`Failed to load mesh "${file.name}": ${e.message}`);
    }
  }

  async loadPlyOrOctreeFile(file, colmap = this.colmap, gen = ++this.loadGen, plyRel = null) {
    const name = file.name, lname = name.toLowerCase();
    try {
      this.clearError();
      this._onProgress(`Reading ${name}...`);
      if (lname.endsWith('.json')) {
        const oct = LooseOctree.fromJSON(await file.text());
        if (this._stale(gen)) return;
        if (this.colmap?.images) oct.updateCameraAndNeighborStats(this.colmap.images);
        this.octree = oct;
        this.drawGen++;
        this._loggedStats = oct.lastStats;
        if (this.trainer) {
          this.trainer.octree = oct;
          this.trainer.sortGen++;
        }
        this._refreshWireframes();
        this.log(`Loaded saved LooseOctree from ${name} (${oct.leaves.length} leaves, β=${oct.beta})`, 'ok');
        return;
      }
      const buf = await file.arrayBuffer();
      if (this._stale(gen)) return;
      const opts = { ply: plyRel ?? undefined, fromFile: true };  // no URL form for local files
      if (isSnapshotName(name)) {
        await this.restoreSnapshot(buf, name, colmap, gen, opts);
        return;
      }
      const { splats, octree } = this._parseSplats(buf, name, colmap?.images || []);
      if (!await this._initOctreeAndTrainer(splats, colmap, gen, { ...opts, octree })) return;
      this.log(`Loaded ${name} (${splats.count} splats${octree ? ', embedded octree' : ''})`, 'ok');
    } catch (e) {
      if (this._stale(gen)) return;
      if (plyRel) throw e;  // resumeFromSelectedPly() reports it
      this.showError(`Failed to load "${name}": ${e.message}`);
    }
  }

  // Build the trainer of a model, then make it current with `colmap` (null for the synthetic scene)
  // unless load `gen` is stale. Returns whether it was installed. octree: pre-built (else built
  // here), snapshot: optimizer state to restore, ply: checkpoint selector value of this model
  // (undefined: unchanged), fromFile: loaded from a local file (no URL form).
  async _initOctreeAndTrainer(splats, colmap, gen, { octree = null, snapshot = null, ply, fromFile = false } = {}) {
    const images = colmap?.images || [];
    if (!octree) {
      octree = this._newOctree();
      octree.build(splats.params, splats.count, images);
    }
    const trainer = await GaussOptTrainer.create(this.gpu.device, octree, splats, images, this.params);
    if (this._stale(gen)) { trainer.destroy?.(); return false; }
    // Restore optimizer state before the trainer becomes visible to frame() / step()
    if (snapshot) trainer.applyTrainingState(snapshot.header.state, snapshot.blobs);
    trainer.points3D = colmap?.points3D || null;

    if (colmap !== this.colmap) this._rebuildPlySelector(colmap?.availablePlys || []);
    this.colmap = colmap;
    if (colmap?.images?.length) this.shell.shapeCanvas(colmap.images[0].camera);
    if (colmap && !colmap.baseUrl) this.params.dataset_url = NOT_A_URL;
    this._rebuildDatasetSelector();
    if (colmap) this.shell.advance('train');
    if (this.trainer) this.retired.push(this.trainer);
    this.trainer = trainer;
    this.drawGen++;
    this.pacer   = new FramePacer();  // another scene: another GPU time per step
    this.octree  = octree;
    if (ply !== undefined) this.params.resume_ply = ply;
    this.loadedPly = this.params.resume_ply;
    this.plyFromFile = fromFile;
    this._refreshGui();
    this._loggedStats = octree.lastStats;  // don't report the initial stats as a rebalance
    this.worldUp = estimateWorldUp(images);
    if (images.length > 0) this.log(`World up (from ${images.length} cams): [${this.worldUp.map(v => v.toFixed(3)).join(', ')}]`);
    this.params.target = [...octree.sceneCenter];
    this.params.radius = octree.sceneRadius * 1.4;
    this.lastView  = null;
    this.droneLive = false;
    this.orbitAnim = null;
    this._onImagesChanged(images);
    this.lastSnapStep = this.lastValStep = this.trainStartStep = trainer.stepCount;
    this._syncUrl();

    // Without the photos (e.g. a sparse/ model only) the geometry is still viewable: cameras,
    // path, points. Training, compare views and "hold target" need them.
    this.gtAvailable = false;
    const gt = images.length > 0 && await trainer.ensureGtTexture(0).then(t => !!t, () => false);
    if (this.trainer !== trainer) return true;
    this.gtAvailable = gt;
    if (images.length > 0 && !gt) {
      this.log(`Training images not found (${images[0].imageUrl || images[0].name}): view only, training disabled`, 'warn');
    }
    if (this.autoTrain) {
      this.autoTrain = false;
      this._setTraining(true);
    }
    return true;
  }

  // Cameras of a repaired SfM model for the running trainer, splats kept (src/ui/sfm_panel.js has
  // aligned them to the splats' frame). keepTextures: the training photos are unchanged.
  async swapCameras(colmap, { keepTextures }) {
    const trainer = this.trainer;
    const r = await trainer.replaceImages(colmap.images, { keepTextures });
    if (this.trainer !== trainer) throw new Error('another model was loaded meanwhile');
    this.colmap = colmap;
    this.drawGen++;
    trainer.points3D = colmap.points3D;
    this._onImagesChanged(colmap.images);
    return r;
  }

  // The trainer's cameras changed (new model or swapCameras): camera slider range, picked /
  // in-flight camera, and per-camera state (lastStepOut.gtTex may be a destroyed texture).
  _onImagesChanged(images) {
    this._applyPathOrder();
    this._offerShortestPath();
    const maxIdx = Math.max(0, images.length - 1);
    this.params.cam_idx = Math.min(this.params.cam_idx | 0, maxIdx);
    this.camSlider?.max(maxIdx);
    this.flyAnim = null;
    this.demoLive = false;  // the demo path follows the images
    this.lastStepOut = null;
    this.frustumFail.clear();
    this.closeThumbnail();  // also refreshes the wireframes
  }

  // The camera path (DESIGN-sfm §13.3 P7): colmap.path, an order of the (name-sorted) images that the
  // path line and the demo flight follow; path_order 'capture' (name order) or 'shortest' (path.js,
  // up to PATH_MAX cameras). The names, the validation split and the training do not change.
  _applyPathOrder() {
    const c = this.colmap;
    if (!c?.images?.length) return;
    c.path = c.images.map((_, i) => i);
    if (this.params.path_order === 'shortest') {
      if (c.images.length > PATH_MAX) this.log(`camera path: shortest only up to ${PATH_MAX} cameras (${c.images.length}): capture order`, 'warn');
      else c.path = shortestPath(pathViews(c.images), { rotWeight: PATH_ROT_WEIGHT });
    }
    this.demoLive = false;  // the demo path follows it
  }

  // A photo set whose capture order goes back and forth (cuisine: 2.1x the shortest path's length):
  // the shortest path is offered when PATH_OFFER times shorter, once per model
  _offerShortestPath() {
    const c = this.colmap;
    if (!c?.fromSfm || !this.sfm?.photos || this.params.path_order !== 'capture' || c.images.length > PATH_MAX || c.pathOffered) return;
    c.pathOffered = true;
    const views = pathViews(c.images), capture = pathStats(c.images.map((_, i) => i), views);
    const short = pathStats(shortestPath(views, { rotWeight: PATH_ROT_WEIGHT }), views), ratio = capture.length / short.length;
    if (ratio < PATH_OFFER) return;
    this.log(`camera path: the shortest is ${ratio.toFixed(1)}x shorter than capture order (largest turn ${short.maxTurnDeg.toFixed(0)}° vs ${capture.maxTurnDeg.toFixed(0)}°): ` +
             'View ▸ Camera ▸ camera path order, for the path line and the demo flight', 'ok');
  }

  // The images in path order (the demo flight)
  _pathImages() {
    const c = this.colmap;
    return c?.path ? c.path.map(i => c.images[i]) : (c?.images ?? []);
  }

  _refreshWireframes() {
    if (!this.renderer || !this.octree) return;
    this.drawGen++;
    this.renderer.updateWireframes({
      octree:       this.octree,
      images:       this.colmap?.images || [],
      activeCamIdx: this._activeCamIdx(),
      showOctree:   this.params.show_octree,
      showLoose:    this.params.show_loose,
      showCameras:  this.params.show_cameras,
      showPath:     this.params.show_path,
      pathOrder:    this.colmap?.path ?? null,
      frustumScale: this.params.frustum_scale,
      splitVectors: this.params.show_splits ? this.trainer?.splitVectors : null,
    });
  }

  // serve.py POST /save: writes `name` into <dataset>/splats_gaussopt/, returns { path }.
  // dataset: baseUrl of a served dataset (dropped folders / SfM models have none: download them).
  async _postSave(name, body, dataset) {
    const q = `name=${encodeURIComponent(name)}&dataset=${encodeURIComponent(dataset)}`;
    const resp = await fetch(`/save?${q}`, { method: 'POST', body });
    if (!resp.ok) {  // serve.py answers {ok: false, error} with what went wrong (e.g. permissions)
      const detail = await resp.json().then(j => j.error).catch(() => null);
      throw new Error(detail ? `serve.py could not save ${name}: ${detail}` : `POST /save -> ${resp.status} ${resp.statusText}`);
    }
    return resp.json();
  }

  // Save into the project's splats/, else via serve.py, else as a browser download
  async _saveOrDownload(bytes, name, type, okMsg) {
    if (this.project) {
      await this.project.write(`splats/${name}`, bytes);
      this.log(okMsg(`${this.project.where} splats/${name}`), 'ok');
      return;
    }
    const dataset = this.colmap?.baseUrl;
    let why = !this.serverOk ? 'no serve.py' : !dataset ? 'dataset not served by serve.py' : null;
    if (!why) {
      try {
        const j = await this._postSave(name, bytes, dataset);
        this.log(okMsg(j.path), 'ok');
        return;
      } catch (e) {
        // TypeError = no server; anything else is the server's answer
        if (e instanceof TypeError) why = 'no serve.py';
        else { this.log(e.message, 'warn'); why = 'serve.py could not save it'; }
      }
    }
    downloadBlob(new Blob([bytes], { type }), name);
    this.log(`Downloaded ${name} (${why})`, 'ok');
  }

  requestSnapshot(kind = 'manual') {
    if (!this.snapshotsOk) { this.log('Snapshot unavailable: no project folder, serve.py not reachable', 'warn'); return; }
    if (this.trainer && this.trainer.count > 0) this.snapRequest = kind;
  }

  // The full training state as a .gosnap: { buf, name, counters }
  async _encodeSnapshot(kind) {
    const trainer = this.trainer;
    const { blobs, ...counters } = await trainer.readbackTrainingState();
    const octJson = this.octree.toJSON();
    const leafIdx = octJson.leafIndices ? new Uint32Array(octJson.leafIndices) : null;
    delete octJson.leafIndices;
    const imgs = this.colmap?.images || [];
    const buf = encodeSnapshot({
      savedAt:     new Date().toISOString(),
      dataset_url: this.colmap?.baseUrl ?? null,
      numImages:   imgs.length,
      firstImage:  imgs[0]?.name || null,
      splatFloats: blobs.params.length / Math.max(1, counters.count),
      params:      Object.fromEntries(SNAPSHOT_PARAMS.map(k => [k, this.params[k]])),
      state:       counters,
      octree:      octJson,
    }, { ...blobs, leaf_idx: leafIdx });
    return { buf, counters, name: `gaussopt_snap_${kind === 'auto' ? 'latest' : pad5(counters.stepCount)}${SNAPSHOT_EXT}` };
  }

  // A snapshot into `project`'s splats/, recorded in its manifest as the one to resume from
  async writeSnapshotTo(project, kind = 'manual') {
    const { buf, name, counters } = await this._encodeSnapshot(kind);
    await project.write(`splats/${name}`, buf);
    const val = this.trainer.valHistory.at(-1);
    await project.update({ phases: { train: { step: counters.stepCount, splats: counters.count, latest: `splats/${name}`,
                                              ...(val ? { valPsnr: +val.psnr.toFixed(2), valSsim: +val.ssim.toFixed(3) } : {}) } } });
    this.log(`Snapshot @ step ${counters.stepCount} (${counters.count} splats, ${mb(buf)} MB) -> ${project.where} splats/${name}`, 'ok');
  }

  // Serialize the full training state; runs from frame() between steps
  async _writeSnapshot(kind) {
    const t0 = performance.now();
    this.log(`${kind === 'auto' ? 'Auto-snapshot' : 'Snapshot'} @ step ${this.trainer.stepCount}: reading back GPU state...`);
    if (this.project) { await this.writeSnapshotTo(this.project, kind); return; }
    const { buf, name, counters } = await this._encodeSnapshot(kind);
    const dataset = this.colmap?.baseUrl;
    if (!dataset) {  // dropped folder / SfM model: nowhere to save on the server
      await this._saveOrDownload(buf, name, 'application/octet-stream', path => `Snapshot saved to ${path}`);
      return;
    }
    this.log(`Snapshot encoded in ${(performance.now() - t0).toFixed(0)} ms (${mb(buf)} MB), uploading ${name}...`);
    // Upload in the background: only the GPU readback needs to be serialized with training
    this._uploadSnapshot(buf, name, counters, dataset).catch(e => this.showError(`Snapshot upload failed: ${e.message}`));
  }

  async _uploadSnapshot(buf, name, counters, dataset) {
    let j;
    try {
      j = await this._postSave(name, buf, dataset);
    } catch (e) {
      if (!(e instanceof TypeError)) throw e;  // TypeError = network error: serve.py went away
      this._setSnapshotsEnabled(false);
      throw new Error(`serve.py unreachable, snapshots disabled (${e.message})`);
    }
    this.log(`Snapshot @ step ${counters.stepCount} (${counters.count} splats, ${mb(buf)} MB) -> ${j.path}`, 'ok');
    if (this.colmap?.baseUrl === dataset) this._addCheckpointEntry(`splats_gaussopt/${name}`);
  }

  // Add a newly saved checkpoint to the selector (URL datasets only)
  _addCheckpointEntry(rel) {
    if (!this.colmap?.baseUrl || this.colmap.availablePlys.map(plyName).includes(rel)) return;
    this.colmap.availablePlys.push(rel);
    this.colmap.availablePlys.sort((a, b) => plyName(a).localeCompare(plyName(b)));
    this._rebuildPlySelector(this.colmap.availablePlys);
  }

  // A .gosnap of `colmap` (load generation `gen`); opts: see _initOctreeAndTrainer()
  async restoreSnapshot(buf, name, colmap, gen, opts = {}) {
    this.log(`Restoring snapshot ${name} (${mb(buf)} MB)...`);
    const snap = decodeSnapshot(buf);
    const h = snap.header;
    const imgs = colmap?.images || [];
    if (h.numImages !== imgs.length || (h.firstImage && imgs[0]?.name !== h.firstImage)) {
      throw new Error(`snapshot was taken on ${h.dataset_url ?? 'a local dataset'} (${h.numImages} cams), ` +
                      `current dataset has ${imgs.length} cams`);
    }
    const count = h.state.count | 0;
    if (!snap.blobs.params || snap.blobs.params.length !== count * SPLAT_FLOATS) {
      throw new Error(`snapshot splat layout mismatch (${h.splatFloats} floats/splat, expected ${SPLAT_FLOATS})`);
    }
    // Params given explicitly in the URL win over the snapshot's (e.g. A/B runs from one snapshot)
    for (const k of SNAPSHOT_PARAMS) {
      if (!this.urlKeys.has(k) && h.params && k in h.params) this.params[k] = h.params[k];
    }
    this._setTraining(false);
    this._refreshGui();

    const octree = LooseOctree.fromJSON({ ...h.octree, leafIndices: snap.blobs.leaf_idx || null });
    if (imgs.length > 0) octree.updateCameraAndNeighborStats(imgs);
    const splats = { count, params: snap.blobs.params, octreeData: null };
    if (!await this._initOctreeAndTrainer(splats, colmap, gen, { ...opts, octree, snapshot: snap })) return;
    this.log(`Resumed snapshot ${name} @ step ${this.trainer.stepCount} (${count} splats, ${octree.leaves.length} leaves, saved ${h.savedAt})`, 'ok');
  }

  async savePlyCheckpoint() {
    if (!this.trainer) return;
    this.log(`Saving .ply @ step ${this.trainer.stepCount} (${this.trainer.count} splats)...`);
    await this.trainer.syncSplatsFromGpu();
    const bytes = exportPlyBuffer(this.trainer.cpuParams, this.trainer.count, this.octree);
    await this._saveOrDownload(bytes, `gaussopt_${pad5(this.trainer.stepCount)}.ply`, 'application/octet-stream',
      path => `Saved PLY + embedded octree (${this.octree.leaves.length} leaves) to: ${path}`);
  }

  // DESIGN §9 E5: the rendered depth of every view fused (TSDF) and meshed (training pauses meanwhile),
  // then simplified in a worker, textured from the photos (mesh_texture; training pauses again) and
  // written as .glb; shown in the 3D view (mesh_view) and saved
  async saveMesh() {
    if (!this.trainer || this.trainer.count === 0) return;
    const P = this.params, wasTraining = P.training, step = this.trainer.stepCount;
    this._setTraining(false);
    let fused;
    try {
      this.log(`Meshing @ step ${step}: fusing the depth of ${this.trainer.images.length} views...`);
      fused = await meshFromTrainer(this.trainer, { res: P.mesh_res });
    } catch (e) {
      this.log(`Mesh failed: ${e.message}`, 'warn');
      return;
    } finally {
      if (wasTraining) this._setTraining(true);
    }
    const { mesh, info } = fused;
    this.log(`Mesh: ${info.vertices} vertices, ${info.faces} faces, grid ${info.grid.join(' x ')} (voxel ${info.voxel.toPrecision(3)}), ` +
             `${(info.ms / 1000).toFixed(1)} s; simplifying...`);
    const size = P.mesh_texture;
    const out = await new Promise((resolve) => {
      const w = new Worker(new URL('./mesh_worker.js', import.meta.url), { type: 'module' });
      w.onmessage = ({ data }) => { w.terminate(); resolve(data); };
      w.onerror = (e) => { w.terminate(); resolve({ error: e.message }); };
      w.postMessage({ mesh, maxError: P.mesh_simplify * info.voxel, up: this.worldUp, glb: !size });
    });
    if (out.error) { this.log(`Mesh simplification failed: ${out.error}`, 'warn'); return; }
    this.log(`Mesh simplified to ${out.mesh.faces.length / 3} faces (error <= ${P.mesh_simplify} voxel) in ${(out.info.ms / 1000).toFixed(1)} s` +
             (size ? `; texturing (${size} x ${size}) from the photos...` : `, ${(out.glb.length / 1e6).toFixed(1)} MB`), 'ok');
    let shown = out.mesh, glb = out.glb;
    if (size) {
      const was = P.training;
      this._setTraining(false);
      try {
        const t = await textureFromTrainer(this.trainer, out.mesh, { size, tol: info.trunc * info.voxel });
        const jpeg = await encodeJpeg(t.image, size);
        glb = writeGlb(t.mesh, { up: this.worldUp, texture: { bytes: jpeg, mimeType: 'image/jpeg' } });
        shown = { ...t.mesh, image: { data: t.image, width: size, height: size } };
        this.log(`Mesh textured: ${t.info.charts} charts, ${(100 * t.info.used).toFixed(0)}% of the texels used, ${(100 * t.info.seen).toFixed(0)}% of them seen ` +
                 `by a photo, in ${(t.info.ms / 1000).toFixed(1)} s; ${(glb.length / 1e6).toFixed(1)} MB (JPEG ${(jpeg.length / 1e6).toFixed(1)} MB)`, 'ok');
      } catch (e) {
        this.log(`Mesh texture failed (${e.message}): saved with its vertex colors`, 'warn');
        glb = writeGlb(out.mesh, { up: this.worldUp });
      } finally {
        if (was) this._setTraining(true);
      }
    }
    this.showMesh(shown);
    await this._saveOrDownload(glb, `gaussopt_${pad5(step)}.mesh.glb`, 'model/gltf-binary', path => `Saved mesh to: ${path}`);
  }

  // A mesh ({ positions, faces, colors, uvs?, image? }, render.js setMesh) into the 3D view, turned on if it was off
  showMesh(mesh) {
    this.renderer.setMesh(mesh);
    if (mesh && this.params.mesh_view === 0) { this.params.mesh_view = 1; this._refreshGui(); }
    this.drawGen++;
  }

  async saveOctreeJson() {
    if (!this.octree) return;
    const bytes = new TextEncoder().encode(JSON.stringify(this.octree.toJSON()));
    await this._saveOrDownload(bytes, `gaussopt_${pad5(this.trainer?.stepCount ?? 0)}.octree.json`, 'application/json',
      path => `Saved standalone octree (${this.octree.leaves.length} leaves) to: ${path}`);
  }

  _logRebalance() {
    const st = this._loggedStats = this.octree.lastStats;
    const tr = this.trainer;
    const extra = (st.seeded ? `, +${st.seeded} stipple` : '') + (st.teleported ? `, ${st.teleported} relocated (MC)` : '') + (st.trimmed ? `, ${st.trimmed} trimmed to the budget` : '') +
                  (tr.lastOpacityReset === tr.stepCount ? ', opacity reset' : '') +
                  (tr.lastSnapInfo?.step === tr.stepCount ? `, ${tr.lastSnapInfo.moved} snapped to the mesh (mean ${(tr.lastSnapInfo.meanShift / tr.lastSnapInfo.voxel).toFixed(2)} voxel)` : '');
    this.log(`Rebalance @ step ${tr.stepCount}: +${st.split} split, +${st.cloned} clone, ` +
             `-${st.pruned} prune, -${st.floaters} floaters${st.floaterRules ? ` (near ${st.floaterRules[0]}, few views ${st.floaterRules[1]}, empty ${st.floaterRules[2]})` : ''}, -${st.merged} merge${extra} -> ${tr.count} splats, ` +
             `${this.octree.leaves.length} leaves` + (tr.lastRebalanceMs ? ` in ${tr.lastRebalanceMs.total.toFixed(0)} ms` : ''), 'info', 'train');
    const si = tr.lastSeedInfo;
    if (si) {
      this.log(`E4 stipple seeds from cam #${si.viewIdx}: ${si.count} sites, ${st.seeded ?? 0} added (depth: ${si.covered} rendered, ` +
               `${si.fromPoints} COLMAP pts, ${si.fromMedian} median) in ${si.ms.toFixed(0)} ms`, 'info', 'train');
    }
    const gi = tr.lastGrowInfo;
    if (gi?.step === tr.stepCount) this.log(`E9 grow by grad: score p50 ${gi.p50.toExponential(2)} p90 ${gi.p90.toExponential(2)} ` +
                                            `p99 ${gi.p99.toExponential(2)}, ${gi.over} over ${this.params.grow_grad}`, 'info', 'train');
    const rp = tr.lastRepairInfo;
    if (rp && this.params.region_repair > 0) this.log(`E7 repair: ${rp.cells} stuck cells, ${rp.splats} splats`, 'info', 'train');
    const sa = tr.lastAnnealInfo;
    if (sa && this.params.opt_method === 'sim_anneal') {
      this.log(`SA @ T=${sa.temp.toFixed(4)}: ${sa.accepted} jumps kept, ${sa.rejected} undone, ${sa.proposed} new`, 'info', 'train');
    }
    this._refreshWireframes();
  }

  async _runValidation() {
    this.valRequest  = false;
    this.lastValStep = this.trainer.stepCount;
    const t0 = performance.now();
    try {
      const r = await this.trainer.evaluateValidation();
      if (r) {
        this.log(`Val PSNR @ step ${r.step}: ${r.psnr.toFixed(2)} dB (1/4 res ${r.psnr4.toFixed(2)} dB), SSIM ${r.ssim.toFixed(3)}, depth spread ${(100 * r.spread).toFixed(1)}% over ${r.views} held-out views (${this.params.val_split}) ` +
                 `(${r.count} splats, ${(performance.now() - t0).toFixed(0)} ms)`, 'ok', 'train');
        if (this.colmap?.fromSfm) this.sfm?.viewPsnr(r.perView);
      } else {
        this.log(`Val PSNR: no validation views (${this.colmap?.images.length < 8 ? 'needs >= 8 cameras' : 'every view trains'})`, 'warn', 'train');
      }
    } catch (e) {
      this.showError(`Validation failed: ${e.message}`);
    }
  }

  // One frame; errors are reported and the next frame is always scheduled
  async frame() {
    try {
      await this._frame();
    } catch (e) {
      this.showError(`Frame failed: ${e.message}`);
    }
    requestAnimationFrame(() => this.frame());
  }

  // Training steps, then (between steps) stop_after / validation / snapshot triggers, then draw
  async _frame() {
    const nowT = performance.now();
    const dtFrame = Math.min(0.05, (nowT - this.lastFrameT) / 1000);
    this.lastFrameT = nowT;
    const P = this.params;
    // Replaced trainers, also when a frame in flight still uses them: destroy() lets work already
    // submitted finish (WebGPU spec, buffer and texture destruction)
    for (const t of this.retired.splice(0)) t.destroy?.();
    const tr = this.trainer;
    let t0 = 0, steps = 0;
    if (tr && P.training) {
      // One frame in flight: the GPU runs the previous frame's work while this one is encoded.
      // More, and training dispatches pile up in the queue and every displayed frame (and
      // input response) waits behind them.
      while (this.gpuFrames.length > 1) await this.gpuFrames.shift().done;
      t0 = performance.now();
      const n = this.pacer.steps(P.steps_per_frame), s0 = tr.stepCount;
      try {
        for (let s = 0; s < n && tr === this.trainer; s++) {
          this.lastStepOut = await tr.step();
          if (performance.now() - t0 > TRAIN_BUDGET_MS) break;
        }
      } catch (e) {  // e.g. a training photo that can't be loaded: pause instead of retrying every frame
        if (tr === this.trainer) {
          this._setTraining(false);
          this.showError(`Training step failed, paused: ${e.message}`);
        }
      }
      steps = tr.stepCount - s0;
      if (tr !== this.trainer) {
        this.lastStepOut = null;  // of the replaced trainer
      } else {
        if (this.octree.lastStats !== this._loggedStats) this._logRebalance();
        const every = P.autosnap_every;
        if ((this.project || (this.serverOk && this.colmap?.baseUrl)) && every > 0 && tr.stepCount - this.lastSnapStep >= every && !this.snapRequest) {
          this.snapRequest = 'auto';
        }
        if (P.stop_after > 0 && P.training && tr.stepCount - this.trainStartStep >= P.stop_after) {
          this._setTraining(false);
          this.valRequest = true;
          this.log(`stop_after: paused @ step ${tr.stepCount} (${tr.count} splats)`, 'ok', 'train');
        }
        if (P.training && P.val_every > 0 && tr.stepCount - this.lastValStep >= P.val_every) {
          this.valRequest = true;
        }
      }
    }
    if (this.trainer && this.valRequest) await this._runValidation();
    if (this.trainer && this.snapRequest) {
      const kind = this.snapRequest;
      this.snapRequest = null;
      this.lastSnapStep = this.trainer.stepCount;
      try {
        await this._writeSnapshot(kind);
      } catch (e) {
        this.showError(`Snapshot failed: ${e.message}`);
      }
    }
    if (this.renderer && this.trainer) await this._draw(dtFrame);
    else if (this.sfm) this.statusEl.textContent = `Video → Cameras: ${this.sfm.state.status}`;
    if (nowT - (this.shellT ?? 0) > 200) { this.shellT = nowT; this.shell?.refresh(); }

    if (P.training && steps > 0) {
      const pacer = this.pacer;
      this.gpuFrames.push({ done: this.gpu.device.queue.onSubmittedWorkDone().then(() => pacer.done(t0, performance.now(), steps)) });
    } else if (!P.training) {
      this.gpuFrames = [];
    }
  }

  // The free / COLMAP camera of this frame: { view, center, intr: { fx, fy, cx, cy }, flyDone }.
  // Precedence: fly-to animation > COLMAP snap > drone > demo > orbit
  _viewCamera(dtFrame, imgs, camIdx, W, H) {
    const P = this.params;
    if (this.flyAnim && imgs[this.flyAnim.camIdx]) {
      const st = evaluateCameraSpline(this.flyAnim);
      return { view: st.view, center: st.camCenter, intr: st, flyDone: st.done };
    }
    if (P.snap_to_colmap && imgs[camIdx]) {
      const im = imgs[camIdx];
      return { view: im.view, center: im.center, intr: fitIntrinsics(im.camera, W, H), flyDone: false };
    }
    if (P.camera_mode === 'drone' && this.droneLive) {
      this.drone.update(dtFrame, { sceneCenter: this.octree.sceneCenter, sceneRadius: this.octree.sceneRadius });
      return { view: this.drone.view(), center: [...this.drone.pos], intr: this._freeIntrinsics(W, H), flyDone: false };
    }
    if (P.camera_mode === 'demo' && (this.lastView || imgs.length > 0)) {
      if (!this.demoLive) {
        this.demo.start(this._pathImages(), this.lastView ?? imgs[0], this.worldUp, this.octree.sceneCenter, this.octree.sceneRadius);
        this.demoLive = true;
      }
      this.demo.speed = P.demo_speed;
      this.demo.wander = P.demo_wander;
      this.demo.update(dtFrame);
      return { view: this.demo.view(), center: [...this.demo.pos], intr: this._freeIntrinsics(W, H), flyDone: false };
    }
    this._stepOrbitAnim();
    this._panOrbit(dtFrame);
    const orb = buildOrbitViewMatrix(P.target, P.radius, P.theta, P.phi, this.worldUp);
    return { view: orb.view, center: orb.center, intr: this._freeIntrinsics(W, H), flyDone: false };
  }

  async _draw(dtFrame) {
    const P = this.params, tr = this.trainer;
    const W = this.canvas.width, H = this.canvas.height;
    const imgs = this.colmap?.images || [];
    const camIdx = Math.min(Math.max(0, P.cam_idx | 0), Math.max(0, imgs.length - 1));
    // what the frame draws, but the camera (added below): unchanged, and the frame is skipped
    const nowT = performance.now();
    const baseKey = [this.drawGen, W, H, tr.stepCount, tr.count, tr.valHistory?.length, tr.gtTextures?.size, camIdx,
                     this.holdTarget, this.gtAvailable, P.compare_mode, P.split_x].join();
    const unchanged = (key) => {
      if (key === this.lastDrawKey && nowT - this.lastDrawT < REDRAW_MS) return true;
      this.lastDrawKey = key;
      this.lastDrawT = nowT;
      return false;
    };

    if (this.holdTarget && imgs[camIdx] && this.gtAvailable) {
      if (unchanged(`hold,${baseKey}`)) return;
      const gtTex = await tr.ensureGtTexture(camIdx);
      if (gtTex) this.renderer.renderCompare({ gtTex, hatTex: tr.hatTex, pixState: tr.pixStateBuf, mode: 0 });
    } else if (P.compare_mode > 0 && imgs.length > 0 && this.gtAvailable) {
      if (unchanged(`compare,${baseKey}`)) return;
      const out = (P.training && this.lastStepOut?.gtTex) ? this.lastStepOut : await tr.renderViewForward(camIdx);
      if (out?.gtTex) {
        this.renderer.renderCompare({
          gtTex: out.gtTex, hatTex: out.hatTex, pixState: tr.pixStateBuf,
          mode: P.compare_mode, splitX: P.split_x,
        });
      }
    } else {
      const { view, center, intr, flyDone } = this._viewCamera(dtFrame, imgs, camIdx, W, H);
      const { fx, fy, cx, cy } = intr;
      const frustumTextures = this._frustumTextures(imgs);  // also starts loading the missing ones
      this.lastView = { view, center, fx, fy, cx, cy };
      const skip = unchanged(`${Array.from(view).join()},${fx},${fy},${cx},${cy},${baseKey}`);
      if (!skip) {
        const numVis = tr.sortForDisplay(center, view, 2 * fx / W, 2 * fy / H);
        this.renderer.renderScene({
          splatsBuf:      tr.splatsBuf,
          momentsBuf:     tr.momentsBuf,
          sortedIdxBuf:   tr.viewIdxBuf,
          numVisible:     numVis,
          view,
          sceneRadius:    this.octree.sceneRadius,
          fx, fy, cx, cy, W, H,
          visuMode:       P.visu_mode,
          colorMode:      P.color_mode,
          haze:           P.haze,
          avgR:           tr.avgR,
          frustumTextures,
          meshMode:       P.mesh_view,
          meshShaded:     P.mesh_shaded,
        });
      }
      if (flyDone) this._finishFly(this.flyAnim.camIdx);  // also when its last frame is skipped
      if (skip) return;
    }

    const psnrTxt = tr.lastPsnr > 0 ? `${tr.lastPsnr.toFixed(2)} dB` : '--';
    const optLabel = P.opt_method === 'sim_anneal'  ? `SA(T=${tr.currentTemp.toFixed(3)})`
                   : P.opt_method === 'monte_carlo' ? `MC(λ=${P.mc_noise})` : 'Adam';
    this.statusEl.textContent =
      `[${optLabel}] Step: ${tr.stepCount} | PSNR: ${psnrTxt} | Splats: ${tr.count} (${tr.numVisibleView} vis) | Leaves: ${this.octree.leaves.length}` +
      (P.camera_mode === 'drone' && this.droneLive ? ` | ${this.drone.hud(this.octree.sceneRadius)}` : '') +
      (P.camera_mode === 'demo' && this.demoLive ? ` | ${this.demo.hud()}` : '') +
      (this.sfm?.running || this.sfm?.busy ? ` | video: ${this.sfm.state.status}` : '');
    drawTelemetryPanel(this.sideCanvas, tr.history, this.octree, this.expSummary, tr.valHistory);
  }
}
