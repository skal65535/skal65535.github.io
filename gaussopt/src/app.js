// Gauss-Opt app: GUI + URL params, dataset / checkpoint / snapshot loading, camera modes
// (orbit, drone, COLMAP snap, spline fly-to), training loop triggers and the action log.
// Needs ../common/args.js (parse_arg*) loaded as a classic script before this module.

import { initGPU } from '../../common/webgpu.js';
import { mkGui }   from '../../common/controls.js';
import {
  SPLAT_FLOATS, rgbToSh0,
  loadColmapFromUrl, loadColmapFromFiles,
  parsePlyBuffer, parseSplatBuffer, exportPlyBuffer,
} from './colmap.js';
import { LooseOctree } from './octree.js';
import {
  GaussOptRenderer, drawTelemetryPanel, createCameraSplineAnimation, evaluateCameraSpline,
} from './render.js';
import { logit, buildOrbitViewMatrix, estimateWorldUp, orbitAnglesFromCenter, fitIntrinsics } from './math.js';
import { GaussOptTrainer } from './trainer.js';
import { Drone } from './drone.js';
import { encodeSnapshot, decodeSnapshot, isSnapshotName, SNAPSHOT_EXT } from './snapshot.js';

const VISU_MODES = { 'gaussian': 0, 'cone': 1, 'ellipsis': 2, 'small_dot': 3, 'axis': 4 };

const COLOR_MODES = {
  'learned_rgb':     0,
  'octree_leaf':     1,
  'wlbg_error (R)':  2,
  'view_div (Neff)': 3,
  'depth':           4,
};

const COMPARE_MODES = {
  '3D Free / Snap View': 0,
  'Split (GT | Render)': 1,
  'Residual |GT - Hat|': 2,
};

const FRUSTUM_IMAGES = { 'Off': 'off', 'Active Cam Only': 'active', 'All Cameras': 'all' };

const OPT_METHODS = {
  'Adam (deterministic)':      'adam',
  'Monte-Carlo (SGLD / MCMC)': 'monte_carlo',
  'Simulated Annealing (SA)':  'sim_anneal',
};

const DATASETS = {
  'jardin1_gs (16 views)':  '/datasets/jardin1_gs',
  'jardin3_gs (120 views)': '/datasets/jardin3_gs',
  'jardin3-2_gs (359 views)': '/datasets/jardin3-2_gs',
  '[synthetic ring]':       '[synth]',
};

const POINTS3D = '[points3D.bin]';  // resume_ply value: init from COLMAP points

// Short URL aliases (?dataset=...&ply=...&train&spf=4...); every param is also settable by name.
const URL_ALIASES = {
  dataset_url: 'dataset', resume_ply: 'ply', training: 'train', steps_per_frame: 'spf',
  downsample: 'ds', opt_method: 'opt', stipple_seed: 'stipple', val_every: 'val',
  autosnap_every: 'autosnap',
  max_splats: 'max_splats',  // parsed (and clamped) by parse_arg below
};

// CPU time budget per frame for training steps (steps_per_frame is the upper bound)
const TRAIN_BUDGET_MS = 20;

// Params saved in / restored from a .gosnap snapshot (optimizer + growth settings)
const SNAPSHOT_PARAMS = [
  'opt_method', 'mc_noise', 'sa_temp', 'sa_cooling', 'sa_reheat',
  'max_leaf_size', 'beta', 'rebalance_every',
  'enable_rebalance', 'enable_split', 'enable_merge', 'remove_floaters',
  'max_splats', 'prune_opacity', 'hysteresis', 'stipple_seed', 'stipple_count',
  'lr_pos', 'lr_scale', 'lr_quat', 'lr_color', 'lr_opacity', 'lr_decay_steps', 'lr_pos_final',
  'max_splat_angle', 'max_splat_scale', 'max_anisotropy', 'opacity_reset',
];

const pad5 = n => String(n).padStart(5, '0');
const mb   = buf => (buf.byteLength / 1048576).toFixed(1);

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
    this.logEl      = document.getElementById('log-body');
    this.trainBtn   = document.getElementById('BtnTrain');

    this.params = {
      dataset_url:     parse_arg_str('dataset', '/datasets/jardin1_gs'),
      resume_ply:      parse_arg_str('ply', POINTS3D),
      training:        parse_arg_bool('train'),
      steps_per_frame: parse_arg('spf', 2, 1, 20),
      downsample:      parse_arg('ds', 4, 1, 8),

      // Optimizer: 'adam' | 'monte_carlo' (SGLD/MCMC) | 'sim_anneal'
      opt_method:      parse_arg_str('opt', 'adam'),
      mc_noise:        0.08,
      sa_temp:         0.15,
      sa_cooling:      0.96,
      sa_reheat:       true,

      // Camera & view
      snap_to_colmap:  false,
      camera_mode:     'orbit',  // free camera: 'orbit' | 'drone' (src/drone.js)
      cam_idx:         0,
      fov:             60,
      radius:          3.5,
      theta:           -90,
      phi:             15,
      target:          [0, 0, 0],
      split_x:         0.5,

      // Visualization
      visu_mode:       0,
      color_mode:      0,
      compare_mode:    0,
      haze:            1.0,
      show_octree:     true,
      show_loose:      false,
      show_cameras:    true,
      show_path:       false,  // COLMAP camera path (centers in capture order)
      show_splits:     true,
      frustum_images:  'off',
      frustum_scale:   1.0,

      // Octree & WLBG growth
      max_leaf_size:   256,
      aniso_init:      true,   // init splats from the PCA of their 8 nearest COLMAP points
      beta:            0.20,
      rebalance_every: 100,
      enable_rebalance:true,
      enable_split:    true,
      enable_merge:    true,
      remove_floaters: true,
      max_splats:      parse_arg('max_splats', 300000, 1000, 2000000),
      prune_opacity:   0.008,
      opacity_reset:   3000,  // clamp all opacities to 0.01 every N steps (0 = off)
      // Shape limits (applied every Adam step): huge splats & needles are almost always wrong
      max_splat_angle: 0.03,  // max axis scale / depth, in the training view
      max_splat_scale: 0.1,   // max axis scale, as a fraction of the scene radius
      max_anisotropy:  10,    // max ratio of the largest axis to the median one
      hysteresis:      0.35,
      stipple_seed:    parse_arg_bool('stipple'), // E4: seed new splats from residual stippling
      stipple_count:   2000,

      // Evaluation: held-out (validation camera) PSNR every N training steps (0 = off)
      val_every:       parse_arg('val', 500, 0, 100000),
      stop_after:      0,  // pause training (and run validation) N steps after load (0 = never)

      // Auto-snapshot every N steps (0 = off), overwriting gaussopt_snap_latest.gosnap
      autosnap_every:  parse_arg('autosnap', 0, 0, 100000),

      // Learning rates
      lr_pos:          0.00016,
      lr_pos_final:    0.01,   // lr_pos decays exponentially to lr_pos * lr_pos_final ...
      lr_decay_steps:  30000,  // ... over this many steps
      lr_scale:        0.005,
      lr_quat:         0.002,
      lr_color:        0.0025,
      lr_opacity:      0.05,
    };
    this.urlKeys = this._applyUrlParams();
    this.autoTrain = this.params.training;  // &train: start training once the first model is loaded
    this.trainStartStep = 0;

    this.gpu         = null;
    this.renderer    = null;
    this.octree      = null;
    this.trainer     = null;
    this.colmap      = null;
    this.holdTarget  = false;
    this.lastStepOut = null;
    this.expSummary  = '';    // last 'ok' / 'error' log line, shown on the side canvas
    this.mouse       = { down: false, x: 0, y: 0, sx: 0, sy: 0 };
    this.worldUp     = [0, -1, 0]; // orbit up axis, estimated from the COLMAP cameras
    this.pickedCam   = -1;    // camera shown in #frustum-thumb (-1 = none)
    this.flyAnim     = null;  // active spline fly-to animation
    this.lastView    = null;  // { view, center, fx, fy, cx, cy } of the last 3D frame (for picking / fly-from)
    this.thumbUrl    = null;  // object URL backing #thumb-img for dropped folders
    this.frustumLoad = false; // sequential frustum texture loader running
    this.frustumFail = new Set(); // cam indices whose image failed to load (don't retry every frame)
    this.snapRequest = null;  // 'manual' | 'auto': snapshot to write at the next frame boundary
    this.valRequest  = false; // validation pass requested for the next frame boundary
    this.lastValStep = 0;
    this.lastSnapStep = 0;
    this.drone       = new Drone();
    this.droneLive   = false;  // drone state initialized from the current view
    this.lastFrameT  = performance.now();
  }

  // Apply ?name=value for every param (aliases are already parsed by parse_arg*).
  // Returns the set of param names given in the URL (they win over snapshot params).
  _applyUrlParams() {
    const args = new URLSearchParams(location.search);
    const aliases = Object.values(URL_ALIASES);
    const keys = new Set();
    for (const [k, def] of Object.entries(this.params)) {
      const alias = URL_ALIASES[k];
      if (alias && args.has(alias)) { keys.add(k); continue; }
      if (!args.has(k)) continue;
      const v = args.get(k);
      if (typeof def === 'boolean') {
        this.params[k] = !['0', 'false', 'off', 'no'].includes(v.toLowerCase());
      } else if (typeof def === 'number') {
        const x = parseFloat(v);
        if (!Number.isFinite(x)) { console.warn(`URL param ${k}=${v}: not a number`); continue; }
        this.params[k] = x;
      } else if (typeof def === 'string') {
        this.params[k] = v;
      } else if (Array.isArray(def)) {
        const a = v.split(',').map(parseFloat);
        if (a.length !== def.length || a.some(x => !Number.isFinite(x))) continue;
        this.params[k] = a;
      } else {
        continue;
      }
      keys.add(k);
    }
    for (const k of args.keys()) {
      if (!(k in this.params) && !aliases.includes(k)) console.warn(`Unknown URL param: ${k}`);
    }
    return keys;
  }

  async init() {
    // &nosg: force the no-subgroup shader path (A/B tests)
    const features = new URLSearchParams(location.search).has('nosg') ? [] : ['subgroups'];
    this.gpu = await initGPU(this.canvas, { alphaMode: 'premultiplied', features });
    this.log(`WebGPU subgroups: ${this.gpu.device.features.has('subgroups') ? 'on' : 'off'}`);
    this.renderer = await GaussOptRenderer.create(this.gpu.device, this.gpu.context, this.gpu.format);
    this._setupEvents();
    this._setupGui();
    await this._probeServer();
    await this.loadDatasetUrl(this.params.dataset_url, this.params.resume_ply);
    requestAnimationFrame(() => this.frame());
  }

  _setupGui() {
    const P = this.params;
    const gui = this.gui = mkGui(document.getElementById('gui-container'), { title: 'Gauss-Opt Controls', draggable: false });
    const wire = () => this._refreshWireframes();

    gui.add(P, 'dataset_url', DATASETS).name('dataset').onChange(v => {
      P.resume_ply = POINTS3D;
      this.loadDatasetUrl(v, POINTS3D);
    });

    this.plySelectFolder = gui.addFolder('Checkpoint / Resume');
    this._rebuildPlySelector([]);

    const optF = gui.addFolder('Optimizer (Adam / MC / SA)');
    optF.add(P, 'opt_method', OPT_METHODS).name('method');
    optF.add(P, 'mc_noise', 0.0, 0.5, 0.01).name('MC Langevin η');
    optF.add(P, 'sa_temp', 0.01, 1.0, 0.01).name('SA temp T₀');
    optF.add(P, 'sa_cooling', 0.80, 0.999, 0.005).name('SA cooling γ');
    optF.add(P, 'sa_reheat').name('SA re-heat @ K');

    const visF = gui.addFolder('Visualization');
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

    const camF = gui.addFolder('Camera', { collapsed: true });
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
    camF.add(P, 'camera_mode', ['orbit', 'drone']).name('free camera (F)').listen()
      .onChange(v => this._setCameraMode(v));
    camF.add(P, 'fov', 20, 120, 1).name('free-view FoV');
    camF.add(P, 'radius', 0.2, 30, 0.1).name('orbit radius');

    const octF = gui.addFolder('Octree & WLBG Growth', { collapsed: true });
    octF.add(P, 'max_leaf_size', [64, 128, 256]).name('leaf N_max').onChange(v => {
      if (!this.octree || !this.trainer) return;
      this.octree.maxLeafSize = Number(v);
      this.octree.build(this.trainer.cpuParams, this.trainer.count, this.colmap?.images);
      this.trainer._uploadSplats();
      wire();
    });
    octF.add(P, 'beta', 0.0, 0.4, 0.05).name('loose overlap β').onChange(v => {
      if (!this.octree) return;
      this.octree.beta = Number(v);
      if (this.trainer) this.trainer.sortGen++;
      wire();
    });
    octF.add(P, 'rebalance_every', 20, 500, 10).name('rebalance K');
    octF.add(P, 'enable_rebalance').name('WLBG grow/prune');
    octF.add(P, 'remove_floaters').name('octree floater kill');
    octF.add(P, 'enable_split').name('error-cov split');
    octF.add(P, 'enable_merge').name('overlap merge');
    octF.add(P, 'max_splats', 10000, 1000000, 10000).name('max splats');
    octF.add(P, 'max_splat_angle', 0.005, 0.2, 0.005).name('max splat angle');
    octF.add(P, 'max_anisotropy', 2, 50, 1).name('max anisotropy');
    octF.add(P, 'opacity_reset', [0, 1000, 2000, 3000, 5000]).name('opacity reset every');
    octF.add(P, 'stipple_seed').name('E4: stipple seeding');
    octF.add(P, 'stipple_count', [500, 1000, 2000, 5000, 10000]).name('stipple seeds / K');

    const evalF = gui.addFolder('Evaluation', { collapsed: true });
    evalF.add(P, 'val_every', [0, 250, 500, 1000, 2500]).name('val PSNR every');
    evalF.add({ val: () => { this.valRequest = true; } }, 'val').name('evaluate val now');

    const snapF = gui.addFolder('Snapshots', { collapsed: true });
    this.snapCtrls = [
      snapF.add(P, 'autosnap_every', [0, 250, 500, 1000, 2500, 5000]).name('auto-snap every'),
      snapF.add({ snap: () => this.requestSnapshot() }, 'snap').name('save snapshot now'),
    ];
  }

  // Snapshots are written via serve.py's POST /save: disable them when it isn't reachable
  async _probeServer() {
    let ok = false;
    try {
      const resp = await fetch('/datasets.json', { cache: 'no-store' });
      ok = resp.ok && !!(await resp.json())?.datasets;
    } catch {}
    this._setSnapshotsEnabled(ok);
  }

  _setSnapshotsEnabled(ok) {
    if (this.serverOk !== ok) {
      this.log(ok ? 'serve.py reachable: snapshots enabled'
                  : 'serve.py not reachable: snapshots disabled (saves fall back to downloads)', ok ? 'info' : 'warn');
    }
    this.serverOk = ok;
    const btn = document.getElementById('BtnSnapshot');
    btn.disabled = !ok;
    btn.title = ok ? 'full training state (splats + Adam + moments + octree), resumable'
                   : 'needs serve.py (POST /save) — not reachable';
    if (!ok) {
      this.params.autosnap_every = 0;
      this.snapRequest = null;
    }
    for (const c of this.snapCtrls || []) c.enable(ok);
    this._refreshGui();
  }

  // Push externally-changed params (e.g. restored from a snapshot) back into the GUI
  _refreshGui() {
    for (const c of this.gui?.controllersRecursive?.() || []) c._refresh?.(c._obj[c._key]);
  }

  _setTraining(on) {
    if (on && !this.gtAvailable) {
      this.log('Training needs the dataset images (none loaded)', 'warn');
      on = false;
    }
    this.params.training = on;
    this.trainBtn.textContent = on ? '⏸ Pause' : '▶ Train';
  }

  // Checkpoint names (availablePlys holds strings for URL datasets, {name, file} for dropped folders)
  _plyNames() {
    return (this.colmap?.availablePlys || []).map(p => (typeof p === 'string' ? p : p.name));
  }

  _rebuildPlySelector(plyList) {
    this.plySelectFolder._host.innerHTML = '';
    this.plySelectFolder._controls = [];
    const opts = { '[init from points3D.bin]': POINTS3D };
    for (const p of plyList) {
      const name = typeof p === 'string' ? p : p.name;
      opts[name] = name;
    }
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
        this.params.split_x = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
      } else if (this.params.camera_mode === 'drone') {
        // stick = drag offset from the press point, full deflection at 120 px
        this._droneTakeover();
        let sx = (e.clientX - this.mouse.sx) / 120, sy = (this.mouse.sy - e.clientY) / 120;
        const l = Math.hypot(sx, sy);
        if (l > 1) { sx /= l; sy /= l; }
        this.drone.stick = [sx, sy];
      } else if (dx !== 0 || dy !== 0) {
        this._leaveColmapView();
        this.params.theta += dx * 0.35;
        this.params.phi = Math.min(89, Math.max(-89, this.params.phi - dy * 0.35));
      }
    });
    cv.addEventListener('wheel', e => {
      e.preventDefault();
      if (this.params.camera_mode === 'drone') {  // wheel = gimbal tilt
        this._droneTakeover();
        this.drone.gimbal = Math.min(1.48, Math.max(-0.5, this.drone.gimbal + (e.deltaY > 0 ? 0.05 : -0.05)));
        return;
      }
      this._leaveColmapView();
      this.params.radius *= (e.deltaY > 0) ? 1.07 : 1 / 1.07;
    }, { passive: false });

    // Keys: H = help, F = orbit / drone, drone flight keys; ignored while typing in the GUI
    const droneKeys = new Set(['arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' ', 'e', 'q']);
    const keyName = e => e.key.toLowerCase();
    const help = $('help-overlay');
    window.addEventListener('keydown', e => {
      if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target?.tagName) || e.ctrlKey || e.metaKey || e.altKey) return;
      const key = keyName(e);
      if (key === 'h' || key === '?') { help.classList.toggle('shown'); return; }
      if (key === 'escape') { help.classList.remove('shown'); return; }
      if (key === 'f') {
        this._setCameraMode(this.params.camera_mode === 'drone' ? 'orbit' : 'drone');
        return;
      }
      if (this.params.camera_mode !== 'drone' || !droneKeys.has(key)) return;
      e.preventDefault();
      this._droneTakeover();
      this.drone.keys.add(key);
    });
    window.addEventListener('keyup', e => this.drone.keys.delete(keyName(e)));
    window.addEventListener('blur', () => { this.drone.keys.clear(); this.drone.stick = [0, 0]; });

    $('thumb-close').addEventListener('click', () => this.closeThumbnail());
    $('thumb-fly').addEventListener('click', () => { if (this.pickedCam >= 0) this.flyToCamera(this.pickedCam); });

    this.trainBtn.addEventListener('click', () => this._setTraining(!this.params.training));
    $('BtnStep').addEventListener('click', async () => {
      if (!this.trainer || !this.gtAvailable) return;
      this.lastStepOut = await this.trainer.step(this.params.snap_to_colmap ? this.params.cam_idx : null);
      this._refreshWireframes();
    });
    $('BtnE1').addEventListener('click', async () => {
      if (!this.trainer) return;
      this.log(`Experiment E1 running on cam #${this.params.cam_idx}...`);
      this.log(await this.trainer.runExperimentE1(this.params.cam_idx), 'ok');
    });
    $('BtnExportPly').addEventListener('click', () => this.savePlyCheckpoint());
    $('BtnExportOctree').addEventListener('click', () => this.saveOctreeJson());
    $('BtnSnapshot').addEventListener('click', () => this.requestSnapshot());
    $('log-clear').addEventListener('click', () => { this.logEl.textContent = ''; });

    $('FolderElmt').addEventListener('change', e => {
      if (e.target.files.length > 0) this.loadDroppedFolder(e.target.files);
    });
    $('PlyElmt').addEventListener('change', e => {
      if (e.target.files.length > 0) this.loadPlyOrOctreeFile(e.target.files[0]);
    });

    const stop = e => { e.preventDefault(); e.stopPropagation(); };
    document.addEventListener('dragenter', stop);
    document.addEventListener('dragover', stop);
    document.addEventListener('drop', e => {
      stop(e);
      const files = e.dataTransfer.files;
      if (!files || files.length === 0) return;
      const n = files[0].name;
      if (files.length === 1 && (/\.(ply|splat|json)$/.test(n) || isSnapshotName(n))) {
        this.loadPlyOrOctreeFile(files[0]);
      } else {
        this.loadDroppedFolder(files);
      }
    });
  }

  // Free camera: 'orbit' (drag = orbit, wheel = zoom) or 'drone' (src/drone.js).
  // Switching keeps the current pose.
  _setCameraMode(mode) {
    if (mode === 'drone') {
      this.droneLive = false;
      this._droneTakeover();
      this.log('Drone mode: drag = pitch/roll, ↑/↓ = forward/back, ←/→ = yaw, E/Q = climb/descend, Space = re-center, wheel = tilt, F = orbit, H = help');
    } else if (this.params.camera_mode === 'drone' && !this.params.snap_to_colmap && !this.flyAnim) {
      this._syncOrbitToLastView();
    }
    this.params.camera_mode = mode;
    this.drone.keys.clear();
    this.drone.stick = [0, 0];
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
    const depth = (sc[0] - c[0]) * f[0] + (sc[1] - c[1]) * f[1] + (sc[2] - c[2]) * f[2];
    const d = Math.max(0.1 * this.octree.sceneRadius, depth);
    const target = [c[0] + f[0] * d, c[1] + f[1] * d, c[2] + f[2] * d];
    const orb = orbitAnglesFromCenter(target, c, this.worldUp);
    this.params.target = target;
    this.params.radius = orb.radius;
    this.params.phi    = Math.min(89, Math.max(-89, orb.phi));
    this.params.theta  = orb.theta;
    this.params.fov    = Math.min(120, Math.max(20, 2 * Math.atan((0.5 * this.canvas.height) / fyPx) * 180 / Math.PI));
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

  // Append to the #log-panel action log. level: 'info' | 'ok' | 'warn' | 'error'.
  // 'ok' / 'error' results also become the telemetry summary line on the side canvas.
  log(msg, level = 'info') {
    (level === 'error' ? console.warn : console.log)('[Gauss-Opt]', msg);
    if (level === 'ok' || level === 'error') this.expSummary = (level === 'error') ? `ERROR: ${msg}` : msg;
    const body = this.logEl;
    if (!body) return;
    const last = body.lastElementChild;
    if (last && last._msg === msg && last._level === level) {  // collapse repeats
      last.querySelector('.log-count').textContent = `×${++last._count}`;
      return;
    }
    const atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 4;
    const line = document.createElement('div');
    line.className = `log-line log-${level}`;
    line._msg = msg; line._level = level; line._count = 1;
    const t = document.createElement('span');
    t.className = 'log-time';
    t.textContent = new Date().toTimeString().slice(0, 8);
    const c = document.createElement('span');
    c.className = 'log-count';
    line.append(t, document.createTextNode(msg), c);
    body.appendChild(line);
    while (body.childElementCount > 300) body.firstElementChild.remove();
    if (atBottom) body.scrollTop = body.scrollHeight;
  }

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

  async loadDatasetUrl(url, plyRel = POINTS3D) {
    try {
      this.clearError();
      this._setTraining(false);
      if (url === '[synth]') {
        this.colmap = null;
        const splats = createSyntheticScene();
        await this._initOctreeAndTrainer(splats, []);
        this.statusEl.textContent = `Loaded synthetic scene (${splats.count} splats)`;
        return;
      }
      this._onProgress(`Loading COLMAP dataset from ${url}...`);
      this.colmap = await loadColmapFromUrl(url, { onProgress: this._onProgress });
      this._rebuildPlySelector(this.colmap.availablePlys);
      if (plyRel && plyRel !== POINTS3D) {
        await this.resumeFromSelectedPly(plyRel);
      } else {
        await this._initFromPoints3D();
      }
    } catch (e) {
      if (!this.serverOk && url.startsWith('/dataset')) {
        // static hosting (e.g. GitHub Pages): the /datasets mounts only exist under serve.py
        this.log(`No dataset server: showing a synthetic scene. Use "load gs/ folder", or run gaussopt/serve.py`, 'warn');
      } else {
        this.showError(`Dataset load failed (${url}): ${e.message}`);
      }
      this.colmap = null;
      await this._initOctreeAndTrainer(createSyntheticScene(), []);
    }
  }

  async loadDroppedFolder(fileList) {
    try {
      this.clearError();
      this.colmap = await loadColmapFromFiles(fileList, { onProgress: this._onProgress });
      this._rebuildPlySelector(this.colmap.availablePlys);
      await this._initFromPoints3D();
    } catch (e) {
      this.showError(`Folder load error: ${e.message}`);
    }
  }

  // Reflect the loaded dataset / checkpoint in the URL (?dataset=...&ply=...) so the page can be
  // reloaded or shared as-is. Other params are kept. Dropped folders / files have no URL form,
  // so those drop the params; a failed dataset load leaves the URL as typed.
  _syncUrl() {
    let dataset = null, ply = null;
    if (this.colmap?.baseUrl) {
      dataset = this.params.dataset_url;
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
    return new LooseOctree({ maxLeafSize: Number(this.params.max_leaf_size), beta: Number(this.params.beta) });
  }

  async _initFromPoints3D() {
    this.plyFromFile = false;
    const octree = this._newOctree();
    const splats = octree.initSplatsFromPoints3D(this.colmap.points3D, this.colmap.images, { aniso: this.params.aniso_init });
    await this._initOctreeAndTrainer(splats, this.colmap.images, octree);
    if (this.colmap.points3D.fallback) {
      this.log(`No sparse/points3D.bin: init from ${splats.count} random points in the camera frustums`, 'warn');
      return;
    }
    const rep = this.colmap.stats.meanReprojError.toFixed(2);
    this.log(`COLMAP verified: ${this.colmap.images.length} cams, ${splats.count} pts, mean reproj err = ${rep} px`, 'ok');
  }

  // Parse a .ply / .splat buffer and restore its embedded octree, if any
  _parseSplats(buf, name) {
    const splats = name.endsWith('.splat') ? parseSplatBuffer(buf) : parsePlyBuffer(buf);
    let octree = null;
    if (splats.octreeData) {
      const t0 = performance.now();
      octree = LooseOctree.fromSerializedNodes(splats.octreeData.meta, splats.octreeData.nodes);
      if (this.colmap?.images) octree.updateCameraAndNeighborStats(this.colmap.images);
      this.log(`Restored embedded octree (${octree.leaves.length} leaves) in ${(performance.now() - t0).toFixed(2)} ms`);
    }
    return { splats, octree };
  }

  // Load a checkpoint (.ply / .splat / .gosnap) of the current dataset, by relative path
  async resumeFromSelectedPly(plyRel) {
    if (!this.colmap) return;
    this.clearError();
    if (plyRel === POINTS3D) {
      await this._initFromPoints3D();
      return;
    }
    try {
      const found = this.colmap.availablePlys.find(p => typeof p === 'object' && p.name === plyRel);
      if (found?.file) {  // dropped folder
        await this.loadPlyOrOctreeFile(found.file);
        return;
      }
      const rel = plyRel.replace(/^\//, '');
      const known = this._plyNames();
      if (known.length > 0 && !known.includes(rel)) {
        throw new Error(`File "${rel}" not found in dataset (available: ${known.join(', ')})`);
      }
      const url = `${this.colmap.baseUrl}/${rel}`;
      this.plyFromFile = false;
      this._onProgress(`Fetching checkpoint ${url}...`);
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText || 'Not Found'} (${url})`);
      const buf = await resp.arrayBuffer();
      if (isSnapshotName(rel)) {
        await this.restoreSnapshot(buf, rel);
        return;
      }
      const { splats, octree } = this._parseSplats(buf, rel);
      await this._initOctreeAndTrainer(splats, this.colmap.images, octree);
      const octMsg = octree ? ` [pre-built octree: ${octree.leaves.length} leaves]` : '';
      this.log(`Resumed ${plyRel} (${splats.count} splats, ${this.colmap.images.length} cams)${octMsg}`, 'ok');
    } catch (e) {
      this.params.resume_ply = POINTS3D;
      this._refreshGui();
      if (!this.trainer) await this._initFromPoints3D();
      this.showError(`Resume failed for "${plyRel}": ${e.message} (using points3D.bin)`);
    }
  }

  // Local file: .json (standalone octree for the current splats), .gosnap, .ply or .splat
  async loadPlyOrOctreeFile(file) {
    const name = file.name;
    try {
      this.clearError();
      this._onProgress(`Reading ${name}...`);
      if (name.endsWith('.json')) {
        const oct = LooseOctree.fromJSON(await file.text());
        if (this.colmap?.images) oct.updateCameraAndNeighborStats(this.colmap.images);
        this.octree = oct;
        this._loggedStats = oct.lastStats;
        if (this.trainer) {
          this.trainer.octree = oct;
          this.trainer.sortGen++;
        }
        this._refreshWireframes();
        this.log(`Loaded saved LooseOctree from ${name} (${oct.leaves.length} leaves, β=${oct.beta})`, 'ok');
        return;
      }
      this.plyFromFile = true;  // no URL form for local files
      const buf = await file.arrayBuffer();
      if (isSnapshotName(name)) {
        await this.restoreSnapshot(buf, name);
        return;
      }
      const { splats, octree } = this._parseSplats(buf, name);
      await this._initOctreeAndTrainer(splats, this.colmap?.images || [], octree);
      this.log(`Loaded ${name} (${splats.count} splats${octree ? ', embedded octree' : ''})`, 'ok');
    } catch (e) {
      this.showError(`Failed to load "${name}": ${e.message}`);
    }
  }

  async _initOctreeAndTrainer(splats, images, existingOctree = null, snapshot = null) {
    this.flyAnim  = null;
    this.lastView = null;
    this.frustumFail.clear();
    this.closeThumbnail();
    if (this.camSlider) {
      const maxIdx = Math.max(0, images.length - 1);
      this.camSlider._max = maxIdx;
      if (this.camSlider._range) this.camSlider._range.max = maxIdx;
      this.params.cam_idx = Math.min(this.params.cam_idx | 0, maxIdx);
    }
    this.octree = existingOctree || this._newOctree();
    if (!existingOctree) this.octree.build(splats.params, splats.count, images);
    this._loggedStats = this.octree.lastStats;  // don't report the initial stats as a rebalance
    this.worldUp = estimateWorldUp(images);
    if (images.length > 0) this.log(`World up (from ${images.length} cams): [${this.worldUp.map(v => v.toFixed(3)).join(', ')}]`);

    this.params.target = [...this.octree.sceneCenter];
    this.params.radius = this.octree.sceneRadius * 1.4;

    const trainer = await GaussOptTrainer.create(this.gpu.device, this.octree, splats, images, this.params);
    // Restore optimizer state before the trainer becomes visible to frame() / step()
    if (snapshot) trainer.applyTrainingState(snapshot.header.state, snapshot.blobs);
    trainer.points3D = this.colmap?.points3D || null;
    this.trainer = trainer;
    this.lastSnapStep = this.lastValStep = this.trainStartStep = trainer.stepCount;
    this._syncUrl();
    // Without the photos (e.g. a sparse/ model only) the geometry is still viewable: cameras,
    // path, points. Training, compare views and "hold target" need them.
    this.gtAvailable = images.length > 0 && await trainer.ensureGtTexture(0).then(t => !!t, () => false);
    if (images.length > 0 && !this.gtAvailable) {
      this.log(`Training images not found (${images[0].imageUrl || images[0].name}): view only, training disabled`, 'warn');
    }
    if (this.autoTrain) {
      this.autoTrain = false;
      this._setTraining(true);
    }
    this._refreshWireframes();
  }

  _refreshWireframes() {
    if (!this.renderer || !this.octree) return;
    this.renderer.updateWireframes({
      octree:       this.octree,
      images:       this.colmap?.images || [],
      activeCamIdx: this._activeCamIdx(),
      showOctree:   this.params.show_octree,
      showLoose:    this.params.show_loose,
      showCameras:  this.params.show_cameras,
      showPath:     this.params.show_path,
      frustumScale: Number(this.params.frustum_scale),
      splitVectors: this.params.show_splits ? this.trainer?.splitVectors : null,
    });
  }

  // serve.py POST /save: writes `name` into <dataset>/splats_gaussopt/, returns { path }
  async _postSave(name, body) {
    const q = `name=${encodeURIComponent(name)}&dataset=${encodeURIComponent(this.params.dataset_url)}`;
    const resp = await fetch(`/save?${q}`, { method: 'POST', body });
    if (!resp.ok) throw new Error(`POST /save -> ${resp.status} ${resp.statusText}`);
    return resp.json();
  }

  // Save via serve.py, falling back to a browser download
  async _saveOrDownload(bytes, name, type, okMsg) {
    try {
      const j = await this._postSave(name, bytes);
      this.log(okMsg(j.path), 'ok');
      return;
    } catch {}
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([bytes], { type }));
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
    this.log(`Downloaded ${name} (no serve.py)`, 'ok');
  }

  requestSnapshot(kind = 'manual') {
    if (!this.serverOk) { this.log('Snapshot unavailable: serve.py not reachable', 'warn'); return; }
    if (this.trainer && this.trainer.count > 0) this.snapRequest = kind;
  }

  // Serialize the full training state; runs from frame() between steps
  async _writeSnapshot(kind) {
    const trainer = this.trainer;
    const t0 = performance.now();
    this.log(`${kind === 'auto' ? 'Auto-snapshot' : 'Snapshot'} @ step ${trainer.stepCount}: reading back GPU state...`);
    const { blobs, ...counters } = await trainer.readbackTrainingState();
    const octJson = this.octree.toJSON();
    const leafIdx = octJson.leafIndices ? new Uint32Array(octJson.leafIndices) : null;
    delete octJson.leafIndices;
    const imgs = this.colmap?.images || [];
    const buf = encodeSnapshot({
      savedAt:     new Date().toISOString(),
      dataset_url: this.params.dataset_url,
      numImages:   imgs.length,
      firstImage:  imgs[0]?.name || null,
      splatFloats: blobs.params.length / Math.max(1, counters.count),
      params:      Object.fromEntries(SNAPSHOT_PARAMS.map(k => [k, this.params[k]])),
      state:       counters,
      octree:      octJson,
    }, { ...blobs, leaf_idx: leafIdx });

    const name = `gaussopt_snap_${kind === 'auto' ? 'latest' : pad5(counters.stepCount)}${SNAPSHOT_EXT}`;
    this.log(`Snapshot encoded in ${(performance.now() - t0).toFixed(0)} ms (${mb(buf)} MB), uploading ${name}...`);
    // Upload in the background: only the GPU readback needs to be serialized with training
    this._uploadSnapshot(buf, name, counters).catch(e => this.showError(`Snapshot upload failed: ${e.message}`));
  }

  async _uploadSnapshot(buf, name, counters) {
    let j;
    try {
      j = await this._postSave(name, buf);
    } catch (e) {
      if (!(e instanceof TypeError)) throw e;  // TypeError = network error: serve.py went away
      this._setSnapshotsEnabled(false);
      throw new Error(`serve.py unreachable, snapshots disabled (${e.message})`);
    }
    this.log(`Snapshot @ step ${counters.stepCount} (${counters.count} splats, ${mb(buf)} MB) -> ${j.path}`, 'ok');
    this._addCheckpointEntry(`splats_gaussopt/${name}`);
  }

  // Add a newly saved checkpoint to the selector (URL datasets only)
  _addCheckpointEntry(rel) {
    if (!this.colmap?.baseUrl || this._plyNames().includes(rel)) return;
    this.colmap.availablePlys.push(rel);
    this.colmap.availablePlys.sort((a, b) => String(a.name ?? a).localeCompare(String(b.name ?? b)));
    this._rebuildPlySelector(this.colmap.availablePlys);
  }

  async restoreSnapshot(buf, name) {
    this.log(`Restoring snapshot ${name} (${mb(buf)} MB)...`);
    const snap = decodeSnapshot(buf);
    const h = snap.header;
    const imgs = this.colmap?.images || [];
    if (h.numImages !== imgs.length || (h.firstImage && imgs[0]?.name !== h.firstImage)) {
      throw new Error(`snapshot was taken on ${h.dataset_url} (${h.numImages} cams), current dataset has ${imgs.length} cams`);
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
    await this._initOctreeAndTrainer({ count, params: snap.blobs.params, octreeData: null }, imgs, octree, snap);
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

  async saveOctreeJson() {
    if (!this.octree) return;
    const bytes = new TextEncoder().encode(JSON.stringify(this.octree.toJSON()));
    await this._saveOrDownload(bytes, `gaussopt_${pad5(this.trainer?.stepCount ?? 0)}.octree.json`, 'application/json',
      path => `Saved standalone octree (${this.octree.leaves.length} leaves) to: ${path}`);
  }

  _logRebalance() {
    const st = this._loggedStats = this.octree.lastStats;
    const tr = this.trainer;
    const extra = (st.seeded ? `, +${st.seeded} stipple` : '') +
                  (tr.lastOpacityReset === tr.stepCount ? ', opacity reset' : '');
    this.log(`Rebalance @ step ${tr.stepCount}: +${st.split} split, +${st.cloned} clone, ` +
             `-${st.pruned} prune, -${st.floaters} floaters, -${st.merged} merge${extra} -> ${tr.count} splats, ` +
             `${this.octree.leaves.length} leaves`);
    const si = tr.lastSeedInfo;
    if (si?.capped) {
      this.log(`E4 stipple seeding skipped: ${tr.count} splats >= max_splats (${this.params.max_splats})`, 'warn');
    } else if (si) {
      this.log(`E4 stipple seeds from cam #${si.viewIdx}: ${si.count} sites (depth: ${si.covered} rendered, ` +
               `${si.fromPoints} COLMAP pts, ${si.fromMedian} median) in ${si.ms.toFixed(0)} ms`);
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
        this.log(`Val PSNR @ step ${r.step}: ${r.psnr.toFixed(2)} dB over ${r.views} held-out views ` +
                 `(${r.count} splats, ${(performance.now() - t0).toFixed(0)} ms)`, 'ok');
      } else {
        this.log('Val PSNR: no validation views (needs >= 8 cameras)', 'warn');
      }
    } catch (e) {
      this.showError(`Validation failed: ${e.message}`);
    }
  }

  // Training steps, then (between steps) stop_after / validation / snapshot triggers, then draw
  async frame() {
    const nowT = performance.now();
    const dtFrame = Math.min(0.05, (nowT - this.lastFrameT) / 1000);
    this.lastFrameT = nowT;
    const P = this.params;
    if (this.trainer && P.training) {
      // Keep the GPU at most ~1 frame behind: otherwise training dispatches pile up in the
      // queue and every displayed frame (and input response) waits behind them.
      if (this.gpuIdle) await this.gpuIdle;
      const t0 = performance.now();
      for (let s = 0; s < P.steps_per_frame; s++) {
        this.lastStepOut = await this.trainer.step();
        if (performance.now() - t0 > TRAIN_BUDGET_MS) break;
      }
      if (this.octree.lastStats !== this._loggedStats) this._logRebalance();
      const every = Number(P.autosnap_every);
      if (this.serverOk && every > 0 && this.trainer.stepCount - this.lastSnapStep >= every && !this.snapRequest) {
        this.snapRequest = 'auto';
      }
      const stopAfter = Number(P.stop_after);
      if (stopAfter > 0 && this.trainer.stepCount - this.trainStartStep >= stopAfter) {
        this._setTraining(false);
        this.valRequest = true;
        this.log(`stop_after: paused @ step ${this.trainer.stepCount} (${this.trainer.count} splats)`, 'ok');
      }
      const valEvery = Number(P.val_every);
      if (P.training && valEvery > 0 && this.trainer.stepCount - this.lastValStep >= valEvery) {
        this.valRequest = true;
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

    this.gpuIdle = P.training ? this.gpu.device.queue.onSubmittedWorkDone() : null;
    requestAnimationFrame(() => this.frame());
  }

  async _draw(dtFrame) {
    const P = this.params, tr = this.trainer;
    const W = this.canvas.width, H = this.canvas.height;
    const imgs = this.colmap?.images || [];
    const camIdx = Math.min(Math.max(0, P.cam_idx | 0), Math.max(0, imgs.length - 1));

    if (this.holdTarget && imgs[camIdx] && this.gtAvailable) {
      const gtTex = await tr.ensureGtTexture(camIdx);
      if (gtTex) this.renderer.renderCompare({ gtTex, hatTex: tr.hatTex, mode: 0 });
    } else if (P.compare_mode > 0 && imgs.length > 0 && this.gtAvailable) {
      const out = (P.training && this.lastStepOut?.gtTex) ? this.lastStepOut : await tr.renderViewForward(camIdx);
      if (out?.gtTex) {
        this.renderer.renderCompare({
          gtTex: out.gtTex, hatTex: out.hatTex,
          mode: Number(P.compare_mode), splitX: Number(P.split_x), amp: 3.0,
        });
      }
    } else {
      // Camera precedence: fly-to animation > COLMAP snap > drone > orbit
      let view, camCenter, intr, flyDone = false;
      if (this.flyAnim && imgs[this.flyAnim.camIdx]) {
        const st = evaluateCameraSpline(this.flyAnim);
        ({ view, camCenter } = st);
        intr = st;
        flyDone = st.done;
      } else if (P.snap_to_colmap && imgs[camIdx]) {
        view      = imgs[camIdx].view;
        camCenter = imgs[camIdx].center;
        intr      = fitIntrinsics(imgs[camIdx].camera, W, H);
      } else if (P.camera_mode === 'drone' && this.droneLive) {
        this.drone.update(dtFrame, { sceneCenter: this.octree.sceneCenter, sceneRadius: this.octree.sceneRadius });
        view      = this.drone.view();
        camCenter = [...this.drone.pos];
        intr      = this._freeIntrinsics(W, H);
      } else {
        const orb = buildOrbitViewMatrix(P.target, P.radius, P.theta, P.phi, this.worldUp);
        view      = orb.view;
        camCenter = orb.center;
        intr      = this._freeIntrinsics(W, H);
      }
      const { fx, fy, cx, cy } = intr;
      const numVis = tr.sortForDisplay(camCenter, view, 2 * fx / W, 2 * fy / H);
      this.renderer.renderScene({
        splatsBuf:      tr.splatsBuf,
        momentsBuf:     tr.momentsBuf,
        sortedIdxBuf:   tr.viewIdxBuf,
        numVisible:     numVis,
        view,
        proj:           null,
        camCenter,
        sceneRadius:    this.octree.sceneRadius,
        fx, fy, cx, cy, W, H,
        visuMode:       Number(P.visu_mode),
        colorMode:      Number(P.color_mode),
        haze:           Number(P.haze),
        avgR:           tr.avgR,
        drawWireframes: P.show_octree || P.show_cameras || P.show_path || P.show_splits,
        frustumTextures: this._frustumTextures(imgs),
      });
      this.lastView = { view, center: camCenter, fx, fy, cx, cy };
      if (flyDone) this._finishFly(this.flyAnim.camIdx);
    }

    const psnrTxt = tr.lastPsnr > 0 ? `${tr.lastPsnr.toFixed(2)} dB` : '--';
    const optLabel = P.opt_method === 'sim_anneal'  ? `SA(T=${tr.currentTemp.toFixed(3)})`
                   : P.opt_method === 'monte_carlo' ? `MC(η=${P.mc_noise})` : 'Adam';
    this.statusEl.textContent =
      `[${optLabel}] Step: ${tr.stepCount} | PSNR: ${psnrTxt} | Splats: ${tr.count} (${tr.numVisibleView} vis) | Leaves: ${this.octree.leaves.length}` +
      (P.camera_mode === 'drone' && this.droneLive ? ` | ${this.drone.hud(this.octree.sceneRadius)}` : '');
    drawTelemetryPanel(this.sideCanvas, tr.history, this.octree, this.expSummary, tr.valHistory);
  }
}
