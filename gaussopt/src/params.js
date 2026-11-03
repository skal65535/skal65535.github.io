// params.js — The app's params: defaults (defaultParams), their URL form (URL_ALIASES,
// applyUrlParams), the GUI's choice lists and tooltips, and which ones snapshots and project.json
// keep. Needs ../common/args.js (parse_arg*) loaded as a classic script before this module.
"use strict";

import { TRAIN_DEFAULTS } from './schedule.js';
import { urlFlagOn } from './url.js';

export const VISU_MODES = { 'gaussian': 0, 'cone': 1, 'ellipsis': 2, 'small_dot': 3, 'axis': 4 };

export const COLOR_MODES = {
  'learned_rgb':     0,
  'octree_leaf':     1,
  'wlbg_error (R)':  2,
  'view_div (Neff)': 3,
  'depth':           4,
};

export const COMPARE_MODES = {
  '3D scene': 0,
  'split: photo | render': 1,
  'residual |photo - render|': 2,
  'depth spread': 3,
};

export const FRUSTUM_IMAGES = { 'Off': 'off', 'Active Cam Only': 'active', 'All Cameras': 'all' };

// The mesh in the free view (render.js MESH_*; DESIGN.md §9 E5)
export const MESH_VIEWS = { 'splats only': 0, 'mesh + splats': 1, 'mesh only': 2, 'wireframe + splats': 3 };
export const MESH_KINDS = { 'surface (fused depth)': 'surface', 'behind the splats': 'behind' };  // Make mesh: mesh.js, mesh_grow.js
export const MESH_TEXTURES = { 'off (vertex colors)': 0, '1024': 1024, '2048': 2048, '4096': 4096 };

export const OPT_METHODS = {
  'Adam (deterministic)':      'adam',
  'Monte-Carlo (SGLD / MCMC)': 'monte_carlo',
};

// Tooltips of the trainer, view and project controls (mkGui tips: by key, or by folder name)
export const TIPS = {
  'Example datasets': 'datasets (gs/ folders: cameras, points, photos) this page can load: the /datasets/ ones when serve.py mounts them',
  dataset_url: 'loads a dataset (a gs/ folder: cameras, points, photos) and starts new splats from its points',
  'Start from': 'new splats from the SfM points, or resume from a saved .ply / snapshot of this dataset',
  resume_ply: 'the splats to start from: new ones from the SfM points (points3D), or a checkpoint saved by serve.py',
  Training: 'the optimizer (DESIGN §5.4), the length and size of the run, the loss, validation and snapshots',
  'Monte-Carlo': 'the settings of the Monte-Carlo optimizer (3DGS-MCMC), shown when it is the method',
  opt_method: 'Adam: plain gradient descent. Monte-Carlo (3DGS-MCMC): Adam plus L1 terms on opacity and scale, Langevin noise on ' +
              'the nearly transparent splats, dead splats moved onto live ones (no opacity reset)',
  mc_noise: 'Monte-Carlo (3DGS-MCMC): noise on the centers of nearly transparent splats (opacity < 0.005), x lr_pos x their covariance',
  mc_opacity_reg: 'Monte-Carlo: weight of the mean opacity in the loss (splats that do not earn their opacity die, and move)',
  mc_scale_reg: 'Monte-Carlo: weight of the mean scale in the loss',
  mc_relocate: 'Monte-Carlo: the dead splats move onto live ones drawn by opacity (3DGS-MCMC) or by their error',
  mc_relocate_mode: 'Monte-Carlo: a live splat drawn takes the dead one as a copy in place (copy, 3DGS-MCMC) or splits into two halves within its shape (shape, brush)',
  View: 'what the view shows (the 3D scene, or the photo against the render) and from where',
  Splats: 'how the splats are drawn (the view only, training is not changed)',
  Overlays: 'cameras, camera path, octree and split axes drawn over the splats',
  Mesh: 'Export: the settings of Make mesh. ⚙ view: how the mesh (Make mesh, or a dropped .glb) is shown',
  'Free camera': 'the orbit, drone and demo cameras (F, G; H lists the keys)',
  visu_mode: 'gaussian: the real render. cone, ellipsis, small dot, axis: shapes to inspect the splats',
  color_mode: 'learned colors, or a diagnostic: octree leaf, error mass R (WLBG), effective view count Neff, depth',
  compare_mode: '3D view (free or snapped to a camera), a split wipe photo | render, the residual |photo - render|, or ' +
                'the depth spread: how far apart in depth the middle half of each pixel\'s opacity is (red at 7% of the depth, white at 20%: ' +
                'e.g. a wall rendered as two layers; blue: not enough opacity to tell). ' +
                'The 2D modes show the camera being trained, or camera # (below) when paused',
  split_x: 'Split mode: where the wipe between the photo (left) and the render (right) is',
  mesh_view: 'the mesh (Make mesh, or a dropped .glb) in the 3D view. mesh + splats: the splats behind its surface hidden, ' +
             'those in front kept (floaters show); mesh only; wireframe + splats: the splats behind its surface hidden, its visible edges over the rest, no fill',
  mesh_shaded: 'the mesh in gray, lit from the camera (shows its shape), instead of its colors',
  mesh_margin: 'mesh + splats, wireframe + splats: a splat is hidden only when its center is this far behind the mesh, in voxels (the mesh is at the ' +
               'splats\' median depth: at 0 it hides the half of a surface\'s splats behind it, 37-55% of what the canyons\' splats draw; 2 voxels: 14-28%)',
  mesh_kind: 'Make mesh: surface: the splats\' rendered depth fused (at their median depth; the scene\'s shape); behind the splats: a solid grown from ' +
             'the splats and carved by the cameras\' rays, its surface a few voxels behind them (a backing that fills the gaps between them, open on the ' +
             'side no camera saw; DESIGN.md E5)',
  mesh_res: 'Make mesh: grid voxels along the scene\'s longest side (finer: more detail, more noise, slower)',
  mesh_simplify: 'Make mesh: the largest shape error allowed when simplifying, in voxels (0: not simplified)',
  mesh_texture: 'Make mesh: a texture of this size (texels per side) baked from the photos where the splats\' depth sees the surface; ' +
                'off: only the vertex colors (from the splat renders)',
  haze: 'draws every splat this much larger (view only, training is not changed)',
  dome: 'the Dome step (src/dome.js): auto: before the training starts, the sky found in the photos (the directions where the ' +
        'views agree: no parallax) and, when it covers enough of them, a dome made from it behind the splats, fixed while ' +
        'they train (the zenith and what the cameras did not see filled in); off: none (the learned background of bg_lr, if any)',
  dome_fill: 'what the dome holds besides the sky found: sky only (the sky spread where no camera looked; what the cameras saw that is ' +
             'not sky left to the splats), panorama (every direction a camera saw, its color: the far things as billboards ' +
             'at infinity), sky everywhere (the sky spread over every other direction). Changed after the dome is made: filled again at once',
  show_dome: 'the background behind the splats in the view: the dome (Dome step), or the learned one (bg_lr)',
  dome_only: 'the view draws the background alone (the dome, or the learned one), without the splats nor the mesh: what the dome took',
  show_octree: 'wireframes of the octree leaves, colored by state',
  octree_hidden: 'the octree boxes\' opacity where behind the splats (their half-opaque parts, and the mesh): 1 drawn through as in front',
  show_loose: 'the loose boxes of the leaves, (1 + 2β) times their nominal size',
  show_cameras: 'the training cameras as frustums (validation ones in another color, the active one highlighted)',
  show_path: 'a line through the camera centers, in the camera path order',
  path_order: 'the camera path (path line, demo flight): capture order, or the shortest path through every camera (photo sets taken back and forth); names and training unchanged',
  show_splits: 'the error axes along which the last rebalance split splats',
  Level: 'turn the picture about the view direction until the scene stands upright ([ / ] keys); orbit to another side and turn again. The orbit, drone and demo cameras and the mesh\'s .glb use this up (estimated from the cameras, turned by the scene\'s transform: saved in the project\'s project.json)',
  tilt_ccw: 'turn the picture 1° counter-clockwise ([)',
  tilt_cw: 'turn the picture 1° clockwise (])',
  tilt_ccw_fine: 'turn the picture 0.2° counter-clockwise',
  tilt_cw_fine: 'turn the picture 0.2° clockwise',
  level_reset: 'back to the up estimated from the cameras (the identity transform)',
  frustum_images: 'the training photos drawn inside the frustums: none, the active camera\'s, or all',
  frustum_scale: 'size of the drawn frustums',
  snap_to_colmap: 'look through the training camera chosen below',
  cam_idx: 'the training camera to look through, or to fly from',
  camera_mode: 'drone: drag to look, wheel or pinch to move forward / back, arrows fly and turn, E / Q climb. ' +
               'demo: a slow flight along the capture path (key G). H lists all the keys',
  demo_speed: 'demo flight: speed along the capture path, in scene radii per second',
  demo_wander: 'demo flight: how far it wanders off the path, in units of the cameras\' near depth',
  fov: 'vertical field of view of the free camera, in degrees',
  radius: 'orbit camera: distance to the target',
  'Growth & pruning': 'at each rebalance (every K steps, Octree): grow, prune and move splats; the splat shape limits (DESIGN §5)',
  Floaters: 'at each rebalance: remove the splats that float in front of the cameras',
  'Schedule (E7)': 'experiment E7: over max splats while training, back to it at the end of the planned steps',
  Octree: 'the loose octree the splats are sorted in, rebuilt at each rebalance (every K steps)',
  max_leaf_size: 'most splats an octree leaf holds (one GPU workgroup); the octree is rebuilt',
  beta: 'looseness of the octree boxes: a splat stays in its leaf while it fits in (1 + 2β) times the box',
  rebalance_every: 'K: steps between two rebalances (growth, pruning, floater removal, octree rebuild)',
  rebalance_every_cap: 'steps between two rebalances once the splat count is at max splats (0: K); also one at each opacity reset',
  enable_rebalance: 'grow and prune splats at each rebalance (off: their number stays fixed)',
  remove_floaters: 'at each rebalance, remove floaters: splats in front of the cameras, single-view billboards, isolated leaves',
  floater_few_views: 'floaters include the splats seen from few views with much error (off: what passes by in a movie stays)',
  decay_occluders: 'at each rebalance, a splat with much error loses a third of its opacity (off: kept as it is)',
  floater_depth: 'which splats count as too near a camera (closer than 0.65 x this depth). observed: the depth of the ' +
                 'points the camera sees (the default, best at 30k steps on 4 of 5 test scenes). view: of all points ' +
                 'in its view, also those only other cameras saw (better early in training on gare-st-lazare)',
  floater_spare: 'a splat this near an SfM point (x the scene radius) is never too near a camera, whichever camera saw ' +
                 'the point (0: off). Without it, cameras walking past an object they did not see carve it',
  enable_split: 'split splats with much unexplained error along the main axis of that error, or clone small ones',
  enable_merge: 'merge overlapping splats of nearly the same color',
  max_splats: 'growth stops at this many splats',
  cap_lease: 'E7: how far over max splats the count may go while training; the consolidation brings it back (needs a planned length)',
  train_steps: 'E7: the planned length of the training (0: open-ended, no lease); the consolidation ends there',
  polish: 'E7: the last part of the planned length without Monte-Carlo\'s noise and L1 terms (plain Adam steps)',
  consolidate_frac: 'E7: the last part of the planned length: the count falls back to max splats (the splats seen least go), then trains there',
  max_splat_angle: 'largest splat size, as a fraction of its depth in the training view (no huge splats; 0: none, the default)',
  drop_rate: 'DropGaussian: the share of splats left out of each training render, the others more opaque by ' +
             '1 / (1 - rate) (0: off; 0.25 helps on sparse photo sets, DESIGN.md E3)',
  robust_loss: 'tau of the robust loss: a pixel whose error e is large counts 1 / (1 + (e / tau)^2), so what one frame ' +
               'shows and the others contradict (a movie\'s debris, a flash) is not chased (0: off)',
  split_min_views: 'growth splits or clones only the splats seen from at least this many views (N_eff = W^2 / S): ' +
                   'detail fitted to a single photo does not multiply (0: any)',
  split_mode: 'E9: how a splat splits: \'error\' (two children along its error moments, or a clone at the error centroid) or ' +
              '\'shape\' (two halves drawn within its own shape, scales / sqrt 2, opacity 1 - sqrt(1 - o), as brush)',
  bg_lr: 'E13: the learning rate of a background behind the splats (a cube map of the directions, at infinity): far ' +
         'content (sky, the far end of a valley) goes there instead of onto a sheet of splats at the scene\'s edge. 0: off (black)',
  exposure_lr: 'E19: each training photo\'s exposure learned with the splats (a gain and an offset per color): its learning ' +
               'rate (0: off). The held-out views have none: their score after the best gain is logged too',
  exposure_from: 'E19: the step the exposures start learning at (the splats formed first)',
  pose_lr: 'E19: each training photo\'s pose refined by the splats: its rate (radians a visit; the position x the scene\'s size). 0: off. ' +
           'The held-out photos take their neighbours\' corrections',
  lens_lr: 'E19: each lens\'s focal and principal point refined by the splats: its rate (log focal a step; the principal point x the width). 0: off',
  pose_from: 'E19: the step the poses and lenses start refining at',
  pose_prior: 'E19: the pull of the refined poses and lenses to the SfM\'s: their corrections held within about this ' +
              '(degrees; the move this x the scene\'s size, the focal this in radians relative). 0: none',
  pose_refit: 'E19: every N steps (with the pose rate on), each training photo\'s camera refitted with the splats frozen ' +
              '(pose refit steps a photo): a pose otherwise moves one step a pass over the photos. 0: never',
  pose_refit_iters: 'E19: the steps of each photo\'s camera in a refit',
  ssim: 'weight of the D-SSIM loss, (1 - w) L1 + w (1 - SSIM) / 2 as in 3DGS (0.2 there). 0: off, ' +
        '0.8 L1 + 0.2 L2 (the default)',
  max_anisotropy: 'largest ratio of a splat\'s longest axis to its middle one (no needles; 0: none, the default)',
  opacity_reset: 'every N steps, all opacities are lowered to 0.01 so useless splats fade out and get pruned (0: never). ' +
                 'Validation right at a reset reads low',
  opacity_reset_until: 'the last step of the opacity resets (and half the planned steps): the splats earn their opacity back after it',
  stipple_seed: 'experiment E4: new splats seeded where the residual is large, by stippling the error image',
  stipple_count: 'E4: how many seeds at each rebalance',
  region_log: 'E7: log the error of fixed cells of the scene at each rebalance (trainer.regionLog)',
  region_res: 'E7: the cells per axis of that log and of the repair',
  region_repair: 'E7: the regions whose error is high and falls slower than the scene\'s split more easily and get more of the relocated splats (0: off)',
  val_every: 'PSNR / SSIM on the held-out cameras (never trained on) every N steps (0: never)',
  val: 'evaluate the held-out views now',
  autosnap_every: 'save a snapshot (the full training state, to resume later) every N steps (0: never); each one replaces the last. Export: save one now',
};

// Example datasets: the /datasets/* ones are listed only when serve.py mounts them
export const DATASETS = {
  'jardin1_gs (16 views)':  '/datasets/jardin1_gs',
  'jardin3_gs (120 views)': '/datasets/jardin3_gs',
  'jardin3-2_gs (359 views)': '/datasets/jardin3-2_gs',
  '[synthetic ring]':       '[synth]',
};
export const NOT_A_URL = '[none]';  // dataset_url while the cameras come from a video or a dropped folder

export const POINTS3D = '[points3D.bin]';  // resume_ply value: init from COLMAP points

// Short URL aliases (?dataset=...&ply=...&train&spf=4...); every param is also settable by name.
const URL_ALIASES = {
  dataset_url: 'dataset', resume_ply: 'ply', training: 'train', steps_per_frame: 'spf',
  downsample: 'ds', opt_method: 'opt', stipple_seed: 'stipple', val_every: 'val',
  autosnap_every: 'autosnap',
  max_splats: 'max_splats',  // parsed (and clamped) by parse_arg below
};
// Other URL params read by the app: &video=URL, &photos=URL,URL (SfM panel), &nosg, &sfm_* (src/ui/sfm_panel.js)
const isOtherUrlParam = k => k === 'video' || k === 'photos' || k === 'nosg' || k.startsWith('sfm_');

// Params saved in / restored from a .gosnap snapshot (optimizer + growth settings)
export const SNAPSHOT_PARAMS = [
  'opt_method', 'mc_noise', 'mc_opacity_reg', 'mc_scale_reg', 'mc_relocate', 'mc_relocate_mode',
  'max_leaf_size', 'beta', 'rebalance_every', 'rebalance_every_cap',
  'enable_rebalance', 'enable_split', 'enable_merge', 'remove_floaters', 'floater_few_views', 'decay_occluders', 'floater_depth', 'floater_spare',
  'max_splats', 'cap_lease', 'train_steps', 'consolidate_frac', 'polish', 'prune_opacity', 'hysteresis', 'stipple_seed', 'stipple_count', 'region_repair', 'region_res',
  'lr_pos', 'lr_scale', 'lr_quat', 'lr_color', 'lr_opacity', 'lr_decay_steps', 'lr_pos_final', 'ssim', 'bg_lr', 'exposure_lr', 'exposure_from', 'pose_lr', 'lens_lr', 'pose_from', 'pose_prior', 'pose_refit', 'pose_refit_iters',
  'drop_rate', 'robust_loss', 'split_min_views', 'split_mode',
  'max_splat_angle', 'max_splat_scale', 'max_anisotropy', 'opacity_reset', 'opacity_reset_until',
];

// Settings kept in a project's project.json and applied when it is reopened: the training ones
// and the view's, not the session state (dataset, checkpoint, camera pose, training on / off)
export const PROJECT_PARAMS = [
  ...SNAPSHOT_PARAMS, 'downsample', 'aniso_init', 'val_every', 'val_split', 'stop_after', 'autosnap_every',
  'split_x', 'visu_mode', 'color_mode', 'compare_mode', 'haze',
  'show_octree', 'octree_hidden', 'show_loose', 'show_cameras', 'show_path', 'path_order', 'show_splits', 'frustum_images', 'frustum_scale',
  'mesh_view', 'mesh_shaded', 'mesh_margin', 'mesh_kind', 'mesh_res', 'mesh_simplify', 'mesh_texture', 'dome', 'dome_fill', 'show_dome', 'dome_only',
];

export const URL_FLAGS = ['train'];  // the boolean params read by presence (parse_arg_bool)
// The URL names of param `key` ([alias, name]), or [] for those app.js _syncUrl writes after a load
// (dataset_url, resume_ply)
export const urlNamesOf = (key) => (key === 'dataset_url' || key === 'resume_ply') ? [] : [URL_ALIASES[key], key].filter(Boolean);

// The params of a new app: defaults, the aliased URL params already parsed (parse_arg*)
export function defaultParams() {
  return {
    dataset_url:     parse_arg_str('dataset', '/datasets/jardin1_gs'),
    resume_ply:      parse_arg_str('ply', POINTS3D),
    training:        parse_arg_bool('train'),
    steps_per_frame: parse_arg('spf', 8, 1, 20),

    // the training's (schedule.js), and those the URL sets by an alias
    ...TRAIN_DEFAULTS,
    downsample:      parse_arg('ds', TRAIN_DEFAULTS.downsample, 1, 8),
    opt_method:      parse_arg_str('opt', TRAIN_DEFAULTS.opt_method),
    max_splats:      parse_arg('max_splats', TRAIN_DEFAULTS.max_splats, 1000, 2000000),
    stipple_seed:    urlFlagOn(parse_arg_str('stipple', '1')),  // on by default (E6; stipple=0: off; a bare 'stipple': on)

    // Camera & view
    snap_to_colmap:  false,
    camera_mode:     'drone',  // free camera: 'drone' (src/drone.js) | 'demo' (src/demo.js)
    demo_speed:      0.05,     // demo flight along the capture path, scene radii per second
    demo_wander:     0.5,      // demo wander around the path, in cameras' near depths (demo.js)
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
    octree_hidden:   0.15,
    show_loose:      false,
    show_cameras:    true,
    show_path:       false,  // COLMAP camera path (centers in path_order)
    path_order:      'capture',  // the camera path: 'capture' (name order) | 'shortest' (path.js, DESIGN-sfm §13.3 P7)
    show_splits:     false,
    frustum_images:  'off',
    frustum_scale:   1.0,
    mesh_view:       0,      // MESH_VIEWS
    mesh_shaded:     false,
    mesh_margin:     2,      // voxels behind the mesh before a splat is hidden
    mesh_kind:       'surface',  // Make mesh: MESH_KINDS
    mesh_res:        384,    // ... grid voxels along the longest side ...
    mesh_simplify:   1,      // ... and the simplification's error limit, in voxels
    mesh_texture:    2048,   // ... and its texture's size (MESH_TEXTURES; 0: vertex colors)
    dome:            'off',  // the Dome step: 'auto' (a dome when the photos show enough sky) | 'off'
    dome_fill:       'unseen', // ... and what it holds besides the sky (dome.js DOME_FILLS): 'unseen' | 'seen' (panorama) | 'all'
    show_dome:       true,   // the background (dome or learned) drawn in the view
    dome_only:       false,  // ... alone, no splats nor mesh

    // Octree & WLBG growth
    max_leaf_size:   256,
    aniso_init:      true,   // init splats from the PCA of their 8 nearest COLMAP points
    beta:            0.20,

    // Evaluation: held-out (validation camera) PSNR every N training steps (0 = off)
    val_every:       parse_arg('val', 500, 0, 100000),
    val_split:       'every8',  // held-out views, on load: 'every8' or 'block' (colmap.js splitValidation)
    stop_after:      0,  // pause training (and run validation) N steps after training (re)starts (0 = never)

    // Auto-snapshot every N steps (0 = off), overwriting gaussopt_snap_latest.gosnap
    autosnap_every:  parse_arg('autosnap', 0, 0, 100000),
  };
}

// Applies ?name=value of `search` to every param of `params` (aliases are already parsed by
// parse_arg*). Returns the set of param names given in the URL (they win over snapshot params).
export function applyUrlParams(params, search = location.search) {
  const args = new URLSearchParams(search);
  const aliases = Object.values(URL_ALIASES);
  const keys = new Set();
  for (const [k, def] of Object.entries(params)) {
    const alias = URL_ALIASES[k];
    const name = (alias && args.has(alias)) ? alias : k;
    if (!args.has(name)) continue;
    // aliased numbers / strings are parsed (and clamped) by parse_arg*; booleans only by presence
    if (name === alias && typeof def !== 'boolean') { keys.add(k); continue; }
    const v = args.get(name);
    if (typeof def === 'boolean') {
      params[k] = urlFlagOn(v);
    } else if (typeof def === 'number') {
      const x = parseFloat(v);
      if (!Number.isFinite(x)) { console.warn(`URL param ${k}=${v}: not a number`); continue; }
      params[k] = x;
    } else if (typeof def === 'string') {
      params[k] = v;
    } else if (Array.isArray(def)) {
      const a = v.split(',').map(parseFloat);
      if (a.length !== def.length || a.some(x => !Number.isFinite(x))) continue;
      params[k] = a;
    } else {
      continue;
    }
    keys.add(k);
  }
  for (const k of args.keys()) {
    if (!(k in params) && !aliases.includes(k) && !isOtherUrlParam(k)) console.warn(`Unknown URL param: ${k}`);
  }
  return keys;
}
