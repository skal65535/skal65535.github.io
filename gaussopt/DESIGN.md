# Gauss-Opt: WebGPU 3D Gaussian Splatting Optimizer — Design Document

## 1. Overview & Motivation

**Gauss-Opt** (`gaussopt/`) is a standalone, browser-native **WebGPU 3D Gaussian Splatting (3DGS) Optimizer** in [`skal65535.github.io`](../), built by combining and extending techniques from two sibling projects:
1. **[`splats/`](../splats/index.html):** WebGPU 3DGS forward EWA covariance projection (`get_cov2d`, `get_axis`), quaternion/scale to $\Sigma_{3D}$ conversion, multi-mode visualization (`kVisus`), and interactive free-view camera/GUI controls.
2. **[`stipple/`](../stipple/stipple-iter.js):** WebGPU compute-shader optimization loop with CAS-loop `f32` atomic accumulation ([`stipple-shaders.wgsl`](../stipple/stipple-shaders.wgsl)), asynchronous GPU$\to$CPU readback (`mapAsync`), 2nd-order moment-based adaptive split/merge/prune, and a zero-compute pure draw layer ([`stipple-render.js`](../stipple/stipple-render.js)).

### Scope & Assumptions
* **Input format:** Already-extracted, registered, and undistorted COLMAP `gs/` directories produced by [`video2splat.py`](../../../brush/video2splat.py) (`sparse/cameras.bin`, `sparse/images.bin`, `sparse/points3D.bin`, and undistorted `images/`), plus optional `.ply` / `.splat` checkpoints to resume training.
* **Target platform:** WebGPU in desktop browsers (Chrome / Dawn on Apple Silicon & discrete GPUs).
* **Key architectural bet:** Replace the monolithic global GPU radix sort of standard 3DGS trainers (like Brush / `gsplat`) with a **3D Loose Octree** whose leaves are sized to fit a single WebGPU compute workgroup ($N_{\text{leaf}} \le 256$ splats), paired with periodic **CPU-driven WLBG/stipple moment growth, octree-guided floater removal, and rebalancing**.

---

## 2. Core Design Decisions

### D1. Hybrid GPU Inner Loop / CPU Outer Loop (the `stipple/` Pattern)
* **Inner Loop (100% GPU Compute, $K \approx 100\text{--}250$ steps):**
  * Fixed octree topology and fixed splat-to-leaf assignment.
  * Runs forward tile rasterization, $L_1$ + $L_2$ loss, backward gradient accumulation, per-splat residual moment accumulation, and parameter updates (`Adam`, `Monte-Carlo / SGLD`, or `Simulated Annealing`) without CPU synchronization.
  * **Frame loop (`app.js` `_frame`, `pace.js`):** each display frame runs some steps then draws. One frame stays in flight: the CPU prepares a frame (the sorts: 1.6–3.9 ms per step) while the GPU runs the previous one; waiting for the previous one instead left the GPU idle while the CPU worked and waited for the display refresh. The step count is chosen so a frame takes about 25 ms of GPU time, from a running estimate of the GPU time per step (`FramePacer`); `?spf=` (default 8) bounds it; it is no longer kept in project.json (a setting of the machine, and the saved 2 would keep the old rate). gare and jardin7 at the 300k cap, headless Chrome, M5 Pro, steps 3,500–5,500: 98 → 126 and 89 → 104 steps/s, the 3D view updated 49 → 39 and 45 → 37 times a second; 94% of the rate of steps run back to back with no display (133 and 111 steps/s). A 16 ms target fell to 1.2 steps per frame and 64 steps/s on gare (the GPU idles between display refreshes); 33 ms gave the same rate as 25 with 27 view updates a second. Before, `?spf=8` reached the same rate only by drawing 16 times a second. With the tile rects fixed (D3, per-tile lists): 178 and 146 steps/s, 95% and 99% of the back-to-back rate with rebalances (5.4 and 6.8 ms per step), the view updated 32 and 33 times a second; the CPU now spends about as long per step as the GPU (3.5 ms on gare).
* **Outer Loop (Async CPU Readback via `mapAsync`, every $K$ steps):**
  * Reads back splat parameters and accumulated 3D error/visibility moments.
  * Performs **WLBG hysteresis prune (or MCMC teleportation), octree-guided floater removal, error-covariance split/clone, drift re-binning, overlap moment reconciliation, and octree split/collapse** in plain JavaScript on the CPU, then re-uploads the compacted leaf buffers.

### D2. Loose Octree Spatial Partitioning ($N_{\text{leaf}} \le 256$)
* **Why an Octree:**
  1. **Cheap exact culling:** hierarchical frustum culling over leaves (plus a per-splat frustum/opacity test) yields the same visible set as an exhaustive pass at ~1/10 the cost.
  2. **Depth order = octree cull + radix sort:** leaf traversal (`flips ^ cam_octant_mask`) is a valid per-ray order for *points*, but not for extended splats — see Experiment E1 (§9). The culled set is therefore sorted exactly by view-space $z$ with a 3-pass 11-bit LSD radix sort on the float bit patterns (`sortOctreeFrontToBack`).
  3. **Hierarchical Culling & Spatial Context:** Frustum culling, early-out transmittance culling ($T_{\text{tile}} < 10^{-4}$), and 3D neighbor/density checks for floater removal operate at the octree node level.
* **Why "Loose" (Overlapping Cells):**
  * Standard octrees clip or pop when a Gaussian ellipsoid ($3\sigma$ radius) straddles a boundary or when its center $\mu_i$ drifts across a face during the $K$ inner-loop steps.
  * Each octree node of nominal side length $L$ has an expanded bounding box of side length $(1 + 2\beta)L$ (with looseness factor $\beta \in [0.15, 0.25]$). A splat remains valid inside its assigned leaf as long as its center and $3\sigma$ extent stay within the expanded box.

### D3. Tile-Based Compute Rasterizer (Replacing Hardware Quad Blending for Training)
* While [`splats/index.html`](../splats/index.html) uses the hardware rasterizer (`triangle-strip` quads) for viewing, training requires exact per-pixel transmittance $T_i$ and the ordered list of contributing splats for the backward pass.
* Training uses $16 \times 16$ pixel **compute workgroup tiles**:
  * **Forward pass:** For each tile, walk visible splats front-to-back in shared-memory batches, integrating:
    $$C_{\text{pix}} \leftarrow C_{\text{pix}} + T_i \alpha_i c_i, \quad T_{i+1} = T_i (1 - \alpha_i)$$
    Also records per-pixel median surface depth $z_{\text{med}}(\mathbf{p})$ (the depth where $T$ crosses $0.5$), `last_k` index, and the **depth spread** $(z_{q3} - z_{q1}) / z_{\text{med}}$, with $z_{q1}, z_{q3}$ the depths where $T$ crosses $0.75$ and $0.25$: how far apart along the ray the middle half of the blending weight is, measured where $T_{\text{final}} < 0.25$. The spread is the compare view's *Depth spread* mode and the validation log's mean (over the measured pixels): at 5.5k steps, 9.4% on jardin7, 17.5% on jardin5, 20.5% on gare, whose map is high over most of the image, while jardin7's is high only at depth edges. A weighted standard deviation of $z$ was tried first: tails of little weight saturate it everywhere.
  * **Backward pass:** Walk contributing splats back-to-front up to `last_k`, recovering $T_i = T_{i+1} / (1 - \alpha_i)$, computing parameter gradients, and reducing gradients across the $16\times 16$ tile in `var<workgroup>` shared memory before a single `atomicCompareExchangeWeak` per workgroup hits global memory.
  * **Depth distortion loss** (`depth_dist`): removed. Mip-NeRF 360's $\lambda \sum_{i,j} w_i w_j |m_i - m_j|$ per pixel ($m = \ln z$), with a linear back-to-front gradient; it lowered the held-out PSNR on every v3 scene (E2). Its kernel code, derivation and closed-form test are in git at `51f0f77` (`src/shaders.wgsl` tile_backward, `e2e/tiles.e2e.js`).
  * **Per-tile splat lists (GPU, one submission):** `project_splats` writes each sorted slot's tile rect (the tiles whose pixel centers meet the bounding box of the 3σ ellipse, half-widths $3\sqrt{\Sigma_{xx}}, 3\sqrt{\Sigma_{yy}}$, padded by 0.1 px) and counts entries per tile; `tile_scan` (prefix sum) → `[start, count]`; `tile_fill` appends slots unordered; `tile_sort` (one workgroup per tile, rank-by-binary-search merge sort) restores depth order. Forward/backward then walk only their tile's list: bit-exact vs scanning every visible splat, ~44× fewer splat–tile tests at `ds=4`. List capacity follows an asynchronously read entry count (grown ×2 on overflow). The rect's first tile in x and in y used to be one too low (`floor` for `ceil` in `tile_span`): 2.1–2.4× the entries on the v3 scenes at the 300k cap (8 training cameras each), renders unchanged (the extra tiles hold no pixel inside the box; forward output byte-identical), steps back to back without rebalances 5.8 → 3.6 ms on gare, 7.0 → 4.5 on jardin7, 4.7–5.1 → 2.7–2.9 on jardin4/5/6 (Chrome, M5 Pro). [`e2e/tiles.e2e.js`](e2e/tiles.e2e.js) checks every rect against the exact box.
  * **Subgroup reduction:** with the `subgroups` feature, backward sums each entry's contributions with `subgroupAdd` before the workgroup atomics (`// @sg` / `// @nosg` lines selected in `loadShaders()`; `&nosg` forces the fallback). The 24 sums (12 grads, 12 moments) then go to workgroup memory one per lane: on lane 0 alone, the CAS loops (23 then) were half of the backward time (gare, 300k splats, 270×480, Chrome, Apple M5 Pro: backward 6.65 → 3.43 ms, step 9.1 → 6.1 ms). [`e2e/tiles.e2e.js`](e2e/tiles.e2e.js) checks the subgroup path against the per-thread one, slot by slot.
  * Measured (Chrome, M1 Pro, 153k-splat jardin3): ~15 ms/step at 478×268, ~20 ms at 956×537, ~55 ms at full 1912×1074 (previously 115–200 ms at 478×268 and a GPU timeout at full resolution).

### D4. Parameterization (Phase 1: SH Degree 0, Brush-Compatible)
* Start with **SH degree 0** (**14 `f32` per splat**, padded to 16 `f32` = 64 bytes for clean `vec4f` alignment in WGSL):
  * `pos`: `vec3f` world space $(x, y, z)$, plus `opacity`: `f32` in logit space $\ln(\alpha / (1 - \alpha))$
  * `scale`: `vec3f` in log-space $(\ln s_x, \ln s_y, \ln s_z)$, plus `leaf_id`: `f32`
  * `quat`: `vec4f` $(q_w, q_x, q_y, q_z)$ normalized
  * `f_dc`: `vec3f` SH degree 0 coefficients where $c_{\text{rgb}} = \text{clamp}(0.5 + C_0 \cdot \mathbf{f}_{\text{dc}}, 0, 1)$ with $C_0 = 0.28209479177$, plus `flags`: `f32` (unused, written 0)
* **Why `f_dc` storage:** Matches Brush and standard 3DGS `.ply` files directly for lossless import/resume/export.

---

## 3. Handover from COLMAP, Resuming from `.ply`, & Octree Serialization

Because `sparse/images.bin` already embeds the exact filenames of all registered images, **no offline file conversion is needed** between `colmap image_undistorter` (which produces `gs/` in [`video2splat.py`](../../../brush/video2splat.py)) and `gaussopt/`.

### 3.1 Path A: CLI Bridge (`gaussopt/serve.py`)
A lightweight Python script (`gaussopt/serve.py`) acts as the local bridge:
```bash
./gaussopt/serve.py /Users/skal/gskal/skal65535/Hierarchical-Localization/datasets/jardin1_gs [--ply splats/jardin1_05000.ply]
```
1. **HTTP Mounting:** Serves `skal65535.github.io/` on `http://localhost:8000/`, mounts the target `gs/` directory at `/dataset/`, and exposes `GET /dataset/info.json` (listing existing checkpoints: `*.ply`, `splats/*.ply`, `splats_*/*.ply`, `*.splat`, `splats/*.splat`, `*.gosnap`, `splats_*/*.gosnap`, in the dataset directory and, when it is named `gs/`, in its parent).
2. **Direct Binary Ingestion (`src/colmap.js`):**
   * Fetches `/dataset/sparse/cameras.bin` $\to$ intrinsics $(W, H, f_x, f_y, c_x, c_y)$.
   * Fetches `/dataset/sparse/images.bin` $\to$ world-to-camera $(q, t)$, observed `point3D_id`s (to compute each camera's near-field depth guard $z_{\min}(c)$), and image filenames.
   * Fetches `/dataset/sparse/points3D.bin` $\to$ initial 3D positions $(x, y, z)$ and RGB colors.
   * Streams `/dataset/images/<filename>` via `createImageBitmap`.
3. **Checkpoint & Octree Export (`POST /save`):**
   * Exposes `POST /save?name=splat_05000.ply` so the browser can write `.ply` / `.splat` / `.octree.json` files directly into `<dataset>/splats_gaussopt/` on disk.
4. **Project folders (`--projects DIR`, default `~/gaussopt-projects`):** `GET /projects.json`, and `GET` / `PUT` / `DELETE /project/<name>/<path>` (`?list` on a directory), the store of §3.6 for any browser. Uploads are streamed to a temporary file and renamed. Every write (`POST /save` too) is refused unless `Host` names the server as localhost and `Origin`, when present, is the server itself, so another site open in the browser cannot write to disk through it (`tests/serve_test.py`).

### 3.2 Path B: Zero-Server Static Browser Mode (Drag-and-Drop / Folder Picker)
When running statically on `https://skal65535.github.io/gaussopt/`:
* Dragging and dropping a `gs/` directory (or picking it via `<input type="file" webkitdirectory>`) maps the dropped `File` handles by relative path (`sparse/*.bin`, `images/*`, `*.ply`) into `src/colmap.js`.

### 3.3 Resuming Training from a `.ply` / `.splat` Checkpoint (with COLMAP Cameras)
Gauss-Opt supports resuming or warm-starting training from any 3DGS `.ply` (including Brush exports like `jardin1_05000.ply`) while keeping the registered COLMAP cameras and ground-truth images from `gs/`:
1. **Header-Driven `.ply` Parser:** Parses arbitrary binary little-endian `.ply` property orders (e.g., Brush's alphabetical property order where `f_dc_*` and `f_rest_*` precede `opacity, rot_*, scale_*, x, y, z`), extracting `(x, y, z, scale_0..2, rot_0..3, opacity, f_dc_0..2)` directly in COLMAP world coordinates.
2. **Warm-Start Learning Rate Ramp:** Applies a 50-step linear learning-rate warmup so existing splats accumulate fresh Adam/SA moments before taking full-size steps.

### 3.4 Octree-Accelerated KNN Initialization (When Starting from `points3D.bin`)
1. Insert `points3D.bin` points into the Loose Octree first ($< 2\text{ ms}$ in JS).
2. Compute 3-NN distances locally within each point's leaf, initializing 30,000 points in $\sim 50\text{ ms}$ with zero offline preprocessing.

### 3.5 Saving the Loose Octree & Using It for Visualizer Speed-Up (`gaussopt/` & `splats/`)
Because `LooseOctree.build()` reorders the splat array so every leaf $l$ occupies a contiguous slice `[leaf.start, leaf.start + leaf.count)`, the octree hierarchy is a lightweight index over the already-compacted splat array:
1. **Embedded in `.ply` Checkpoints (`element octree_node <M>`):**
   * When exporting a `.ply` from Gauss-Opt (`exportPlyBuffer(params, count, octree)`), we append a second standard PLY element block after `element vertex <N>`:
     ```text
     comment gaussopt_octree_v1 beta=0.20 max_leaf=256 max_depth=10 cx=... cy=... cz=... radius=...
     element octree_node <M>
     property float cx
     property float cy
     property float cz
     property float half_size
     property int is_leaf
     property int first_child_or_start
     property int count_or_depth
     property int state
     ```
   * Fields missing from the comment take the `LooseOctree` defaults (`OCTREE_DEFAULTS`: `max_leaf` 256, `max_depth` 10, `beta` 0.20).
   * Standard 3DGS viewers read `element vertex <N>` and ignore `element octree_node <M>`, while `parsePlyBuffer` in `gaussopt/` and `splats/` restores the entire `LooseOctree` in **$< 0.2\text{ ms}$** without rebuilding or re-compacting.
2. **Standalone `.octree.json` Export / Import:**
   * `octree.toJSON()` / `LooseOctree.fromJSON(json)` exports/imports the serialized node flat-array (`nodes`, `beta`, `maxLeafSize`, `sceneCenter`, `sceneRadius`) as a compact JSON sidecar (`save .octree` button in UI).
3. **Visualizer Speed-Up (in both `gaussopt/` and `splats/index.html`):**
   * Instead of running a global $O(N)$ radix sort over all $N$ splats every camera move, the visualizer uses **Octree Hierarchical Frustum Culling + `rank ^ mask` Front-to-Back Traversal**:
     * Subtrees completely outside the view frustum are culled in $O(1)$ at high-level internal nodes (skipping $50\%\text{--}80\%$ of splats in a single test when zoomed in).
     * Visible leaves are visited in exact front-to-back order (`rank ^ mask`) and sorted locally inside each leaf (`sort_name = 'Octree'` in [`splats/index.html`](../splats/index.html)).

### 3.6 Project Folders (`src/project/`, Project phase)
Everything a project produces goes into one folder, laid out like `video2splat.py`'s `OUT/` so COLMAP and serve.py read it: `project.json` (format, source video, one entry per phase: options and counts), `images/` (the keyframes as extracted, distorted), `sharpness.csv` (every sampled frame), `sparse/0/` (the distorted model of `images/`, the SfM camera, e.g. OPENCV; `colmap model_analyzer` reads it), `gs/` (the trainer's undistorted dataset), `splats/` (snapshots, `.ply`, `.octree.json`) and `cache/` (the SfM session, to resume repairs). The worker writes `images/`, `sharpness.csv`, `sparse/0/`, `gs/` and `cache/` as they are made; the page writes `project.json` (last, after each change) and `splats/`.
* **Where** (`store.js`, one interface: read, write, list, remove): a folder the user picks (File System Access API, Chrome / Edge; the handle is kept in IndexedDB and offered again after a reload, which needs a click to re-grant write permission), or a project on serve.py (any browser, local). Without either, **Download as .zip** packs the same layout (`zip.js`, stored, no ZIP64). The keyed OPFS session cache of DESIGN-sfm §4 stays the default when no folder is chosen: copying every keyframe again into the browser's storage would double its size.
* **Tested** by `e2e/project.e2e.js`: a 12 s clip to 300 training steps, a new project on serve.py (the files on disk, `colmap model_analyzer` on `sparse/0/`), a reload without the video that reopens it at the same step and repairs, the same project into a folder handle (the browser's private storage hands out the same `FileSystemDirectoryHandle` a picked folder does, so the folder path is covered headless), and the `.zip` (Python's `zipfile`).
* **Attaching** a folder: an empty one is filled with what the page has (the worker writes its whole session, `flush`; the trainer a snapshot); an existing project is reopened: its SfM session (repairs work without the video), its `gs/` dataset and the snapshot `project.json` names as the latest. Snapshots, autosnaps and saves then go to `splats/`.


---

## 4. Ray Integration & Sub-Task Training Modes

$$\hat{C}(\mathbf{p}) = C_{\text{fg}}(\mathbf{p}) + T_{\text{fg}}(\mathbf{p}) \cdot \Big( C_{\text{active}}(\mathbf{p}) + T_{\text{active}}(\mathbf{p}) \cdot C_{\text{bg}}(\mathbf{p}) \Big)$$

* **Mode A: Full-Ray Octree Traversal (Global View Step):** All visible octree leaves for the selected training camera are queued in front-to-back order.
* **Mode B: Block-Coordinate Sub-Task Optimization (Local Subtree Step):** Pre-renders frozen foreground $(C_{\text{fg}}, T_{\text{fg}})$ and background $(C_{\text{bg}})$ for a local octree subtree $B$. **Not implemented** (the leaf state 1 reserved for it is never assigned).

---

## 5. Adapting `stipple/` (WLBG) to 3DGS Growth, Floater Removal & Stochastic Optimizers

### 5.1 Mechanism 1: Soft-Cell 3D Error Moments on Existing Splats
During the backward pass, we accumulate per splat $i$:

| `stipple` (2D Voronoi) | `gaussopt` (3D Soft-Cell Splat) | Meaning |
| :--- | :--- | :--- |
| `acc` $= \sum 1$ | $W_i = \sum_{c,\mathbf{p}} w_{i,c}(\mathbf{p})$ | **Visibility mass:** effective pixels owned by splat $i$ across views |
| *(new for 3D)* | $S_i = \sum_c \big(\sum_{\mathbf{p}} w_{i,c}(\mathbf{p})\big)^2$ | **View concentration:** yields effective view count $N_{\text{eff}}(i) = W_i^2 / S_i$ |
| `r_acc` $= \sum r(\mathbf{p})$ | $R_i = \sum_{c,\mathbf{p}} w_{i,c}(\mathbf{p}) \, e(\mathbf{p})$ | **Error mass:** unexplained photometric error inside splat $i$'s footprint |
| `rx, ry` $= \sum r(\mathbf{p}) \mathbf{p}$ | $\mathbf{R}_{1,i} = \sum_{c,\mathbf{p}} w_{i,c}(\mathbf{p}) \, e(\mathbf{p}) \, \boldsymbol{\delta}_i(\mathbf{p})$ | **1st error moment:** 3D vector to the error centroid inside splat $i$ |
| `rxx, rxy, ryy` | $\mathbf{R}_{2,i} = \sum_{c,\mathbf{p}} w_{i,c}(\mathbf{p}) \, e(\mathbf{p}) \, \boldsymbol{\delta}_i(\mathbf{p}) \boldsymbol{\delta}_i(\mathbf{p})^T$ | **2nd error moment:** $3\times 3$ covariance of the error distribution |

Every $K$ steps, we compute the mean error mass $\bar{R}$ and hysteresis thresholds $T_l = (1 - h)\bar{R}$, $T_u = (1 + h)\bar{R}$:
1. **Prune ($\alpha_i < \epsilon_{\text{prune}}$):** Drops transparent/dead splats (or teleports them in `monte_carlo` mode).
2. **2nd-Moment Error-Covariance Split ($R_i > T_u$):**
   * Compute empirical error centroid $\bar{\boldsymbol{\delta}}_i = \mathbf{R}_{1,i} / R_i$ and error covariance $\mathbf{E}_i = \mathbf{R}_{2,i} / R_i - \bar{\boldsymbol{\delta}}_i \bar{\boldsymbol{\delta}}_i^T$.
   * Split along the **principal eigenvector $\mathbf{v}_{\max}(\mathbf{E}_i)$ of the unexplained error** (4-step power iteration).
   * **Asymmetric Clone (3D Lloyd Jump):** If the splat is already small, keep the parent and spawn a child at $\boldsymbol{\mu}_i + \bar{\boldsymbol{\delta}}_i$.
   * This is `split_mode` `'error'`. The default since E9 is `'shape'` (brush's): the splat chosen by $R_i > T_u$ becomes two halves at $\pm\mathbf o$ drawn within its own shape, scales / $\sqrt 2$, opacity $1 - \sqrt{1 - o}$; +0.3 to +0.9 dB and a better rendered depth on the videos (§9 E9).

### 5.2 Octree-Guided Floater Removal (Integrated in the $K$-Step Loop)
1. **Near-Camera Free-Space Carve:** Prune any splat inside a camera's near-field cone $z < 0.65 \, z_{\min}(c)$. $z_{\min}(c)$ is the 5th-percentile depth of the points camera $c$ observes (`floater_depth: 'observed'`, the default). The alternative, `'view'`, uses $z_{\text{view}}(c)$, the 5th percentile over a sample (up to 50k) of all the points in its view, which also counts points only other cameras saw (e.g. the pavement a walking camera passes over); where those are nearer, it carves less. 30k-step A/B on the v3 bench (val PSNR at step 29,500, all at the 300k cap): `'observed'` wins on jardin7 (+1.44 dB, 10 val views), jardin4 (+0.27), gare (+0.21) and jardin5 (+0.17); `'view'` wins on jardin6 (+0.42). On gare `'view'` leads early (+1.75 dB at 2.5k, +1.3 at 5.5k) and loses the lead between steps 20.5k and 26.5k. Nothing seen so far predicts jardin6, so there is no per-scene rule.
   `'observed'` carves real objects that some cameras walk past without seeing them: on gare-st-lazare all 5271 SfM points of the suitcase statue nearer than 4 units to the camera of t0055648 lie in some training camera's cone (118 cameras; 61% of the scene's points are in some cone), and the statue is gone from every arm below. Monte-Carlo's noise then fills the carved places with dark streaks, the floaters seen in the gallery's retrained gare (Adam and `mc_noise=0` leave them empty, black). `floater_spare` keeps a splat in a cone when an SfM point, whichever camera saw it, is within that fraction of the scene radius. Gare, 10k steps, Monte-Carlo, one run each (val PSNR; splats carved by the cone at step 9,800): `'observed'` 14.90 dB (1177), `'view'` 15.61 (705), `remove_floaters=0` 15.20 (0), spare 0.005 15.44 (1225), spare 0.01 15.59 (1127), spare 0.02 15.52 (757); with spare 0.005 and 0.01 the statue comes back in the held-out frame (`'view'`: a smear). On the jardin scenes, `'observed'` vs `'observed'` + spare 0.01, same protocol: jardin7 24.12 → 24.18, jardin4 17.15 → 18.66 (rerun: 17.15 → 18.63), jardin5 18.03 → 18.23, jardin6 17.68 → 17.93. Never worse on the five scenes with held-out frames, hence `floater_spare = 0.01` by default (with `'observed'`); cuisine, salon and minerals, without held-out frames, are not measured. Off-path renders with their depth: `photo_runs/tools/train.mjs` `NOVEL=1`.
2. **Multi-View Participation Ratio ($N_{\text{eff}}$ Check):** In cells intersected by $M_{\text{cell}} \ge 4$ camera frustums, prune single-view billboards with $N_{\text{eff}}(i) = W_i^2 / S_i < 1.45$ (and $W_i > 0.5$, $R_i / W_i > 0.12$).
3. **Octree Topological Isolation:** A leaf with $\le 3$ splats and no other leaf within $1.25 \times$ (sum of half-sizes) is marked isolated (state 3); only its splats with $\alpha < 0.25$ are removed (`isFloater`).
4. **Selective Sightline Opacity Decay:** Decay opacity for high-error occluders ($R_i / W_i > 0.22$).

### 5.3 Overlap Reconciliation & Octree Rebalancing
1. **Overlap Reconciliation (Moment-Matching Merge):** Merge redundant overlapping Gaussians with small relative distance and color difference.
2. **Octree Rebuild:** Re-bin splats into contiguous leaves ($N_{\text{leaf}} \le 256$).
3. **Cost:** the whole outer loop (readback, `reconcileAndGrow`, Adam gather, upload) runs on the main thread inside a step, so the page freezes for its duration. jardin5 at 300k splats: ~1 s, then ~210 ms once the near-camera floater test used a grid of the camera cones (it projected every splat into every training camera) and kept splats went into one buffer. Now 160-200 ms at the 300k cap (jardin4/5/6/7, gare), every 100 steps; ~33% of the training time since the backward got faster and the tile lists shorter (D3; steps back to back: 5.4 vs 3.6 ms per step without rebalances on gare, 6.8 vs 4.5 on jardin7). A worker is not planned: running `reconcileAndGrow` there (most of the pause; the readback and upload stay on the main thread) gains at most that 33%, while the splats it replaces keep training and their gradients and Adam state would have to be reconciled at the swap. Rebalancing less often once the count is at the cap (4.) cut it instead.
4. **Schedule (`rebalanceDue`, trainer.js):** every `rebalance_every` = 100 steps, every `rebalance_every_cap` = 400 once the count is at 99% of `max_splats`, and at each opacity reset (so the reset stays on its step; with the 100-step default those were rebalance steps already). 30k-step A/B on the v3 bench, the gap at the cap 100 (the old schedule) / 200 / 400 / 800, everything else default; val PSNR / SSIM at the last validation before the step-30,000 opacity reset (29,500 to 29,680: the later schedules shift the validation steps), and the wall time (two runs at a time on one GPU, so rough):

| Scene (val views) | 100 | 200 | 400 | 800 | wall time, 100 → 400 → 800 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| gare (49) | 16.35 / 0.497 | 16.64 / 0.498 | 16.48 / 0.494 | 16.61 / 0.493 | 282 → 232 → 227 s |
| jardin4 (110) | 17.20 / 0.423 | 18.06 / 0.455 | 18.42 / 0.468 | 18.42 / 0.471 | 370 → 341 → 320 s |
| jardin5 (54) | 18.29 / 0.430 | 18.88 / 0.457 | 19.13 / 0.468 | 19.21 / 0.474 | 307 → 215 → 293 s |
| jardin6 (65) | 17.08 / 0.347 | 17.94 / 0.380 | 18.27 / 0.392 | 18.41 / 0.397 | 261 → 187 → 197 s |
| jardin7 (10) | 23.32 / 0.734 | 23.36 / 0.737 | 23.37 / 0.737 | 23.35 / 0.736 | 276 → 244 → 253 s |

   The count is at the cap by steps 2,000–4,000. The 100 runs repeat the 0 runs of the depth distortion A/B (E2, §9) within 0.06 dB. On jardin4–6 the gap opens early (+0.5 to +0.6 dB at 10k for 400) and grows: from 20k to 29k the 100 runs stay flat or drop (jardin4 17.45 → 17.06) while the 400 ones hold or improve (jardin5 19.01 → 19.12). Why (e.g. the splats replaced at each rebalance losing their training) was not measured. 800 (run later, one run per scene, so its wall times ran beside other runs than the others' did) is +0.07 dB on average over 400 (−0.02 on jardin7 to +0.14 on jardin6, against the 0.06 dB repeat above), SSIM up on three scenes and down on two, and not faster: 400 stays the default.

### 5.4 Extra Optimizer Methods: Monte-Carlo (SGLD / MCMC) & Simulated Annealing (SA)
Three methods, selectable live in the GUI: **Adam** (`opt_method = 0`, URL `opt=adam`) and the two below, reworked in E6 (§9), which measures them. Monte-Carlo with the stipple seeds (E4) is the default since E6.

1. **Monte-Carlo (`opt_method = 1`, `monte_carlo`), as [3DGS-MCMC](https://arxiv.org/abs/2404.09591) (Kheradmand et al. 2024):**
   * **L1 terms** (`mc_opacity_reg`, `mc_scale_reg`, 0.01 each as the paper): $\lambda_o \overline{|o|} + \lambda_s \overline{|s|}$ added to the loss (the mean over the splats, as the photometric loss is a mean over the pixels), in `adam_step` (`tu.reg`). Splats that do not earn their opacity fade, and die.
   * **SGLD noise** on the centers of the nearly transparent splats only (eq. 9): $\boldsymbol{\mu}_i \mathrel{+}= \lambda_{\text{noise}}\, \eta_{\text{pos}}\, \sigma(-k(\alpha_i - t))\, \Sigma_i \boldsymbol{\xi}_i$, $k = 100$, $t = 0.005$, $\eta_{\text{pos}}$ the current (decayed) position learning rate, `mc_noise` $= \lambda_{\text{noise}} = 10^6$ (the paper: $5 \cdot 10^5$).
   * **Relocation** (§4.2): at each rebalance the dead splats (pruned, floaters) are not dropped but moved onto live ones drawn with probability $\propto$ their opacity (`mc_relocate = 'opacity'`, the paper; `'error'`: $\propto R_i$). A splat drawn $j$ times becomes $N = j + 1$ copies at its place with $o' = 1 - (1 - o)^{1/N}$ and scales times $o / \sum_{i=1}^{N} \sum_{k=0}^{i-1} \binom{i-1}{k} (-1)^k o'^{k+1} / \sqrt{k+1}$, which keep its opacity at the center and its integrated opacity (`relocate`, `relocatedOpacityScale` in `octree.js`; `tests/optimizers_test.js`). No opacity reset (the L1 term does its work). Stipple seeds (E4, `stipple_seed`) take the dead splats' places first.
   * Before E6: $(1 - \alpha_i)^2$-gated noise of a fixed size on every splat (no learning rate factor), teleports of the dead onto high-error splats as unscaled jittered copies, and the opacity reset.
2. **Simulated annealing (`opt_method = 2`, `sim_anneal`), at the rebalances (`anneal.js`):**
   * **Energy:** over the window since the last rebalance, a splat's mean footprint error $e_i = R_i / W_i$ (§5.1), over the visible splats' mean.
   * **Metropolis test:** a jump made at the last rebalance is judged on $\Delta e_i$ less the median $\Delta e$ of the visible splats that did not jump (the training lowers every $e$ meanwhile): kept if $\le 0$, else with probability $\exp(-\Delta e / T)$, else undone. Then `sa_share` (0.25) of the visible splats jump by $T \cdot$ `sa_size` $\cdot \mathbf{R}_i \mathbf{s}_i \boldsymbol{\xi}$. $T = T_0 \gamma^{t/50}$ (`sa_temp`, `sa_cooling`; `sa_reheat`: back to $T_0$ every $\max(1000, 10K)$ steps).
   * **Split acceptance:** splits with $R_i \in [T_l, T_u]$ are accepted with probability $0.35 \exp(-(T_u - R_i) / (T \bar{R}))$.
   * Before E6 the test ran in `adam_step` at every step, against the splat's error in the *previous step's* view: one view per step, so it compared views, not jumps.
3. **E7's options** (§9, off by default): `cap_lease` (a larger budget until a consolidation at the end of a planned length `train_steps`), `region_repair` (stuck regions split and draw relocations more), `polish` (no Monte-Carlo terms at the end). None gained.

---

## 6. Live Visualization & Interactive Debugging (`src/render.js`)

1. **Camera Modes (`#main-canvas`):** Free-View Orbit Camera (`theta, phi, radius, fov`), Snap-to-COLMAP View (`cam_idx`), and `hold target` / split-screen wipe comparing $I_{\text{gt}}$, $\hat{I}$, and $|I_{\text{gt}} - \hat{I}|$.
2. **Splat Shape & Diagnostic Color Modes:**
   * Shapes: `gaussian`, `cone`, `ellipsis`, `small_dot`, `axis`.
   * Colors: `learned_rgb`, `octree_leaf`, `wlbg_error` ($R_i$), `view_div` ($N_{\text{eff}}$), `depth`.
3. **3D Overlays:**
   * `3D octree boxes`: Nominal ($L$) or loose ($(1+2\beta)L$) leaf wireframes colored by state.
   * `COLMAP frustums`: Camera pyramids colored by train/val/active status.
   * `WLBG split axes`: 3D principal error axes on newly split splats.
4. **Side-Canvas Inspector & Telemetry (`#side-canvas`):** Rolling train PSNR curve with held-out validation PSNR points, and Octree/WLBG rebalance stats; the SA temperature $T(t)$ is shown in the status line only, not plotted.
5. **Validation (`?val=N`, every N steps):** the held-out views (never trained on) are rendered at training resolution and scored by PSNR, SSIM (11x11 Gaussian window, as the 3DGS code) and PSNR at 1/4 of that resolution (`metrics.js`), each the mean over the views; all three are logged and kept in `valHistory`. PSNR is dominated by large areas; the 1/4-resolution PSNR shows coarse agreement only (colors, shapes, small pose errors forgiven) and SSIM structure and sharpness, so a variant that wins on one and loses on another trades detail for structure. Held-out views (`?val_split=`, applied on load): `every8` (default) is one view in 8, each between two training views, so it only tests interpolation along the path; `block` holds out a contiguous tenth of the path around its middle (name order), views away from any training view, where floaters that hide from the training views show.
6. **Layout and phases (`src/ui/`):** one window-high grid (`gaussopt.css`): a header, a **stepper** of the project phases (Project · Video · Keyframes · Cameras · Train · Export), the 3D view with its toolbar (the keyframe contact sheet `frames.js`, shown over the view in the Keyframes phase; hold target; a *view* popover with the Visualization and Camera controls), an **inspector** on the right with the current phase's controls and its next-step buttons pinned at the bottom, and a **dock** under the view with Timeline (the video's frame strip), Charts (the side canvas) and Log tabs. Each step's state (todo, running, done, stale, skipped) and one-line summary are derived from what is loaded and running (`phases.js`, never stored); a finished step moves the inspector forward, never back. Loading a COLMAP dataset marks Video and Keyframes skipped and Cameras done. The canvas backing store takes the dataset's aspect (long side 960) and is scaled to fit the view (`shell.js`). Every control has a tooltip: the GUI panels pass a table of texts by key or folder name (`mkGui(..., { tips })`, `common/controls.js`), HTML elements a `title`; one styled popup shows both. Below 1000 px the columns stack and the page scrolls. All lines go to one log (`log.js`), tagged by source (app, sfm, train) with a filter.

---

## 7. Adversarial Design & Code Review Mitigations

| ID | Risk / Failure Mode | Concrete Mitigation Built into Implementation |
| :--- | :--- | :--- |
| **A1** | **COLMAP vs. `splats/index.html` Camera Convention Mismatch** | Standardized all projection and EWA Jacobian math on **OpenCV/COLMAP camera space ($+Z$ forward)**: $u = f_x (x/z) + c_x$, $v = f_y (y/z) + c_y$, while supporting $z_{\text{sign}} = -1$ in `LooseOctree.sortIndexFrontToBack` for `splats/index.html`. |
| **A2** | **Far-Field SfM Outliers Crushing Octree Resolution** | Scene center / radius from the **10th–90th percentile** of point coordinates (the root box still bounds all finite points). |
| **A3** | **Backward Transmittance Division-by-Zero ($T_i = T_{i+1} / (1 - \alpha_i)$)** | Clamped $\alpha_i \le 0.99$; forward integration stops when $T < 10^{-4}$, recording `last_k` per pixel. |
| **A4** | **Workgroup Shared-Memory Contention in Backward Pass** | Contributing splats are loaded in batches of 32 into `var<workgroup>` shared memory and reduced across the $16\times 16$ tile before at most 1 global atomic update per splat per tile. |
| **A5** | **2D$\to$3D EWA Chain Rule Overhead** | Pixel threads accumulate only 2D gradients (`dL/drgb`, `dL/dop`, `dL/duv`, `dL/dconic`); the 2D$\to$3D EWA covariance & quaternion chain rule runs once per splat in `adam_step`. |
| **A6** | **Suffix Color Transmittance Scaling in `dL_dalpha`** | Since `accum_rec` tracks the unattenuated suffix color $\tilde{C}_{>i} = T_{i+1}^{-1}\sum_{j>i} T_j \alpha_j c_j$, `dL_dalpha` uses exact $T_i \cdot (\text{dL\_dC} \cdot (c_i - \tilde{C}_{>i}))$ without double-dividing by $(1 - \alpha_i)$. |
| **A7** | **Column-Major Quaternion Rotation Gradient (`dL_dq`)** | Matched skew-symmetric cross-terms `(G1.z - G2.y)`, `(G2.x - G0.z)`, `(G0.y - G1.x)` to WGSL's column-major `mat3x3f` layout. |
| **A8** | **Stale Adam Momentum / SA State Across Octree Reordering** | Adam state is carried across rebalances: `reconcileAndGrow()` returns `srcIdx` (source splat of each survivor, -1 for new children) and `build()` records `lastPerm` (its leaf-order permutation); the trainer gathers m/v through both. New splats start from zero. |
| **A9** | **Concurrent `step()` / `mapAsync` Re-Entrancy** | Readbacks that touch shared state (validation, snapshot, .ply sync, and the rebalance inside `step()`) run under the step lock (`GaussOptTrainer._withStepLock()`), so they cannot overlap a step. The loss poll and the tile-list size poll each read their own staging buffer, guarded by a pending promise (no second `mapAsync` while one is in flight). |
| **A10** | **Rank-Deficient Single-View 2nd Error Moment $\mathbf{E}_i$** | Added isotropic diagonal regularization $\epsilon_{\text{iso}} = 10^{-4} s_{\max}^2 \mathbf{I}_3$ before power iteration in `reconcileAndGrow()`. |
| **A11** | **Progressive Streaming Octree Rebuild Thrashing in `splats/`** | Throttled `LooseOctree.buildIndexOnly()` during HTTP streaming to only rebuild every $\ge 16{,}384$ newly arrived splats and once upon stream completion. |

---

## 8. File Structure in `gaussopt/`

```text
skal65535.github.io/gaussopt/
├── DESIGN.md              # Architecture, decisions, adversarial review, MC/SA & Octree spec
├── DESIGN-sfm.md          # In-browser video -> cameras (SfM); its §10 lists src/sfm/, tests/, tools/, e2e/
├── deno.json              # deno task test / test:slow
├── serve.py               # Local HTTP bridge: mounts COLMAP gs/ dirs, lists .ply/.splat/.gosnap checkpoints, POST /save, project folders
├── index.html             # Page markup + bootstrap (imports src/app.js)
├── gaussopt.css           # Page layout (grid: header, stepper, view, inspector, dock), overlays
├── e2e/                   # Browser end-to-end tests (Playwright, headless Chrome; BROWSER=safari: safaridriver): the SfM
│                          #   panel (sfm.e2e.js), project folders (project.e2e.js), tile kernels (tiles.e2e.js), Save mesh and the mesh view (mesh.e2e.js); harness.js: server, clip, browser, run()
├── tests/                 # deno test: SfM unit / solver / GPU / synthetic tests, .ply octree round trip
├── tools/                 # Deno / Python command-line tools for the SfM (DESIGN-sfm §10)
└── src/
    ├── app.js             # App: GUI, dataset/snapshot loading, frame loop, cameras (orbit/drone/fly-to)
    ├── params.js          # The app's params: defaults, URL params, GUI choices and tooltips, what snapshots / project.json keep
    ├── colmap.js          # Binary parser for COLMAP *.bin + .ply/.splat import/export (with embedded octree)
    ├── math.js            # Shared math: vec3, quaternions, 3x3 eigen, OpenCV view matrices, pinhole projection, orbit frame, intrinsics fit
    ├── octree.js          # Loose Octree build, JSON/PLY serialization, front-to-back sort, WLBG/MCMC/SA growth
    ├── render.js          # Pure draw layer: free-view splats, 3D octree & camera wireframes, side-canvas telemetry
    ├── gpu.js             # WebGPU compute helpers: bind groups, one-dispatch passes, pipelines from a .wgsl file
    ├── drone.js           # 'Drone fly' free camera: angle-mode quadcopter model (mouse stick, arrows forward/back + yaw, E/Q climb/descend)
    ├── demo.js            # 'Demo' free camera (G): slow elastic flight along the capture path; any input hands over to the drone
    ├── trainer.js         # GPU/CPU hybrid training loop (Adam, Monte-Carlo SGLD, Simulated Annealing)
    ├── metrics.js         # Validation metrics: PSNR, SSIM (3DGS convention), PSNR at 1/4 resolution
    ├── pace.js            # FramePacer: training steps per display frame, from their GPU time
    ├── shaders.wgsl       # WGSL kernels: projection, tile forward/backward+WLBG moments, Adam/MC/SA update, render (+ shared _* sections)
    ├── snapshot.js        # .gosnap full training-state snapshot codec (splats + Adam + moments + octree)
    ├── stipple.js         # E4: residual stippling -> unprojected seed splats (reuses ../stipple/ Voronoi)
    ├── anneal.js          # sim_anneal: the Metropolis test of the centers' jumps at the rebalances (§5.4)
    ├── regions.js         # E7: fixed cells of the scene, their window error (region log), stuck regions (repair)
    ├── mesh.js            # E5: a mesh from the splats: rendered depth -> TSDF fusion -> surface nets (+ mesh in the loop)
    ├── mesh_simplify.js   # E5: quadric edge collapse (surface nets puts ~2 triangles in every voxel)
    ├── mesh_worker.js     # E5: simplify + .glb off the page's thread (Save mesh)
    ├── glb.js             # E5: meshes as binary glTF (16-bit positions, vertex colors or a JPEG texture), read back for the viewers
    ├── mesh_texture.js    # E5: the mesh's texture: charts, atlas, the photos baked into it
    ├── mesh_image.js      # E5: the texture between RGBA, JPEG and GPU mip levels (app and gallery)
    ├── project/           # Project folders (§3.6): store.js (folder / serve.py / memory), project.js (manifest), zip.js
    ├── ui/                # Page code (DOM; engine modules stay DOM-free): shell.js (layout, stepper, inspector, dock),
    │                      #   phases.js (phase states, DOM-free), log.js, frames.js (keyframe contact sheet),
    │                      #   project_panel.js (Project phase), sfm_panel.js (the video phases' controls)
    └── sfm/               # In-browser SfM: worker, GPU SIFT / matching, mapper (DESIGN-sfm §10)
```

---

## 9. Experiments

### E1 — Depth order: octree traversal vs exact z-sort (closed, 2026-10-20)
* **Setup:** `jardin3_gs`, `splat_05000.ply` and a step-8088 snapshot, 6 COLMAP cameras (0, 16, 34, 69, 85, 107). Each variant is rendered with the training forward pass (`tile_forward`) and compared (PSNR) against an exhaustive exact $z$-sort over the **same** culled set, under the step lock (an early version raced with training steps and gave non-reproducible numbers).
* **Results (dB vs exact $z$, same set; higher = closer):**

| Variant | Extra cost | cam 16 | cam 34 | cam 69 | cam 85 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Octree leaf order + intra-leaf sort (`rank ^ mask`) | — | 35.00 | 37.11 | 34.80 | 27.53 |
| + weighted child order (ray-aligned axis flipped last) | 0 | 35.54 | 37.69 | 35.08 | 29.18 |
| + cross-leaf sliding window W=64 (CPU) | 2–10 ms | 35.19 | 37.68 | 35.34 | 27.62 |
| + GPU chunked insertion refine ×32 (reach ≈ ±500) | 5 ms GPU | 36.23 | 38.54 | 36.35 | 29.15 |
| Exact radial (distance to camera) order | — | 41.50 | 45.61 | 44.35 | 44.37 |
| **Octree cull + radix sort by $z$** | **0.7–2.7 ms total** | **exact** | **exact** | **exact** | **exact** |

* **Findings:**
  1. Culling is exact: octree and exhaustive visible sets match once both use the same opacity cut / frustum margin.
  2. Leaf traversal order is a valid visibility order for points, but splats have extent: two splats in neighbouring leaves overlap on screen while lying on different rays, and the leaf order decides arbitrarily. Mis-orders span whole leaves (mean rank displacement 11–18% of the list), so **no local fix** (sliding window, GPU refine) recovers more than ~1.5 dB.
  3. $z$-order and radial order agree to 41–46 dB, so $z$-sort is a valid reference — the octree order really is worse, not just different.
  4. Weighted child order is free and strictly better (+0.3 to +1.7 dB); kept for traversal (also used by `splats/`).
* **Decision:** training and display use octree culling + exact radix sort (bit-exact vs the exhaustive sort, 10–14× faster: 0.7–2.7 ms vs 7–30 ms for 27k–112k visible splats). Sliding window and GPU refine were removed. The "Run Exp E1" button that checked octree+radix against the exhaustive sort in the page (99.9 dB) was removed once settled; `tests/ply_octree_test.js` checks the same visible sets on the CPU (`sortGlobalExact`).
* **Open:** does exact order improve *training* PSNR vs the old leaf order? (A/B on held-out views, once validation PSNR exists.)

### E2 — Depth distortion loss from step 0 (closed, 2026-10-23)
* **Setup:** the v3 bench (gare, jardin4–7, 300k-splat cap), 30k steps, `depth_dist` = 0 / 0.1 / 0.3 from the first step, everything else default. Val PSNR / SSIM / mean depth spread at step 29,500 (not an opacity-reset step).
* **Results:**

| Scene (val views) | 0 | 0.1 | 0.3 |
| :--- | :---: | :---: | :---: |
| gare (49) | 16.34 / 0.497 / 19.8% | 16.19 / 0.489 / 10.3% | 15.95 / 0.478 / 6.6% |
| jardin4 (110) | 17.19 / 0.423 / 27.5% | 16.73 / 0.403 / 16.4% | 15.99 / 0.374 / 11.3% |
| jardin5 (54) | 18.30 / 0.434 / 18.9% | 17.59 / 0.398 / 9.4% | 17.12 / 0.371 / 5.9% |
| jardin6 (65) | 17.02 / 0.349 / 14.8% | 16.61 / 0.328 / 8.5% | 15.80 / 0.296 / 6.0% |
| jardin7 (10) | 23.28 / 0.736 / 9.1% | 23.12 / 0.729 / 4.2% | 22.91 / 0.714 / 2.9% |

* **Conclusion:** the loss does what it says (the spread halves at 0.1) but costs held-out PSNR and SSIM everywhere, more with the weight, and the gap is there from 5.5k steps on. The 0 runs repeat the `'observed'` runs of the near-camera carve A/B (§5.2) within 0.14 dB (gare 16.48 then), so gare's and jardin7's −0.15 dB at 0.1 is about the noise and jardin4–6's −0.4 to −0.7 dB is not. Not tried: a late start or a ramp (e.g. once the splat count reaches the cap). The loss was removed after E3 confirmed it on sparse sets; the code is in git at `51f0f77`.

### E3 — Depth coherence on sparse photo sets (closed, 2026-10-25)
* **Problem:** on 10–13-photo sets the splats fit each photo but float between cameras: surfaces line up only near a training view.
* **Setup:** 5k steps, every 4th view held out. cuisine (12 photos, COLMAP cameras), out_scene1 (13 photos, three lenses), and a synthetic room (`photo_runs/tools/synth_room.py`: 10 cameras on an arc, a nearly plain far wall and ceiling, posters, a table, a cabinet, a sphere; exact depth per pixel, 27 probe views between neighbouring cameras, the geometry in `scene_sdf.json`). Measures (`photo_runs/tools/train.mjs`): held-out PSNR; on the synthetic room the probe-view PSNR and the **surface measure**, each splat centre's distance to the true surfaces, opacity-weighted (depth compared per pixel is biased on oblique surfaces, since a splat has one depth); on the real sets the render's depth at the COLMAP observations. Repeat noise: ~0.1 dB on the room, ~0.3 dB on the real sets.
* **Findings:**
  1. Growth is what drifts: on the room 72% of the splats are within 3 cm of a surface at 1k steps (29k splats), 52% at 5k (299k), while the training PSNR keeps rising (55 dB). The init points are on the surfaces.
  2. Floaters are not faint: mean opacity 0.41–0.46, most in front of the surface in two or more training views.
  3. Every term that moves splats along the camera rays hurt: `depth_smooth` 1 (rendered-depth smoothness, RegNeRF-style) took the splats beyond 10 cm from 6% to 20%; virtual views between cameras (`pseudo_every`) 11–21%; `depth_dist` 0.1 / 1 36% / 65% (see E2). Adam scales a lone along-ray gradient up to a full step, since the photometric one is tiny there; applied by plain gradient descent instead (`reg_sgd`) `depth_smooth` is harmless but still does not pull splats onto the surfaces.
  4. The single-view floater rule (§5.2) hardly fires on photo sets: $N_{\text{eff}} = W^2/S$ counts view *visits* since the last rebalance, $K/n$ per view (100 steps over ~10 photos). Counting views instead (`billboard_views`) changed nothing measurable.
* **Results (held-out PSNR, dB; room: probes):**

| Variant | room | cuisine | out_scene1 |
| :--- | :---: | :---: | :---: |
| default | 38.82 | 12.84 | 13.94 |
| `edge_prune` 0.05 (prune splats seen only near frame edges) | — (room val 21.98, 47k splats left) | 12.83 | 14.12 |
| `border_weight` 0.05 / 0.1 (loss ramp at frame edges) | 37.59 / 37.20 | 12.94 / 13.15 | 14.11 / 14.19 |
| `depth_smooth` 1 | 38.06 | 13.04 | 14.06 |
| `opacity_entropy` 1e-4 | 29.48 | 13.13 | 14.59 |
| `opacity_decay` 0.995 | 36.26 | 12.57 | 14.05 |
| `drop_rate` 0.25 (DropGaussian) | 39.00 | 13.31 | 14.72 |
| "sb": `ssim` 0.2, `max_splat_angle` 0.3, `max_splat_scale` 1 | 40.09 | 14.60 | 14.22 |
| sb + `drop_rate` 0.25 | 41.18 | 15.13 | 14.98 |
| sb + `drop_rate` 0.25 + `split_min_views` 2 | 38.41 | **15.97** | **15.18** |

  No effect: `flatten` 1e-4..1e-2 (2DGS-style smallest-scale penalty), regularizer weights 1e-2..1e-3, a 30k / 100k splat cap on the real sets. At 14k steps (flat from ~8k): cuisine 12.99 (default, 300k splats) / 15.25 (sb + drop) / 15.60 (+ `split_min_views` 2, 53k splats); out_scene1 13.71 / 15.02 / 15.15 (65k).
* **Conclusion:** neither edge rejection nor a depth-smoothness penalty helps in these forms. What helps is DropGaussian and larger splats with D-SSIM; on the real sets gating growth on two views helps further. The size limits carry most of it: with `split_min_views` 2 + drop + D-SSIM, the default limits give 12.65 / 13.56 (cuisine / out_scene1), angle 0.1 / scale 0.3 14.77 / 14.85, 0.3 / 1 15.97 / 15.18. The large splats fill what few photos see, and are the blurry blobs seen from outside the camera path (gallery `cuisine-sb-gated`). What few photos see (cuisine's chairs and floor) is unchanged by all of it. Every option stays off by default; sb + drop is a candidate preset for sparse photo sets only. Open: a size limit that depends on how many views see a splat.
* **On a video (jardin6, 525 frames, our cameras, `ds=2`, 66 held-out views `every8`, 10k steps; val PSNR / SSIM / depth spread):** default 17.18 / 0.325 / 17.7%; sb 16.86 / 0.337 / 23.5%; sb + `drop_rate` 0.25 16.86 / 0.331 / 25.2%; + `split_min_views` 2 16.86 / 0.331 / 23.8%. The default run repeats an earlier one within 0.01 dB. Every run is at the 300k cap by 4k steps: on a video each splat is seen from many views, so `split_min_views` 2 does not hold growth back. The sparse-set settings lose 0.3 dB on a dense video, so they do not become defaults. (Gallery: `jardin6` default, `jardin6-sb` sb + drop.)
* **Options kept** (off by default, `params.js`, in the GUI's Optimizer folder): `drop_rate`, `split_min_views`; tests: `e2e/ssim.e2e.js` (DropGaussian), `tests/split_views_test.js`. **Removed** after this study, as harmful or without effect: `edge_prune`, `border_weight`, `depth_smooth` / `depth_smooth_edge`, `pseudo_every` / `pseudo_smooth`, `opacity_entropy`, `opacity_decay`, `flatten`, `reg_sgd`, `billboard_views` (and `depth_dist`, E2). Their code and tests (`e2e/ssim.e2e.js`'s depth smoothness and border checks, `tests/pose_interp_test.js`) are in git at `51f0f77`, so the runs above can be repeated from there.
* **Sources:** [DropGaussian](https://arxiv.org/abs/2504.00773), [Binocular-guided 3DGS](https://arxiv.org/abs/2410.18822) (opacity decay, stereo consistency), [RegNeRF](https://arxiv.org/abs/2112.00724) (depth smoothness at unseen views), [FSGS](https://arxiv.org/abs/2312.00451), [SparseGS](https://arxiv.org/abs/2312.00206), [CoR-GS](https://arxiv.org/abs/2405.12110), [2DGS](https://arxiv.org/abs/2403.17888) (distortion, flat splats), [Mip-Splatting](https://arxiv.org/abs/2311.16493), [StableGS](https://arxiv.org/abs/2503.18458), [TIDI-GS](https://arxiv.org/abs/2601.09291) (view-count pruning), a [sparse-view 3DGS survey](https://arxiv.org/abs/2507.16406).

### E5 — A mesh from the trained splats (open, started 2026-10-25)
* **Goal:** a triangle mesh of the scene (`.ply`, vertex colors) next to the splats, from what the pipeline already has: the SfM cameras and the trained splats. The SfM's own points are too sparse to mesh (a few thousand on a photo set: a Poisson surface from them is a blob), and a dense multi-view stereo (PatchMatch, as COLMAP / OpenMVS do it) is a project of its own.
* **Literature, by what each needs:**

| Route | Papers | What it needs here |
| :--- | :--- | :--- |
| Rendered depth from many views, fused into a truncated signed distance volume (TSDF), then marching cubes | [2DGS](https://arxiv.org/abs/2403.17888) meshes this way; [PD²GS](https://arxiv.org/abs/2506.09663), [fVDB `mesh_from_splats`](https://fvdb-reality-capture.readthedocs.io/latest/_modules/fvdb_reality_capture/tools/_mesh_from_splats.html) | Only the forward render: `tile_forward` already writes the median depth `z_med` per pixel (`pix_state.y`). **Chosen.** |
| The same fusion, with depth from a stereo network run on rendered image pairs | [GS2Mesh](https://arxiv.org/abs/2404.01810) | A stereo network in the browser: heavy. |
| Splats trained to lie on surfaces (flat disks, depth-correct rasterization, surface regularizers) | [2DGS](https://arxiv.org/abs/2403.17888), [RaDe-GS](https://arxiv.org/abs/2406.01467), [SuGaR](https://arxiv.org/abs/2311.12775) | Changes to the trainer; E3's flatness penalty (`flatten`) did nothing on its own. |
| The level set of the splats' opacity field, or a mesh optimized during training | [GOF](https://arxiv.org/abs/2404.10772), [MILo](https://arxiv.org/abs/2506.24096) | The best published accuracy, at a cost (a tetrahedralization of all the splats, training-time meshing): not for a browser page first. |

  No WebGPU depth fusion was found; browser splat projects stop at rendering ([Visionary](https://arxiv.org/abs/2512.08478), [WebSplatter](https://arxiv.org/abs/2602.03207)).
* **Method (`src/mesh.js`):** render each camera's median depth (pixels with $T_{\text{final}} < 0.5$ only, optionally only where the depth spread `pix_state.w` is small), fuse into a TSDF grid (KinectFusion-style running average; the distance is along the optical axis, truncated at a few voxels; voxels in front of the depth are written as free space, which is what removes a floater seen by few views), extract the zero level set with surface nets (a vertex per cell, a quad per crossing edge; cells with an unobserved corner and crossings that jump a whole truncation width are skipped), then color each vertex from the views that see it. Bounds: percentiles of the rendered depth points.
  CPU only: fusion 28 ms per 960 x 540 view on a 256 x 154 x 256 grid; the room (10 views at 480 x 360) meshes in 0.8 s at 256, 1.4 s at 384. A GPU port can wait. A floater can be carved only by views that have depth behind it: against the sky or a hole ($T_{\text{final}} \ge 0.5$, no depth) it stays (`tests/mesh_test.js` needs a background for its floater to go). In the page: "Save mesh" (Export; see Export and display below); `.mesh.ply` files are not listed as checkpoints.
* **Expected limit:** the mesh is as good as the depth. E3 found the splats drifting off the surfaces as they grow (52% within 3 cm on the synthetic room at 5k steps) and the depth wrong on plain walls; the fusion averages noise out but not a bias several views agree on.
* **Results, synthetic room** (`photo_runs/tools/mesh_eval.mjs`, `train.mjs` `MESH=`; 5k steps unless said, 3 views held out but meshed too; F-score at 3 cm, grid 256 / 384 = 3.4 / 2.3 cm voxels; one run each, two repeats of the best moved it 1.3 and 1.7; by surface: grid 384, the share of its vertices within 3 cm of it / of its points within 3 cm of the mesh):

| Training | F 3 cm (256 / 384) | F 5 cm (384) | far wall: acc. / compl. at 3 cm | ceiling | table |
| :--- | :---: | :---: | :---: | :---: | :---: |
| exact depth (the method's ceiling) | 98.8 / 99.6 | 99.9 | 100 / 100 | 100 / 100 | 99 / 100 |
| default | 41.0 / 42.1 | 54.5 | 11 / 5 | 11 / 6 | 86 / 99 |
| default, 1k steps (29k splats) | 55.9 / 56.0 | 75.3 | 42 / 41 | 24 / 16 | 89 / 99 |
| `lr_pos` 5e-5 | 37.2 / 37.3 | 50.0 | 9 / 4 | 8 / 4 | 89 / 99 |
| `split_min_views` 2 | 45.8 / 48.0 | 60.6 | 8 / 2 | 26 / 21 | 77 / 98 |
| sb + `drop_rate` 0.25 | 57.7 / 59.7 | 69.6 | 54 / 56 | 10 / 5 | 90 / 99 |
| sb + drop + `split_min_views` 2 | 61.7 / 62.4 (repeats 63.7, 64.1) | 77.3 | 45 / 46 | 55 / 44 | 79 / 98 |

  The fusion settings do not matter (default training: 41.0 / 40.9 / 42.1 / 41.4 for grid 256 trunc 3, 256 trunc 6, 384 trunc 4, 256 with spread < 0.1). The training does: textured surfaces (table, cabinet, posters) mesh well in every run; the plain far wall and ceiling are what fails. The sparse-set settings help them (E3): sb + drop fixes half of the far wall, gating growth on 2 views on top half of the ceiling. Fewer splats help too (1k steps: 56). The splat centers are closer to the surfaces than the mesh is (default: 53% within 3 cm vs 36% of the vertices): a guess, not checked: `z_med` is the center depth of the splat where $T$ crosses 0.5, so an opaque floater in front sets it, and an alpha-weighted depth may do better.
* **Real sets:** cuisine and out_scene1 (sb + drop + `split_min_views` 2, grid 384) mesh in 0.9 / 1.5 s. Their SfM has no metric scale, so voxel sizes there are in SfM units (capping the bounds at 3x each view's median depth changed nothing on the corner sets below: the grid spans the content).
* **Kitchen corner shot for E5** (2026-10-25: plain walls and a sloped ceiling, a cabinet, two windows, a cluttered worktop; `corner1`: 25 photos, `corner2`: a 26 s video, the same corner; our SfM on both, every photo and all 53 video keyframes registered; `photo_runs/tools/corner.sh`, `logs/corner.txt`, renders in `photo_runs/renders/mesh_c*`). Held-out PSNR, three / two runs: photos (every 4th held out, 5k steps) default 16.75-16.78 dB, sb + drop + `split_min_views` 2 18.07-18.13; video (the page's split, 10k steps) 21.71-21.72 / 21.94-21.97. By eye (`mesh_shot.py`, seen 35° off a camera): the layout is right in all four (worktop corner, cabinet, window frames, bottles), the surfaces are rough and layered, the sparse-set settings a little smoother. The video meshes have large holes in the plain wall above the window and on the white worktop, the photo meshes few; not looked into yet (no depth there, $T_{\text{final}} \ge 0.5$, or carved by views that disagree).
* **Export and display** (2026-10-25): Save mesh writes `gaussopt_NNNNN.mesh.glb`, simplified in a worker (`mesh_simplify.js`, quadric edge collapse with border, flip and link-condition guards; no dependency) to `mesh_simplify` voxels of error (default 1), as glTF with `KHR_mesh_quantization` (16-bit positions, byte colors, unlit), turned to glTF's +Y up, the world up kept in the node's `extras` so `readGlb` turns it back; Khronos' glTF validator: no errors or warnings. Sizes, corner photos mesh (1.27M faces, grid 384): `.ply` 27.0 MB; `.glb` unsimplified 23.6 MB (10.7 gzipped); simplified at 0.5 / 1 voxel 5.4 / 4.5 MB (2.8 / 2.2 gzipped). Simplifying costs no accuracy on the room (grid 384, scored on points of the faces 1 cm apart): F 3 cm 65.8 with 464k faces, 65.9 / 66.2 / 66.6 at 0.5 / 1 / 2 cm of error with 100k / 72k / 55k faces. Viewers: the app's Visualization > mesh and the gallery's mesh menu (scenes with `mesh: true`, `mesh=` in the URL): `mesh + splats` draws the mesh's depth first and hides the splats behind it (those in front, floaters, stay), then blends the mesh under them; `mesh only`; `wireframe + splats`; `mesh shaded`: gray lit from the camera. The app draws the mesh with its depth linear in view z, as its splats are; the gallery with its reversed perspective depth (1 near). Dropping a `.glb` (or `.mesh.ply`) shows it in either. Gallery: `corner1` (photos) and `corner2` (video), sb + drop + `split_min_views` 2, meshes 4.4 / 2.7 MB beside 16.9 MB of splats. Tests: `tests/mesh_simplify_test.js`, `tests/glb_test.js`, `e2e/mesh.e2e.js` (the .glb inside the room; mesh only all gray when shaded; mesh + splats 9-10% gray, 0.7% with the splats drawn unhidden).
* **Texture** (2026-10-25; `mesh_texture.js`, `mesh_texture` = 2048 texels a side by default, 0 for vertex colors; `photo_runs/tools/train.mjs` `TEXTURE=`, `tex_eval.mjs`, logs `tex1`-`tex5.txt`):
  * **The simplifier folded the mesh.** On the corner photos mesh, 42% of the simplified area lay in folds (two neighbouring faces more than 90° apart), against 6.2% before simplifying: the TSDF surface is bumpy at the voxel scale, and the 60° limit per collapse adds up over the passes. `simplify` now also refuses a collapse that turns a face away from the original surface there (`minFacing` 0.3 against the merged area-weighted normal of the original faces): 18% of the area in folds, 340,672 faces instead of 238,637 (0.5 / 0.7 / 0.8: 13 / 9 / 8%, 396k / 494k / 576k faces). Accuracy on the room is unchanged (the 458k-face mesh: F 3 cm 66.1 / 66.2 / 66.5 at 0.5 / 1 / 2 cm of error, 107k / 83k / 68k faces).
  * **Atlas, own code:** charts grown from the largest faces left, across faces that share a vertex, while their smoothed normals (two passes face → vertex → face) stay within 60° of the seed's; each projected flat on its mean normal, turned to its principal axes, and shelf-packed at the largest texel density that fits. Folds overlap inside their chart, and the face facing the chart's normal most keeps the shared texels. On the corner photos mesh, raw normals made 85,576 charts (47,260 single faces, 8% of the texels used); smoothed ones 9,187 charts, 27% used; the room 723 charts, 38%.
  * **Bake:** each texel the mean of the views whose rendered depth is within the truncation width of it (as the vertex colors), weighted by the view's pixels per unit of surface, (cos f / z)^2. Photos at their own size (the SfM's undistorted JPEGs, 4x the training size on the corner sets), read at the level of a 2 x 2 pyramid whose pixels match the texel. Texels no view sees keep the vertex colors; the space between the charts is filled (4 rings of neighbours, then pull-push), so mip levels and JPEG blocks do not mix in black. Stored as a JPEG (quality 0.9) in the .glb, on 16-bit uvs, the vertex colors left out (glTF multiplies them in); Khronos' validator: no errors or warnings.
  * **Results** (held-out views, rendered at the training size on the CPU, on the pixels every mesh covers, sb + drop + `split_min_views` 2, size 2048; PSNR dB / SSIM, ranges over the runs of `tex4` and `tex5`):

    | set | textured | simplified, vertex colors | full mesh, vertex colors | splats |
    |---|---|---|---|---|
    | room, 3 views, 86% covered | 31.0-31.1 / 0.974 | 27.8-27.9 / 0.86 | 30.7-30.8 / 0.96 | 36.2-36.4 / 0.987 |
    | corner photos, 6 views, 92% | 17.5-17.7 / 0.59-0.62 | 18.2 / 0.67 | 18.2-18.3 / 0.70 | 18.3-18.5 / 0.71 |
    | corner video, 5 views, 90-91% | 19.6-19.7 / 0.73-0.74 | 20.0 / 0.79-0.80 | 20.4 / 0.84 | 22.4 / 0.93 |

    Where the surface is right (the room), the texture wins by 3.3 dB; on the corners it loses 0.3-0.7 dB and 0.05-0.08 SSIM to the same mesh's vertex colors: off the true surface by a voxel or more, sharp colors from one view land on the wrong spot in another (ghosts: the cabinet's edges on the wall), which the vertex colors blur away. By eye it is sharper on the corner photos' held-out view (labels, the toaster). The geometry is the limit, not the texture.
  * **What did not help** (corner sets, same runs): a tighter depth test (1 or 2 voxels instead of 4), sharper view weights (power 4 or 16, nearly the best view alone): lower on all three sets (the room's 2 voxels: equal); the splat renders instead of the photos: SSIM above the vertex colors on the video only (0.81 vs 0.80, `tex4`), PSNR below; 1024 / 4096 texels: within 0.2 dB, and full-size vs training-size photos the same score (scored at the training size, which cannot show the difference).
  * **Is an atlas library needed? No.** xatlas (`xatlas-wasm` 0.1.3, `photo_runs/tools/xatlas_uv.mjs`, a reference outside the repository) on the room: 30.88 / 0.973 vs 30.79 / 0.973 with ours at 2048 (30.70 vs 30.74 at 1024), the same; it packs tighter (56% of the texels used vs 38%) but needs 6.9 s for the uvs (ours: 0.3-0.4 s on the larger corner meshes); on the corner meshes (144k and 340k faces) it had not finished after more than 20 and 15 minutes.
  * **Cost:** 0.3-0.9 s on the room, 1.6 / 2.4 s on the corner sets at 2048 (4.3 / 6.2 s at 4096). The corner video's .glb: 5.3 MB (193k faces, the JPEG 1.2 MB), was 2.7 MB with vertex colors and the old simplifier. In the app (`e2e/mesh.e2e.js`, the room after 300 steps, seen from camera 0, a training view): the textured mesh is 1.5-2.8 dB closer to the photo than with its vertex colors (0.9-1.5 dB with the plane depth below, its vertex colors better too). Tests: `tests/mesh_texture_test.js` (the sphere's atlas, a plane baked from a view: colors, occlusion, unseen texels, gutters, the photo's level), `tests/mesh_image_test.js`, the noisy-plane fold test in `tests/mesh_simplify_test.js`.
* **Holes** (2026-10-25; `photo_runs/tools/holes.mjs`, train.mjs `HOLES=1`, offline from the saved renders: `node holes.mjs RENDERED ROOM|- OPTS...`): the points of the true surface with no vertex within 3 cm (the room; elsewhere the views' own rendered depth, 1.5 voxels), explained by the fused grid there and by each view that truly sees the point (its render: transparent, depth in front, behind, or right).
  * **The room** (default training, 5k steps, grid 384 trunc 4): 49% of the true surface missed, the plain far wall and ceiling 94%. Not missing depth (no view renders them transparent) nor carving (no view sees past them): every view's depth lies in front of the ceiling. `depth_trace.mjs` (the splats composited on the CPU, against the exact depth): the splats float 21-30 cm below the nearly plain ceiling and 7-9 cm in front of the far wall whatever depth is read (center, alpha-weighted mean, plane), but on the side walls and the sphere the center depth of large tilted splats is the error: -5.5 / -6.4 cm median on the walls, -0.7 / -1.0 cm where the ray crosses the splat's plane.
  * **Plane depth** (`tile_forward`): z_med (and the spread's quartiles) where the ray crosses the plane of the splat when it is flat (thinnest scale at most half the middle one), as 2DGS / RaDe-GS read depth. The room: F-score 43.6 -> 53.6 (3 cm; 54.4 with every splat's plane, round ones included), completeness 51 -> 61%, the side walls missed 64-65% -> 9-18%, held-out depth within 5% 83 -> 88%; the far wall and ceiling 94 -> 83 / 76% (the splats are there). Training is unchanged (z_med is not in the loss; held-out PSNR 33.7 vs 33.2-33.9 dB). `e2e/mesh.e2e.js`: 'mesh + splats' shows less of the mesh (3.8-4.1% gray instead of 9.3-9.7%: it lies on the splats' surface), its floor lowered to 2% (0.7% with the splats unhidden).
  * **The corner video** (corner2, 10k steps, the gallery's settings; no truth, the views' own depth): 28% missed, and there the grid says free space (56% of the holes 'carved'): views disagree, 28% of the other views' renders at a hole lie behind it (they see past the splats of the plain wall and worktop) and carve it, a full vote of free space against each one that sees it. **`freeWeight`**: a view's free-space vote beyond the band counts 0.1 (`meshFromTrainer`, Save mesh; `Tsdf`'s default stays 1, a view count, for the mesh in the loop): 28 -> 19% missed, the holes in the walls filled by eye (the window panes stay open); the room F-score 53.3 -> 54.3 (accuracy up: free-space-only cells now weigh < 1 and make no faces), and a floater seen by one view of 20 is still carved (`tests/mesh_test.js`). Tighter fusion bounds (the garden through the windows makes the voxel 7.9 cm): no better (5-10% percentiles: 6.8 / 5.9 cm voxels, 28 / 44% missed, the points outside counted).
  * **The plane depth cracks the corner video's mesh** (found while retraining the gallery, E7): the corner2 mesh fused from plane-depth renders is cut in many small pieces, the 'filled' wall included, where the center depth's mesh had fewer, larger holes (gallery `corner2` vs `corner2-latest`, the same training settings). Offline (`holes.mjs`'s fusion, grid 384 trunc 4, unsimplified; open edges = edges of one face; `holes_c2c` vs `holes_c2_center`, the second trained by the app at `f6964a4`, before the plane depth): plane depth 1.70M faces, 9.8% open edges, 10,560 pieces (9,754 under 50 faces); center depth 0.91M faces, 6.4%, 1,670 pieces (1,505). `freeWeight` does not change it (0.1: 10.6% / 10,291 against 6.8% / 3,033). The missed shares (28 / 19% plane, 20 / 16% center, `freeWeight` 1 / 0.1) are each against its own renders' depth, so they do not compare across the two. A guess, not checked: the corner's splats are not trained flat, so neighbouring splats' planes give depths that disagree from pixel to pixel. The synthetic room's gain from the plane depth (above) stands; it is now a trade-off, E8.
* **Next:** the plain surfaces the splats float off (the room's ceiling, far wall): no depth reading fixes them; a planar prior for the large surfaces (planes fitted to the confident depth where plain surfaces meet textured ones) is the candidate, the splats kept for the detail. And splats trained flat, so that the plane depth stops cracking real meshes: E8.
* **Mesh in the loop** (`mesh_snap`, off by default): the fused surface fed back into training, like [SuGaR](https://arxiv.org/abs/2311.12775) binding splats to its mesh or [MILo](https://arxiv.org/abs/2506.24096) meshing during training, but as a CPU step. At the first rebalance every `mesh_snap_every` steps (1000), the training views' depth is fused (`fuseTrainer`, grid 256), then every splat center where the grid is confident (inside the truncation band, each corner of its cell seen by `mesh_snap_views` = 3 views or more) moves `mesh_snap` of the way to the surface along the grid's normal, by at most one voxel (`snapToSurface`). Colors, opacities, scales and the Adam state are left alone, so the photometric loss restores the detail. A direct position edit rather than a loss term: E3 found Adam turning a small along-ray pull into a full step. The distance is the tsdf over its gradient's magnitude: the tsdf itself is measured along the cameras' axes, 1.7-2x the true distance on the test sphere. Even so 7 of 87 test points ended a voxel or more off (up to 2.8 after a full move), hence the cap. Risk: on the plain wall and ceiling the surface is wrong, so the snap can lock the error in; the 3-view rule is the only guard.
  **Results** (room, 5k steps, one run each; held-out PSNR / probe PSNR in dB; splat centers: median distance to the true surface, share within 1 / 3 cm; mesh F-score at 3 cm, grid 384; `photo_runs/logs/snap1.txt`):

| Training | val | probes | splat centers | mesh F 3 cm |
| :--- | :---: | :---: | :---: | :---: |
| sb + drop + `split_min_views` 2 (two runs) | 34.27 / 34.23 | 38.22 / 38.07 | 1.7-1.8 cm, 35-36% / 63-65% | 62.4 / 63.7 |
| + `mesh_snap` 0.5 | 33.62 | 37.07 | 1.4 cm, 42% / 67% | 60.9 |
| + `mesh_snap` 0.5, every 500 steps | 33.67 | 37.10 | 1.4 cm, 42% / 66% | 61.3 |
| + `mesh_snap` 1 | 32.08 | 34.79 | 1.4 cm, 43% / 66% | 62.0 |
| default | 33.58 | 38.75 | 2.8 cm, 17% / 53% | 42.1 |
| default + `mesh_snap` 0.5 | 31.43 | 36.00 | 2.5 cm, 25% / 57% | 38.2 |

  The snap does move the splats closer to the surfaces (within 1 cm: 35% -> 42%), but the mesh does not get better and the renders get worse (-0.6 dB held out and -1 dB at the probes at 0.5, -2.2 / -3.3 dB at 1). The loop brings no new information: the surface it pulls towards is the splats' own, right where it was right already, and wrong on the plain far wall and ceiling (snap 0.5: 43 / 43% and 50 / 37%, against 45 / 46% and 55 / 44% without). Off by default; not tried: snapping the orientation, or pulling only splats that several views agree on photometrically (an MVS-like check, the new information the loop lacks).
* **Experiment path:**
  1. Synthetic room (exact geometry): accuracy (mesh to true surface) and completeness (true surface, from the exact depth maps, to the mesh) at 1 / 3 / 5 cm, an F-score; against voxel size, truncation width, the spread filter, and default vs sparse-set training (sb + drop).
  2. Real photo sets (cuisine, out_scene1) and a video (jardin6): by eye, and the depth at the COLMAP observations.
  3. If the depth is the limit: mesh in the loop (above); a multi-view depth-consistency term in training, or flat splats (2DGS); GS2Mesh-style stereo depth as a later option. Snap also the splats' orientation (thinnest axis along the normal) if moving the centers helps.
  4. Simple real scenes shot for it (a box-like room, a table with objects: planes and edges, measurable with a tape).

### E6 — Monte-Carlo, annealing and stipple seeding, reworked (closed, 2026-10-25)
* **Question:** the two stochastic optimizers (§5.4) and the stipple seeding (E4, `stipple_seed`) had no measured result. Do they work, and can they be made to?
* **Setup:** `photo_runs/tools/opt_ab.sh` on frozen copies of the app: exp.sh's sparse scenes (the synthetic room, cuisine, out_scene1; every 4th view held out, 5k steps) and jardin6 (gs_ours, ds=2, the page's split, 10k steps); held-out PSNR; default training otherwise (the splat count reaches the 300k cap on all four). Lines in `photo_runs/logs/opt_ab.txt`. Repeats of a configuration differ by up to 0.5 dB on the room, 0.26 elsewhere.
* **What was wrong:**
  * Annealing compared a splat's error in the current step's view with its error in the previous steps' views: the difference measures the views, so its accept / undo was close to random jumps.
  * Monte-Carlo shook every splat by (1 - alpha)^2 x 0.08 of its size at every step (half-opaque ones a quarter of that), the learning rate left out.
  * The stipple seeds were skipped at the cap (`max_splats - count` was their budget), where every run spends most of its steps.
* **Results** (held-out PSNR, the change against Adam; two runs each unless one value is given):

  | | synth | cuisine | out | jardin6 | mean |
  |---|---|---|---|---|---|
  | Adam (before) | 33.46 33.84 | 12.88 12.74 | 14.08 13.96 | 17.18 17.17 | |
  | Monte-Carlo, before | +0.41 | +0.22 | +0.20 | +0.01 | +0.21 |
  | annealing, before | +0.12 | +0.12 | +0.12 | -0.05 | +0.08 |
  | Adam, after (plane depth, seeds' budget) | +0.07 | +0.01 | +0.14 | -0.02 | +0.05 |
  | Monte-Carlo (3DGS-MCMC, noise 5e5; 1 run) | -1.36 | -0.07 | -0.40 | +0.54 | -0.32 |
  | Monte-Carlo, noise 1e6 (1 run) | -0.03 | +0.13 | -0.32 | +0.50 | +0.07 |
  | stipple seeds (Adam) | -0.21 | +1.04 | +0.14 | -0.09 | +0.22 |
  | **Monte-Carlo 1e6 + stipple seeds** | +0.01 | +1.05 | +0.05 | +0.51 | **+0.41** |
  | annealing, after | -0.04 | +0.07 | +0.05 | -0.02 | +0.02 |
  | annealing, T0 0.3, jumps x4 (1 run) | +0.08 | +0.13 | -0.08 | -0.03 | +0.03 |

  Monte-Carlo variants on one run: relocation by error instead of opacity, L1 terms at 0.001, noise 2e6: within 0.1 dB of each other on jardin6 (+0.46 to +0.56); on the room 2e6 is the best (+0.02) and L1 at 0.001 the worst (-1.48).
* **Conclusion:** Monte-Carlo now gains 0.5 dB on the video at the cap and the stipple seeds 1 dB on cuisine (they now take the room the prunes free at the cap, before the splits); together, 1e6 the noise, the best configuration on every scene but the room, where it equals Adam. Which part of Monte-Carlo makes the video's gain (the L1 terms, the relocation of ~1000 dead and near-camera splats per rebalance, no opacity reset) was not measured. The old Monte-Carlo was better on the room (+0.41, both runs): a guess, not checked: its noise on every splat regularized the sparse sets, which the paper's noise on the dead only does not. Annealing is now a Metropolis test (cuisine, step 1000: 1106 jumps kept, 309 undone) but changes nothing; why was not measured. Defaults: Monte-Carlo with the stipple seeds (`opt=adam`, `stipple=0` for the earlier ones); `mc_noise` 1e6.
* **Not tried:** Monte-Carlo with the sparse-set settings (E3: ssim 0.2, bigger splats, DropGaussian, split gate); the old noise on every splat added to the new Monte-Carlo for sparse sets; dropping annealing.

### E7 — The cap lease, stuck regions, an Adam polish (closed, 2026-10-25)
* **Question:** at the cap the training plateaus. Does a larger splat budget during training, given back at the end, help? Are there parts of the scene that stop learning while the rest goes on, and can they be helped locally? Does plain Adam at the end (no Monte-Carlo noise or L1 terms) help?
* **Tools (off by default, `params.js`, GUI Octree folder):**
  * **Region log** (`region_log`, `src/regions.js`, `photo_runs/tools/regions.mjs`): the scene cut in fixed cells (`region_res`^3 over the 2-98% box of the centers, fixed at the first rebalance, unlike the octree's leaves), each cell's window error $e_c = \sum R / \sum W$ at every rebalance; `train.mjs` writes it with the query `region_log=1`.
  * **Cap lease** (`cap_lease`, `train_steps`, `consolidate_frac`; `trainer.js` `splatBudget`): with a planned length, the budget is `max_splats` x (1 + `cap_lease`) until the last `consolidate_frac` (0.2) of the steps, falls linearly to `max_splats` over 3/4 of that, then stays; while it falls the rebalances come every `rebalance_every` steps, and one at the end of the fall, so the count ends at `max_splats`. Over the budget, the rows of the lowest keep score go (`octree.js` `trimToCap`): each splat's visibility $W$ per window, averaged over the windows it lived with a decay of 0.7 a window (a video's window does not visit every view); Monte-Carlo relocates only within the budget, and the stipple seeds stop during the fall.
  * **Stuck-region repair** (`region_repair`, `RegionTracker`): over the last 6 windows, a cell is stuck when its error is above the scene's and fell slower than it (3 windows at each end); its splats split at $R > T_u / (1 + b)$ and draw relocations $(1 + b)$ times as often, $b$ = `region_repair`.
  * **Polish** (`polish`): with a planned length, the last `consolidate_frac` of the steps without Monte-Carlo's noise and L1 terms.
  * Tests: `tests/optimizers_test.js` (the budget's schedule and rebalances, the trim, the keep score), `tests/regions_test.js` (the cells, the stuck test, the repair's splits and relocations).
* **Do regions get stuck?** (default training, region log at `region_res` 8; the last half of the training cut in two spans A and B; $d_c$: a cell's log error change over a span less the scene's; `logs/opt_ab.txt` `reg_r1`, `models/train__*_reg_r1.regions.json`)

  | | scene error over A | slow cells stay slow, r(d A, d B) | slow cells are patches, r(d, neighbours) | mass slow in A and B, high error / low error |
  |---|---|---|---|---|
  | synth | -30% | 0.51 | 0.90 | 8% / 30% |
  | cuisine | -50% | 0.64 | 0.61 | 26% / 10% |
  | out | -35% | 0.43 | 0.63 | 29% / 12% |
  | jardin6 | -7% | 0.28 | 0.17 | 8% / 13% |

  No cell stops learning (at most 2% of the mass without a drop over A). On the sparse photo sets about a quarter of the visible mass is above the scene's error and learns slower in both spans, in patches; on the video the scene plateaus as a whole, the cells' differences small and neither lasting nor spatial.
* **Results** (held-out PSNR, dB, each run; `photo_runs/tools/opt_ab.sh` as E6, `@STEPS` in its query for the planned length; the default's runs are E6's two and `reg_r1`):

  | | synth | cuisine | out | jardin6 |
  |---|---|---|---|---|
  | default (Monte-Carlo + seeds) | 33.60 33.73 32.75 | 13.79 13.94 13.81 | 14.15 13.98 14.28 | 17.69 17.68 17.66 |
  | `cap_lease` 0.3 | 33.85 33.31 | 13.83 13.69 | 13.83 13.90 | 17.71 17.70 |
  | `cap_lease` 0.6 | 33.61 33.78 | 13.64 13.76 | 13.99 13.78 | 17.68 17.70 |
  | `region_repair` 1 | 32.33 32.79 | 13.77 13.80 | 14.25 14.14 | 17.68 17.67 |
  | `polish` | 32.84 33.55 | 13.94 13.88 | 14.08 14.17 | 17.71 17.69 |

  Every run ends at 297.8-300k splats. The lease fits the training views better (cuisine 30.6-31.4 dB against 29.7 for `reg_r1`) without a better held-out view; on out_scene1 its runs are 0.15-0.36 dB under the default's mean. The lease runs ran before the rebalance fix above; their counts ended at the cap anyway.
* **Conclusion:** none of the three gains beyond the repeat noise (up to 1 dB on synth, 0.3 elsewhere). The region log finds what it was built for (lasting, high-error patches on the sparse sets), but more splits and relocations there do not release them; a guess, not checked: what limits them is what few photos constrain, not how many splats they have. The video's plateau is global. Everything stays off; the region log stays as a diagnostic.
* **The gallery, retrained with the E6 defaults** (`photo_runs/tools/gallery_latest.sh`, the original settings and the same SfM, the `-latest` scenes beside the originals; `logs/gallery_latest.txt`): jardin7 22.98 -> 24.15 dB held out, jardin5 17.46 -> 18.02, cuisine 28.89 -> 29.93 and salon 27.03 -> 29.05 on the training views (no view held out). corner2 is trained with Adam as before (21.86 vs 22.05 held out) to show the mesh's new fusion: it covers more of the wall but is cracked (E5); trained with Monte-Carlo (21.76) the mesh is as cracked.

### E9 — Geometry: the trainer smears what the SfM gets right (2026-10-25; splits by shape, `lr_pos` 5e-5 and `lr_opacity` 0.01 the defaults since 2026-10-26, depth still behind brush)
* **Question:** the videos (jardin5, gare) train into a splat soup with no 3D structure, though a video has many views of everything. Poses or trainer?
* **Not the poses.** jardin5: COLMAP's poses differ from ours by up to the whole path extent over the first 1500 frames, and our trainer does no better on them (17.79 / 18.17 dB held out / training views, vs 18.23 / 18.70 on ours). Brush (10k steps, 300k cap, max resolution 960) trained on *our* poses and scored by our page (SH 0 only, so brush's view-dependent color is lost; our size caps off, see *Scoring brush* below): 19.13 dB held out, training views 18.98 (ours 18.23 / 18.70), and its rendered depth at the SfM points is 1.0% median relative error, 84% within 5%, against ours 4.6% / 60%; its renders have the bench, chair and branch where ours have blur. Gare is worse: depth off by 25% on average, even training views render as a near-camera smear.
* **A clean case and a measure.** `~/gskal/brush/dodecahedron`: 44 s circling a dodecahedron, two cork disks and a tile on a glossy tiled floor, two loops at two heights. Our SfM registers 90/90 keyframes, 7,566 points; the floor is a thin plane. `photo_runs/tools/video_gs.py` turns the run into a COLMAP folder (`dodecahedron/gs`: the page's exact cameras, the keyframes pulled from the video to ~46 dB of the page's own images) for brush and `train.mjs gs`. `photo_runs/tools/floor_slab.mjs`: the floor plane fitted to the SfM points, then the distance to it of the opaque splats (opacity $\ge$ 0.5) inside the camera ring, over the cameras' height: median, p90, the share more than 2% *under* the floor (physically impossible, the cleanest error signal). `splat_views.mjs` draws it from above and the side.
* **Tools (off by default):** `split_min_err` (a splat splits only if its mean L1 error under its footprint $R/W$ is above this; the split threshold $T_u = 1.35\,\bar R$ is relative, so growth otherwise never stops before the cap), `grow_rate` (at most this share of new splats per rebalance, the highest $R$ first), `split_in_plane` (split / clone offsets lose their component along the splat's thinnest axis: the error offsets are taken in each view's image plane, `shaders.wgsl`, far from a floor seen at a grazing angle). `tests/growth_gates_test.js`; each test seen failing with its code reverted.
* **Results** (dodecahedron, 5k steps, `photo_runs/tools/e9_growth.sh`, `logs/e9.txt`; floor: median / p90 / under; repeat noise ~0.2 dB):

  | run | splats | val dB | floor |
  |---|---|---|---|
  | SfM points | | | 0.7% / 2.5% / 9% |
  | brush (SH 3) / brush `--sh-degree 0` | 18k / 37k | | 0.7% / 3.2% / 7%, 0.7% / 3.6% / 9% |
  | default (Monte-Carlo + stipple) | 297k | 21.63 | 2.3% / 10.9% / 22% |
  | `max_splats` 20k | 20k | 21.93 | 1.6% / 7.8% / 22% |
  | `split_min_err` 0.08 / `grow_rate` 0.02 | 40k / 45k | 21.69 / 21.82 | 1.5% / 11.9% / 18%, 1.8% / 11.1% / 22% |
  | `split_in_plane` | 298k | 21.70 | 2.4% / 12.6% / 23% |
  | no growth (Adam; split, stipple, merge off) | 7k | 20.13 | 0.8% / 2.8% / 14% |
  | Monte-Carlo only (no split, no stipple) | 6k | 21.87 | 1.6% / 8.4% / 25% |
  | stipple only (Adam, no split) | 42k | 21.63 | 1.2% / 5.4% / 18% |
  | Adam, splits only, `grow_rate` 0.02 | 18k | 22.00 | 1.3% / 5.8% / 14% |
  | Adam, `grow_rate` 0.01, `lr_pos` 5e-5 | 10k | 21.77 | 0.9% / 3.9% / 10% |
  | default, `lr_pos` 5e-5 | 298k | 22.03 | 1.9% / 8.5% / 19% |

  Without growth the trained positions keep the floor flat: the slab is made by the splats added during training, Monte-Carlo's relocation the worst, then stipple seeds, then splits. Where they split matters little (`split_in_plane` changes nothing), how many does: 10-45k splats give the held-out PSNR of 300k, faster, flatter.
* **It does not carry over to the real videos** (10k steps; val dB, depth at the SfM points: median / within 5%):

  | | jardin5 | gare |
  |---|---|---|
  | default | 18.22, 4.9% / 59% | 15.59, 25.4% / 21% |
  | Adam, `grow_rate` 0.01, `lr_pos` 5e-5 | 16.44, 3.7% / 60% (50k splats, fewer than the 76k SfM points it started from) | 13.75, 21.1% / 30% |
  | Adam, no stipple | 17.63, 6.7% / 51% | |
  | Adam, no stipple, `grow_rate` 0.05 | 16.75, 12.1% / 40% | |
  | default, `lr_pos` 5e-5 | **18.51, 3.1% / 67%** | 15.47, 25.2% / 26% |
  | default, `lr_pos` 1e-5 / 2e-6 | 18.12, 3.4% / 66% / 17.43, 4.3% / 63% | |
  | brush on our poses | 1.0% / 84% | |

  On the big scenes Monte-Carlo helps and slow growth starves them. `lr_pos` 5e-5 (from 1.6e-4, x the scene radius): +0.3 dB and better depth on jardin5, +0.4 dB and a thinner floor on the dodecahedron, gare the same; but jardin7 −0.25 dB (24.21 → 23.96) and E5's synthetic-room mesh F-score 41.0 → 37.2 with the old split, so it went in only with the split below.
* **Brush on gare** (our poses, same flags as on jardin5, scored by our page): depth 15.3% / 42% against our 25% / 21-26%, so gare's depth is better recoverable from our cameras than we get it, though brush is far from its jardin5 1.0% there too; 16.35 dB held out against our 15.59 (SH 0 only, size caps off).
* **Brush's training, from its source** (v0.3.0, `crates/brush-train/src/train.rs`), against ours: no opacity reset (ours is off under Monte-Carlo too, `trainer.js` `_rebalance`); growth every 200 steps by the positional gradient (the per-pixel $|\partial L/\partial \mathbf{u}|$ summed unsigned, max over steps / views seen, over 4e-5: 10% of those, drawn by it), until step 15k; **a grown splat becomes two halves at $\pm\mathbf{o}$, $\mathbf{o} = R\,(\mathcal N(0, 0.5)\,\mathbf s)$ drawn within its own shape, scales / $\sqrt 2$, opacity $1 - \sqrt{1 - o}$**, where ours splits along the error moments and clones at the error centroid with the parent's full opacity; dead splats (opacity < 2/255) are moved onto live ones drawn by opacity and split the same way; noise on the means of nearly transparent splats ($(1-o)^{100}$); 0.8 L1 + 0.2 D-SSIM; Adam $\epsilon$ 1e-15; opacity and scale L1 terms of 1e-9 / 1e-8 (`opac_loss_weight`, `scale_loss_weight`); `lr_scale` 1e-2 → 6e-3; init opacity 0.1-0.25, isotropic, random rotations; no size limits.
* **One difference at a time** (jardin5, 10k steps, `photo_runs/tools/e9_real.sh`; val dB, depth median / within 5%; the default 18.53, 3.3% / 67%, repeating 18.51): `ssim` 0.2 18.61, 2.5% / 71%; `lr_scale` 0.01 18.63, 2.7% / 69%; no stipple 18.63, 2.9% / 69%; `mc_scale_reg` 0 18.50, 3.2% / 66%; `mc_noise` 1e5 18.37, 3.1% / 67%; `mc_opacity_reg` 0 18.37, 3.7% / 65%; `rebalance_every_cap` 200 18.35, 3.3% / 65%; isotropic init 18.43, 3.5% / 65%; no size limits 18.55, 5.7% / 62%. New options: `split_mode` `'shape'` (brush's split, `octree.js` `splitByShape`) **18.86, 2.1% / 75%**; `grow_by` `'grad'` (brush's score: `tile_backward` sums $|\partial L/\partial u_0| + |\partial L/\partial v_0|$ per pixel into grads slot 11, `adam_step` adds it with a view count into `grow_acc`; a splat grows when the mean per view is over `grow_grad`): 2e-5 17.66, 3.2% / 65% (254k splats); with the shape split 18.37 (2e-5), 17.55 (5e-5, 131k splats). Tests: `tests/growth_gates_test.js`, each seen failing with its code reverted.
* **On the other scenes** (10k steps; cuisine: training views, no view held out):

  | | jardin5 | jardin7 | gare | cuisine |
  |---|---|---|---|---|
  | `lr_pos` 1.6e-4 (the old default) | 18.22, 4.9% / 59% | 24.21, 10.8% / 46% | 15.59, 25.4% / 21% | |
  | `lr_pos` 5e-5 | 18.53, 3.3% / 67% | 23.96, 10.0% / 48% | 15.48, 25.5% / 26% | 30.10 |
  | `split_mode` shape, `lr_pos` 1.6e-4 | 18.52, 3.4% / 65% | 24.73, 16.1% / 47% | 15.64, 25.3% / 21% | |
  | **`split_mode` shape, `lr_pos` 5e-5 (the new default)** | **18.86, 2.1% / 75%** | **24.84, 8.7% / 53%** | 15.57, 21.5% / 28% | **30.77** |
  | + `ssim` 0.2 | 18.91, 1.8% / 78% | 24.68, 6.6% / 58% | 15.52, 20.1% / 30% | 30.34 |
  | + `ssim` 0.2, `lr_scale` 0.01, no stipple | 18.97, 1.8% / 77% | 24.67, 9.6% / 54% | 15.49, 20.9% / 30% | |

  The split by shape is the fix: +0.3 to +0.9 dB and a better depth on every scene, more with the lower position rate (with the old rate it loses its depth gain on jardin7 and gare). D-SSIM adds depth (jardin7 8.7% → 6.6%) but costs PSNR on three of four scenes and ~20% time: an option. On the dodecahedron (5k steps) the split by shape does not flatten the floor: the default 21.87 dB, 2.0% / 13.0% / 19% under; shape 21.97, 2.2% / 12.1% / 24%; shape + `ssim` + `lr_scale` 0.01 + no stipple 22.27, 1.4% / 9.1% / 27%.
* **Scoring brush:** a brush PLY scored by our page (`ply=`, every learning rate 0) still went through `adam_step`'s size caps (`max_splat_angle` 0.03, `max_anisotropy` 10): they shrank brush's long thin splats (16% of them over anisotropy 10, 11-15% over the angle) and opened holes. With the caps off (`max_splat_angle=100&max_splat_scale=100&max_anisotropy=100000`): jardin5 19.13 dB held out (was 18.80), gare 16.35 (13.41), the dodecahedron 22.86 at SH 0 (`--sh-degree 0`, 37k splats; was 17.69) and 22.08 for the SH 3 model rendered at SH 0. The depth figures barely move (jardin5 1.0%, gare 16.0%). Brush leads on every scene (corrected in E10: those brush runs had trained on our held-out frames).
* **The dodecahedron's floor** (5k steps unless said; floor: median |distance| / p90 / share more than 2% under, opaque splats, `photo_runs/tools/floor_slab.mjs`; the SfM points 0.7% / 2.5% / 9%, brush 0.7% / 3.2% / 7%; our default 2.1% / 11.1% / 23%, 22.17 dB):
  - **Not the dead splats:** brush's relocation (`mc_relocate_mode` `'shape'`: each live splat drawn splits by shape, `octree.js` `relocate`; test `tests/optimizers_test.js`, seen failing with the path off) 22% under, 22.00 dB; jardin5 18.89, jardin7 24.89, gare 15.57 (noise). An option, off.
  - **Not the count:** brush-like counts (`grow_by` grad 4e-5: 22k splats; `grow_rate` 0.01 / 0.003; cap 30k) 17-21% under; with no growth at all (7.8k splats, no stipple) 20%, already 20% after 500 steps: the initial splats sink (14% of all splats under at 500 steps, the opaque ones more).
  - **Not the gradient:** `e2e/posgrad.e2e.js` checks `adam_step`'s position and scale gradients against central differences of `project_splats` (grads_2d injected, Adam's first moment read back): 0.01-0.03% on four tilted splats, one beyond the frustum clamp, the conic's path alone too; it fails with the clamp term doubled (2-16%) or the covariance term dropped (100%).
  - **Not** `mc_scale_reg` 0.05 / 0.2, `lr_scale` 0.01, `max_splat_angle` 0.01, `ssim` 0.2, `mc_noise` 0, no regularizers, `lr_pos` 2e-5, isotropic init, no carve, `split_mode` error, `lr_quat` 0.001 (all 16-19% under, 22-42k splats), nor the size caps off (with `lr_opacity` 0.01: 10-13%, 22.38-22.42 dB), nor brush's init (opacity 0.15, 0.5 x the neighbor distance, isotropic: 22.37-22.41 dB, 10-12%).
  - **The opacity learning rate:** ours was 0.05, brush's 0.01. 0.02: 22.12 dB, 21% under; **0.01: 22.44 / 22.40 dB, 0.6% / 3.3% / 11%**; 0.005: 22.20, 0.6% / 2.4% / 9%. At 10k steps: 0.05 22.04 dB, 24% under; 0.01 22.39, 15%; 0.005 22.39, 10%. By eye (the gallery's viewer) the floor's streaks and wavy grout lines go away. With 0.005, `ssim` 0.2, `lr_scale` 0.01, Adam, no stipple or `grow_by` grad add nothing (21.62-22.28 dB).
  - **On the videos** (10k; val dB, depth median / within 5%): jardin5 18.86, 2.1% / 75% → 18.75, 1.8% / 77% (0.01), 18.56, 2.0% / 75% (0.005); jardin7 24.84, 8.7% / 53% → 25.40, 10.3% / 54% / 24.96, 10.6% / 54%; gare 15.57, 21.5% / 28% → 15.56, 17.9% / 33% / 15.53, 17.5% / 34%. 0.01 the default (+0.8 dB summed over the four scenes, 0.005 +0.1).
  - **On the photo sets** the training views lose 0.6-1.4 dB (cuisine 30.80 → 29.39, salon 28.99 → 27.95, minerals 26.05 → 25.45), but held out (every 4th registered photo, `VAL`; 0.05 / 0.02 / 0.01): cuisine 13.41 / 13.15 / **13.49**, salon 16.49 / 17.00 / **17.37**, minerals 15.15 / **15.35** / 15.24: less overfit, not worse. Training is slower at 0.01: 145-191 s for the gallery's 10k-step runs against 101-157 s at 0.05.
* **Open:** jardin5 1.8% against brush's 1.0%, gare ~18% against 16%, the dodecahedron 22.4 dB at 300k splats against brush's 22.86 at 37k (crisper edges on the dodecahedron itself); semi-transparent splats still under the floor (opacity >= 0.2: 16% at 0.01). Not tried: the opacity L1 near zero with the noise kept, brush's position rate schedule (`lr_mean` 2e-5 → `lr_mean_end` 1e-6).

### E8 — Flat splats for the mesh, the plane depth as an option (planned)
* **Why:** the plane depth (E5) lifts the synthetic room's mesh F-score from 43.6 to 53.6 but cuts the corner video's mesh into pieces (6x as many); the room's plain ceiling and far wall stay wrong whatever the depth reading. Both point at the splats' shape: they are fitted to the photos, not to the surfaces. 2DGS ([Huang et al. 2024](https://arxiv.org/abs/2403.17888)) trains flat disks (surfels) with an exact ray-disk intersection, a normal-consistency term (the disk normals against the normals of the rendered depth) and a depth-distortion term, and meshes them by the same TSDF fusion of rendered depth as `src/mesh.js`; SuGaR ([Guédon & Lepetit 2024](https://arxiv.org/abs/2311.12775)) flattens 3D Gaussians onto the surfaces, meshes them (Poisson) and binds them to the mesh: the hybrid of track c, later.
* **What we already know:** a flatness penalty alone (`flatten`, E3) changed nothing; the depth distortion from step 0 (`depth_dist`, E2) cost 0.4-0.7 dB on the videos; terms along the camera rays hurt under Adam (E3, finding 3). So E8 does not repeat them as penalties.
* **Steps, cheapest first:**
  1. **`plane_depth` option** (`tile_forward`: off = the center depth, flat-only = now, a stricter flatness ratio): measured on the room (F-score 3 cm, grid 384) and on corner2 (open edges and pieces, `holes.mjs` offline, by eye in the gallery). Picks the Save mesh default.
  2. **Flat splats by construction:** the thinnest scale pinned (a fixed fraction of the middle one) instead of penalized, the plane depth on every splat. Held-out PSNR, room F-score, corner2 pieces.
  3. **Normal consistency** (2DGS): the rendered normal (the splats' plane normals, alpha-blended) against the normal of the rendered depth, from a step late enough not to cost the fit (E2's open question: a late start), applied so that Adam does not turn it into a full step (E3).
  4. **Ray-disk intersection** in the forward and backward passes (2DGS's exact footprint instead of the projected 2D Gaussian): only if 2-3 help, as it touches both rasterization passes.
* **Measures:** held-out PSNR on E6's bench (the cost), room F-score and its far wall / ceiling, corner2 open edges and pieces, the gallery's corner scenes by eye.

### E10 — Where the gap to brush lives (steps 1-2 done, 2026-10-26)
* **Why:** after E9 brush still leads on every scene it was run on, scored by our page with its size caps off and at SH 0 only (which understates it; jardin5 and gare were trained at SH 3): jardin5 19.13 dB, 1.0% / 84% against our 18.75, 1.8% / 77%; gare 16.35, 16.0% / 41% against 15.56, 17.9% / 33%; the dodecahedron 22.86 dB with 37k splats in 5k steps (`--sh-degree 0`) against our 22.39 with 300k in 10k, its floor 7% under against 15%. The brush differences tried one at a time in E9 (the split, the relocation, gradient growth, the init, the size caps, the regularizers, D-SSIM, the opacity rate) close part of it, none all of it.
* **Steps, the two diagnostics first** (each about an hour; they decide between 3a and 3b):
  1. **Our renderer against brush's.** Brush's own renders of its held-out frames (`--eval-split-every`, `--eval-save-to-disk`) for its SH 0 dodecahedron model, against our page's render of the same PLY on the same frames: their PSNR, then per pixel where they differ. Candidates: the 0.3 px² added to the 2D covariance, the opacity cap at 0.99, the cut at power 4.5 (3 sigma) and alpha >= 1/255. If the renderers disagree, so do the gradients we train on.
  2. **Our trainer from brush's solution.** Brush's SH 0 PLY loaded (`ply=`, size caps off) and trained 5k more steps by ours, against the same PLY scored untrained (22.86 dB). Holds or improves: our optimizer keeps a better solution, the gap is in getting there (3a). Drifts toward 22.4: our optimizer pulls away from it (3b).
* **Results of 1-2: most of the gap was a leak.** Brush's earlier runs trained on our held-out frames: the dodecahedron run had no eval split, jardin5 and gare used `--eval-split-every 8`, which holds out sorted index 0 mod 8 (`brush-dataset` `formats/colmap.rs`) while ours are index 4 mod 8. Retrained with 4 missing images prepended to `images.bin` (`photo_runs/tools/brush_split.py`; brush skips missing images after numbering them), so its eval frames are ours (checked name by name: 11, 46, 47), scored by our page with the caps off: the dodecahedron (SH 0, 5k) **22.41 dB** (18k splats; was 22.86) against our 22.44 at 5k, at SH 3 21.94 rendered at SH 0 (was 22.08), floor 7% under; jardin5 (SH 3 rendered at SH 0, 10k) **18.80**, depth 1.1% / 85% (was 19.13) against our 18.75, 1.8% / 77%; gare **15.71**, 18.3% / 40% (was 16.35) against our 15.56, 17.9% / 33%. Held-out PSNR is level (+0.05, +0.15 dB at brush's SH 0 only); what stays is the depth on jardin5, the dodecahedron's floor (brush 0.7% / 2.8% / 8% under, 1% above; ours 0.6% / 3.3% / 11%, 6% above) and the splat count (18k against 300k).
  1. **The renderers agree:** brush's own eval renders against our page's render of the same PLY (`photo_runs/tools/brush_cmp.py`): 43.7 dB apart on average, brush 22.44 dB against ours 22.41 on our ground truth. The difference sits on one object, the patterned tile, brush brighter in every pixel off by more than 4/255: brush clamps a splat's color only below 0 (`rasterize.wgsl`, `max(color.rgb, 0)`), ours to [0, 1] with no color gradient where clamped (`shaders.wgsl` `rgb_raw`, `wg_cmask`); 11% of brush's splats have a channel above 1. Not the covariance term, the opacity cap or the cuts.
  2. **Our trainer holds brush's solution:** from brush's PLY (caps off), 5k steps: our defaults 20.79 dB at 1k (growth 18k → 112k splats), then 22.21, 22.34, 22.48, **22.40** at 5k (training views 22.27 → 23.36, 300k splats); with `enable_split=0&enable_merge=0` 22.24 → **22.41** (111k splats: growth still runs by another path). Neither 3a nor 3b on held-out PSNR: what is left is geometry and count, for 3a.
* **3a. Placement and growth:** both trainers' splats at matched steps (500, 1k, 2k: counts, sizes, `floor_slab.mjs`, the error by region); brush's position rate schedule (`lr_mean` 2e-5 → `lr_mean_end` 1e-6, ours exponential over `lr_decay_steps` to `lr_pos_final` x); its opacity L1 (`opac_loss_weight` 1e-9; ours `mc_opacity_reg` 0.01 on the mean) with the noise kept.
* **3b. The optimizer:** one at a time from brush's PLY: the loss (ours 0.8 L1 + 0.2 L2 by default, brush 0.8 L1 + 0.2 D-SSIM), the 50-step learning-rate warmup, Adam's gradient clamp at ±1 (`adam_step` `adam4` / `adam3`; brush has none), the size caps again.
* **4. View-dependent color** (independent; scoped after 1-2): brush trains SH 3, we train SH 0, and the dodecahedron's floor is glossy. 45 more floats per splat (16 now): GPU memory, the PLY, the gallery's viewer; worth it only if 1-3 leave a gap SH could explain.
* **5. The time cost of `lr_opacity` 0.01** (independent): its 10k-step gallery runs take 145-191 s against 101-157 s at 0.05. The splats blended per pixel at both rates; if near-transparent splats explain it, a stricter `prune_opacity` or a cull at blend time.
* **Measures:** the dodecahedron (5k and 10k steps: held-out PSNR, `floor_slab.mjs`, the gallery's viewer by eye), jardin5 and gare (held-out PSNR, depth at the SfM points, `photo_runs/tools/e9_real.sh`), brush's PLYs on the same frames with the caps off.
