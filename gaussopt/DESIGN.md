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
    Also records per-pixel median surface depth $z_{\text{med}}(\mathbf{p})$ (the depth where $T$ crosses $0.5$) and `last_k` index.
  * **Backward pass:** Walk contributing splats back-to-front up to `last_k`, recovering $T_i = T_{i+1} / (1 - \alpha_i)$, computing parameter gradients, and reducing gradients across the $16\times 16$ tile in `var<workgroup>` shared memory before a single `atomicCompareExchangeWeak` per workgroup hits global memory.
  * **Per-tile splat lists (GPU, one submission):** `project_splats` writes each sorted slot's tile rect (bounding box of the 3σ ellipse, half-widths $3\sqrt{\Sigma_{xx}}, 3\sqrt{\Sigma_{yy}}$) and counts entries per tile; `tile_scan` (prefix sum) → `[start, count]`; `tile_fill` appends slots unordered; `tile_sort` (one workgroup per tile, rank-by-binary-search merge sort) restores depth order. Forward/backward then walk only their tile's list: bit-exact vs scanning every visible splat, ~44× fewer splat–tile tests at `ds=4`. List capacity follows an asynchronously read entry count (grown ×2 on overflow).
  * **Subgroup reduction:** with the `subgroups` feature, backward sums each entry's contributions with `subgroupAdd` before the workgroup atomics (`// @sg` / `// @nosg` lines selected in `loadShaders()`; `&nosg` forces the fallback).
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

### 5.2 Octree-Guided Floater Removal (Integrated in the $K$-Step Loop)
1. **Near-Camera Free-Space Carve:** Prune any splat inside a camera's near-field cone $z < 0.65 \, z_{\min}(c)$.
2. **Multi-View Participation Ratio ($N_{\text{eff}}$ Check):** In cells intersected by $M_{\text{cell}} \ge 4$ camera frustums, prune single-view billboards with $N_{\text{eff}}(i) = W_i^2 / S_i < 1.45$ (and $W_i > 0.5$, $R_i / W_i > 0.12$).
3. **Octree Topological Isolation:** A leaf with $\le 3$ splats and no other leaf within $1.25 \times$ (sum of half-sizes) is marked isolated (state 3); only its splats with $\alpha < 0.25$ are removed (`isFloater`).
4. **Selective Sightline Opacity Decay:** Decay opacity for high-error occluders ($R_i / W_i > 0.22$).

### 5.3 Overlap Reconciliation & Octree Rebalancing
1. **Overlap Reconciliation (Moment-Matching Merge):** Merge redundant overlapping Gaussians with small relative distance and color difference.
2. **Octree Rebuild:** Re-bin splats into contiguous leaves ($N_{\text{leaf}} \le 256$).
3. **Cost (TODO):** the whole outer loop (readback, `reconcileAndGrow`, Adam gather, upload) runs on the main thread inside a step, so the page freezes for its duration. jardin5 at 300k splats: ~1 s, then ~210 ms once the near-camera floater test used a grid of the camera cones (it projected every splat into every training camera) and kept splats went into one buffer. Next: `reconcileAndGrow` and the gather in a worker on transferred buffers, training going on with the old splats until the result is swapped in.

### 5.4 Extra Optimizer Methods: Monte-Carlo (SGLD / MCMC) & Simulated Annealing (SA)
In addition to standard **Adam** (`opt_method = 0`), Gauss-Opt provides two stochastic optimization methods selectable live in the GUI:

1. **Monte-Carlo / SGLD (`opt_method = 1`, `monte_carlo`):**
   * **Inner Loop (GPU Covariance-Scaled Langevin Noise):** Injects anisotropic Gaussian noise aligned with each splat's 3D covariance $\Sigma_i = \mathbf{R}_i \mathbf{S}_i^2 \mathbf{R}_i^T$ and gated by transparency $(1 - \alpha_i)^2$ (inspired by *3DGS-MCMC*):
     $$\boldsymbol{\mu}_i \leftarrow \boldsymbol{\mu}_i - \eta_{\text{pos}} \hat{\mathbf{m}}_{\mu,i} + \eta_{\text{mc}} (1 - \alpha_i)^2 (\mathbf{R}_i \mathbf{s}_i \odot \boldsymbol{\xi}_i), \quad \boldsymbol{\xi}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_3)$$
     Low-opacity splats diffuse along their ellipsoid axes to explore uncovered surfaces, while high-opacity converged splats ($\alpha_i \to 1$) experience near-zero noise.
   * **Outer Loop (MCMC Error-Proportional Teleportation):** During `reconcileAndGrow`, pruned/dead splats are **teleported (resampled)** onto high-error parent splats chosen via roulette-wheel sampling proportional to WLBG error mass $R_j$, preserving total splat count while concentrating density where residual error is highest.

2. **Simulated Annealing (`opt_method = 2`, `sim_anneal`):**
   * **Temperature Schedule:** Maintains an annealing temperature $T(t) = T_0 \cdot \gamma^{t/50}$ with optional periodic re-heating at rebalance intervals.
   * **Inner Loop (Metropolis-Hastings Per-Splat Perturbation):** Each splat tracks its local WLBG mean error energy $E_i^{(t)} = e_{\text{view}}(i)$ in `adam_v[b+1].w` and its last accepted position displacement in `adam_m[b+1].w`, `adam_m[b+3].w`, `adam_v[b+3].w`. At each step:
     * Compare current local error $E_i^{(t)}$ against previous $E_i^{(t-1)}$.
     * Compute Boltzmann acceptance probability $P_{\text{acc}} = \exp(-\max(0, E_i^{(t)} - E_i^{(t-1)}) / T(t))$.
     * If rejected ($u > P_{\text{acc}}$), fully undo the last jump and draw a new Gaussian proposal $\Delta \boldsymbol{\mu}_i = 0.25\, T(t)\, \mathbf{R}_i \mathbf{s}_i \boldsymbol{\xi}_i$.
     * If accepted, update the energy baseline ($E \leftarrow 0.7 E + 0.3 E_i^{(t)}$, the first value as is) and jump $0.15\, T(t) \max(0.05, E_i^{(t)})\, \mathbf{R}_i \mathbf{s}_i \boldsymbol{\xi}_i$.
     * Positions only: no Cauchy proposal, no scale perturbation.
   * **Outer Loop (Boltzmann Split Acceptance):** Candidate splat splits with intermediate error $R_i \in [T_l, T_u]$ are accepted stochastically with probability $\exp(-(T_u - R_i) / (T(t) \bar{R}))$.

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
│                          #   panel (sfm.e2e.js), project folders (project.e2e.js); harness.js: server, clip, browser
├── tests/                 # deno test: SfM unit / solver / GPU / synthetic tests, .ply octree round trip
├── tools/                 # Deno / Python command-line tools for the SfM (DESIGN-sfm §10)
└── src/
    ├── app.js             # App: GUI & URL params, dataset/snapshot loading, frame loop, cameras (orbit/drone/fly-to)
    ├── colmap.js          # Binary parser for COLMAP *.bin + .ply/.splat import/export (with embedded octree)
    ├── math.js            # Shared math: vec3, quaternions, 3x3 eigen, OpenCV view matrices, pinhole projection, orbit frame, intrinsics fit
    ├── octree.js          # Loose Octree build, JSON/PLY serialization, front-to-back sort, WLBG/MCMC/SA growth
    ├── render.js          # Pure draw layer: free-view splats, 3D octree & camera wireframes, side-canvas telemetry
    ├── drone.js           # 'Drone fly' free camera: angle-mode quadcopter model (mouse stick, arrows forward/back + yaw, E/Q climb/descend)
    ├── demo.js            # 'Demo' free camera (G): slow elastic flight along the capture path; any input hands over to the drone
    ├── trainer.js         # GPU/CPU hybrid training loop (Adam, Monte-Carlo SGLD, Simulated Annealing)
    ├── metrics.js         # Validation metrics: PSNR, SSIM (3DGS convention), PSNR at 1/4 resolution
    ├── shaders.wgsl       # WGSL kernels: projection, tile forward/backward+WLBG moments, Adam/MC/SA update, render
    ├── snapshot.js        # .gosnap full training-state snapshot codec (splats + Adam + moments + octree)
    ├── stipple.js         # E4: residual stippling -> unprojected seed splats (reuses ../stipple/ Voronoi)
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

