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
  1. **Eliminates Global GPU Radix Sort:** Any octree can be traversed in exact front-to-back order for a given camera position $\mathbf{c}$ by ordering a node's 8 children using a 3-bit XOR mask on the octant index (`octant ^ cam_octant_mask`).
  2. **Workgroup-Local Sorting:** Because each leaf holds at most $N_{\max} = 256$ splats, sorting splats by depth within a leaf happens locally on small contiguous leaf slices.
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

### D4. Parameterization (Phase 1: SH Degree 0, Brush-Compatible)
* Start with **SH degree 0** (**14 `f32` per splat**, padded to 16 `f32` = 64 bytes for clean `vec4f` alignment in WGSL):
  * `pos`: `vec3f` world space $(x, y, z)$, plus `opacity`: `f32` in logit space $\ln(\alpha / (1 - \alpha))$
  * `scale`: `vec3f` in log-space $(\ln s_x, \ln s_y, \ln s_z)$, plus `leaf_id`: `f32`
  * `quat`: `vec4f` $(q_w, q_x, q_y, q_z)$ normalized
  * `f_dc`: `vec3f` SH degree 0 coefficients where $c_{\text{rgb}} = \text{clamp}(0.5 + C_0 \cdot \mathbf{f}_{\text{dc}}, 0, 1)$ with $C_0 = 0.28209479177$, plus `_pad1`: `f32`
* **Why `f_dc` storage:** Matches Brush and standard 3DGS `.ply` files directly for lossless import/resume/export.

---

## 3. Handover from COLMAP, Resuming from `.ply`, & Octree Serialization

Because `sparse/images.bin` already embeds the exact filenames of all registered images, **no offline file conversion is needed** between `colmap image_undistorter` (which produces `gs/` in [`video2splat.py`](../../../brush/video2splat.py)) and `gaussopt/`.

### 3.1 Path A: CLI Bridge (`gaussopt/serve.py`)
A lightweight Python script (`gaussopt/serve.py`) acts as the local bridge:
```bash
./gaussopt/serve.py /Users/skal/gskal/skal65535/Hierarchical-Localization/datasets/jardin1_gs [--ply splats/jardin1_05000.ply]
```
1. **HTTP Mounting:** Serves `skal65535.github.io/` on `http://localhost:8000/`, mounts the target `gs/` directory at `/dataset/`, and exposes `GET /dataset/info.json` (listing any existing `.ply` checkpoints in `splats*/` or `init.ply`).
2. **Direct Binary Ingestion (`gaussopt-colmap.js`):**
   * Fetches `/dataset/sparse/cameras.bin` $\to$ intrinsics $(W, H, f_x, f_y, c_x, c_y)$.
   * Fetches `/dataset/sparse/images.bin` $\to$ world-to-camera $(q, t)$, observed `point3D_id`s (to compute each camera's near-field depth guard $z_{\min}(c)$), and image filenames.
   * Fetches `/dataset/sparse/points3D.bin` $\to$ initial 3D positions $(x, y, z)$ and RGB colors.
   * Streams `/dataset/images/<filename>` via `createImageBitmap`.
3. **Checkpoint & Octree Export (`POST /save`):**
   * Exposes `POST /save?name=splat_05000.ply` so the browser can write `.ply` / `.splat` / `.octree.json` files directly into `<dataset>/splats_gaussopt/` on disk.

### 3.2 Path B: Zero-Server Static Browser Mode (Drag-and-Drop / Folder Picker)
When running statically on `https://skal65535.github.io/gaussopt/`:
* Dragging and dropping a `gs/` directory (or picking it via `<input type="file" webkitdirectory>`) maps the dropped `File` handles by relative path (`sparse/*.bin`, `images/*`, `*.ply`) into `gaussopt-colmap.js`.

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
     comment gaussopt_octree_v1 beta=0.20 max_leaf=256 cx=... cy=... cz=... radius=...
     element octree_node <M>
     property float cx
     property float cy
     property float cz
     property float half_size
     property int is_leaf
     property int start_or_child0
     property int count_or_depth
     property int state
     ```
   * Standard 3DGS viewers read `element vertex <N>` and ignore `element octree_node <M>`, while `parsePlyBuffer` in `gaussopt/` and `splats/` restores the entire `LooseOctree` in **$< 0.2\text{ ms}$** without rebuilding or re-compacting.
2. **Standalone `.octree.json` Export / Import:**
   * `octree.toJSON()` / `LooseOctree.fromJSON(json)` exports/imports the serialized node flat-array (`nodes`, `beta`, `maxLeafSize`, `sceneCenter`, `sceneRadius`) as a compact JSON sidecar (`save .octree` button in UI).
3. **Visualizer Speed-Up (in both `gaussopt/` and `splats/index.html`):**
   * Instead of running a global $O(N)$ radix sort over all $N$ splats every camera move, the visualizer uses **Octree Hierarchical Frustum Culling + `rank ^ mask` Front-to-Back Traversal**:
     * Subtrees completely outside the view frustum are culled in $O(1)$ at high-level internal nodes (skipping $50\%\text{--}80\%$ of splats in a single test when zoomed in).
     * Visible leaves are visited in exact front-to-back order (`rank ^ mask`) and sorted locally inside each leaf (`sort_name = 'Octree'` in [`splats/index.html`](../splats/index.html)).

---

## 4. Ray Integration & Sub-Task Training Modes

$$\hat{C}(\mathbf{p}) = C_{\text{fg}}(\mathbf{p}) + T_{\text{fg}}(\mathbf{p}) \cdot \Big( C_{\text{active}}(\mathbf{p}) + T_{\text{active}}(\mathbf{p}) \cdot C_{\text{bg}}(\mathbf{p}) \Big)$$

* **Mode A: Full-Ray Octree Traversal (Global View Step):** All visible octree leaves for the selected training camera are queued in front-to-back order.
* **Mode B: Block-Coordinate Sub-Task Optimization (Local Subtree Step):** Pre-renders frozen foreground $(C_{\text{fg}}, T_{\text{fg}})$ and background $(C_{\text{bg}})$ for a local octree subtree $B$.

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
2. **Multi-View Participation Ratio ($N_{\text{eff}}$ Check):** In cells intersected by $M_{\text{cell}} \ge 4$ camera frustums, prune single-view billboards with $N_{\text{eff}}(i) = W_i^2 / S_i < 1.45$.
3. **Octree Topological Isolation:** Prune sparse leaves ($\le 3$ splats) that have zero populated neighbor leaves.
4. **Selective Sightline Opacity Decay:** Decay opacity for high-error occluders ($R_i / W_i > 0.22$).

### 5.3 Overlap Reconciliation & Octree Rebalancing
1. **Overlap Reconciliation (Moment-Matching Merge):** Merge redundant overlapping Gaussians with small relative distance and color difference.
2. **Octree Rebuild:** Re-bin splats into contiguous leaves ($N_{\text{leaf}} \le 256$).

### 5.4 Extra Optimizer Methods: Monte-Carlo (SGLD / MCMC) & Simulated Annealing (SA)
In addition to standard **Adam** (`opt_method = 0`), Gauss-Opt provides two stochastic optimization methods selectable live in the GUI:

1. **Monte-Carlo / SGLD (`opt_method = 1`, `monte_carlo`):**
   * **Inner Loop (GPU Covariance-Scaled Langevin Noise):** Injects anisotropic Gaussian noise aligned with each splat's 3D covariance $\Sigma_i = \mathbf{R}_i \mathbf{S}_i^2 \mathbf{R}_i^T$ and gated by transparency $(1 - \alpha_i)^2$ (inspired by *3DGS-MCMC*):
     $$\boldsymbol{\mu}_i \leftarrow \boldsymbol{\mu}_i - \eta_{\text{pos}} \hat{\mathbf{m}}_{\mu,i} + \eta_{\text{mc}} (1 - \alpha_i)^2 (\mathbf{R}_i \mathbf{s}_i \odot \boldsymbol{\xi}_i), \quad \boldsymbol{\xi}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_3)$$
     Low-opacity splats diffuse along their ellipsoid axes to explore uncovered surfaces, while high-opacity converged splats ($\alpha_i \to 1$) experience near-zero noise.
   * **Outer Loop (MCMC Error-Proportional Teleportation):** During `reconcileAndGrow`, pruned/dead splats are **teleported (resampled)** onto high-error parent splats chosen via roulette-wheel sampling proportional to WLBG error mass $R_j$, preserving total splat count while concentrating density where residual error is highest.

2. **Simulated Annealing (`opt_method = 2`, `sim_anneal`):**
   * **Temperature Schedule:** Maintains an annealing temperature $T(t) = T_0 \cdot \gamma^{t/100}$ with optional periodic re-heating at rebalance intervals.
   * **Inner Loop (Metropolis-Hastings Per-Splat Perturbation):** Each splat tracks its local WLBG mean error energy $E_i^{(t)} = e_{\text{view}}(i)$ in `adam_v[b+1].w` and its last accepted position displacement in `adam_m[b+2]`. At each step:
     * Compare current local error $E_i^{(t)}$ against previous $E_i^{(t-1)}$.
     * Compute Boltzmann acceptance probability $P_{\text{acc}} = \exp(-\max(0, E_i^{(t)} - E_i^{(t-1)}) / T(t))$.
     * If rejected ($u > P_{\text{acc}}$), partially revert the last position/scale perturbation and sample a new temperature-scaled Cauchy/Gaussian proposal $\Delta \boldsymbol{\mu}_i \propto T(t) \mathbf{R}_i \mathbf{s}_i \boldsymbol{\xi}_i$.
   * **Outer Loop (Boltzmann Split Acceptance):** Candidate splat splits with intermediate error $R_i \in [T_l, T_u]$ are accepted stochastically with probability $\exp(-(T_u - R_i) / (T(t) \bar{R}))$.

---

## 6. Live Visualization & Interactive Debugging (`gaussopt-render.js`)

1. **Camera Modes (`#main-canvas`):** Free-View Orbit Camera (`theta, phi, radius, fov`), Snap-to-COLMAP View (`cam_idx`), and `hold target` / split-screen wipe comparing $I_{\text{gt}}$, $\hat{I}$, and $|I_{\text{gt}} - \hat{I}|$.
2. **Splat Shape & Diagnostic Color Modes:**
   * Shapes: `gaussian`, `cone`, `ellipsis`, `small_dot`, `axis`.
   * Colors: `learned_rgb`, `octree_leaf`, `wlbg_error` ($R_i$), `view_div` ($N_{\text{eff}}$), `depth`.
3. **3D Overlays:**
   * `3D octree boxes`: Nominal ($L$) or loose ($(1+2\beta)L$) leaf wireframes colored by state.
   * `COLMAP frustums`: Camera pyramids colored by train/val/active status.
   * `WLBG split axes`: 3D principal error axes on newly split splats.
4. **Side-Canvas Inspector & Telemetry (`#side-canvas`):** Rolling PSNR curve, optimizer temperature $T(t)$, and Octree/WLBG rebalance stats.

---

## 7. Adversarial Design & Code Review Mitigations

| ID | Risk / Failure Mode | Concrete Mitigation Built into Implementation |
| :--- | :--- | :--- |
| **A1** | **COLMAP vs. `splats/index.html` Camera Convention Mismatch** | Standardized all projection and EWA Jacobian math on **OpenCV/COLMAP camera space ($+Z$ forward)**: $u = f_x (x/z) + c_x$, $v = f_y (y/z) + c_y$, while supporting $z_{\text{sign}} = -1$ in `LooseOctree.sortIndexFrontToBack` for `splats/index.html`. |
| **A2** | **Far-Field SfM Outliers Crushing Octree Resolution** | Root octree bounding box computed from the **2nd–98th percentile** of point coordinates. |
| **A3** | **Backward Transmittance Division-by-Zero ($T_i = T_{i+1} / (1 - \alpha_i)$)** | Clamped $\alpha_i \le 0.99$; forward integration stops when $T < 10^{-4}$, recording `last_k` per pixel. |
| **A4** | **Workgroup Shared-Memory Contention in Backward Pass** | Contributing splats are loaded in batches of 32 into `var<workgroup>` shared memory and reduced across the $16\times 16$ tile before at most 1 global atomic update per splat per tile. |
| **A5** | **2D$\to$3D EWA Chain Rule Overhead** | Pixel threads accumulate only 2D gradients (`dL/drgb`, `dL/dop`, `dL/duv`, `dL/dconic`); the 2D$\to$3D EWA covariance & quaternion chain rule runs once per splat in `adam_step`. |
| **A6** | **Suffix Color Transmittance Scaling in `dL_dalpha`** | Since `accum_rec` tracks the unattenuated suffix color $\tilde{C}_{>i} = T_{i+1}^{-1}\sum_{j>i} T_j \alpha_j c_j$, `dL_dalpha` uses exact $T_i \cdot (\text{dL\_dC} \cdot (c_i - \tilde{C}_{>i}))$ without double-dividing by $(1 - \alpha_i)$. |
| **A7** | **Column-Major Quaternion Rotation Gradient (`dL_dq`)** | Matched skew-symmetric cross-terms `(G1.z - G2.y)`, `(G2.x - G0.z)`, `(G0.y - G1.x)` to WGSL's column-major `mat3x3f` layout. |
| **A8** | **Stale Adam Momentum / SA State Across Octree Reordering** | `_uploadSplats()` zeroes `adamMBuf` and `adamVBuf` and resets `adamStepCount = 0` whenever splats are reordered or split/pruned by the octree. |
| **A9** | **Concurrent `step()` / `mapAsync` Re-Entrancy** | Guarded `GaussOptTrainer.step()` with `stepInFlight` to prevent overlapping `mapAsync()` calls when clicking UI step controls during live training. |
| **A10** | **Rank-Deficient Single-View 2nd Error Moment $\mathbf{E}_i$** | Added isotropic diagonal regularization $\epsilon_{\text{iso}} = 10^{-4} s_{\max}^2 \mathbf{I}_3$ before power iteration in `reconcileAndGrow()`. |
| **A11** | **Progressive Streaming Octree Rebuild Thrashing in `splats/`** | Throttled `LooseOctree.buildIndexOnly()` during HTTP streaming to only rebuild every $\ge 16{,}384$ newly arrived splats and once upon stream completion. |

---

## 8. File Structure in `gaussopt/`

```text
skal65535.github.io/gaussopt/
├── DESIGN.md              # Architecture, decisions, adversarial review, MC/SA & Octree spec
├── serve.py               # Local HTTP bridge: mounts COLMAP gs/ dirs, lists .ply checkpoints, handles POST /save
├── index.html             # UI, main 3D canvas, side-canvas inspector, GUI controls & training loop
├── gaussopt-colmap.js     # Binary parser for COLMAP *.bin + .ply/.splat import/export (with embedded octree)
├── gaussopt-octree.js     # Loose Octree build, JSON/PLY serialization, front-to-back sort, WLBG/MCMC/SA growth
├── gaussopt-render.js     # Pure draw layer: free-view splats, 3D octree & camera wireframes, side-canvas telemetry
├── gaussopt-trainer.js    # GPU/CPU hybrid training loop (Adam, Monte-Carlo SGLD, Simulated Annealing)
└── gaussopt-shaders.wgsl  # WGSL kernels: projection, tile forward/backward+WLBG moments, Adam/MC/SA update, render
```
