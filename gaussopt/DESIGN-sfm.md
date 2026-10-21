# Gauss-Opt SfM: In-Browser Video → Registered Cameras — Design Document

## 1. Overview & Motivation

Gauss-Opt trains from a COLMAP `gs/` directory produced offline by [`video2splat.py`](../../../brush/video2splat.py): ffmpeg frame extraction, sharpness/spacing selection, COLMAP SIFT + sequential matching, `global_mapper`, `image_undistorter`. This document specifies a second, **in-browser** path that turns a **video file** into the same dataset, with WebGPU for the heavy kernels and an interactive UI to inspect and repair weak sections. The COLMAP path stays exactly as it is.

### 1.1 Where the time goes today (measured, M5 Pro, COLMAP 88065471, no CUDA)

| Stage | jardin5 (425 frames) | jardin6 (525 frames) | Runs on |
| :--- | :---: | :---: | :--- |
| SIFT extraction | 13 s | 16 s | GPU (SiftGPU, OpenGL) |
| Sequential + loop matching | 4.2 min | 5.7 min | GPU (SiftGPU, OpenGL) + CPU verification |
| `global_mapper` | **8.2 min** | **10.2 min** | CPU (Ceres) |
| `image_undistorter` | 3 s | 3 s | CPU |

* `global_mapper` breakdown (jardin5): global positioning 65–74 s, iterative BA 290 s (3 rounds of fixed-rotation + joint stage), retriangulation 125 s. The problem has 425 cameras, 308k tracks, 2.4M observations.
* Profile (`sample`): BA time is in Ceres `SchurEliminator<2,3,-1>::ChunkOuterProduct` and `BlockRandomAccessSparseMatrix::GetCell`, with most of the 18 threads in `__psynch_mutexwait`. All frames share one camera, so every point's outer product hits the same intrinsics cells: the solver is lock-bound at 5–14 cores.
* Matching cost is $O(n^2)$ in features per image; frames carry 10k–18k SIFT features and SiftGPU's OpenGL matcher clamps at 16384.
* Nothing in the pipeline is CUDA-only on this machine. The motivation is **speed, portability and interactivity**, not getting rid of CUDA.

### 1.2 Locality of a video (measured on the COLMAP models)

| | jardin5 | jardin6 |
| :--- | :---: | :---: |
| Track span (keyframes), median / p90 / p99 | 7 / 32 / 115 | 5 / 16 / 195 |
| Tracks contained in 10 consecutive keyframes | 66% | 78% |
| Tracks contained in 20 consecutive keyframes | 83% | 93% |
| Keyframes sharing ≥ 30 points with a keyframe > 30 apart | 219 / 425 | 287 / 525 |

* A sliding window of $W = 10\text{--}20$ keyframes holds most of the geometry, so mapping can be **local**: cost $O(N \cdot W)$ instead of a global problem.
* About half of the keyframes revisit earlier parts of the scene (drone orbits). **Loop closure is a core stage, not an option**, and windows must be built on **covisibility**, not just time.

### Scope & Assumptions
* **Input:** one video file (`.mp4` / `.mov` / `.webm`), single camera, constant intrinsics (no zoom). Drone footage is the primary target.
* **Output:** the exact in-memory dataset `buildDataset()` returns in [`src/colmap.js`](src/colmap.js) (`cameras`, `images`, `points3D`), so the trainer, renderer and snapshot code are unchanged. Optionally written out as COLMAP `sparse/*.bin` + undistorted images.
* **External COLMAP input stays first-class:** `?dataset=`, folder drop and `serve.py` keep working as today. The in-browser SfM is an alternative producer, and COLMAP is also its reference for validation (§8).
* **Target:** desktop Chrome / Dawn, WebGPU available in a Dedicated Worker.

---

## 2. Core Design Decisions

### D1. Video-Specific Incremental SfM, Not a Port of `global_mapper`
* Temporal order gives the neighbours for free: no exhaustive or vocabulary-tree pairing for the sequential part, and a new frame is located against the previous window (PnP), which is far more stable than global positioning from scratch.
* Structure follows visual SLAM (ORB-SLAM: tracking, local BA on covisible keyframes, loop closure with Sim(3) pose graph), but **offline**: the whole video is known, so loop closure and refinement run as separate passes over the finished sequential map, with no real-time budget.

### D2. One Solver Core, Two Backends
* Window BA (≈ 100 camera parameters, 20–30k points, ~10 LM iterations) runs in **plain JS, f64** in the worker. It is small enough for the CPU and serves as the reference implementation.
* Large problems (global refinement, if needed) use the same Schur-complement LM in **WGSL, f32**: per-observation residual/Jacobian, per-point $3\times3$ elimination, implicit-Schur PCG for the camera system. The small reduced systems and all accumulations that need precision stay in f64 on the CPU.
* WGSL has no f64. Mitigation: coordinates are centred and scaled per window, poses are parameterized as local increments, and the JS backend is the oracle for the WGSL one. Precedent: Wu et al., *Multicore Bundle Adjustment* (CVPR 2011) runs single precision on the GPU; COLMAP's Caspar BA defaults to f32 (`CASPAR_USE_DOUBLE OFF`).

### D3. Covisibility Windows Everywhere
* A **window** is a seed keyframe set (temporal) **plus** the keyframes that share ≥ $m$ points with it (from loops). Local BA, refinement sweeps and user-triggered repairs (§6) all use the same window builder.
* Keyframes outside the window that observe window points are held fixed: they anchor position, orientation and scale (the monocular gauge).

### D4. Low Frequency by Pose Graph, High Frequency by Window Sweeps
* Block-coordinate descent over gliding windows converges slowly on low-frequency errors (scale drift, loop tension). These are removed first by a **Sim(3) pose graph** over keyframes (sequential + loop edges; ~500 nodes × 7 parameters, milliseconds in JS).
* Gliding-window BA sweeps then fix the remaining high-frequency error. A single global GPU BA is a fallback, used only if sweeps fall short of COLMAP accuracy (§8).

### D5. SIFT Features, 4096 per Keyframe
* SIFT keeps the descriptors loop closure needs and allows stage-by-stage comparison with COLMAP's database. uint8 descriptors: 4096 × 128 B = 512 KB per keyframe, ~270 MB for 525 keyframes, kept on the CPU and uploaded per window.
* KLT tracking between keyframes is a possible later addition for the sequential part.

### D6. Separate, Lazily Loaded Panel
* The main UI is already crowded. Everything SfM lives in its own panel (§7), loaded with a dynamic `import()` the first time it opens: zero cost for users who load COLMAP datasets.

---

## 3. Data Model

All state lives in the worker; the panel receives snapshots for display.

* **`Video`:** source `File`/URL, fps, duration, resolution, decoded on demand by timestamp.
* **`Keyframe`:** video timestamp, compressed image blob (JPEG/WebP, ~300 KB at 1080p; raw RGBA for 500 frames would be 4 GB), sharpness, keypoints (f32 $x, y$, scale, orientation), descriptors (uint8), pose $T_{cw}$ (or `null` = unregistered), segment id, per-keyframe quality (§6.1).
* **`Camera`:** model (`SIMPLE_RADIAL` during SfM: $f, c_x, c_y, k_1$; `PINHOLE` after undistortion), refine flags.
* **`Track` / `Point`:** position, colour, observations (keyframe, keypoint index), reprojection error, reference keyframe (for Sim(3) corrections).
* **`Edge`:** keyframe pair with match/inlier count and type: `seq`, `loop`, `user-loop`, `rejected`.
* **`Segment`:** maximal run of keyframes in one connected map; a tracking loss starts a new segment, merged later by loop closure (§5.4).
* **Operation log:** every stage run and user action (§6) is an entry with its parameters, so a model can be replayed, undone, and saved.

---

## 4. Output Contract & Persistence

* **To the trainer:** `{ cameras, images, points3D }` with the same fields `parseImagesBin` produces (`R`, `t`, `center`, `forward`, `view`, `proj`, `obs2D`, `zMin`/`zMax`), plus `fileBlob` per image (the path the dropped-folder loader already uses for thumbnails). Then `computeCameraDepthStats` and the validation split run unchanged.
* **Undistortion:** a GPU texture remap from `SIMPLE_RADIAL` to `PINHOLE` (same maximal-rectangle cropping as `image_undistorter`), applied once when handing over.
* **Export:** `sparse/cameras.bin`, `images.bin`, `points3D.bin` and `images/`, so a browser result can be opened in the COLMAP GUI or compared with `colmap model_comparer` / `model_aligner`. The writer mirrors the parsers in `colmap.js`.
* **Session persistence:** keyframes, features and model go to OPFS (Origin Private File System), so a reload resumes without decoding and extracting again. A `.gosfm` project file (operation log + model, optionally features) can be saved and dropped back in.

---

## 5. Pipeline

Each stage is restartable from its inputs, which is what makes interactive repair (§6) possible.

### 5.1 S0 — Decode & Frame Selection
* Decode with WebCodecs `VideoDecoder` (demux via mp4box.js). A first version can use `<video>` + `requestVideoFrameCallback` with no dependency.
* Port of `video2splat.py --spacing`: sharpness (Laplacian variance, reject < 0.35 × median) and keep one frame every `spacing` × image diagonal of optical-flow path, computed on the GPU at low resolution.
* **Change vs the Python version:** the path length uses the flow **after removing a fitted homography**, i.e. parallax rather than raw motion. Pure yaw in place produces flow but no parallax, and today it adds keyframes that cannot be triangulated.
* Keyframes can be **inserted later** at any timestamp (§6.3): the video stays available, which COLMAP never had.

### 5.2 S1 — Features
* SIFT in WGSL following the SiftGPU structure: Gaussian pyramid, DoG extrema with subpixel refinement, edge rejection, orientation histogram, 4×4×8 descriptor, RootSIFT normalisation, top-$K$ by response ($K$ = 4096 default).
* Validated against COLMAP's SIFT on the same frames: repeatability and match counts on sequential pairs, not bitwise equality.

### 5.3 S2 — Sequential Mapping
* **Matching:** keyframe $k$ against $k-1 \dots k-W$ on the GPU: brute-force nearest neighbour on f16 (a batched matrix product), ratio test 0.8, mutual check. No 16384 cap. ~$N \cdot W$ pairs total.
* **Initialisation:** first pair with median parallax above a threshold; fit essential matrix (5-point, port from PoseLib, BSD-3) and homography. If the homography explains the matches (rotation only, or a flat scene), move to a later pair. Triangulate.
* **Per new keyframe:**
  1. Locate against the window's 3D points (P3P + LO-RANSAC, then pose-only refinement).
  2. Triangulate new tracks with window keyframes: viewing angle ≥ 1.5°, positive depth, reprojection error < 4 px.
  3. Window BA (D2/D3): last $W$ poses + their points; older observers fixed.
* **Intrinsics:** no prior. Start at $f = 1.2 \cdot \max(w, h)$ (COLMAP's default) or a `focal=` URL parameter, estimate $f$ and $k_1$ jointly over the first ~50 keyframes, then freeze; free again in the final refinement.
* **Tracking loss:** too few PnP inliers. Try the BoW index (§5.4) to relocalise against all keyframes; otherwise start a new segment from a new initialisation.

### 5.4 S3 — Loop Closure (Offline Pass)
* **Candidates:**
  * Appearance: bag of words built from this video's own descriptors (k-means to ~4k words on a sample, TF-IDF, cosine similarity on the GPU; 500² is trivial). No pre-trained vocabulary to ship.
  * Geometry: keyframes far apart in time whose estimated frusta overlap, with a tolerance that grows with the temporal distance (drift).
  * User: pairs designated in the UI (§6.3).
* **Verification:** full descriptor match → PnP RANSAC of the query against the candidate's points → Sim(3) from matched 3D points (Umeyama + RANSAC). The scale component corrects monocular drift.
* **Correction:** Sim(3) pose graph over all keyframes, points moved with their reference keyframe, then **track fusion**: points triangulated independently on both sides of the loop that match the same keypoints are merged. Without fusion, BA cannot pull the two sides together.
* Segments (§3) are merged by the same mechanism.

### 5.5 S4 — Global Refinement
* Gliding-window BA sweeps over covisibility windows (D3/D4): window offsets alternate by $W/2$ so boundaries do not stay fixed; 2–3 sweeps or until the largest pose change is below a threshold.
* Non-overlapping windows are independent: on the GPU, solve all even windows in one dispatch, then all odd ones.
* Intrinsics are freed in this stage (one shared $f, k_1$ accumulated across windows).
* Fallback: one global GPU BA (D2) if sweeps fall short of COLMAP (§8).

### 5.6 S5 — Hand-Over
* Undistort (§4), build the dataset, close the panel or keep it open, start training. Every 8th view stays held out for validation, as today.

### 5.7 Expected Budget (jardin5, estimates, not measured)

| Stage | Estimate |
| :--- | :---: |
| S0 decode + selection | 5–10 s |
| S1 SIFT | 10–20 s |
| S2 matching | ~10 s |
| S2 mapping + window BA | 30–90 s |
| S3 loop closure | a few s |
| S4 sweeps | ~30 s |
| **Total** | **≈ 2 min** (COLMAP today: 12.5 min) |

---

## 6. Interactive Inspection & Repair

The browser can do what a batch COLMAP run cannot: show where the model is weak and let the user point at it. Repairs are **local**: they re-run the stages of §5 on a window and re-merge the result, never the whole video.

### 6.1 Quality Signals (per keyframe, per edge)
* Registration: unregistered, or PnP inlier count / ratio below threshold.
* Support: number of tracked points, mean track length, reprojection error.
* Geometry: parallax to the previous keyframe; homography-vs-essential score (rotation-only stretches).
* **Path smoothness:** drone motion is smooth, so second-difference outliers of the camera centres (jerk) and of rotations flag bad poses that every other metric misses.
* Loop edges: Sim(3) residual after the pose graph (a large residual suggests a false loop or a broken section).
* **After training:** per-view validation PSNR from the trainer. A view that trains badly next to views that train well is a pose problem candidate. This closes the loop from Gauss-Opt back to SfM.
* Signals are combined into one per-keyframe score (green / amber / red) with the dominant reason.

### 6.2 Display
* **Timeline strip** (one cell per keyframe, coloured by score; thumbnails on hover; segments shown as breaks; loop edges as arcs).
* **3D view:** the camera path (existing `show_path` line) coloured by score; frustums (existing picking) of red keyframes highlighted; points coloured by reprojection error on request.
* Selection is shared: brushing a timeline range highlights the path and frusta, clicking a frustum selects its timeline cell.

### 6.3 Actions on a Selection (keyframe range, or two keyframes)

| Action | What it does | Typical fix for |
| :--- | :--- | :--- |
| **Retry** | Re-match the range with more features, larger $W$, looser ratio test, re-locate and window BA | Low inliers, short tracks |
| **Densify** | Insert extra keyframes from the video inside the range (smaller `spacing`), then retry | Fast motion, blur gaps, tracking loss |
| **Drop** | Remove keyframes (blurry, moving objects, lens flare) and re-run window BA around them | Outlier frames |
| **Refine** | Window BA with a wider covisibility window and intrinsics free | Residual jitter |
| **Redo** | Cut the range into a new segment, re-initialise and re-map it, merge back with Sim(3) (§5.4) | A broken stretch |
| **Force loop** | User picks two keyframes that see the same place: verify and add a `user-loop` edge, re-run pose graph + fusion | Missed loops |
| **Reject loop** | Mark a loop edge `rejected`, re-run pose graph | False loops (repetitive texture) |
| **Set focal** | Fix or re-estimate intrinsics, re-run refinement | Wrong focal (curved paths, bad scale) |

* Every action is an operation-log entry (§3): **undo** restores the previous model, and the before/after scores of the affected keyframes are shown.
* Actions run in the worker; the panel stays responsive and shows progress per stage.
* While training is running, repairs apply to the SfM model only. Pushing an updated model to the trainer swaps the cameras (poses change, splats stay). This needs a new trainer entry point that replaces camera poses in place, keyed by image name; splats remain in the old frame, so pushes are only sensible after a local repair, not after a change of gauge.

---

## 7. UI: the "Video → Cameras" Panel

### 7.1 Opening
* **Drop a video file** anywhere on the page: the existing `drop` handler in `src/app.js` routes `video/*` MIME types or `.mp4/.mov/.webm` to the panel instead of `loadDroppedFolder`.
* **"Process video…" button** in the main button row (one button, the only addition to the main UI), with a file picker.
* **URL:** `?video=URL` opens the panel with that video; SfM parameters are settable by name like every other parameter (`sfm_spacing=0.05`, `sfm_features=4096`, …).

### 7.2 Layout
* A **full-height panel that replaces the right column** (`#gui-container`) while open, with its own lil-gui instance, plus the **timeline strip** docked under the canvas. The main canvas shows the SfM state live (points, path, frusta) through the existing renderer.
* **Close** restores the main GUI; the SfM session stays in the worker and in OPFS, so reopening is instant.

### 7.3 Contents (top to bottom)
1. **Source:** file name, resolution, fps, duration; thumbnail scrubber.
2. **Frames (S0):** sample fps, `spacing`, blur threshold, max resolution; live count of kept keyframes.
3. **Features & matching (S1–S2):** max features, window $W$, ratio test.
4. **Camera:** model (`SIMPLE_RADIAL` / `PINHOLE`), initial focal, refine flags.
5. **Mapping (S2–S4):** min parallax, loop detection on/off, BoW size, number of sweeps.
6. **Run controls:** Run all, Run stage, Pause, Stop; per-stage progress bars and timings.
7. **Quality (§6):** score legend, filters (show red only), selection info, action buttons of §6.3, undo.
8. **Output:** downscale for training, **Send to trainer**, **Export COLMAP**, **Save project (.gosfm)**.
9. **Log:** SfM-specific log (the main `#log-panel` receives one summary line per stage).

---

## 8. Validation Against COLMAP & Milestones

Each milestone is compared with the COLMAP model of the same frames on jardin4/5/6 after a Sim(3) alignment: RMS camera position error relative to path length, rotation error (degrees), focal ratio, reprojection error, % registered, and finally validation PSNR from the trainer.

* **Reference input:** a small Python script dumps keypoints, descriptors and verified matches from `database_sift.db` to a compact binary, so each stage can run on COLMAP's features and be tested in isolation. Solver modules have no DOM dependency and also run under Node for regression tests.

| Milestone | Content | Estimate |
| :--- | :--- | :---: |
| M1 | S2 mapping + window BA (JS) on COLMAP's features/matches | 2–3 weeks |
| M2 | S3 loop closure, pose graph, track fusion; S4 sweeps | 1–2 weeks |
| M3 | WGSL SIFT + matching replacing COLMAP's features | 2–3 weeks |
| M4 | S0 video front end, panel (§7), undistortion, hand-over | 1–2 weeks |
| M5 | Interactive repair (§6) | 1–2 weeks |
| M6 | Global GPU BA, only if M2 sweeps fall short | 2–4 weeks |

M1 carries most of the risk (geometry, BA, intrinsics without prior) and needs no GPU work, so it goes first.

---

## 9. Adversarial Review & Mitigations

* **"f32 BA will not converge on a 90-unit path."** Window problems are solved in f64 JS; GPU problems use per-window centring/scaling and incremental parameters; the JS backend checks the WGSL one on the same inputs.
* **"No focal prior: the whole map is wrong."** Joint estimation on the first ~50 keyframes, re-estimated in S4, user override (§6.3); M1 measures focal error against COLMAP.
* **"Scale drift without loops."** Remains, as for any monocular method; the quality view shows it as loop residuals or path jerk. Not worse than COLMAP for a single pass without revisits.
* **"Repetitive texture (grass, roofs) causes false loops."** Loops need PnP + Sim(3) RANSAC agreement and a pose-graph residual check; the UI shows and lets the user reject loop edges.
* **"Rotation-only segments break triangulation."** Homography test at initialisation and in S0 spacing (parallax, not raw flow); such keyframes are located by PnP only and do not triangulate.
* **"Browser memory."** Keyframes stored compressed; descriptors on the CPU, uploaded per window; full-resolution decode only for SIFT, then dropped.
* **"Locked-in to one browser."** WebGPU in a worker is required; feature-detect and keep the COLMAP path as the fallback.

---

## 10. File Structure (new files)

```text
gaussopt/
├── DESIGN-sfm.md          # This document
└── src/sfm/
    ├── panel.js           # §7 panel: lil-gui, timeline strip, selection, actions; dynamic import from app.js
    ├── worker.js          # Worker entry: owns the model, runs stages and actions, operation log, OPFS
    ├── video.js           # S0 decode (WebCodecs / <video>), timestamp access, keyframe blobs
    ├── select.js          # S0 sharpness + parallax spacing (GPU, low resolution)
    ├── sift.js / sift.wgsl    # S1 SIFT
    ├── match.js / match.wgsl  # S2/S3 brute-force matching, ratio + mutual test
    ├── geometry.js        # 5-point (PoseLib port), homography, P3P, triangulation, LO-RANSAC, Umeyama
    ├── ba.js / ba.wgsl    # Schur-complement LM: JS f64 core, WGSL f32 backend
    ├── posegraph.js       # Sim(3) pose graph
    ├── loop.js            # BoW vocabulary, candidates, verification, track fusion
    ├── quality.js         # §6.1 signals and scores
    └── export.js          # COLMAP *.bin writer, undistortion, dataset hand-over, .gosfm
```

---

## 11. Open Questions

* Is a 1080p keyframe needed for SIFT, or does half resolution register as well (4× less SIFT and memory)?
* KLT tracking for S2 instead of descriptor matching: fewer outliers on small baselines, but a second code path.
* Does window-sweep refinement match COLMAP's global BA on jardin5/6, or is M6 required?
* Can the trainer's photometric error refine poses directly (pose gradients in the backward pass), making S4 partly redundant?
