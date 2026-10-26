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
* **Input:** one video file (`.mp4` / `.mov` / `.m4v`, non-fragmented), single camera, constant intrinsics (no zoom). Drone footage is the primary target.
* **Photo sets** (~10 photos and more, one phone, mixed lenses, zooms and orientations) are planned in §13.
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
* **`Camera`:** model (`OPENCV` during SfM: $f_x = f_y = f, c_x, c_y, k_1$ refined, $k_2, p_1, p_2$ held at 0; `PINHOLE` after undistortion), refine flags.
* **`Track` / `Point`:** position, colour, observations (keyframe, keypoint index), reprojection error, reference keyframe (for Sim(3) corrections).
* **`Edge`:** keyframe pair with match/inlier count and type: `seq`, `loop`, `user-loop`, `rejected`.
* **`Segment`:** maximal run of keyframes in one connected map; a tracking loss starts a new segment, merged later by loop closure (§5.4).
* **Operation log** (in memory, not persisted): every stage run and user action (§6) is an entry with its parameters and a **model snapshot** (copy-on-write: poses, intrinsics, tracks; features are immutable and shared). Undo restores the snapshot. Replaying the log is not an undo mechanism: RANSAC and GPU float reductions make a replay close but not bit-exact.
* **Keyframe names** are the presentation timestamp, zero-padded (`t0012345.jpg`, milliseconds). Name order is time order, which the camera path and the trainer rely on, and frames inserted later (§6.3) keep that order.
* **Validation hold-out** is chosen by a hash of the keyframe name, not by index: the trainer's `i % 8 == 4` split shifts every view after an inserted or dropped frame, which would make PSNR comparisons across repairs meaningless.

---

## 4. Output Contract & Persistence

* **To the trainer:** `{ cameras, images, points3D }` with the same fields `parseImagesBin` produces (`R`, `t`, `center`, `forward`, `view`, `obs2D`, `zMin`/`zMax`), plus `fileBlob` per image (the path the dropped-folder loader already uses for thumbnails). Then `computeCameraDepthStats` and the validation split run unchanged.
* **Undistortion:** a GPU texture remap from `OPENCV` to a same-size `PINHOLE` (no maximal-rectangle crop, unlike `image_undistorter`; as `tools/undistort.py`), applied once when handing over.
* **Export:** `sparse/cameras.bin`, `images.bin`, `points3D.bin` and `images/`, so a browser result can be opened in the COLMAP GUI or compared with `colmap model_comparer` / `model_aligner`. The writer mirrors the parsers in `colmap.js`.
* **Session persistence:** keyframes, features and model go to OPFS (Origin Private File System), so a reload resumes without decoding and extracting again. A `.gosfm` project file (features + model, no operation log) can be saved and dropped back in; the features are required. The features container carries an id, the model stores the id it was built from, and a model is restored only over the same features: after an interrupted run the old model is ignored, with a warning.

---

## 5. Pipeline

Each stage is restartable from its inputs, which is what makes interactive repair (§6) possible.

### 5.1 S0 — Decode & Frame Selection
* Decode with WebCodecs `VideoDecoder`; demux by `mp4.js` (MP4 / MOV sample tables). Fragmented MP4 and compact sample sizes (`stz2`) are rejected with a remux hint (`ffmpeg -i in.mp4 -c copy out.mp4`). Decoding starts one GOP before the first wanted frame (open GOPs reference it).
* Port of `video2splat.py --spacing`: sharpness (Laplacian variance, reject < 0.35 × median) and keep one frame every `spacing` × image diagonal of optical-flow path, computed in JS at low resolution by a pool of workers.
* The path length is the **raw** median LK displacement (`frames.js`), as in the Python version; no homography is removed. A parallax-based path (flow after a fitted homography, so pure yaw in place adds no keyframes) was planned and is not implemented (R4).
* Keyframes can be **inserted later** at any timestamp (§6.3): the video stays available, which COLMAP never had.

### 5.2 S1 — Features
* SIFT in WGSL following the SiftGPU structure: Gaussian pyramid, DoG extrema with subpixel refinement, edge rejection, orientation histogram, 4×4×8 descriptor, RootSIFT normalisation, top-$K$ by response ($K$ = 4096 default).
* Validated against COLMAP's SIFT on the same frames: repeatability and match counts on sequential pairs, not bitwise equality.
* At most 65536 candidate / oriented keypoints per frame on the GPU; past that, which ones are kept depends on the GPU's append order, and the worker warns.

### 5.3 S2 — Sequential Mapping
* **Matching:** keyframe $k$ against $k-1 \dots k-W$ on the GPU: brute-force nearest neighbour on f16 (a batched matrix product), ratio test 0.8, mutual check. No 16384 cap. ~$N \cdot W$ pairs total.
* **Initialisation:** first pair with median parallax above a threshold; relative pose (`relativePose`: 8-point F RANSAC on the undistorted normalized rays, threshold = pixels / $f$, so F is the essential matrix; cheirality) and homography. If the homography explains > 80% of the F inliers (rotation only, or a flat scene), move to a later pair. Triangulate.
* **Per new keyframe:**
  1. Locate against the window's 3D points (P3P + LO-RANSAC, then pose-only refinement).
  2. Triangulate new tracks with window keyframes: viewing angle ≥ 1.5°, positive depth, reprojection error < 4 px.
  3. Window BA (D2/D3): last $W$ poses + their points; older observers fixed.
* **Intrinsics: self-calibrate before mapping.** COLMAP's default initial guess $f = 1.2 \cdot \max(w, h) = 2304$ px is **1.87× off** on jardin5/6, where COLMAP converged to $f = 1230$ px (horizontal FOV 76°, OPENCV $k_1 = 0.005$); its global mapper warns about the missing prior. A window BA started that far off can converge to a wrong focal with bent paths. So:
  1. Estimate fundamental matrices on ~100 sequential pairs (7/8-point LO-RANSAC).
  2. Choose the focal that makes $E = K^\top F K$ most essential-like (two equal singular values), robustly over all pairs: the criterion of COLMAP's `view_graph_calibration`.
  3. Confirm with a 1-D sweep of the first window's BA cost around that value.
  4. `sfm_focal=` (pixels) or `sfm_hfovDeg=` (degrees) URL parameters override.
  Then $f$ and $k_1$ are refined jointly over the first ~50 keyframes, frozen, and freed again in S4.
* **Critical motion** (measured, R1): translation without rotation, typically a nadir mapping pass, leaves the focal unobservable from $F$. The calibration reports a **sharpness** (cost at $f/1.25$ and $1.25f$ over the minimum): jardin5 ≈ 6, a synthetic orbit ≈ 16, a forward flight ≈ 2 (weak: BA corrects it), a nadir pass ≈ 1.05. Below 1.3 the mapper falls back to `sfm_hfovDeg=` / `sfm_focal=`, or to a 76° drone-typical HFOV, and says so.
* **Segments:** when no frame can be registered any more, a new segment starts with its own initial pair. On jardin5 this happens at a cut with **no sequential matches at all** (frames 48→49, joined only by loop pairs) and at two weak links; S3 merges segments.
* **Tracking loss:** too few PnP inliers. Try the BoW index (§5.4) to relocalise against all keyframes; otherwise start a new segment from a new initialisation.

### 5.4 S3 — Loop Closure (Offline Pass)
* **Candidates:**
  * Appearance: bag of words built from this video's own descriptors (k-means to ~4k words on a sample, TF-IDF, cosine similarity on the CPU; 500² is trivial; word assignment uses the GPU matcher). No pre-trained vocabulary to ship.
  * Geometry: keyframes far apart in time whose estimated frusta overlap, with a tolerance that grows with the temporal distance (drift).
  * User: pairs designated in the UI (§6.3).
* **Verification:** full descriptor match → PnP RANSAC of the query against the candidate's points → Sim(3) from matched 3D points (Umeyama + RANSAC). The scale component corrects monocular drift.
* **Correction:** Sim(3) pose graph over all keyframes, points moved with their reference keyframe, then **track fusion**: points triangulated independently on both sides of the loop that match the same keypoints are merged. Without fusion, BA cannot pull the two sides together.
* Segments (§3) are merged by the same mechanism.

### 5.5 S4 — Global Refinement
* Gliding-window BA sweeps over covisibility windows (D3/D4): window offsets alternate by $W/2$ so boundaries do not stay fixed; 2–3 sweeps or until the largest pose change is below a threshold.
* Non-overlapping windows are independent: on the GPU, solve all even windows in one dispatch, then all odd ones.
* Intrinsics are freed in this stage: each window BA refines $f, k_1$ in turn, starting from the previous window's, so the result is the last window's estimate, not a joint fit over all frames. A last sweep with the intrinsics fixed then refits every window to that camera. Without it, each window keeps the poses and points it fitted to its own estimate: the median per-frame reprojection error on gare-st-lazare was 1.97 px (half of its frames red), 0.79 px with it; on jardin5 the p90 drops from 1.62 to 0.73 px and jardin4's from 0.95 to 0.76. A joint fit of the intrinsics over all frames (poses fixed) before that sweep did no better.
* Fallback: one global GPU BA (D2) if sweeps fall short of COLMAP (§8).
* **Final retriangulation and filtering** (COLMAP spends 125 s on this on jardin5, and the trainer initialises splats from its output). Mapping side (`mapper.extendTracks`, formerly `retriangulate`): with the final poses, extend tracks and triangulate new ones over all usable pairs, then drop observations that reproject > 4 px; existing points keep their BA positions. Trainer side (`export.js` `datasetFromMapper`): keep observations ≤ 2 px, then points with ≥ 3 of them and two camera centres seeing the point ≥ 1.5° apart; no re-triangulation, no track split. Colours come from the undistorted images (S5). On jardin5 the trainer filter keeps 121,831 of 202,339 points (≥ 3 observations alone: 139,001; p90 point error 2.01 → 1.34 px). The point count and spatial spread feed the trainer's `sceneRadius` and initial splat scales, so this stage is checked against COLMAP's points3D too (§8).

### 5.6 S5 — Hand-Over
* Undistort (§4), build the dataset, close the panel or keep it open, start training. Every 8th view stays held out for validation, as today.

### 5.7 Platform Constraints
* **Codecs:** the test videos are H.264 High 1080p (jardin5/6 at 30 fps, re-exported through Core Media without metadata; jardin4 is a DJI NEO original at 59.94 fps with audio). DJI originals are often HEVC, sometimes 10-bit: WebCodecs HEVC decoding depends on the platform's hardware decoder. Check `VideoDecoder.isConfigSupported()` first and tell the user to transcode (or use the COLMAP path) when it fails.
* **Demuxing:** frames come out in decode order (B-frames); select by presentation timestamp. Honour the rotation matrix of the track header, and use timestamps rather than frame indices (variable frame rate).
* **Keyframe storage:** JPEG at quality ≥ 95, like `video2splat.py` (`-q:v 2`); these are the trainer's ground-truth images, so a lower quality costs PSNR.
* **One GPU, two devices:** the SfM worker and the trainer each own a `GPUDevice` and cannot share buffers. Running both at once competes for memory (up to 2M splats plus SfM descriptors) and time. Policy: SfM stages pause training and resume it when done; images and the dataset cross over as blobs.
* **Determinism:** every RANSAC and k-means takes a seeded RNG (pair verification: a seed per pair, the same on any number of workers); GPU kernels avoid float atomics where results feed decisions (matching, verification), so a stage re-run on the same input gives the same model. Tests depend on this (§8).
* **Headless WebGPU:** Deno 2.9 on this Mac exposes `navigator.gpu` with `shader-f16` and `timestamp-query` but no `subgroups`. Kernels need a subgroup-free path to run under tests; Chrome-only features stay optional.

### 5.8 Expected Budget (jardin5, estimates; S2 measured in R1)

| Stage | Estimate |
| :--- | :---: |
| S0 decode + selection | 5–10 s |
| S1 SIFT | 10–20 s |
| S2 matching | ~10 s |
| S2 mapping + window BA | 30–90 s (measured: 93 s, plain JS, R1) |
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
| **Retry** | Re-match the range with a wider window ($2W$) and a looser ratio test (0.9), re-locate and window BA | Low inliers, short tracks |
| **Densify** | Insert extra keyframes from the video inside the range (smaller `spacing`), then retry | Fast motion, blur gaps, tracking loss |
| **Drop** | Remove keyframes (blurry, moving objects, lens flare) and re-run window BA around them | Outlier frames |
| **Refine** | Window BA over the range itself (no margin) plus its covisible frames, intrinsics fixed, then new tracks | Residual jitter |
| **Redo** | Cut the range into a new segment, re-initialise and re-map it, merge back with Sim(3) (§5.4) | A broken stretch |
| **Force loop** | User picks two keyframes that see the same place: verify and add a `user-loop` edge, re-run pose graph + fusion | Missed loops |
| **Reject loop** | Mark a loop edge `rejected`, re-run pose graph | False loops (repetitive texture) |
| **Set focal** | Fix or re-estimate intrinsics, re-run refinement | Wrong focal (curved paths, bad scale) |

* **Repair all** (no selection; also run once after a reconstruction unless `autoRepair` is off): retry each stretch of amber / red keyframes, skipping rotation-only frames (their poses are right, a re-match cannot add parallax) and separate maps (they need redo). A retry is kept only when the scores improve over the whole sequence (fewer red, or as many red and more green), otherwise undone: a retry can break a neighbour. Kinks show up as red (scale break, rotation or baseline disagreeing with the matches), so this check covers them. The kept retries are one log entry, and one undo reverts them all. On gare-st-lazare (388 keyframes) it took red from 28 to 9 and green from 326 to 340, with 14 of 24 retries kept, in 216 s. A second pass over the stretches left adds little (2 of 14 kept); `passes` allows it.

* Every action is an operation-log entry (§3): **undo** restores the previous snapshot, and the before/after scores of the affected keyframes are shown.
* Repair matching uses a per-action RNG (seeded by the action's position in the log), so a repair does not depend on what was matched before it. The matching window $W$ is also the mapper's BA window.
* Actions run in the worker; the panel stays responsive and shows progress per stage.
* While training is running, repairs apply to the SfM model only. Pushing an updated model to the trainer swaps the cameras (poses change, splats stay). This needs a new trainer entry point that replaces camera poses in place, keyed by image name; splats remain in the old frame, so pushes are only sensible after a local repair, not after a change of gauge.

### 6.4 As Built (M5, R5)
* **Signals** (`quality.js`): registration and exported segment, support, mean reprojection error, parallax, a **two-view check** (the relative pose of consecutive keyframes from their own matches against the model's, used only when well conditioned: a homography explains < 70% of the F inliers and two RANSAC runs agree), scale breaks (baseline / depth), loop residuals after the pose graph, and validation PSNR per view from the trainer. Two-view thresholds scale with the video's median disagreement.
* **Path smoothness is reported, not scored:** keyframed by optical flow, drone footage stops, starts and yaws in place; on jardin5 every frame it flagged had the same step ratios in COLMAP's model (R5). The two-view check found the real local error instead.
* **Actions** (`session.js`): as in §6.3. Redo merges the re-mapped stretch back by Sim(3) only (a local action must not re-run the global pose graph); Force loop and Reject loops re-run loop closure **and** the refinement sweeps after it; Reject loops replays S3–S4 from the sequential map (later repairs are not kept, the log says so); Set focal re-runs the reconstruction (a focal change reshapes the whole map, local sweeps cannot undo it).
* **Update trainer cameras** (the default after repairs): the repaired model is Sim(3)-aligned onto the trainer's cameras (keyframes in common, refitted without those that moved), the trainer's views are replaced by name (`replaceImages`: photos kept unless the undistortion changed, inserted / dropped keyframes added / removed, octree camera stats refreshed), and the splats, their Adam state and the step count stay. How far the cameras moved is logged, with a warning past 2% of the scene radius or 1°. **Send to trainer (new splats)** re-initialises from the points as before.
* **Persistence:** analysis + keyframes (JPEG, SIFT) and the model are saved in OPFS per video (name, size, date) and restored when it is opened again; **Save project** writes both as one `.gosfm` (`project.js`), which can be dropped back (repairs work; densify and re-runs need the video). The undo history is not saved. With a project folder (DESIGN.md §3.6) the two containers go to its `cache/` instead, and the keyframes, `sharpness.csv`, the distorted model `sparse/0/` (`export.js` with `pinhole: false`) and `gs/` are written into it as they are made.
* **Worker protocol** (`worker.js`): in `open`, `project`, `run`, `action`, `undo`, `send`, `save`, `psnr`; out `video`, `log`, `progress`, `selection`, `quality`, `result`, `restored`, `project`, `idle` (a request with nothing to do, e.g. nothing to undo), `error`. Messages are handled one at a time, in order. A dropped project posts `video` with `{ project, name, duration }`. The options and their defaults live in `src/sfm/defaults.js` (`SESSION_DEFAULTS`, `PIPELINE_DEFAULTS`); the panel imports them.

---

## 7. UI: the "Video → Cameras" Panel

### 7.1 Opening
* **Drop a video file** anywhere on the page: the existing `drop` handler in `src/app.js` routes `video/*` MIME types, `.mp4/.mov/.m4v` and `.gosfm` projects to the panel instead of `loadDroppedFolder`.
* **"Open video or photos…"** in the Project phase, with a file picker (several JPEGs: a photo set, §13).
* **URL:** `?video=URL` opens the panel with that video; SfM parameters are settable as `sfm_<key>`, the keys of `SESSION_DEFAULTS` / `PIPELINE_DEFAULTS` in `src/sfm/defaults.js` plus the panel's `autoSend`, `fixFocal` (`sfm_spacing=0.05`, `sfm_maxFeatures=4096`, `sfm_focal=1230`, `sfm_hfovDeg=76`, …).

### 7.2 Layout (as built: phases, `src/ui/`)
* The page is organized in **phases** (DESIGN.md §6, item 6): the panel's controls (`src/ui/sfm_panel.js`) fill the **Video**, **Keyframes** and **Cameras** phases of the inspector, its exports the Export phase, and the **timeline strip** is a tab of the dock under the canvas. The inspector's footer holds the next step: *Find keyframes →* (a run that stops after the keyframes and their features, `until: 'keyframes'`), *Find cameras →* (the rest, cached stages skipped, then the trainer when *train when found*), *Train on these cameras →* / *Update training cameras*, and *■ Stop* while the worker runs.
* The SfM session stays in the worker and in OPFS while other phases are shown; training pauses while the worker runs (one GPU), with a log line saying so.
* The panel's own log is gone: its lines go to the page log, tagged `sfm`.
* **Contact sheet** (`src/ui/frames.js`): the keyframes as 320 px thumbnails (made once per keyframe in the worker, message `thumbs`) over the 3D view, shown in the Keyframes phase and from the toolbar; framed by score once the cameras exist; a click (shift-click: a range) selects keyframes for the repairs, as the timeline does.

### 7.3 Contents (top to bottom)
1. **Source:** file name, resolution, fps, duration; thumbnail scrubber.
2. **Frames (S0):** sample fps, `spacing`, blur threshold, max resolution; live count of kept keyframes.
3. **Features & matching (S1–S2):** max features, window $W$, ratio test.
4. **Camera:** initial focal (`auto` or pixels) or HFOV, fix the focal (for *Set focal*); the SfM model is `OPENCV`, the export a same-size `PINHOLE`.
5. **Mapping (S2–S4):** min parallax, loop detection on/off, BoW size, number of sweeps.
6. **Run controls:** Run all, Run stage, Pause, Stop; per-stage progress bars and timings.
7. **Quality (§6):** score legend, filters (show red only), selection info, action buttons of §6.3, undo.
8. **Output:** downscale for training, **Send to trainer**, **Export COLMAP**, **Save project (.gosfm)**.
9. **Log:** SfM-specific log (the main `#log-panel` receives one summary line per stage).

---

## 8. Testing, Validation & Milestones

COLMAP is a **reference, not ground truth**: agreeing with it does not prove either is right. The plan therefore has three layers: unit tests with exact answers, synthetic videos with exact poses, and comparison with COLMAP on real videos.

### 8.1 Test Layers

| Layer | Runs under | What | Pass criterion |
| :--- | :--- | :--- | :--- |
| **Unit (CPU)** | `deno test` | 8-point F / relative pose, homography, P3P, triangulation, Umeyama, focal self-calibration on random synthetic scenes with noise and outliers | Recovers the generating pose / focal within a noise-scaled tolerance, over 50–500 seeded trials per test |
| **Solver** | `deno test` | BA Jacobians vs central finite differences; LM convergence from perturbed starts; gauge fixing (no drift of fixed frames); a synthetic loop with injected drift closed by the Sim(3) pose graph | Jacobian error < 1e-4 × (1 + \|J\|); cost reaches the noise floor; loop residual removed |
| **GPU kernels** | `deno test` (WebGPU) | Matcher vs CPU brute force and vs COLMAP's raw matches (slow); SIFT: determinism and 90° rotation invariance on synthetic texture, and (slow) verified matches on jardin5 frames vs COLMAP's (no CPU SIFT reference); undistortion vs a CPU reference; WGSL BA vs JS f64 BA (not built, M6) | Identical match indices; SIFT match counts within tolerance of COLMAP's; undistortion within 1 level |
| **Synthetic video (ground truth)** | `SLOW=1` (`deno task test:slow`), pending the renderer | Videos rendered by Gauss-Opt from a trained jardin model along drone-like paths with known $K$, distortion, blur, a yaw-in-place segment, a revisit loop, and a cut (§8.2) | Thresholds in §8.3 against the **true** poses |
| **Real video vs COLMAP** | `SLOW=1` (`deno task test:slow`) + `pytest --slow` | `samples/garden_flight_5fps_720p.mp4` (already used by `tests/test_pipeline.py`), jardin4/5/6 | Thresholds in §8.3 against COLMAP; validation PSNR after a fixed training budget |
| **Browser end-to-end** | Playwright, headless Chrome with WebGPU (`e2e/sfm.e2e.js`); Safari through safaridriver with `BROWSER=safari` (`e2e/safari.js`, a window, not headless) | `?video=` → panel opens → run → send to trainer → 500 steps → PSNR reported; one repair action + undo | No errors; PSNR within tolerance of the stored baseline |
| **Performance** | same runs | Per-stage wall time logged to JSON | Within 1.5× of the stored baseline (catches regressions, not absolute speed) |

### 8.2 Test Assets
* **COLMAP dump** (`tools/colmap_dump.py`): keypoints, descriptors, verified matches **and the pair list with its origin** (sequential vs loop detection) from `database_sift.db`, plus the model. M1 must only use sequential pairs inside its window; loop pairs are held back so M2 tests loop **detection**, not just verification of pairs COLMAP already found.
* **Synthetic videos** (planned `tools/render_path.js`, not written yet; renderer pending, §8.4): Gauss-Opt renders a trained model from an arbitrary camera path (the trainer already renders from any `view` and intrinsics). Pipeline: spline path with drone dynamics → render frames → apply OPENCV distortion, motion blur and H.264 encoding (ffmpeg) → video plus exact poses and intrinsics. Variants each target one failure mode: wrong focal start, rotation-only segment, loop, tracking loss (scene cut), low texture (grass close-up).
* **Metrics tool** (`tools/compare.js`): Sim(3)-align the camera centres (Umeyama), then report RMS position error / path length, rotation error, focal ratio, % registered, reprojection error, point count and spread. Used by every layer above (`src/sfm/compare.js`, `reference.js`); there is no "compare with COLMAP" view in the UI.

### 8.3 Pass Thresholds (initial values, tightened once baselines exist)

| Metric | vs ground truth (synthetic) | vs COLMAP (real) |
| :--- | :---: | :---: |
| Registered keyframes | ≥ 99% | ≥ COLMAP − 1% |
| RMS centre error / path length | ≤ 0.3% | ≤ 1% |
| Median rotation error | ≤ 0.2° | ≤ 0.5° |
| Focal error | ≤ 1% | ≤ 2% |
| Mean reprojection error | ≤ 1.0 px | ≤ COLMAP + 0.2 px |
| Validation PSNR after the same training budget | — | ≥ COLMAP − 0.3 dB |

### 8.4 Milestones

| Milestone | Content | Estimate | Status |
| :--- | :--- | :---: | :--- |
| **M0** | Test harness: `deno test` layout, seeded RNG, COLMAP dump, metrics tool, synthetic scene generator for unit tests | 1 week | done |
| M1 | Focal self-calibration, S2 mapping + window BA (JS) on COLMAP's sequential matches | 2–3 weeks | done (R1) |
| M2 | S3 loop detection + closure, pose graph, track fusion; S4 sweeps + final retriangulation | 1–2 weeks | done (R2) |
| M3 | WGSL SIFT + matching replacing COLMAP's features; synthetic video renderer | 2–3 weeks | done (R3), renderer pending |
| M4 | S0 video front end, panel (§7), undistortion, hand-over; Playwright smoke test | 1–2 weeks | done (R4) |
| M5 | Interactive repair (§6) with snapshot undo | 1–2 weeks | done (R5) |
| M6 | Global GPU BA, only if M2 sweeps fail §8.3 | 2–4 weeks | not needed so far |

M0 comes first because every later milestone is accepted by these tests. M1 carries most of the algorithmic risk (geometry, BA, intrinsics without prior) and needs no GPU work.

---

## 9. Adversarial Review & Mitigations

* **"f32 BA will not converge on a 90-unit path."** Window problems are solved in f64 JS; GPU problems use per-window centring/scaling and incremental parameters; the JS backend checks the WGSL one on the same inputs.
* **"No focal prior: the whole map is wrong."** Joint estimation on the first ~50 keyframes, re-estimated in S4, user override (§6.3); M1 measures focal error against COLMAP.
* **"Scale drift without loops."** Remains, as for any monocular method; the quality view shows it as loop residuals or path jerk. Not worse than COLMAP for a single pass without revisits.
* **"Repetitive texture (grass, roofs) causes false loops."** Loops need PnP + Sim(3) RANSAC agreement and a pose-graph residual check; the UI shows and lets the user reject loop edges.
* **"Rotation-only segments break triangulation."** Homography test at initialisation and in S0 spacing (parallax, not raw flow); such keyframes are located by PnP only and do not triangulate.
* **"Browser memory."** Keyframes stored compressed; descriptors on the CPU, uploaded per window; full-resolution decode only for SIFT, then dropped.
* **"Locked-in to one browser."** WebGPU in a worker is required; feature-detect and keep the COLMAP path as the fallback.
* **"Rolling shutter."** Drone sensors are rolling shutter; fast yaw skews frames. COLMAP ignores it too. Not modelled; the quality view flags keyframes with high angular velocity so they can be dropped or densified around.
* **"Testing against COLMAP only proves we copy COLMAP."** Synthetic videos with exact poses (§8.2) are the ground truth; COLMAP is a second, independent check on real footage.
* **"Licences."** COLMAP and VLFeat are BSD (no PoseLib code is ported); SIFT's patent has expired. SiftGPU's code is not ported: the WGSL SIFT follows Lowe's paper and VLFeat, and SiftGPU only informs the GPU structure.

---

## 10. File Structure (as built)

```text
gaussopt/
├── DESIGN-sfm.md          # This document
├── deno.json              # deno task test / test:slow
├── e2e/sfm.e2e.js         # Browser end-to-end test (Playwright, headless Chrome)
├── e2e/project.e2e.js     # Project folders end to end (DESIGN.md §3.6)
├── e2e/harness.js         # What the e2e tests share: clip, serve.py, browser and its errors, run()
├── e2e/safari.js          # The page calls the e2e tests use, on Safari through safaridriver (W3C WebDriver)
├── src/ui/sfm_panel.js    # The "Video → Cameras" panel (§7), loaded on first use
├── src/sfm/
│   ├── worker.js          # The session in a worker: S0..S5, stage results cached
│   ├── defaults.js        # Session / pipeline option defaults (worker and panel, URL sfm_<key>)
│   ├── frames_worker.js   # S0 analysis of a chunk of frames (pool of workers)
│   ├── mp4.js             # MP4 / MOV demuxer: sample table, WebCodecs configuration
│   ├── video.js           # WebCodecs decoding, frames drawn upright at a given size
│   ├── frames.js          # S0: sharpness, pyramidal LK motion, keyframe selection (video2splat.py `select`)
│   ├── exif.js            # The EXIF fields a photo set needs (§13.3): capture time, focal, zoom, camera and lens, orientation
│   ├── photos.js          # A photo set as keyframes (§13.3): capture order, names, labels, 90° rotations, a camera per lens and zoom
│   ├── coverage.js        # What to shoot next once a photo set is mapped (§13.3 P4, P6): pieces apart, gaps, points seen by 2 / 3+
│   ├── pipeline.js        # S1..S4 on keyframes (shared by the worker and tools/run_frames.js), geometric loop proposals
│   ├── session.js         # A repairable reconstruction: S2..S4 (with pipeline.js's loop proposals), actions of §6.3, operation log with undo
│   ├── quality.js         # Per-keyframe quality signals and scores (§6.1)
│   ├── project.js         # .gosfm containers (OPFS persistence, Save project)
│   ├── undistort.js / undistort.wgsl  # S5: GPU remap to the PINHOLE camera, point colors
│   ├── linalg.js          # Small dense linear algebra: Jacobi SVD, Cholesky, SO(3) exp / log
│   ├── util.js            # median / quantile, degrees, matched keypoint pixels
│   ├── camera.js          # COLMAP camera models, (un)distortion, poses, field-of-view test
│   ├── align.js           # Umeyama Sim(3) alignment
│   ├── sim3.js            # Similarity transforms for loop closure
│   ├── geometry.js        # F (8-point), H, adaptive RANSAC, essential matrix, relative pose, triangulation, P3P / PnP
│   ├── focal.js           # Focal self-calibration from F, with an observability measure (sharpness)
│   ├── ba.js              # Robust LM bundle adjustment, Schur complement on the points (JS f64)
│   ├── posegraph.js       # Sim(3) pose graph, LM + block-Jacobi PCG
│   ├── mapper.js          # S2: sequential mapping, window BA, segments; S4 sweeps and extendTracks
│   ├── loop.js            # S3: loop verification (3D-3D Sim(3)), segment merging, pose graph, track fusion
│   ├── reconstruct.js     # S2..S4 in order: mapSequential, finish (loops, sweeps, extendTracks, late merge)
│   ├── sift.js / sift.wgsl    # S1: GPU SIFT
│   ├── match.js / match.wgsl  # GPU brute-force matching (tiled), COLMAP's ratio / distance / cross check
│   ├── frontend.js        # Window + loop pairs matched on the GPU, F verification
│   ├── verify_worker.js   # F verification of matched pairs, in frontend.js's pool of workers
│   ├── loopdetect.js      # Loop candidates by appearance (vocabulary, TF-IDF) and by geometry
│   ├── export.js          # One segment -> COLMAP *.bin (PINHOLE, undistorted keypoints), trainer point filter
│   ├── compare.js         # Accuracy vs a reference, §8.3 thresholds
│   ├── reference.js       # Defects of a COLMAP reference, per-piece assessment
│   ├── synth.js           # Seeded synthetic scenes with ground truth
│   └── dump.js            # Loader for tools/colmap_dump.py
├── tools/                 # Deno / Python command-line tools (README "In-browser SfM")
│   ├── common.js          # CLI parsing, COLMAP model loading, dataset export, reference report
│   ├── map.js             # Pipeline on a COLMAP dump (--ref, --export)
│   ├── run_frames.js      # End to end on frames, no COLMAP (--ref, --export)
│   ├── compare.js         # Two COLMAP models
│   ├── colmap_dump.py     # COLMAP database -> dump
│   ├── gray.py / pgm.js   # Frames -> PGM for Deno
│   └── undistort.py       # Undistorted images + point colors for an export
└── tests/                 # deno test: unit, solver, GPU, synthetic, real data (SLOW=1)
```

---

## 11. Open Questions

* Is a 1080p keyframe needed for SIFT, or does half resolution register as well (4× less SIFT and memory)?
* KLT tracking for S2 instead of descriptor matching: fewer outliers on small baselines, but a second code path.
* Does window-sweep refinement match COLMAP's global BA on jardin5/6, or is M6 required?
* DJI originals may carry telemetry (GPS, altitude, gimbal attitude) in a metadata or subtitle track; the jardin test files do not. When present, it gives scale, gravity and loop candidates. Worth a probe once an original is available.
* Do COLMAP's scale breaks on jardin5 (R1) explain the splat scale problems seen when training on that model (far "shell" splats with the same parallax as near content)? Train on both models and compare.
* Can the trainer's photometric error refine poses directly (pose gradients in the backward pass), making S4 partly redundant?
* *Update trainer cameras* keeps splats that were trained on a wrong pose: after a camera repair they still encode the error. Would a targeted opacity reset of the splats seen mainly by the moved cameras recover faster than plain updating? Experiment: rotate one camera by 1.5°, train, swap in the correct camera, and compare that view's PSNR for plain update / update + targeted reset / new splats.

---

## 12. Results

### R1 — M1 sequential mapper on synthetic scenes and jardin5/6 (2026-10-21)
* **Setup:** `tools/map.js` on COLMAP database dumps (`tools/colmap_dump.py`), **sequential pairs only** (loop pairs held back for S3), window 10, focal from F (prior 2304 px). Synthetic scenes from `synth.js` compared with exact ground truth; real data compared with COLMAP's `global_mapper` model, globally (Sim(3)) and with the gauge-free local metric (`compareLocal`: relative rotation and baseline direction of consecutive frames).
* **Synthetic** (30–50 frames, 0.5 px noise, `tests/mapper_test.js`): all frames registered in one segment; RMS center error 0.004–0.03% of the path, median rotation 0.01–0.2°, focal 0.1–0.5%, also with 20% outlier matches, a yaw-in-place segment and a wide lens. Nadir pass: focal unobservable (§5.3), detected; with the HFOV given, focal 0.2%, centers 0.2%, rotations 0.8° (bas-relief ambiguity over flat ground).

| | jardin5 (425 frames) | jardin6 (525 frames) |
| :--- | :---: | :---: |
| Registered | 425 (4 segments) | 525 (2 segments) |
| Focal (COLMAP) | 1221–1224 px (1231) | 1229 px (1228.5) |
| Time (S2, plain JS, M5 Pro) | 93 s | 116 s |
| Local relative rotation vs COLMAP, median / p90 | 0.017° / 0.033° | 0.016° / 0.42° |
| Local baseline direction vs COLMAP, median / p90 | 0.18° / 2.3° | 0.19° / 0.9° |
| Sim(3) per segment vs COLMAP (RMS center) | 0.10%, 0.42% (segments 1, 2); 10%, 8% (0, 3) | 1.4% (479 frames, no loops), 0.08% |

* **COLMAP's model is not a reliable reference everywhere.** Where the two disagree, the evidence points at COLMAP:
  1. jardin5 has **scale breaks**: at 26→27 the baseline is 34× the scene depth for 20 video frames (0.67 s, 38 px median flow), and the scene depth jumps 150×; also 48→49 (4×, across the cut with no sequential matches) and 175→176 (1×). Scale-free (baseline / depth), our model agrees with COLMAP everywhere else in frames 0–48.
  2. jardin5 frame 379 differs by 41° in relative rotation: COLMAP registered it from **97** observations, ours from 1398 at 0.53 px.
  3. jardin6 frames 222–243 differ by ~1° per consecutive pair; independent two-view rotations (essential matrix from each pair's matches) agree with ours to 0.03–0.37° and with COLMAP to 0.35–2.4°. The stretch is near-planar (H/F inlier ratio 0.86–1.0).
* **Decision: never match COLMAP where it is shown wrong.** Its defects are found from independent evidence and excluded, and the strict `colmap` thresholds of §8.3 apply to what remains (`src/sfm/reference.js`):
  * weak frames (observations < max(150, 10% of the median)), excluded;
  * scale breaks (a step's baseline / depth > max(0.3, 10x the median)), where the reference is cut;
  * arbitrated steps: where the estimate and COLMAP disagree on a relative rotation by > 0.2°, the two-view rotation from that pair's matches decides; when it sides with the estimate, the reference is cut there.
  
  Each consistent piece is compared with its own Sim(3) (rotation from the camera orientations, scale and translation from the centers; errors relative to the path, or to the scene depth when hovering). Tolerances are never loosened to absorb a reference error; synthetic ground truth stays the primary acceptance test.
* **Result with defects excluded:** jardin5 excludes 3 weak frames and 6 breaks (26→27, 48→49 and 175→176, three steps around frame 379) and passes: 6 pieces, RMS center 0.04–0.45%, median rotation 0.09–0.37°. jardin6 excludes 38 arbitrated steps (around frames 222–243 and 390–440); three pieces still fail (RMS 1.3–2.7%, up to 1.1°): sequential-only drift on a loop-heavy flight, or COLMAP bending by less per step than two-view rotations can resolve. Not attributed yet: M2 must explain them.
* **Next (M2):** merge segments and close loops (Sim(3) + pose graph), then window sweeps; the three failing jardin6 pieces are the first target.

### R2 — M2 loop closure, segment merging, refinement (2026-10-21)
* **Pipeline** (`src/sfm/reconstruct.js`): sequential mapping (R1) → `loop.js`: long-range and cross-segment pairs verified by a RANSAC Sim(3) on 3D-3D track correspondences, scored by reprojection in both frames (scale-free); segments merged by the pooled Sim(3) of all pairs between them (largest support first); per segment a Sim(3) pose graph (`posegraph.js`: covisibility edges + the best 3 loop edges per frame, LM with block-Jacobi PCG), points moved with their reference frame; tracks fused across verified loops → gliding covisibility-window BA sweeps with intrinsics free → final retriangulation over all pairs.
* **Loop detection is not part of M2:** loop pairs are given (COLMAP's vocabulary-tree pairs, or synthetic revisits). Detecting them needs descriptors and the matcher, so it moves to M3.
* **Synthetic** (80 frames, 1 px noise, 1.3 turns): a cut with no sequential pair across it goes from 2 segments and 11.5% RMS error to 1 segment and 0.003%; with 20% outlier matches as well, 8.2% → 0.007%; the plain orbit 0.008% → 0.003%. Loop closure takes 6–10 s (adaptive RANSAC, at most 3 loop edges per frame; the first version verified all 2415 loop pairs in 109 s).
* **jardin5:** segments 0–2 merge (frames 0–365); frames 366–424 stay a separate segment. The hover at frames 362–373 (yaw without translation, baseline / depth ≈ 0.001) triangulates nothing, and no structure is shared across it: 43 cross pairs with 47,515 matches give **zero** 3D-3D correspondences, so the scale and position relation is unobservable there. COLMAP joined them, and that is where its defects are (frame 379, breaks at 378–380). Every consistent COLMAP piece passes the `colmap` thresholds (RMS 0.04–0.45%). Times: map 93 s, loops 4 s, refine 39 s, retriangulate 6 s.
* **jardin6:** merges into one segment, pose graph cost 103 → 2.7, but the three pieces of R1 still fail (RMS 0.6–4.2%, rotation 0.5–1.9°). Two-view arbitration over **all** matched pairs (not only consecutive ones) settles part of it:

| Frames | Pairs | Two-view vs ours (median) | Two-view vs COLMAP (median) | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| 0–30, loop pairs | 9 | 0.08° | 1.46° | COLMAP wrong |
| 205–240, gaps 4–15 | 56 | 0.29° | 3.39° | COLMAP wrong |
| 31–200, 245–420, 421–524 | 76–786 | 0.04–0.75° | 0.03–0.72° | equal (COLMAP closer on 55–65% of pairs) |

  The failing piece around frames 245–420 is therefore not a rotation error of either model; the disagreement is in positions / scale, which two-view rotations cannot see. **Unattributed.** The decisive check is the one of §8.3: train on both models and compare validation PSNR.
* **Open:** a hover that rotates past the shared structure cannot be bridged geometrically; §6 must show it as a weak section (the repair is more video around it, or a revisit). Loop detection (M3). PSNR comparison of the two models (export: `tools/map.js --export`).

### R3 — M3 GPU features, matching and loop detection, end to end without COLMAP (2026-10-21)
* **Matching** (`match.wgsl`, tiled 64 x 64 with 4 x 4 blocks per thread, `dot4U8Packed`): bit-exact with the CPU reference and with COLMAP's own matches (91,900 of 91,900 on 20 jardin5 pairs). 4.7 ms per pair at 4096 features (21 ms for the first one-query-per-thread kernel); each frame's pairs go in one submit and one readback.
* **SIFT** (`sift.wgsl`, first octave 0): ~55 ms per 1080p frame; deterministic, invariant to a 90° rotation. On the pairs COLMAP verified, our features and matcher give 0.95x its verified matches (COLMAP upsamples the first octave).
* **Loop candidates:** appearance alone (bag of words learned from the video, 4096 words) finds only 22% of COLMAP's strong loop pairs in the top 8 per frame. After sequential mapping, **geometric** candidates count the sampled points of frame i that land in frame j seen from within 25° and at a depth ratio < 1.5 (what SIFT can match): 52% at 8 per frame, 69% at 16. Both are used: appearance candidates are matched with the sequential window, geometric ones once the sequential map exists (then `reconstructAsync`; now `pipeline.js` / `session.js`).
* **End to end** (`tools/run_frames.js`, frames -> GPU SIFT (4096) -> GPU matching (window 10 + loops) -> F verification -> reconstruction), against COLMAP's ~755 s (13 s features, 252 s matching, 490 s mapping):

| | jardin5 (425) | jardin6 (525) |
| :--- | :---: | :---: |
| Features / loop detection / matching | 23 s / 1 s / 44 s | 29 s / 1 s / 54 s |
| Reconstruction (incl. geometric loops) | 75 s | 114 s |
| **Total** | **142 s** | **198 s** |
| Registered, segments | 425, 3 | 525, 1 |
| Focal (COLMAP) | 1236 px (1231) | 1228 px (1228.5) |
| Consistent COLMAP pieces passing | 6 / 6 | 6 / 7 |

  Re-run after the review pass (`2da2852`), whose fix applies the field-of-view test to geometric candidates; before it: 141 s / 203 s, 6 / 6 and 4 / 6 pieces, the same conclusions.
* jardin6's frames 3080–4009 (1.4% RMS, 0.50°) disagree with COLMAP in every variant (COLMAP's matches or ours, with or without loops); still unattributed, see R2.
* **Datasets for the PSNR comparison** (§8.3): `jardin5/out/gs_sfm` (366 frames, COLMAP's matches) vs `jardin5/out/gs_colmap366` (COLMAP, same frames); `jardin6/out/gs_sfm` vs `jardin6/out/gs`; fully in-house: `jardin5/out/gs_ours` (365 frames), `jardin6/out/gs_ours`.


### R4 — M4 video front end, panel, undistortion and hand-over (2026-10-21)
* **Demuxing** (`mp4.js`): jardin5.mp4 (H.264 High, 3723 samples, B-frames) in 6 ms, jardin4.mov (8126 samples) in 3 ms; presentation times from `ctts` and the edit list.
* **Keyframe selection** (`frames.js`, a port of `video2splat.py select`: Laplacian variance at 1000 px, median pyramidal-LK motion of 400 Shi-Tomasi corners at 640 px, spacing bins, blur rejection). Under Deno, on frames decoded by ffmpeg: jardin5 keeps 421 frames (Python: 425), jardin6 515 (525); the path between Python's picks is 0.996x / 0.994x theirs (p10–p90 0.97–1.00); sharpness is a constant 1.027x (their JPEG round trip). The parallax-based spacing of §5.1 is not implemented: the port keeps raw flow, so selections stay comparable.
* **Browser S0:** the hardware `VideoDecoder` runs at ~1200 fps (headless Chrome, M5 Pro); drawing frames into a `willReadFrequently` canvas cost 16 ms per frame (CPU readback and scaling of the full frame), a GPU canvas 2.5 ms. With the analysis spread over 8 workers, the 3723 frames of jardin5 are analyzed in 15 s. Chrome's scaler gives a sharpness 1.47x (p5–p95 1.40–1.55) the OpenCV one; the selection is relative to the median: 419 keyframes vs Python's 425, 175 identical and 122 one frame away.
* **Undistortion** (`undistort.wgsl`): within 1 level of a CPU reference built on `camera.js`; on a jardin5 frame, 1.47 levels mean difference to `tools/undistort.py` (cv2, one JPEG round trip each side; 14.4 without the remap). Same-size pinhole as `undistort.py`, no maximal-rectangle crop.
* **Determinism:** the GPU appends SIFT keypoints in a varying order and real frames have DoG response ties, so the same frames gave slightly different models from run to run (jardin5 focal 1236.0 / 1236.3 / 1237.3 px). Ties are now broken by octave, position and orientation (`sift.js`): two runs of `tools/run_frames.js` on jardin5 give byte-identical models (focal 1235.6 px), and the slow SIFT test checks re-runs on real frames.
* **End to end in the page** (headless Chrome, `e2e/sfm.e2e.js`, `?video=`, no COLMAP):

| | 12 s of jardin5 (362 frames) | jardin5, whole video (3723 frames) |
| :--- | :---: | :---: |
| S0 analysis / S1 keyframes + SIFT | 2.4 s / 1.8 s | 15 s / 20 s |
| Matching / reconstruction / hand-over | 2 s / 9 s / 0.8 s | 31 s / 90 s / 8 s |
| **Total** | **16 s** | **164 s** (COLMAP path: ~12.5 min) |
| Keyframes, registered, segments | 35, 35, 1 | 419, 419, 3 |
| Exported (largest segment), points | 35, 7,343 | 359, 77,298 |
| Focal | 1231 px | 1236 px |
| Validation PSNR after 500 steps | 17.0–17.5 dB | 13.1 dB (500 steps ≈ one pass over 359 views) |

* **Not in M4** (moved to M5, where repairs need them): OPFS persistence and `.gosfm` projects, the live map in the main canvas while mapping, the name-hash validation split, `keepFps` and per-stage re-runs in the panel (stages are cached in the worker, so re-running after changing a mapping option skips S0–S1). Rotated and HEVC videos are untested (no assets).

### R5 — M5 quality signals, repairs with undo, persistence (2026-10-21)
* **Quality on jardin5** (COLMAP's matches, `tests/quality_test.js`): 374 green, 49 amber (mostly rotation-only stretches and thin support), 2 red: frame 49, a scale break (baseline 0.31 x depth; COLMAP breaks there too), and **frame 176, a real error of ours**: its rotation from 175 is 1.5° off both the pair's own matches and COLMAP, which agree within 0.4°. No loop is suspect. 6 s per full evaluation (two-view RANSAC), 0.23 s when only the model changed (two-view poses cached by name, seeded per pair).
* **What did not work:** path smoothness flagged ~100 frames of jardin5 whose irregular steps COLMAP's model shows too (the drone stops, starts and yaws in place): reported, not scored. The two-view rotation from F is unstable on planar ground (1–3° between RANSAC runs, a homography explaining 75% of the matches): it is only trusted when a homography explains < 70% of the F inliers and two runs agree.
* **Repairs** (`tests/session*_test.js`, synthetic, exact truth): retry and redo bring a damaged stretch (2–4°, 0.5 m) back within the `truth` thresholds; drop, refine, insert (densify), force loop (joins two segments), reject loop and set focal end within them too; undo restores the exact poses. On jardin5's frame 176, refine, retry and redo leave the 1.5° (the link has 258 matches); densify, the intended fix, needs the video (the page).
* **Late merge** (end of S4, `reconstruct.js`): refinement extends tracks across a segment gap, so segments are offered for merging once more, then retriangulated and swept. jardin5 on COLMAP's matches: one segment of 425 frames (was 366 + 59), every COLMAP piece passing. With our matches: no change in `tools/run_frames.js` (no pairs across the hover), but in the page (other keyframes, R4) 417 of 419 keyframes go to the trainer (was 359).
* **End to end in the page** (`e2e/sfm.e2e.js`): 12 s clip: retry + undo and repair all + undo restore the scores exactly, densify adds 3 keyframes that reach the trainer, a reload restores the 38 keyframes from OPFS without a run, 37 s in all. Whole jardin5: 222 s to the cameras (OPFS writes included), 417 cameras, validation 12.9 dB at step 500, the same repair checks, 272 s in all.
* **jardin6 vs COLMAP since the deterministic SIFT (M4):** pieces 6 / 8 pass; frames 2160–4009 (1.19%, 0.56°) and 4024–4038 (10 frames, 3.1%) fail, the stretch R2 / R3 could not attribute (R3: 3080–4009 at 1.44%). R4 did not re-run this comparison; the M5 changes do not affect it (same numbers as `5263178`).
* **Tests:** 60 fast tests in 69 s (`--parallel`), up from 51 in 50 s.
* **Camera update without retraining:** 12 s clip, densify then *Update trainer cameras*: 35 views kept, 3 added, unchanged cameras moved by 0.001% of the scene, step count and 27,791 splats kept, validation 16.8 dB (step 500) -> 17.5 dB (step 1000). After *Set focal* 1180 px (from 1231, a full re-run in another gauge): aligned within 0.25% of the scene radius (median; 1.0° median turn), photos reloaded, validation 17.4 (step 1000, before) -> 17.8 -> 18.1 dB (steps 1500, 2000).

### R6 — review pass (2026-10-21)
* **Tests:** 71 fast tests pass, plus 9 slow ones (`SLOW=1`); new: `frontend_test`, `session_replay_test`, `sfm_fixes_test`, `ply_octree_test`. The matcher test now asserts equality with COLMAP's raw matches (91,900 / 91,900, R3).
* **jardin5 (COLMAP's matches) unchanged where it matters:** 1 segment, 3 merges, quality 374 / 49 / 2 (red: 49 and 176), every consistent COLMAP piece passes. One piece improves from 0.317° to 0.277° median rotation with `relativePose` on undistorted rays (§5.3).
* **Points for the trainer** (§5.5): `extendTracks` on the mapping side, the ≤ 2 px / ≥ 3 obs / 1.5° filter in `datasetFromMapper`: 121,831 of 202,339 points kept, p90 point error 2.01 → 1.34 px.
* **Timings** now accumulate repeated stages (refine, retriangulate run again after a late merge), so those read higher than in R2–R5.
* **Tried and reverted:** a ceil stride in `calibrateFocal` (loses the late merge: 2 segments, 73 red); a both-ends gap in `partners()` (a jardin5 piece 0.317° → 0.352°, 0.505° with the `relativePose` fix, fails: the frames just after a window must overlap the next one). The `grow()` frontier was measured at 212 ms of 125 s and not optimised.
* **Undistortion:** point colours sample the containing pixel (floor, COLMAP's centres at +0.5); `tools/undistort.py` peak memory 1.95 GB → 220 MB.
* **Slow / e2e:** 78 slow tests pass (16 min). The 12 s jardin5 clip in the page: validation 16.48 dB at step 500, 17.65 dB after densify + update (as in R5), every repair check passing.
* **VP9** (`mp4.js`): `vp09.PP.LL.DD` from the vpcC record, the profile from the first frame's header (jardin10.mp4's vpcC says profile 1, 4:4:4 for a profile 0, 4:2:0 stream); decoding falls back to software without a hardware decoder. A 12 s jardin10 clip: 33 / 33 keyframes in one segment, repairs passing, validation 15.6 dB at step 500, 18.7 dB at 1000.
* **Safari 27** (`BROWSER=safari`, safaridriver): the 12 s clip passes every check in 104 s (Chrome 43 s). No WebGPU `subgroups` (the fallback shaders run); WebCodecs decodes H.264, VP9 and HEVC. Frame analysis 17 s vs 2.7 s, SIFT 10.7 s vs 1.8 s, undistortion 8.8 s vs 0.8 s. Validation differs by a few tenths of a dB for input reasons: Safari's canvas scaler gives 0.54x the sharpness, so 3 of 35 keyframes are neighbouring frames, which changes the name-hashed validation set (6 views vs 7); Safari also varies from run to run (16.24-16.33 dB at step 500), Chrome does not (16.48).

---

## 13. Photo Sets: ~10 Photos, Mixed Lenses and Orientations (plan)

### 13.1 Scope
* **Input:** about 10 photos of one scene (the test sets have 17–23), taken with **one phone or camera**, which may mix **lenses** (e.g. ultrawide and main), **digital zoom** and **portrait / landscape** (P8). Photos of another device, or of another aspect ratio once rotated, are refused with the reason.
* **Labels:** every photo keeps its lens, zoom and orientation as separate labels (from EXIF), shown in the UI and usable as filters: *only ultrawide photos*, *only main 1x*, *only landscape*, … (P8). The lens and zoom decide the camera model; the orientation does not (portraits are rotated).
* **Output:** the same as for a video (§4): a `gs/` dataset, the distorted model in `sparse/0/`, a project folder (DESIGN.md §3.6).
* **Not a video:** no frame selection (S0), no temporal neighbours, no camera path to interpolate. The photos are taken on purpose, so each is kept: a blurry one is reported, not dropped.

### 13.2 Assessment (measured 2026-10-24, M5 Pro, `192bc53`)
Ten images in, every pair matched (`matchFeatures` with `window` ≥ 9), then `reconstructFromFeatures`. The scripts were one-off and are not in the repository.

| Input | Pairs verified | Registered | Notes |
| :--- | :---: | :---: | :--- |
| Synthetic arc (`synth.js` orbit, `numFrames: 10`, `overlap: 9`), 4°, 9°, 18°, 36° between views | 45 / 45 | **10 / 10**, 1 segment | focal 1228–1229 px (true 1230), centres ≤ 0.036% RMS, rotations ≤ 0.04° median |
| gare-st-lazare, every 4th keyframe (`t0000167` … `t0008568`, a walking chain) | 19 / 45 | 7 / 10, 2 segments | focal 948 px (our full-video model: 942) |
| gare-st-lazare, every 8th | 13 / 45 | 0 / 10 | |
| jardin5 / jardin7, 10 keyframes over the whole flight | 3 / 45, 12 / 45 | 0 / 10 | too sparse to stand for photos taken on purpose |

* **The geometry holds** (focal calibration, PnP, BA, segments): with correct matches between all pairs, 10 views 4° to 36° apart are solved exactly. The synthetic matches do not degrade with the viewpoint, as SIFT's do, so this tests the solver only.
* **Features are not the limit:** on the gare chain, 4096 and 16384 SIFT features give the same 19 verified pairs (inliers per pair 16–979 vs 15–1173).
* **The mapper loses a well-matched chain.** Consecutive gare pairs have 189–979 inliers, yet:
  1. `initializeWith` takes the **first** pair in index order that passes (0–1, 184 F inliers, median angle 14.8°), not the strongest (4–5, 979). Forcing 4–5 first was rejected by the initial-pair checks; which one (homography ratio, angle, the 100 points within 4 px) is not established yet.
  2. Frame 2 (59 matches with frame 0, 268 with frame 1) is registered, then triangulates only **17** new points. Frame 3 then has 9 2D-3D correspondences (511 matches with frame 2) and is never registered. Not explained yet (a poor pose of frame 2 from few correspondences is the first suspect).
* **Already fit for ~10 photos:** with `window` 10, `matchFeatures` matches every pair up to 11 images; loop candidates need a gap > 2 × `window`, so none are proposed (harmless). Window BA and the S4 sweeps take the `window` frames nearest by index: with n ≤ `window` that is one global BA. `session.insert` sorts frames by name, so photos added later keep the order of §13.3.
* **Real photo sets** (Pixel 10, 3000 × 4000 portrait, 24 mm equivalent, EXIF orientation 1 throughout; two interiors, `cuisine` 18 photos and `salon` 23). Ours at 1440 × 1920 (`maxDim`), 4096 features, every pair matched; COLMAP `feature_extractor` (one OPENCV camera) + `exhaustive_matcher` + `mapper` on the original photos (default options) as the reference:

| | cuisine (18) | salon (23) |
| :--- | :---: | :---: |
| Pairs verified (ours) | 51 / 153 | 83 / 253 |
| COLMAP: registered, models | 12, 1 | 19, 1 |
| COLMAP: mean reprojection, track length | 1.15 px, 3.2 | 1.12 px, 3.2 |
| Ours: registered, segment sizes | 14: 2 + 3 + 3 + 4 + 2 | 21: 14 + 3 + 2 + 2 |
| Ours, 8192 features or `minInliers` 15 | 17: 8 + 3 + 4 + 2 | 19: 14 + 3 + 2 (8192); 18: 16 + 2 (`minInliers` 15) |
| Focal at 1440 px wide: ours / COLMAP / EXIF | 1485–1515 / 1379 / 1331 | 1388–1392 / 1392 / 1331 |
| Ours vs COLMAP per segment (Sim(3)), centers RMS / rotation median | 0.24–0.75% / 0.8–1.9° | 0.26–0.44% / 0.2–0.7° |

  * **Fragmentation is the failure:** all but 1–5 photos get a pose, but in 4–5 segments that are never merged (`loop.js` needs 30 3D-3D inliers between two segments; tracks of 2–3 views rarely give them). The trainer would get the largest segment: 4 (or 8) of 18 cuisine photos, 14 (or 16) of 23 salon photos, against COLMAP's 12 and 19 in one model.
  * **Focal:** on cuisine BA drifts it 8–10% above COLMAP's while the segments are small; salon agrees within 0.3%. The EXIF focal ($f_{35} \sqrt{w^2 + h^2} / 43.27$) is 3.5% / 4.4% below COLMAP's (which also fits $k_2 = -0.14$ / $-0.09$, $p_1$, $p_2$; we fit $k_1$ only): a prior, not a value to fix. Starting from it joins no segments (largest still 4 and 14).
  * COLMAP is a reference only (§8): it registers 12 of 18 cuisine photos itself.
* **Mixed outdoor sets** (`out_scene1`–`4`, a garden, 23 / 21 / 17 / 19 photos, Pixel 10, mixing on purpose the ultrawide (14 mm equivalent) and main (24 mm) lenses, digital zoom 2x (1.7x on two ultrawide photos of `out_scene4`) and both orientations; every photo is 4000 × 3000 once portraits are rotated by 90°). Match graph over all photos of a set (ours, 1920 × 1440, every pair, F-verified):

| | out_scene1 | out_scene2 | out_scene3 | out_scene4 |
| :--- | :---: | :---: | :---: | :---: |
| Pairs verified | 69 / 253 | 25 / 210 | 17 / 136 | 20 / 171 |
| Largest component, pairs ≥ 15 / ≥ 50 inliers | 21 / 14 | 10 / 5 | 6 / 5 | 9 / 6 |
| Verified pairs joining two lenses or zooms | 33 / 69 | 16 / 25 | 12 / 17 | 14 / 20 |
| COLMAP, a camera per lens, zoom and orientation | 13 | 6 | 4 | 3 |
| COLMAP, a camera per lens and zoom, portraits rotated | 13 | 6 | 4 | 3 |
| COLMAP, main lens 1x only, rotated, one camera | 4 / 13 | no model (9) | no model (8) | no model (5) |
| Ours, main lens 1x only, rotated | 0 / 13 | 0 / 9 | 0 / 8 | 0 / 5 |
| Ours, ultrawide 1x only, rotated | **7 / 7**, 1 segment | 0 / 9 | 0 / 5 | 2 / 6 |

  * **Coherent single-lens subsets do not exist in these sets:** a scene holds together through pairs of different lenses (half of the verified pairs or more). The only connected one-lens subset, `out_scene1`'s 7 ultrawide photos (4 landscape, 3 rotated portraits), is solved by ours in one segment, focal 755 px (COLMAP 753, EXIF 777 at 1920 px wide). With all 23 photos forced into one camera, ours registers 2.
  * **Orientation is free:** rotating the portraits gives COLMAP the same registrations with fewer cameras (models with 3 / 2 / 2 / 2 cameras instead of 5 / 4 / 3 / 2), and our ultrawide subset mixes both orientations in one camera.
  * **Per-lens focals match EXIF** where a lens has enough photos (`out_scene1`, COLMAP at 1920 px wide): main 1362 (EXIF 1331, +2.3%), 2x 2690 (EXIF 1331 × 2 = 2662, +1.1%), ultrawide 753 (777, −3.1%). The 2x camera is the main one with twice the focal (2690 / 1362 = 1.98). With 6 photos per camera, of which few registered (`out_scene4`, 3 in the model), COLMAP's focals are off: 2x 2055, ultrawide 424 (EXIF 2662, 777).
  * **`out_scene2`–`4` are too sparse for any SfM tried:** their match graphs are in pieces of 5–10 photos even with every lens. They test graceful failure: which photos join, and what to shoot to link the rest.
* **Tied to video:** `worker.js` `run` refuses to run without a video, S0 and S1 decode it; `app.js` sends any multi-file drop to the COLMAP folder loader; the focal falls back to a 76° drone HFOV (`mapper.js` `fallbackHfovDeg`); `quality.js` scores each keyframe against the **previous** one (parallax, two-view check, scale break, path smoothness); densify decodes the video; retry and insert match by index distance (`session.js` `_neighbourTodo`); the phases have Video and Keyframes, the dock a video timeline.
* **Single camera already assumed downstream:** the trainer takes one training size from `images[0].camera` (`trainer.js`) and `replaceImages` refuses a size change. The one-camera scope needs no change there.

### 13.3 Design
* **P1. Ingestion.** A drop (or picker) of image files with no `sparse/` opens the photo path; a folder with `sparse/*.bin` stays the COLMAP path. The worker gets `{ type: 'photos', files }`: decode, apply the EXIF orientation, label (lens, zoom, orientation, P8), rotate portraits to landscape, check one aspect ratio for all (else refuse, naming the odd ones), scale to `maxDim`, JPEG at `jpegQuality`, SIFT. Order by EXIF `DateTimeOriginal`, then file name; names are `p<NNN>_<original stem>.jpg`, so name order is capture order, as the trainer and `session.insert` expect. The OPFS session key is a hash of the files' names, sizes and dates. To check on the test photos: HEIC decoding (iPhone default) in Chrome and Safari, and whether `createImageBitmap` applies the EXIF orientation by default.
* **P2. Focal prior from EXIF.** `FocalLengthIn35mmFilm` gives $f$ in pixels (35 mm equivalent over the 43.27 mm diagonal: $f_{px} = f_{35} \sqrt{w^2 + h^2} / 43.27$; 3–4% below COLMAP's on the Pixel 10 sets, §13.2). It replaces the 76° drone fallback, and is compared with the self-calibration (§5.3) when that is sharp: a disagreement past a few percent is logged. Without EXIF, the self-calibration, then a phone-typical HFOV, said so in the log.
* **P3. Mapping an unordered set.**
  1. **Initial pair by score**, not by index: pairs ranked by F inliers among those passing the homography and angle checks. Find out why 4–5 fails those checks on gare first.
  2. **Thin registration:** find the cause of §13.2 item 2 before changing thresholds (`minInliers` 30 for PnP, 100 for the initial pair, `minAngleDeg` 1.5°). Candidates: refine the new pose with BA over its own triangulation (a local BA before `triangulateNew`, not after), retriangulate against every registered view instead of the first neighbours, guided matching (re-match a verified pair along its epipolar lines).
  3. Window BA stays as it is (global for n ≤ `window`); `window` is set to at least n − 1 for photo sets so every pair is matched.
  4. Loop closure is off for photo sets (every pair is matched already); segment merging stays (a set may split).
* **P4. Quality and repairs.** Per photo: registered, support (points, mean track length), reprojection error, and its strongest link (the most inliers with a registered photo), with no notion of "previous". Repairs: drop, refine, retry (re-match against **every** photo with the looser ratio test), redo, set focal; densify is replaced by **Add photos** (`session.insert`, matched against every photo). The panel tells the user where to shoot more: a photo whose strongest link is weak, or a segment of its own.
  * **Add photos (as built):** *Add photos…* in the Cameras phase (photo sets only). `photos.js` `planMore` keeps the new photos of the set's camera and size (rotated to its orientation), named after the set's last in capture order (names are padded to 3 digits, so a set numbers up to 999); the worker extracts them as at S1 and `session.insert` matches them against every photo, registers them, and with `regrowSegments` regrows the pieces apart into the main one (`tests/session_edit_test.js`: a photo between two pieces joins them). Undone like a repair. cuisine, the last 4 photos added to the first 14 (`e2e/photos_add.e2e.js`): the main piece goes from 9 to 11 photos; mapping all 18 at once gives 12.
* **P5. UI.** One **Photos** phase in place of Video and Keyframes (the contact sheet `frames.js` as the main view, thumbnails in capture order, each with its labels); no timeline tab for photo sets. The label filters of P8 sit above the sheet. Cameras, Train, Export unchanged.
  * **As built:** `phases.js` `phaseLabel` / `phaseShown`: the stepper of a photo set is Project · Photos · Cameras · Train · Export (Keyframes hidden, its state *skipped*), the Photos step's summary *N photos*. The dock opens on the log and has no timeline tab. The Photos phase (the Video section) holds the source, the *Use only…* filters and the shooting guide; its footer *Find cameras →* and *Run all, then train* (before, the footer ran only an open video: a photo set's buttons were disabled). The contact sheet opens with the set: every photo in capture order, upright as shot (not rotated as mapped), captioned *p007 · 14 mm · portrait · 25 s* (35 mm equivalent focal with the zoom, orientation as shot, time from the first); photos left out (a filter, another size) are greyed with the reason and cannot be selected. After mapping, the sheet's frames take the quality colours as for a video. `e2e/photos.e2e.js` checks the phase, the sheet's caption and runs from the footer button.
* **P6. Training on ~10 views** (measured 2026-10-24, `photo_runs/NOTES.md`; the scores below are the mean PSNR of 3 held-out photos, set before the trainer is made, at the default 1/4 resolution, 360 × 480).
  * **Validation: none for photo sets** (`splitValidation` mode `'none'`, `sfm_panel.js` `valSplit`): the hashed 1-in-8 split held out 3 of cuisine's 12 photos. Measurements hold photos out by name instead (`photo_runs/tools/train.mjs`, `VAL=`).
  * **Unseen views do not train, whatever the settings.** cuisine, 9 training photos, 5000 steps: the training photos reach 25.6–32.7 dB, the held-out ones 12.9–13.6 dB in every run (14 runs; these held-out figures may be one checkpoint, 1000 steps, early: the tool read the page's last validation before it was done, fixed since), where a flat image of each photo's mean colour scores 13.1–13.4 dB. A held-out render has the scene's layout, smeared by floaters, although the held-out photo p02 was taken 4 s after p01 (about 40 dB). COLMAP's cameras score the same (13.2–13.6 dB), so the cameras are not the cause. No setting changed the held-out score past the run-to-run spread (about 0.5 dB): `max_splats` 30k, `opacity_reset` 0, both, `floater_depth` `'view'`, `lr_pos` / 2, points of 2 photos for the trainer (`trainMinObs` 2); and in a first series (10k steps, the held-out photos set after the trainer was made) `max_splats` 100k and `ds` 2. No sparse-view preset, then: none measured earns one. For scale, the gare video scores 16.4–16.6 dB on its 49 held-out views after 30k steps (DESIGN.md §5.3).
  * **More photos help a little:** salon, COLMAP's undistorted model, the same 3 photos held out (their flat colour: 13.8 dB on average), the others thinned (the points stay, including those of photos left out): 16 training photos 15.5 dB, 11 photos 15.0, 8 photos 14.0, 6 photos 14.1 (one run each).
  * Hence the shooting guide and the advice below, rather than training settings.
  * **The near-camera carve was the exception** (`minerals`, 40 photos: a room, then close-ups of a bookshelf; COLMAP 39 in one model, ours 38, centres 0.32% RMS, rotations 0.64° median). With 5 photos held out, our cameras fitted their training photos at 17.3 dB and COLMAP's at 21.8, the gap on the 11 close-ups (23.7–30.1 dB with COLMAP's cameras, 13.0–21.2 with ours). Not the points (COLMAP's cut to our 1716: 22.2 dB), not the lens model (COLMAP refitted with ours, RADIAL: 21.3), not the refinement (4 sweeps: 17.4). The floater removal's near-camera cone did it: 93,272 splats removed over the run (the other two rules: 4 and 0), because a photo taken next to the bookshelf observes mostly far points (its `zMin`, DESIGN.md §5.2) and its cone covers the shelf. Removal off: ours 25.6, COLMAP's 25.6 on the training photos, 14.2 / 14.8 held out (from 11.5 / 12.2). `floater_depth` `'view'` (the depth of every point in view) does as well while keeping the carve.
  * **Photo-set training defaults** (cuisine, salon, minerals; held-out PSNR at 10k steps, the page's own validation, `photo_runs/logs/photo_preset_from_page_log.txt`):

| | `'observed'`, points of 3+ / 2+ | `'view'`, 3+ / 2+ | carve off, 3+ / 2+ |
| :--- | :---: | :---: | :---: |
| cuisine | 13.01 / 13.58 | 13.37 / 13.52 | 13.12 / 13.55 |
| salon | 14.59 / 15.14 | 14.70 / 15.13 | 14.49 / 15.26 |
| minerals | 11.18 / 11.56 | 13.80 / 14.22 | 13.47 / 14.22 |

    Points of 2 photos for the trainer score higher in all 9 comparisons (+0.15 to +0.77 dB, single runs: the spread is about 0.5 dB); `'view'` equals the others on cuisine and salon and gains 2.7 dB on minerals. So for photo sets: `trainMinObs` 2 (its default 0 means 3 for a video, 2 for photos) and `floater_depth` `'view'` when the trainer gets the cameras (`sfm_panel.js` `send`, unless the URL sets it).
* **P6.1 Shooting guide** (`index.html`, the Project phase, cloned into the Video phase): walk rather than pivot; from one photo to the next turn at most 30° and move at most half the distance to the subject (12 photos per full turn around an object); every part of the scene in 3 photos or more (salon: 16 training photos 15.5 dB held out, 8 photos 14.0); lenses, zoom and orientations can mix (P8: a camera model each; one with fewer than 5 registered photos keeps its EXIF focal ratio). The limits come from every pair COLMAP registers in cuisine and salon, on COLMAP's poses (`photo_runs/tools/pairstats.py`): of the 32 pairs within both limits, 27 have 50 F inliers or more (ours); of the 126 pairs turning more than 60°, 7.
* **P6.2 Advice after mapping** (`coverage.js`, `tests/coverage_test.js`): the pieces outside the main segment and their strongest pair into it, the steps between consecutive photos (capture order) with fewer than 50 matches that pass a limit of P6.1 (the 3 worst in the Cameras phase, all in the log), and the share of the main segment's points only 2 photos see. In the Cameras phase above the controls, refreshed after every action. cuisine: *p07, p13, p14, p15 are a piece of their own (4 photos): the best link to the main piece is p13–p06 (58 matches)*; 79% of its points are seen by only 2 photos.

* **P7. Path order, shared with video.** The camera path is today the name order: the path line (`render.js`, "images are name-sorted"), the demo flight (`demo.js`, centres in capture (name) order, smoothed), the `block` validation split (`colmap.js` `splitValidation`); on the SfM side the quality signals against the previous keyframe, the index windows and `session.insert` (sorts by name). Names are also the images' identity (OPFS, the hashed validation split, `replaceImages`), so a path cannot be changed by renaming.
  1. **The path becomes data:** an ordered list of view names in the dataset and in `project.json`, name order by default. Video is unchanged (name order is time order); the path line, the demo flight, the `block` split and the contact sheet read the list instead of sorting by name.
  2. **Photos start in capture order** (EXIF time, then file name, §13.3 P1): the path actually walked, known before any pose. No synthetic poses in between: they would have no image to train or validate on, and the demo flight smooths the path already.
  3. **After SfM, a shortest path is offered:** an open path visiting every camera once (not a spanning tree, which branches), cost = centre distance + a rotation term, exact by dynamic programming over subsets at this size (~10–30 views). It is proposed only when clearly shorter than capture order, with a capture / shortest switch; names stay. Measured on COLMAP's poses (`cuisine` 12 cameras, `salon` 19):

| | capture order | shortest (distance only) | ratio |
| :--- | :---: | :---: | :---: |
| cuisine | 41.2 | 18.9 | 2.18x (the photos go back and forth) |
| salon | 28.2 | 24.0 | 1.18x (mostly capture order, a few stretches reversed) |

  On distance alone, single steps of the shortest path turn by 92° (cuisine) and 117° (salon), hence the rotation term; its weight is chosen on these sets.
  4. **SfM uses covisibility, not the path:** matching windows, BA windows and the quality signals' "previous keyframe" become the most covisible registered views, for video as for photos. For a video these are nearly its time neighbours; for n ≤ `window` the index windows already span every view, so photo sets do not wait for this.
  5. To check on both sets once photos load in the page: whether the smoothed demo flight cuts through furniture indoors, where consecutive photos are up to a sixth of the whole path apart (cuisine, capture order: 6.8 of 41.2).
  6. **As built** (`src/path.js`, `tests/path_test.js`): `colmap.path`, an order of the name-sorted images, is what the path line (`render.js` `pathOrder`) and the demo flight (`app.js` `_pathImages`) follow; the setting `path_order` (*capture* | *shortest*, View ▸ Camera, kept in `project.json` with the other view settings) recomputes it when the images change, so the order is stored as the setting, not as a list of names (the shortest path is a deterministic function of the poses). The `block` validation split and the contact sheet stay in name (capture) order: the split is made at load, and the sheet holds photos that have no pose. Item 4 is not done.
     * **The search:** a step costs the distance between the centres plus `rotWeight` × the turn between the viewing directions (radians) × the median nearest-neighbour distance. Exact (Held-Karp, either end free) up to 16 cameras, which is the size it is exact for, not ~30 as item 3 said (30 × 2³⁰ states); beyond, the best 2-opt optimum from a nearest-neighbour start at every camera (on 16 random views within 5% of the exact cost, `tests/path_test.js`), and up to 100 cameras in the page (`PATH_MAX`).
     * **The weight, 2** (COLMAP's poses, `photo_runs/logs/pathweights.log`; capture order / shortest length, largest turn of a step capture → shortest): cuisine 2.11×, 75° → 53°; salon 1.08×, 78° → 78°; minerals 1.45×, 130° → 56°; out_scene1 1.57×, 102° → 42°. Distance alone (0) leaves turns of 92° (cuisine) and 117° (salon); at 2 no step turns more than capture order's largest on these four.
     * **The offer:** for a photo set, when the shortest path is at least 1.25× shorter, the log says so once per model (cuisine on our cameras: *2.1x shorter, largest turn 52° vs 75°*; salon, 1.08×, is not offered). `e2e/photos.e2e.js` checks the offer and both orders. Item 5 (furniture) is not checked.

* **P8. Mixed lenses and orientations.** Measured on the mixed sets above (§13.2: the outdoor scenes hold together only through pairs of different lenses):
  1. **Orientation:** rotate portraits (or the minority orientation) by 90° at ingestion, so every photo of a lens has one size and one camera. The trainer sees the rotated photos (its views are rolled by 90°; nothing else changes).
  2. **One camera per lens and zoom**, keyed by EXIF (`FocalLength`, `DigitalZoomRatio`), with the focal prior from EXIF ($f_{35}$, times the zoom ratio). The trainer already takes intrinsics per image (`trainer.js` `_intrinsics(img)` reads `img.camera`) at one training size, which rotation gives; the SfM side has one shared `camera` (mapper, BA, export, undistortion, session state) and needs a camera per image index. A digital zoom could share the main lens's distortion with $f \times$ zoom, rather than a free camera fitted to a handful of photos.
  3. A camera with too few photos keeps its EXIF prior (refinement off) rather than fitting $f$, $k_1$ to 2–3 views.
  4. **Labels, kept separately:** lens (EXIF `FocalLength`, and `LensModel` when present), zoom (`DigitalZoomRatio`, 1 when 0 or missing) and orientation as shot (before the rotation of item 1), per photo, in the session and in `project.json`. They are data about the photo, not about the camera model: two photos with the same lens and zoom and different orientations share a camera and keep different labels.
  5. **Filters (*only use … photos*):** one button per label value present (*only ultrawide*, *only main 1x*, *only 2x*, *only landscape*, *only portrait*), combinable, plus *all photos*. A filter sets the **active photos**: SfM S2–S4 re-run on them from the cached features (S1 is not repeated), the trainer gets only them, the contact sheet greys out the others. The inactive photos keep their features and labels, so switching back costs a re-run of S2–S4 only. Changing the filter is an operation-log entry (§3), undone like a repair. Why: a mixed set can be solved best by all its photos (`out_scene1`: the lenses link the scene) or by one coherent part (`out_scene1`'s 7 ultrawide photos, 7 / 7 in one segment), and which one is better for training is measured, not assumed.

* **P8 as built.**
  * `photos.js` `planPhotos`: every group (device, lens, zoom) of the one common size once rotated is used, a camera model each, the largest first (camera 0); photos of the minority orientation of the whole set are rotated, whatever the filter (a filter change re-extracts no feature). `filter(labels)` keeps some photos (P8.5).
  * The mapper takes `cameras` and each image's `camera` index: PnP, `relativePose` (`camera2`), triangulation, reprojection, the window BAs (`ba.js`: an intrinsics block per camera model, `refineCameras`), snapshots and saved sessions per camera. Camera 0's focal is self-calibrated from its own pairs; the others start from it times their EXIF ratio (f35 × zoom), and a camera's intrinsics are refined once it has `minCameraFrames` (5, not measured) registered frames. With one camera every result is bit-identical to before (cuisine: focal, lens and poses).
  * `export.js`: a PINHOLE camera per lens in `cameras.bin`, each image on its own; the worker undistorts each photo with its lens's camera. The trainer already took a camera per image.
  * **Filters (P8.5):** the Video phase of a photo set has *Use only…*: a toggle per lens, zoom and orientation the set has, and *All photos*. A change re-plans the set (`photoFilter`); Run maps the photos left from the cached features. Undone like a repair: the worker keeps the filter and the map before each change, and Undo, once the current map has no action left to undo, restores both (the map's cameras go to the trainer again with *Train on these cameras*); `e2e/photos_lenses.e2e.js` undoes "only the ultrawide" back to the 13 cameras of every lens.
  * `out_scene1`, every lens (`e2e/photos_lenses.e2e.js`, `photo_runs/logs/cmp__out_scene1_p8_all.log`): 13 photos in the main piece, COLMAP's 13 (with a camera per lens and zoom), centres 0.99% RMS of the extent, viewing axes 0.74° median (above the colmap threshold of 0.5° on rotations; arbitrated by training, 3 photos held out, 10k steps, the photo defaults for both: our cameras fit their training photos at 28.2 dB, COLMAP's at 26.8, better on 6 of 10, all of the ultrawide by +0.8 to +6.3 dB, COLMAP's better on the main lens by 0.2 to 1.5; held out 13.3 / 13.4 dB; one run each, `photo_runs/logs/train__out_scene1_arb_*`); focals main 1382 / COLMAP 1362, ultrawide 757 / 753, 2x 2794 / 2690 (3 photos: kept at the EXIF ratio). Only the ultrawide: 7 / 7, focal 756, 0.36% and 0.23°. Before P8: 2 photos with every lens in one camera, 0 / 13 with the main lens alone. `out_scene2` and `3`: no pair starts a map; `out_scene4`: 6 placed, a largest piece of 2 photos without points. All three end without error, on the advice (*no two photos could start a map: the strongest pair has … matches*), and nothing is handed to the trainer (it would wait on 0 splats).
  * A synthetic test (`tests/mapper_test.js`: one photo in three on a lens of f 700): a camera per lens registers all 30 photos within the truth thresholds, focals within 1%; one camera for both does not. From a self-calibration 9% high (strong distortion, 16 photos of camera 0), two lenses end 5% high where one lens recovers (`tests/export_test.js`, `photo_runs/NOTES.md`).
  * **Two mixed sets shot for it** (`outside-mixed`: 31 photos, 15 main 1x + 15 ultrawide taken in pairs at each spot, 1 at 2x; `inside-mixed`: 27, 13 at 2x + 12 ultrawide + 2 main 1x; all portrait; `photo_runs/logs/colmap_lenses.log`, `cmp__*`). Focals at 1920 px, ours with every lens / COLMAP with a camera per lens (its model of every lens; each lens alone in brackets):

| | camera 0 start (source) | camera 0 final | ultrawide | registered, main piece (COLMAP) |
| :--- | :---: | :---: | :---: | :---: |
| outside | 1331 (EXIF: F unobservable, 4 pairs) | 1379 / 1403 (1436), −1.7% | 765 / 761 (754), +0.5% | 25, 21 (31) |
| inside | 2663 (EXIF: 2 pairs) | 2699 / 2750, −1.9% | 754 / 759 (757), −0.7% | 22, 22 (23) |

    The synthetic case does not occur here: camera 0's start is its EXIF focal (3–5% low; the pairs of one lens are too few for F), and the BA ends within 2% of COLMAP's, which is about COLMAP's own spread (main lens 1403 with every lens, 1436 alone). Each lens alone: ultrawide 754 / 754 and 755 / 757; main 1x 1464 / 1436 (8 and 9 of 15). **The EXIF ratio of the ultrawide is 7% off** (14 / 24 = 0.583; COLMAP 0.542 and 0.552), the 2x one exact (2.00): a lens with fewer than `minCameraFrames` photos keeps that error. `outside-mixed` registered 21 photos in its main piece where COLMAP has all 31 (`photo_runs/tools/run.js why=`):
    * frame 22 was registered by the regrow and lost when that regrow was undone (the smaller segment did not come back whole): `regrowSegments` now ends with the largest segment trying the frames still unregistered (21 → 22; the other six sets unchanged; *Add photos* on cuisine 11 → 12).
    * the rest have 0–7 2D-3D correspondences and pairs of 20–75 matches into the main piece, where COLMAP's (about 15,000 features at full size) have 71–184: features. At 16,384 features `outside-mixed` registers 31 / 31 in one segment (0.26%, 0.39° against COLMAP), but the count is no clear default: minerals 38 → 36, salon 17 → 17, out_scene1 13 → 14, inside-mixed 22 → 23, cuisine's near-planar photo 0 placed 19.7° off again (`photo_runs/logs/features_matrix.log`). Left at 4096.
  * **A lens with a few photos** (subsets of `outside-mixed` and of `inside2-mixed`, all main-lens photos and 2–4 ultrawide ones): kept at its EXIF ratio, the ultrawide's ratio to the main camera is 3.0–7.8% off COLMAP's; refined from 2 photos while mapping, it lands closer (−0.6 to +2.8%) but the full sets suffer (`outside-mixed`: rotations 0.60 → 2.95° median against COLMAP): a lens refined from its second photo, before the map is there. So a lens is refined while mapping from `minCameraFrames` (5) registered photos, and in the final sweeps from `minCameraFramesSweeps` (2): the subsets' ratio errors +0.2 to +3.6% (were +3.0 to +7.8%), the full sets unchanged or better (`inside-mixed`: its 2 main 1x photos' focal −3.3 → −0.8%, the 2x −1.9 → −0.7%, centres 0.43 → 0.23%). `tests/mapper_test.js`: a 3-photo lens 7% off ends within 2% (6.4% with the old rule).
  * **`inside2-mixed`** (45 photos: 19 main 1x, 20 ultrawide, 6 main 1x, a lens change mid-walk): 41 in one segment (COLMAP 39), centres 0.11%, rotations 0.24°; main 1385 / 1380, ultrawide 758 / 760 from an EXIF start of 821.
* **Speed** (minerals, 40 photos, headless Chrome, M5 Pro): matching every pair took 9.2 s, 2.1 s of it on the GPU and 7.1 s verifying pairs on one CPU core (RANSAC F runs its full 2,000 iterations on the pairs that fail). Each pair is now verified on a pool of workers (`frontend.js`, `verify_worker.js`, up to 8: `hardwareConcurrency` − 1) while the GPU matches the next batches: 2.7 s. A pair's RANSAC draws from a seed of its own (the call's seed, mixed with the pair's frame indices), so the worker count and the finishing order do not change the model (`tests/frontend_test.js`, which fails when the workers draw one sequence each). The contact sheet's thumbnails decode 4 photos at a time: 1.8 → 0.9 s (decoding at a reduced size alone: no change). SIFT itself is 1.0 s on the GPU (25 ms a photo); the photo decode is 1.2 s.
  * The new draws move the models as much as another seed does (`photo_runs/logs/ab_seeds.log`, `ab_cuisine.log`, before / after on seeds 1–4, cuisine 1–12): minerals' main piece 38/33/34/34 before, 34/34/39/34 after; cuisine's 12 on 9 of 12 seeds before, 8 of 12 after (else 8 photos, focal ~9% high). The slow photo tests now run 3 seeds and pass on 2. cuisine at browser size is left out as a known failure: its first photo (a near-planar wall, ~100 2D-3D) registers 20–29° off on most seeds, before (4 right of 12) and after (2 of 11); seed 1 was one of the right ones.

### 13.4 Tests and Acceptance
* **Photo sets:** `cuisine` (18), `salon` (23) and `minerals` (40, a room then close-ups of a bookshelf; COLMAP 39), §13.2, `outside-mixed` (31) and `inside-mixed` (27), two lenses at each spot (P8), and the mixed outdoor sets `out_scene1`–`4` (in `brush/out_scene*/images`, per-group folders and the COLMAP runs beside them), in `brush/cuisine/images` and `brush/salon/images` (originals; COLMAP reference in `colmap/sparse/0`), copies on the VPS in `test_videos/`. Still wanted: an object turned around (convergent, all pairs overlap) and a facade or street (a chain, like the gare subset). COLMAP `exhaustive_matcher` + `mapper` on the same photos is the reference (§8: not ground truth, defects excluded as in R1).
* **Synthetic** (`tests/`, fast): the 10-view arcs of §13.2 as a regression test (passing today), plus a variant with the matches of wide pairs thinned, to stand for SIFT's loss with the viewpoint.
* **Stress:** the gare every-4th chain (10 / 10 registered in one segment is the target of P3).
* **End to end** (`e2e/`): a photo drop → cameras → a few hundred training steps, no errors; one repair + undo.
* **Accepted when:** one segment holding at least the photos COLMAP registers (cuisine 12, salon 19), and every photo where COLMAP's own model shows no defect; rotations and centres within the `colmap` thresholds of §8.3 on the consistent pieces; focal within 2% of COLMAP's.
* **Mixed sets accepted when:** `out_scene1` with all its photos gives one segment holding at least COLMAP's 13 (a camera per lens and zoom), each camera's focal within 2% of COLMAP's where COLMAP has ≥ 7 photos of it; *only ultrawide* on `out_scene1` stays 7 / 7 in one segment (as today); `out_scene2`–`4` end without error and report their pieces. End to end: one filter change, training on the active photos only, undo.

### 13.5 Order
P1 + P2 (photos reach the existing pipeline, with the EXIF focal and the labels of P8.4) → P3 on cuisine, salon and `out_scene1`'s ultrawide photos, where the target is one segment instead of 4–5 (§13.2) → P8.1–3 (a camera per lens and zoom; target: `out_scene1` with every lens) and P8.5 (filters) → P4 / P5 / P7 (path order) → P6.
