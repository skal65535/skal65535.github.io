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
* Intrinsics are freed in this stage: each window BA refines $f, k_1$ in turn, starting from the previous window's, so the result is the last window's estimate, not a joint fit over all frames.
* Fallback: one global GPU BA (D2) if sweeps fall short of COLMAP (§8).
* **Final retriangulation and filtering** (COLMAP spends 125 s on this on jardin5, and the trainer initialises splats from its output). Mapping side (`mapper.extendTracks`, formerly `retriangulate`): with the final poses, extend tracks and triangulate new ones over all usable pairs, then drop observations that reproject > 4 px; existing points keep their BA positions. Trainer side (`export.js` `datasetFromMapper`): keep observations ≤ 2 px, then points with ≥ 3 of them and two camera centres seeing the point ≥ 1.5° apart; no re-triangulation, no track split. Colours come from the undistorted images (S5). On jardin5 the trainer filter keeps 121,831 of 202,339 points (≥ 3 observations alone: 139,001; p90 point error 2.01 → 1.34 px). The point count and spatial spread feed the trainer's `sceneRadius` and initial splat scales, so this stage is checked against COLMAP's points3D too (§8).

### 5.6 S5 — Hand-Over
* Undistort (§4), build the dataset, close the panel or keep it open, start training. Every 8th view stays held out for validation, as today.

### 5.7 Platform Constraints
* **Codecs:** the test videos are H.264 High 1080p (jardin5/6 at 30 fps, re-exported through Core Media without metadata; jardin4 is a DJI NEO original at 59.94 fps with audio). DJI originals are often HEVC, sometimes 10-bit: WebCodecs HEVC decoding depends on the platform's hardware decoder. Check `VideoDecoder.isConfigSupported()` first and tell the user to transcode (or use the COLMAP path) when it fails.
* **Demuxing:** frames come out in decode order (B-frames); select by presentation timestamp. Honour the rotation matrix of the track header, and use timestamps rather than frame indices (variable frame rate).
* **Keyframe storage:** JPEG at quality ≥ 95, like `video2splat.py` (`-q:v 2`); these are the trainer's ground-truth images, so a lower quality costs PSNR.
* **One GPU, two devices:** the SfM worker and the trainer each own a `GPUDevice` and cannot share buffers. Running both at once competes for memory (up to 2M splats plus SfM descriptors) and time. Policy: SfM stages pause training and resume it when done; images and the dataset cross over as blobs.
* **Determinism:** every RANSAC and k-means takes a seeded RNG; GPU kernels avoid float atomics where results feed decisions (matching, verification), so a stage re-run on the same input gives the same model. Tests depend on this (§8).
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
* **"Open video…"** in the Project phase, with a file picker.
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
├── e2e/harness.js         # What the e2e tests share: clip, serve.py, browser and its errors
├── e2e/safari.js          # The page calls the e2e tests use, on Safari through safaridriver (W3C WebDriver)
├── src/ui/sfm_panel.js    # The "Video → Cameras" panel (§7), loaded on first use
├── src/sfm/
│   ├── worker.js          # The session in a worker: S0..S5, stage results cached
│   ├── defaults.js        # Session / pipeline option defaults (worker and panel, URL sfm_<key>)
│   ├── frames_worker.js   # S0 analysis of a chunk of frames (pool of workers)
│   ├── mp4.js             # MP4 / MOV demuxer: sample table, WebCodecs configuration
│   ├── video.js           # WebCodecs decoding, frames drawn upright at a given size
│   ├── frames.js          # S0: sharpness, pyramidal LK motion, keyframe selection (video2splat.py `select`)
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
