# Gauss-Opt — Design Archive

What `DESIGN.md` no longer carries: the closed experiments of its §9 in full (their status table stays there), and the specs of what was removed, to put it back from.

## Removed: simulated annealing (`opt_method = 2`, `sim_anneal`; DESIGN.md §5.4 until 2026-10-29)

It changed nothing in E6 (-0.08 to +0.13 dB against Adam per scene, within the repeat noise). The code (`src/anneal.js`, its branches in `trainer.js`, `octree.js` and `app.js`, `tests/optimizers_test.js`'s two tests) is at commit 194528d.

**At the rebalances (`anneal.js`):**
   * **Energy:** over the window since the last rebalance, a splat's mean footprint error $e_i = R_i / W_i$ (§5.1), over the visible splats' mean.
   * **Metropolis test:** a jump made at the last rebalance is judged on $\Delta e_i$ less the median $\Delta e$ of the visible splats that did not jump (the training lowers every $e$ meanwhile): kept if $\le 0$, else with probability $\exp(-\Delta e / T)$, else undone. Then `sa_share` (0.25) of the visible splats jump by $T \cdot$ `sa_size` $\cdot \mathbf{R}_i \mathbf{s}_i \boldsymbol{\xi}$. $T = T_0 \gamma^{t/50}$ (`sa_temp`, `sa_cooling`; `sa_reheat`: back to $T_0$ every $\max(1000, 10K)$ steps).
   * **Split acceptance:** splits with $R_i \in [T_l, T_u]$ are accepted with probability $0.35 \exp(-(T_u - R_i) / (T \bar{R}))$.
   * Before E6 the test ran in `adam_step` at every step, against the splat's error in the *previous step's* view: one view per step, so it compared views, not jumps.

## Closed experiments

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
* **Tools (off by default; removed 2026-10-26 with `grow_by` `'grad'`, none gained, below; the shape split's test is now `tests/split_shape_test.js`):** `split_min_err` (a splat splits only if its mean L1 error under its footprint $R/W$ is above this; the split threshold $T_u = 1.35\,\bar R$ is relative, so growth otherwise never stops before the cap), `grow_rate` (at most this share of new splats per rebalance, the highest $R$ first), `split_in_plane` (split / clone offsets lose their component along the splat's thinnest axis: the error offsets are taken in each view's image plane, `shaders.wgsl`, far from a floor seen at a grazing angle). `tests/growth_gates_test.js`; each test seen failing with its code reverted.
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
* **One difference at a time** (jardin5, 10k steps, `photo_runs/tools/e9_real.sh`; val dB, depth median / within 5%; the default 18.53, 3.3% / 67%, repeating 18.51): `ssim` 0.2 18.61, 2.5% / 71%; `lr_scale` 0.01 18.63, 2.7% / 69%; no stipple 18.63, 2.9% / 69%; `mc_scale_reg` 0 18.50, 3.2% / 66%; `mc_noise` 1e5 18.37, 3.1% / 67%; `mc_opacity_reg` 0 18.37, 3.7% / 65%; `rebalance_every_cap` 200 18.35, 3.3% / 65%; isotropic init 18.43, 3.5% / 65%; no size limits 18.55, 5.7% / 62%. New options: `split_mode` `'shape'` (brush's split, `octree.js` `splitByShape`) **18.86, 2.1% / 75%**; `grow_by` `'grad'` (removed 2026-10-26; brush's score: `tile_backward` sums $|\partial L/\partial u_0| + |\partial L/\partial v_0|$ per pixel into grads slot 11, `adam_step` adds it with a view count into `grow_acc`; a splat grows when the mean per view is over `grow_grad`): 2e-5 17.66, 3.2% / 65% (254k splats); with the shape split 18.37 (2e-5), 17.55 (5e-5, 131k splats). Tests: `tests/growth_gates_test.js`, each seen failing with its code reverted.
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
  - **Not the octree's tilt** (2026-10-26): the dodecahedron's up (`scene.json`) is [-0.052, -0.657, -0.752], 49° from the octree's Y axis, so the floor cuts the leaves at a slant. The leaves decide only the floater rules 2 and 3 (`octree.js` `isFloater`, the leaf states from `updateCameraAndNeighborStats`) and which pairs `mergeOverlaps` tries; the render is the exact sort (E1). The COLMAP model rotated so that up is +Y (`R' = R R_w^T`, points `R_w p`; the images copied, as `serve.py` refuses symlinks out of the root), 5k steps, defaults, two runs each: tilted 22.39 / 22.36 dB, 12% / 13% under, 7% / 6% above; level 22.40 / 22.39 dB, 10% / 9% under, 9% / 9% above. The same within the runs' spread.
  - **The opacity learning rate:** ours was 0.05, brush's 0.01. 0.02: 22.12 dB, 21% under; **0.01: 22.44 / 22.40 dB, 0.6% / 3.3% / 11%**; 0.005: 22.20, 0.6% / 2.4% / 9%. At 10k steps: 0.05 22.04 dB, 24% under; 0.01 22.39, 15%; 0.005 22.39, 10%. By eye (the gallery's viewer) the floor's streaks and wavy grout lines go away. With 0.005, `ssim` 0.2, `lr_scale` 0.01, Adam, no stipple or `grow_by` grad add nothing (21.62-22.28 dB).
  - **On the videos** (10k; val dB, depth median / within 5%): jardin5 18.86, 2.1% / 75% → 18.75, 1.8% / 77% (0.01), 18.56, 2.0% / 75% (0.005); jardin7 24.84, 8.7% / 53% → 25.40, 10.3% / 54% / 24.96, 10.6% / 54%; gare 15.57, 21.5% / 28% → 15.56, 17.9% / 33% / 15.53, 17.5% / 34%. 0.01 the default (+0.8 dB summed over the four scenes, 0.005 +0.1).
  - **On the photo sets** the training views lose 0.6-1.4 dB (cuisine 30.80 → 29.39, salon 28.99 → 27.95, minerals 26.05 → 25.45), but held out (every 4th registered photo, `VAL`; 0.05 / 0.02 / 0.01): cuisine 13.41 / 13.15 / **13.49**, salon 16.49 / 17.00 / **17.37**, minerals 15.15 / **15.35** / 15.24: less overfit, not worse. Training is slower at 0.01: 145-191 s for the gallery's 10k-step runs against 101-157 s at 0.05.
* **Open:** jardin5 1.8% against brush's 1.0%, gare ~18% against 16%, the dodecahedron 22.4 dB at 300k splats against brush's 22.86 at 37k (crisper edges on the dodecahedron itself); semi-transparent splats still under the floor (opacity >= 0.2: 16% at 0.01). Not tried: the opacity L1 near zero with the noise kept, brush's position rate schedule (`lr_mean` 2e-5 → `lr_mean_end` 1e-6).

### E11 — Brush's defaults where they win (2026-10-27)
* **Sweep** (`photo_runs/tools/e11_sweep.sh`, `logs/e11.txt`; 10k steps, 300k splats, ds 2, one change at a time from the E10 defaults): held-out PSNR on jardin5 / gare / jardin7 / the dodecahedron, training views on cuisine (no held-out photos):

  | | jardin5 | gare | jardin7 | dodeca | cuisine |
  |---|---|---|---|---|---|
  | E10 defaults | 18.74 | 15.57 | 25.37 | 22.37 | 29.79 |
  | shape limits off | 19.15 | 15.78 | 25.92 | 22.40 | 31.38 |
  | + color clamped only below 0 (brush) | 19.19 | 15.84 | 25.84 | 22.38 | 31.72 |
  | + D-SSIM 0.2, `lr_scale` 0.01 | 19.16 | 15.50 | 25.54 | 22.53 | 31.47 |
  | + brush's `lr_mean`: 2e-5 → 1e-6 over the run | 19.12 | 15.81 | 25.76 | 22.53 | 30.91 |

  Depth at the SfM points (median rel. error / within 5%), limits off → + brush's `lr_mean`: jardin5 2.1% / 76% → 1.5% / 81%, gare 21.9% / 29% → 19.4% / 35%, jardin7 18.5% / 51% → 15.4% / 55%; the dodecahedron's opaque splats off the floor plane, p90 11.2% → 8.8% (6.3% with the limits on).
* **Taken:** the shape limits off (`max_splat_angle`, `max_splat_scale`, `max_anisotropy` 0: none; on, they shrank the long thin splats of grass and branches) and brush's position rate (`lr_pos` 2e-5 decaying to 1e-6 over `train_steps`, or 10k steps open-ended; brush v0.3.0 `config.rs`, `train.rs`: over `total_steps`, x its bounds' median side where ours is x the scene radius). **Not taken:** the clamp below 0 only (within noise held out), D-SSIM (loses on gare and jardin7).
* **Gallery's `-latest` scenes** retrained with them (`logs/gallery_latest.txt` `e11_new`, against `opacity`): jardin5 18.73 → 19.14 dB held out, gare 15.55 → 15.83, jardin7 25.38 → 25.49 (25.76 in the sweep: one run each, so about 0.3 dB of noise there); training views cuisine 29.39 → 30.92, salon 27.95 → 29.37, minerals 25.45 → 26.75.
* **jardin5 at 1M splats, 30k steps** (brush's run of the gallery's jardin5-brush-1m: 20.01 dB held out at SH 0, depth 1.1% / 86%): the limits off alone 20.51 dB, 1.7% / 79%; with brush's `lr_mean` too (`train_steps=30000`) 20.57 dB, 1.4% / 82%, 1038 s.
* **Still open:** the depth gap to brush (1.4% against 1.1%), the dodecahedron's floor (p90 8.8% against 6.3% with the limits on), and SH 3 (E10 step 4). Brush's gradient growth was tried in E9 (`grow_by=grad`, -0.9 dB on jardin5) and is not retried here.
