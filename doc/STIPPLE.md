# Stipple: WebGPU port — handoff

Weighted Linde-Buzo-Gray stippling, fully on WebGPU (compute + render).
The original CPU + WebGL2 implementation lives in `stipple/stipple.js`
and is exposed as an opt-in toggle for A/B comparison; everything else
on the page is WebGPU.

## Status

| Phase | What | State |
| --- | --- | --- |
| 1 | Scaffold; drop all WebGL; WebGPU render path | ✅ |
| 2 | JFA Voronoi (clear / seed / step + JFA+1) | ✅ |
| 3 | GPU accumulate + Lloyd update | ✅ |
| 4 | Split/merge readback + JS port | ✅ |
| 5 | Multi-resolution pyramid + JFA early-stop | **skipped** (GPU is ~1 ms / iter) |
| 6 | 1-iter-per-frame + video (continuous mode) | ✅ |
| – | CPU iterator adapter, 'Use CPU' toggle | ✅ |

**Visual quality** at N = 80 K: matches the CPU reference. **Perf**: ~1 ms
GPU per Lloyd iter at 1024² + N = 80 K.

## Files

```
stipple/
  index.html              page + UI + tick loop
  stipple-shaders.wgsl    all WGSL kernels (compute + render)
  stipple-gpu.js          initGPU + iterator + renderer + grabFrameGPU
                          + CPUStipplingIteratorAdapter
  stipple.js              original CPU iterator (unchanged; consumed by adapter)

doc/STIPPLE.md            this file
```

`navigator.gpu` is a hard requirement; on absence the page alerts and bails.

## Pipeline (one Lloyd iter)

1. **Source ingestion (`grabFrameGPU`)** — only when the source frame
   changes (first load; every tick during video playback):
   - `drawImage` source onto a 2D scratch canvas at work resolution
     (`copyExternalImageToTexture` doesn't scale).
   - `copyExternalImageToTexture` to `tex_src` (rgba8unorm).
   - `edge_detect` (or `luma`) compute → `tex_gray` (rgba8unorm,
     grayscale in `.r`). Same kernel reduces the grayscale sum into a
     single `atomic<u32>` via workgroup-local reduction + one global
     `atomicAdd` per workgroup.
   - 4-byte `mapAsync` of the sum, fire-and-forget. Exposed as
     `grays.average` (live getter on a module-level cached value).
2. **JFA Voronoi (`clear_jfa` → `seed_jfa` → step×N → step=1)**: result
   in `tex_jfa_a` (r32uint, 1024²). Step sequence: W/2, W/4, …, 1, **1**
   (trailing extra step=1 = "JFA+1" for boundary cleanup).
3. **`clear_cells`** zeros the per-cell accumulator buffer.
4. **`accumulate`**: one thread per pixel; reads `site_ids[p]` (from
   JFA texture), `pixels[p].r` (grayscale), atomic-adds f32 (via
   bitcast-u32 + CAS loop) into `cells[sid * 9 + slot]`. Slots:
   `[acc, r, rx, ry, rxx, rxy, ryy, x, y]`. The last five (full
   moments) are written only when `params.full_moments == 1` — set on
   split/merge frames.
5. **`lloyd_update`**: one thread per site; reads `cells`, writes new
   position to `sites_a` in place: `(rx / r, ry / r)`.
6. **Split/merge frames** (every `num_Lloyd_iters` iters, default 4):
   `copyBufferToBuffer(cells → cells_read)`, submit, `mapAsync`, run
   JS `_splitMerge` on the f32 array, upload new positions, update
   `_N`.

The renderer (`GPURenderer`) draws points (instanced quads at
`sites_a[i]`), the source target (textured quad blit), or false-color
Voronoi (samples `tex_jfa_a` and looks up `site_color[id]`). For Voronoi
render the iter exposes `refreshJFA()` which re-runs the JFA passes on
the current `sites_a` — needed because the iter's pre-Lloyd JFA may not
correspond to the post-split/merge ID space.

## Resources

| Name | Type | Size | Notes |
| --- | --- | --- | --- |
| `tex_src` | rgba8unorm tex | 1024×1024 | RGB source, scaled. |
| `tex_gray` | rgba8unorm tex | 1024×1024 | Edge-detected grayscale. Only `.r` used. `COPY_SRC` for the CPU-mode pixel readback. |
| `tex_jfa_a / tex_jfa_b` | r32uint tex | 1024×1024 | JFA ping-pong. Input bound as sampled `texture_2d<u32>`; output as `texture_storage_2d<r32uint, write>`. |
| `sites_a` | storage `vec2<f32>[]` | N_max | Point positions in [0,1]². In-place Lloyd update. |
| `cells` | storage `atomic<u32>[]` | N_max × 9 × 4 B | f32 accumulators via bitcast. |
| `cells_read` | MAP_READ buffer | same | Split/merge readback. |
| `sum_buf` / `sum_read` | storage + MAP_READ | 4 B each | Image-sum reduction + readback for `grays.average`. |
| `read_buf` (CPU mode) | MAP_READ | row-aligned × H | Lazily allocated pixel readback for the CPU iterator. |
| `buf_params` | uniform | 48 B | W, H, N, …, full_moments. Per-iter. |
| `buf_jfa_params[]` | uniform | 48 B × ~12 | One per JFA pass. All pre-written before submit. |

## Work resolution

**1024 × 1024.** At N = 80 K → ~13 pixels per cell, ~8% seed-collision
rate (vs 30% at 512²). The seed pass races (multiple threads
`textureStore` to the same texel = last writer wins); cells whose
site got stolen end up with `acc == 0` and are pruned by split/merge.

## Lessons (the journey, not the destination)

These were all dead ends or surprises that consumed real time. They
look obvious in hindsight; they weren't.

### Quantization bias on weighted reductions

Initial accumulator design: `atomic<i32>` with Q4 (×16) fixed-point on
coordinates. Worst-case sum at 512² fit i32 with 2× headroom; sub-pixel
centroid precision in the analysis. Shipped.

The artifact: at high N, points lined up on cell boundaries — clearly
visible after ~10 iters. Spent time investigating JFA correctness
(false suspicion), tried JFA+1, tried Q3 at 1024² — only partial fix.

Root cause: per-pixel rounding is `±0.5 Q-units in the direction of
the value`. For a cell with mostly background pixels punctuated by a
few bright edge pixels, the bright pixels' biases stack coherently in
the same direction. Centroid drifts toward the cell border where
edge pixels live; next iter the border moves with the centroid;
repeat. The "math is sub-pixel" precision analysis missed the
coherent-bias mode entirely.

Fix: **f32 atomics**. WebGPU has no native f32 atomic add; emulate via
`atomicCompareExchangeWeak` on u32 + bitcast:

```wgsl
fn add_f32(idx : u32, value : f32) {
  var old : u32 = atomicLoad(&cells[idx]);
  loop {
    let new_bits = bitcast<u32>(bitcast<f32>(old) + value);
    let r = atomicCompareExchangeWeak(&cells[idx], old, new_bits);
    if (r.exchanged) { break; }
    old = r.old_value;
  }
}
```

At ~13 pixels/cell average, CAS retries are negligible. Total iter
cost ~1 ms.

**Takeaway**: don't quantize accumulators that get summed under
non-symmetric weighting. Coherent rounding bias is invisible in
single-sample precision analysis but is real and accumulates.

### Dead 1MB readback in `grabFrameGPU`

Phase 1 returned `{ texture, pixels, average }`. After Phase 3 ported
Lloyd to GPU, only `texture` and `average` were consumed — but the
`pixels` path stayed in place: `copyTextureToBuffer` (1 MB),
`mapAsync` stall (~16 ms), and a 1 M-iter JS unpack loop (~10–30 ms).

This was the dominant per-frame cost for video, hidden behind the
"GPU is fast" assumption. Fix: drop it. Compute `average` on the GPU
during edge-detect via workgroup-local atomic reduction → one global
atomicAdd per workgroup. Readback shrinks from 1 MB to 4 bytes; JS
unpack vanishes. Reinstated as opt-in (`needPixels: true`) only for
the CPU-iterator path.

**Takeaway**: leftovers from earlier phases are silent perf killers.
Audit return shapes when phasing.

### `read`-mode storage textures are not universal

First JFA bound the read side of the ping-pong as
`texture_storage_2d<r32uint, read>`. That access mode is gated behind
the `readonly_and_readwrite_storage_textures` feature and silently
misbehaves on implementations that don't support it. Switch to
`texture_2d<u32>` (sampled binding) for input — universally supported,
same `textureLoad` API.

### One uniform buffer per pass

`queue.writeBuffer` in a loop, followed by one `queue.submit`, means
**all passes in the submitted command buffer see the final value** of
the uniform. The JFA passes were all running with the last
`jfa_step` value, silently. Fix: pre-allocate one uniform buffer per
JFA pass and pre-write each before encoding.

### `copyExternalImageToTexture` does not scale

It crops to copy size. We pre-draw onto a 2D scratch canvas at work
resolution, then copy. `GPUExternalTexture` would skip both for video
(see Deferred).

### Voronoi viz shader was upside-down

Easy bug, recorded for the next time someone touches the render
shaders. `target_render` uses `textureSampleLevel(uv)` which treats
(0,0) as top-left. The Voronoi fragment shader was applying an extra
`(1 - uv.y)` flip before `textureLoad`, inverting the result. Use
`uv.y` directly.

### `show_voronoi` was gated on `iter.done`

Continuous mode (Phase 6) made `iter.done` always false, so the
Voronoi render branch never executed. Fix: drop the guard; gate only
on `params.show_voronoi`.

Also after a split/merge frame, `tex_jfa_a` corresponds to *pre*-Lloyd
site positions and the IDs may have been reshuffled. Calling
`iter.refreshJFA()` before `renderVoronoi` re-runs JFA on the current
`sites_a` so the IDs match `_points`.

## Continuous mode (Phase 6)

The iterator runs forever:
- `iter.done` is intrinsically `false`.
- The tick loop applies an iter cap externally: for static images,
  `state.iter.ops < state.iter.maxOps` gates `requestAnimationFrame`
  scheduling. After convergence the rAF loop stops; source change /
  Reset / param change rebuilds state and unblocks.
- Hysteresis ramps from 0.01 to 0.61 over the first `max_ops` iters,
  then saturates. The ramp gives initial-convergence room; the cap
  bounds steady-state split/merge churn.

For **video**: the tick loop never gates on the cap; on each frame it
calls `grabFrameGPU` (which reuses the same `tex_gray` handle —
content updates in place) and continues iterating. Point positions
persist across video frames; the stippling tracks motion rather than
restarting state.

## CPU adapter

`CPUStipplingIteratorAdapter` wraps `stipple.js`'s `StipplingIterator`
behind the GPU iter's surface. The CPU does Voronoi + Lloyd +
split/merge in JS; the adapter uploads positions to a GPU buffer for
the WebGPU renderer to draw. `refreshJFA` runs the CPU `computeVoronoi`
and uploads the result to a r32uint texture — so `show_voronoi` works
in both modes.

Performance: CPU at 1024² + N = 80 K is 5–10× slower than GPU
(`computeVoronoi` dominates). Video CPU mode works but won't be
smooth. The pixel readback (`needPixels: true`) adds ~16 ms mapAsync +
~10 ms JS unpack per grab.

The toggle is a comparison tool, not a production fallback.

## Deferred / open

Ordered by likely visible impact:

- **`GPUExternalTexture` for video**:
  `device.importExternalTexture({source: videoElement})` binds video
  frames directly in compute shaders (`texture_external`); GPU sampler
  handles YUV→RGB + scaling. Skips the 2D-canvas `drawImage` +
  `copyExternalImageToTexture` round-trip. Add when video ingestion
  becomes the next visible bottleneck.
- **Path 2 convergence sensing**: GPU-side displacement metric
  (per-workgroup tree-reduce of `|new − old|²`), periodic mapAsync
  readback (one slot per workgroup, sum on CPU). Replaces the
  iter-cap heuristic with real convergence detection. Don't bother
  unless the cap turns out to be too coarse.
- **Split/merge pipelining**: kick off the next iter's GPU work while
  the cells `mapAsync` resolves in the background. Cuts the
  per-split-merge stall to ~0.
- **Hysteresis for video**: today the ramp is static (capped at 0.61
  after `max_ops` iters). For continuously-changing sources a schedule
  tied to a moving average of displacement might track better.
- **Multi-resolution pyramid (Phase 5)**: ~1 ms / iter × ~40 iters =
  ~40 ms across full convergence. Mipmaps would only help the
  `gPixels` texture (bilinear-averaged density field is the right
  downsample); JFA textures (categorical IDs) and per-site buffers
  don't benefit. Skipped because GPU is already fast enough.
- **GPU split/merge**: variable output length + PCA per cell + rarity.
  Cells readback (~1 frame stall every 4 iters) is the only cost the
  CPU port currently pays. Revisit if profile shows it dominates.

## Out of scope

- **WebGL2 fallback.** Hard WebGPU requirement.
- **Service worker.** No WebGPU access (network/cache layer).
- **Web worker + OffscreenCanvas.** Could move JS `_splitMerge` off
  the main thread, but main-thread CPU isn't the bottleneck. Not
  worth the refactor.
- **Temporal multi-resolution.** Speculative; revisit if needed.
- **Persistent compiled pipelines across canvas recreation.**
  Source-change path rebuilds; same pattern in the GPU iter.

## Key params (UI-exposed)

| Name | Default | Notes |
| --- | --- | --- |
| `num_points` | 80000 | Target N. Buffers allocated at 2× headroom. |
| `num_iters` | 10 | Convergence budget × `num_Lloyd_iters` = iter cap. |
| `num_Lloyd_iters` | 4 | Lloyd iters between split/merge. |
| `rho` | 0 | Density cutoff bias for split/merge thresholds. |
| `alpha`, `strength` | 0.01, 2.6 | Edge-detect params. |
| `use_edges` | true | Toggle edge_detect vs luma compute. |
| `radius` | 1.5 | Point sprite size in px. |
| `invert` | false | Render colors. |
| `use_cpu` | false | Switch to CPUStipplingIteratorAdapter for A/B. |
| `show_voronoi` | false | Render false-color cells instead of points. |
| `show_target` | false | Render the edge-detected source. |
