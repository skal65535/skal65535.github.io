# Triangle Preview Encoder — implementation plan

In-browser encoder for WP2's triangle-preview format. Produces base64
strings byte-compatible with the existing decoder at
[`triangle/index.html`](../triangle/index.html). Mirrors the libwebp2
algorithm: shake-and-mutate optimization, AYCoCg19b palette, ANS
coding, and a Gouraud-shaded rasterizer offloaded to WebGPU compute.

C++ reference: `libwebp2/src/{enc,common,dec}/preview/`.

## Goals / non-goals

- **Goal**: take an `HTMLImageElement` / `Blob` / `ImageData`, produce
  a base64 string identical in *bitstream layout* to `mk_preview`
  output (not necessarily bit-identical; float vs. fixed-point and RNG
  paths will diverge). The string must round-trip through the existing
  JS decoder unchanged.
- **Goal**: real-time-ish — a few seconds at default effort on a
  midrange laptop. GPU rasterizer is the leverage.
- **Goal**: share code with the decoder where possible (constants,
  AYCoCg↔RGB, Delaunay, ANS state). Refactor `triangle/index.html`
  into a reusable module before adding encoder code.
- **Non-goal**: CLI/Node build. Browser only.
- **Non-goal**: WebGL fallback. WebGPU is required. On absence, encoder
  UI is disabled; decoder still works.
- **Non-goal**: bit-exact match with C++ encoder output. The decoder
  is the contract.

## Format recap (what the encoder must emit)

ANS-coded stream, prefix `"preview"`. Sections in order
(`preview_enc.cc:370`):

1. `grid_w`  — Range[2, 256]
2. `grid_h`  — Range[2, 256]
3. `use_noise` — adaptive bit (`kPreviewNoiseProba/Total = 2/4`)
4. **Palette**: `num_colors` Range[2, 32]; `has_alpha` bit
   (`kPreviewOpaqueProba/Total = 3/4`); then per color, sorted by
   `(cg, co, y, a)`:
   - if `has_alpha`: residual alpha bit via `ANSBinSymbol(2,2)`
   - `y` residual via `PutAValue` (zero + 6 magnitude bits + sign),
     `stats_yco` context
   - `co` residual via `PutAValue`, same `stats_yco`
   - `cg` residual via `PutAValue`, `stats_cg`, **no sign** (sorted
     non-decreasing)
   - Starting prediction: `{a=1, y=31, co=31, cg=0}`. Each color
     predicts from the previous.
5. **Vertices**: `num_vertices` Range[min, max] where
   `min = max(num_colors-4, 0)`, `max = min(1024, grid_w*grid_h - 4)`.
   Four corner indices first (TL, TR, BL, BR): adaptive "match" bit
   (`ANSBinSymbol(2,2)`), then if mismatch `ReadRange(0, nb_colors-2)`
   skipping the predicted index. Then row-major scan over all non-corner
   cells: `position_match` bit with proba
   `kProbaMax - (vertices_left << 16) / cells_left`; if set, color
   index with the same predict-and-skip scheme.

Vertices live on the integer grid `(0,0)..(grid_w-1, grid_h-1)`.
No duplicate vertices.

## Refactor first: shared module

Split [`triangle/index.html`](../triangle/index.html) into:

```
triangle/
  index.html             page + UI (decode + encode)
  triangle-core.js       constants, ANSBinSymbol, ValueStats,
                         AYCoCg↔RGB, Vtx/Color, Preview (decoder),
                         Delaunay
  triangle-enc.js        Encoder, ANS writer, mutation kernels,
                         optimization driver
  triangle-gpu.js        WebGPU rasterizer + loss reduction
  triangle-shaders.wgsl  WGSL compute kernels
```

Decoder behaviour is unchanged. Verify with all `setPrecalc(0..6)`
samples before proceeding.

## Encoder pipeline

### A. Input prep

Accept `HTMLImageElement` / `HTMLCanvasElement` / `ImageData`. Convert
to premultiplied RGBA8. Downsample to grid-density target via canvas 2D
`drawImage` to a scratch canvas at `grid_w × grid_h`. Default: grid 64,
density 1.5. No need to replicate libwebp2's halving + blur chain
exactly.

### B. Initial vertex placement

Port `CollectVerticesFromColorDiffMaximization`
(`preview_analysis.cc:209`) — the default strategy, no Lloyd iterations,
reuses the rasterizer:

1. Start with 4 corners.
2. Rasterize current triangulation; compute per-pixel SAD to reference.
3. Pick the grid cell with the largest residual; add it as a vertex.
4. Repeat until target vertex count.

GPU-friendly: each candidate-add just re-runs the rasterizer.

### C. Initial palette

Port `CollectColors2` (`preview_color.cc:307`):

1. Per triangle: 2×2 Wiener fit of `(x, y) → (A, R, G, B)`. Closed
   form, ~16 muls per triangle.
2. Evaluate fit at each vertex; accumulate color sum + count per vertex.
3. Sort by count; take top `target_num_colors` colors as palette seed.
4. `ReduceColors` (`preview_color.cc:172`): iteratively merge
   least-used entry into its perceptually-closest survivor using
   weighted YCoCg distance
   `0.3*dy² + 0.1*dco² + 0.6*dcg² + alpha_pen`.
5. Per-vertex/corner assignment: `FindClosestColorIndex` on a 5-px
   canvas average around each vertex.

All small CPU work.

### D. Optimization loop

CPU driver (`PreviewData::Optimize`, `preview_opt.cc:303`); GPU handles
the scoring rasterizer. Per iteration:

1. Restore best state.
2. Apply `num_mutations_per_iteration` (default 1–4) mutations, each
   drawn independently from 7 kernels with C++ probabilities
   (50/20/25/25/20/1/3):
   `MoveVertex`, `AddVertex`, `RemoveVertex`, `MoveColorIndex`,
   `MoveColor`, `AddColor`, `RemoveColor`.
3. Remove duplicate vertices if add/move happened.
4. Every `optimize_color_indices_every_n_iterations` (=10): brute-force
   sweep all vertex palette indices — one GPU dispatch per call (§ E).
5. `GetScore = loss / num_pixels + λ * num_bytes`:
   - `num_bytes`: real ANS-encoded size.
   - `loss`: GPU rasterize + weighted SAD reduction (§ E).
6. Accept with annealing tolerance `tol = score_tol * (N - i) / N`.
7. Early stop after `N/6` iterations with no improvement.
8. After 70% of iterations, nudge `λ` ×0.9 / ÷0.9 toward
   `target_num_bytes` (default 200).

Two `PreviewData` copies (current + best). All mutations are O(1) or
O(vertices).

### E. WebGPU rasterizer (hot path)

Rasterizes the Delaunay triangulation into a canvas-sized ARGB buffer
with Gouraud shading, then computes weighted SAD against the reference.

**Two shader designs under consideration**:

- **Render pipeline** (vertex + fragment): triangle list, smooth color
  interpolation, render target → storage texture, then compute pass for
  the SAD reduction. Simpler, native interpolation.
- **Compute-only scanline rasterizer**: one workgroup per triangle,
  following `RecordTriangleSpan` (`preview_rasterizer.cc:277`). Needed
  for `OptimizeColorIndices` so many color-index variants can be scored
  in one dispatch without re-triangulating.

Plan: start with the render pipeline, prove correctness against C++
output on a fixed bitstream, then add the compute rasterizer for
`OptimizeColorIndices`.

Loss reduction: workgroup-local SAD → `atomic<u32>` per-workgroup sums
→ global reduction. Final 4-byte `mapAsync` readback, pipelined so the
CPU prepares the next mutation while the GPU is busy.

**Reference behaviors the shader must reproduce** (`preview_rasterizer.cc`):

- Grid→pixel: `mult = ceil((img << 17) / (grid - 1))`;
  `pixel = (grid_coord * mult) >> 17` (`:53,283`).
- Top-left fill rule; no AA, no sub-pixel.
- Gouraud interpolation with fixed-point gradient; channel order ARGB.
- Loss weights: `sad_A + 0.3*sad_R + 0.6*sad_G + 0.1*sad_B`.

The render-pipeline smooth-shaded result will score correlated enough
with the C++ reference for optimization to converge. Bit-exactness is
a decoder round-trip concern, not a loss-function concern.

### F. ANS encoder

The decoder (`triangle/index.html:263–321`) defines the wire format.
The encoder needs the inverse ops: `PutBit`, `PutRange`, `PutAValue`,
`PutRValue`. Reference: `libwebp2/src/utils/ans.cc`.

Strategy: implement a tANS-style writer matching the decoder's 16-bit
word size and `kProbaMax = 1 << 16`.

Validation — round-trip harness:

```js
function roundTrip(encode_fn, expected_fields) {
  const bytes = encode_fn();
  const p = new Preview();
  p.decode(new ANSDec(bytes));
  assert.deepEqual(p.fields, expected_fields);
}
```

Build confidence in order: header fields → palette → vertices. This is
the riskiest piece; bit-exact compatibility with the existing decoder is
mandatory. Tackle it in Phase 1 before any optimization work.

### G. Sort + emit

Before encoding:

- Palette: sort by `(cg, co, y, a)`; remap vertex + corner indices.
- Vertices: sort by `(y, x)`. Corners emitted separately first.
- Drop unused palette entries.

Then write the sections per § "Format recap" and return
`btoa(bytes)`.

## UI additions

Augment [`triangle/index.html`](../triangle/index.html):

- File input + drop zone for image upload.
- Sliders: grid size (32–128), target colors (2–32), target bytes
  (50–500), effort (1–9 → maps to iterations / mutations-per-iter).
- "Encode" button. Live preview re-renders via the existing decode path
  on every accepted mutation (or every N) so the user watches
  convergence.
- Output: base64 textarea (auto-populates the existing decode area),
  byte count, palette swatches, iter/s, GPU ms.

## Phase plan

| Phase | Scope | Done when |
| --- | --- | --- |
| 0 | Split `index.html` → `triangle-core.js` + thin wrapper. No logic change. | All `setPrecalc(0..6)` decode identically. |
| 1 | ANS encoder + header/palette/vertex emit. CPU-only score (naive JS Gouraud). | Encode hand-crafted `PreviewData`; decode; fields match. |
| 2 | Initial state: input resample → vertex placement → palette init. | Visual quality ballpark vs. `mk_preview` (informal). |
| 3 | WebGPU rasterizer (render pipeline) + f32-CAS SAD reduction. | Loss agrees with CPU raster ±1% on a fixed input. |
| 4 | Optimization driver: 7 mutation kernels + annealing + size penalty. | Bitstream size converges toward target; quality improves. |
| 5 | `OptimizeColorIndices` via compute-shader rasterizer (variants in one dispatch). | Iter time ≤ 2× single-raster despite scoring N variants. |
| 6 | UI: upload, sliders, live preview, perf overlay. | Drop an image → base64 in < 5 s at default effort. |

## Shared code map

| Concern | Decoder | Encoder | Location |
| --- | --- | --- | --- |
| Constants (limits, probas) | ✓ | ✓ | `triangle-core.js` |
| `ANSBinSymbol`, `ValueStats` | ✓ | ✓ | `triangle-core.js` |
| `YCoCg_to_RGB` + new `RGB_to_YCoCg` | ✓ | ✓ | `triangle-core.js` |
| `Vtx`, `Color`, `Preview` (data) | ✓ | ✓ | `triangle-core.js` |
| Delaunay (Bowyer–Watson) | ✓ | ✓ | `triangle-core.js` |
| ANS reader (`ReadRange`, `ReadAValue`, …) | ✓ | – | `triangle-core.js` |
| ANS writer (`PutRange`, `PutAValue`, …) | – | ✓ | `triangle-enc.js` |
| Mutation kernels, `GetScore`, opt driver | – | ✓ | `triangle-enc.js` |
| CPU rasterizer (reference) | – | ✓ | `triangle-enc.js` |
| GPU rasterizer + loss reduce | – | ✓ | `triangle-gpu.js` + `.wgsl` |

## Risks

- **ANS bit-exactness.** The JS decoder's state machine is the spec.
  Any protocol mismatch silently corrupts the stream. Mitigation:
  round-trip tests before any optimization work.
- **Delaunay robustness.** Float Bowyer–Watson works fine for decoding
  a few hundred fixed points; the encoder hammers it during mutations.
  Watch for cocircular edge cases. If needed, port the int64 `Circle`
  math from `preview.cc:33–61` — BigInt is acceptable at mutation
  granularity.
- **`use_noise_` is a TODO in the decoder** (`preview_dec.cc:195`).
  Emit `false`; revisit if/when the decoder honours it.
- **`GetScore` calls full ANS encode every iteration.** If profiling
  shows this dominates, switch to an arithmetic-coder *estimator* for
  accept/reject and only do the real encode at the end.
- **`OptimizeColorIndices` is O(vertices × colors) rasters** per call.
  Without GPU this is unusable at effort > 3. Phase 5 compute-shader
  rasterizer is non-optional.
