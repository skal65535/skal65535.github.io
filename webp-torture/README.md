# WebP torture bitstreams

Small WebP files that exercise corners of the format a normal encoder never
emits, one layer of it at a time:

* **62 lossless (VP8L) streams**, written bit by bit by
  [`vp8l.py`](vp8l.py).
* **81 lossy VP8 frames**. All but seven are assembled by
  [`vp8_asm.py`](vp8_asm.py) from a text description of the bitstream, one
  file per case in [`cases/`](cases), under the field names
  [RFC 6386](https://www.rfc-editor.org/rfc/rfc6386.html) gives them. The
  other seven start from an encoder-API call
  ([`make_partition_sources.c`](make_partition_sources.c)) and are patched
  by [`lossy_parts.py`](lossy_parts.py).
* **19 RIFF containers**, wrapped by
  [`webp_asm.py`](webp_asm.py) in
  [RFC 9649](https://www.rfc-editor.org/rfc/rfc9649.html)'s names: the
  extended-format VP8X chunk, the optional chunks a decoder must step over,
  and sizes that lie about what is behind them.
* **18 alpha chunks**, where the plane is either stored one byte per
  pixel or compressed with the lossless coder in its 8-bit mode -- a
  different path through the decoder from the one every VP8L file here
  takes.

A case is a text file and the notes below are its own: each one carries what
it is, what the decoder should do with it, and which path that answer comes
from.

Each entry says what the reference decoder is expected to do:

* **ok** -- must decode, and must keep decoding to the same pixels. Several
  are not something cwebp can produce, so nothing else pins the behaviour.
* **reject** -- must fail cleanly and report a status, with no crash and no
  out-of-bounds access. Which status varies: a malformed Huffman code gives
  BITSTREAM_ERROR, a short partition table gives NOT_ENOUGH_DATA.

## The bitstreams

Every name below links straight to the file. The whole set lives
in **[`files/`](files/)**, which lists each one with its size and
expected verdict; the notes further down say what each targets.

| Group | Files | must decode | must be rejected |
| --- | ---: | ---: | ---: |
| Simple codes | 9 | 6 | 3 |
| The code-length code | 15 | 9 | 6 |
| Meta Huffman / entropy image | 5 | 5 | 0 |
| Color cache | 5 | 3 | 2 |
| Palette packing | 6 | 6 | 0 |
| Transforms | 3 | 2 | 1 |
| Back-references | 8 | 6 | 2 |
| Predictor modes | 7 | 7 | 0 |
| Frame header | 4 | 2 | 2 |
| The RIFF container | 19 | 9 | 10 |
| The alpha chunk | 18 | 10 | 8 |
| Lossy: frame tag and picture header | 16 | 8 | 8 |
| Lossy: segmentation | 7 | 7 | 0 |
| Lossy: loop filter | 6 | 6 | 0 |
| Lossy: quantizer | 6 | 6 | 0 |
| Lossy: coefficient probabilities | 5 | 5 | 0 |
| Lossy: prediction modes | 5 | 5 | 0 |
| Lossy: coefficients | 16 | 16 | 0 |
| Lossy: skipped macroblocks | 3 | 3 | 0 |
| Lossy: token partitions | 5 | 2 | 3 |
| Lossy: truncation | 4 | 1 | 3 |
| Lossy: partition sizes, from real encodes | 8 | 5 | 3 |
| **total** | **180** | **129** | **51** |

**Simple codes** —
 [simple-dist-2sym-first-oob](files/simple-dist-2sym-first-oob.webp) ·
 [simple-dist-2sym-second-oob](files/simple-dist-2sym-second-oob.webp) ·
 [simple-dist-2sym-both-oob](files/simple-dist-2sym-both-oob.webp) ·
 [simple-dist-1sym-oob](files/simple-dist-1sym-oob.webp) ·
 [simple-dist-sym-39-last-valid](files/simple-dist-sym-39-last-valid.webp) ·
 [simple-dist-sym-40-first-oob](files/simple-dist-sym-40-first-oob.webp) ·
 [simple-green-1bit-symbol](files/simple-green-1bit-symbol.webp) ·
 [simple-dist-2sym-duplicate](files/simple-dist-2sym-duplicate.webp) ·
 [simple-green-2sym-1bit-each](files/simple-green-2sym-1bit-each.webp)

**The code-length code** —
 [codelen-repeat16-no-previous](files/codelen-repeat16-no-previous.webp) ·
 [codelen-repeat18-138-zeros](files/codelen-repeat18-138-zeros.webp) ·
 [codelen-repeat17-short-zeros](files/codelen-repeat17-short-zeros.webp) ·
 [codelen-max-symbol-early-stop](files/codelen-max-symbol-early-stop.webp) ·
 [codelen-max-symbol-too-big](files/codelen-max-symbol-too-big.webp) ·
 [codelen-repeat-past-end](files/codelen-repeat-past-end.webp) ·
 [codelen-num-codes-4](files/codelen-num-codes-4.webp) ·
 [codelen-num-codes-19](files/codelen-num-codes-19.webp) ·
 [codelen-depth-15](files/codelen-depth-15.webp) ·
 [codelen-single-symbol-complex-form](files/codelen-single-symbol-complex-form.webp) ·
 [codelen-over-capacity](files/codelen-over-capacity.webp) ·
 [codelen-oversubscribed](files/codelen-oversubscribed.webp) ·
 [codelen-two-level-table](files/codelen-two-level-table.webp) ·
 [codelen-incomplete](files/codelen-incomplete.webp) ·
 [codelen-all-zero-lengths](files/codelen-all-zero-lengths.webp)

**Meta Huffman / entropy image** —
 [meta-huffman-precision-min](files/meta-huffman-precision-min.webp) ·
 [meta-huffman-precision-max](files/meta-huffman-precision-max.webp) ·
 [meta-huffman-two-groups](files/meta-huffman-two-groups.webp) ·
 [meta-huffman-sparse-groups](files/meta-huffman-sparse-groups.webp) ·
 [meta-huffman-1001-groups](files/meta-huffman-1001-groups.webp)

**Color cache** — [cache-bits-1](files/cache-bits-1.webp) ·
 [cache-bits-11](files/cache-bits-11.webp) ·
 [cache-bits-0-invalid](files/cache-bits-0-invalid.webp) ·
 [cache-bits-12-invalid](files/cache-bits-12-invalid.webp) ·
 [cache-index-literal](files/cache-index-literal.webp)

**Palette packing** —
 [transform-palette-2-colors](files/transform-palette-2-colors.webp) ·
 [transform-palette-256-colors](files/transform-palette-256-colors.webp) ·
 [transform-palette-3-colors](files/transform-palette-3-colors.webp) ·
 [transform-palette-index-past-end](files/transform-palette-index-past-end.webp) ·
 [transform-palette-16-colors](files/transform-palette-16-colors.webp) ·
 [transform-palette-1-color](files/transform-palette-1-color.webp)

**Transforms** — [transform-all-four](files/transform-all-four.webp) ·
 [transform-repeated](files/transform-repeated.webp) ·
 [transform-predictor-bits-max](files/transform-predictor-bits-max.webp)

**Back-references** — [lz77-distance-1-run](files/lz77-distance-1-run.webp) ·
 [lz77-max-length-symbol](files/lz77-max-length-symbol.webp) ·
 [lz77-plane-code-1](files/lz77-plane-code-1.webp) ·
 [lz77-plane-code-clamped-to-1](files/lz77-plane-code-clamped-to-1.webp) ·
 [lz77-plane-code-120](files/lz77-plane-code-120.webp) ·
 [lz77-distance-direct-121](files/lz77-distance-direct-121.webp) ·
 [lz77-distance-past-start](files/lz77-distance-past-start.webp) ·
 [lz77-length-past-end](files/lz77-length-past-end.webp)

**Predictor modes** —
 [predictor-all-16-modes](files/predictor-all-16-modes.webp) ·
 [predictor-mode-14-undefined](files/predictor-mode-14-undefined.webp) ·
 [predictor-mode-15-undefined](files/predictor-mode-15-undefined.webp) ·
 [predictor-mode-11-select](files/predictor-mode-11-select.webp) ·
 [predictor-mode-13-clamp-half](files/predictor-mode-13-clamp-half.webp) ·
 [predictor-single-row](files/predictor-single-row.webp) ·
 [predictor-tile-bits-min](files/predictor-tile-bits-min.webp)

**Frame header** — [header-width-16384](files/header-width-16384.webp) ·
 [header-max-area-bomb](files/header-max-area-bomb.webp) ·
 [header-max-area-truncated](files/header-max-area-truncated.webp) ·
 [header-version-nonzero](files/header-version-nonzero.webp)

**The RIFF container** —
 [container-chunk-header-truncated](files/container-chunk-header-truncated.webp) ·
 [container-duplicate-image-chunk](files/container-duplicate-image-chunk.webp) ·
 [container-metadata-chunks](files/container-metadata-chunks.webp) ·
 [container-no-image-chunk](files/container-no-image-chunk.webp) ·
 [container-odd-chunk-no-pad](files/container-odd-chunk-no-pad.webp) ·
 [container-odd-chunk-payload](files/container-odd-chunk-payload.webp) ·
 [container-riff-size-past-end](files/container-riff-size-past-end.webp) ·
 [container-riff-size-short](files/container-riff-size-short.webp) ·
 [container-riff-size-truncates-chunks](files/container-riff-size-truncates-chunks.webp) ·
 [container-trailing-bytes](files/container-trailing-bytes.webp) ·
 [container-unknown-chunk](files/container-unknown-chunk.webp) ·
 [container-vp8x-animation](files/container-vp8x-animation.webp) ·
 [container-vp8x-area-overflow](files/container-vp8x-area-overflow.webp) ·
 [container-vp8x-canvas-mismatch](files/container-vp8x-canvas-mismatch.webp) ·
 [container-vp8x-reserved-bits](files/container-vp8x-reserved-bits.webp) ·
 [container-vp8x-still-flags](files/container-vp8x-still-flags.webp) ·
 [container-vp8x-wrong-size](files/container-vp8x-wrong-size.webp) ·
 [container-vp8x](files/container-vp8x.webp) ·
 [container-zero-size-chunk](files/container-zero-size-chunk.webp)

**The alpha chunk** — [alph-after-image](files/alph-after-image.webp) ·
 [alph-compression-invalid](files/alph-compression-invalid.webp) ·
 [alph-empty-payload](files/alph-empty-payload.webp) ·
 [alph-lossless-byte-flipped](files/alph-lossless-byte-flipped.webp) ·
 [alph-lossless-palette](files/alph-lossless-palette.webp) ·
 [alph-lossless-predictor](files/alph-lossless-predictor.webp) ·
 [alph-lossless-truncated](files/alph-lossless-truncated.webp) ·
 [alph-no-vp8x](files/alph-no-vp8x.webp) ·
 [alph-preprocessing-invalid](files/alph-preprocessing-invalid.webp) ·
 [alph-raw-filter-gradient](files/alph-raw-filter-gradient.webp) ·
 [alph-raw-filter-horizontal](files/alph-raw-filter-horizontal.webp) ·
 [alph-raw-filter-none](files/alph-raw-filter-none.webp) ·
 [alph-raw-filter-vertical](files/alph-raw-filter-vertical.webp) ·
 [alph-raw-oversized](files/alph-raw-oversized.webp) ·
 [alph-raw-preprocessing](files/alph-raw-preprocessing.webp) ·
 [alph-raw-short](files/alph-raw-short.webp) ·
 [alph-reserved-set](files/alph-reserved-set.webp) ·
 [alph-without-vp8x-flag](files/alph-without-vp8x-flag.webp)

**Lossy: frame tag and picture header** —
 [lossy-frame-bad-start-code](files/lossy-frame-bad-start-code.webp) ·
 [lossy-frame-colorspace-clamp](files/lossy-frame-colorspace-clamp.webp) ·
 [lossy-frame-interframe](files/lossy-frame-interframe.webp) ·
 [lossy-frame-not-shown](files/lossy-frame-not-shown.webp) ·
 [lossy-frame-part0-empty](files/lossy-frame-part0-empty.webp) ·
 [lossy-frame-part0-past-end](files/lossy-frame-part0-past-end.webp) ·
 [lossy-frame-scale-1](files/lossy-frame-scale-1.webp) ·
 [lossy-frame-scale-2](files/lossy-frame-scale-2.webp) ·
 [lossy-frame-scaled](files/lossy-frame-scaled.webp) ·
 [lossy-frame-version-1](files/lossy-frame-version-1.webp) ·
 [lossy-frame-version-2](files/lossy-frame-version-2.webp) ·
 [lossy-frame-version-3](files/lossy-frame-version-3.webp) ·
 [lossy-frame-version-4](files/lossy-frame-version-4.webp) ·
 [lossy-frame-version-7](files/lossy-frame-version-7.webp) ·
 [lossy-frame-width-16383](files/lossy-frame-width-16383.webp) ·
 [lossy-frame-zero-width](files/lossy-frame-zero-width.webp)

**Lossy: segmentation** —
 [lossy-segment-delta-quantizers](files/lossy-segment-delta-quantizers.webp) ·
 [lossy-segment-filter-strengths](files/lossy-segment-filter-strengths.webp) ·
 [lossy-segment-four-quantizers](files/lossy-segment-four-quantizers.webp) ·
 [lossy-segment-map-only](files/lossy-segment-map-only.webp) ·
 [lossy-segment-no-map](files/lossy-segment-no-map.webp) ·
 [lossy-segment-prob-extremes](files/lossy-segment-prob-extremes.webp) ·
 [lossy-segment-quant-extremes](files/lossy-segment-quant-extremes.webp)

**Lossy: loop filter** —
 [lossy-filter-lf-delta-extremes](files/lossy-filter-lf-delta-extremes.webp) ·
 [lossy-filter-lf-delta](files/lossy-filter-lf-delta.webp) ·
 [lossy-filter-normal-max](files/lossy-filter-normal-max.webp) ·
 [lossy-filter-sharpness-4](files/lossy-filter-sharpness-4.webp) ·
 [lossy-filter-sharpness-5](files/lossy-filter-sharpness-5.webp) ·
 [lossy-filter-simple-max](files/lossy-filter-simple-max.webp)

**Lossy: quantizer** —
 [lossy-quant-deltas-mirrored](files/lossy-quant-deltas-mirrored.webp) ·
 [lossy-quant-deltas](files/lossy-quant-deltas.webp) ·
 [lossy-quant-dequant-overflow](files/lossy-quant-dequant-overflow.webp) ·
 [lossy-quant-max](files/lossy-quant-max.webp) ·
 [lossy-quant-min](files/lossy-quant-min.webp) ·
 [lossy-quant-uv-dc-clamp](files/lossy-quant-uv-dc-clamp.webp)

**Lossy: coefficient probabilities** —
 [lossy-proba-all-updated](files/lossy-proba-all-updated.webp) ·
 [lossy-proba-one-update](files/lossy-proba-one-update.webp) ·
 [lossy-proba-refresh-and-skip-zero](files/lossy-proba-refresh-and-skip-zero.webp) ·
 [lossy-proba-skip-extremes](files/lossy-proba-skip-extremes.webp) ·
 [lossy-proba-zero](files/lossy-proba-zero.webp)

**Lossy: prediction modes** —
 [lossy-mode-i16-all-four](files/lossy-mode-i16-all-four.webp) ·
 [lossy-mode-i4-all-ten](files/lossy-mode-i4-all-ten.webp) ·
 [lossy-mode-i4-context](files/lossy-mode-i4-context.webp) ·
 [lossy-mode-mixed](files/lossy-mode-mixed.webp) ·
 [lossy-mode-uv-all-four](files/lossy-mode-uv-all-four.webp)

**Lossy: coefficients** —
 [lossy-coeff-all-types](files/lossy-coeff-all-types.webp) ·
 [lossy-coeff-bands-chroma](files/lossy-coeff-bands-chroma.webp) ·
 [lossy-coeff-bands-i16](files/lossy-coeff-bands-i16.webp) ·
 [lossy-coeff-bands-i4](files/lossy-coeff-bands-i4.webp) ·
 [lossy-coeff-cat3](files/lossy-coeff-cat3.webp) ·
 [lossy-coeff-cat4](files/lossy-coeff-cat4.webp) ·
 [lossy-coeff-cat5](files/lossy-coeff-cat5.webp) ·
 [lossy-coeff-cat6-max](files/lossy-coeff-cat6-max.webp) ·
 [lossy-coeff-cat6](files/lossy-coeff-cat6.webp) ·
 [lossy-coeff-context](files/lossy-coeff-context.webp) ·
 [lossy-coeff-empty-blocks](files/lossy-coeff-empty-blocks.webp) ·
 [lossy-coeff-full-block](files/lossy-coeff-full-block.webp) ·
 [lossy-coeff-medium-magnitudes](files/lossy-coeff-medium-magnitudes.webp) ·
 [lossy-coeff-small-magnitudes](files/lossy-coeff-small-magnitudes.webp) ·
 [lossy-coeff-wht-full](files/lossy-coeff-wht-full.webp) ·
 [lossy-coeff-zero-runs](files/lossy-coeff-zero-runs.webp)

**Lossy: skipped macroblocks** — [lossy-skip-all](files/lossy-skip-all.webp) ·
 [lossy-skip-i4x4-nz-dc](files/lossy-skip-i4x4-nz-dc.webp) ·
 [lossy-skip-mixed](files/lossy-skip-mixed.webp)

**Lossy: token partitions** —
 [lossy-parts-2-wrap](files/lossy-parts-2-wrap.webp) ·
 [lossy-parts-8-rows](files/lossy-parts-8-rows.webp) ·
 [lossy-parts-last-empty](files/lossy-parts-last-empty.webp) ·
 [lossy-parts-size-past-end](files/lossy-parts-size-past-end.webp) ·
 [lossy-parts-table-too-small](files/lossy-parts-table-too-small.webp)

**Lossy: truncation** —
 [lossy-truncated-header](files/lossy-truncated-header.webp) ·
 [lossy-truncated-modes](files/lossy-truncated-modes.webp) ·
 [lossy-truncated-short-modes](files/lossy-truncated-short-modes.webp) ·
 [lossy-truncated-tokens](files/lossy-truncated-tokens.webp)

**Lossy: partition sizes, from real encodes** —
 [lossy-1-partitions](files/lossy-1-partitions.webp) ·
 [lossy-2-partitions](files/lossy-2-partitions.webp) ·
 [lossy-4-partitions](files/lossy-4-partitions.webp) ·
 [lossy-8-partitions](files/lossy-8-partitions.webp) ·
 [lossy-8-partitions-size-overflow](files/lossy-8-partitions-size-overflow.webp) ·
 [lossy-8-partitions-zero-sizes](files/lossy-8-partitions-zero-sizes.webp) ·
 [lossy-8-partitions-sizes-sum-past-end](files/lossy-8-partitions-sizes-sum-past-end.webp) ·
 [lossy-combo-all-features](files/lossy-combo-all-features.webp)

## The code

| file | what it is |
| --- | --- |
| [`vp8l.py`](vp8l.py) | VP8L bitstream writer: bit packing, canonical Huffman codes, prefix coding, RIFF wrapping. |
| [`generate.py`](generate.py) | Writes the lossless cases, assembles the rest from `cases/`, and produces `expected.txt`, this README and the two `index.html` listings. |
| [`vp8.py`](vp8.py) | VP8 lossy bitstream writer: the boolean coder, the frame header, the mode trees, the coefficients. |
| [`vp8_asm.py`](vp8_asm.py) | Assembles a lossy frame from a text case, in RFC 6386's field names. Its docstring is the format. |
| [`webp_asm.py`](webp_asm.py) | Wraps that frame in a RIFF container, in RFC 9649's field names, for the cases that need one. |
| [`vp8_dis.py`](vp8_dis.py) | The other direction: a lossy .webp back into that text. `--check` round trips one against libwebp. |
| [`vp8_selftest.py`](vp8_selftest.py) | Round trips real encodes through both, and checks what cwebp cannot emit against dwebp. |
| [`vp8_tables.py`](vp8_tables.py) | The VP8 constant tables, extracted from libwebp. |
| [`make_vp8_tables.py`](make_vp8_tables.py) | Extracts them, so they are never retyped. |
| [`lossy_parts.py`](lossy_parts.py) | The multi-partition lossy cases, patched from `sources/`. |
| [`make_partition_sources.c`](make_partition_sources.c) | Rebuilds `sources/`: cwebp cannot emit more than one token partition. |
| [`check.sh`](check.sh) | Decodes every file; checks the verdict and the pixels. |
| [`make_hashes.sh`](make_hashes.sh) | Rewrites `hashes.txt` when the new output is known to be right. |
| [`asan_sweep.sh`](asan_sweep.sh) | Decodes every file in 14 modes, under a sanitizer build. |
| [`probes.py`](probes.py) | The `fprintf` probes `make_coverage.sh` patches in. |
| [`make_coverage.sh`](make_coverage.sh) | Rebuilds `coverage.txt` in a throwaway worktree. |
| [`expected.txt`](expected.txt) | Name and expected verdict, one line per file. |
| [`hashes.txt`](hashes.txt) | SHA-256 of each decoding file's `-pam` output. |
| [`coverage.txt`](coverage.txt) | Which decoder path each file actually reached. |
| [`COPYING`](COPYING) | BSD 3-clause, the same as libwebp. |

## Using them

    ./check.sh              # verdict + decoded-pixel hash for every file
    ./asan_sweep.sh         # 14 decode modes, under a sanitizer build
    ./vp8_selftest.py       # checks the lossy writer against libwebp
    ./make_coverage.sh      # regenerate coverage.txt
    ./make_hashes.sh        # regenerate hashes.txt, once the output is right
    python3 generate.py     # rebuild files/, expected.txt and this README

A case is a text file, one field per line under the name the specification
gives it -- RFC 6386 for the frame, RFC 9649 for the container -- so a case
reads against the format rather than against the decoder that happens to be
under test. `webp_asm.py` assembles any of them and hands the frame part to
`vp8_asm.py`; use either directly, or read an existing frame back out as
text to start from:

    ./webp_asm.py cases/alph-raw-filter-gradient.bitstream /tmp/out.webp
    ./vp8_asm.py cases/lossy-coeff-cat6.bitstream /tmp/out.webp
    ./vp8_dis.py some-photo.webp

Each tool's docstring is the reference for the fields it owns:
`vp8_asm.py` for the frame, `webp_asm.py` for the container and the alpha
chunk.

`files/` is pure output and is wiped on every rebuild. The four lossy encodes
the multi-partition cases are patched from live in `sources/` --
[1](sources/lossy-1-partitions.webp), [2](sources/lossy-2-partitions.webp),
[4](sources/lossy-4-partitions.webp), [8](sources/lossy-8-partitions.webp)
partitions -- and are themselves rebuilt by `make_partition_sources.c`.

`check.sh`, `make_hashes.sh` and `vp8_selftest.py` honour `$DWEBP` and
`asan_sweep.sh` honours `$ASAN_DWEBP`, so all of them can be pointed at any
build, or at another decoder implementation; they fall back to whatever
`dwebp` is on `$PATH`. `make_coverage.sh` and `make_vp8_tables.py` need
`$LIBWEBP` set to a libwebp git checkout. `SKIP_SLOW=1` skips the one file
that allocates a gigabyte.

`hashes.txt` holds the SHA-256 of each decoding file's `-pam` output, so the
suite catches a *silent* change in decoded pixels, not just a crash or a
changed verdict.

## How they were verified

Verdicts alone prove little -- a file can be rejected for the wrong reason,
and several of these were before being corrected. Each file was also run
against a decoder instrumented with probes on the exact lines the notes below
refer to; `coverage.txt` records which paths each file actually reached, and
`make_coverage.sh` regenerates it from `probes.py` in a throwaway worktree.
The notes are written from that output, not from reading the code.

The source line numbers they quote (`vp8l_dec.c:111` and friends) are only
meaningful against one revision of libwebp: the one recorded at the top of
`coverage.txt`, which `make_coverage.sh` stamps automatically. If those two
disagree, trust `coverage.txt`.

The lossy writer is checked a second way, against libwebp rather than against
itself: `vp8_dis.py` reads a frame back into the same text `vp8_asm.py`
assembles, so a real cwebp encode can be disassembled, reassembled and
compared byte for byte. Every file in `sources/` survives that, as do encodes
from 1x1 to 128x128 across the whole quality range -- 596 macroblocks, all
four 16x16 modes, all ten 4x4 modes and coefficients in every escape
category. `vp8_selftest.py` runs it, along with every coefficient magnitude
up to the format's largest and a handful of frames that say the same thing
two different ways and must decode alike.

What the corpus reaches is measured rather than assumed, and the measurement
is what says where to add files next. As it stands: every field of the lossy
frame header is written at both ends of its range; all 93 reachable
(coefficient type, band, context) probability cells are read, which is the
whole grid bar the three that no bitstream can select; all 28 pairs of
optional tools appear together in some frame; and one probe out of 88 is
unreached, a version check that `VP8LCheckSignature()` has already made by
the time it runs.

## What is not covered

Animation. There is no real ANIM or ANMF chunk, only the VP8X flag that
claims one, and nothing here goes near the demux API that would be needed to
walk a sequence of frames. No inter frames either, which libwebp refuses
outright, so there is little to pin beyond the one file that checks it does.

The two compressed alpha planes are cwebp output pasted in, so that path has
two points in it rather than a swept range: one that reaches the lossless
decoder's 8-bit loop and one that misses it. `vp8l.py` could generate them --
an alpha plane is a VP8L image whose green channel carries the values -- and
then they could be malformed like every other lossless case here.

Within a lossy key frame, what is left is what the decoder does not read:
the profile selects no reconstruction filter, and the entropy-refresh bit is
parsed and dropped. Both are written anyway, so a decoder that started
acting on either would fail a pixel hash rather than pass unnoticed.

## License

BSD 3-clause, the same as libwebp. See [`COPYING`](COPYING). That covers the
generators, the scripts and the bitstreams in `files/` alike.

## Simple codes

The 1-or-2-symbol shorthand a Huffman code can take. Its symbols are read as
raw 8-bit values and are never checked against the alphabet size, so this is
where a stream can say things an encoder cannot.

### [`simple-dist-2sym-first-oob.webp`](files/simple-dist-2sym-first-oob.webp) -- ok

Distance code: simple form, 2 symbols, the first one 200 >= alphabet_size 40.

ReadHuffmanCode() writes code_lengths[200] with alphabet_size 40; the code then
has one symbol left and is accepted. Pins the behaviour CL 8256621 documents.

### [`simple-dist-2sym-second-oob.webp`](files/simple-dist-2sym-second-oob.webp) -- ok

Distance code: simple form, 2 symbols, the second one 200 >= 40.

Same as above but the out-of-range symbol is the second 8-bit field.

### [`simple-dist-2sym-both-oob.webp`](files/simple-dist-2sym-both-oob.webp) -- reject

Distance code: both simple-form symbols out of range (200, 201).

No symbol is left inside alphabet_size, so BuildHuffmanTable() sees an empty
code and fails. Must stay a clean BITSTREAM_ERROR, not a crash.

### [`simple-dist-1sym-oob.webp`](files/simple-dist-1sym-oob.webp) -- reject

Distance code: simple form, single symbol 255, alphabet_size is 40.

The single write lands past the logical alphabet but inside the shared
max_alphabet_size buffer. Rejected because no symbol remains.

### [`simple-dist-sym-39-last-valid.webp`](files/simple-dist-sym-39-last-valid.webp) -- ok

Distance code: single symbol 39, the last in-range value.

Boundary partner of simple-dist-sym-40-first-oob: 39 == NUM_DISTANCE_CODES - 1
must be accepted.

### [`simple-dist-sym-40-first-oob.webp`](files/simple-dist-sym-40-first-oob.webp) -- reject

Distance code: single symbol 40, the first out-of-range value.

Exact boundary of the check that does not exist in ReadHuffmanCode(). If
someone adds one, these two files pin where it goes.

### [`simple-green-1bit-symbol.webp`](files/simple-green-1bit-symbol.webp) -- ok

Green code: simple form with first_symbol_len_code = 0, so the symbol is 1 bit
wide.

The short form of the simple code, only reachable when the symbol is 0 or 1.
cwebp emits it rarely.

### [`simple-dist-2sym-duplicate.webp`](files/simple-dist-2sym-duplicate.webp) -- ok

Distance code: simple form declaring 2 symbols that are the same (5, 5).

code_lengths[5] is written twice, so the code really has one symbol.
BuildHuffmanTable() takes its single-value shortcut.

### [`simple-green-2sym-1bit-each.webp`](files/simple-green-2sym-1bit-each.webp) -- ok

Green code with two real symbols, so every pixel costs exactly 1 bit.

The smallest non-trivial code. 4x1 pixels alternate between the two green
values.

## The code-length code

The Huffman code that describes the lengths of another Huffman code, plus its
repeat escapes (16, 17, 18) and the optional max_symbol field. cwebp only ever
emits a narrow slice of this.

### [`codelen-repeat16-no-previous.webp`](files/codelen-repeat16-no-previous.webp) -- ok

Code-length stream starting with code 16 (repeat previous), before any non-zero
length was seen.

Hits DEFAULT_CODE_LENGTH: 'prev_code_len' is still 8 at vp8l_dec.c:254, so the
first symbols get length 8 out of nowhere.

### [`codelen-repeat18-138-zeros.webp`](files/codelen-repeat18-138-zeros.webp) -- ok

Code-length stream using code 18 with its maximum run of 138 zeros.

Longest repeat the format allows (11 + 127). Green alphabet is 280 symbols so
two of them fit.

### [`codelen-repeat17-short-zeros.webp`](files/codelen-repeat17-short-zeros.webp) -- ok

Code-length stream using code 17 (3..10 zeros) rather than 18.

The short zero-run escape. Its extra field is 3 bits, offset 3.

### [`codelen-max-symbol-early-stop.webp`](files/codelen-max-symbol-early-stop.webp) -- ok

Code-length stream with an explicit max_symbol far below the alphabet size.

ReadHuffmanCodeLengths() breaks out at vp8l_dec.c:284 with most lengths still
zero. Exercises the use_length branch cwebp never takes.

### [`codelen-max-symbol-too-big.webp`](files/codelen-max-symbol-too-big.webp) -- reject

Explicit max_symbol greater than the alphabet size.

Must be caught by the max_symbol > num_symbols test at vp8l_dec.c:273.

### [`codelen-repeat-past-end.webp`](files/codelen-repeat-past-end.webp) -- reject

A repeat run that would write past the end of the alphabet.

Must be caught by the symbol + repeat > num_symbols test at vp8l_dec.c:298.

### [`codelen-num-codes-4.webp`](files/codelen-num-codes-4.webp) -- ok

Only 4 code-length codes declared, the minimum the 4-bit field allows.

Restricts the code-length alphabet to {17, 18, 0, 1}, so lengths can only be 0
or 1 plus the two zero-run escapes.

### [`codelen-num-codes-19.webp`](files/codelen-num-codes-19.webp) -- ok

All 19 code-length codes declared.

Maximum of the 4-bit num_codes field; every entry of kCodeLengthCodeOrder[]
gets a 3-bit length.

### [`codelen-depth-15.webp`](files/codelen-depth-15.webp) -- ok

A green code containing a symbol of depth 15, MAX_ALLOWED_CODE_LENGTH.

The deepest code the format allows; forces the two-level lookup in
BuildHuffmanTable() past HUFFMAN_TABLE_BITS.

### [`codelen-single-symbol-complex-form.webp`](files/codelen-single-symbol-complex-form.webp) -- ok

The complex form used to describe a code with exactly one symbol.

Takes BuildHuffmanTable()'s offset[MAX_ALLOWED_CODE_LENGTH] == 1 shortcut,
which makes the code 0 bits wide.

### [`codelen-over-capacity.webp`](files/codelen-over-capacity.webp) -- reject

Three symbols of depth 1, more than the two codes of that length that exist.

Caught early, by the count[len] > (1 << len) guard in BuildHuffmanTable(),
before the tree walk runs.

### [`codelen-oversubscribed.webp`](files/codelen-oversubscribed.webp) -- reject

Lengths 1, 2, 2, 2: each length is individually possible, but together they
over-subscribe the tree.

Slips past the per-length capacity guard and is caught later, when num_open
goes negative during the tree walk.

### [`codelen-two-level-table.webp`](files/codelen-two-level-table.webp) -- ok

A green code with depths up to 10, past the 8-bit root table.

Forces BuildHuffmanTable() to allocate a second-level table and ReadSymbol() to
take its two-step lookup.

### [`codelen-incomplete.webp`](files/codelen-incomplete.webp) -- reject

A code whose lengths leave the tree incomplete (two symbols of depth 2).

Caught by the num_nodes != 2 * num_symbols - 1 test at the end of
BuildHuffmanTable().

### [`codelen-all-zero-lengths.webp`](files/codelen-all-zero-lengths.webp) -- reject

A code-length stream that assigns length 0 to every symbol.

Empty code. Different route to the same rejection as simple-dist-1sym-oob.

## Meta Huffman / entropy image

The sub-image that picks one of several code groups per tile, and the remapping
the decoder does when the group count looks implausible.

### [`meta-huffman-precision-min.webp`](files/meta-huffman-precision-min.webp) -- ok

Meta Huffman with the smallest tile size (precision 2, 4x4 pixels).

MIN_HUFFMAN_BITS. A 16x16 image is split into 4x4 = 16 tiles, all pointing at
group 0.

### [`meta-huffman-precision-max.webp`](files/meta-huffman-precision-max.webp) -- ok

Meta Huffman with the largest tile size (precision 9, 512x512 pixels).

MAX_HUFFMAN_BITS. One tile covers the whole image, so the entropy image is 1x1.

### [`meta-huffman-two-groups.webp`](files/meta-huffman-two-groups.webp) -- ok

Two Huffman groups selected per tile by the entropy image.

The left half of the image uses group 0 (green 0x20), the right half group 1
(green 0xd0).

### [`meta-huffman-sparse-groups.webp`](files/meta-huffman-sparse-groups.webp) -- ok

Entropy image referencing groups 0 and 900 only, leaving a 900-entry hole.

num_htree_groups_max (901) exceeds the pixel count, so ReadHuffmanCodes()
builds the mapping[] remap and the 899 unused groups take the "validate but do
not store" branch.

### [`meta-huffman-1001-groups.webp`](files/meta-huffman-1001-groups.webp) -- ok

Entropy image whose highest group index is 1000, one past the decoder's
arbitrary limit.

Crosses the num_htree_groups_max > 1000 test at vp8l_dec.c:409, which forces
the mapping[] path even when the count is plausible.

## Color cache

Size bounds, and cache-index literals.

### [`cache-bits-1.webp`](files/cache-bits-1.webp) -- ok

Color cache with the minimum size, 1 bit (2 entries).

Lower bound of the cache_bits >= 1 check in DecodeImageStream().

### [`cache-bits-11.webp`](files/cache-bits-11.webp) -- ok

Color cache with the maximum size, 11 bits (2048 entries).

MAX_CACHE_BITS. Also stretches the green alphabet to 280 + 2048 symbols.

### [`cache-bits-0-invalid.webp`](files/cache-bits-0-invalid.webp) -- reject

Color cache flagged as present but with 0 bits.

Must be rejected: the format reserves "no cache" for the flag bit, so 0 is not
a legal size.

### [`cache-bits-12-invalid.webp`](files/cache-bits-12-invalid.webp) -- reject

Color cache with 12 bits, one past MAX_CACHE_BITS.

Upper bound of the same check. The 4-bit field can hold up to 15.

### [`cache-index-literal.webp`](files/cache-index-literal.webp) -- ok

A pixel coded as a color-cache index rather than as a literal.

Green symbols >= NUM_LITERAL_CODES + NUM_LENGTH_CODES address the cache. Pixel
2 replays pixel 1 through cache slot 0.

## Palette packing

Index width follows the palette size, and the map is padded out to the packing
capacity with black.

### [`transform-palette-2-colors.webp`](files/transform-palette-2-colors.webp) -- ok

Color-indexing transform with 2 colors, so 8 pixels are packed per byte.

num_colors <= 2 gives bits = 3, the densest packing, and shrinks xsize to
ceil(w / 8).

### [`transform-palette-256-colors.webp`](files/transform-palette-256-colors.webp) -- ok

Color-indexing transform with the full 256-entry palette.

MAX_PALETTE_SIZE, bits = 0 so there is no packing; also the largest value the
8-bit num_colors field can hold.

### [`transform-palette-3-colors.webp`](files/transform-palette-3-colors.webp) -- ok

Palette of 3 colors, so indices are 2 bits and 4 pixels share a byte.

num_colors in 3..4 selects bits = 2, the middle packing density. The byte 0xe4
holds indices 0, 1, 2, 3 least-significant first.

### [`transform-palette-index-past-end.webp`](files/transform-palette-index-past-end.webp) -- ok

Palette of 3 colors addressed with index 3, which does not exist.

Reads ExpandColorMap()'s black tail (vp8l_dec.c:1412): the map is padded out to
the packing capacity of 4, so the pixel comes back transparent black instead of
out of bounds.

### [`transform-palette-16-colors.webp`](files/transform-palette-16-colors.webp) -- ok

Palette of 16 colors, so indices are 4 bits and 2 pixels share a byte.

The bits = 1 packing, and the largest palette that still packs.

### [`transform-palette-1-color.webp`](files/transform-palette-1-color.webp) -- ok

Palette with a single color, the smallest the 8-bit field can express.

bits = 3, so 8 pixels share a byte and the map is padded from 1 entry to 2.
Index 1 is the black tail.

## Transforms

Presence, repetition and tile sizes.

### [`transform-all-four.webp`](files/transform-all-four.webp) -- ok

All four transforms present in one stream.

NUM_TRANSFORMS in a row: color-indexing, subtract-green, cross-color and
predictor, each with its own sub-image.

### [`transform-repeated.webp`](files/transform-repeated.webp) -- reject

The subtract-green transform declared twice.

Each transform type may appear once. Caught by the transforms_seen bitmask in
ReadTransform().

### [`transform-predictor-bits-max.webp`](files/transform-predictor-bits-max.webp) -- ok

Predictor transform with bits = 9, the maximum tile size.

MIN_TRANSFORM_BITS + 7. The predictor sub-image is a single pixel for any image
up to 512x512.

## Back-references

Copy lengths and distances.

### [`lz77-distance-1-run.webp`](files/lz77-distance-1-run.webp) -- ok

A single literal followed by a length-8 copy at distance 1.

The degenerate overlapping copy: the copy loop reads bytes it has just written.

### [`lz77-max-length-symbol.webp`](files/lz77-max-length-symbol.webp) -- ok

A back-reference using length symbol 23, the largest the format defines.

NUM_LENGTH_CODES - 1: 10 extra bits, copy lengths up to 4096. Here it copies
1200 pixels.

### [`lz77-plane-code-1.webp`](files/lz77-plane-code-1.webp) -- ok

Back-reference with plane code 1, which means "the pixel directly above".

kCodeToPlane[0] is 0x18: yoffset 1, xoffset 0, so the distance is a whole row
rather than a small number.

### [`lz77-plane-code-clamped-to-1.webp`](files/lz77-plane-code-clamped-to-1.webp) -- ok

Plane code 4 on a 1-pixel-wide image, where the 2-D offset computes to 0.

kCodeToPlane[3] is 0x19: yoffset 1, xoffset -1, so dist = xsize - 1 = 0 and the
"dist < 1 ? 1" clamp at vp8l_dec.c:173 fires. Only reachable at xsize 1.

### [`lz77-plane-code-120.webp`](files/lz77-plane-code-120.webp) -- ok

Plane code 120, the last entry of the 2-D offset table.

kCodeToPlane[119] is 0x70: yoffset 7, xoffset 8, so on a 16-wide image the
distance is 120. Upper bound of the mapped range.

### [`lz77-distance-direct-121.webp`](files/lz77-distance-direct-121.webp) -- ok

Plane code 121, the first value past the table.

Distances above CODE_TO_PLANE_CODES bypass the 2-D mapping entirely: the
distance is plane_code - 120, so 121 means 1.

### [`lz77-distance-past-start.webp`](files/lz77-distance-past-start.webp) -- reject

A back-reference pointing further back than the pixels decoded so far.

One literal, then a copy at a distance of one whole row. Must be rejected
rather than reading before the buffer.

### [`lz77-length-past-end.webp`](files/lz77-length-past-end.webp) -- reject

A copy whose length runs past the last pixel of the image.

Four literals then an 8-pixel copy in an 8-pixel image. Must be rejected rather
than writing past the buffer.

## Predictor modes

The per-tile predictor index. It is read as a 4-bit field, so all 16 values are
reachable, but the format only defines 14 of them.

### [`predictor-all-16-modes.webp`](files/predictor-all-16-modes.webp) -- ok

One tile per predictor index, 0 to 15, across a 64x4 image.

The mode is read as ((pixel >> 8) & 0xf) at lossless.c:247, so all 16 indices
are reachable even though the format defines only 0..13.

### [`predictor-mode-14-undefined.webp`](files/predictor-mode-14-undefined.webp) -- ok

Every tile selects predictor 14, which the format does not define.

Must decode, not crash: VP8LPredictorsAdd[14] is a padding sentinel pointing at
PredictorAdd0_C (lossless.c:653), so the tile comes out as mode 0. Shrink the
table to 14 entries and this is an out-of-bounds indirect call instead.

### [`predictor-mode-15-undefined.webp`](files/predictor-mode-15-undefined.webp) -- ok

Every tile selects predictor 15, the other undefined index.

Partner of predictor-mode-14-undefined and the largest value the 4-bit mask can
produce. Verified to decode identically to mode 14, i.e. both really do land on
PredictorAdd0_C.

### [`predictor-mode-11-select.webp`](files/predictor-mode-11-select.webp) -- ok

Predictor 11 (Select) over the whole image.

The only predictor with a data-dependent branch, Select() at lossless.c:100.

### [`predictor-mode-13-clamp-half.webp`](files/predictor-mode-13-clamp-half.webp) -- ok

Predictor 13 (ClampAddSubtractHalf) over the whole image.

Exercises AddSubtractComponentHalf() and its Clip255(), the arithmetic most
likely to differ between the C and SIMD paths.

### [`predictor-single-row.webp`](files/predictor-single-row.webp) -- ok

A predictor transform on a one-row image.

Only the y_start == 0 shortcut runs (lossless.c:223): the first pixel takes
mode 0 and the rest mode 1, so the tile modes are never read.

### [`predictor-tile-bits-min.webp`](files/predictor-tile-bits-min.webp) -- ok

Predictor tiles of 4x4 pixels, the smallest the format allows.

MIN_TRANSFORM_BITS, so the sub-image is as large as it can get and the mode
changes every four pixels.

## Frame header

The 14-bit dimension fields and the version escape.

### [`header-width-16384.webp`](files/header-width-16384.webp) -- ok

Width 16384, one past WEBP_MAX_DIMENSION.

The header stores width - 1 in 14 bits, so 16384 is expressible and the decoder
accepts it. WEBP_MAX_DIMENSION (16383) is enforced only in the encoder, at
webp_enc.c:347, so cwebp can never produce this.

### [`header-max-area-bomb.webp`](files/header-max-area-bomb.webp) -- ok

34 bytes declaring 16384x16384, every pixel one color.

A bare VP8L stream has no area limit -- MAX_IMAGE_AREA is only checked in
ParseVP8X (webp_dec.c:138) -- and single-symbol codes cost zero bits per pixel,
so this decodes for real: 1.83GB peak RSS and 3.4s. The backstop is
WEBP_MAX_ALLOCABLE_MEMORY (utils.c:185), nothing earlier.

### [`header-max-area-truncated.webp`](files/header-max-area-truncated.webp) -- reject

The same 16384x16384 header, cut off before the Huffman codes.

Must fail on the missing data rather than allocating the gigabyte first.
Partner of header-max-area-bomb: together they say where the allocation sits
relative to the parse.

### [`header-version-nonzero.webp`](files/header-version-nonzero.webp) -- reject

Header with the 3-bit version field set to 1.

Rejected by VP8LCheckSignature() at vp8l_dec.c:111, which tests (data[4] >> 5)
before ReadImageInfo() ever runs. The version field is the format's only
forward-compatibility escape.

## The RIFF container

The layer above the image: the RIFF header, the extended-format VP8X chunk and
its canvas size, the optional chunks a decoder must step over by their declared
length alone, and the padding rule that makes an odd-sized one even on disk.
Everything here is read by webp_dec.c before the frame is looked at.

### [`container-chunk-header-truncated.webp`](files/container-chunk-header-truncated.webp) -- reject -- from [`container-chunk-header-truncated.bitstream`](cases/container-chunk-header-truncated.bitstream)

The file cut four bytes into the last chunk header.

The "buf_size < CHUNK_HEADER_SIZE" test at the top of the walk: a tag with no
size behind it. Four bytes is enough to read the fourcc and not the length.

### [`container-duplicate-image-chunk.webp`](files/container-duplicate-image-chunk.webp) -- ok -- from [`container-duplicate-image-chunk.bitstream`](cases/container-duplicate-image-chunk.bitstream)

Two VP8 chunks, one after the other.

The walk stops at the first VP8 chunk, so the second is never looked at and
whatever it holds is dead weight. Nothing rejects the duplicate.

### [`container-metadata-chunks.webp`](files/container-metadata-chunks.webp) -- ok -- from [`container-metadata-chunks.bitstream`](cases/container-metadata-chunks.bitstream)

ICCP, EXIF and XMP chunks around the frame.

ParseOptionalChunks() walks and skips anything that is not VP8 or VP8L. The
three metadata chunks the format defines are the ones a real file is most
likely to carry, and the decoder must step over them by their declared size
alone.

### [`container-no-image-chunk.webp`](files/container-no-image-chunk.webp) -- reject -- from [`container-no-image-chunk.bitstream`](cases/container-no-image-chunk.bitstream)

A VP8X chunk and some metadata, and no image chunk at all.

ParseOptionalChunks() walks to the end of the data without meeting VP8 or VP8L
and runs out. A container that promises a picture and does not carry one.

### [`container-odd-chunk-no-pad.webp`](files/container-odd-chunk-no-pad.webp) -- reject -- from [`container-odd-chunk-no-pad.bitstream`](cases/container-odd-chunk-no-pad.bitstream)

An odd-sized chunk whose header rounds its length up, taking the pad byte away.

The walk steps by (8 + size + 1) & ~1, so an even declared size means no pad
byte, and everything after it is read one byte off. The partner of container-
odd-chunk-payload: same chunk, one byte shorter.

### [`container-odd-chunk-payload.webp`](files/container-odd-chunk-payload.webp) -- ok -- from [`container-odd-chunk-payload.bitstream`](cases/container-odd-chunk-payload.bitstream)

An optional chunk with an odd-sized payload, and the pad byte that must follow
it.

disk_chunk_size rounds a payload up to an even length. Every lossy file in the
corpus has an odd image chunk and so pads incidentally, but this is the only
one where the padded chunk has something after it, which is where getting the
rounding wrong would show.

### [`container-riff-size-past-end.webp`](files/container-riff-size-past-end.webp) -- reject -- from [`container-riff-size-past-end.bitstream`](cases/container-riff-size-past-end.bitstream)

A RIFF header claiming far more bytes than the file holds.

The "size > *data_size - CHUNK_HEADER_SIZE" test, which only fires when the
whole file is in hand -- the same lie is tolerated while a stream is still
arriving.

### [`container-riff-size-short.webp`](files/container-riff-size-short.webp) -- reject -- from [`container-riff-size-short.bitstream`](cases/container-riff-size-short.bitstream)

A RIFF header claiming 11 bytes, one less than the smallest legal value.

The "size < TAG_SIZE + CHUNK_HEADER_SIZE" test in ParseRIFF(): a RIFF size must
leave room for "WEBP" and one chunk header. 11 is the last value that does not.

### [`container-riff-size-truncates-chunks.webp`](files/container-riff-size-truncates-chunks.webp) -- reject -- from [`container-riff-size-truncates-chunks.bitstream`](cases/container-riff-size-truncates-chunks.bitstream)

A RIFF size that stops in the middle of the chunks behind it.

The "total_size > riff_size" test in ParseOptionalChunks(): the walk adds up
what it has skipped and refuses to walk past what the RIFF header said was
there.

### [`container-trailing-bytes.webp`](files/container-trailing-bytes.webp) -- ok -- from [`container-trailing-bytes.bitstream`](cases/container-trailing-bytes.bitstream)

Bytes after the last chunk that the RIFF size does not account for.

The decoder stops at the image chunk and never looks past it, so junk at the
end of the file is ignored rather than refused.

### [`container-unknown-chunk.webp`](files/container-unknown-chunk.webp) -- ok -- from [`container-unknown-chunk.bitstream`](cases/container-unknown-chunk.bitstream)

A chunk with a fourcc the format does not define, ahead of the frame.

The same skip path as the metadata chunks, but with a tag no version of libwebp
knows. An unknown chunk must be stepped over, not refused: that is what makes
the format extensible.

### [`container-vp8x-animation.webp`](files/container-vp8x-animation.webp) -- reject -- from [`container-vp8x-animation.bitstream`](cases/container-vp8x-animation.bitstream)

The VP8X animation flag set on a file with no animation chunks.

The flag alone is enough: WebPParseHeaders() turns any file claiming animation
into UNSUPPORTED_FEATURE, since a still decoder cannot compose frames. Nothing
looks for the ANIM chunk the flag implies.

### [`container-vp8x-area-overflow.webp`](files/container-vp8x-area-overflow.webp) -- reject -- from [`container-vp8x-area-overflow.bitstream`](cases/container-vp8x-area-overflow.bitstream)

A VP8X canvas of 16777216 by 16777216, the largest the two 24-bit fields can
describe.

The "width * height >= MAX_IMAGE_AREA" test in ParseVP8X(), computed in 64 bits
precisely so this cannot wrap. Both fields are at their maximum, so the product
is 2^48.

### [`container-vp8x-canvas-mismatch.webp`](files/container-vp8x-canvas-mismatch.webp) -- reject -- from [`container-vp8x-canvas-mismatch.bitstream`](cases/container-vp8x-canvas-mismatch.bitstream)

A VP8X canvas of 64x64 in front of a 32x32 frame.

The "Validates image size coherency" check at the end of
ParseHeadersInternal(): a VP8X chunk must agree with the frame behind it, in
both dimensions. The container is the only place in the format where the
picture size is stated twice.

### [`container-vp8x-reserved-bits.webp`](files/container-vp8x-reserved-bits.webp) -- ok -- from [`container-vp8x-reserved-bits.bitstream`](cases/container-vp8x-reserved-bits.bitstream)

The reserved bits of the VP8X flags field all set.

Everything outside ALL_VALID_FLAGS (0x3e). The decoder reads the field as a
whole and tests only the bits it knows, so the reserved ones ride through; the
muxer is stricter than the decoder here.

### [`container-vp8x-still-flags.webp`](files/container-vp8x-still-flags.webp) -- ok -- from [`container-vp8x-still-flags.bitstream`](cases/container-vp8x-still-flags.bitstream)

The four still-image VP8X flags set, with none of the chunks they promise.

Alpha, ICC, EXIF and XMP together. Nothing checks that a flag is backed by its
chunk, so this decodes as the plain lossy frame it is. The animation flag is
the one exception, which container-vp8x-animation covers.

### [`container-vp8x-wrong-size.webp`](files/container-vp8x-wrong-size.webp) -- reject -- from [`container-vp8x-wrong-size.bitstream`](cases/container-vp8x-wrong-size.bitstream)

A VP8X chunk whose header claims 9 bytes rather than 10.

The "chunk_size != VP8X_CHUNK_SIZE" test in ParseVP8X(), which is an equality:
a VP8X chunk is 10 bytes and no other length is tolerated, in either direction.

### [`container-vp8x.webp`](files/container-vp8x.webp) -- ok -- from [`container-vp8x.bitstream`](cases/container-vp8x.bitstream)

The extended format: a VP8X chunk ahead of the frame.

ParseVP8X(). The canvas size is written as width and height less one, and here
it agrees with the frame behind it. No file in the corpus had a VP8X chunk at
all before this one.

### [`container-zero-size-chunk.webp`](files/container-zero-size-chunk.webp) -- ok -- from [`container-zero-size-chunk.bitstream`](cases/container-zero-size-chunk.bitstream)

An optional chunk declaring a zero-length payload.

disk_chunk_size is then just the 8-byte header, which is the smallest step the
walk can take. A zero-length chunk is legal and must be stepped over like any
other.

## The alpha chunk

ALPH carries the alpha plane beside a lossy frame: a header byte of four two-
bit fields, then the plane itself, either stored as it is or compressed with
the lossless coder in its 8-bit mode. That mode is a separate path through
vp8l_dec.c from the one every VP8L file here takes, and these are the only
files that reach it.

### [`alph-after-image.webp`](files/alph-after-image.webp) -- ok -- from [`alph-after-image.bitstream`](cases/alph-after-image.bitstream)

An ALPH chunk placed after the image chunk instead of before it.

ParseOptionalChunks() stops at the first VP8 chunk, so an ALPH behind it is
never seen and the picture decodes fully opaque. Ordering is not diagnosed, it
is silently obeyed -- which the all-255 alpha in the hash is what records.

### [`alph-compression-invalid.webp`](files/alph-compression-invalid.webp) -- reject -- from [`alph-compression-invalid.bitstream`](cases/alph-compression-invalid.bitstream)

A compression method of 2, past the lossless one.

The header byte packs four fields into two bits each, and only the method and
the pre-processing have values the format does not define. This is the first of
them.

### [`alph-empty-payload.webp`](files/alph-empty-payload.webp) -- reject -- from [`alph-empty-payload.bitstream`](cases/alph-empty-payload.bitstream)

An ALPH chunk holding its header byte and nothing else.

The "data_size <= ALPHA_HEADER_LEN" test at the top of ALPHInit(), which is
what stops the header byte itself from being read out of an empty chunk.

### [`alph-lossless-byte-flipped.webp`](files/alph-lossless-byte-flipped.webp) -- reject -- from [`alph-lossless-byte-flipped.bitstream`](cases/alph-lossless-byte-flipped.bitstream)

The same plane with its last byte replaced.

One byte, not a truncation: the stream is the right length and stops making
sense at the end. Together with alph-lossless-truncated this pins both ways the
compressed plane can fail.

### [`alph-lossless-palette.webp`](files/alph-lossless-palette.webp) -- ok -- from [`alph-lossless-palette.bitstream`](cases/alph-lossless-palette.bitstream)

A losslessly compressed alpha plane carrying a palette transform, from a cwebp
encode of a two-valued plane.

The one shape that reaches DecodeAlphaData(): exactly one transform, that
transform colour-indexing, no colour cache, and the red, blue and alpha codes
each a single symbol. That is the lossless decoder's 8-bit mode, a different
loop from the one all 62 VP8L files here take, and this is the only file in the
corpus that runs it. Its partner alph- lossless-predictor is the same feature
failing the same test.

### [`alph-lossless-predictor.webp`](files/alph-lossless-predictor.webp) -- ok -- from [`alph-lossless-predictor.bitstream`](cases/alph-lossless-predictor.bitstream)

A losslessly compressed alpha plane carrying a predictor transform, from a
cwebp encode of a gradient.

Compression method 1 hands the payload to the lossless decoder. A predictor
transform leaves the red, blue and alpha codes non-trivial, so
Is8bOptimizable() says no and the plane is decoded through DecodeImageData()
with ExtractAlphaRows() pulling the green channel out afterwards. The 21
payload bytes are cwebp -q 60 output; nothing here can write one yet.

### [`alph-lossless-truncated.webp`](files/alph-lossless-truncated.webp) -- reject -- from [`alph-lossless-truncated.bitstream`](cases/alph-lossless-truncated.bitstream)

The predictor-transform plane cut to ten bytes.

The lossless decoder runs out part way through the alpha image. The failure
comes back through ALPHInit() as a bitstream error rather than as a short read.

### [`alph-no-vp8x.webp`](files/alph-no-vp8x.webp) -- reject -- from [`alph-no-vp8x.bitstream`](cases/alph-no-vp8x.bitstream)

An ALPH chunk in a RIFF file with no VP8X ahead of it.

The extended format is what an ALPH chunk lives in: libwebp only accepts a
leading ALPH when there is no RIFF header at all, the bare stream case. With
RIFF and no VP8X it is refused.

### [`alph-preprocessing-invalid.webp`](files/alph-preprocessing-invalid.webp) -- reject -- from [`alph-preprocessing-invalid.bitstream`](cases/alph-preprocessing-invalid.bitstream)

A pre-processing value of 2, past level reduction.

The second undefined value in the header byte, refused by the same condition in
ALPHInit(). All four filter values are legal, so the filter field has no
partner to this.

### [`alph-raw-filter-gradient.webp`](files/alph-raw-filter-gradient.webp) -- ok -- from [`alph-raw-filter-gradient.bitstream`](cases/alph-raw-filter-gradient.bitstream)

An uncompressed alpha plane under the gradient filter.

WebPUnfilters[3], from left plus above minus above-left. The four filters have
a routine each in dsp/filters.c and the same stored bytes come out as four
different planes, so the pixel hash is what tells them apart. Nothing in the
corpus reached any of them before.

### [`alph-raw-filter-horizontal.webp`](files/alph-raw-filter-horizontal.webp) -- ok -- from [`alph-raw-filter-horizontal.bitstream`](cases/alph-raw-filter-horizontal.bitstream)

An uncompressed alpha plane under the horizontal filter.

WebPUnfilters[1], each byte a difference from the one to its left. The four
filters have a routine each in dsp/filters.c and the same stored bytes come out
as four different planes, so the pixel hash is what tells them apart. Nothing
in the corpus reached any of them before.

### [`alph-raw-filter-none.webp`](files/alph-raw-filter-none.webp) -- ok -- from [`alph-raw-filter-none.bitstream`](cases/alph-raw-filter-none.bitstream)

An uncompressed alpha plane under the none filter.

WebPUnfilters[0], stored as it is. The four filters have a routine each in
dsp/filters.c and the same stored bytes come out as four different planes, so
the pixel hash is what tells them apart. Nothing in the corpus reached any of
them before.

### [`alph-raw-filter-vertical.webp`](files/alph-raw-filter-vertical.webp) -- ok -- from [`alph-raw-filter-vertical.bitstream`](cases/alph-raw-filter-vertical.bitstream)

An uncompressed alpha plane under the vertical filter.

WebPUnfilters[2], from the one above. The four filters have a routine each in
dsp/filters.c and the same stored bytes come out as four different planes, so
the pixel hash is what tells them apart. Nothing in the corpus reached any of
them before.

### [`alph-raw-oversized.webp`](files/alph-raw-oversized.webp) -- ok -- from [`alph-raw-oversized.bitstream`](cases/alph-raw-oversized.bitstream)

An uncompressed plane 44 bytes longer than the picture needs.

ALPHInit() tests "alpha_data_size >= alpha_decoded_size", so a plane may be
longer than width by height and the tail is simply never read. The boundary
partner of alph-raw-short.

### [`alph-raw-preprocessing.webp`](files/alph-raw-preprocessing.webp) -- ok -- from [`alph-raw-preprocessing.bitstream`](cases/alph-raw-preprocessing.bitstream)

An uncompressed plane declaring the level-reduction pre-processing.

The one bit of ALPHA_PREPROCESSED_LEVELS: it makes the decoder take the "decode
everything in one pass" branch and keeps alpha dithering alive instead of
switching it off. The plane itself comes out the same as with no pre-
processing, so this pins the control path rather than the pixels.

### [`alph-raw-short.webp`](files/alph-raw-short.webp) -- reject -- from [`alph-raw-short.bitstream`](cases/alph-raw-short.bitstream)

An uncompressed plane one byte short of the picture.

The other side of that same test, one byte away: 255 bytes where a 16x16
picture needs 256.

### [`alph-reserved-set.webp`](files/alph-reserved-set.webp) -- reject -- from [`alph-reserved-set.bitstream`](cases/alph-reserved-set.bitstream)

The two reserved bits of the ALPH header byte set.

The "rsrv != 0" arm of the same test. Unlike VP8X, whose reserved bits ride
through untouched, ALPH refuses a header with anything in its top two bits.

### [`alph-without-vp8x-flag.webp`](files/alph-without-vp8x-flag.webp) -- ok -- from [`alph-without-vp8x-flag.bitstream`](cases/alph-without-vp8x-flag.bitstream)

An ALPH chunk with the VP8X alpha flag left clear.

The flag and the chunk are independent: the decoder walks to the ALPH chunk and
uses it whatever VP8X claimed. A file the muxer would call inconsistent and the
decoder does not.

## Lossy: frame tag and picture header

The ten uncompressed bytes every lossy frame starts with: the profile, the
visibility and key-frame bits, the length of partition 0, the start code and
the two 14-bit dimensions.

### [`lossy-frame-bad-start-code.webp`](files/lossy-frame-bad-start-code.webp) -- reject -- from [`lossy-frame-bad-start-code.bitstream`](cases/lossy-frame-bad-start-code.bitstream)

The three-byte start code changed from 9d 01 2a to 9d 01 29.

VP8CheckSignature(). One bit away from valid, so it also checks that the
signature is compared, not merely skipped over.

### [`lossy-frame-colorspace-clamp.webp`](files/lossy-frame-colorspace-clamp.webp) -- ok -- from [`lossy-frame-colorspace-clamp.bitstream`](cases/lossy-frame-colorspace-clamp.bitstream)

The colour-space and clamping-type bits both set.

The two bits at the very top of partition 0. libwebp stores both and acts on
neither, so a decoder that started honouring either would fail this file's hash
rather than its verdict. Nothing else in the corpus sets them.

### [`lossy-frame-interframe.webp`](files/lossy-frame-interframe.webp) -- reject -- from [`lossy-frame-interframe.bitstream`](cases/lossy-frame-interframe.bitstream)

The key-frame bit cleared, so the frame claims to be an inter frame.

libwebp decodes single key frames only, and VP8GetInfo() turns this away on the
key-frame bit alone: the picture header behind it is never looked at, and
VP8GetHeaders() never runs.

### [`lossy-frame-not-shown.webp`](files/lossy-frame-not-shown.webp) -- reject -- from [`lossy-frame-not-shown.bitstream`](cases/lossy-frame-not-shown.bitstream)

A key frame with the show_frame bit cleared.

VP8GetInfo() bails on "first frame is invisible"; the VP8 layer would have said
UNSUPPORTED_FEATURE. Nothing else in the corpus reaches either.

### [`lossy-frame-part0-empty.webp`](files/lossy-frame-part0-empty.webp) -- reject -- from [`lossy-frame-part0-empty.bitstream`](cases/lossy-frame-part0-empty.bitstream)

The frame tag claims a zero-byte partition 0.

The boolean reader is handed no data at all: every header field reads past the
end, and ParseSegmentHeader() returns on br->eof.

### [`lossy-frame-part0-past-end.webp`](files/lossy-frame-part0-past-end.webp) -- reject -- from [`lossy-frame-part0-past-end.bitstream`](cases/lossy-frame-part0-past-end.bitstream)

The frame tag claims a partition 0 far larger than the file.

The 19-bit partition length. VP8GetInfo() catches "partition_length >=
chunk_size" first; VP8GetHeaders() has its own "bad partition length" behind
it.

### [`lossy-frame-scale-1.webp`](files/lossy-frame-scale-1.webp) -- ok -- from [`lossy-frame-scale-1.bitstream`](cases/lossy-frame-scale-1.bitstream)

A horizontal upscaling hint of 1 and a vertical one of 3.

Two of the four values of the two 2-bit scale fields; lossy- frame-scaled has 3
and 2 and lossy-frame-scale-2 the rest, so between the three every value of
both is written. libwebp reads them into pic_hdr and acts on neither.

### [`lossy-frame-scale-2.webp`](files/lossy-frame-scale-2.webp) -- ok -- from [`lossy-frame-scale-2.bitstream`](cases/lossy-frame-scale-2.bitstream)

A horizontal upscaling hint of 2 and a vertical one of 1.

The two values the other two scale cases leave out, so all four are seen in
each field. A decoder that started honouring the hint would resize the output
and fail the hash rather than the verdict.

### [`lossy-frame-scaled.webp`](files/lossy-frame-scaled.webp) -- ok -- from [`lossy-frame-scaled.bitstream`](cases/lossy-frame-scaled.bitstream)

Horizontal and vertical upscaling hints of 3 (2x) in the top bits of the
dimension fields.

pic_hdr->xscale and yscale. libwebp parses and ignores them, so the output
stays 32x32; a decoder that honoured them would fail the hash.

### [`lossy-frame-version-1.webp`](files/lossy-frame-version-1.webp) -- ok -- from [`lossy-frame-version-1.bitstream`](cases/lossy-frame-version-1.bitstream)

A frame declaring profile 1 instead of 0.

The 3-bit version field of the frame tag. libwebp accepts 0 to 3 and
reconstructs them all the same way, so this pins the acceptance, not the
pixels.

### [`lossy-frame-version-2.webp`](files/lossy-frame-version-2.webp) -- ok -- from [`lossy-frame-version-2.bitstream`](cases/lossy-frame-version-2.bitstream)

Profile 2, the value between the two the corpus already had.

Completes the four accepted values of the 3-bit version field: 0 and 1 and 3
were covered, 2 was not.

### [`lossy-frame-version-3.webp`](files/lossy-frame-version-3.webp) -- ok -- from [`lossy-frame-version-3.bitstream`](cases/lossy-frame-version-3.bitstream)

Profile 3, the largest the decoder accepts.

Boundary partner of lossy-frame-version-4: "profile > 3" is the whole check in
VP8GetHeaders().

### [`lossy-frame-version-4.webp`](files/lossy-frame-version-4.webp) -- reject -- from [`lossy-frame-version-4.bitstream`](cases/lossy-frame-version-4.bitstream)

Profile 4, one past the last valid value.

VP8GetInfo() rejects it before VP8GetHeaders() ever runs, so this comes back as
BITSTREAM_ERROR rather than the "Incorrect keyframe parameters" message the VP8
layer would give.

### [`lossy-frame-version-7.webp`](files/lossy-frame-version-7.webp) -- reject -- from [`lossy-frame-version-7.bitstream`](cases/lossy-frame-version-7.bitstream)

Profile 7, the largest the 3-bit field can hold.

The far end of the field, past the "profile > 3" test that lossy-frame-
version-4 sits on. Same rejection, opposite end of the range, which is what
pins the test as a comparison rather than an equality.

### [`lossy-frame-width-16383.webp`](files/lossy-frame-width-16383.webp) -- ok -- from [`lossy-frame-width-16383.bitstream`](cases/lossy-frame-width-16383.bitstream)

The widest frame the 14-bit field can describe, one macroblock tall.

1024 macroblocks in a single row, so the whole frame goes through one partition
and the per-column contexts are exercised 1024 wide.

### [`lossy-frame-zero-width.webp`](files/lossy-frame-zero-width.webp) -- reject -- from [`lossy-frame-zero-width.bitstream`](cases/lossy-frame-zero-width.bitstream)

A frame whose width field is zero, with a height of 32.

The "w == 0 || h == 0" check in VP8GetInfo(), which the comment above it
describes as not supporting both being zero while the code refuses either.
Nothing else in the corpus reaches it.

## Lossy: segmentation

Up to four segments, each with its own quantizer and loop-filter strength, and
a per-macroblock map saying which is which. cwebp uses the feature but only
ever writes absolute values, and always writes the map and the data together.

### [`lossy-segment-delta-quantizers.webp`](files/lossy-segment-delta-quantizers.webp) -- ok -- from [`lossy-segment-delta-quantizers.bitstream`](cases/lossy-segment-delta-quantizers.bitstream)

Segment quantizers read as deltas on the frame quantizer instead of absolute
values.

segment_feature_mode = 0, the "q += base_q0" branch of VP8ParseQuant(). cwebp
always writes absolute values (syntax_enc.c:196), so nothing else reaches this.

### [`lossy-segment-filter-strengths.webp`](files/lossy-segment-filter-strengths.webp) -- ok -- from [`lossy-segment-filter-strengths.bitstream`](cases/lossy-segment-filter-strengths.bitstream)

Per-segment loop-filter strengths, from 0 to 63, under a frame filter level of
40.

PrecomputeFilterStrengths() with use_segment: the per-segment base level
replaces the frame level outright when the deltas are absolute.

### [`lossy-segment-four-quantizers.webp`](files/lossy-segment-four-quantizers.webp) -- ok -- from [`lossy-segment-four-quantizers.bitstream`](cases/lossy-segment-four-quantizers.bitstream)

Four segments with four different absolute quantizers, one macroblock each.

The segment tree in ParseIntraMode() with all three probabilities used, and
four distinct VP8QuantMatrix rows in VP8ParseQuant().

### [`lossy-segment-map-only.webp`](files/lossy-segment-map-only.webp) -- ok -- from [`lossy-segment-map-only.bitstream`](cases/lossy-segment-map-only.bitstream)

A segment map with no segment data behind it.

update_map without update_data: the ids are read and used to index dqm[], but
every entry is the frame quantizer, so the map changes nothing. Pins that the
two flags are independent.

### [`lossy-segment-no-map.webp`](files/lossy-segment-no-map.webp) -- ok -- from [`lossy-segment-no-map.bitstream`](cases/lossy-segment-no-map.bitstream)

Segmentation on, quantizers given, but no segment map: every macroblock is
segment 0.

use_segment without update_map. No per-macroblock segment bits are read, and
only dqm[0] is ever selected, but the other three are still built.

### [`lossy-segment-prob-extremes.webp`](files/lossy-segment-prob-extremes.webp) -- ok -- from [`lossy-segment-prob-extremes.bitstream`](cases/lossy-segment-prob-extremes.bitstream)

Segment probabilities of 0 and 255, and loop-filter updates at both ends of
their range.

A tree probability of 0 or 255 makes one branch of the segment id free and the
other maximally expensive; -63 and 63 are the ends of the 6-bit signed loop-
filter update. All four macroblocks carry a different segment id, so every
branch of the tree is taken under those probabilities.

### [`lossy-segment-quant-extremes.webp`](files/lossy-segment-quant-extremes.webp) -- ok -- from [`lossy-segment-quant-extremes.bitstream`](cases/lossy-segment-quant-extremes.bitstream)

Segment quantizers at 127, -127, 0 and absent.

clip(q, 127) at both ends of VP8ParseQuant(), and the difference between a
field written as zero and one left out, which libwebp's own encoder cannot
express.

## Lossy: loop filter

The in-loop deblocking filter: simple or normal, its level and sharpness, and
the per-reference and per-mode deltas.

### [`lossy-filter-lf-delta-extremes.webp`](files/lossy-filter-lf-delta-extremes.webp) -- ok -- from [`lossy-filter-lf-delta-extremes.bitstream`](cases/lossy-filter-lf-delta-extremes.bitstream)

Loop-filter mode deltas at -63 and 63.

The mode deltas are applied per macroblock coding type, and only the first of
the four is ever used by an intra frame. lossy-filter-lf-delta writes -20 and
31; this writes both ends of the 6-bit signed field, against a filter level
that leaves room to move.

### [`lossy-filter-lf-delta.webp`](files/lossy-filter-lf-delta.webp) -- ok -- from [`lossy-filter-lf-delta.bitstream`](cases/lossy-filter-lf-delta.bitstream)

Loop-filter deltas: 63 and -63 on the reference deltas, and a delta on the 4x4
mode.

The mode_lf_delta[0] path for 4x4-coded macroblocks and the ref_lf_delta that
only inter frames would use. cwebp writes four zero flags and one i4x4 delta
(syntax_enc.c:226), never these.

### [`lossy-filter-normal-max.webp`](files/lossy-filter-normal-max.webp) -- ok -- from [`lossy-filter-normal-max.bitstream`](cases/lossy-filter-normal-max.bitstream)

The normal loop filter at level 63, sharpness 0.

filter_type 2 with the widest possible filter, so both the 8-pixel and the
4-pixel variants run at their limits.

### [`lossy-filter-sharpness-4.webp`](files/lossy-filter-sharpness-4.webp) -- ok -- from [`lossy-filter-sharpness-4.bitstream`](cases/lossy-filter-sharpness-4.bitstream)

Sharpness 4, the last level that halves the interior limit.

PrecomputeFilterStrengths() shifts the interior limit right by one for
sharpness 1 to 4 and by two for 5 to 7, then clamps it to 9 - sharpness. These
two files sit on either side of that boundary; 0, 1, 2, 3 and 7 were already
covered.

### [`lossy-filter-sharpness-5.webp`](files/lossy-filter-sharpness-5.webp) -- ok -- from [`lossy-filter-sharpness-5.bitstream`](cases/lossy-filter-sharpness-5.bitstream)

Sharpness 5, the first level that quarters it.

PrecomputeFilterStrengths() shifts the interior limit right by one for
sharpness 1 to 4 and by two for 5 to 7, then clamps it to 9 - sharpness. These
two files sit on either side of that boundary; 0, 1, 2, 3 and 7 were already
covered.

### [`lossy-filter-simple-max.webp`](files/lossy-filter-simple-max.webp) -- ok -- from [`lossy-filter-simple-max.bitstream`](cases/lossy-filter-simple-max.bitstream)

The simple loop filter at level 63 and sharpness 7.

filter_type 1, and the sharpness clamp on the interior limit. cwebp picks its
own level and never gets near the top of the range.

## Lossy: quantizer

The frame quantizer index and the five deltas around it, one per plane and
coefficient kind, with clamps that are not all the same.

### [`lossy-quant-deltas-mirrored.webp`](files/lossy-quant-deltas-mirrored.webp) -- ok -- from [`lossy-quant-deltas-mirrored.bitstream`](cases/lossy-quant-deltas-mirrored.bitstream)

The five quantizer deltas at the ends lossy-quant-deltas does not use.

Each of the five 4-bit signed fields written at its other extreme, so between
the two files every one of them is seen at both -15 and +15.

### [`lossy-quant-deltas.webp`](files/lossy-quant-deltas.webp) -- ok -- from [`lossy-quant-deltas.bitstream`](cases/lossy-quant-deltas.bitstream)

All five quantizer deltas present, at the ends of their 4-bit range.

dqy1_dc, dqy2_dc, dqy2_ac, dquv_dc and dquv_ac at once. cwebp writes only the
two chroma deltas, and only small ones.

### [`lossy-quant-dequant-overflow.webp`](files/lossy-quant-dequant-overflow.webp) -- ok -- from [`lossy-quant-dequant-overflow.bitstream`](cases/lossy-quant-dequant-overflow.bitstream)

A coefficient of 2114 at the coarsest quantizer, so the dequantized value does
not fit the int16 it is stored in.

"out[kZigzag[n]] = VP8GetSigned(br, v) * dq[n > 0]" with v * dq of 600376
against an int16_t destination. Nothing an encoder can produce, and worth
watching under a sanitizer.

### [`lossy-quant-max.webp`](files/lossy-quant-max.webp) -- ok -- from [`lossy-quant-max.bitstream`](cases/lossy-quant-max.bitstream)

The frame quantizer at 127, the coarsest.

The last entry of both quantizer tables, and the top of the clip() range that
the delta cases push against.

### [`lossy-quant-min.webp`](files/lossy-quant-min.webp) -- ok -- from [`lossy-quant-min.bitstream`](cases/lossy-quant-min.bitstream)

The frame quantizer at 0, the finest the format allows.

kDcTable[0] and kAcTable[0] are both 4, and the Y2 AC quantizer hits the "if
(m->y2_mat[1] < 8) m->y2_mat[1] = 8" floor in VP8ParseQuant().

### [`lossy-quant-uv-dc-clamp.webp`](files/lossy-quant-uv-dc-clamp.webp) -- ok -- from [`lossy-quant-uv-dc-clamp.bitstream`](cases/lossy-quant-uv-dc-clamp.bitstream)

A chroma DC quantizer index pushed past 117, where it is clamped rather than at
127.

The odd "clip(q + dquv_dc, 117)" in VP8ParseQuant(), a limit the four other
planes do not have. base 110 plus 15 lands on 125, so the clamp is what makes
the difference.

## Lossy: coefficient probabilities

The 1056 probabilities that drive the coefficient coder, each one optionally
replaced in the frame header, plus the skip probability.

### [`lossy-proba-all-updated.webp`](files/lossy-proba-all-updated.webp) -- ok -- from [`lossy-proba-all-updated.bitstream`](cases/lossy-proba-all-updated.bitstream)

Every one of the 1056 coefficient probabilities updated.

The largest partition 0 the corpus has: 1056 set flags, each followed by a raw
byte. cwebp updates a handful at most.

### [`lossy-proba-one-update.webp`](files/lossy-proba-one-update.webp) -- ok -- from [`lossy-proba-one-update.bitstream`](cases/lossy-proba-one-update.bitstream)

A single coefficient probability updated, the other 1055 left alone.

One 1 among the update flags of VP8ParseProba(), on the [i4-AC][band 0][ctx 0]
EOB probability, which the coefficients below then use.

### [`lossy-proba-refresh-and-skip-zero.webp`](files/lossy-proba-refresh-and-skip-zero.webp) -- ok -- from [`lossy-proba-refresh-and-skip-zero.bitstream`](cases/lossy-proba-refresh-and-skip-zero.bitstream)

The entropy-refresh bit set, and a skip probability of 0.

refresh_entropy_probs is read and dropped by libwebp, so no other file sets it.
A prob_skip_false of 0 says no macroblock is skipped while two are, which is
the most expensive way the flag can be coded and the bottom of its range.

### [`lossy-proba-skip-extremes.webp`](files/lossy-proba-skip-extremes.webp) -- ok -- from [`lossy-proba-skip-extremes.bitstream`](cases/lossy-proba-skip-extremes.bitstream)

A skip probability of 255 with nothing skipped, and the flag itself written
out.

use_skip_proba with a probability that says every macroblock should be skipped
while none is, which is the most expensive way to code it.

### [`lossy-proba-zero.webp`](files/lossy-proba-zero.webp) -- ok -- from [`lossy-proba-zero.bitstream`](cases/lossy-proba-zero.bitstream)

Coefficient probabilities of 0 and of 255, the ends of the range.

A probability of 0 makes the boolean split 0, so the "bit is 0" branch
renormalizes from an empty range. Legal, and never emitted.

## Lossy: prediction modes

The 16x16 and 4x4 luma modes and the chroma modes, and the neighbour-indexed
probability table the 4x4 modes are coded with.

### [`lossy-mode-i16-all-four.webp`](files/lossy-mode-i16-all-four.webp) -- ok -- from [`lossy-mode-i16-all-four.bitstream`](cases/lossy-mode-i16-all-four.bitstream)

The four 16x16 luma modes, one per macroblock.

DC_PRED, V_PRED, H_PRED and TM_PRED through the hardcoded tree at probabilities
156, 128 and 163, and all four 16x16 reconstructions.

### [`lossy-mode-i4-all-ten.webp`](files/lossy-mode-i4-all-ten.webp) -- ok -- from [`lossy-mode-i4-all-ten.bitstream`](cases/lossy-mode-i4-all-ten.bitstream)

All ten 4x4 luma modes inside one macroblock, twice over.

Every leaf of the B_PRED tree, and every 4x4 predictor including the ones that
need the four pixels above and to the right.

### [`lossy-mode-i4-context.webp`](files/lossy-mode-i4-context.webp) -- ok -- from [`lossy-mode-i4-context.bitstream`](cases/lossy-mode-i4-context.bitstream)

Four B_PRED macroblocks whose 4x4 modes walk the [above][left] probability
table.

kBModesProba is indexed by the two neighbouring modes, so this is the only way
to reach entries other than [B_DC][B_DC]. The top row of one macroblock is the
context of the one below it.

### [`lossy-mode-mixed.webp`](files/lossy-mode-mixed.webp) -- ok -- from [`lossy-mode-mixed.bitstream`](cases/lossy-mode-mixed.bitstream)

16x16 and 4x4 macroblocks alternating, in both directions.

A 16x16 macroblock writes its mode into all four of its neighbours' 4x4
contexts, so this is what checks that the two mode paths agree on the context
they leave behind.

### [`lossy-mode-uv-all-four.webp`](files/lossy-mode-uv-all-four.webp) -- ok -- from [`lossy-mode-uv-all-four.bitstream`](cases/lossy-mode-uv-all-four.bitstream)

The four chroma modes, one per macroblock.

The uvmode tree at probabilities 142, 114 and 183, and the 8x8 chroma
predictors.

## Lossy: coefficients

The token coder of section 13: magnitudes and their escape categories, end-of-
block, zero runs, and the four coefficient types.

### [`lossy-coeff-all-types.webp`](files/lossy-coeff-all-types.webp) -- ok -- from [`lossy-coeff-all-types.bitstream`](cases/lossy-coeff-all-types.bitstream)

All four coefficient types in one macroblock, and both luma types across two.

The Y2 block (type 1), luma after Y2 starting at position 1 (type 0), chroma
(type 2) and luma with its own DC (type 3). No single macroblock can reach all
four, since types 0 and 3 are exclusive.

### [`lossy-coeff-bands-chroma.webp`](files/lossy-coeff-bands-chroma.webp) -- ok -- from [`lossy-coeff-bands-chroma.bitstream`](cases/lossy-coeff-bands-chroma.bitstream)

The same sweep across the four blocks of a chroma plane.

Type 2 (chroma). The 2x2 layout means block 3 is the only one with both a left
and an above neighbour, so it is the only way to read the chroma band 0 at
context 2. The three token classes drive the context of the next position: a
zero gives 0, a +-1 gives 1, anything larger gives 2. A block of each, walked
to position 15, reads every band at that class, and the blocks are placed so
their neighbour contexts are 0, 1 and 2 in turn -- which is the only way to
reach band 0, the one that is never a token's successor.

### [`lossy-coeff-bands-i16.webp`](files/lossy-coeff-bands-i16.webp) -- ok -- from [`lossy-coeff-bands-i16.bitstream`](cases/lossy-coeff-bands-i16.bitstream)

The same sweep for the two block kinds a 16x16 macroblock has: the Y2 block and
the luma blocks that follow it.

Types 1 (Y2) and 0 (luma after Y2). The luma blocks start at position 1, so
band 0 is unreachable for them and band 1 is what their neighbour context
selects. One Y2 block per macroblock means the four macroblocks are what give
it contexts 0, 1, 1 and 2. The three token classes drive the context of the
next position: a zero gives 0, a +-1 gives 1, anything larger gives 2. A block
of each, walked to position 15, reads every band at that class, and the blocks
are placed so their neighbour contexts are 0, 1 and 2 in turn -- which is the
only way to reach band 0, the one that is never a token's successor.

### [`lossy-coeff-bands-i4.webp`](files/lossy-coeff-bands-i4.webp) -- ok -- from [`lossy-coeff-bands-i4.bitstream`](cases/lossy-coeff-bands-i4.bitstream)

Three 4x4 luma blocks that sweep every coefficient band, one per context.

Type 3 (luma with its own DC). The three token classes drive the context of the
next position: a zero gives 0, a +-1 gives 1, anything larger gives 2. A block
of each, walked to position 15, reads every band at that class, and the blocks
are placed so their neighbour contexts are 0, 1 and 2 in turn -- which is the
only way to reach band 0, the one that is never a token's successor.

### [`lossy-coeff-cat3.webp`](files/lossy-coeff-cat3.webp) -- ok -- from [`lossy-coeff-cat3.bitstream`](cases/lossy-coeff-cat3.bitstream)

Category-3 coefficients: 11 to 18, three extra bits each.

The first escape category, kCat3 = {173, 148, 140}, at both ends of its range.

### [`lossy-coeff-cat4.webp`](files/lossy-coeff-cat4.webp) -- ok -- from [`lossy-coeff-cat4.bitstream`](cases/lossy-coeff-cat4.bitstream)

Category-4 coefficients: 19 to 34, four extra bits.

kCat4, and the p[9] branch that separates category 4 from category 3.

### [`lossy-coeff-cat5.webp`](files/lossy-coeff-cat5.webp) -- ok -- from [`lossy-coeff-cat5.bitstream`](cases/lossy-coeff-cat5.bitstream)

Category-5 coefficients: 35 to 66, five extra bits.

kCat5, reached through p[8] = 1 and p[10] = 0, which is a different pair of
probabilities from the lower categories.

### [`lossy-coeff-cat6-max.webp`](files/lossy-coeff-cat6-max.webp) -- ok -- from [`lossy-coeff-cat6-max.bitstream`](cases/lossy-coeff-cat6-max.bitstream)

The largest coefficient the format can encode: 2114.

3 + (8 << 3) + 2047, every one of kCat6's eleven extra bits set. One more would
wrap round inside the same eleven bits.

### [`lossy-coeff-cat6.webp`](files/lossy-coeff-cat6.webp) -- ok -- from [`lossy-coeff-cat6.bitstream`](cases/lossy-coeff-cat6.bitstream)

Category-6 coefficients: 67 upwards, eleven extra bits.

kCat6, the longest escape. 67 is the first value it can hold.

### [`lossy-coeff-context.webp`](files/lossy-coeff-context.webp) -- ok -- from [`lossy-coeff-context.bitstream`](cases/lossy-coeff-context.bitstream)

Neighbouring blocks with and without coefficients, so that every context value
from 0 to 2 is used.

The "ctx = left + top" that picks one of the three probability sets. Context 2
needs both neighbours non-empty, which only happens a few blocks into a
macroblock.

### [`lossy-coeff-empty-blocks.webp`](files/lossy-coeff-empty-blocks.webp) -- ok -- from [`lossy-coeff-empty-blocks.bitstream`](cases/lossy-coeff-empty-blocks.bitstream)

Every one of the 25 blocks empty, but the macroblock not skipped.

An end-of-block at the very first position of every block. Without a skip
probability in the frame there is no other way to say it, and ParseResiduals()
still runs in full.

### [`lossy-coeff-full-block.webp`](files/lossy-coeff-full-block.webp) -- ok -- from [`lossy-coeff-full-block.bitstream`](cases/lossy-coeff-full-block.bitstream)

A block with all sixteen coefficients non-zero, so the loop ends by running out
of positions rather than on an end-of-block.

The "n == 16" exit of GetCoeffs(), which is the only way out that does not read
an end-of-block bit, and every band from 0 to 7.

### [`lossy-coeff-medium-magnitudes.webp`](files/lossy-coeff-medium-magnitudes.webp) -- ok -- from [`lossy-coeff-medium-magnitudes.bitstream`](cases/lossy-coeff-medium-magnitudes.bitstream)

Coefficients of 5 through 10, coded with the fixed probabilities 159, 165 and
145.

The two branches under p[6]: 5 and 6 through probability 159, then 7 to 10
through 165 and 145. Those three constants appear nowhere else.

### [`lossy-coeff-small-magnitudes.webp`](files/lossy-coeff-small-magnitudes.webp) -- ok -- from [`lossy-coeff-small-magnitudes.bitstream`](cases/lossy-coeff-small-magnitudes.bitstream)

Coefficients of 1, 2, 3 and 4, the magnitudes with their own tree branches.

The v == 1 shortcut, then the p[4]/p[5] pair of GetLargeValue() that separates
2 from 3 and 4.

### [`lossy-coeff-wht-full.webp`](files/lossy-coeff-wht-full.webp) -- ok -- from [`lossy-coeff-wht-full.bitstream`](cases/lossy-coeff-wht-full.bitstream)

A Y2 block with more than one coefficient, next to one with only a DC.

The two halves of the Y2 branch in ParseResiduals(): "nz > 1" runs the full
VP8TransformWHT, while a lone DC takes the inlined "(dc[0] + 3) >> 3" shortcut.

### [`lossy-coeff-zero-runs.webp`](files/lossy-coeff-zero-runs.webp) -- ok -- from [`lossy-coeff-zero-runs.bitstream`](cases/lossy-coeff-zero-runs.bitstream)

Single coefficients at positions 15, 12, 8 and 1, each behind a run of zeros.

The inner "sequence of zero coeffs" loop of GetCoeffs(), which walks the band
table without reading an end-of-block bit, and reaches position 15 in band 7.

## Lossy: skipped macroblocks

The per-macroblock skip flag, which drops the residual entirely and clears the
neighbouring non-zero flags -- almost all of them.

### [`lossy-skip-all.webp`](files/lossy-skip-all.webp) -- ok -- from [`lossy-skip-all.bitstream`](cases/lossy-skip-all.bitstream)

Every macroblock skipped.

The skip branch of VP8DecodeMB(), which clears the neighbour flags without
reading a single coefficient. Every token partition holds nothing but its own
padding.

### [`lossy-skip-i4x4-nz-dc.webp`](files/lossy-skip-i4x4-nz-dc.webp) -- ok -- from [`lossy-skip-i4x4-nz-dc.bitstream`](cases/lossy-skip-i4x4-nz-dc.bitstream)

A skipped 4x4 macroblock between two 16x16 ones that both carry a Y2 block.

VP8DecodeMB() clears nz_dc only when the skipped macroblock is not 4x4-coded,
so the second 16x16 macroblock sees the Y2 context left behind by the first.
Reproducing that quirk is the whole point.

### [`lossy-skip-mixed.webp`](files/lossy-skip-mixed.webp) -- ok -- from [`lossy-skip-mixed.bitstream`](cases/lossy-skip-mixed.bitstream)

Skipped and coded macroblocks alternating, with a skip probability of 1.

The most expensive coding of the skip flag: the probability says almost nothing
is skipped while half of it is. Also checks that a skipped macroblock leaves
its neighbours' contexts cleared.

## Lossy: token partitions

A lossy frame may carry 1, 2, 4 or 8 token partitions, macroblock row r being
read from partition r & (n - 1). cwebp does not expose config.partitions and
libwebp forces it back to 1 whenever the token path is used (webp_enc.c:124),
so none of this is reachable through the tools.

### [`lossy-parts-2-wrap.webp`](files/lossy-parts-2-wrap.webp) -- ok -- from [`lossy-parts-2-wrap.bitstream`](cases/lossy-parts-2-wrap.bitstream)

Four rows over two partitions, so each partition holds two non- adjacent rows.

The "mb_y & num_parts_minus_one" wrap. Each bit reader is left in the middle of
a row and resumed two rows later.

### [`lossy-parts-8-rows.webp`](files/lossy-parts-8-rows.webp) -- ok -- from [`lossy-parts-8-rows.bitstream`](cases/lossy-parts-8-rows.bitstream)

Eight macroblock rows over eight token partitions, one row each, every row
different.

Row r is read from partition r & 7, so a decoder that got the mapping wrong
would still parse but decode the rows in the wrong order. The pixel hash is
what catches that.

### [`lossy-parts-last-empty.webp`](files/lossy-parts-last-empty.webp) -- reject -- from [`lossy-parts-last-empty.bitstream`](cases/lossy-parts-last-empty.bitstream)

Four partitions whose declared sizes leave nothing for the last one.

The last partition is not declared anywhere; it gets whatever is left, which
the clamp on the third size has already reduced to nothing. ParsePartitions()
notices that part_start is no longer inside the buffer and refuses the frame
rather than handing out an empty reader.

### [`lossy-parts-size-past-end.webp`](files/lossy-parts-size-past-end.webp) -- reject -- from [`lossy-parts-size-past-end.bitstream`](cases/lossy-parts-size-past-end.bitstream)

Four partitions, the first declaring 16 MB of data.

The "if (psize > size_left) psize = size_left" clamp. Partition 0 swallows the
rest of the frame, so part_start reaches the end of the buffer and
ParsePartitions() returns NOT_ENOUGH_DATA: the frame is refused there, before a
single macroblock is decoded.

### [`lossy-parts-table-too-small.webp`](files/lossy-parts-table-too-small.webp) -- reject -- from [`lossy-parts-table-too-small.bitstream`](cases/lossy-parts-table-too-small.bitstream)

Eight partitions declared in a frame with only ten bytes left for the twenty-
one-byte size table.

The "size < 3 * last_part" test in ParsePartitions(), the one failure there
that cannot be papered over: the sizes themselves are unreadable, so there is
nothing to clamp. Confirmed to reach parts_size_table_truncated and no further.

## Lossy: truncation

Frames that stop early, at each of the places the decoder can notice: inside
partition 0, inside the macroblock modes, and inside the token data.

### [`lossy-truncated-header.webp`](files/lossy-truncated-header.webp) -- reject -- from [`lossy-truncated-header.bitstream`](cases/lossy-truncated-header.bitstream)

Partition 0 cut to two bytes, which is enough for the segment header and not
for the filter header.

The "cannot parse filter header" exit of VP8GetHeaders(). ParseFilterHeader()
returns !br->eof, and this is the only file that makes it the one to fail: one
byte less and the segment header goes first, one more and the failure moves to
the macroblock modes.

### [`lossy-truncated-modes.webp`](files/lossy-truncated-modes.webp) -- reject -- from [`lossy-truncated-modes.bitstream`](cases/lossy-truncated-modes.bitstream)

Partition 0 long enough for the whole frame header and not for the macroblock
modes that follow it.

"Premature end-of-partition0 encountered", out of the !dec->br.eof that
VP8ParseIntraModeRow() checks once per macroblock row rather than once per
macroblock. The token partitions are untouched, so this is the mode data
failing on its own.

### [`lossy-truncated-short-modes.webp`](files/lossy-truncated-short-modes.webp) -- ok -- from [`lossy-truncated-short-modes.bitstream`](cases/lossy-truncated-short-modes.bitstream)

Mode data for 15 macroblocks in a frame whose dimensions call for 16.

The missing macroblock is decoded out of partition 0's padding without the
reader ever running out, so the frame is accepted and the last macroblock
decodes to whatever the padding happens to say. Nothing checks that the mode
data is as long as the frame claims to be. Reading the file back gives 16
macroblocks, so it is the one case here that cannot be disassembled and
reassembled unchanged.

### [`lossy-truncated-tokens.webp`](files/lossy-truncated-tokens.webp) -- reject -- from [`lossy-truncated-tokens.bitstream`](cases/lossy-truncated-tokens.bitstream)

Partition 0 intact, the token partition cut in half.

"Premature end-of-file encountered" out of VP8DecodeMB(), the other truncation
path. The modes all parse, so the decoder gets several rows in before it fails.

## Lossy: partition sizes, from real encodes

The four files behind these are genuine encoder output, made through the
encoder API rather than cwebp, and the broken ones rewrite the raw size table
that follows partition 0. They carry 256 macroblocks of real coefficients,
which the assembled cases do not.

### [`lossy-1-partitions.webp`](files/lossy-1-partitions.webp) -- ok

A single token partition: the default, and the control for the others.

Same encode settings as the 2/4/8 files, so a size or hash difference against
them is entirely the partitioning.

### [`lossy-2-partitions.webp`](files/lossy-2-partitions.webp) -- ok

A plain 2-partition lossy frame.

cwebp never emits this: config.partitions is API-only and is forced back to 1
for method >= 3 unless low_memory is set.

### [`lossy-4-partitions.webp`](files/lossy-4-partitions.webp) -- ok

A plain 4-partition lossy frame.

Same, with the size table holding three entries.

### [`lossy-8-partitions.webp`](files/lossy-8-partitions.webp) -- ok

A plain 8-partition lossy frame, the maximum the 2-bit field allows.

MAX_NUM_PARTITIONS. Seven 3-byte size-table entries, and eight independent bit-
readers in the decoder.

### [`lossy-8-partitions-size-overflow.webp`](files/lossy-8-partitions-size-overflow.webp) -- reject

Eight partitions whose first declared size is 0xffffff, far past the data.

Hits the "if (psize > size_left) psize = size_left" clamp in ParsePartitions():
partition 0 swallows the whole remainder and the other seven get zero-length
readers.

### [`lossy-8-partitions-zero-sizes.webp`](files/lossy-8-partitions-zero-sizes.webp) -- reject

Eight partitions all declared as zero bytes long.

Every token partition but the last is empty, so the last one is handed the
whole remainder. Legal to parse, garbage to decode.

### [`lossy-8-partitions-sizes-sum-past-end.webp`](files/lossy-8-partitions-sizes-sum-past-end.webp) -- reject

Eight partitions whose declared sizes add up to more than the chunk holds.

The clamp fires part-way through the loop, so later partitions get zero-length
readers while earlier ones look valid.

### [`lossy-combo-all-features.webp`](files/lossy-combo-all-features.webp) -- ok -- from [`lossy-combo-all-features.bitstream`](cases/lossy-combo-all-features.bitstream)

Every optional tool switched on in one frame at once.

Segmentation with a map, the loop filter with per-segment strengths and mode
deltas, four token partitions, the skip flag, a probability update, and both
macroblock types. Each of those has a file of its own that isolates it; this is
the one that makes them interact, and it is what covers the eight pairs of
features that never met anywhere else.

---

180 files, 46988 bytes total. Rebuild with `generate.py`: it writes the
lossless cases itself with `vp8l.py`, and assembles everything in `cases/`
through `webp_asm.py`, which hands the frame to `vp8_asm.py` and that to
`vp8.py`.
