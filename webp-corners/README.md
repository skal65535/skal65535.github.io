# WebP stress bitstreams

Small WebP files that exercise corners of the format a normal encoder never
emits, one layer of it at a time.

**They are written, not captured.** Each is a text file naming bitstream
fields, one per line. Nothing is validated on the way, so a case can say
what no encoder ever would.

The names are the specifications' own:
[RFC 6386](https://www.rfc-editor.org/rfc/rfc6386.html) for the lossy
bitstream, [RFC 9649](https://www.rfc-editor.org/rfc/rfc9649.html) for the
container.

What they reach:

* **82 lossless (VP8L) streams** -- Huffman codes and the code-length code that
  describes them, colour caches, back-references, the four transforms, and the
  entropy image that changes codes mid-row.

* **81 lossy VP8 frames** -- every field of the frame header, the segmentation,
  loop-filter and quantizer records, the coefficient token coder out to its
  escape categories, and the 1, 2, 4 or 8 token partitions.

* **19 RIFF containers** -- the VP8X chunk and the canvas it declares, the
  optional chunks a decoder must step over by their declared length alone, and
  headers that lie about what sits behind them.

* **33 alpha chunks** -- the plane stored a byte per pixel through each of the
  four filters, and the plane compressed by the lossless coder in an 8-bit mode
  nothing else here reaches.

* **48 animations** -- frame position, duration, disposal and blending,
  composed over a canvas one frame at a time. No still decoder will open one of
  these at all.

Every file carries a verdict, which is what a decoder must do with it:

* **ok** -- must decode, and must keep decoding to the same pixels. Several
  are not something cwebp can produce, so nothing else pins the behaviour.

* **reject** -- must fail cleanly and report a status, with no crash and no
  out-of-bounds access. Which status varies: a malformed Huffman code gives
  BITSTREAM_ERROR, a short partition table gives NOT_ENOUGH_DATA.

A file read by more than one decoder carries more than one verdict, and
they do not always agree. An animation reads `reject, anim_dump ok`: a
still decoder refuses anything claiming animation before it looks at a
frame, so that half says nothing about the file and the second half is the
one that does. `webpinfo` and the incremental decoder are named where they
disagree, and `check.sh` holds all of them to it.

## What to run

The first two are the point: they run every file here through a decoder and say
whether it behaved. The rest rebuild the corpus or check the tools that write
it.

| file | what it is |
| --- | --- |
| [`check.sh`](check.sh) | Decodes every file and checks the verdict and the pixels, through `dwebp` or -- for the animations -- `$ANIM_DUMP` and `$WEBPINFO`. The one to run. |
| [`asan_sweep.sh`](asan_sweep.sh) | The same, in 14 output modes under a sanitizer build. Point `$ASAN_DWEBP` at one. |
| [`coverage.sh`](coverage.sh) | How much of `src/dec` and `src/demux` these files reach, from an instrumented build in a throwaway worktree. Also reports how much further a caller can get with the same files, which is how you tell a gap in the corpus from a path no bitstream controls. |
| [`generate.py`](generate.py) | Rebuilds `files/` from `cases/`, and writes `expected.txt`, this README, `SYNTAX.md`, `src/README.md`, and an `index.html` for each directory that needs one. |
| [`make_hashes.sh`](make_hashes.sh) | Rewrites `hashes.txt`, once the new output is known to be right. |
| [`make_coverage.sh`](make_coverage.sh) | Rebuilds `coverage.txt` in a throwaway worktree. |
| [`vp8_selftest.py`](vp8_selftest.py) | Checks the assemblers themselves, not the corpus: only needed if you change them. |

## The code

The layers underneath, in **[`src/`](src)**, which has its own README
describing how they fit together.

| file | what it is |
| --- | --- |
| [`src/vp8l.py`](src/vp8l.py) | VP8L lossless bitstream writer: bit packing, canonical Huffman codes, prefix coding, sub-images. |
| [`src/vp8l_asm.py`](src/vp8l_asm.py) | Assembles a lossless image from a text case. Its docstring is the format. |
| [`src/vp8.py`](src/vp8.py) | VP8 lossy bitstream writer: the boolean coder, the frame header, the mode trees, the coefficients. |
| [`src/vp8_asm.py`](src/vp8_asm.py) | Assembles a lossy frame from a text case, in RFC 6386's field names. Its docstring is the format. |
| [`src/webp_asm.py`](src/webp_asm.py) | Wraps either in a RIFF container, in RFC 9649's field names, and picks which assembler a case belongs to. |
| [`src/vp8_dis.py`](src/vp8_dis.py) | The other direction for a lossy frame. `--check` round trips one against libwebp. |
| [`src/vp8l_dis.py`](src/vp8l_dis.py) | The other direction for a lossless image, the same way. |
| [`src/webp_dis.py`](src/webp_dis.py) | The other direction for a whole file: chunks, animation frames and alpha planes, delegating each image to one of those two. |
| [`src/grammar.py`](src/grammar.py) | Every keyword and the range of every value, as data. `SYNTAX.md` is generated from it. |
| [`src/vp8_tables.py`](src/vp8_tables.py) | The VP8 constant tables, extracted from libwebp. |
| [`src/make_vp8_tables.py`](src/make_vp8_tables.py) | Extracts them, so they are never retyped. |
| [`src/lossy_parts.py`](src/lossy_parts.py) | The multi-partition lossy cases, patched from `sources/`. |
| [`src/make_partition_sources.c`](src/make_partition_sources.c) | Rebuilds `sources/`: cwebp cannot emit more than one token partition. |
| [`src/probes.py`](src/probes.py) | The `fprintf` probes `make_coverage.sh` patches in. |
| [`src/api_sweep.c`](src/api_sweep.c) | Every decoding entry point libwebp exports, for `coverage.sh`: the incremental decoder fed a few bytes at a time, caller-allocated buffers, the colorspaces dwebp cannot ask for, the demuxer's iterators. |
| [`src/check_refs.py`](src/check_refs.py) | Checks that the source lines the notes point at still say what the notes claim. |

## The data

| file | what it is |
| --- | --- |
| [`HOWTO.md`](HOWTO.md) | How to write a case, read a real file back into one, and add one here. |
| [`SYNTAX.md`](SYNTAX.md) | The whole case syntax, generated from `src/grammar.py`. |
| [`expected.txt`](expected.txt) | Name and expected verdict, one line per file. |
| [`hashes.txt`](hashes.txt) | SHA-256 of each decoding file's `-pam` output, so a silent change in decoded pixels fails too. |
| [`coverage.txt`](coverage.txt) | Which decoder path each file actually reached. |
| [`refs.txt`](refs.txt) | What each line a note points at said when it was written. |
| [`COPYING`](COPYING) | BSD 3-clause, the same as libwebp. |

## The bitstreams

The whole set lives in **[`files/`](files)**, which lists each one with its
size and expected verdict, and the text they are assembled from is in
**[`cases/`](cases)**. Each row below links to both: the line is what the case
calls itself, and the case says the rest -- which decoder path it reaches and
why that is worth a file.

| Group | Files | must decode | must be rejected |
| --- | ---: | ---: | ---: |
| Whole lossless images | 2 | 2 | 0 |
| Simple codes | 9 | 6 | 3 |
| The code-length code | 15 | 9 | 6 |
| Meta Huffman / entropy image | 7 | 6 | 1 |
| Color cache | 5 | 3 | 2 |
| Sub-images | 9 | 5 | 4 |
| Palette packing | 6 | 6 | 0 |
| Transforms | 5 | 4 | 1 |
| Back-references | 10 | 8 | 2 |
| Predictor modes | 7 | 7 | 0 |
| Frame header | 7 | 3 | 4 |
| The RIFF container | 19 | 9 | 10 |
| Animation | 48 | 27 | 21 |
| The alpha chunk | 33 | 23 | 10 |
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
| **total** | **263** | **182** | **81** |

### Whole lossless images

One image with nothing optional in it, and one with all four transforms, a
colour cache, an entropy image and every kind of pixel item. Every other
lossless file here sets one field of those two to something an encoder would
not.

| file | | what it is |
| --- | --- | --- |
| [`lossless-all-features`](files/lossless-all-features.webp) [txt](cases/lossless-all-features.txt) | ok | Every optional part of the lossless format in one image |
| [`lossless-plain`](files/lossless-plain.webp) [txt](cases/lossless-plain.txt) | ok | An ordinary lossless image, with none of the corners |

### Simple codes

The 1-or-2-symbol shorthand a Huffman code can take. Its symbols are read as
raw 8-bit values and are never checked against the alphabet size, so this is
where a stream can say things an encoder cannot.

| file | | what it is |
| --- | --- | --- |
| [`simple-dist-1sym-oob`](files/simple-dist-1sym-oob.webp) [txt](cases/simple-dist-1sym-oob.txt) | reject | Distance code: simple form, single symbol 255, alphabet_size is 40 |
| [`simple-dist-2sym-both-oob`](files/simple-dist-2sym-both-oob.webp) [txt](cases/simple-dist-2sym-both-oob.txt) | reject | Distance code: both simple-form symbols out of range (200, 201) |
| [`simple-dist-2sym-duplicate`](files/simple-dist-2sym-duplicate.webp) [txt](cases/simple-dist-2sym-duplicate.txt) | ok | Distance code: simple form declaring 2 symbols that are the same (5, 5) |
| [`simple-dist-2sym-first-oob`](files/simple-dist-2sym-first-oob.webp) [txt](cases/simple-dist-2sym-first-oob.txt) | ok | Distance code: simple form, 2 symbols, the first one 200 >= alphabet_size 40 |
| [`simple-dist-2sym-second-oob`](files/simple-dist-2sym-second-oob.webp) [txt](cases/simple-dist-2sym-second-oob.txt) | ok | Distance code: simple form, 2 symbols, the second one 200 >= 40 |
| [`simple-dist-sym-39-last-valid`](files/simple-dist-sym-39-last-valid.webp) [txt](cases/simple-dist-sym-39-last-valid.txt) | ok | Distance code: single symbol 39, the last in-range value |
| [`simple-dist-sym-40-first-oob`](files/simple-dist-sym-40-first-oob.webp) [txt](cases/simple-dist-sym-40-first-oob.txt) | reject | Distance code: single symbol 40, the first out-of-range value |
| [`simple-green-1bit-symbol`](files/simple-green-1bit-symbol.webp) [txt](cases/simple-green-1bit-symbol.txt) | ok | Green code: simple form with first_symbol_len_code = 0, so the symbol is 1 bit wide |
| [`simple-green-2sym-1bit-each`](files/simple-green-2sym-1bit-each.webp) [txt](cases/simple-green-2sym-1bit-each.txt) | ok | Green code with two real symbols, so every pixel costs exactly 1 bit |

### The code-length code

The Huffman code that describes the lengths of another Huffman code, plus its
repeat escapes (16, 17, 18) and the optional max_symbol field. cwebp only ever
emits a narrow slice of this.

| file | | what it is |
| --- | --- | --- |
| [`codelen-all-zero-lengths`](files/codelen-all-zero-lengths.webp) [txt](cases/codelen-all-zero-lengths.txt) | reject | A code-length stream that assigns length 0 to every symbol |
| [`codelen-depth-15`](files/codelen-depth-15.webp) [txt](cases/codelen-depth-15.txt) | ok | A green code containing a symbol of depth 15, MAX_ALLOWED_CODE_LENGTH |
| [`codelen-incomplete`](files/codelen-incomplete.webp) [txt](cases/codelen-incomplete.txt) | reject | A code whose lengths leave the tree incomplete (two symbols of depth 2) |
| [`codelen-max-symbol-early-stop`](files/codelen-max-symbol-early-stop.webp) [txt](cases/codelen-max-symbol-early-stop.txt) | ok | Code-length stream with an explicit max_symbol far below the alphabet size |
| [`codelen-max-symbol-too-big`](files/codelen-max-symbol-too-big.webp) [txt](cases/codelen-max-symbol-too-big.txt) | reject | Explicit max_symbol greater than the alphabet size |
| [`codelen-num-codes-19`](files/codelen-num-codes-19.webp) [txt](cases/codelen-num-codes-19.txt) | ok | All 19 code-length codes declared |
| [`codelen-num-codes-4`](files/codelen-num-codes-4.webp) [txt](cases/codelen-num-codes-4.txt) | ok | Only 4 code-length codes declared, the minimum the 4-bit field allows |
| [`codelen-over-capacity`](files/codelen-over-capacity.webp) [txt](cases/codelen-over-capacity.txt) | reject | Three symbols of depth 1, more than the two codes of that length that exist |
| [`codelen-oversubscribed`](files/codelen-oversubscribed.webp) [txt](cases/codelen-oversubscribed.txt) | reject | Lengths 1, 2, 2, 2: each length is individually possible, but together they over-subscribe the tree |
| [`codelen-repeat-past-end`](files/codelen-repeat-past-end.webp) [txt](cases/codelen-repeat-past-end.txt) | reject | A repeat run that would write past the end of the alphabet |
| [`codelen-repeat16-no-previous`](files/codelen-repeat16-no-previous.webp) [txt](cases/codelen-repeat16-no-previous.txt) | ok | Code-length stream starting with code 16 (repeat previous), before any non-zero length was seen |
| [`codelen-repeat17-short-zeros`](files/codelen-repeat17-short-zeros.webp) [txt](cases/codelen-repeat17-short-zeros.txt) | ok | Code-length stream using code 17 (3..10 zeros) rather than 18 |
| [`codelen-repeat18-138-zeros`](files/codelen-repeat18-138-zeros.webp) [txt](cases/codelen-repeat18-138-zeros.txt) | ok | Code-length stream using code 18 with its maximum run of 138 zeros |
| [`codelen-single-symbol-complex-form`](files/codelen-single-symbol-complex-form.webp) [txt](cases/codelen-single-symbol-complex-form.txt) | ok | The complex form used to describe a code with exactly one symbol |
| [`codelen-two-level-table`](files/codelen-two-level-table.webp) [txt](cases/codelen-two-level-table.txt) | ok | A green code with depths up to 10, past the 8-bit root table |

### Meta Huffman / entropy image

The sub-image that picks one of several code groups per tile, and the remapping
the decoder does when the group count looks implausible.

| file | | what it is |
| --- | --- | --- |
| [`meta-huffman-1001-groups`](files/meta-huffman-1001-groups.webp) [txt](cases/meta-huffman-1001-groups.txt) | ok | Entropy image whose highest group index is 1000, one past the decoder's arbitrary limit |
| [`meta-huffman-groups-truncated`](files/meta-huffman-groups-truncated.webp) [txt](cases/meta-huffman-groups-truncated.txt) | reject | An entropy image naming group 1 when only one group of codes follows |
| [`meta-huffman-per-tile-data`](files/meta-huffman-per-tile-data.webp) [txt](cases/meta-huffman-per-tile-data.txt) | ok | Two Huffman groups that both carry real data, so the code in use changes four pixels into every row |
| [`meta-huffman-precision-max`](files/meta-huffman-precision-max.webp) [txt](cases/meta-huffman-precision-max.txt) | ok | Meta Huffman with the largest tile size (precision 9, 512x512 pixels) |
| [`meta-huffman-precision-min`](files/meta-huffman-precision-min.webp) [txt](cases/meta-huffman-precision-min.txt) | ok | Meta Huffman with the smallest tile size (precision 2, 4x4 pixels) |
| [`meta-huffman-sparse-groups`](files/meta-huffman-sparse-groups.webp) [txt](cases/meta-huffman-sparse-groups.txt) | ok | Entropy image referencing groups 0 and 900 only, leaving a 900-entry hole |
| [`meta-huffman-two-groups`](files/meta-huffman-two-groups.webp) [txt](cases/meta-huffman-two-groups.txt) | ok | Two Huffman groups selected per tile by the entropy image |

### Color cache

Size bounds, and cache-index literals.

| file | | what it is |
| --- | --- | --- |
| [`cache-bits-0-invalid`](files/cache-bits-0-invalid.webp) [txt](cases/cache-bits-0-invalid.txt) | reject | Color cache flagged as present but with 0 bits |
| [`cache-bits-1`](files/cache-bits-1.webp) [txt](cases/cache-bits-1.txt) | ok | Color cache with the minimum size, 1 bit (2 entries) |
| [`cache-bits-11`](files/cache-bits-11.webp) [txt](cases/cache-bits-11.txt) | ok | Color cache with the maximum size, 11 bits (2048 entries) |
| [`cache-bits-12-invalid`](files/cache-bits-12-invalid.webp) [txt](cases/cache-bits-12-invalid.txt) | reject | Color cache with 12 bits, one past MAX_CACHE_BITS |
| [`cache-index-literal`](files/cache-index-literal.webp) [txt](cases/cache-index-literal.txt) | ok | A pixel coded as a color-cache index rather than as a literal |

### Sub-images

A lossless file carries whole image streams inside itself: one for each
transform that needs a per-tile parameter, and one for the entropy image. Each
is read by the same DecodeImageStream() as the outer image, minus the
transforms and the entropy image it is not allowed to have of its own -- so
each has a color cache and five Huffman codes that a file can say something
about, and that cwebp always writes the same dull way.

| file | | what it is |
| --- | --- | --- |
| [`subimage-cache-12-invalid`](files/subimage-cache-12-invalid.webp) [txt](cases/subimage-cache-12-invalid.txt) | reject | A sub-image color cache of 12 bits, one past MAX_CACHE_BITS |
| [`subimage-cache-entropy-image`](files/subimage-cache-entropy-image.webp) [txt](cases/subimage-cache-entropy-image.txt) | ok | A color cache inside the entropy image itself |
| [`subimage-cache-palette-max`](files/subimage-cache-palette-max.webp) [txt](cases/subimage-cache-palette-max.txt) | ok | A color cache of 11 bits inside the palette sub-image of the color-indexing transform |
| [`subimage-cache-predictor-min`](files/subimage-cache-predictor-min.webp) [txt](cases/subimage-cache-predictor-min.txt) | ok | A color cache declared inside the predictor transform's sub-image, 1 bit |
| [`subimage-cache-zero-invalid`](files/subimage-cache-zero-invalid.webp) [txt](cases/subimage-cache-zero-invalid.txt) | reject | A sub-image color cache flagged present but with 0 bits |
| [`subimage-code-complex-form`](files/subimage-code-complex-form.webp) [txt](cases/subimage-code-complex-form.txt) | ok | The predictor sub-image's green code written with the code-length repeat escapes |
| [`subimage-code-empty`](files/subimage-code-empty.webp) [txt](cases/subimage-code-empty.txt) | reject | A sub-image code-length stream that really does assign length 0 to all 280 symbols |
| [`subimage-code-max-symbol`](files/subimage-code-max-symbol.webp) [txt](cases/subimage-code-max-symbol.txt) | ok | A sub-image code using the explicit max_symbol early stop |
| [`subimage-code-oversubscribed`](files/subimage-code-oversubscribed.webp) [txt](cases/subimage-code-oversubscribed.txt) | reject | A sub-image Huffman code with lengths 1, 2, 2, 2, which over-subscribes the tree |

### Palette packing

Index width follows the palette size, and the map is padded out to the packing
capacity with black.

| file | | what it is |
| --- | --- | --- |
| [`transform-palette-1-color`](files/transform-palette-1-color.webp) [txt](cases/transform-palette-1-color.txt) | ok | Palette with a single color, the smallest the 8-bit field can express |
| [`transform-palette-16-colors`](files/transform-palette-16-colors.webp) [txt](cases/transform-palette-16-colors.txt) | ok | Palette of 16 colors, so indices are 4 bits and 2 pixels share a byte |
| [`transform-palette-2-colors`](files/transform-palette-2-colors.webp) [txt](cases/transform-palette-2-colors.txt) | ok | Color-indexing transform with 2 colors, so 8 pixels are packed per byte |
| [`transform-palette-256-colors`](files/transform-palette-256-colors.webp) [txt](cases/transform-palette-256-colors.txt) | ok | Color-indexing transform with the full 256-entry palette |
| [`transform-palette-3-colors`](files/transform-palette-3-colors.webp) [txt](cases/transform-palette-3-colors.txt) | ok | Palette of 3 colors, so indices are 2 bits and 4 pixels share a byte |
| [`transform-palette-index-past-end`](files/transform-palette-index-past-end.webp) [txt](cases/transform-palette-index-past-end.txt) | ok | Palette of 3 colors addressed with index 3, which does not exist |

### Transforms

Presence, repetition and tile sizes.

| file | | what it is |
| --- | --- | --- |
| [`transform-all-four`](files/transform-all-four.webp) [txt](cases/transform-all-four.txt) | ok | All four transforms present in one stream |
| [`transform-cross-color-bits-max`](files/transform-cross-color-bits-max.webp) [txt](cases/transform-cross-color-bits-max.txt) | ok | Cross-color transform with bits = 9, the largest tile size |
| [`transform-cross-color-multipliers`](files/transform-cross-color-multipliers.webp) [txt](cases/transform-cross-color-multipliers.txt) | ok | A cross-color transform with real multipliers rather than the identity |
| [`transform-predictor-bits-max`](files/transform-predictor-bits-max.webp) [txt](cases/transform-predictor-bits-max.txt) | ok | Predictor transform with bits = 9, the maximum tile size |
| [`transform-repeated`](files/transform-repeated.webp) [txt](cases/transform-repeated.txt) | reject | The subtract-green transform declared twice |

### Back-references

Copy lengths and distances.

| file | | what it is |
| --- | --- | --- |
| [`lz77-distance-1-run`](files/lz77-distance-1-run.webp) [txt](cases/lz77-distance-1-run.txt) | ok | A single literal followed by a length-8 copy at distance 1 |
| [`lz77-distance-2-pattern`](files/lz77-distance-2-pattern.webp) [txt](cases/lz77-distance-2-pattern.txt) | ok | A back-reference at distance 2, five pixels long |
| [`lz77-distance-3-overlap`](files/lz77-distance-3-overlap.webp) [txt](cases/lz77-distance-3-overlap.txt) | ok | A ten-pixel back-reference at distance 3 |
| [`lz77-distance-direct-121`](files/lz77-distance-direct-121.webp) [txt](cases/lz77-distance-direct-121.txt) | ok | Plane code 121, the first value past the table |
| [`lz77-distance-past-start`](files/lz77-distance-past-start.webp) [txt](cases/lz77-distance-past-start.txt) | reject | A back-reference pointing further back than the pixels decoded so far |
| [`lz77-length-past-end`](files/lz77-length-past-end.webp) [txt](cases/lz77-length-past-end.txt) | reject | A copy whose length runs past the last pixel of the image |
| [`lz77-max-length-symbol`](files/lz77-max-length-symbol.webp) [txt](cases/lz77-max-length-symbol.txt) | ok | A back-reference using length symbol 23, the largest the format defines |
| [`lz77-plane-code-1`](files/lz77-plane-code-1.webp) [txt](cases/lz77-plane-code-1.txt) | ok | Back-reference with plane code 1, which means "the pixel directly above" |
| [`lz77-plane-code-120`](files/lz77-plane-code-120.webp) [txt](cases/lz77-plane-code-120.txt) | ok | Plane code 120, the last entry of the 2-D offset table |
| [`lz77-plane-code-clamped-to-1`](files/lz77-plane-code-clamped-to-1.webp) [txt](cases/lz77-plane-code-clamped-to-1.txt) | ok | Plane code 4 on a 1-pixel-wide image, where the 2-D offset computes to 0 |

### Predictor modes

The per-tile predictor index. It is read as a 4-bit field, so all 16 values are
reachable, but the format only defines 14 of them.

| file | | what it is |
| --- | --- | --- |
| [`predictor-all-16-modes`](files/predictor-all-16-modes.webp) [txt](cases/predictor-all-16-modes.txt) | ok | One tile per predictor index, 0 to 15, across a 64x4 image |
| [`predictor-mode-11-select`](files/predictor-mode-11-select.webp) [txt](cases/predictor-mode-11-select.txt) | ok | Predictor 11 (Select) over the whole image |
| [`predictor-mode-13-clamp-half`](files/predictor-mode-13-clamp-half.webp) [txt](cases/predictor-mode-13-clamp-half.txt) | ok | Predictor 13 (ClampAddSubtractHalf) over the whole image |
| [`predictor-mode-14-undefined`](files/predictor-mode-14-undefined.webp) [txt](cases/predictor-mode-14-undefined.txt) | ok | Every tile selects predictor 14, which the format does not define |
| [`predictor-mode-15-undefined`](files/predictor-mode-15-undefined.webp) [txt](cases/predictor-mode-15-undefined.txt) | ok | Every tile selects predictor 15, the other undefined index |
| [`predictor-single-row`](files/predictor-single-row.webp) [txt](cases/predictor-single-row.txt) | ok | A predictor transform on a one-row image |
| [`predictor-tile-bits-min`](files/predictor-tile-bits-min.webp) [txt](cases/predictor-tile-bits-min.txt) | ok | Predictor tiles of 4x4 pixels, the smallest the format allows |

### Frame header

The 14-bit dimension fields and the version escape.

| file | | what it is |
| --- | --- | --- |
| [`header-alpha-is-used`](files/header-alpha-is-used.webp) [txt](cases/header-alpha-is-used.txt) | ok | The alpha_is_used hint set on an image whose every pixel is opaque |
| [`header-magic-wrong`](files/header-magic-wrong.webp) [txt](cases/header-magic-wrong.txt) | reject | Signature byte 0x2e instead of the 0x2f the format defines |
| [`header-max-area-bomb`](files/header-max-area-bomb.webp) [txt](cases/header-max-area-bomb.txt) | ok | 34 bytes declaring 16384x16384, every pixel one color |
| [`header-max-area-truncated`](files/header-max-area-truncated.webp) [txt](cases/header-max-area-truncated.txt) | reject | The same 16384x16384 header, cut off before the Huffman codes |
| [`header-version-max`](files/header-version-max.webp) [txt](cases/header-version-max.txt) | reject | Header with the 3-bit version field at 7, the largest it holds |
| [`header-version-nonzero`](files/header-version-nonzero.webp) [txt](cases/header-version-nonzero.txt) | reject | Header with the 3-bit version field set to 1 |
| [`header-width-16384`](files/header-width-16384.webp) [txt](cases/header-width-16384.txt) | ok | Width 16384, one past WEBP_MAX_DIMENSION |

### The RIFF container

The layer above the image: the RIFF header, the extended-format VP8X chunk and
its canvas size, the optional chunks a decoder must step over by their declared
length alone, and the padding rule that makes an odd-sized one even on disk.
Everything here is read by webp_dec.c before the frame is looked at.

| file | | what it is |
| --- | --- | --- |
| [`container-chunk-header-truncated`](files/container-chunk-header-truncated.webp) [txt](cases/container-chunk-header-truncated.txt) | reject | The file cut four bytes into the last chunk header |
| [`container-duplicate-image-chunk`](files/container-duplicate-image-chunk.webp) [txt](cases/container-duplicate-image-chunk.txt) | ok | Two VP8 chunks, one after the other |
| [`container-metadata-chunks`](files/container-metadata-chunks.webp) [txt](cases/container-metadata-chunks.txt) | ok | ICCP, EXIF and XMP chunks around the frame |
| [`container-no-image-chunk`](files/container-no-image-chunk.webp) [txt](cases/container-no-image-chunk.txt) | reject | A VP8X chunk and some metadata, and no image chunk at all |
| [`container-odd-chunk-no-pad`](files/container-odd-chunk-no-pad.webp) [txt](cases/container-odd-chunk-no-pad.txt) | reject | An odd-sized chunk whose header rounds its length up, taking the pad byte away |
| [`container-odd-chunk-payload`](files/container-odd-chunk-payload.webp) [txt](cases/container-odd-chunk-payload.txt) | ok | An optional chunk with an odd-sized payload, and the pad byte that must follow it |
| [`container-riff-size-past-end`](files/container-riff-size-past-end.webp) [txt](cases/container-riff-size-past-end.txt) | reject, incremental ok | A RIFF header claiming far more bytes than the file holds |
| [`container-riff-size-short`](files/container-riff-size-short.webp) [txt](cases/container-riff-size-short.txt) | reject | A RIFF header claiming 11 bytes, one less than the smallest legal value |
| [`container-riff-size-truncates-chunks`](files/container-riff-size-truncates-chunks.webp) [txt](cases/container-riff-size-truncates-chunks.txt) | reject | A RIFF size that stops in the middle of the chunks behind it |
| [`container-trailing-bytes`](files/container-trailing-bytes.webp) [txt](cases/container-trailing-bytes.txt) | ok | Bytes after the last chunk that the RIFF size does not account for |
| [`container-unknown-chunk`](files/container-unknown-chunk.webp) [txt](cases/container-unknown-chunk.txt) | ok | A chunk with a fourcc the format does not define, ahead of the frame |
| [`container-vp8x-animation`](files/container-vp8x-animation.webp) [txt](cases/container-vp8x-animation.txt) | reject | The VP8X animation flag set on a file with no animation chunks |
| [`container-vp8x-area-overflow`](files/container-vp8x-area-overflow.webp) [txt](cases/container-vp8x-area-overflow.txt) | reject | A VP8X canvas of 16777216 by 16777216, the largest the two 24-bit fields can describe |
| [`container-vp8x-canvas-mismatch`](files/container-vp8x-canvas-mismatch.webp) [txt](cases/container-vp8x-canvas-mismatch.txt) | reject | A VP8X canvas of 64x64 in front of a 32x32 frame |
| [`container-vp8x-reserved-bits`](files/container-vp8x-reserved-bits.webp) [txt](cases/container-vp8x-reserved-bits.txt) | ok | The reserved bits of the VP8X flags field all set |
| [`container-vp8x-still-flags`](files/container-vp8x-still-flags.webp) [txt](cases/container-vp8x-still-flags.txt) | ok | The four still-image VP8X flags set, with none of the chunks they promise |
| [`container-vp8x-wrong-size`](files/container-vp8x-wrong-size.webp) [txt](cases/container-vp8x-wrong-size.txt) | reject | A VP8X chunk whose header claims 9 bytes rather than 10 |
| [`container-vp8x`](files/container-vp8x.webp) [txt](cases/container-vp8x.txt) | ok | The extended format: a VP8X chunk ahead of the frame |
| [`container-zero-size-chunk`](files/container-zero-size-chunk.webp) [txt](cases/container-zero-size-chunk.txt) | ok | An optional chunk declaring a zero-length payload |

### Animation

An ANIM chunk carrying the loop count, then an ANMF per frame with its own
position, duration, disposal and blending, and its own image chunks. anim_dump
is what reads these: the demuxer of demux.c, the frame composition of
anim_decode.c, and one decode per frame.

| file | | what it is |
| --- | --- | --- |
| [`anim-alph-after-image`](files/anim-alph-after-image.webp) [txt](cases/anim-alph-after-image.txt) | reject, anim_dump reject | A frame whose alpha chunk comes after its image |
| [`anim-alpha-flag-missing`](files/anim-alpha-flag-missing.webp) [txt](cases/anim-alpha-flag-missing.txt) | reject, anim_dump ok, webpinfo reject | Frames with transparency in a file whose VP8X does not admit it |
| [`anim-alpha-lossless-frame`](files/anim-alpha-lossless-frame.webp) [txt](cases/anim-alpha-lossless-frame.txt) | reject, anim_dump ok | An animation frame whose alpha plane is a compressed image stream |
| [`anim-alpha-raw-frame`](files/anim-alpha-raw-frame.webp) [txt](cases/anim-alpha-raw-frame.txt) | reject, anim_dump ok | An animation frame carrying an uncompressed ALPH chunk beside its lossy image |
| [`anim-anim-chunk-padded`](files/anim-anim-chunk-padded.webp) [txt](cases/anim-anim-chunk-padded.txt) | reject, anim_dump ok, webpinfo reject | An ANIM chunk with four bytes of padding after its two fields |
| [`anim-anim-chunk-short`](files/anim-anim-chunk-short.webp) [txt](cases/anim-anim-chunk-short.txt) | reject, anim_dump reject | An ANIM chunk of two bytes |
| [`anim-anmf-header-truncated`](files/anim-anmf-header-truncated.webp) [txt](cases/anim-anmf-header-truncated.txt) | reject, anim_dump reject | An ANMF chunk declaring fewer bytes than its own header needs |
| [`anim-anmf-odd-size`](files/anim-anmf-odd-size.webp) [txt](cases/anim-anmf-odd-size.txt) | reject, anim_dump reject | An ANMF chunk whose declared length is odd |
| [`anim-anmf-size-past-end`](files/anim-anmf-size-past-end.webp) [txt](cases/anim-anmf-size-past-end.txt) | reject, anim_dump reject | An ANMF chunk claiming to be far longer than the file |
| [`anim-anmf-size-short`](files/anim-anmf-size-short.webp) [txt](cases/anim-anmf-size-short.txt) | reject, anim_dump reject | An ANMF chunk declaring less than the image inside it takes up |
| [`anim-background-color`](files/anim-background-color.webp) [txt](cases/anim-background-color.txt) | reject, anim_dump ok | A background colour that is neither white nor transparent |
| [`anim-blend-none`](files/anim-blend-none.webp) [txt](cases/anim-blend-none.txt) | reject, anim_dump ok | A half-transparent frame that says not to blend it |
| [`anim-blend-over`](files/anim-blend-over.webp) [txt](cases/anim-blend-over.txt) | reject, anim_dump ok | The same half-transparent frame, blended |
| [`anim-blend-ranges`](files/anim-blend-ranges.webp) [txt](cases/anim-blend-ranges.txt) | reject, anim_dump ok | Three frames arranged so the blended region is split in two |
| [`anim-canvas-larger-than-frames`](files/anim-canvas-larger-than-frames.webp) [txt](cases/anim-canvas-larger-than-frames.txt) | reject, anim_dump ok | A canvas bigger than any frame in it |
| [`anim-dispose-background`](files/anim-dispose-background.webp) [txt](cases/anim-dispose-background.txt) | reject, anim_dump ok | A partial frame that asks to be cleared away after it is shown |
| [`anim-dispose-blend-matrix`](files/anim-dispose-blend-matrix.webp) [txt](cases/anim-dispose-blend-matrix.txt) | reject, anim_dump ok | The four combinations of the disposal and blending bits, in one sequence |
| [`anim-duplicate-anim`](files/anim-duplicate-anim.webp) [txt](cases/anim-duplicate-anim.txt) | reject, anim_dump ok | A second ANIM chunk between the frames |
| [`anim-duration-extremes`](files/anim-duration-extremes.webp) [txt](cases/anim-duration-extremes.txt) | reject, anim_dump ok | One frame of zero milliseconds and one of the longest the field can say |
| [`anim-eight-frames`](files/anim-eight-frames.webp) [txt](cases/anim-eight-frames.txt) | reject, anim_dump ok | Eight frames, each a different flat colour |
| [`anim-empty-frame-alone`](files/anim-empty-frame-alone.webp) [txt](cases/anim-empty-frame-alone.txt) | reject, anim_dump reject | An animation whose only frame carries no image |
| [`anim-empty-frame`](files/anim-empty-frame.webp) [txt](cases/anim-empty-frame.txt) | reject, anim_dump ok, webpinfo reject | An ANMF chunk holding its header and nothing else, followed by a real frame |
| [`anim-frame-1x1`](files/anim-frame-1x1.webp) [txt](cases/anim-frame-1x1.txt) | reject, anim_dump ok | A single-pixel frame in the middle of a canvas |
| [`anim-frame-alpha-only`](files/anim-frame-alpha-only.webp) [txt](cases/anim-frame-alpha-only.txt) | reject, anim_dump reject | A frame carrying an alpha chunk and no image |
| [`anim-frame-area-overflow`](files/anim-frame-area-overflow.webp) [txt](cases/anim-frame-area-overflow.txt) | reject, anim_dump reject | An ANMF header claiming a frame of sixteen million by sixteen million, inside a canvas of sixteen by sixteen |
| [`anim-frame-image-past-canvas`](files/anim-frame-image-past-canvas.webp) [txt](cases/anim-frame-image-past-canvas.txt) | reject, anim_dump reject | A frame whose header fits the canvas but whose image does not |
| [`anim-frame-offsets`](files/anim-frame-offsets.webp) [txt](cases/anim-frame-offsets.txt) | reject, anim_dump ok | A second frame smaller than the canvas, placed at an offset |
| [`anim-frame-past-canvas`](files/anim-frame-past-canvas.webp) [txt](cases/anim-frame-past-canvas.txt) | reject, anim_dump reject | A frame whose rectangle runs off the edge of the canvas |
| [`anim-frame-reserved-bits`](files/anim-frame-reserved-bits.webp) [txt](cases/anim-frame-reserved-bits.txt) | reject, anim_dump ok | The six reserved bits of the ANMF flag byte all set |
| [`anim-frame-size-mismatch`](files/anim-frame-size-mismatch.webp) [txt](cases/anim-frame-size-mismatch.txt) | reject, anim_dump ok, webpinfo reject | An ANMF header claiming a size its own image disagrees with |
| [`anim-frames-without-flag`](files/anim-frames-without-flag.webp) [txt](cases/anim-frames-without-flag.txt) | reject, anim_dump reject | A full animation whose VP8X does not claim to be one |
| [`anim-image-chunk-beside-frames`](files/anim-image-chunk-beside-frames.webp) [txt](cases/anim-image-chunk-beside-frames.txt) | reject, anim_dump reject | A top-level image chunk in a file that also has frames |
| [`anim-loop-count-max`](files/anim-loop-count-max.webp) [txt](cases/anim-loop-count-max.txt) | reject, anim_dump ok | Loop count 65535, the largest the field holds |
| [`anim-lossy-frames`](files/anim-lossy-frames.webp) [txt](cases/anim-lossy-frames.txt) | reject, anim_dump ok | Two lossy frames rather than lossless ones |
| [`anim-metadata-chunks`](files/anim-metadata-chunks.webp) [txt](cases/anim-metadata-chunks.txt) | reject, anim_dump ok | An animation carrying an ICC profile and both metadata chunks |
| [`anim-metadata-without-flags`](files/anim-metadata-without-flags.webp) [txt](cases/anim-metadata-without-flags.txt) | reject, anim_dump ok, webpinfo reject | ICCP and EXIF chunks in an animation that declares neither |
| [`anim-mixed-formats`](files/anim-mixed-formats.webp) [txt](cases/anim-mixed-formats.txt) | reject, anim_dump ok | A lossy frame and a lossless one in the same animation |
| [`anim-nested-anmf`](files/anim-nested-anmf.webp) [txt](cases/anim-nested-anmf.txt) | reject, anim_dump reject | An ANMF chunk inside another ANMF chunk |
| [`anim-no-anim-chunk`](files/anim-no-anim-chunk.webp) [txt](cases/anim-no-anim-chunk.txt) | reject, anim_dump reject | Frames with no ANIM chunk in front of them |
| [`anim-no-vp8x`](files/anim-no-vp8x.webp) [txt](cases/anim-no-vp8x.txt) | reject, anim_dump reject | ANIM and ANMF chunks with no VP8X in front of them |
| [`anim-riff-size-past-end`](files/anim-riff-size-past-end.webp) [txt](cases/anim-riff-size-past-end.txt) | reject, anim_dump reject | A RIFF size claiming more bytes than the file has |
| [`anim-riff-size-truncates-frames`](files/anim-riff-size-truncates-frames.webp) [txt](cases/anim-riff-size-truncates-frames.txt) | reject, anim_dump reject | A RIFF size that stops in the middle of the last frame |
| [`anim-second-vp8x`](files/anim-second-vp8x.webp) [txt](cases/anim-second-vp8x.txt) | reject, anim_dump reject | A second VP8X chunk after the frames have started |
| [`anim-single-frame`](files/anim-single-frame.webp) [txt](cases/anim-single-frame.txt) | reject, anim_dump ok | An animation of one frame |
| [`anim-two-frames`](files/anim-two-frames.webp) [txt](cases/anim-two-frames.txt) | reject, anim_dump ok | Two full-canvas lossless frames, the plain animation everything else here is a variation on |
| [`anim-two-images-in-frame`](files/anim-two-images-in-frame.webp) [txt](cases/anim-two-images-in-frame.txt) | reject, anim_dump reject | One ANMF carrying two image chunks |
| [`anim-unknown-chunk-between-frames`](files/anim-unknown-chunk-between-frames.webp) [txt](cases/anim-unknown-chunk-between-frames.txt) | reject, anim_dump ok | An unrecognised chunk sitting between two frames |
| [`anim-vp8l-with-alph`](files/anim-vp8l-with-alph.webp) [txt](cases/anim-vp8l-with-alph.txt) | reject, anim_dump reject | A frame carrying both an alpha chunk and a lossless image |

### The alpha chunk

ALPH carries the alpha plane beside a lossy frame: a header byte of four two-
bit fields, then the plane itself, either stored as it is or compressed with
the lossless coder in its 8-bit mode. That mode is a separate path through
vp8l_dec.c from the one every VP8L image here takes, and an alpha chunk is the
only thing that reaches it, beside a still frame or inside an animation frame
alike. Each of the four filters has a routine of its own in dsp/filters.c and
turns the same stored bytes into a different plane, so the pixel hash is what
tells those apart. A compressed plane is a lossless stream with its header left
off: the alph-plane files write one from text, and break each of the four
conditions the 8-bit mode asks for in turn.

| file | | what it is |
| --- | --- | --- |
| [`alph-after-image`](files/alph-after-image.webp) [txt](cases/alph-after-image.txt) | ok | An ALPH chunk placed after the image chunk instead of before it |
| [`alph-compression-invalid`](files/alph-compression-invalid.webp) [txt](cases/alph-compression-invalid.txt) | reject | A compression method of 2, past the lossless one |
| [`alph-empty-payload`](files/alph-empty-payload.webp) [txt](cases/alph-empty-payload.txt) | reject | An ALPH chunk holding its header byte and nothing else |
| [`alph-lossless-byte-flipped`](files/alph-lossless-byte-flipped.webp) [txt](cases/alph-lossless-byte-flipped.txt) | reject | The same plane with its last byte replaced |
| [`alph-lossless-palette`](files/alph-lossless-palette.webp) [txt](cases/alph-lossless-palette.txt) | ok | A losslessly compressed alpha plane carrying a palette transform, from a cwebp encode of a two-valued plane |
| [`alph-lossless-predictor`](files/alph-lossless-predictor.webp) [txt](cases/alph-lossless-predictor.txt) | ok | A losslessly compressed alpha plane carrying a predictor transform, from a cwebp encode of a gradient |
| [`alph-lossless-truncated`](files/alph-lossless-truncated.webp) [txt](cases/alph-lossless-truncated.txt) | reject | The predictor-transform plane cut to ten bytes |
| [`alph-no-vp8x`](files/alph-no-vp8x.webp) [txt](cases/alph-no-vp8x.txt) | reject | An ALPH chunk in a RIFF file with no VP8X ahead of it |
| [`alph-plane-cache`](files/alph-plane-cache.webp) [txt](cases/alph-plane-cache.txt) | ok | A palette-coded alpha plane that also declares a colour cache |
| [`alph-plane-filtered`](files/alph-plane-filtered.webp) [txt](cases/alph-plane-filtered.txt) | ok | A compressed alpha plane with the gradient filter on top |
| [`alph-plane-literals`](files/alph-plane-literals.webp) [txt](cases/alph-plane-literals.txt) | ok | A compressed alpha plane written as plain green literals |
| [`alph-plane-lz77-dist-2`](files/alph-plane-lz77-dist-2.webp) [txt](cases/alph-plane-lz77-dist-2.txt) | ok | An alpha plane back-reference at distance 2 |
| [`alph-plane-lz77-dist-4`](files/alph-plane-lz77-dist-4.webp) [txt](cases/alph-plane-lz77-dist-4.txt) | ok | An alpha plane back-reference at distance 4 |
| [`alph-plane-lz77-past-start`](files/alph-plane-lz77-past-start.webp) [txt](cases/alph-plane-lz77-past-start.txt) | reject | An alpha plane copying from before its own beginning |
| [`alph-plane-lz77-plain`](files/alph-plane-lz77-plain.webp) [txt](cases/alph-plane-lz77-plain.txt) | ok | Two alpha-plane back-references that take neither pattern arm: one at distance 3, one shorter than its own distance |
| [`alph-plane-lz77-unaligned`](files/alph-plane-lz77-unaligned.webp) [txt](cases/alph-plane-lz77-unaligned.txt) | ok | An alpha plane back-reference starting one byte off a word boundary |
| [`alph-plane-lz77`](files/alph-plane-lz77.webp) [txt](cases/alph-plane-lz77.txt) | ok | An alpha plane whose second half is a back-reference to its first |
| [`alph-plane-meta-huffman`](files/alph-plane-meta-huffman.webp) [txt](cases/alph-plane-meta-huffman.txt) | ok | A palette-coded alpha plane with an entropy image inside it |
| [`alph-plane-nontrivial-red`](files/alph-plane-nontrivial-red.webp) [txt](cases/alph-plane-nontrivial-red.txt) | ok | A palette-coded alpha plane whose red code carries two symbols |
| [`alph-plane-oversubscribed`](files/alph-plane-oversubscribed.webp) [txt](cases/alph-plane-oversubscribed.txt) | reject | An alpha plane whose green code is over-subscribed |
| [`alph-plane-palette`](files/alph-plane-palette.webp) [txt](cases/alph-plane-palette.txt) | ok | A compressed alpha plane of three values, through a palette |
| [`alph-plane-preprocessed`](files/alph-plane-preprocessed.webp) [txt](cases/alph-plane-preprocessed.txt) | ok | A compressed alpha plane flagged as level-reduced |
| [`alph-plane-two-transforms`](files/alph-plane-two-transforms.webp) [txt](cases/alph-plane-two-transforms.txt) | ok | An alpha plane carrying a palette and a subtract-green transform |
| [`alph-preprocessing-invalid`](files/alph-preprocessing-invalid.webp) [txt](cases/alph-preprocessing-invalid.txt) | reject | A pre-processing value of 2, past level reduction |
| [`alph-raw-filter-gradient`](files/alph-raw-filter-gradient.webp) [txt](cases/alph-raw-filter-gradient.txt) | ok | An uncompressed alpha plane under the gradient filter |
| [`alph-raw-filter-horizontal`](files/alph-raw-filter-horizontal.webp) [txt](cases/alph-raw-filter-horizontal.txt) | ok | An uncompressed alpha plane under the horizontal filter |
| [`alph-raw-filter-none`](files/alph-raw-filter-none.webp) [txt](cases/alph-raw-filter-none.txt) | ok | An uncompressed alpha plane under the none filter |
| [`alph-raw-filter-vertical`](files/alph-raw-filter-vertical.webp) [txt](cases/alph-raw-filter-vertical.txt) | ok | An uncompressed alpha plane under the vertical filter |
| [`alph-raw-oversized`](files/alph-raw-oversized.webp) [txt](cases/alph-raw-oversized.txt) | ok | An uncompressed plane 44 bytes longer than the picture needs |
| [`alph-raw-preprocessing`](files/alph-raw-preprocessing.webp) [txt](cases/alph-raw-preprocessing.txt) | ok | An uncompressed plane declaring the level-reduction pre-processing |
| [`alph-raw-short`](files/alph-raw-short.webp) [txt](cases/alph-raw-short.txt) | reject | An uncompressed plane one byte short of the picture |
| [`alph-reserved-set`](files/alph-reserved-set.webp) [txt](cases/alph-reserved-set.txt) | reject | The two reserved bits of the ALPH header byte set |
| [`alph-without-vp8x-flag`](files/alph-without-vp8x-flag.webp) [txt](cases/alph-without-vp8x-flag.txt) | ok | An ALPH chunk with the VP8X alpha flag left clear |

### Lossy: frame tag and picture header

The ten uncompressed bytes every lossy frame starts with: the profile, the
visibility and key-frame bits, the length of partition 0, the start code and
the two 14-bit dimensions.

| file | | what it is |
| --- | --- | --- |
| [`lossy-frame-bad-start-code`](files/lossy-frame-bad-start-code.webp) [txt](cases/lossy-frame-bad-start-code.txt) | reject | The three-byte start code changed from 9d 01 2a to 9d 01 29 |
| [`lossy-frame-colorspace-clamp`](files/lossy-frame-colorspace-clamp.webp) [txt](cases/lossy-frame-colorspace-clamp.txt) | ok | The colour-space and clamping-type bits both set |
| [`lossy-frame-interframe`](files/lossy-frame-interframe.webp) [txt](cases/lossy-frame-interframe.txt) | reject | The key-frame bit cleared, so the frame claims to be an inter frame |
| [`lossy-frame-not-shown`](files/lossy-frame-not-shown.webp) [txt](cases/lossy-frame-not-shown.txt) | reject | A key frame with the show_frame bit cleared |
| [`lossy-frame-part0-empty`](files/lossy-frame-part0-empty.webp) [txt](cases/lossy-frame-part0-empty.txt) | reject | The frame tag claims a zero-byte partition 0 |
| [`lossy-frame-part0-past-end`](files/lossy-frame-part0-past-end.webp) [txt](cases/lossy-frame-part0-past-end.txt) | reject | The frame tag claims a partition 0 far larger than the file |
| [`lossy-frame-scale-1`](files/lossy-frame-scale-1.webp) [txt](cases/lossy-frame-scale-1.txt) | ok | A horizontal upscaling hint of 1 and a vertical one of 3 |
| [`lossy-frame-scale-2`](files/lossy-frame-scale-2.webp) [txt](cases/lossy-frame-scale-2.txt) | ok | A horizontal upscaling hint of 2 and a vertical one of 1 |
| [`lossy-frame-scaled`](files/lossy-frame-scaled.webp) [txt](cases/lossy-frame-scaled.txt) | ok | Horizontal and vertical upscaling hints of 3 (2x) in the top bits of the dimension fields |
| [`lossy-frame-version-1`](files/lossy-frame-version-1.webp) [txt](cases/lossy-frame-version-1.txt) | ok | A frame declaring profile 1 instead of 0 |
| [`lossy-frame-version-2`](files/lossy-frame-version-2.webp) [txt](cases/lossy-frame-version-2.txt) | ok | Profile 2, one of the four values the decoder accepts |
| [`lossy-frame-version-3`](files/lossy-frame-version-3.webp) [txt](cases/lossy-frame-version-3.txt) | ok | Profile 3, the largest the decoder accepts |
| [`lossy-frame-version-4`](files/lossy-frame-version-4.webp) [txt](cases/lossy-frame-version-4.txt) | reject | Profile 4, one past the last valid value |
| [`lossy-frame-version-7`](files/lossy-frame-version-7.webp) [txt](cases/lossy-frame-version-7.txt) | reject | Profile 7, the largest the 3-bit field can hold |
| [`lossy-frame-width-16383`](files/lossy-frame-width-16383.webp) [txt](cases/lossy-frame-width-16383.txt) | ok | The widest frame the 14-bit field can describe, one macroblock tall |
| [`lossy-frame-zero-width`](files/lossy-frame-zero-width.webp) [txt](cases/lossy-frame-zero-width.txt) | reject | A frame whose width field is zero, with a height of 32 |

### Lossy: segmentation

Up to four segments, each with its own quantizer and loop-filter strength, and
a per-macroblock map saying which is which. cwebp uses the feature but only
ever writes absolute values, and always writes the map and the data together.

| file | | what it is |
| --- | --- | --- |
| [`lossy-segment-delta-quantizers`](files/lossy-segment-delta-quantizers.webp) [txt](cases/lossy-segment-delta-quantizers.txt) | ok | Segment quantizers read as deltas on the frame quantizer instead of absolute values |
| [`lossy-segment-filter-strengths`](files/lossy-segment-filter-strengths.webp) [txt](cases/lossy-segment-filter-strengths.txt) | ok | Per-segment loop-filter strengths, from 0 to 63, under a frame filter level of 40 |
| [`lossy-segment-four-quantizers`](files/lossy-segment-four-quantizers.webp) [txt](cases/lossy-segment-four-quantizers.txt) | ok | Four segments with four different absolute quantizers, one macroblock each |
| [`lossy-segment-map-only`](files/lossy-segment-map-only.webp) [txt](cases/lossy-segment-map-only.txt) | ok | A segment map with no segment data behind it |
| [`lossy-segment-no-map`](files/lossy-segment-no-map.webp) [txt](cases/lossy-segment-no-map.txt) | ok | Segmentation on, quantizers given, but no segment map: every macroblock is segment 0 |
| [`lossy-segment-prob-extremes`](files/lossy-segment-prob-extremes.webp) [txt](cases/lossy-segment-prob-extremes.txt) | ok | Segment probabilities of 0 and 255, and loop-filter updates at both ends of their range |
| [`lossy-segment-quant-extremes`](files/lossy-segment-quant-extremes.webp) [txt](cases/lossy-segment-quant-extremes.txt) | ok | Segment quantizers at 127, -127, 0 and absent |

### Lossy: loop filter

The in-loop deblocking filter: simple or normal, its level and sharpness, and
the per-reference and per-mode deltas. PrecomputeFilterStrengths() shifts the
interior limit right by one for sharpness 1 to 4 and by two for 5 to 7, then
clamps it to 9 - sharpness, which is what the sharpness files sit either side
of.

| file | | what it is |
| --- | --- | --- |
| [`lossy-filter-lf-delta-extremes`](files/lossy-filter-lf-delta-extremes.webp) [txt](cases/lossy-filter-lf-delta-extremes.txt) | ok | Loop-filter mode deltas at -63 and 63 |
| [`lossy-filter-lf-delta`](files/lossy-filter-lf-delta.webp) [txt](cases/lossy-filter-lf-delta.txt) | ok | Loop-filter deltas: 63 and -63 on the reference deltas, and a delta on the 4x4 mode |
| [`lossy-filter-normal-max`](files/lossy-filter-normal-max.webp) [txt](cases/lossy-filter-normal-max.txt) | ok | The normal loop filter at level 63, sharpness 0 |
| [`lossy-filter-sharpness-4`](files/lossy-filter-sharpness-4.webp) [txt](cases/lossy-filter-sharpness-4.txt) | ok | Sharpness 4, the last level that halves the interior limit |
| [`lossy-filter-sharpness-5`](files/lossy-filter-sharpness-5.webp) [txt](cases/lossy-filter-sharpness-5.txt) | ok | Sharpness 5, the first level that quarters it |
| [`lossy-filter-simple-max`](files/lossy-filter-simple-max.webp) [txt](cases/lossy-filter-simple-max.txt) | ok | The simple loop filter at level 63 and sharpness 7 |

### Lossy: quantizer

The frame quantizer index and the five deltas around it, one per plane and
coefficient kind, with clamps that are not all the same.

| file | | what it is |
| --- | --- | --- |
| [`lossy-quant-deltas-mirrored`](files/lossy-quant-deltas-mirrored.webp) [txt](cases/lossy-quant-deltas-mirrored.txt) | ok | The five quantizer deltas at the ends lossy-quant-deltas does not use |
| [`lossy-quant-deltas`](files/lossy-quant-deltas.webp) [txt](cases/lossy-quant-deltas.txt) | ok | All five quantizer deltas present, at the ends of their 4-bit range |
| [`lossy-quant-dequant-overflow`](files/lossy-quant-dequant-overflow.webp) [txt](cases/lossy-quant-dequant-overflow.txt) | ok | A coefficient of 2114 at the coarsest quantizer, so the dequantized value does not fit the int16 it is stored in |
| [`lossy-quant-max`](files/lossy-quant-max.webp) [txt](cases/lossy-quant-max.txt) | ok | The frame quantizer at 127, the coarsest |
| [`lossy-quant-min`](files/lossy-quant-min.webp) [txt](cases/lossy-quant-min.txt) | ok | The frame quantizer at 0, the finest the format allows |
| [`lossy-quant-uv-dc-clamp`](files/lossy-quant-uv-dc-clamp.webp) [txt](cases/lossy-quant-uv-dc-clamp.txt) | ok | A chroma DC quantizer index pushed past 117, where it is clamped rather than at 127 |

### Lossy: coefficient probabilities

The 1056 probabilities that drive the coefficient coder, each one optionally
replaced in the frame header, plus the skip probability.

| file | | what it is |
| --- | --- | --- |
| [`lossy-proba-all-updated`](files/lossy-proba-all-updated.webp) [txt](cases/lossy-proba-all-updated.txt) | ok | Every one of the 1056 coefficient probabilities updated |
| [`lossy-proba-one-update`](files/lossy-proba-one-update.webp) [txt](cases/lossy-proba-one-update.txt) | ok | A single coefficient probability updated, the other 1055 left alone |
| [`lossy-proba-refresh-and-skip-zero`](files/lossy-proba-refresh-and-skip-zero.webp) [txt](cases/lossy-proba-refresh-and-skip-zero.txt) | ok | The entropy-refresh bit set, and a skip probability of 0 |
| [`lossy-proba-skip-extremes`](files/lossy-proba-skip-extremes.webp) [txt](cases/lossy-proba-skip-extremes.txt) | ok | A skip probability of 255 with nothing skipped, and the flag itself written out |
| [`lossy-proba-zero`](files/lossy-proba-zero.webp) [txt](cases/lossy-proba-zero.txt) | ok | Coefficient probabilities of 0 and of 255, the ends of the range |

### Lossy: prediction modes

The 16x16 and 4x4 luma modes and the chroma modes, and the neighbour-indexed
probability table the 4x4 modes are coded with.

| file | | what it is |
| --- | --- | --- |
| [`lossy-mode-i16-all-four`](files/lossy-mode-i16-all-four.webp) [txt](cases/lossy-mode-i16-all-four.txt) | ok | The four 16x16 luma modes, one per macroblock |
| [`lossy-mode-i4-all-ten`](files/lossy-mode-i4-all-ten.webp) [txt](cases/lossy-mode-i4-all-ten.txt) | ok | All ten 4x4 luma modes inside one macroblock, twice over |
| [`lossy-mode-i4-context`](files/lossy-mode-i4-context.webp) [txt](cases/lossy-mode-i4-context.txt) | ok | Four B_PRED macroblocks whose 4x4 modes walk the [above][left] probability table |
| [`lossy-mode-mixed`](files/lossy-mode-mixed.webp) [txt](cases/lossy-mode-mixed.txt) | ok | 16x16 and 4x4 macroblocks alternating, in both directions |
| [`lossy-mode-uv-all-four`](files/lossy-mode-uv-all-four.webp) [txt](cases/lossy-mode-uv-all-four.txt) | ok | The four chroma modes, one per macroblock |

### Lossy: coefficients

The token coder of section 13: magnitudes and their escape categories, end-of-
block, zero runs, and the four coefficient types. The band sweeps below share a
trick: the three token classes drive the context of the next position -- a zero
gives 0, a +-1 gives 1, anything larger gives 2 -- so a block of each, walked
to position 15, reads every band at that class, and placing the blocks so their
neighbour contexts are 0, 1 and 2 in turn is the only way to reach band 0,
which is never a token's successor.

| file | | what it is |
| --- | --- | --- |
| [`lossy-coeff-all-types`](files/lossy-coeff-all-types.webp) [txt](cases/lossy-coeff-all-types.txt) | ok | All four coefficient types in one macroblock, and both luma types across two |
| [`lossy-coeff-bands-chroma`](files/lossy-coeff-bands-chroma.webp) [txt](cases/lossy-coeff-bands-chroma.txt) | ok | The same sweep across the four blocks of a chroma plane |
| [`lossy-coeff-bands-i16`](files/lossy-coeff-bands-i16.webp) [txt](cases/lossy-coeff-bands-i16.txt) | ok | The same sweep for the two block kinds a 16x16 macroblock has: the Y2 block and the luma blocks that follow it |
| [`lossy-coeff-bands-i4`](files/lossy-coeff-bands-i4.webp) [txt](cases/lossy-coeff-bands-i4.txt) | ok | Three 4x4 luma blocks that sweep every coefficient band, one per context |
| [`lossy-coeff-cat3`](files/lossy-coeff-cat3.webp) [txt](cases/lossy-coeff-cat3.txt) | ok | Category-3 coefficients: 11 to 18, three extra bits each |
| [`lossy-coeff-cat4`](files/lossy-coeff-cat4.webp) [txt](cases/lossy-coeff-cat4.txt) | ok | Category-4 coefficients: 19 to 34, four extra bits |
| [`lossy-coeff-cat5`](files/lossy-coeff-cat5.webp) [txt](cases/lossy-coeff-cat5.txt) | ok | Category-5 coefficients: 35 to 66, five extra bits |
| [`lossy-coeff-cat6-max`](files/lossy-coeff-cat6-max.webp) [txt](cases/lossy-coeff-cat6-max.txt) | ok | The largest coefficient the format can encode: 2114 |
| [`lossy-coeff-cat6`](files/lossy-coeff-cat6.webp) [txt](cases/lossy-coeff-cat6.txt) | ok | Category-6 coefficients: 67 upwards, eleven extra bits |
| [`lossy-coeff-context`](files/lossy-coeff-context.webp) [txt](cases/lossy-coeff-context.txt) | ok | Neighbouring blocks with and without coefficients, so that every context value from 0 to 2 is used |
| [`lossy-coeff-empty-blocks`](files/lossy-coeff-empty-blocks.webp) [txt](cases/lossy-coeff-empty-blocks.txt) | ok | Every one of the 25 blocks empty, but the macroblock not skipped |
| [`lossy-coeff-full-block`](files/lossy-coeff-full-block.webp) [txt](cases/lossy-coeff-full-block.txt) | ok | A block with all sixteen coefficients non-zero, so the loop ends by running out of positions rather than on an end-of-block |
| [`lossy-coeff-medium-magnitudes`](files/lossy-coeff-medium-magnitudes.webp) [txt](cases/lossy-coeff-medium-magnitudes.txt) | ok | Coefficients of 5 through 10, coded with the fixed probabilities 159, 165 and 145 |
| [`lossy-coeff-small-magnitudes`](files/lossy-coeff-small-magnitudes.webp) [txt](cases/lossy-coeff-small-magnitudes.txt) | ok | Coefficients of 1, 2, 3 and 4, the magnitudes with their own tree branches |
| [`lossy-coeff-wht-full`](files/lossy-coeff-wht-full.webp) [txt](cases/lossy-coeff-wht-full.txt) | ok | A Y2 block with more than one coefficient, next to one with only a DC |
| [`lossy-coeff-zero-runs`](files/lossy-coeff-zero-runs.webp) [txt](cases/lossy-coeff-zero-runs.txt) | ok | Single coefficients at positions 15, 12, 8 and 1, each behind a run of zeros |

### Lossy: skipped macroblocks

The per-macroblock skip flag, which drops the residual entirely and clears the
neighbouring non-zero flags -- almost all of them.

| file | | what it is |
| --- | --- | --- |
| [`lossy-skip-all`](files/lossy-skip-all.webp) [txt](cases/lossy-skip-all.txt) | ok | Every macroblock skipped |
| [`lossy-skip-i4x4-nz-dc`](files/lossy-skip-i4x4-nz-dc.webp) [txt](cases/lossy-skip-i4x4-nz-dc.txt) | ok | A skipped 4x4 macroblock between two 16x16 ones that both carry a Y2 block |
| [`lossy-skip-mixed`](files/lossy-skip-mixed.webp) [txt](cases/lossy-skip-mixed.txt) | ok | Skipped and coded macroblocks alternating, with a skip probability of 1 |

### Lossy: token partitions

A lossy frame may carry 1, 2, 4 or 8 token partitions, macroblock row r being
read from partition r & (n - 1). cwebp does not expose config.partitions and
libwebp forces it back to 1 whenever the token path is used (webp_enc.c:124),
so none of this is reachable through the tools.

| file | | what it is |
| --- | --- | --- |
| [`lossy-parts-2-wrap`](files/lossy-parts-2-wrap.webp) [txt](cases/lossy-parts-2-wrap.txt) | ok | Four rows over two partitions, so each partition holds two non-adjacent rows |
| [`lossy-parts-8-rows`](files/lossy-parts-8-rows.webp) [txt](cases/lossy-parts-8-rows.txt) | ok | Eight macroblock rows over eight token partitions, one row each, every row different |
| [`lossy-parts-last-empty`](files/lossy-parts-last-empty.webp) [txt](cases/lossy-parts-last-empty.txt) | reject | Four partitions whose declared sizes leave nothing for the last one |
| [`lossy-parts-size-past-end`](files/lossy-parts-size-past-end.webp) [txt](cases/lossy-parts-size-past-end.txt) | reject | Four partitions, the first declaring 16 MB of data |
| [`lossy-parts-table-too-small`](files/lossy-parts-table-too-small.webp) [txt](cases/lossy-parts-table-too-small.txt) | reject | Eight partitions declared in a frame with only ten bytes left for the twenty-one-byte size table |

### Lossy: truncation

Frames that stop early, at each of the places the decoder can notice: inside
partition 0, inside the macroblock modes, and inside the token data.

| file | | what it is |
| --- | --- | --- |
| [`lossy-truncated-header`](files/lossy-truncated-header.webp) [txt](cases/lossy-truncated-header.txt) | reject | Partition 0 cut to two bytes, which is enough for the segment header and not for the filter header |
| [`lossy-truncated-modes`](files/lossy-truncated-modes.webp) [txt](cases/lossy-truncated-modes.txt) | reject | Partition 0 long enough for the whole frame header and not for the macroblock modes that follow it |
| [`lossy-truncated-short-modes`](files/lossy-truncated-short-modes.webp) [txt](cases/lossy-truncated-short-modes.txt) | ok | Mode data for 15 macroblocks in a frame whose dimensions call for 16 |
| [`lossy-truncated-tokens`](files/lossy-truncated-tokens.webp) [txt](cases/lossy-truncated-tokens.txt) | reject | Partition 0 intact, the token partition cut in half |

### Lossy: partition sizes, from real encodes

The four files behind these are genuine encoder output, made through the
encoder API rather than cwebp, and the broken ones rewrite the raw size table
that follows partition 0. They carry a whole frame of real coefficients, which
the assembled cases do not.

| file | | what it is |
| --- | --- | --- |
| [`lossy-1-partitions`](files/lossy-1-partitions.webp) | ok | A single token partition: the default, and the control for the others |
| [`lossy-2-partitions`](files/lossy-2-partitions.webp) | ok | A plain 2-partition lossy frame |
| [`lossy-4-partitions`](files/lossy-4-partitions.webp) | ok | A plain 4-partition lossy frame |
| [`lossy-8-partitions`](files/lossy-8-partitions.webp) | ok | A plain 8-partition lossy frame, the maximum the 2-bit field allows |
| [`lossy-8-partitions-size-overflow`](files/lossy-8-partitions-size-overflow.webp) | reject | Eight partitions whose first declared size is 0xffffff, far past the data |
| [`lossy-8-partitions-zero-sizes`](files/lossy-8-partitions-zero-sizes.webp) | reject | Eight partitions all declared as zero bytes long |
| [`lossy-8-partitions-sizes-sum-past-end`](files/lossy-8-partitions-sizes-sum-past-end.webp) | reject | Eight partitions whose declared sizes add up to more than the chunk holds |
| [`lossy-combo-all-features`](files/lossy-combo-all-features.webp) [txt](cases/lossy-combo-all-features.txt) | ok | Every optional tool switched on in one frame at once |

### What reaches what

Every decoder path `src/probes.py` measures, and the files that reach it -- the
question a decoder leaves you with, which the list above answers only
backwards. Generated from `coverage.txt`.

| decoder path | files |
| --- | --- |
| `alpha_8b_blocks` | `alph-plane-meta-huffman` |
| `alpha_8b_copy` | `alph-lossless-palette`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-plain`, `alph-plane-lz77-unaligned` |
| `alpha_8b_copy_oob` | `alph-plane-lz77-past-start` |
| `alpha_8b_data` | `alph-lossless-palette`, `alph-plane-filtered`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-past-start` +6 |
| `alpha_8b_no_blocks` | `alph-lossless-palette`, `alph-plane-filtered`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-past-start` +5 |
| `alpha_is_used` | `anim-alpha-flag-missing`, `anim-blend-none`, `anim-blend-over`, `anim-blend-ranges`, `anim-dispose-background`, `anim-dispose-blend-matrix` +2 |
| `anim_blend_around_prev` | `anim-blend-ranges` |
| `anim_blend_nonpremult` | `anim-alph-after-image`, `anim-alpha-flag-missing`, `anim-alpha-lossless-frame`, `anim-alpha-raw-frame`, `anim-anim-chunk-padded`, `anim-anim-chunk-short` +41 |
| `anim_blend_whole_rows` | `anim-alpha-flag-missing`, `anim-blend-over`, `anim-blend-ranges`, `anim-canvas-larger-than-frames`, `anim-dispose-background`, `anim-dispose-blend-matrix` +2 |
| `anim_dispose_background` | `anim-blend-ranges`, `anim-dispose-background`, `anim-dispose-blend-matrix`, `anim-frame-1x1` |
| `anim_inter_frame` | `anim-alpha-flag-missing`, `anim-blend-over`, `anim-blend-ranges`, `anim-canvas-larger-than-frames`, `anim-dispose-background`, `anim-dispose-blend-matrix` +2 |
| `anim_key_first_frame` | `anim-alpha-flag-missing`, `anim-alpha-lossless-frame`, `anim-alpha-raw-frame`, `anim-anim-chunk-padded`, `anim-background-color`, `anim-blend-none` +21 |
| `anim_key_frame` | `anim-alpha-flag-missing`, `anim-alpha-lossless-frame`, `anim-alpha-raw-frame`, `anim-anim-chunk-padded`, `anim-background-color`, `anim-blend-none` +21 |
| `anim_key_from_prev_dispose` | `anim-alpha-flag-missing`, `anim-blend-over`, `anim-blend-ranges`, `anim-canvas-larger-than-frames`, `anim-dispose-background`, `anim-dispose-blend-matrix` +2 |
| `anim_key_opaque_full_frame` | `anim-alpha-raw-frame`, `anim-blend-none`, `anim-dispose-background`, `anim-duplicate-anim`, `anim-duration-extremes`, `anim-eight-frames` +6 |
| `anim_range_disjoint` | `anim-blend-ranges` |
| `anim_range_left` | `anim-blend-ranges` |
| `anim_range_right` | `anim-blend-ranges` |
| `bmode_0` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +21 |
| `bmode_1` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +7 |
| `bmode_2` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-coeff-all-types`, `lossy-combo-all-features` +6 |
| `bmode_3` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +7 |
| `bmode_4` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +7 |
| `bmode_5` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-coeff-all-types`, `lossy-combo-all-features` +6 |
| `bmode_6` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +7 |
| `bmode_7` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-coeff-all-types`, `lossy-combo-all-features` +6 |
| `bmode_8` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +7 |
| `bmode_9` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +7 |
| `build_depth_15` | `codelen-depth-15` |
| `build_empty_code` | `codelen-all-zero-lengths`, `header-max-area-truncated`, `meta-huffman-groups-truncated`, `simple-dist-1sym-oob`, `simple-dist-2sym-both-oob`, `simple-dist-sym-40-first-oob` +1 |
| `build_single_value` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-lossless-truncated`, `alph-plane-cache`, `alph-plane-filtered` +109 |
| `build_two_level` | `codelen-depth-15`, `codelen-two-level-table` |
| `cache_bits_max` | `cache-bits-11`, `subimage-cache-palette-max` |
| `cache_bits_min` | `cache-bits-1`, `cache-index-literal`, `subimage-cache-predictor-min` |
| `cache_index` | `cache-index-literal`, `lossless-all-features` |
| `codelen_16` | `codelen-repeat16-no-previous` |
| `codelen_16_default_prev` | `codelen-repeat16-no-previous` |
| `codelen_17` | `alph-lossless-byte-flipped`, `alph-lossless-predictor`, `alph-lossless-truncated`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-past-start`, `alph-plane-lz77-plain` +2 |
| `codelen_17_max_run` | `alph-plane-lz77-past-start`, `codelen-repeat17-short-zeros`, `lz77-plane-code-120` |
| `codelen_18` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4` +31 |
| `codelen_18_max_run` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-past-start` +20 |
| `coeff_cat3` | `anim-lossy-frames`, `container-duplicate-image-chunk`, `container-metadata-chunks`, `container-odd-chunk-payload`, `container-trailing-bytes`, `container-unknown-chunk` +39 |
| `coeff_cat4` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +72 |
| `coeff_cat5` | `lossy-coeff-cat5` |
| `coeff_cat6` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-coeff-cat6`, `lossy-coeff-cat6-max` +1 |
| `coeff_max` | `lossy-coeff-cat6-max`, `lossy-quant-dequant-overflow` |
| `coeff_v2` | `container-duplicate-image-chunk`, `container-metadata-chunks`, `container-odd-chunk-payload`, `container-trailing-bytes`, `container-unknown-chunk`, `container-vp8x` +35 |
| `coeff_v3_4` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +65 |
| `coeff_v5_6` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +43 |
| `coeff_v7_10` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-coeff-all-types`, `lossy-coeff-full-block` +17 |
| `complex_code` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-lossless-truncated`, `alph-plane-cache`, `alph-plane-lz77` +51 |
| `copy32b_dist_2` | `lz77-distance-2-pattern` |
| `copy32b_overlap` | `lz77-distance-3-overlap` |
| `copy32b_tail` | `lz77-distance-2-pattern` |
| `copy8b_dist_1` | `alph-plane-lz77`, `alph-plane-lz77-unaligned` |
| `copy8b_dist_2` | `alph-plane-lz77-dist-2` |
| `copy8b_dist_4` | `alph-plane-lz77-dist-4` |
| `copy8b_memcpy` | `alph-plane-lz77-plain` |
| `copy8b_no_pattern` | `alph-plane-lz77-plain` |
| `copy8b_tail` | `alph-plane-lz77-unaligned` |
| `copy8b_unaligned` | `alph-plane-lz77-unaligned` |
| `cross_color_multipliers` | `transform-cross-color-multipliers` |
| `demux_alph_after_image` | `anim-alph-after-image` |
| `demux_anim_duplicate` | `anim-duplicate-anim` |
| `demux_anim_padded` | `anim-anim-chunk-padded` |
| `demux_anim_too_small` | `anim-anim-chunk-short` |
| `demux_bounds_inexact` | `anim-alpha-flag-missing`, `anim-alpha-lossless-frame`, `anim-alpha-raw-frame`, `anim-anim-chunk-padded`, `anim-background-color`, `anim-blend-none` +23 |
| `demux_chunk_flagless` | `anim-duplicate-anim`, `anim-metadata-without-flags` |
| `demux_frame_added` | `anim-alph-after-image`, `anim-alpha-flag-missing`, `anim-alpha-lossless-frame`, `anim-alpha-raw-frame`, `anim-anim-chunk-padded`, `anim-anmf-odd-size` +29 |
| `demux_frame_alph` | `anim-alph-after-image`, `anim-alpha-lossless-frame`, `anim-alpha-raw-frame`, `anim-frame-alpha-only`, `anim-vp8l-with-alph` |
| `demux_frame_area_overflow` | `anim-frame-area-overflow` |
| `demux_frame_before_anim` | `anim-no-anim-chunk` |
| `demux_frame_chunk_done` | `anim-image-chunk-beside-frames`, `anim-two-images-in-frame` |
| `demux_frame_dropped` | `anim-empty-frame`, `anim-empty-frame-alone`, `anim-nested-anmf` |
| `demux_frame_overran_payload` | `anim-anmf-size-short` |
| `demux_frame_past_canvas` | `anim-frame-image-past-canvas`, `anim-frame-past-canvas` |
| `demux_frame_too_small` | `anim-anmf-header-truncated` |
| `demux_image_in_animation` | `anim-image-chunk-beside-frames`, `anim-two-images-in-frame` |
| `demux_no_frames` | `anim-frames-without-flag` |
| `demux_partial` | `anim-riff-size-past-end` |
| `demux_second_vp8x` | `anim-second-vp8x` |
| `demux_vp8l_with_alph` | `anim-vp8l-with-alph` |
| `dist_clamped_to_1` | `lz77-plane-code-clamped-to-1` |
| `dist_direct` | `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-plain`, `alph-plane-lz77-unaligned`, `lz77-distance-2-pattern`, `lz77-distance-3-overlap` +2 |
| `dist_plane_code_1` | `lz77-distance-past-start`, `lz77-plane-code-1` |
| `dist_plane_code_120` | `lz77-plane-code-120` |
| `error:Alpha-decoder-initialization-failed.` | `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-truncated`, `alph-plane-oversubscribed`, `alph-preprocessing-invalid`, `alph-raw-short` +1 |
| `error:Could-not-decode-alpha-data.` | `alph-lossless-byte-flipped`, `alph-plane-lz77-past-start` |
| `error:Premature-end-of-file-encountered.` | `lossy-8-partitions-zero-sizes`, `lossy-truncated-tokens` |
| `error:Premature-end-of-partition0-encountered.` | `lossy-truncated-modes` |
| `error:cannot-parse-filter-header` | `lossy-truncated-header` |
| `error:cannot-parse-partitions` | `lossy-8-partitions-size-overflow`, `lossy-8-partitions-sizes-sum-past-end`, `lossy-parts-last-empty`, `lossy-parts-size-past-end`, `lossy-parts-table-too-small` |
| `error:cannot-parse-segment-header` | `lossy-frame-part0-empty` |
| `header_huge_area` | `header-max-area-bomb`, `header-max-area-truncated` |
| `header_width_16384` | `header-max-area-bomb`, `header-max-area-truncated`, `header-width-16384` |
| `info:bad-signature` | `alph-no-vp8x`, `anim-no-vp8x`, `lossy-frame-bad-start-code` |
| `info:not-a-keyframe` | `lossy-frame-interframe` |
| `info:not-displayable` | `lossy-frame-not-shown` |
| `info:partition-length` | `lossy-frame-part0-past-end` |
| `info:unknown-profile` | `lossy-frame-version-4`, `lossy-frame-version-7` |
| `info:zero-dimension` | `lossy-frame-zero-width` |
| `literal_four_channels` | `alph-lossless-byte-flipped`, `alph-lossless-predictor`, `lossless-plain`, `transform-palette-3-colors`, `transform-palette-index-past-end` |
| `literal_trivial_arb` | `codelen-depth-15`, `codelen-two-level-table`, `lz77-distance-2-pattern`, `lz77-distance-3-overlap`, `predictor-all-16-modes`, `predictor-tile-bits-min` +1 |
| `literal_trivial_code` | `alph-lossless-byte-flipped`, `alph-lossless-predictor`, `alph-lossless-truncated`, `alph-plane-cache`, `alph-plane-literals`, `alph-plane-two-transforms` +62 |
| `lz77_copy` | `alph-lossless-byte-flipped`, `alph-lossless-predictor`, `lossless-all-features`, `lz77-distance-1-run`, `lz77-distance-2-pattern`, `lz77-distance-3-overlap` +7 |
| `lz77_max_len_sym` | `lz77-max-length-symbol` |
| `max_symbol_early_stop` | `codelen-max-symbol-early-stop`, `subimage-code-max-symbol` |
| `mb_i16` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +91 |
| `mb_i4x4` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +21 |
| `mb_skip_flag` | `lossy-combo-all-features`, `lossy-proba-refresh-and-skip-zero`, `lossy-skip-all`, `lossy-skip-i4x4-nz-dc`, `lossy-skip-mixed`, `lossy-truncated-short-modes` |
| `mb_skipped` | `lossy-combo-all-features`, `lossy-proba-refresh-and-skip-zero`, `lossy-skip-all`, `lossy-skip-i4x4-nz-dc`, `lossy-skip-mixed`, `lossy-truncated-short-modes` |
| `meta_group_switch` | `lossless-all-features`, `meta-huffman-1001-groups`, `meta-huffman-per-tile-data`, `meta-huffman-sparse-groups`, `meta-huffman-two-groups`, `subimage-cache-entropy-image` |
| `meta_groups_over_1000` | `meta-huffman-1001-groups` |
| `meta_huffman` | `alph-plane-meta-huffman`, `lossless-all-features`, `meta-huffman-1001-groups`, `meta-huffman-groups-truncated`, `meta-huffman-per-tile-data`, `meta-huffman-precision-max` +4 |
| `meta_mapping_path` | `meta-huffman-1001-groups`, `meta-huffman-sparse-groups` |
| `meta_precision_max` | `meta-huffman-precision-max` |
| `meta_precision_min` | `alph-plane-meta-huffman`, `lossless-all-features`, `meta-huffman-1001-groups`, `meta-huffman-groups-truncated`, `meta-huffman-per-tile-data`, `meta-huffman-precision-min` +3 |
| `meta_unused_group` | `meta-huffman-1001-groups`, `meta-huffman-sparse-groups` |
| `num_codes_max` | `codelen-depth-15`, `codelen-num-codes-19` |
| `num_codes_min` | `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-past-start`, `alph-plane-lz77-plain`, `alph-plane-lz77-unaligned` +26 |
| `palette_1` | `transform-palette-1-color` |
| `palette_16` | `transform-palette-16-colors` |
| `palette_2` | `alph-lossless-palette`, `subimage-cache-palette-max`, `transform-all-four`, `transform-palette-2-colors` |
| `palette_256` | `transform-palette-256-colors` |
| `palette_3` | `alph-plane-cache`, `alph-plane-filtered`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-past-start` +10 |
| `palette_black_tail` | `alph-plane-cache`, `alph-plane-filtered`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-past-start` +11 |
| `parts_1` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +99 |
| `parts_2` | `lossy-2-partitions`, `lossy-parts-2-wrap` |
| `parts_4` | `lossy-4-partitions`, `lossy-combo-all-features`, `lossy-parts-last-empty`, `lossy-parts-size-past-end` |
| `parts_8` | `lossy-8-partitions`, `lossy-8-partitions-size-overflow`, `lossy-8-partitions-sizes-sum-past-end`, `lossy-8-partitions-zero-sizes`, `lossy-parts-8-rows`, `lossy-parts-table-too-small` |
| `parts_no_data_left` | `lossy-8-partitions-size-overflow`, `lossy-8-partitions-sizes-sum-past-end`, `lossy-parts-last-empty`, `lossy-parts-size-past-end` |
| `parts_psize_clamped` | `lossy-8-partitions-size-overflow`, `lossy-8-partitions-sizes-sum-past-end`, `lossy-parts-last-empty`, `lossy-parts-size-past-end` |
| `parts_psize_zero` | `lossy-8-partitions-size-overflow`, `lossy-8-partitions-sizes-sum-past-end`, `lossy-8-partitions-zero-sizes`, `lossy-parts-size-past-end` |
| `parts_size_table_truncated` | `lossy-parts-table-too-small` |
| `pred_first_row` | `alph-lossless-predictor`, `lossless-all-features`, `predictor-all-16-modes`, `predictor-mode-11-select`, `predictor-mode-13-clamp-half`, `predictor-mode-14-undefined` +8 |
| `pred_mode_0` | `lossless-all-features`, `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form`, `transform-predictor-bits-max` |
| `pred_mode_1` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form`, `subimage-code-max-symbol` |
| `pred_mode_10` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_11` | `predictor-all-16-modes`, `predictor-mode-11-select`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_12` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_13` | `predictor-all-16-modes`, `predictor-mode-13-clamp-half`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_14` | `predictor-all-16-modes`, `predictor-mode-14-undefined`, `subimage-code-complex-form` |
| `pred_mode_15` | `predictor-all-16-modes`, `predictor-mode-15-undefined`, `subimage-code-complex-form` |
| `pred_mode_2` | `alph-lossless-predictor`, `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-cache-predictor-min`, `subimage-code-complex-form` |
| `pred_mode_3` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_4` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_5` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_6` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_7` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_8` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_9` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `proba_255` | `lossy-proba-zero` |
| `proba_update` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-combo-all-features` +3 |
| `proba_zero` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-proba-zero` |
| `quant_delta` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-quant-deltas` +2 |
| `quant_max` | `lossy-quant-dequant-overflow`, `lossy-quant-max` |
| `quant_min` | `lossy-coeff-cat3`, `lossy-coeff-cat4`, `lossy-coeff-cat5`, `lossy-coeff-cat6`, `lossy-coeff-cat6-max`, `lossy-coeff-full-block` +4 |
| `quant_segment_absolute` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-combo-all-features` +6 |
| `quant_segment_delta` | `lossy-segment-delta-quantizers` |
| `quant_uv_dc_clamped_at_117` | `lossy-quant-dequant-overflow`, `lossy-quant-max`, `lossy-quant-uv-dc-clamp`, `lossy-segment-four-quantizers`, `lossy-segment-no-map`, `lossy-segment-quant-extremes` |
| `quant_y1_dc_clipped` | `lossy-segment-quant-extremes` |
| `quant_y2_ac_floor` | `lossy-coeff-cat3`, `lossy-coeff-cat4`, `lossy-coeff-cat5`, `lossy-coeff-cat6`, `lossy-coeff-cat6-max`, `lossy-coeff-full-block` +6 |
| `reject_cache_bits` | `cache-bits-0-invalid`, `cache-bits-12-invalid`, `subimage-cache-12-invalid`, `subimage-cache-zero-invalid` |
| `reject_count_over_capacity` | `alph-plane-oversubscribed`, `codelen-over-capacity` |
| `reject_incomplete` | `codelen-incomplete` |
| `reject_max_symbol_too_big` | `codelen-max-symbol-too-big` |
| `reject_oversubscribed` | `codelen-oversubscribed`, `subimage-code-oversubscribed` |
| `reject_repeat_past_end` | `codelen-repeat-past-end` |
| `reject_signature_magic` | `alph-no-vp8x`, `anim-no-vp8x`, `header-magic-wrong` |
| `reject_signature_version` | `header-version-max`, `header-version-nonzero` |
| `reject_transform_repeated` | `transform-repeated` |
| `segment_id_0` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-combo-all-features`, `lossy-segment-delta-quantizers` +5 |
| `segment_id_1` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-combo-all-features`, `lossy-segment-delta-quantizers` +5 |
| `segment_id_2` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-combo-all-features` +6 |
| `segment_id_3` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-combo-all-features` +6 |
| `simple_1sym` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-lossless-truncated`, `alph-plane-cache`, `alph-plane-filtered` +109 |
| `simple_2sym` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-plane-filtered`, `alph-plane-meta-huffman`, `alph-plane-nontrivial-red` +19 |
| `simple_sym1_oob` | `simple-dist-1sym-oob`, `simple-dist-2sym-both-oob`, `simple-dist-2sym-first-oob`, `simple-dist-sym-40-first-oob` |
| `simple_sym2_oob` | `simple-dist-2sym-both-oob`, `simple-dist-2sym-second-oob` |
| `simple_sym_1bit` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-lossless-truncated`, `simple-green-1bit-symbol` |
| `skip_i4x4_keeps_nz_dc` | `lossy-skip-i4x4-nz-dc` |
| `skip_proba` | `lossy-combo-all-features`, `lossy-proba-refresh-and-skip-zero`, `lossy-proba-skip-extremes`, `lossy-skip-all`, `lossy-skip-i4x4-nz-dc`, `lossy-skip-mixed` +1 |
| `subimage_cache` | `subimage-cache-12-invalid`, `subimage-cache-entropy-image`, `subimage-cache-palette-max`, `subimage-cache-predictor-min`, `subimage-cache-zero-invalid` |
| `subimage_stream` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-lossless-truncated`, `alph-plane-cache`, `alph-plane-filtered` +46 |
| `tf_bits_max` | `alph-lossless-byte-flipped`, `alph-lossless-predictor`, `alph-lossless-truncated`, `transform-cross-color-bits-max`, `transform-predictor-bits-max` |
| `tf_bits_min` | `lossless-all-features`, `predictor-all-16-modes`, `predictor-mode-11-select`, `predictor-mode-13-clamp-half`, `predictor-mode-14-undefined`, `predictor-mode-15-undefined` +11 |
| `tf_color_indexing` | `alph-lossless-palette`, `alph-plane-cache`, `alph-plane-filtered`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4` +18 |
| `tf_cross_color` | `lossless-all-features`, `subimage-cache-zero-invalid`, `transform-all-four`, `transform-cross-color-bits-max`, `transform-cross-color-multipliers` |
| `tf_predictor` | `alph-lossless-byte-flipped`, `alph-lossless-predictor`, `alph-lossless-truncated`, `lossless-all-features`, `predictor-all-16-modes`, `predictor-mode-11-select` +13 |
| `tf_subtract_green` | `alph-plane-two-transforms`, `lossless-all-features`, `transform-all-four`, `transform-repeated` |
| `use_length` | `codelen-max-symbol-early-stop`, `codelen-max-symbol-too-big`, `subimage-code-max-symbol` |
| `uvmode_0` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +106 |
| `uvmode_1` | `lossy-coeff-all-types`, `lossy-combo-all-features`, `lossy-filter-lf-delta`, `lossy-filter-normal-max`, `lossy-filter-simple-max`, `lossy-mode-mixed` +5 |
| `uvmode_2` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-filter-lf-delta`, `lossy-filter-normal-max` +6 |
| `uvmode_3` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-combo-all-features` +18 |
| `wht_dc_only` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +79 |
| `wht_empty` | `anim-alpha-lossless-frame`, `anim-alpha-raw-frame`, `anim-lossy-frames`, `anim-mixed-formats`, `container-duplicate-image-chunk`, `container-metadata-chunks` +34 |
| `wht_full` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-coeff-all-types`, `lossy-coeff-bands-i16` +8 |
| `ymode_0` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +91 |
| `ymode_1` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-combo-all-features` +12 |
| `ymode_2` | `anim-lossy-frames`, `container-duplicate-image-chunk`, `container-metadata-chunks`, `container-odd-chunk-payload`, `container-trailing-bytes`, `container-unknown-chunk` +33 |
| `ymode_3` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-filter-normal-max`, `lossy-filter-simple-max` +9 |

## Using them

Every file is one click away from this page, but the corpus is one
directory of a much larger repository, so to take the whole thing at once
ask git for just that directory:

    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/skal65535/skal65535.github.io.git
    cd skal65535.github.io
    git sparse-checkout set webp-corners

Run the scripts above from `webp-corners/`.

To write one of your own, or read a real encode back into a case,
[`HOWTO.md`](HOWTO.md) is the walk-through and [`SYNTAX.md`](SYNTAX.md) the
reference. The short version:

    ./src/webp_asm.py cases/alph-raw-filter-gradient.txt /tmp/out.webp
    ./src/webp_dis.py --check some-animation.webp

[`files/`](files) is pure output and is wiped on every rebuild. The only
input that is not text is [`sources/`](sources): four lossy frames made
through the encoder API, because a frame may carry up to eight token
partitions and cwebp emits only one. Nothing in this directory can write
them.

Nothing here looks for a decoder on its own account: every tool is named
in the environment, so the thing under test is always the one you meant.

| variable | what for |
| --- | --- |
| `$DWEBP` | The decoder under test -- `check.sh`, `make_hashes.sh`, `vp8_selftest.py`. Defaults to whatever `dwebp` is on `$PATH`. |
| `$ANIM_DUMP` | The animation decoder, which is the only thing that opens the animated files. libwebp does not build it by default: `cmake --build . --target anim_dump`. |
| `$WEBPINFO` | The container reader `check.sh` holds the animations to as a second opinion. Ships with libwebp. |
| `$CWEBP` | The encoder, for the half of `vp8_selftest.py` that starts from real encodes rather than from this corpus. |
| `$WEBPMUX` | The muxer, likewise, for real animations. |
| `$ASAN_DWEBP` | A sanitizer build, for `asan_sweep.sh`. |
| `$ASAN_ANIM_DUMP` | The same for the animations. Looked for beside `$ASAN_DWEBP` when unset, which is where the build puts it. |
| `$ASAN_OPTIONS` | Passed through to those two; `detect_leaks=0` unless you say otherwise. |
| `$LIBWEBP` | A libwebp git checkout, for `make_coverage.sh`, `coverage.sh` and `make_vp8_tables.py`. |
| `$PROFDATA` | `llvm-profdata`, for `coverage.sh`. Taken from `$PATH` or `xcrun` when unset. |
| `$COV` | `llvm-cov`, likewise. |
| `$SKIP_SLOW` | Set it to skip the one file that allocates a gigabyte. |

A missing tool is reported and skipped, never silently passed over.

## How they were verified

A verdict alone proves little: a file can be refused for the wrong reason,
and several of these were until the reason was measured. So nothing here
rests on having read the code.

* **What each file reaches is measured.** `make_coverage.sh` builds an
  instrumented decoder in a throwaway worktree -- probes on the exact lines
  the notes name -- runs every file through `dwebp` and, for the animations,
  `anim_dump`, and writes `coverage.txt`. Every note is written from that
  output; two were rewritten when it disagreed with them.

* **A second reader checks the first.** `webpinfo` walks every chunk and
  decodes nothing. Its verdict is recorded beside the demuxer's, so the
  files the two disagree about are a checked fact rather than a remark --
  and they disagree in one direction only, webpinfo testing things the
  demuxer never looks at.

* **The writers are checked against libwebp, not against themselves.** The
  three disassemblers read a file back into case text; assemble that, and
  the bytes have to return. `vp8_selftest.py` runs it over `sources/`, over
  images it asks cwebp to encode both ways, over animations it asks webpmux
  to build, and over the corpus itself. It also writes every coefficient
  magnitude the format allows, and pairs of frames that say the same thing
  two ways and must decode alike.

* **The source lines the notes quote are pinned.** They mean nothing except
  against the libwebp revision stamped in `coverage.txt`, so
  `src/check_refs.py` records what each cited line said and checks it still
  says it. Three were already wrong when it was written.

* **How much of the decoder this reaches is a number.** `LIBWEBP=...
  ./coverage.sh` builds an instrumented libwebp in a throwaway worktree and
  reports `src/dec` and `src/demux` three times over: the corpus as
  `check.sh` runs it, the same files through every `dwebp` output and
  scaling knob, and then through every entry point libwebp exports. At
  0be8ddd1 that is 61% of regions and 68% of lines from the files alone,
  81% and 92% with a caller driving them. The distance between the two is
  the part no bitstream decides -- output formats, rescaling, allocation
  failures -- and keeping them apart is what stops the corpus being blamed
  for paths a file cannot reach. `dec/quant_dec.c` and `dec/tree_dec.c` are
  at 100% from the files alone; `dec/io_dec.c`, which is output format and
  nothing else, is at 26% and belongs there. `coverage.sh` says when the
  numbers in this paragraph have moved.

Where that leaves the corpus: every field of the lossy frame header is
written at both ends of its range, every reachable (coefficient type, band,
context) probability cell is read, and every pair of optional tools appears
together in some frame.

The probes nothing reaches are all checks that cannot fire: the magic-byte
and version tests inside `ReadImageInfo()`, which `VP8LCheckSignature()` has
already made; and in `demux.c`, a negative loop count no 16-bit field can
hold, a complete frame carrying neither image nor alpha, and a second frame
in a file without the animation flag. One more is real but out of reach from
here -- the master-chunk table matching nothing, which `WebPGetInfo()`
refuses before the demuxer is ever called.

## What is not covered

No inter frames -- libwebp refuses them outright, so there is nothing to pin
beyond the one file that checks it does. Nothing on the encoding side
either: `WebPAnimEncoder` builds animations and no file here is written to
be read back by it.

Partial parsing. The demuxer can be asked to accept a file it has not seen
all of, and a caller that streams one uses that; every tool here hands it
the whole file, so the only thing these reach is the refusal that follows
when it is not complete. The same goes for `libwebpmux`'s own reader, which
is a third implementation of this container that nothing here runs.

What is left inside a lossy key frame is what libwebp does not act on. The
profile picks the reconstruction and loop filters in RFC 6386 and libwebp
reads it only to refuse a value above 3; the entropy-refresh bit is parsed
and dropped. Both are written anyway, so a decoder that started obeying
either would fail a pixel hash rather than pass unnoticed.

## License

BSD 3-clause, the same as libwebp. See [`COPYING`](COPYING). That covers the
generators, the scripts and the bitstreams in `files/` alike.

---

263 files, 58061 bytes total. Rebuild with `generate.py`: it assembles
everything in `cases/` through `webp_asm.py`, which hands each case to
`vp8l_asm.py` or `vp8_asm.py`, and those to `vp8l.py` and `vp8.py`.
