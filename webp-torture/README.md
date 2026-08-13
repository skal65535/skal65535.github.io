# WebP torture bitstreams

Small WebP files that exercise corners of the format a normal encoder never
emits. 62 of them are lossless (VP8L) streams written bit by bit by
`vp8l.py`. The other 7 are lossy VP8 frames carrying multiple token
partitions, which cwebp cannot produce at all: those start from an
encoder-API call (`make_partition_sources.c`) and are then patched by
`lossy_parts.py`.

Each entry says what the reference decoder is expected to do:

* **ok** -- must decode, and must keep decoding to the same pixels. Several
  are not something cwebp can produce, so nothing else pins the behaviour.
* **reject** -- must fail cleanly and report a status, with no crash and no
  out-of-bounds access. Which status varies: a malformed Huffman code gives
  BITSTREAM_ERROR, a short partition table gives NOT_ENOUGH_DATA.

## Using them

    ./check.sh           # verdict + decoded-pixel hash for every file
    ./asan_sweep.sh      # 13 decode modes, under a sanitizer build
    ./make_coverage.sh   # regenerate coverage.txt
    python3 generate.py  # rebuild files/, expected.txt and this README

`files/` is pure output and is wiped on every rebuild. The four lossy encodes
the multi-partition cases are patched from live in `sources/`, and are
themselves rebuilt by `make_partition_sources.c`.

`check.sh` honours `$DWEBP` and `asan_sweep.sh` honours `$ASAN_DWEBP`, so both
can be pointed at any build, or at another decoder implementation; both fall
back to whatever `dwebp` is on `$PATH`. `make_coverage.sh` needs `$LIBWEBP`
set to a libwebp git checkout. `SKIP_SLOW=1` skips the one file that
allocates a gigabyte.

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

## What is not covered

The lossy VP8 syntax, apart from the partition size table. Everything
bool-coded -- segment header, filter header, quantizer deltas, intra modes,
coefficient tokens -- would need a bool encoder and is untouched. There is
also no ALPH-chunk case, since that needs a valid lossy frame to sit behind.

## License

BSD 3-clause, the same as libwebp. See `COPYING`. That covers the generators,
the scripts and the bitstreams in `files/` alike.

## Index

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
| Multiple token partitions (lossy VP8) | 7 | 4 | 3 |
| **total** | **69** | **50** | **19** |

## Simple codes

The 1-or-2-symbol shorthand a Huffman code can take. Its symbols are read as
raw 8-bit values and are never checked against the alphabet size, so this is
where a stream can say things an encoder cannot.

### `simple-dist-2sym-first-oob.webp` -- ok

Distance code: simple form, 2 symbols, the first one 200 >= alphabet_size 40.

ReadHuffmanCode() writes code_lengths[200] with alphabet_size 40; the code then
has one symbol left and is accepted. Pins the behaviour CL 8256621 documents.

### `simple-dist-2sym-second-oob.webp` -- ok

Distance code: simple form, 2 symbols, the second one 200 >= 40.

Same as above but the out-of-range symbol is the second 8-bit field.

### `simple-dist-2sym-both-oob.webp` -- reject

Distance code: both simple-form symbols out of range (200, 201).

No symbol is left inside alphabet_size, so BuildHuffmanTable() sees an empty
code and fails. Must stay a clean BITSTREAM_ERROR, not a crash.

### `simple-dist-1sym-oob.webp` -- reject

Distance code: simple form, single symbol 255, alphabet_size is 40.

The single write lands past the logical alphabet but inside the shared
max_alphabet_size buffer. Rejected because no symbol remains.

### `simple-dist-sym-39-last-valid.webp` -- ok

Distance code: single symbol 39, the last in-range value.

Boundary partner of simple-dist-sym-40-first-oob: 39 == NUM_DISTANCE_CODES - 1
must be accepted.

### `simple-dist-sym-40-first-oob.webp` -- reject

Distance code: single symbol 40, the first out-of-range value.

Exact boundary of the check that does not exist in ReadHuffmanCode(). If
someone adds one, these two files pin where it goes.

### `simple-green-1bit-symbol.webp` -- ok

Green code: simple form with first_symbol_len_code = 0, so the symbol is 1 bit
wide.

The short form of the simple code, only reachable when the symbol is 0 or 1.
cwebp emits it rarely.

### `simple-dist-2sym-duplicate.webp` -- ok

Distance code: simple form declaring 2 symbols that are the same (5, 5).

code_lengths[5] is written twice, so the code really has one symbol.
BuildHuffmanTable() takes its single-value shortcut.

### `simple-green-2sym-1bit-each.webp` -- ok

Green code with two real symbols, so every pixel costs exactly 1 bit.

The smallest non-trivial code. 4x1 pixels alternate between the two green
values.

## The code-length code

The Huffman code that describes the lengths of another Huffman code, plus its
repeat escapes (16, 17, 18) and the optional max_symbol field. cwebp only ever
emits a narrow slice of this.

### `codelen-repeat16-no-previous.webp` -- ok

Code-length stream starting with code 16 (repeat previous), before any non-zero
length was seen.

Hits DEFAULT_CODE_LENGTH: 'prev_code_len' is still 8 at vp8l_dec.c:254, so the
first symbols get length 8 out of nowhere.

### `codelen-repeat18-138-zeros.webp` -- ok

Code-length stream using code 18 with its maximum run of 138 zeros.

Longest repeat the format allows (11 + 127). Green alphabet is 280 symbols so
two of them fit.

### `codelen-repeat17-short-zeros.webp` -- ok

Code-length stream using code 17 (3..10 zeros) rather than 18.

The short zero-run escape. Its extra field is 3 bits, offset 3.

### `codelen-max-symbol-early-stop.webp` -- ok

Code-length stream with an explicit max_symbol far below the alphabet size.

ReadHuffmanCodeLengths() breaks out at vp8l_dec.c:284 with most lengths still
zero. Exercises the use_length branch cwebp never takes.

### `codelen-max-symbol-too-big.webp` -- reject

Explicit max_symbol greater than the alphabet size.

Must be caught by the max_symbol > num_symbols test at vp8l_dec.c:273.

### `codelen-repeat-past-end.webp` -- reject

A repeat run that would write past the end of the alphabet.

Must be caught by the symbol + repeat > num_symbols test at vp8l_dec.c:298.

### `codelen-num-codes-4.webp` -- ok

Only 4 code-length codes declared, the minimum the 4-bit field allows.

Restricts the code-length alphabet to {17, 18, 0, 1}, so lengths can only be 0
or 1 plus the two zero-run escapes.

### `codelen-num-codes-19.webp` -- ok

All 19 code-length codes declared.

Maximum of the 4-bit num_codes field; every entry of kCodeLengthCodeOrder[]
gets a 3-bit length.

### `codelen-depth-15.webp` -- ok

A green code containing a symbol of depth 15, MAX_ALLOWED_CODE_LENGTH.

The deepest code the format allows; forces the two-level lookup in
BuildHuffmanTable() past HUFFMAN_TABLE_BITS.

### `codelen-single-symbol-complex-form.webp` -- ok

The complex form used to describe a code with exactly one symbol.

Takes BuildHuffmanTable()'s offset[MAX_ALLOWED_CODE_LENGTH] == 1 shortcut,
which makes the code 0 bits wide.

### `codelen-over-capacity.webp` -- reject

Three symbols of depth 1, more than the two codes of that length that exist.

Caught early, by the count[len] > (1 << len) guard in BuildHuffmanTable(),
before the tree walk runs.

### `codelen-oversubscribed.webp` -- reject

Lengths 1, 2, 2, 2: each length is individually possible, but together they
over-subscribe the tree.

Slips past the per-length capacity guard and is caught later, when num_open
goes negative during the tree walk.

### `codelen-two-level-table.webp` -- ok

A green code with depths up to 10, past the 8-bit root table.

Forces BuildHuffmanTable() to allocate a second-level table and ReadSymbol() to
take its two-step lookup.

### `codelen-incomplete.webp` -- reject

A code whose lengths leave the tree incomplete (two symbols of depth 2).

Caught by the num_nodes != 2 * num_symbols - 1 test at the end of
BuildHuffmanTable().

### `codelen-all-zero-lengths.webp` -- reject

A code-length stream that assigns length 0 to every symbol.

Empty code. Different route to the same rejection as simple-dist-1sym-oob.

## Meta Huffman / entropy image

The sub-image that picks one of several code groups per tile, and the remapping
the decoder does when the group count looks implausible.

### `meta-huffman-precision-min.webp` -- ok

Meta Huffman with the smallest tile size (precision 2, 4x4 pixels).

MIN_HUFFMAN_BITS. A 16x16 image is split into 4x4 = 16 tiles, all pointing at
group 0.

### `meta-huffman-precision-max.webp` -- ok

Meta Huffman with the largest tile size (precision 9, 512x512 pixels).

MAX_HUFFMAN_BITS. One tile covers the whole image, so the entropy image is 1x1.

### `meta-huffman-two-groups.webp` -- ok

Two Huffman groups selected per tile by the entropy image.

The left half of the image uses group 0 (green 0x20), the right half group 1
(green 0xd0).

### `meta-huffman-sparse-groups.webp` -- ok

Entropy image referencing groups 0 and 900 only, leaving a 900-entry hole.

num_htree_groups_max (901) exceeds the pixel count, so ReadHuffmanCodes()
builds the mapping[] remap and the 899 unused groups take the "validate but do
not store" branch.

### `meta-huffman-1001-groups.webp` -- ok

Entropy image whose highest group index is 1000, one past the decoder's
arbitrary limit.

Crosses the num_htree_groups_max > 1000 test at vp8l_dec.c:409, which forces
the mapping[] path even when the count is plausible.

## Color cache

Size bounds, and cache-index literals.

### `cache-bits-1.webp` -- ok

Color cache with the minimum size, 1 bit (2 entries).

Lower bound of the cache_bits >= 1 check in DecodeImageStream().

### `cache-bits-11.webp` -- ok

Color cache with the maximum size, 11 bits (2048 entries).

MAX_CACHE_BITS. Also stretches the green alphabet to 280 + 2048 symbols.

### `cache-bits-0-invalid.webp` -- reject

Color cache flagged as present but with 0 bits.

Must be rejected: the format reserves "no cache" for the flag bit, so 0 is not
a legal size.

### `cache-bits-12-invalid.webp` -- reject

Color cache with 12 bits, one past MAX_CACHE_BITS.

Upper bound of the same check. The 4-bit field can hold up to 15.

### `cache-index-literal.webp` -- ok

A pixel coded as a color-cache index rather than as a literal.

Green symbols >= NUM_LITERAL_CODES + NUM_LENGTH_CODES address the cache. Pixel
2 replays pixel 1 through cache slot 0.

## Palette packing

Index width follows the palette size, and the map is padded out to the packing
capacity with black.

### `transform-palette-2-colors.webp` -- ok

Color-indexing transform with 2 colors, so 8 pixels are packed per byte.

num_colors <= 2 gives bits = 3, the densest packing, and shrinks xsize to
ceil(w / 8).

### `transform-palette-256-colors.webp` -- ok

Color-indexing transform with the full 256-entry palette.

MAX_PALETTE_SIZE, bits = 0 so there is no packing; also the largest value the
8-bit num_colors field can hold.

### `transform-palette-3-colors.webp` -- ok

Palette of 3 colors, so indices are 2 bits and 4 pixels share a byte.

num_colors in 3..4 selects bits = 2, the middle packing density. The byte 0xe4
holds indices 0, 1, 2, 3 least-significant first.

### `transform-palette-index-past-end.webp` -- ok

Palette of 3 colors addressed with index 3, which does not exist.

Reads ExpandColorMap()'s black tail (vp8l_dec.c:1412): the map is padded out to
the packing capacity of 4, so the pixel comes back transparent black instead of
out of bounds.

### `transform-palette-16-colors.webp` -- ok

Palette of 16 colors, so indices are 4 bits and 2 pixels share a byte.

The bits = 1 packing, and the largest palette that still packs.

### `transform-palette-1-color.webp` -- ok

Palette with a single color, the smallest the 8-bit field can express.

bits = 3, so 8 pixels share a byte and the map is padded from 1 entry to 2.
Index 1 is the black tail.

## Transforms

Presence, repetition and tile sizes.

### `transform-all-four.webp` -- ok

All four transforms present in one stream.

NUM_TRANSFORMS in a row: color-indexing, subtract-green, cross-color and
predictor, each with its own sub-image.

### `transform-repeated.webp` -- reject

The subtract-green transform declared twice.

Each transform type may appear once. Caught by the transforms_seen bitmask in
ReadTransform().

### `transform-predictor-bits-max.webp` -- ok

Predictor transform with bits = 9, the maximum tile size.

MIN_TRANSFORM_BITS + 7. The predictor sub-image is a single pixel for any image
up to 512x512.

## Back-references

Copy lengths and distances.

### `lz77-distance-1-run.webp` -- ok

A single literal followed by a length-8 copy at distance 1.

The degenerate overlapping copy: the copy loop reads bytes it has just written.

### `lz77-max-length-symbol.webp` -- ok

A back-reference using length symbol 23, the largest the format defines.

NUM_LENGTH_CODES - 1: 10 extra bits, copy lengths up to 4096. Here it copies
1200 pixels.

### `lz77-plane-code-1.webp` -- ok

Back-reference with plane code 1, which means "the pixel directly above".

kCodeToPlane[0] is 0x18: yoffset 1, xoffset 0, so the distance is a whole row
rather than a small number.

### `lz77-plane-code-clamped-to-1.webp` -- ok

Plane code 4 on a 1-pixel-wide image, where the 2-D offset computes to 0.

kCodeToPlane[3] is 0x19: yoffset 1, xoffset -1, so dist = xsize - 1 = 0 and the
"dist < 1 ? 1" clamp at vp8l_dec.c:173 fires. Only reachable at xsize 1.

### `lz77-plane-code-120.webp` -- ok

Plane code 120, the last entry of the 2-D offset table.

kCodeToPlane[119] is 0x70: yoffset 7, xoffset 8, so on a 16-wide image the
distance is 120. Upper bound of the mapped range.

### `lz77-distance-direct-121.webp` -- ok

Plane code 121, the first value past the table.

Distances above CODE_TO_PLANE_CODES bypass the 2-D mapping entirely: the
distance is plane_code - 120, so 121 means 1.

### `lz77-distance-past-start.webp` -- reject

A back-reference pointing further back than the pixels decoded so far.

One literal, then a copy at a distance of one whole row. Must be rejected
rather than reading before the buffer.

### `lz77-length-past-end.webp` -- reject

A copy whose length runs past the last pixel of the image.

Four literals then an 8-pixel copy in an 8-pixel image. Must be rejected rather
than writing past the buffer.

## Predictor modes

The per-tile predictor index. It is read as a 4-bit field, so all 16 values are
reachable, but the format only defines 14 of them.

### `predictor-all-16-modes.webp` -- ok

One tile per predictor index, 0 to 15, across a 64x4 image.

The mode is read as ((pixel >> 8) & 0xf) at lossless.c:247, so all 16 indices
are reachable even though the format defines only 0..13.

### `predictor-mode-14-undefined.webp` -- ok

Every tile selects predictor 14, which the format does not define.

Must decode, not crash: VP8LPredictorsAdd[14] is a padding sentinel pointing at
PredictorAdd0_C (lossless.c:653), so the tile comes out as mode 0. Shrink the
table to 14 entries and this is an out-of-bounds indirect call instead.

### `predictor-mode-15-undefined.webp` -- ok

Every tile selects predictor 15, the other undefined index.

Partner of predictor-mode-14-undefined and the largest value the 4-bit mask can
produce. Verified to decode identically to mode 14, i.e. both really do land on
PredictorAdd0_C.

### `predictor-mode-11-select.webp` -- ok

Predictor 11 (Select) over the whole image.

The only predictor with a data-dependent branch, Select() at lossless.c:100.

### `predictor-mode-13-clamp-half.webp` -- ok

Predictor 13 (ClampAddSubtractHalf) over the whole image.

Exercises AddSubtractComponentHalf() and its Clip255(), the arithmetic most
likely to differ between the C and SIMD paths.

### `predictor-single-row.webp` -- ok

A predictor transform on a one-row image.

Only the y_start == 0 shortcut runs (lossless.c:223): the first pixel takes
mode 0 and the rest mode 1, so the tile modes are never read.

### `predictor-tile-bits-min.webp` -- ok

Predictor tiles of 4x4 pixels, the smallest the format allows.

MIN_TRANSFORM_BITS, so the sub-image is as large as it can get and the mode
changes every four pixels.

## Frame header

The 14-bit dimension fields and the version escape.

### `header-width-16384.webp` -- ok

Width 16384, one past WEBP_MAX_DIMENSION.

The header stores width - 1 in 14 bits, so 16384 is expressible and the decoder
accepts it. WEBP_MAX_DIMENSION (16383) is enforced only in the encoder, at
webp_enc.c:347, so cwebp can never produce this.

### `header-max-area-bomb.webp` -- ok

34 bytes declaring 16384x16384, every pixel one color.

A bare VP8L stream has no area limit -- MAX_IMAGE_AREA is only checked in
ParseVP8X (webp_dec.c:138) -- and single-symbol codes cost zero bits per pixel,
so this decodes for real: 1.83GB peak RSS and 3.4s. The backstop is
WEBP_MAX_ALLOCABLE_MEMORY (utils.c:185), nothing earlier.

### `header-max-area-truncated.webp` -- reject

The same 16384x16384 header, cut off before the Huffman codes.

Must fail on the missing data rather than allocating the gigabyte first.
Partner of header-max-area-bomb: together they say where the allocation sits
relative to the parse.

### `header-version-nonzero.webp` -- reject

Header with the 3-bit version field set to 1.

Rejected by VP8LCheckSignature() at vp8l_dec.c:111, which tests (data[4] >> 5)
before ReadImageInfo() ever runs. The version field is the format's only
forward-compatibility escape.

## Multiple token partitions (lossy VP8)

The only non-VP8L group. A lossy frame may carry 1, 2, 4 or 8 token partitions,
but cwebp does not expose config.partitions and libwebp forces it back to 1
whenever the token path is used (webp_enc.c:124), so these are hard to come by.
The corrupted ones rewrite the raw size table that follows partition 0.

### `lossy-1-partitions.webp` -- ok

A single token partition: the default, and the control for the others.

Same encode settings as the 2/4/8 files, so a size or hash difference against
them is entirely the partitioning.

### `lossy-2-partitions.webp` -- ok

A plain 2-partition lossy frame.

cwebp never emits this: config.partitions is API-only and is forced back to 1
for method >= 3 unless low_memory is set.

### `lossy-4-partitions.webp` -- ok

A plain 4-partition lossy frame.

Same, with the size table holding three entries.

### `lossy-8-partitions.webp` -- ok

A plain 8-partition lossy frame, the maximum the 2-bit field allows.

MAX_NUM_PARTITIONS. Seven 3-byte size-table entries, and eight independent bit-
readers in the decoder.

### `lossy-8-partitions-size-overflow.webp` -- reject

Eight partitions whose first declared size is 0xffffff, far past the data.

Hits the "if (psize > size_left) psize = size_left" clamp in ParsePartitions():
partition 0 swallows the whole remainder and the other seven get zero-length
readers.

### `lossy-8-partitions-zero-sizes.webp` -- reject

Eight partitions all declared as zero bytes long.

Every token partition but the last is empty, so the last one is handed the
whole remainder. Legal to parse, garbage to decode.

### `lossy-8-partitions-sizes-sum-past-end.webp` -- reject

Eight partitions whose declared sizes add up to more than the chunk holds.

The clamp fires part-way through the loop, so later partitions get zero-length
readers while earlier ones look valid.

---

69 files, 33558 bytes total. Rebuild with `generate.py`; the VP8L bitstream
writer is `vp8l.py` and the lossy cases are in `lossy_parts.py`.
