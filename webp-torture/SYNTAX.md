# The case syntax

Every file in [`files/`](files) but seven is assembled from one text case in
[`cases/`](cases). This is the whole vocabulary those cases use, generated
from [`src/grammar.py`](src/grammar.py), which is also where a generator
should read it from -- `./src/grammar.py` prints the same thing as JSON.

A case is a keyed comment header, then one field per line, `#` starting a
comment. Every field has a default, so a case says only what it is about:

    # note: what this file is
    # expect: ok
    # exercises: which decoder path it reaches, and why that matters
    lossless
    width 16
    code dist 200 3

**Nothing is validated or clamped.** The ranges below are what the
*bitstream field* holds, not what a decoder accepts: a value past one loses
its top bits rather than being refused, which for a torture case is usually
the point. The handful of things the assemblers do refuse are the ones they
could not write at all -- a symbol a declared code has no entry for, a tile
list that is not the length the transform implies.

The header keys are `anim`, `exercises`, `expect`, `info`, `note`, `roundtrip`, `slow`. `expect` is `ok` or `reject`; `slow` marks
the one file that allocates a gigabyte; `roundtrip: no` says the case cannot
be read back by `src/vp8_dis.py`; `anim` is the same verdict from the
animation decoder, for the files a still one refuses on sight; `info` is
webpinfo's, which is a second reader of the container and not always of the
same opinion.

Which assembler owns a case follows from its keywords: a case saying
`lossless` is a VP8L image, anything else a lossy VP8 frame, and container
keywords may be added to either. Two keywords open a block -- `frame` and
`alph_plane` -- and everything after one belongs to it until the next block
or the end of the case; that is the only nesting there is.

A value is written as a plain number -- `12`, `0x0c`, `0b1100` all work --
except where the table says otherwise. `-` in place of a number means the
field is *absent*, which is not the same as zero: it writes the flag alone.

## The lossless image (VP8L)

Written in the order the format carries them: the header, the transforms, the
color cache and entropy image, the Huffman codes, then the pixel data.

| keyword | values | range | what it is |
| --- | --- | --- | --- |
| `alpha_is_used` | 1 | `0`..`1` | a hint; libwebp drops it |
| `argb` | any | `0`..`4294967295` | whole pixels; 'V xN' repeats one |
| `cache_bits` | 1 | `0`..`15` | absent by default; 1..11 is the legal range |
| `cross_color_bits` | 1 | `2`..`9` |  |
| `cross_color_tiles` | any | `0`..`4294967295` | one multiplier triple per tile, as a color |
| `group` | 0 | -- | opens another group of five codes |
| `group_count` | 1 | `0`..`65535` | how many groups to write, when that should differ from what the entropy image asks for |
| `lossless` | 0 | -- | this case is a VP8L image, not a VP8 frame |
| `magic` | 1 | `0`..`255` | the signature byte, 0x2f; anything else is refused |
| `meta_bits` | 1 | `2`..`9` | the entropy image, absent by default |
| `meta_tiles` | any | `0`..`65535` | one Huffman-group index per tile |
| `palette_colors` | any | `0`..`4294967295` | up to 256; the stream carries their per-byte deltas |
| `pixels` | any | a word | green-code symbols, 'cache N', or 'copy LENGTH PLANE' |
| `predictor_bits` | 1 | `2`..`9` | log2 of the tile size |
| `predictor_tiles` | any | `0`..`15` | one predictor index per tile, read as 4 bits |
| `subimage` | 1 | `cross_color`, `meta`, `palette`, `predictor` | aims the 'code' lines that follow at one sub-image |
| `subimage_cache_bits` | 2 | `0`..`15` / `cross_color`, `meta`, `palette`, `predictor` | that sub-image's own color cache |
| `transforms` | any | `color_indexing`, `cross_color`, `predictor`, `subtract_green` | in the order written. A type may legally appear once |

## The lossy frame (VP8)

RFC 6386's field names, section by section, so a case reads against the
specification rather than against the decoder.

| keyword | values | range | what it is |
| --- | --- | --- | --- |
| `clamping_type` | 1 | `0`..`1` |  |
| `coeff_prob` | 4 | `*` / `0`..`255` | type, band, context and index, then the probability; '*' stands for every value of that one |
| `color_space` | 1 | `0`..`1` |  |
| `filter_type` | 1 | `0`..`1` | 1 is the simple filter |
| `frame_type` | 1 | `0`..`1` | 0 is a key frame, 1 an interframe |
| `horizontal_scale` | 1 | `0`..`3` | the top 2 bits of the width field |
| `lf_update_value` | 4 | `-63`..`63`, or `-` | one per segment |
| `log2_nbr_of_DCT_partitions` | 1 | `0`..`3` | so 1, 2, 4 or 8 partitions |
| `loop_filter_adj_enable` | 1 | `0`..`1` |  |
| `loop_filter_level` | 1 | `0`..`63` |  |
| `macroblock` | 0 | -- | opens one; 'macroblock count N' writes it N times |
| `macroblock_count` | 1 | `0`..`65535` | how many to write, when that should differ from what width and height call for |
| `mb_mode_delta` | 4 | `-63`..`63`, or `-` | one per mode class |
| `mb_no_skip_coeff` | 1 | `0`..`1` |  |
| `mode_ref_lf_delta_update` | 1 | `0`..`1` |  |
| `patch` | any | `0`..`16777215` / a word | partition0_size N | part0_bytes N | part_size I N | truncate N | truncate tokens N -- rewrites what the frame says about itself once it is assembled |
| `prob_skip_false` | 1 | `0`..`255` |  |
| `quantizer_update_value` | 4 | `-127`..`127`, or `-` | one per segment |
| `raw` | any | a word / hex digits | 'raw part0 HEX' or 'raw token N HEX': bytes a partition's own syntax cannot produce |
| `ref_frame_delta` | 4 | `-63`..`63`, or `-` | one per reference frame |
| `refresh_entropy_probs` | 1 | `0`..`1` | parsed and dropped by libwebp |
| `segment_feature_mode` | 1 | `0`..`1` | 1 absolute, 0 delta |
| `segment_prob` | 3 | `0`..`255`, or `-` | the three tree probabilities |
| `segmentation_enabled` | 1 | `0`..`1` |  |
| `sharpness_level` | 1 | `0`..`7` |  |
| `show_frame` | 1 | `0`..`1` |  |
| `start_code` | 1 | hex digits | the three bytes after the tag, normally 9d012a |
| `update_mb_segmentation_map` | 1 | `0`..`1` |  |
| `update_segment_feature_data` | 1 | `0`..`1` |  |
| `uvac_delta` | 1 | `-15`..`15`, or `-` |  |
| `uvdc_delta` | 1 | `-15`..`15`, or `-` |  |
| `vertical_scale` | 1 | `0`..`3` |  |
| `y2ac_delta` | 1 | `-15`..`15`, or `-` |  |
| `y2dc_delta` | 1 | `-15`..`15`, or `-` |  |
| `yac_qi` | 1 | `0`..`127` |  |
| `ydc_delta` | 1 | `-15`..`15`, or `-` |  |

## Both

The two image formats spell these the same way.

| keyword | values | range | what it is |
| --- | --- | --- | --- |
| `height` | 1 | `0`..`16383` |  |
| `version` | 1 | `0`..`7` | the profile |
| `width` | 1 | `0`..`16383` |  |

## Inside a macroblock

`macroblock` opens one; these fill it in. Macroblocks are laid out in raster
order, and any the frame still needs at the end are added with default
everything.

| keyword | values | range | what it is |
| --- | --- | --- | --- |
| `coeffs` | any | `-2114`..`2114` / a word | a block name -- y2, y[0]..y[15], u[0]..u[3], v[0]..v[3], or y[*] u[*] v[*] uv[*] -- then levels in zigzag order; 'N:V' jumps to position N |
| `intra_b_mode` | any | `B_DC`, `B_HD`, `B_HE`, `B_HU`, `B_LD`, `B_RD`, `B_TM`, `B_VE`, `B_VL`, `B_VR` | 16 in all, over any number of lines |
| `intra_chroma_mode` | 1 | `DC`, `H`, `TM`, `V` |  |
| `intra_y_mode` | 1 | `B`, `DC`, `H`, `TM`, `V` | B_PRED makes the macroblock 4x4 |
| `mb_skip_coeff` | 1 | `0`..`1` | needs mb_no_skip_coeff signalled with it |
| `segment_id` | 1 | `0`..`3` |  |

## Inside a group of Huffman codes

A lossless image has one group per entry of its entropy image, and each group
holds five codes: green, red, blue, alpha and dist.

| keyword | values | range | what it is |
| --- | --- | --- | --- |
| `code` | any | `alpha`, `blue`, `dist`, `green`, `red` / a word | one Huffman code; see the forms below |

## The RIFF container (RFC 9649)

A case that says nothing here gets a plain `RIFF....WEBP` around its one image
chunk. A fourcc is padded to four characters, so `VP8` means `'VP8 '`. A
payload spelled out with `payload` replaces whatever this would otherwise have
built for that chunk.

| keyword | values | range | what it is |
| --- | --- | --- | --- |
| `alph_compression` | 1 | `0`..`3` | an ALPH header field |
| `alph_data` | 1 | hex digits | the bytes after the ALPH header byte, spelled out |
| `alph_filtering` | 1 | `0`..`3` | an ALPH header field |
| `alph_plane` | 0 | -- | opens a block: the ALPH payload as a lossless image stream, with no signature of its own |
| `alph_preprocessing` | 1 | `0`..`3` | an ALPH header field |
| `alph_raw` | 1 | `0`..`4294967295` | that many bytes of uncompressed alpha plane |
| `alph_reserved` | 1 | `0`..`3` | an ALPH header field |
| `alpha` | 1 | `0`..`1` | a VP8X feature flag |
| `animation` | 1 | `0`..`1` | a VP8X feature flag |
| `canvas_height_minus_one` | 1 | `0`..`16777215` |  |
| `canvas_width_minus_one` | 1 | `0`..`16777215` |  |
| `chunk_size` | 2 | `0`..`4294967295` / a word | the same lie, per chunk; it also decides the pad byte |
| `chunks` | any | a word | the fourccs to write, in order; listing one twice is allowed |
| `exif_metadata` | 1 | `0`..`1` | a VP8X feature flag |
| `icc_profile` | 1 | `0`..`1` | a VP8X feature flag |
| `payload` | 2 | a word / hex digits | the bytes of a chunk this file has no builder for |
| `riff_size` | 1 | `0`..`4294967295` | what the RIFF header claims, when that should not be what the file holds |
| `trailing` | 1 | hex digits | bytes after the last chunk |
| `vp8x_reserved` | 1 | `0`..`16777215` | the two reserved bits and the reserved byte, raw |
| `xmp_metadata` | 1 | `0`..`1` | a VP8X feature flag |

## Animation (ANIM and ANMF)

`frame` opens a block, and everything after it belongs to that frame until the
next block or the end of the case: its ANMF header fields, its chunk list, and
the image it carries. A file with frames in it defaults to `VP8X ANIM ANMF...`
with the animation flag set and a canvas the frames fit in, so a case says only
what it is changing.

| keyword | values | range | what it is |
| --- | --- | --- | --- |
| `background_color` | 1 | `0`..`4294967295` | ARGB; read back out again and never drawn |
| `blending_method` | 1 | `0`..`1` | 1 is 'do not blend' |
| `disposal_method` | 1 | `0`..`1` | 1 clears the frame's area to the background |
| `frame` | 0 | -- | opens a block: one ANMF chunk, with its own image and its own chunk list |
| `frame_duration` | 1 | `0`..`16777215` | milliseconds |
| `frame_height_minus_one` | 1 | `0`..`16777215` |  |
| `frame_reserved` | 1 | `0`..`63` | the six bits above those two |
| `frame_width_minus_one` | 1 | `0`..`16777215` | default is the frame's own image |
| `frame_x` | 1 | `0`..`16777215` | the offset field: the pixel offset is twice it |
| `frame_y` | 1 | `0`..`16777215` |  |
| `loop_count` | 1 | `0`..`65535` | 0 means forever |

## The forms a Huffman code takes

`code NAME ...` names one of `green`, `red`, `blue`, `alpha`, `dist`, and the
word after it picks what the rest of the line means. A code the case says
nothing about covers whatever the pixel data asks of it.

| form | values | range | what it is |
| --- | --- | --- | --- |
| `code NAME cl_lengths` | any | `0`..`7` | the code-length code itself, as symbol:length pairs; without it one is built from how often each code-length symbol is used, which two encoders may do differently |
| `code NAME codelen` | any | a word | the code-length stream itself: a length 0..15, or 16xN, 17xN, 18xN for a run of N |
| `code NAME complex` | 0 | -- | the normal form, over whatever the pixels need |
| `code NAME lengths` | any | `0`..`15` | a length per symbol, positional from 0; 'N:L' jumps to symbol N, trailing zeros implied |
| `code NAME max_symbol` | 1 | `0`..`65535` | the optional early stop |
| `code NAME num_codes` | 1 | `4`..`19` | how many of the 19 code-length codes to declare |
| `code NAME simple` | any | `0`..`255` | one or two symbols, written raw and never range-checked |
| `code NAME simple1` | 1 | `0`..`1` | one symbol in a 1-bit field rather than 8 |

## The items a pixel list takes

`pixels` lists symbols of the green code, in order. `argb` spells whole pixels
instead, and the two append to the same stream.

| item | what it is |
| --- | --- |
| `V` | a green-code symbol; V xN repeats it |
| `cache` | the color-cache index N |
| `copy` | a back-reference: the length, then the plane code |

## Constants

| name | value |
| --- | --- |
| `CODE_LENGTH_CODES` | 19 |
| `MAX_ALLOWED_CODE_LENGTH` | 15 |
| `MAX_CACHE_BITS` | 11 |
| `MIN_HUFFMAN_BITS` | 2 |
| `MIN_TRANSFORM_BITS` | 2 |
| `NUM_BANDS` | 8 |
| `NUM_CTX` | 3 |
| `NUM_DISTANCE_CODES` | 40 |
| `NUM_LENGTH_CODES` | 24 |
| `NUM_LITERAL_CODES` | 256 |
| `NUM_PROBAS` | 11 |
| `NUM_TYPES` | 4 |
