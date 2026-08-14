# WebP torture bitstreams

Small WebP files that exercise corners of the format a normal encoder never
emits, one layer of it at a time.

**They are written, not captured.** Each one is a text case in
[`cases/`](cases) naming bitstream fields, one per line, under the names the
specification gives them; [`SYNTAX.md`](SYNTAX.md) is the grammar, generated
from [`src/grammar.py`](src/grammar.py); and an assembler in [`src/`](src)
turns the case into bytes. Nothing is checked on the way, so a case can say
what no encoder would -- and the file it produces is readable as the text
that describes it.

* **80 lossless (VP8L) streams**
  ([`vp8l_asm.py`](src/vp8l_asm.py)): the header, the transforms, the
  Huffman codes down to the code-length stream that describes them, and the
  pixel data.
* **81 lossy VP8 frames** ([`vp8_asm.py`](src/vp8_asm.py)), under the
  field names [RFC 6386](https://www.rfc-editor.org/rfc/rfc6386.html) gives
  them. Seven of them are the exception to the paragraph above: a frame may
  carry up to eight token partitions and cwebp emits only one, so
  [`make_partition_sources.c`](src/make_partition_sources.c) makes four
  through the encoder API into [`sources/`](sources), and
  [`lossy_parts.py`](src/lossy_parts.py) turns those into seven -- the four
  as they are, plus three with the partition-size table rewritten.
* **19 RIFF containers** ([`webp_asm.py`](src/webp_asm.py)), in
  [RFC 9649](https://www.rfc-editor.org/rfc/rfc9649.html)'s names: the
  extended-format VP8X chunk, the optional chunks a decoder must step over,
  and sizes that lie about what is behind them.
* **29 alpha chunks**, where the plane is either stored one byte per
  pixel or compressed with the lossless coder in its 8-bit mode -- a
  different path through the decoder from the one every VP8L file here
  takes. A compressed plane is a lossless image stream without a header, so
  `vp8l_asm.py` writes those too.
* **48 animations** ([`webp_asm.py`](src/webp_asm.py) again): an ANIM
  chunk and one ANMF per frame, each frame carrying its own image, position,
  duration, disposal and blending. No still decoder will open one, so these
  are checked with `anim_dump` instead.

Each note below is its case's own, and says what the reference decoder is
expected to do with it:

* **ok** -- must decode, and must keep decoding to the same pixels. Several
  are not something cwebp can produce, so nothing else pins the behaviour.
* **reject** -- must fail cleanly and report a status, with no crash and no
  out-of-bounds access. Which status varies: a malformed Huffman code gives
  BITSTREAM_ERROR, a short partition table gives NOT_ENOUGH_DATA.

An animation carries both verdicts, written `reject, anim_dump ok`. dwebp
returns UNSUPPORTED_FEATURE for any file claiming animation, before it looks
at a frame, so the first half of that is the same for every one of them and
the second half is the one about the file.

## The bitstreams

The whole set lives in **[`files/`](files)**, which lists each one with its
size and expected verdict. Every note below links straight to the file it is
about.

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
| Back-references | 8 | 6 | 2 |
| Predictor modes | 7 | 7 | 0 |
| Frame header | 7 | 3 | 4 |
| The RIFF container | 19 | 9 | 10 |
| Animation | 48 | 27 | 21 |
| The alpha chunk | 29 | 19 | 10 |
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
| **total** | **257** | **176** | **81** |

## What to run

The first two are the point: they run every file here through a decoder and say
whether it behaved. The rest rebuild the corpus or check the tools that write
it.

| file | what it is |
| --- | --- |
| [`check.sh`](check.sh) | Decodes every file and checks the verdict and the pixels, through `dwebp` or -- for the animations -- `$ANIM_DUMP` and `$WEBPINFO`. The one to run. |
| [`asan_sweep.sh`](asan_sweep.sh) | The same, in 14 output modes under a sanitizer build. Point `$ASAN_DWEBP` at one. |
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
| [`src/grammar.py`](src/grammar.py) | Every keyword and the range of every value, as data. `SYNTAX.md` is generated from it. |
| [`src/vp8_tables.py`](src/vp8_tables.py) | The VP8 constant tables, extracted from libwebp. |
| [`src/make_vp8_tables.py`](src/make_vp8_tables.py) | Extracts them, so they are never retyped. |
| [`src/lossy_parts.py`](src/lossy_parts.py) | The multi-partition lossy cases, patched from `sources/`. |
| [`src/make_partition_sources.c`](src/make_partition_sources.c) | Rebuilds `sources/`: cwebp cannot emit more than one token partition. |
| [`src/probes.py`](src/probes.py) | The `fprintf` probes `make_coverage.sh` patches in. |
| [`src/check_refs.py`](src/check_refs.py) | Checks that the source lines the notes point at still say what the notes claim. |

## The data

| file | what it is |
| --- | --- |
| [`SYNTAX.md`](SYNTAX.md) | The whole case syntax, generated from `src/grammar.py`. |
| [`expected.txt`](expected.txt) | Name and expected verdict, one line per file. |
| [`hashes.txt`](hashes.txt) | SHA-256 of each decoding file's `-pam` output, so a silent change in decoded pixels fails too. |
| [`coverage.txt`](coverage.txt) | Which decoder path each file actually reached. |
| [`refs.txt`](refs.txt) | What each line a note points at said when it was written. |
| [`COPYING`](COPYING) | BSD 3-clause, the same as libwebp. |

## Using them

Every file is one click away from this page, but the corpus is one
directory of a much larger repository, so to take the whole thing at once
ask git for just that directory:

    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/skal65535/skal65535.github.io.git
    cd skal65535.github.io
    git sparse-checkout set webp-torture

That fetches about 2MB instead of the ~90MB the rest of the site comes to.
Run the scripts above from `webp-torture/`.

Every field of a case has a default, so it says only what it is about, and
a value too big for its field loses its top bits rather than being refused.
`./src/grammar.py` prints the grammar as JSON, which is what a generator
should read rather than the prose.

Any one case, or any real encode read back into a case:

    ./src/webp_asm.py cases/alph-raw-filter-gradient.txt /tmp/out.webp
    ./src/vp8_asm.py cases/lossy-coeff-cat6.txt /tmp/out.webp
    ./src/vp8l_asm.py cases/codelen-depth-15.txt /tmp/out.webp
    ./src/vp8_dis.py some-photo.webp
    ./src/vp8l_dis.py --check some-lossless.webp

[`files/`](files) is pure output and is wiped on every rebuild. The only
input here that is not text is [`sources/`](sources), the four frames above:
nothing in this directory can write them, only the encoder can.

`check.sh`, `make_hashes.sh` and `vp8_selftest.py` honour `$DWEBP`,
`asan_sweep.sh` honours `$ASAN_DWEBP`, and both fall back to whatever
`dwebp` is on `$PATH`. The animation files need two more: `$ANIM_DUMP` for
`check.sh` and `make_hashes.sh`, which `asan_sweep.sh` instead looks for
beside `$ASAN_DWEBP`, and `$WEBPINFO` for `check.sh`. `webpinfo` ships with
libwebp; `anim_dump` does not, and is built with `cmake --build . --target
anim_dump`. Without either, `check.sh` carries on and says what it skipped.
`make_coverage.sh` and `make_vp8_tables.py` need `$LIBWEBP` set to a libwebp
git checkout. `SKIP_SLOW=1` skips the one file that allocates a gigabyte.

## How they were verified

Verdicts alone prove little -- a file can be rejected for the wrong reason,
and several of these were before being corrected. Every file is also run
against decoders instrumented with probes on the exact lines the notes
name; `coverage.txt` records which paths each one actually reached, and
`make_coverage.sh` regenerates it from `src/probes.py` in a throwaway
worktree, through `dwebp` and -- for the animations, which dwebp cannot
open -- `anim_dump`. The notes are written from that output, not from
reading the code. Two were rewritten because of it: an ANMF with no image
turned out to be dropped rather than refused, and a frame claiming an
impossible area was being caught by the canvas check one layer up.

The animation files are read a third time, by `webpinfo`, which walks every
chunk of the container and decodes nothing. Its verdict is recorded beside
the others and checked with them, so a file the two readers disagree about
is a checked fact rather than a remark, and its heading says so. The
disagreements all have one shape -- webpinfo tests something the demuxer
does not look at: a frame whose ANMF header size disagrees with the image
inside it, an alpha frame in a file that does not set the alpha flag, an
ANIM chunk with padding after its two fields.

The source line numbers the notes quote are only meaningful against one
revision of libwebp: the one stamped at the top of `coverage.txt`.
`src/check_refs.py` records what each cited line said and checks it still
says it, so upstream moving underneath a note is reported rather than
discovered later. Three were already wrong when that check was written.

Both writers are checked against libwebp rather than against themselves:
`src/vp8_dis.py` and `src/vp8l_dis.py` read a file back into the text the
assemblers take, so a real encode can be disassembled, reassembled and
compared byte for byte. `vp8_selftest.py` runs that over `sources/`, over a
spread of images it asks cwebp to encode both ways, and over the corpus
itself; it also writes every coefficient magnitude the format allows, and
pairs of frames that say the same thing two different ways and must decode
alike.

What the corpus reaches is measured rather than assumed, and the measurement
is what says where to add files next. As it stands: every field of the lossy
frame header is written at both ends of its range, every reachable
(coefficient type, band, context) probability cell is read, and every pair
of optional tools appears together in some frame.

What the probes do *not* reach is worth naming, because all of it turned out
to be checks that cannot fire. The magic-byte and version tests inside
`ReadImageInfo()` have already been made by `VP8LCheckSignature()` by the
time they run. In `demux.c`: a negative loop count, which a 16-bit unsigned
field cannot produce; a complete frame carrying neither an image nor an
alpha chunk, which is the one thing that stops a frame being added at all,
so it can never be found later; and a second frame in a file without the
animation flag, which cannot be reached because nothing without that flag
ever numbers a frame past the first. One more, the master-chunk table
matching nothing, is real but out of reach from here: every tool checks the
format with `WebPGetInfo()` first, and that refuses such a file before the
demuxer is called.

**The counts are deliberately not repeated here.** They change with every
file added, and prose does not: `check.sh`, `vp8_selftest.py` and
`make_coverage.sh` each print what they covered, and `coverage.txt`,
`hashes.txt` and `expected.txt` are the record.

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

## Whole lossless images

The two ends of the format rather than one corner of it: an image with nothing
optional in it at all, and one with everything. Between them they are what the
rest of the lossless files here are a departure from.

### [`lossless-all-features.webp`](files/lossless-all-features.webp) -- ok -- from [`lossless-all-features.txt`](cases/lossless-all-features.txt)

Every optional part of the lossless format in one image.

The maximal stack: all four transforms, a colour cache, an entropy image
selecting between two Huffman groups, and pixel data using literals, a cache
index and a back-reference. transform-all-four has the transforms alone; this
adds everything that is read after them, so the header walk of
DecodeImageStream() runs to its full length and the pixel loop consults the
entropy image, the cache and the copy path in turn.

### [`lossless-plain.webp`](files/lossless-plain.webp) -- ok -- from [`lossless-plain.txt`](cases/lossless-plain.txt)

An ordinary lossless image, with none of the corners.

The control. No transform, no colour cache, no entropy image: whole pixels
straight through, which is the one shape every other lossless file here departs
from. The four colours differ in all four channels, so the red, blue and alpha
codes each carry several symbols and are read once per pixel -- the path a
stream takes when is_trivial_literal is false, which a single-colour image
never reaches.

## Simple codes

The 1-or-2-symbol shorthand a Huffman code can take. Its symbols are read as
raw 8-bit values and are never checked against the alphabet size, so this is
where a stream can say things an encoder cannot.

### [`simple-dist-1sym-oob.webp`](files/simple-dist-1sym-oob.webp) -- reject -- from [`simple-dist-1sym-oob.txt`](cases/simple-dist-1sym-oob.txt)

Distance code: simple form, single symbol 255, alphabet_size is 40.

The single write lands past the logical alphabet but inside the shared
max_alphabet_size buffer. Rejected because no symbol remains.

### [`simple-dist-2sym-both-oob.webp`](files/simple-dist-2sym-both-oob.webp) -- reject -- from [`simple-dist-2sym-both-oob.txt`](cases/simple-dist-2sym-both-oob.txt)

Distance code: both simple-form symbols out of range (200, 201).

No symbol is left inside alphabet_size, so BuildHuffmanTable() sees an empty
code and fails. Must stay a clean BITSTREAM_ERROR, not a crash.

### [`simple-dist-2sym-duplicate.webp`](files/simple-dist-2sym-duplicate.webp) -- ok -- from [`simple-dist-2sym-duplicate.txt`](cases/simple-dist-2sym-duplicate.txt)

Distance code: simple form declaring 2 symbols that are the same (5, 5).

code_lengths[5] is written twice, so the code really has one symbol.
BuildHuffmanTable() takes its single-value shortcut.

### [`simple-dist-2sym-first-oob.webp`](files/simple-dist-2sym-first-oob.webp) -- ok -- from [`simple-dist-2sym-first-oob.txt`](cases/simple-dist-2sym-first-oob.txt)

Distance code: simple form, 2 symbols, the first one 200 >= alphabet_size 40.

ReadHuffmanCode() writes code_lengths[200] with alphabet_size 40; the code then
has one symbol left and is accepted. Pins the behaviour CL 8256621 documents.

### [`simple-dist-2sym-second-oob.webp`](files/simple-dist-2sym-second-oob.webp) -- ok -- from [`simple-dist-2sym-second-oob.txt`](cases/simple-dist-2sym-second-oob.txt)

Distance code: simple form, 2 symbols, the second one 200 >= 40.

Same as above but the out-of-range symbol is the second 8-bit field.

### [`simple-dist-sym-39-last-valid.webp`](files/simple-dist-sym-39-last-valid.webp) -- ok -- from [`simple-dist-sym-39-last-valid.txt`](cases/simple-dist-sym-39-last-valid.txt)

Distance code: single symbol 39, the last in-range value.

Boundary partner of simple-dist-sym-40-first-oob: 39 == NUM_DISTANCE_CODES - 1
must be accepted.

### [`simple-dist-sym-40-first-oob.webp`](files/simple-dist-sym-40-first-oob.webp) -- reject -- from [`simple-dist-sym-40-first-oob.txt`](cases/simple-dist-sym-40-first-oob.txt)

Distance code: single symbol 40, the first out-of-range value.

Exact boundary of the check that does not exist in ReadHuffmanCode(). If
someone adds one, these two files pin where it goes.

### [`simple-green-1bit-symbol.webp`](files/simple-green-1bit-symbol.webp) -- ok -- from [`simple-green-1bit-symbol.txt`](cases/simple-green-1bit-symbol.txt)

Green code: simple form with first_symbol_len_code = 0, so the symbol is 1 bit
wide.

The short form of the simple code, only reachable when the symbol is 0 or 1.
cwebp emits it rarely.

### [`simple-green-2sym-1bit-each.webp`](files/simple-green-2sym-1bit-each.webp) -- ok -- from [`simple-green-2sym-1bit-each.txt`](cases/simple-green-2sym-1bit-each.txt)

Green code with two real symbols, so every pixel costs exactly 1 bit.

The smallest non-trivial code. 4x1 pixels alternate between the two green
values.

## The code-length code

The Huffman code that describes the lengths of another Huffman code, plus its
repeat escapes (16, 17, 18) and the optional max_symbol field. cwebp only ever
emits a narrow slice of this.

### [`codelen-all-zero-lengths.webp`](files/codelen-all-zero-lengths.webp) -- reject -- from [`codelen-all-zero-lengths.txt`](cases/codelen-all-zero-lengths.txt)

A code-length stream that assigns length 0 to every symbol.

Empty code. Different route to the same rejection as simple-dist-1sym-oob.

### [`codelen-depth-15.webp`](files/codelen-depth-15.webp) -- ok -- from [`codelen-depth-15.txt`](cases/codelen-depth-15.txt)

A green code containing a symbol of depth 15, MAX_ALLOWED_CODE_LENGTH.

The deepest code the format allows; forces the two-level lookup in
BuildHuffmanTable() past HUFFMAN_TABLE_BITS.

### [`codelen-incomplete.webp`](files/codelen-incomplete.webp) -- reject -- from [`codelen-incomplete.txt`](cases/codelen-incomplete.txt)

A code whose lengths leave the tree incomplete (two symbols of depth 2).

Caught by the num_nodes != 2 * num_symbols - 1 test at the end of
BuildHuffmanTable().

### [`codelen-max-symbol-early-stop.webp`](files/codelen-max-symbol-early-stop.webp) -- ok -- from [`codelen-max-symbol-early-stop.txt`](cases/codelen-max-symbol-early-stop.txt)

Code-length stream with an explicit max_symbol far below the alphabet size.

ReadHuffmanCodeLengths() breaks out at vp8l_dec.c:284 with most lengths still
zero. Exercises the use_length branch cwebp never takes.

### [`codelen-max-symbol-too-big.webp`](files/codelen-max-symbol-too-big.webp) -- reject -- from [`codelen-max-symbol-too-big.txt`](cases/codelen-max-symbol-too-big.txt)

Explicit max_symbol greater than the alphabet size.

Must be caught by the max_symbol > num_symbols test at vp8l_dec.c:273.

### [`codelen-num-codes-19.webp`](files/codelen-num-codes-19.webp) -- ok -- from [`codelen-num-codes-19.txt`](cases/codelen-num-codes-19.txt)

All 19 code-length codes declared.

Maximum of the 4-bit num_codes field; every entry of kCodeLengthCodeOrder[]
gets a 3-bit length.

### [`codelen-num-codes-4.webp`](files/codelen-num-codes-4.webp) -- ok -- from [`codelen-num-codes-4.txt`](cases/codelen-num-codes-4.txt)

Only 4 code-length codes declared, the minimum the 4-bit field allows.

Restricts the code-length alphabet to {17, 18, 0, 1}, so lengths can only be 0
or 1 plus the two zero-run escapes.

### [`codelen-over-capacity.webp`](files/codelen-over-capacity.webp) -- reject -- from [`codelen-over-capacity.txt`](cases/codelen-over-capacity.txt)

Three symbols of depth 1, more than the two codes of that length that exist.

Caught early, by the count[len] > (1 << len) guard in BuildHuffmanTable(),
before the tree walk runs.

### [`codelen-oversubscribed.webp`](files/codelen-oversubscribed.webp) -- reject -- from [`codelen-oversubscribed.txt`](cases/codelen-oversubscribed.txt)

Lengths 1, 2, 2, 2: each length is individually possible, but together they
over-subscribe the tree.

Slips past the per-length capacity guard and is caught later, when num_open
goes negative during the tree walk.

### [`codelen-repeat-past-end.webp`](files/codelen-repeat-past-end.webp) -- reject -- from [`codelen-repeat-past-end.txt`](cases/codelen-repeat-past-end.txt)

A repeat run that would write past the end of the alphabet.

Must be caught by the symbol + repeat > num_symbols test at vp8l_dec.c:298.

### [`codelen-repeat16-no-previous.webp`](files/codelen-repeat16-no-previous.webp) -- ok -- from [`codelen-repeat16-no-previous.txt`](cases/codelen-repeat16-no-previous.txt)

Code-length stream starting with code 16 (repeat previous), before any non-zero
length was seen.

Hits DEFAULT_CODE_LENGTH: 'prev_code_len' is still 8 at vp8l_dec.c:254, so the
first symbols get length 8 out of nowhere.

### [`codelen-repeat17-short-zeros.webp`](files/codelen-repeat17-short-zeros.webp) -- ok -- from [`codelen-repeat17-short-zeros.txt`](cases/codelen-repeat17-short-zeros.txt)

Code-length stream using code 17 (3..10 zeros) rather than 18.

The short zero-run escape. Its extra field is 3 bits, offset 3.

### [`codelen-repeat18-138-zeros.webp`](files/codelen-repeat18-138-zeros.webp) -- ok -- from [`codelen-repeat18-138-zeros.txt`](cases/codelen-repeat18-138-zeros.txt)

Code-length stream using code 18 with its maximum run of 138 zeros.

Longest repeat the format allows (11 + 127). Green alphabet is 280 symbols so
two of them fit.

### [`codelen-single-symbol-complex-form.webp`](files/codelen-single-symbol-complex-form.webp) -- ok -- from [`codelen-single-symbol-complex-form.txt`](cases/codelen-single-symbol-complex-form.txt)

The complex form used to describe a code with exactly one symbol.

Takes BuildHuffmanTable()'s offset[MAX_ALLOWED_CODE_LENGTH] == 1 shortcut,
which makes the code 0 bits wide.

### [`codelen-two-level-table.webp`](files/codelen-two-level-table.webp) -- ok -- from [`codelen-two-level-table.txt`](cases/codelen-two-level-table.txt)

A green code with depths up to 10, past the 8-bit root table.

Forces BuildHuffmanTable() to allocate a second-level table and ReadSymbol() to
take its two-step lookup.

## Meta Huffman / entropy image

The sub-image that picks one of several code groups per tile, and the remapping
the decoder does when the group count looks implausible.

### [`meta-huffman-1001-groups.webp`](files/meta-huffman-1001-groups.webp) -- ok -- from [`meta-huffman-1001-groups.txt`](cases/meta-huffman-1001-groups.txt)

Entropy image whose highest group index is 1000, one past the decoder's
arbitrary limit.

Crosses the num_htree_groups_max > 1000 test at vp8l_dec.c:409, which forces
the mapping[] path even when the count is plausible.

### [`meta-huffman-groups-truncated.webp`](files/meta-huffman-groups-truncated.webp) -- reject -- from [`meta-huffman-groups-truncated.txt`](cases/meta-huffman-groups-truncated.txt)

An entropy image naming group 1 when only one group of codes follows.

num_htree_groups is derived from the entropy image rather than declared, so the
decoder reads a second group out of a stream that carries none and takes
whatever the pixel data happens to say as a Huffman code. It comes out empty,
which BuildHuffmanTable() refuses -- a clean failure on data that was never a
code at all.

### [`meta-huffman-per-tile-data.webp`](files/meta-huffman-per-tile-data.webp) -- ok -- from [`meta-huffman-per-tile-data.txt`](cases/meta-huffman-per-tile-data.txt)

Two Huffman groups that both carry real data, so the code in use changes four
pixels into every row.

Every other meta-Huffman file here has groups whose codes cost no bits, so none
of them proves the entropy image is consulted at all. Here each group holds two
symbols of one bit and the halves of a row decode to different values, so a
decoder that used group 0 throughout would come out with the wrong pixels
rather than the right ones by luck.

### [`meta-huffman-precision-max.webp`](files/meta-huffman-precision-max.webp) -- ok -- from [`meta-huffman-precision-max.txt`](cases/meta-huffman-precision-max.txt)

Meta Huffman with the largest tile size (precision 9, 512x512 pixels).

MAX_HUFFMAN_BITS. One tile covers the whole image, so the entropy image is 1x1.

### [`meta-huffman-precision-min.webp`](files/meta-huffman-precision-min.webp) -- ok -- from [`meta-huffman-precision-min.txt`](cases/meta-huffman-precision-min.txt)

Meta Huffman with the smallest tile size (precision 2, 4x4 pixels).

MIN_HUFFMAN_BITS. A 16x16 image is split into 4x4 = 16 tiles, all pointing at
group 0.

### [`meta-huffman-sparse-groups.webp`](files/meta-huffman-sparse-groups.webp) -- ok -- from [`meta-huffman-sparse-groups.txt`](cases/meta-huffman-sparse-groups.txt)

Entropy image referencing groups 0 and 900 only, leaving a 900-entry hole.

num_htree_groups_max (901) exceeds the pixel count, so ReadHuffmanCodes()
builds the mapping[] remap and the 899 unused groups take the "validate but do
not store" branch.

### [`meta-huffman-two-groups.webp`](files/meta-huffman-two-groups.webp) -- ok -- from [`meta-huffman-two-groups.txt`](cases/meta-huffman-two-groups.txt)

Two Huffman groups selected per tile by the entropy image.

The left half of the image uses group 0 (green 0x20), the right half group 1
(green 0xd0).

## Color cache

Size bounds, and cache-index literals.

### [`cache-bits-0-invalid.webp`](files/cache-bits-0-invalid.webp) -- reject -- from [`cache-bits-0-invalid.txt`](cases/cache-bits-0-invalid.txt)

Color cache flagged as present but with 0 bits.

Must be rejected: the format reserves "no cache" for the flag bit, so 0 is not
a legal size.

### [`cache-bits-1.webp`](files/cache-bits-1.webp) -- ok -- from [`cache-bits-1.txt`](cases/cache-bits-1.txt)

Color cache with the minimum size, 1 bit (2 entries).

Lower bound of the cache_bits >= 1 check in DecodeImageStream().

### [`cache-bits-11.webp`](files/cache-bits-11.webp) -- ok -- from [`cache-bits-11.txt`](cases/cache-bits-11.txt)

Color cache with the maximum size, 11 bits (2048 entries).

MAX_CACHE_BITS. Also stretches the green alphabet to 280 + 2048 symbols.

### [`cache-bits-12-invalid.webp`](files/cache-bits-12-invalid.webp) -- reject -- from [`cache-bits-12-invalid.txt`](cases/cache-bits-12-invalid.txt)

Color cache with 12 bits, one past MAX_CACHE_BITS.

Upper bound of the same check. The 4-bit field can hold up to 15.

### [`cache-index-literal.webp`](files/cache-index-literal.webp) -- ok -- from [`cache-index-literal.txt`](cases/cache-index-literal.txt)

A pixel coded as a color-cache index rather than as a literal.

Green symbols >= NUM_LITERAL_CODES + NUM_LENGTH_CODES address the cache. Pixel
2 replays pixel 1 through cache slot 0.

## Sub-images

A lossless file carries whole image streams inside itself: one for each
transform that needs a per-tile parameter, and one for the entropy image. Each
is read by the same DecodeImageStream() as the outer image, minus the
transforms and the entropy image it is not allowed to have of its own -- so
each has a color cache and five Huffman codes that a file can say something
about, and that cwebp always writes the same dull way.

### [`subimage-cache-12-invalid.webp`](files/subimage-cache-12-invalid.webp) -- reject -- from [`subimage-cache-12-invalid.txt`](cases/subimage-cache-12-invalid.txt)

A sub-image color cache of 12 bits, one past MAX_CACHE_BITS.

Upper bound of the same check, one level down. The 4-bit field can hold up to
15.

### [`subimage-cache-entropy-image.webp`](files/subimage-cache-entropy-image.webp) -- ok -- from [`subimage-cache-entropy-image.txt`](cases/subimage-cache-entropy-image.txt)

A color cache inside the entropy image itself.

The entropy image is read by DecodeImageStream() through ReadHuffmanCodes(), so
it too may declare a cache. The image it selects groups for has none.

### [`subimage-cache-palette-max.webp`](files/subimage-cache-palette-max.webp) -- ok -- from [`subimage-cache-palette-max.txt`](cases/subimage-cache-palette-max.txt)

A color cache of 11 bits inside the palette sub-image of the color-indexing
transform.

MAX_CACHE_BITS one level down, which stretches that stream's green alphabet to
280 + 2048 symbols in order to describe two palette entries. The outer image
has no cache, so the two alphabets differ within one file.

### [`subimage-cache-predictor-min.webp`](files/subimage-cache-predictor-min.webp) -- ok -- from [`subimage-cache-predictor-min.txt`](cases/subimage-cache-predictor-min.txt)

A color cache declared inside the predictor transform's sub-image, 1 bit.

DecodeImageStream() reads the cache flag at every level, so a sub-image may
carry one of its own. cwebp never writes one there. Verified to decode to the
same pixels as the same transform without it.

### [`subimage-cache-zero-invalid.webp`](files/subimage-cache-zero-invalid.webp) -- reject -- from [`subimage-cache-zero-invalid.txt`](cases/subimage-cache-zero-invalid.txt)

A sub-image color cache flagged present but with 0 bits.

The cache_bits >= 1 test sits in DecodeImageStream() at vp8l_dec.c:1576, so it
guards every level. Partner of cache-bits-0-invalid, which trips the same line
at level 0.

### [`subimage-code-complex-form.webp`](files/subimage-code-complex-form.webp) -- ok -- from [`subimage-code-complex-form.txt`](cases/subimage-code-complex-form.txt)

The predictor sub-image's green code written with the code-length repeat
escapes.

A sub-image's five codes are a full ReadHuffmanCode() each, with nothing
special-cased about them. The form the automatic path picks spells all 280
lengths out one by one; this one says the same thing in 18 symbols, two of them
code 18 skipping 138 and 126 unused entries. 32 bytes shorter than predictor-
all-16-modes and verified to decode to the same pixels.

### [`subimage-code-empty.webp`](files/subimage-code-empty.webp) -- reject -- from [`subimage-code-empty.txt`](cases/subimage-code-empty.txt)

A sub-image code-length stream that really does assign length 0 to all 280
symbols.

Empty code inside the predictor sub-image, and a third route into the rejection
codelen-all-zero-lengths and simple-dist-1sym-oob reach from level 0. Getting
there took saying it in full: a shorter run leaves the rest of the alphabet
unread, and the decoder trips over the next code instead.

### [`subimage-code-max-symbol.webp`](files/subimage-code-max-symbol.webp) -- ok -- from [`subimage-code-max-symbol.txt`](cases/subimage-code-max-symbol.txt)

A sub-image code using the explicit max_symbol early stop.

ReadHuffmanCodeLengths()'s use_length branch, reached through a transform sub-
image rather than at level 0. cwebp emits it nowhere.

### [`subimage-code-oversubscribed.webp`](files/subimage-code-oversubscribed.webp) -- reject -- from [`subimage-code-oversubscribed.txt`](cases/subimage-code-oversubscribed.txt)

A sub-image Huffman code with lengths 1, 2, 2, 2, which over-subscribes the
tree.

The same defect as codelen-oversubscribed, one level down: BuildHuffmanTable()
fails inside the predictor sub-image, and the failure has to travel back out
through DecodeImageStream() and ReadTransform() rather than being noticed at
the top.

## Palette packing

Index width follows the palette size, and the map is padded out to the packing
capacity with black.

### [`transform-palette-1-color.webp`](files/transform-palette-1-color.webp) -- ok -- from [`transform-palette-1-color.txt`](cases/transform-palette-1-color.txt)

Palette with a single color, the smallest the 8-bit field can express.

bits = 3, so 8 pixels share a byte and the map is padded from 1 entry to 2.
Index 1 is the black tail.

### [`transform-palette-16-colors.webp`](files/transform-palette-16-colors.webp) -- ok -- from [`transform-palette-16-colors.txt`](cases/transform-palette-16-colors.txt)

Palette of 16 colors, so indices are 4 bits and 2 pixels share a byte.

The bits = 1 packing, and the largest palette that still packs.

### [`transform-palette-2-colors.webp`](files/transform-palette-2-colors.webp) -- ok -- from [`transform-palette-2-colors.txt`](cases/transform-palette-2-colors.txt)

Color-indexing transform with 2 colors, so 8 pixels are packed per byte.

num_colors <= 2 gives bits = 3, the densest packing, and shrinks xsize to
ceil(w / 8).

### [`transform-palette-256-colors.webp`](files/transform-palette-256-colors.webp) -- ok -- from [`transform-palette-256-colors.txt`](cases/transform-palette-256-colors.txt)

Color-indexing transform with the full 256-entry palette.

MAX_PALETTE_SIZE, bits = 0 so there is no packing; also the largest value the
8-bit num_colors field can hold.

### [`transform-palette-3-colors.webp`](files/transform-palette-3-colors.webp) -- ok -- from [`transform-palette-3-colors.txt`](cases/transform-palette-3-colors.txt)

Palette of 3 colors, so indices are 2 bits and 4 pixels share a byte.

num_colors in 3..4 selects bits = 2, the middle packing density. The byte 0xe4
holds indices 0, 1, 2, 3 least-significant first.

### [`transform-palette-index-past-end.webp`](files/transform-palette-index-past-end.webp) -- ok -- from [`transform-palette-index-past-end.txt`](cases/transform-palette-index-past-end.txt)

Palette of 3 colors addressed with index 3, which does not exist.

Reads ExpandColorMap()'s black tail (vp8l_dec.c:1412): the map is padded out to
the packing capacity of 4, so the pixel comes back transparent black instead of
out of bounds.

## Transforms

Presence, repetition and tile sizes.

### [`transform-all-four.webp`](files/transform-all-four.webp) -- ok -- from [`transform-all-four.txt`](cases/transform-all-four.txt)

All four transforms present in one stream.

NUM_TRANSFORMS in a row: color-indexing, subtract-green, cross-color and
predictor, each with its own sub-image.

### [`transform-cross-color-bits-max.webp`](files/transform-cross-color-bits-max.webp) -- ok -- from [`transform-cross-color-bits-max.txt`](cases/transform-cross-color-bits-max.txt)

Cross-color transform with bits = 9, the largest tile size.

MIN_TRANSFORM_BITS + 7, so one tile covers any image up to 512x512. Partner of
transform-all-four, which uses the minimum.

### [`transform-cross-color-multipliers.webp`](files/transform-cross-color-multipliers.webp) -- ok -- from [`transform-cross-color-multipliers.txt`](cases/transform-cross-color-multipliers.txt)

A cross-color transform with real multipliers rather than the identity.

The three signed multipliers live in the blue, green and red bytes of a tile
(lossless.c:284), and here they turn a stored 0xff404060 into 0xff604080. Every
other file leaves them at zero, so this is the only one where
ColorTransformDelta() and its two chained additions do anything.

### [`transform-predictor-bits-max.webp`](files/transform-predictor-bits-max.webp) -- ok -- from [`transform-predictor-bits-max.txt`](cases/transform-predictor-bits-max.txt)

Predictor transform with bits = 9, the maximum tile size.

MIN_TRANSFORM_BITS + 7. The predictor sub-image is a single pixel for any image
up to 512x512.

### [`transform-repeated.webp`](files/transform-repeated.webp) -- reject -- from [`transform-repeated.txt`](cases/transform-repeated.txt)

The subtract-green transform declared twice.

Each transform type may appear once. Caught by the transforms_seen bitmask in
ReadTransform().

## Back-references

Copy lengths and distances.

### [`lz77-distance-1-run.webp`](files/lz77-distance-1-run.webp) -- ok -- from [`lz77-distance-1-run.txt`](cases/lz77-distance-1-run.txt)

A single literal followed by a length-8 copy at distance 1.

The degenerate overlapping copy: the copy loop reads bytes it has just written.

### [`lz77-distance-direct-121.webp`](files/lz77-distance-direct-121.webp) -- ok -- from [`lz77-distance-direct-121.txt`](cases/lz77-distance-direct-121.txt)

Plane code 121, the first value past the table.

Distances above CODE_TO_PLANE_CODES bypass the 2-D mapping entirely: the
distance is plane_code - 120, so 121 means 1.

### [`lz77-distance-past-start.webp`](files/lz77-distance-past-start.webp) -- reject -- from [`lz77-distance-past-start.txt`](cases/lz77-distance-past-start.txt)

A back-reference pointing further back than the pixels decoded so far.

One literal, then a copy at a distance of one whole row. Must be rejected
rather than reading before the buffer.

### [`lz77-length-past-end.webp`](files/lz77-length-past-end.webp) -- reject -- from [`lz77-length-past-end.txt`](cases/lz77-length-past-end.txt)

A copy whose length runs past the last pixel of the image.

Four literals then an 8-pixel copy in an 8-pixel image. Must be rejected rather
than writing past the buffer.

### [`lz77-max-length-symbol.webp`](files/lz77-max-length-symbol.webp) -- ok -- from [`lz77-max-length-symbol.txt`](cases/lz77-max-length-symbol.txt)

A back-reference using length symbol 23, the largest the format defines.

NUM_LENGTH_CODES - 1: 10 extra bits, copy lengths up to 4096. Here it copies
1200 pixels.

### [`lz77-plane-code-1.webp`](files/lz77-plane-code-1.webp) -- ok -- from [`lz77-plane-code-1.txt`](cases/lz77-plane-code-1.txt)

Back-reference with plane code 1, which means "the pixel directly above".

kCodeToPlane[0] is 0x18: yoffset 1, xoffset 0, so the distance is a whole row
rather than a small number.

### [`lz77-plane-code-120.webp`](files/lz77-plane-code-120.webp) -- ok -- from [`lz77-plane-code-120.txt`](cases/lz77-plane-code-120.txt)

Plane code 120, the last entry of the 2-D offset table.

kCodeToPlane[119] is 0x70: yoffset 7, xoffset 8, so on a 16-wide image the
distance is 120. Upper bound of the mapped range.

### [`lz77-plane-code-clamped-to-1.webp`](files/lz77-plane-code-clamped-to-1.webp) -- ok -- from [`lz77-plane-code-clamped-to-1.txt`](cases/lz77-plane-code-clamped-to-1.txt)

Plane code 4 on a 1-pixel-wide image, where the 2-D offset computes to 0.

kCodeToPlane[3] is 0x19: yoffset 1, xoffset -1, so dist = xsize - 1 = 0 and the
"dist < 1 ? 1" clamp at vp8l_dec.c:172 fires. Only reachable at xsize 1.

## Predictor modes

The per-tile predictor index. It is read as a 4-bit field, so all 16 values are
reachable, but the format only defines 14 of them.

### [`predictor-all-16-modes.webp`](files/predictor-all-16-modes.webp) -- ok -- from [`predictor-all-16-modes.txt`](cases/predictor-all-16-modes.txt)

One tile per predictor index, 0 to 15, across a 64x4 image.

The mode is read as ((pixel >> 8) & 0xf) at lossless.c:247, so all 16 indices
are reachable even though the format defines only 0..13.

### [`predictor-mode-11-select.webp`](files/predictor-mode-11-select.webp) -- ok -- from [`predictor-mode-11-select.txt`](cases/predictor-mode-11-select.txt)

Predictor 11 (Select) over the whole image.

The only predictor with a data-dependent branch, Select() at lossless.c:102.

### [`predictor-mode-13-clamp-half.webp`](files/predictor-mode-13-clamp-half.webp) -- ok -- from [`predictor-mode-13-clamp-half.txt`](cases/predictor-mode-13-clamp-half.txt)

Predictor 13 (ClampAddSubtractHalf) over the whole image.

Exercises AddSubtractComponentHalf() and its Clip255(), the arithmetic most
likely to differ between the C and SIMD paths.

### [`predictor-mode-14-undefined.webp`](files/predictor-mode-14-undefined.webp) -- ok -- from [`predictor-mode-14-undefined.txt`](cases/predictor-mode-14-undefined.txt)

Every tile selects predictor 14, which the format does not define.

Must decode, not crash: VP8LPredictorsAdd[14] is a padding sentinel pointing at
PredictorAdd0_C (lossless.c:654), so the tile comes out as mode 0. Shrink the
table to 14 entries and this is an out-of-bounds indirect call instead.

### [`predictor-mode-15-undefined.webp`](files/predictor-mode-15-undefined.webp) -- ok -- from [`predictor-mode-15-undefined.txt`](cases/predictor-mode-15-undefined.txt)

Every tile selects predictor 15, the other undefined index.

Partner of predictor-mode-14-undefined and the largest value the 4-bit mask can
produce. Verified to decode identically to mode 14, i.e. both really do land on
PredictorAdd0_C.

### [`predictor-single-row.webp`](files/predictor-single-row.webp) -- ok -- from [`predictor-single-row.txt`](cases/predictor-single-row.txt)

A predictor transform on a one-row image.

Only the y_start == 0 shortcut runs (lossless.c:223): the first pixel takes
mode 0 and the rest mode 1, so the tile modes are never read.

### [`predictor-tile-bits-min.webp`](files/predictor-tile-bits-min.webp) -- ok -- from [`predictor-tile-bits-min.txt`](cases/predictor-tile-bits-min.txt)

Predictor tiles of 4x4 pixels, the smallest the format allows.

MIN_TRANSFORM_BITS, so the sub-image is as large as it can get and the mode
changes every four pixels.

## Frame header

The 14-bit dimension fields and the version escape.

### [`header-alpha-is-used.webp`](files/header-alpha-is-used.webp) -- ok -- from [`header-alpha-is-used.txt`](cases/header-alpha-is-used.txt)

The alpha_is_used hint set on an image whose every pixel is opaque.

ReadImageInfo() reads the bit at vp8l_dec.c:119 and VP8LDecodeHeader() drops it
on the floor; only VP8LGetInfo() passes it out. A decoder that started
believing it over the pixels would fail this hash rather than pass unnoticed.

### [`header-magic-wrong.webp`](files/header-magic-wrong.webp) -- reject -- from [`header-magic-wrong.txt`](cases/header-magic-wrong.txt)

Signature byte 0x2e instead of the 0x2f the format defines.

Refused by VP8LCheckSignature() at vp8l_dec.c:110, which runs before a bit of
the stream is read. ReadImageInfo() tests the same byte again at vp8l_dec.c:116
and can never see a wrong one: like the version field beside it, that second
test is dead for anything that reaches the decoder through the public API, and
coverage.txt records both as the only probes here nothing hits.

### [`header-max-area-bomb.webp`](files/header-max-area-bomb.webp) -- ok -- from [`header-max-area-bomb.txt`](cases/header-max-area-bomb.txt)

34 bytes declaring 16384x16384, every pixel one color.

A bare VP8L stream has no area limit -- MAX_IMAGE_AREA is only checked in
ParseVP8X (webp_dec.c:138) -- and single-symbol codes cost zero bits per pixel,
so this decodes for real: 1.83GB peak RSS and 3.4s. The backstop is
WEBP_MAX_ALLOCABLE_MEMORY (utils.c:185), nothing earlier.

### [`header-max-area-truncated.webp`](files/header-max-area-truncated.webp) -- reject -- from [`header-max-area-truncated.txt`](cases/header-max-area-truncated.txt)

The same 16384x16384 header, cut off before the Huffman codes.

Must fail on the missing data rather than allocating the gigabyte first.
Partner of header-max-area-bomb: together they say where the allocation sits
relative to the parse.

### [`header-version-max.webp`](files/header-version-max.webp) -- reject -- from [`header-version-max.txt`](cases/header-version-max.txt)

Header with the 3-bit version field at 7, the largest it holds.

Upper bound of the field header-version-nonzero takes at 1. Both are refused by
the same (data[4] >> 5) test at vp8l_dec.c:111, which never looks at which non-
zero value it is.

### [`header-version-nonzero.webp`](files/header-version-nonzero.webp) -- reject -- from [`header-version-nonzero.txt`](cases/header-version-nonzero.txt)

Header with the 3-bit version field set to 1.

Rejected by VP8LCheckSignature() at vp8l_dec.c:111, which tests (data[4] >> 5)
before ReadImageInfo() ever runs. The version field is the format's only
forward-compatibility escape.

### [`header-width-16384.webp`](files/header-width-16384.webp) -- ok -- from [`header-width-16384.txt`](cases/header-width-16384.txt)

Width 16384, one past WEBP_MAX_DIMENSION.

The header stores width - 1 in 14 bits, so 16384 is expressible and the decoder
accepts it. WEBP_MAX_DIMENSION (16383) is enforced only in the encoder, at
webp_enc.c:347, so cwebp can never produce this.

## The RIFF container

The layer above the image: the RIFF header, the extended-format VP8X chunk and
its canvas size, the optional chunks a decoder must step over by their declared
length alone, and the padding rule that makes an odd-sized one even on disk.
Everything here is read by webp_dec.c before the frame is looked at.

### [`container-chunk-header-truncated.webp`](files/container-chunk-header-truncated.webp) -- reject -- from [`container-chunk-header-truncated.txt`](cases/container-chunk-header-truncated.txt)

The file cut four bytes into the last chunk header.

The "buf_size < CHUNK_HEADER_SIZE" test at the top of the walk: a tag with no
size behind it. Four bytes is enough to read the fourcc and not the length.

### [`container-duplicate-image-chunk.webp`](files/container-duplicate-image-chunk.webp) -- ok -- from [`container-duplicate-image-chunk.txt`](cases/container-duplicate-image-chunk.txt)

Two VP8 chunks, one after the other.

The walk stops at the first VP8 chunk, so the second is never looked at and
whatever it holds is dead weight. Nothing rejects the duplicate.

### [`container-metadata-chunks.webp`](files/container-metadata-chunks.webp) -- ok -- from [`container-metadata-chunks.txt`](cases/container-metadata-chunks.txt)

ICCP, EXIF and XMP chunks around the frame.

ParseOptionalChunks() walks and skips anything that is not VP8 or VP8L. The
three metadata chunks the format defines are the ones a real file is most
likely to carry, and the decoder must step over them by their declared size
alone.

### [`container-no-image-chunk.webp`](files/container-no-image-chunk.webp) -- reject -- from [`container-no-image-chunk.txt`](cases/container-no-image-chunk.txt)

A VP8X chunk and some metadata, and no image chunk at all.

ParseOptionalChunks() walks to the end of the data without meeting VP8 or VP8L
and runs out. A container that promises a picture and does not carry one.

### [`container-odd-chunk-no-pad.webp`](files/container-odd-chunk-no-pad.webp) -- reject -- from [`container-odd-chunk-no-pad.txt`](cases/container-odd-chunk-no-pad.txt)

An odd-sized chunk whose header rounds its length up, taking the pad byte away.

The walk steps by (8 + size + 1) & ~1, so an even declared size means no pad
byte, and everything after it is read one byte off. The partner of container-
odd-chunk-payload: same chunk, one byte shorter.

### [`container-odd-chunk-payload.webp`](files/container-odd-chunk-payload.webp) -- ok -- from [`container-odd-chunk-payload.txt`](cases/container-odd-chunk-payload.txt)

An optional chunk with an odd-sized payload, and the pad byte that must follow
it.

disk_chunk_size rounds a payload up to an even length, which only matters when
something follows the chunk it rounded. Here a JUNK payload of three bytes sits
in front of the image, so getting the rounding wrong moves the frame rather
than the end of the file.

### [`container-riff-size-past-end.webp`](files/container-riff-size-past-end.webp) -- reject, incremental ok -- from [`container-riff-size-past-end.txt`](cases/container-riff-size-past-end.txt)

A RIFF header claiming far more bytes than the file holds.

The "size > *data_size - CHUNK_HEADER_SIZE" test, which only fires when the
whole file is in hand -- the same lie is tolerated while a stream is still
arriving, so this is the one file in the corpus the incremental decoder accepts
and the one-shot one does not.

### [`container-riff-size-short.webp`](files/container-riff-size-short.webp) -- reject -- from [`container-riff-size-short.txt`](cases/container-riff-size-short.txt)

A RIFF header claiming 11 bytes, one less than the smallest legal value.

The "size < TAG_SIZE + CHUNK_HEADER_SIZE" test in ParseRIFF(): a RIFF size must
leave room for "WEBP" and one chunk header. 11 is the last value that does not.

### [`container-riff-size-truncates-chunks.webp`](files/container-riff-size-truncates-chunks.webp) -- reject -- from [`container-riff-size-truncates-chunks.txt`](cases/container-riff-size-truncates-chunks.txt)

A RIFF size that stops in the middle of the chunks behind it.

The "total_size > riff_size" test in ParseOptionalChunks(): the walk adds up
what it has skipped and refuses to walk past what the RIFF header said was
there.

### [`container-trailing-bytes.webp`](files/container-trailing-bytes.webp) -- ok -- from [`container-trailing-bytes.txt`](cases/container-trailing-bytes.txt)

Bytes after the last chunk that the RIFF size does not account for.

The decoder stops at the image chunk and never looks past it, so junk at the
end of the file is ignored rather than refused.

### [`container-unknown-chunk.webp`](files/container-unknown-chunk.webp) -- ok -- from [`container-unknown-chunk.txt`](cases/container-unknown-chunk.txt)

A chunk with a fourcc the format does not define, ahead of the frame.

The same skip path as the metadata chunks, but with a tag no version of libwebp
knows. An unknown chunk must be stepped over, not refused: that is what makes
the format extensible.

### [`container-vp8x-animation.webp`](files/container-vp8x-animation.webp) -- reject -- from [`container-vp8x-animation.txt`](cases/container-vp8x-animation.txt)

The VP8X animation flag set on a file with no animation chunks.

The flag alone is enough: WebPParseHeaders() turns any file claiming animation
into UNSUPPORTED_FEATURE, since a still decoder cannot compose frames. Nothing
looks for the ANIM chunk the flag implies.

### [`container-vp8x-area-overflow.webp`](files/container-vp8x-area-overflow.webp) -- reject -- from [`container-vp8x-area-overflow.txt`](cases/container-vp8x-area-overflow.txt)

A VP8X canvas of 16777216 by 16777216, the largest the two 24-bit fields can
describe.

The "width * height >= MAX_IMAGE_AREA" test in ParseVP8X(), computed in 64 bits
precisely so this cannot wrap. Both fields are at their maximum, so the product
is 2^48.

### [`container-vp8x-canvas-mismatch.webp`](files/container-vp8x-canvas-mismatch.webp) -- reject -- from [`container-vp8x-canvas-mismatch.txt`](cases/container-vp8x-canvas-mismatch.txt)

A VP8X canvas of 64x64 in front of a 32x32 frame.

The "Validates image size coherency" check at the end of
ParseHeadersInternal(): a VP8X chunk must agree with the frame behind it, in
both dimensions. The container is the only place in the format where the
picture size is stated twice.

### [`container-vp8x-reserved-bits.webp`](files/container-vp8x-reserved-bits.webp) -- ok -- from [`container-vp8x-reserved-bits.txt`](cases/container-vp8x-reserved-bits.txt)

The reserved bits of the VP8X flags field all set.

Everything outside ALL_VALID_FLAGS (0x3e). The decoder reads the field as a
whole and tests only the bits it knows, so the reserved ones ride through; the
muxer is stricter than the decoder here.

### [`container-vp8x-still-flags.webp`](files/container-vp8x-still-flags.webp) -- ok -- from [`container-vp8x-still-flags.txt`](cases/container-vp8x-still-flags.txt)

The four still-image VP8X flags set, with none of the chunks they promise.

Alpha, ICC, EXIF and XMP together. Nothing checks that a flag is backed by its
chunk, so this decodes as the plain lossy frame it is. The animation flag is
the one exception, which container-vp8x-animation covers.

### [`container-vp8x-wrong-size.webp`](files/container-vp8x-wrong-size.webp) -- reject -- from [`container-vp8x-wrong-size.txt`](cases/container-vp8x-wrong-size.txt)

A VP8X chunk whose header claims 9 bytes rather than 10.

The "chunk_size != VP8X_CHUNK_SIZE" test in ParseVP8X(), which is an equality:
a VP8X chunk is 10 bytes and no other length is tolerated, in either direction.

### [`container-vp8x.webp`](files/container-vp8x.webp) -- ok -- from [`container-vp8x.txt`](cases/container-vp8x.txt)

The extended format: a VP8X chunk ahead of the frame.

ParseVP8X(). The canvas size is written as width and height less one, and here
everything agrees: it is the shape each of the other container-vp8x-* cases
breaks in one way.

### [`container-zero-size-chunk.webp`](files/container-zero-size-chunk.webp) -- ok -- from [`container-zero-size-chunk.txt`](cases/container-zero-size-chunk.txt)

An optional chunk declaring a zero-length payload.

disk_chunk_size is then just the 8-byte header, which is the smallest step the
walk can take. A zero-length chunk is legal and must be stepped over like any
other.

## Animation

A sequence of frames rather than one image: an ANIM chunk carrying the loop
count, then an ANMF per frame carrying its own position, duration, disposal and
blending, and its own image chunks. None of it is reachable through dwebp,
which refuses any file claiming animation before looking at a frame, so these
are checked with anim_dump instead -- the demuxer of demux.c, the composition
of anim_decode.c, and one decode per frame. The verdict quoted for each is that
one; every file here is still a reject to a still decoder.

### [`anim-alph-after-image.webp`](files/anim-alph-after-image.webp) -- reject, anim_dump reject -- from [`anim-alph-after-image.txt`](cases/anim-alph-after-image.txt)

A frame whose alpha chunk comes after its image.

Both chunks are stored -- StoreFrame() has no opinion on their order -- and it
is IsValidExtendedFormat() that compares the two offsets afterwards and
requires the alpha to precede the image. The still-image form of the same
mistake is alph-after-image, which libwebp's own decoder accepts, so the two
layers do not agree about it.

### [`anim-alpha-flag-missing.webp`](files/anim-alpha-flag-missing.webp) -- reject, anim_dump ok, webpinfo reject -- from [`anim-alpha-flag-missing.txt`](cases/anim-alpha-flag-missing.txt)

Frames with transparency in a file whose VP8X does not admit it.

The reverse of container-vp8x-still-flags, which sets flags with nothing behind
them. Here the data is there and the flag is not. Nothing in the demuxer
compares the two, so the frames blend exactly as they would with the flag set;
webpinfo refuses the file -- "Unexpected alpha data detected" -- which makes
this the pair of cases where the two readers here disagree in opposite
directions.

### [`anim-alpha-lossless-frame.webp`](files/anim-alpha-lossless-frame.webp) -- reject, anim_dump ok -- from [`anim-alpha-lossless-frame.txt`](cases/anim-alpha-lossless-frame.txt)

An animation frame whose alpha plane is a compressed image stream.

The other ALPH compression, inside an animation: the payload is a headerless
VP8L stream read by VP8LDecodeAlphaHeader(), and this one is a palette of three
alpha values, so it also takes the 8-bit decoding path -- a lossless decode
nested inside a frame of an animation, two chunk layers below the file.

### [`anim-alpha-raw-frame.webp`](files/anim-alpha-raw-frame.webp) -- reject, anim_dump ok -- from [`anim-alpha-raw-frame.txt`](cases/anim-alpha-raw-frame.txt)

An animation frame carrying an uncompressed ALPH chunk beside its lossy image.

An ANMF holds a chunk list, not a chunk: ALPH then VP8 is the same pair a still
file uses, and StoreFrame() fills both slots of img_components. The alpha plane
is stored one byte per pixel, so this is the animation route into ALPHInit()
and the unfiltered plane.

### [`anim-anim-chunk-padded.webp`](files/anim-anim-chunk-padded.webp) -- reject, anim_dump ok, webpinfo reject -- from [`anim-anim-chunk-padded.txt`](cases/anim-anim-chunk-padded.txt)

An ANIM chunk with four bytes of padding after its two fields.

The ANIM chunk is six bytes of content and the demuxer treats that as a
minimum, not a size: it reads the background colour and loop count and then
skips whatever else the chunk declares, the same forward-compatibility the VP8X
chunk gets. webpinfo takes the other view and requires the length to be exactly
six -- "Corrupted ANIM chunk" -- so this is a file the two disagree about.

### [`anim-anim-chunk-short.webp`](files/anim-anim-chunk-short.webp) -- reject, anim_dump reject -- from [`anim-anim-chunk-short.txt`](cases/anim-anim-chunk-short.txt)

An ANIM chunk of two bytes.

The other side of anim-anim-chunk-padded. The demuxer treats six bytes as the
minimum an ANIM chunk may be and refuses anything shorter outright, because the
background colour and loop count it is about to read are not there. Longer is
tolerated; shorter is not.

### [`anim-anmf-header-truncated.webp`](files/anim-anmf-header-truncated.webp) -- reject, anim_dump reject -- from [`anim-anmf-header-truncated.txt`](cases/anim-anmf-header-truncated.txt)

An ANMF chunk declaring fewer bytes than its own header needs.

NewFrame()'s "actual_size < min_size" test. The sixteen bytes of ANMF header
are the minimum a frame chunk can be, and the check is made against the
declared length before a single field is read, so nothing here depends on what
the chunk actually contains.

### [`anim-anmf-odd-size.webp`](files/anim-anmf-odd-size.webp) -- reject, anim_dump reject -- from [`anim-anmf-odd-size.txt`](cases/anim-anmf-odd-size.txt)

An ANMF chunk whose declared length is odd.

The padding rule, one layer in. A frame chunk always comes out even -- its
header is sixteen bytes and every sub-chunk inside it is padded already -- so
an odd length is a contradiction. The demuxer does not test for it: it rounds
the length up like any other chunk, and what refuses the file is the pad byte
left over at the end, one short of a chunk header. webpinfo does test for it,
by name -- "ANMF chunk size should always be even".

### [`anim-anmf-size-past-end.webp`](files/anim-anmf-size-past-end.webp) -- reject, anim_dump reject -- from [`anim-anmf-size-past-end.txt`](cases/anim-anmf-size-past-end.txt)

An ANMF chunk claiming to be far longer than the file.

SizeIsInvalid(), the test the walk makes before trusting any declared length: a
chunk may not claim to extend past the end of the RIFF chunk it sits in. This
one claims four kilobytes inside a file of about a hundred bytes, so it is
refused before ParseAnimationFrame() is called at all.

### [`anim-anmf-size-short.webp`](files/anim-anmf-size-short.webp) -- reject, anim_dump reject -- from [`anim-anmf-size-short.txt`](cases/anim-anmf-size-short.txt)

An ANMF chunk declaring less than the image inside it takes up.

The other of the two size tests in ParseAnimationFrame(): after StoreFrame()
has read the frame's chunks, what it consumed is compared against what the ANMF
said its payload was. The image is really there and reads back fine; it is the
ANMF header's own arithmetic that does not add up, and the frame is refused for
overrunning itself.

### [`anim-background-color.webp`](files/anim-background-color.webp) -- reject, anim_dump ok -- from [`anim-background-color.txt`](cases/anim-background-color.txt)

A background colour that is neither white nor transparent.

ANIM's first four bytes. The demuxer reads them into dmux->bgcolor and hands
them to the caller, and that is all: the animation decoder zero-fills its
canvas for a key frame rather than filling it with this, so the field cannot
change a pixel of the output. Written so that a decoder that started obeying it
would fail the hash.

### [`anim-blend-none.webp`](files/anim-blend-none.webp) -- reject, anim_dump ok -- from [`anim-blend-none.txt`](cases/anim-blend-none.txt)

A half-transparent frame that says not to blend it.

blending_method 1 is NO_BLEND, and it changes two things at once: the frame is
written over the canvas as it is, alpha and all, and IsKeyFrame() takes its
second branch, because a full-canvas frame that does not blend needs nothing
underneath it. Same bytes as anim-blend-over, one bit apart, and the frames
come out different.

### [`anim-blend-over.webp`](files/anim-blend-over.webp) -- reject, anim_dump ok -- from [`anim-blend-over.txt`](cases/anim-blend-over.txt)

The same half-transparent frame, blended.

The pair to anim-blend-none. blending_method 0 with alpha and a frame that does
not fill the canvas keeps IsKeyFrame() false, so BlendPixelRowNonPremult() runs
over the previous canvas -- the non-premultiplied formula of
BlendPixelNonPremult(), which is what MODE_RGBA output asks for.

### [`anim-blend-ranges.webp`](files/anim-blend-ranges.webp) -- reject, anim_dump ok -- from [`anim-blend-ranges.txt`](cases/anim-blend-ranges.txt)

Three frames arranged so the blended region is split in two.

FindBlendRangeAtRow(), which is only reached when the previous frame was
disposed to background and the current one blends. Frame 3 sticks out on both
sides of frame 2's rectangle, so rows that cross it yield two ranges to blend
and the strip between them is left alone; rows above and below frame 2 take the
disjoint branch and blend whole. Neither frame 2 nor frame 3 is a key frame,
which is what it takes to get here at all.

### [`anim-canvas-larger-than-frames.webp`](files/anim-canvas-larger-than-frames.webp) -- reject, anim_dump ok -- from [`anim-canvas-larger-than-frames.txt`](cases/anim-canvas-larger-than-frames.txt)

A canvas bigger than any frame in it.

Nothing requires a frame to reach the edges of the canvas. The 16x16 frames sit
in the corner of a 64x48 one and the rest stays as ZeroFillCanvas() left it --
transparent black -- for the whole sequence, so the output is mostly the canvas
rather than the frames.

### [`anim-dispose-background.webp`](files/anim-dispose-background.webp) -- reject, anim_dump ok -- from [`anim-dispose-background.txt`](cases/anim-dispose-background.txt)

A partial frame that asks to be cleared away after it is shown.

disposal_method 1 is DISPOSE_BACKGROUND: once frame 2 has been handed out,
ZeroFillFrameRect() clears just its rectangle out of the canvas the next frame
starts from, so frame 3 sees frame 1 everywhere except that hole. Frame 3 is
opaque and full-canvas, so it overwrites all of it anyway -- what is being
pinned is that the clearing does not reach past the rectangle.

### [`anim-dispose-blend-matrix.webp`](files/anim-dispose-blend-matrix.webp) -- reject, anim_dump ok -- from [`anim-dispose-blend-matrix.txt`](cases/anim-dispose-blend-matrix.txt)

The four combinations of the disposal and blending bits, in one sequence.

Both bits of the ANMF flag byte, in every combination, over partial frames with
alpha so that each combination actually changes the canvas. Frame 1 is the
opaque background; frames 2 to 5 are (dispose, blend) = (0,0), (0,1), (1,0) and
(1,1) in turn, which walks IsKeyFrame(), the copy-or-zero-fill choice, both
arms of the blending branch and ZeroFillFrameRect() one after another in a
single file.

### [`anim-duplicate-anim.webp`](files/anim-duplicate-anim.webp) -- reject, anim_dump ok -- from [`anim-duplicate-anim.txt`](cases/anim-duplicate-anim.txt)

A second ANIM chunk between the frames.

The one place the demuxer silently drops a chunk rather than refusing the file:
with anim_chunks already at one, the second ANIM takes the store_chunk = 0 path
and is skipped by its declared length. Its loop count and background colour are
not read, so the first ANIM still decides both.

### [`anim-duration-extremes.webp`](files/anim-duration-extremes.webp) -- reject, anim_dump ok -- from [`anim-duration-extremes.txt`](cases/anim-duration-extremes.txt)

One frame of zero milliseconds and one of the longest the field can say.

Both ends of the 24-bit duration field in one file. Neither is refused: a zero-
duration frame is legal and common as an interstitial, and 0xffffff is just
under MAX_DURATION, so nothing in the field's range is out of range. The
durations accumulate into the timestamp anim_dump reports, and change no pixel.

### [`anim-eight-frames.webp`](files/anim-eight-frames.webp) -- reject, anim_dump ok -- from [`anim-eight-frames.txt`](cases/anim-eight-frames.txt)

Eight frames, each a different flat colour.

A longer frame list than anything else here: the demuxer chains eight Frame
records and AddFrame() walks to the tail each time. Every one is a full-canvas
opaque key frame, so the canvas is zero-filled and overwritten eight times with
no blending anywhere.

### [`anim-empty-frame-alone.webp`](files/anim-empty-frame-alone.webp) -- reject, anim_dump reject -- from [`anim-empty-frame-alone.txt`](cases/anim-empty-frame-alone.txt)

An animation whose only frame carries no image.

The same silent drop as anim-empty-frame, with nothing left behind it. The
frame list stays empty, and IsValidExtendedFormat() refuses a finished file
with no frames at all -- so what the demuxer tolerates in the middle of a
sequence it will not tolerate as the whole of one.

### [`anim-empty-frame.webp`](files/anim-empty-frame.webp) -- reject, anim_dump ok, webpinfo reject -- from [`anim-empty-frame.txt`](cases/anim-empty-frame.txt)

An ANMF chunk holding its header and nothing else, followed by a real frame.

A frame with no image is dropped rather than refused, and quietly. StoreFrame()
is handed a payload size of zero, reads the chunk header belonging to the next
frame, rewinds and returns having stored nothing; the frame record it was
filling still has the frame_num zero WebPSafeCalloc() left, so the
"frame->frame_num > 0" test in ParseAnimationFrame() fails and it is freed. One
frame comes out of a file that declares two. webpinfo refuses the same file --
"No VP8/VP8L chunk detected in an ANMF chunk".

### [`anim-frame-1x1.webp`](files/anim-frame-1x1.webp) -- reject, anim_dump ok -- from [`anim-frame-1x1.txt`](cases/anim-frame-1x1.txt)

A single-pixel frame in the middle of a canvas.

The smallest frame the ANMF header can describe, since the size fields hold
width - 1. Its rectangle is one pixel wide, which is what ZeroFillFrameRect()
and the blend ranges are handed after it, and the copy from the previous canvas
supplies every other pixel.

### [`anim-frame-alpha-only.webp`](files/anim-frame-alpha-only.webp) -- reject, anim_dump reject -- from [`anim-frame-alpha-only.txt`](cases/anim-frame-alpha-only.txt)

A frame carrying an alpha chunk and no image.

The ALPH arm of StoreFrame() sets frame_num where the image arm would have, so
unlike anim-empty-frame this frame is kept: it has an alpha component, no image
component, and the width and height the ANMF header claimed. That is enough to
pass every test in IsValidExtendedFormat(), which only refuses a frame with
neither. The decoder is then handed an ALPH chunk to decode as an image.

### [`anim-frame-area-overflow.webp`](files/anim-frame-area-overflow.webp) -- reject, anim_dump reject -- from [`anim-frame-area-overflow.txt`](cases/anim-frame-area-overflow.txt)

An ANMF header claiming a frame of sixteen million by sixteen million, inside a
canvas of sixteen by sixteen.

The MAX_IMAGE_AREA test inside ParseAnimationFrame(), which runs on the header
fields alone -- before the frame's own chunks are looked at, and before
anything is allocated for it. Both size fields are 0xffffff, so the product
overflows 32 bits and is computed as a uint64. The canvas is pinned small on
purpose: left to follow the frame it would be refused by the VP8X area check
first, and this test would never run.

### [`anim-frame-image-past-canvas.webp`](files/anim-frame-image-past-canvas.webp) -- reject, anim_dump reject -- from [`anim-frame-image-past-canvas.txt`](cases/anim-frame-image-past-canvas.txt)

A frame whose header fits the canvas but whose image does not.

The other half of anim-frame-size-mismatch. The ANMF says 16x16 at (16,16),
which fits; the image inside is 32x32, and since that is the size
CheckFrameBounds() ends up testing, the frame runs 16 pixels past the canvas
and the file is refused. The header numbers being reasonable is no protection.

### [`anim-frame-offsets.webp`](files/anim-frame-offsets.webp) -- reject, anim_dump ok -- from [`anim-frame-offsets.txt`](cases/anim-frame-offsets.txt)

A second frame smaller than the canvas, placed at an offset.

The ANMF offset fields, which hold half the pixel offset: frame_x 4 puts the
frame at x = 8. The frame covers part of the canvas, so the rest of it is
whatever frame 1 left there -- CopyCanvas() rather than ZeroFillCanvas(), the
only way a frame is not composed from scratch.

### [`anim-frame-past-canvas.webp`](files/anim-frame-past-canvas.webp) -- reject, anim_dump reject -- from [`anim-frame-past-canvas.txt`](cases/anim-frame-past-canvas.txt)

A frame whose rectangle runs off the edge of the canvas.

CheckFrameBounds() with exact = 0, the animation form: a frame need not fill
the canvas, but x_offset + width must still fit inside it. 16 wide at x = 48 in
a 32-wide canvas does not, and the check runs after the whole file has parsed,
so this is refused by the validator rather than by the walk.

### [`anim-frame-reserved-bits.webp`](files/anim-frame-reserved-bits.webp) -- reject, anim_dump ok -- from [`anim-frame-reserved-bits.txt`](cases/anim-frame-reserved-bits.txt)

The six reserved bits of the ANMF flag byte all set.

The byte after the duration holds six reserved bits above the blending and
disposal ones. Both readers take it with a mask -- bits & 1 and bits & 2 -- so
the rest is dropped rather than refused, and the frame composes as if the byte
were zero.

### [`anim-frame-size-mismatch.webp`](files/anim-frame-size-mismatch.webp) -- reject, anim_dump ok, webpinfo reject -- from [`anim-frame-size-mismatch.txt`](cases/anim-frame-size-mismatch.txt)

An ANMF header claiming a size its own image disagrees with.

Which of the two sizes wins. The ANMF header says 16x16 and the VP8L inside
says 32x32; ParseAnimationFrame() takes the header's numbers and then
SetFrameInfo() overwrites them with the image's, so the header fields are
dropped and the frame composes at 32x32. webpinfo refuses the same file --
"Frame size in VP8/VP8L sub-chunk differs from ANMF header" -- so the two
readers in this repository disagree about it.

### [`anim-frames-without-flag.webp`](files/anim-frames-without-flag.webp) -- reject, anim_dump reject -- from [`anim-frames-without-flag.txt`](cases/anim-frames-without-flag.txt)

A full animation whose VP8X does not claim to be one.

The animation flag is what makes the frames real. Without it
ParseAnimationFrame() parses each ANMF, allocates a frame, and then throws it
away rather than adding it, so a file with two frames in it ends with none and
is refused for being empty. The ANIM chunk itself is read either way -- nothing
checks the flag before that.

### [`anim-image-chunk-beside-frames.webp`](files/anim-image-chunk-beside-frames.webp) -- reject, anim_dump reject -- from [`anim-image-chunk-beside-frames.txt`](cases/anim-image-chunk-beside-frames.txt)

A top-level image chunk in a file that also has frames.

"check that this isn't an animation (all frames should be in an ANMF)". An
animated file has no image of its own; the VP8 here sits beside the frames
rather than inside one, and the walk refuses it on sight of either the flag or
a preceding ANIM chunk.

### [`anim-loop-count-max.webp`](files/anim-loop-count-max.webp) -- reject, anim_dump ok -- from [`anim-loop-count-max.txt`](cases/anim-loop-count-max.txt)

Loop count 65535, the largest the field holds.

ANIM's loop count is read by ReadLE16s() into an int, so every value of the
field is non-negative and the "loop_count < 0" test in IsValidExtendedFormat()
can never fire. 0 means forever; this is the other end, and the decoder
composes the frames the same either way.

### [`anim-lossy-frames.webp`](files/anim-lossy-frames.webp) -- reject, anim_dump ok -- from [`anim-lossy-frames.txt`](cases/anim-lossy-frames.txt)

Two lossy frames rather than lossless ones.

An ANMF may carry either image format. This is the VP8 side: StoreFrame() takes
the MKFOURCC('V','P','8',' ') arm and WebPGetFeatures() reads the frame tag
rather than the lossless header.

### [`anim-metadata-chunks.webp`](files/anim-metadata-chunks.webp) -- reject, anim_dump ok -- from [`anim-metadata-chunks.txt`](cases/anim-metadata-chunks.txt)

An animation carrying an ICC profile and both metadata chunks.

Every optional chunk the format defines, around an animation rather than around
a still frame: ICCP has to come before the image data and EXIF and XMP after
it, and each is stored for the caller only because its VP8X flag is set.
container-metadata-chunks is the same three around a lossy frame.

### [`anim-metadata-without-flags.webp`](files/anim-metadata-without-flags.webp) -- reject, anim_dump ok, webpinfo reject -- from [`anim-metadata-without-flags.txt`](cases/anim-metadata-without-flags.txt)

ICCP and EXIF chunks in an animation that declares neither.

A metadata chunk whose flag is missing is not stored: the store_chunk test in
ParseVP8XChunks() drops it, and the walk then skips its bytes by their declared
length like any chunk it does not want. So the animation decodes and the two
chunks are simply invisible to the caller -- no error, no chunk. webpinfo
refuses the same file, twice over: "Unexpected ICCP chunk detected".

### [`anim-mixed-formats.webp`](files/anim-mixed-formats.webp) -- reject, anim_dump ok -- from [`anim-mixed-formats.txt`](cases/anim-mixed-formats.txt)

A lossy frame and a lossless one in the same animation.

Nothing ties the frames of an animation to one format. The decoder re-reads the
chunk tag per frame, so WebPDecode() switches between VP8 and VP8L mid-
sequence, and both write into the same RGBA canvas.

### [`anim-nested-anmf.webp`](files/anim-nested-anmf.webp) -- reject, anim_dump reject -- from [`anim-nested-anmf.txt`](cases/anim-nested-anmf.txt)

An ANMF chunk inside another ANMF chunk.

There is no nesting in the format: a frame holds image chunks and nothing else.
StoreFrame() does not recognise the inner ANMF, so it rewinds and leaves the
outer frame with no image, which drops it; the walk then meets the inner one as
though it were a top-level frame, and that one declares a payload of nothing
while an image follows it.

### [`anim-no-anim-chunk.webp`](files/anim-no-anim-chunk.webp) -- reject, anim_dump reject -- from [`anim-no-anim-chunk.txt`](cases/anim-no-anim-chunk.txt)

Frames with no ANIM chunk in front of them.

"'ANIM' precedes frames": ParseVP8XChunks() refuses an ANMF while anim_chunks
is still zero. The ANIM chunk carries the loop count and background colour, and
the format puts it before the frames rather than anywhere among them.

### [`anim-no-vp8x.webp`](files/anim-no-vp8x.webp) -- reject, anim_dump reject -- from [`anim-no-vp8x.txt`](cases/anim-no-vp8x.txt)

ANIM and ANMF chunks with no VP8X in front of them.

How far such a file gets, which is not far. An application reaches the demuxer
through a format check first -- anim_util.c calls WebPGetInfo() -- and that is
the still-image header walk, which looks for VP8 or VP8L where this file has
ANIM and refuses it as a bad signature. The demuxer would have refused it too,
one layer down: its master chunk table has three entries, VP8, VP8L and VP8X,
and a file beginning with none of them matches no parser at all.

### [`anim-riff-size-past-end.webp`](files/anim-riff-size-past-end.webp) -- reject, anim_dump reject -- from [`anim-riff-size-past-end.txt`](cases/anim-riff-size-past-end.txt)

A RIFF size claiming more bytes than the file has.

Where the demuxer parts company with the still decoder. It computes partial =
(buffer < riff_end) up front and, unless the caller asked for partial parsing,
refuses the file there and then -- before the master chunk table is consulted,
so none of the animation code runs at all. container-riff-size-past-end is the
same lie told to dwebp.

### [`anim-riff-size-truncates-frames.webp`](files/anim-riff-size-truncates-frames.webp) -- reject, anim_dump reject -- from [`anim-riff-size-truncates-frames.txt`](cases/anim-riff-size-truncates-frames.txt)

A RIFF size that stops in the middle of the last frame.

The RIFF length is the ceiling every other length is measured against: the
demuxer clamps its buffer to it, so the second frame's own header -- which is
intact on disk, and correct -- now declares more bytes than are left, and
SizeIsInvalid() refuses it. The bytes are there; the file just says they are
not.

### [`anim-second-vp8x.webp`](files/anim-second-vp8x.webp) -- reject, anim_dump reject -- from [`anim-second-vp8x.txt`](cases/anim-second-vp8x.txt)

A second VP8X chunk after the frames have started.

The one fourcc the chunk walk refuses on sight. VP8X is the header that started
the walk and it carries the canvas size and the feature flags, so a second one
would be re-declaring what has already been acted on; the walk returns an error
rather than picking either.

### [`anim-single-frame.webp`](files/anim-single-frame.webp) -- reject, anim_dump ok -- from [`anim-single-frame.txt`](cases/anim-single-frame.txt)

An animation of one frame.

The degenerate sequence. Nothing requires more than one ANMF, so the
composition loop runs once and stops; the file is still an animation to every
decoder, and still refused by the still one.

### [`anim-two-frames.webp`](files/anim-two-frames.webp) -- reject, anim_dump ok -- from [`anim-two-frames.txt`](cases/anim-two-frames.txt)

Two full-canvas lossless frames, the plain animation everything else here is a
variation on.

The whole ordinary path: VP8X with the animation flag, ANIM, then one ANMF per
frame carrying a VP8L image the size of the canvas. Both frames are key frames
-- frame 1 always is, and frame 2 is because it is opaque and fills the canvas
-- so IsKeyFrame() takes its first two branches and neither frame is blended
against anything.

### [`anim-two-images-in-frame.webp`](files/anim-two-images-in-frame.webp) -- reject, anim_dump reject -- from [`anim-two-images-in-frame.txt`](cases/anim-two-images-in-frame.txt)

One ANMF carrying two image chunks.

How the walk leaves a frame it cannot finish. StoreFrame() takes the first
VP8L, sees a second while image_chunks is already one, and jumps to the label
that rewinds and stops -- leaving the reader positioned inside the ANMF
payload, at the second image's header. The outer walk then reads that as a top-
level chunk and refuses it, because in an animation every image belongs to a
frame.

### [`anim-unknown-chunk-between-frames.webp`](files/anim-unknown-chunk-between-frames.webp) -- reject, anim_dump ok -- from [`anim-unknown-chunk-between-frames.txt`](cases/anim-unknown-chunk-between-frames.txt)

An unrecognised chunk sitting between two frames.

The chunk walk between frames. A fourcc the demuxer knows nothing about is
stored by offset and stepped over by its declared length, exactly as it would
be in a still file, and the frame that follows is unaffected -- the frames of
an animation need not be adjacent.

### [`anim-vp8l-with-alph.webp`](files/anim-vp8l-with-alph.webp) -- reject, anim_dump reject -- from [`anim-vp8l-with-alph.txt`](cases/anim-vp8l-with-alph.txt)

A frame carrying both an alpha chunk and a lossless image.

"VP8L has its own alpha": the lossless format carries a fourth channel already,
so an ALPH beside it is contradictory rather than redundant, and StoreFrame()
refuses the pair outright as soon as it reads the VP8L tag with alpha_chunks
non-zero.

## The alpha chunk

ALPH carries the alpha plane beside a lossy frame: a header byte of four two-
bit fields, then the plane itself, either stored as it is or compressed with
the lossless coder in its 8-bit mode. That mode is a separate path through
vp8l_dec.c from the one every VP8L image here takes, and an alpha chunk is the
only thing that reaches it -- whether it sits beside a still frame or inside an
animation frame. Each of the four filters has a routine of its own in
dsp/filters.c, and the same stored bytes come out as four different planes, so
the pixel hash is what tells those apart. A compressed plane is a lossless
image stream with its header left off, so the alph-plane cases write one from
text and can break each of the four conditions the 8-bit mode asks for.

### [`alph-after-image.webp`](files/alph-after-image.webp) -- ok -- from [`alph-after-image.txt`](cases/alph-after-image.txt)

An ALPH chunk placed after the image chunk instead of before it.

ParseOptionalChunks() stops at the first VP8 chunk, so an ALPH behind it is
never seen and the picture decodes fully opaque. Ordering is not diagnosed, it
is silently obeyed -- which the all-255 alpha in the hash is what records.

### [`alph-compression-invalid.webp`](files/alph-compression-invalid.webp) -- reject -- from [`alph-compression-invalid.txt`](cases/alph-compression-invalid.txt)

A compression method of 2, past the lossless one.

The header byte packs four fields into two bits each, and only the method and
the pre-processing have values the format does not define. This is the first of
them.

### [`alph-empty-payload.webp`](files/alph-empty-payload.webp) -- reject -- from [`alph-empty-payload.txt`](cases/alph-empty-payload.txt)

An ALPH chunk holding its header byte and nothing else.

The "data_size <= ALPHA_HEADER_LEN" test at the top of ALPHInit(), which is
what stops the header byte itself from being read out of an empty chunk.

### [`alph-lossless-byte-flipped.webp`](files/alph-lossless-byte-flipped.webp) -- reject -- from [`alph-lossless-byte-flipped.txt`](cases/alph-lossless-byte-flipped.txt)

The same plane with its last byte replaced.

One byte, not a truncation: the stream is the right length and stops making
sense at the end. Together with alph-lossless-truncated this pins both ways the
compressed plane can fail.

### [`alph-lossless-palette.webp`](files/alph-lossless-palette.webp) -- ok -- from [`alph-lossless-palette.txt`](cases/alph-lossless-palette.txt)

A losslessly compressed alpha plane carrying a palette transform, from a cwebp
encode of a two-valued plane.

The one shape that reaches DecodeAlphaData(): exactly one transform, that
transform colour-indexing, no colour cache, and the red, blue and alpha codes
each a single symbol. That is the lossless decoder's 8-bit mode, a different
loop from the one every VP8L file here takes. This one is real encoder output,
which is what it is for: the alph-plane-* cases reach the same loop from a
written case, and can be made to miss it one condition at a time.

### [`alph-lossless-predictor.webp`](files/alph-lossless-predictor.webp) -- ok -- from [`alph-lossless-predictor.txt`](cases/alph-lossless-predictor.txt)

A losslessly compressed alpha plane carrying a predictor transform, from a
cwebp encode of a gradient.

Compression method 1 hands the payload to the lossless decoder. A predictor
transform leaves the red, blue and alpha codes non-trivial, so
Is8bOptimizable() says no and the plane is decoded through DecodeImageData()
with ExtractAlphaRows() pulling the green channel out afterwards. The 21
payload bytes are cwebp -q 60 output, kept as the control that a written plane
is shaped like a real one.

### [`alph-lossless-truncated.webp`](files/alph-lossless-truncated.webp) -- reject -- from [`alph-lossless-truncated.txt`](cases/alph-lossless-truncated.txt)

The predictor-transform plane cut to ten bytes.

The lossless decoder runs out part way through the alpha image. The failure
comes back through ALPHInit() as a bitstream error rather than as a short read.

### [`alph-no-vp8x.webp`](files/alph-no-vp8x.webp) -- reject -- from [`alph-no-vp8x.txt`](cases/alph-no-vp8x.txt)

An ALPH chunk in a RIFF file with no VP8X ahead of it.

The extended format is what an ALPH chunk lives in: libwebp only accepts a
leading ALPH when there is no RIFF header at all, the bare stream case. With
RIFF and no VP8X it is refused.

### [`alph-plane-cache.webp`](files/alph-plane-cache.webp) -- ok -- from [`alph-plane-cache.txt`](cases/alph-plane-cache.txt)

A palette-coded alpha plane that also declares a colour cache.

The first test in Is8bOptimizable(), and the only one that is about the stream
rather than about its Huffman codes: any colour cache at all disqualifies the
8-bit path, because that path keeps one byte per pixel and the cache is keyed
on whole ARGB values. Same palette as alph-plane-palette, one field apart, and
it decodes through the 32-bit loop instead.

### [`alph-plane-filtered.webp`](files/alph-plane-filtered.webp) -- ok -- from [`alph-plane-filtered.txt`](cases/alph-plane-filtered.txt)

A compressed alpha plane with the gradient filter on top.

The two alpha stages together. Filtering and compression are separate fields of
the ALPH header byte and the raw cases sweep the filters on their own; here the
plane is decompressed first and the gradient unfilter then runs over what came
out, so the values in the stream are differences rather than levels.

### [`alph-plane-literals.webp`](files/alph-plane-literals.webp) -- ok -- from [`alph-plane-literals.txt`](cases/alph-plane-literals.txt)

A compressed alpha plane written as plain green literals.

The simplest compressed plane there is: no transform, so Is8bOptimizable() is
never consulted and the plane goes through DecodeImageData() with
ExtractAlphaRows() pulling the green channel out of a full ARGB buffer. The
alpha value is the green symbol, which is why this needs no palette to say
anything.

### [`alph-plane-lz77-past-start.webp`](files/alph-plane-lz77-past-start.webp) -- reject -- from [`alph-plane-lz77-past-start.txt`](cases/alph-plane-lz77-past-start.txt)

An alpha plane copying from before its own beginning.

The bounds test inside the 8-bit loop, which is written separately from the
32-bit one and refuses the copy rather than clamping it: the distance reaches
back further than the pixels decoded so far. lz77-distance-past-start is the
same mistake on the 32-bit path.

### [`alph-plane-lz77.webp`](files/alph-plane-lz77.webp) -- ok -- from [`alph-plane-lz77.txt`](cases/alph-plane-lz77.txt)

An alpha plane whose second half is a back-reference to its first.

CopyBlock8b(), the byte-wide copy the 8-bit alpha path uses in place of the
32-bit one every other back-reference here goes through. The distance is a
plane code, resolved against the packed width of the palette-coded image rather
than against the width of the alpha plane.

### [`alph-plane-meta-huffman.webp`](files/alph-plane-meta-huffman.webp) -- ok -- from [`alph-plane-meta-huffman.txt`](cases/alph-plane-meta-huffman.txt)

A palette-coded alpha plane with an entropy image inside it.

The block loop of DecodeAlphaData(), which nothing else here reaches: with an
entropy image the Huffman mask stops being ~0, so the decoder stops reading
straight through to the last row and instead walks one tile-sized block at a
time, asking GetHtreeGroupForPos() which group each one belongs to. Four tiles
down the plane, alternating between two groups, and both groups keep their red,
blue and alpha codes single so the 8-bit path still applies.

### [`alph-plane-nontrivial-red.webp`](files/alph-plane-nontrivial-red.webp) -- ok -- from [`alph-plane-nontrivial-red.txt`](cases/alph-plane-nontrivial-red.txt)

A palette-coded alpha plane whose red code carries two symbols.

The Huffman half of Is8bOptimizable(), which asks that the red, blue and alpha
codes each hold a single symbol so their bits can be skipped entirely. Here the
packed pixels vary in a channel the palette lookup never reads, which costs the
plane its 8-bit path for nothing: the alpha that comes out is the same as alph-
plane-palette's, decoded the long way.

### [`alph-plane-oversubscribed.webp`](files/alph-plane-oversubscribed.webp) -- reject -- from [`alph-plane-oversubscribed.txt`](cases/alph-plane-oversubscribed.txt)

An alpha plane whose green code is over-subscribed.

A malformed Huffman code inside the alpha stream rather than inside a VP8L
image. It fails in the same BuildHuffmanTable() as codelen-oversubscribed, but
the error has to travel back out through VP8LDecodeAlphaHeader() and ALPHInit()
to become the decoder's verdict.

### [`alph-plane-palette.webp`](files/alph-plane-palette.webp) -- ok -- from [`alph-plane-palette.txt`](cases/alph-plane-palette.txt)

A compressed alpha plane of three values, through a palette.

The shape that reaches DecodeAlphaData(): one transform, that transform colour-
indexing, no colour cache, and the red, blue and alpha codes each a single
symbol. Written rather than pasted from an encoder, so the neighbouring cases
can each break one of those four conditions and watch the decoder fall back.
Three values in a two-bit index, so a packed byte holds four pixels and the
fourth index is the unused one.

### [`alph-plane-preprocessed.webp`](files/alph-plane-preprocessed.webp) -- ok -- from [`alph-plane-preprocessed.txt`](cases/alph-plane-preprocessed.txt)

A compressed alpha plane flagged as level-reduced.

ALPHA_PREPROCESSED_LEVELS on a compressed plane. The bit makes the decoder take
the "decode everything in one pass" branch rather than the row-by-row one,
which here means the whole lossless stream is read before any of it is handed
on. WebPDequantizeLevels() is then called with the dithering strength, which is
zero unless the caller asks for it, so the plane comes out unchanged: this pins
the control path, not the pixels. alph-raw-preprocessing is the same bit on a
stored plane.

### [`alph-plane-two-transforms.webp`](files/alph-plane-two-transforms.webp) -- ok -- from [`alph-plane-two-transforms.txt`](cases/alph-plane-two-transforms.txt)

An alpha plane carrying a palette and a subtract-green transform.

The transform count, the other half of the 8-bit test: the decoder asks for
exactly one transform and for it to be the palette, so a second one -- however
harmless subtracting green from a plane that is all green may be -- sends it
back to the 32-bit loop.

### [`alph-preprocessing-invalid.webp`](files/alph-preprocessing-invalid.webp) -- reject -- from [`alph-preprocessing-invalid.txt`](cases/alph-preprocessing-invalid.txt)

A pre-processing value of 2, past level reduction.

The second undefined value in the header byte, refused by the same condition in
ALPHInit(). All four filter values are legal, so the filter field has no
partner to this.

### [`alph-raw-filter-gradient.webp`](files/alph-raw-filter-gradient.webp) -- ok -- from [`alph-raw-filter-gradient.txt`](cases/alph-raw-filter-gradient.txt)

An uncompressed alpha plane under the gradient filter.

WebPUnfilters[3], from left plus above minus above-left.

### [`alph-raw-filter-horizontal.webp`](files/alph-raw-filter-horizontal.webp) -- ok -- from [`alph-raw-filter-horizontal.txt`](cases/alph-raw-filter-horizontal.txt)

An uncompressed alpha plane under the horizontal filter.

WebPUnfilters[1], each byte a difference from the one to its left.

### [`alph-raw-filter-none.webp`](files/alph-raw-filter-none.webp) -- ok -- from [`alph-raw-filter-none.txt`](cases/alph-raw-filter-none.txt)

An uncompressed alpha plane under the none filter.

WebPUnfilters[0], stored as it is.

### [`alph-raw-filter-vertical.webp`](files/alph-raw-filter-vertical.webp) -- ok -- from [`alph-raw-filter-vertical.txt`](cases/alph-raw-filter-vertical.txt)

An uncompressed alpha plane under the vertical filter.

WebPUnfilters[2], from the one above.

### [`alph-raw-oversized.webp`](files/alph-raw-oversized.webp) -- ok -- from [`alph-raw-oversized.txt`](cases/alph-raw-oversized.txt)

An uncompressed plane 44 bytes longer than the picture needs.

ALPHInit() tests "alpha_data_size >= alpha_decoded_size", so a plane may be
longer than width by height and the tail is simply never read. The boundary
partner of alph-raw-short.

### [`alph-raw-preprocessing.webp`](files/alph-raw-preprocessing.webp) -- ok -- from [`alph-raw-preprocessing.txt`](cases/alph-raw-preprocessing.txt)

An uncompressed plane declaring the level-reduction pre-processing.

The one bit of ALPHA_PREPROCESSED_LEVELS: it makes the decoder take the "decode
everything in one pass" branch and keeps alpha dithering alive instead of
switching it off. The plane itself comes out the same as with no pre-
processing, so this pins the control path rather than the pixels.

### [`alph-raw-short.webp`](files/alph-raw-short.webp) -- reject -- from [`alph-raw-short.txt`](cases/alph-raw-short.txt)

An uncompressed plane one byte short of the picture.

The other side of that same test, one byte away: 255 bytes where a 16x16
picture needs 256.

### [`alph-reserved-set.webp`](files/alph-reserved-set.webp) -- reject -- from [`alph-reserved-set.txt`](cases/alph-reserved-set.txt)

The two reserved bits of the ALPH header byte set.

The "rsrv != 0" arm of the same test. Unlike VP8X, whose reserved bits ride
through untouched, ALPH refuses a header with anything in its top two bits.

### [`alph-without-vp8x-flag.webp`](files/alph-without-vp8x-flag.webp) -- ok -- from [`alph-without-vp8x-flag.txt`](cases/alph-without-vp8x-flag.txt)

An ALPH chunk with the VP8X alpha flag left clear.

The flag and the chunk are independent: the decoder walks to the ALPH chunk and
uses it whatever VP8X claimed. A file the muxer would call inconsistent and the
decoder does not.

## Lossy: frame tag and picture header

The ten uncompressed bytes every lossy frame starts with: the profile, the
visibility and key-frame bits, the length of partition 0, the start code and
the two 14-bit dimensions.

### [`lossy-frame-bad-start-code.webp`](files/lossy-frame-bad-start-code.webp) -- reject -- from [`lossy-frame-bad-start-code.txt`](cases/lossy-frame-bad-start-code.txt)

The three-byte start code changed from 9d 01 2a to 9d 01 29.

VP8CheckSignature(). One bit away from valid, so it also checks that the
signature is compared, not merely skipped over.

### [`lossy-frame-colorspace-clamp.webp`](files/lossy-frame-colorspace-clamp.webp) -- ok -- from [`lossy-frame-colorspace-clamp.txt`](cases/lossy-frame-colorspace-clamp.txt)

The colour-space and clamping-type bits both set.

The two bits at the very top of partition 0. libwebp stores both and acts on
neither, so a decoder that started honouring either would fail this file's hash
rather than its verdict. Nothing else in the corpus sets them.

### [`lossy-frame-interframe.webp`](files/lossy-frame-interframe.webp) -- reject -- from [`lossy-frame-interframe.txt`](cases/lossy-frame-interframe.txt)

The key-frame bit cleared, so the frame claims to be an inter frame.

libwebp decodes single key frames only, and VP8GetInfo() turns this away on the
key-frame bit alone: the picture header behind it is never looked at, and
VP8GetHeaders() never runs.

### [`lossy-frame-not-shown.webp`](files/lossy-frame-not-shown.webp) -- reject -- from [`lossy-frame-not-shown.txt`](cases/lossy-frame-not-shown.txt)

A key frame with the show_frame bit cleared.

VP8GetInfo() bails on "first frame is invisible"; the VP8 layer would have said
UNSUPPORTED_FEATURE. Nothing else in the corpus reaches either.

### [`lossy-frame-part0-empty.webp`](files/lossy-frame-part0-empty.webp) -- reject -- from [`lossy-frame-part0-empty.txt`](cases/lossy-frame-part0-empty.txt)

The frame tag claims a zero-byte partition 0.

The boolean reader is handed no data at all: every header field reads past the
end, and ParseSegmentHeader() returns on br->eof.

### [`lossy-frame-part0-past-end.webp`](files/lossy-frame-part0-past-end.webp) -- reject -- from [`lossy-frame-part0-past-end.txt`](cases/lossy-frame-part0-past-end.txt)

The frame tag claims a partition 0 far larger than the file.

The 19-bit partition length. VP8GetInfo() catches "partition_length >=
chunk_size" first; VP8GetHeaders() has its own "bad partition length" behind
it.

### [`lossy-frame-scale-1.webp`](files/lossy-frame-scale-1.webp) -- ok -- from [`lossy-frame-scale-1.txt`](cases/lossy-frame-scale-1.txt)

A horizontal upscaling hint of 1 and a vertical one of 3.

Two of the four values of the two 2-bit scale fields; lossy-frame-scaled has 3
and 2 and lossy-frame-scale-2 the rest, so between them every value of both is
written. libwebp reads them into pic_hdr and acts on neither.

### [`lossy-frame-scale-2.webp`](files/lossy-frame-scale-2.webp) -- ok -- from [`lossy-frame-scale-2.txt`](cases/lossy-frame-scale-2.txt)

A horizontal upscaling hint of 2 and a vertical one of 1.

The values the other scale cases leave out, so every one of the four is seen in
each field. A decoder that started honouring the hint would resize the output
and fail the hash rather than the verdict.

### [`lossy-frame-scaled.webp`](files/lossy-frame-scaled.webp) -- ok -- from [`lossy-frame-scaled.txt`](cases/lossy-frame-scaled.txt)

Horizontal and vertical upscaling hints of 3 (2x) in the top bits of the
dimension fields.

pic_hdr->xscale and yscale. libwebp parses and ignores them, so the output
stays 32x32; a decoder that honoured them would fail the hash.

### [`lossy-frame-version-1.webp`](files/lossy-frame-version-1.webp) -- ok -- from [`lossy-frame-version-1.txt`](cases/lossy-frame-version-1.txt)

A frame declaring profile 1 instead of 0.

The 3-bit version field of the frame tag. libwebp accepts 0 to 3 and
reconstructs them all the same way, so this pins the acceptance, not the
pixels.

### [`lossy-frame-version-2.webp`](files/lossy-frame-version-2.webp) -- ok -- from [`lossy-frame-version-2.txt`](cases/lossy-frame-version-2.txt)

Profile 2, one of the four values the decoder accepts.

One of the four values "profile > 3" lets through. libwebp reads the field only
to refuse the others, so what this pins is that it does not act on the value.

### [`lossy-frame-version-3.webp`](files/lossy-frame-version-3.webp) -- ok -- from [`lossy-frame-version-3.txt`](cases/lossy-frame-version-3.txt)

Profile 3, the largest the decoder accepts.

Boundary partner of lossy-frame-version-4: "profile > 3" is the whole check in
VP8GetHeaders().

### [`lossy-frame-version-4.webp`](files/lossy-frame-version-4.webp) -- reject -- from [`lossy-frame-version-4.txt`](cases/lossy-frame-version-4.txt)

Profile 4, one past the last valid value.

VP8GetInfo() rejects it before VP8GetHeaders() ever runs, so this comes back as
BITSTREAM_ERROR rather than the "Incorrect keyframe parameters" message the VP8
layer would give.

### [`lossy-frame-version-7.webp`](files/lossy-frame-version-7.webp) -- reject -- from [`lossy-frame-version-7.txt`](cases/lossy-frame-version-7.txt)

Profile 7, the largest the 3-bit field can hold.

The far end of the field, past the "profile > 3" test that lossy-frame-
version-4 sits on. Same rejection, opposite end of the range, which is what
pins the test as a comparison rather than an equality.

### [`lossy-frame-width-16383.webp`](files/lossy-frame-width-16383.webp) -- ok -- from [`lossy-frame-width-16383.txt`](cases/lossy-frame-width-16383.txt)

The widest frame the 14-bit field can describe, one macroblock tall.

1024 macroblocks in a single row, so the whole frame goes through one partition
and the per-column contexts are exercised 1024 wide.

### [`lossy-frame-zero-width.webp`](files/lossy-frame-zero-width.webp) -- reject -- from [`lossy-frame-zero-width.txt`](cases/lossy-frame-zero-width.txt)

A frame whose width field is zero, with a height of 32.

The "w == 0 || h == 0" check in VP8GetInfo(), which the comment above it
describes as not supporting both being zero while the code refuses either.
Nothing else in the corpus reaches it.

## Lossy: segmentation

Up to four segments, each with its own quantizer and loop-filter strength, and
a per-macroblock map saying which is which. cwebp uses the feature but only
ever writes absolute values, and always writes the map and the data together.

### [`lossy-segment-delta-quantizers.webp`](files/lossy-segment-delta-quantizers.webp) -- ok -- from [`lossy-segment-delta-quantizers.txt`](cases/lossy-segment-delta-quantizers.txt)

Segment quantizers read as deltas on the frame quantizer instead of absolute
values.

segment_feature_mode = 0, the "q += base_q0" branch of VP8ParseQuant(). cwebp
always writes absolute values (syntax_enc.c:196), so nothing else reaches this.

### [`lossy-segment-filter-strengths.webp`](files/lossy-segment-filter-strengths.webp) -- ok -- from [`lossy-segment-filter-strengths.txt`](cases/lossy-segment-filter-strengths.txt)

Per-segment loop-filter strengths, from 0 to 63, under a frame filter level of
40.

PrecomputeFilterStrengths() with use_segment: the per-segment base level
replaces the frame level outright when the deltas are absolute.

### [`lossy-segment-four-quantizers.webp`](files/lossy-segment-four-quantizers.webp) -- ok -- from [`lossy-segment-four-quantizers.txt`](cases/lossy-segment-four-quantizers.txt)

Four segments with four different absolute quantizers, one macroblock each.

The segment tree in ParseIntraMode() with all three probabilities used, and
four distinct VP8QuantMatrix rows in VP8ParseQuant().

### [`lossy-segment-map-only.webp`](files/lossy-segment-map-only.webp) -- ok -- from [`lossy-segment-map-only.txt`](cases/lossy-segment-map-only.txt)

A segment map with no segment data behind it.

update_map without update_data: the ids are read and used to index dqm[], but
every entry is the frame quantizer, so the map changes nothing. Pins that the
two flags are independent.

### [`lossy-segment-no-map.webp`](files/lossy-segment-no-map.webp) -- ok -- from [`lossy-segment-no-map.txt`](cases/lossy-segment-no-map.txt)

Segmentation on, quantizers given, but no segment map: every macroblock is
segment 0.

use_segment without update_map. No per-macroblock segment bits are read, and
only dqm[0] is ever selected, but the other three are still built.

### [`lossy-segment-prob-extremes.webp`](files/lossy-segment-prob-extremes.webp) -- ok -- from [`lossy-segment-prob-extremes.txt`](cases/lossy-segment-prob-extremes.txt)

Segment probabilities of 0 and 255, and loop-filter updates at both ends of
their range.

A tree probability of 0 or 255 makes one branch of the segment id free and the
other maximally expensive; -63 and 63 are the ends of the 6-bit signed loop-
filter update. All four macroblocks carry a different segment id, so every
branch of the tree is taken under those probabilities.

### [`lossy-segment-quant-extremes.webp`](files/lossy-segment-quant-extremes.webp) -- ok -- from [`lossy-segment-quant-extremes.txt`](cases/lossy-segment-quant-extremes.txt)

Segment quantizers at 127, -127, 0 and absent.

clip(q, 127) at both ends of VP8ParseQuant(), and the difference between a
field written as zero and one left out, which libwebp's own encoder cannot
express.

## Lossy: loop filter

The in-loop deblocking filter: simple or normal, its level and sharpness, and
the per-reference and per-mode deltas. PrecomputeFilterStrengths() shifts the
interior limit right by one for sharpness 1 to 4 and by two for 5 to 7, then
clamps it to 9 - sharpness, which is what the sharpness files sit either side
of.

### [`lossy-filter-lf-delta-extremes.webp`](files/lossy-filter-lf-delta-extremes.webp) -- ok -- from [`lossy-filter-lf-delta-extremes.txt`](cases/lossy-filter-lf-delta-extremes.txt)

Loop-filter mode deltas at -63 and 63.

The mode deltas are applied per macroblock coding type, and only the first of
the four is ever used by an intra frame. lossy-filter-lf-delta writes -20 and
31; this writes both ends of the 6-bit signed field, against a filter level
that leaves room to move.

### [`lossy-filter-lf-delta.webp`](files/lossy-filter-lf-delta.webp) -- ok -- from [`lossy-filter-lf-delta.txt`](cases/lossy-filter-lf-delta.txt)

Loop-filter deltas: 63 and -63 on the reference deltas, and a delta on the 4x4
mode.

The mode_lf_delta[0] path for 4x4-coded macroblocks and the ref_lf_delta that
only inter frames would use. cwebp writes four zero flags and one i4x4 delta
(syntax_enc.c:226), never these.

### [`lossy-filter-normal-max.webp`](files/lossy-filter-normal-max.webp) -- ok -- from [`lossy-filter-normal-max.txt`](cases/lossy-filter-normal-max.txt)

The normal loop filter at level 63, sharpness 0.

filter_type 2 with the widest possible filter, so both the 8-pixel and the
4-pixel variants run at their limits.

### [`lossy-filter-sharpness-4.webp`](files/lossy-filter-sharpness-4.webp) -- ok -- from [`lossy-filter-sharpness-4.txt`](cases/lossy-filter-sharpness-4.txt)

Sharpness 4, the last level that halves the interior limit.

This is the last level of the first shift.

### [`lossy-filter-sharpness-5.webp`](files/lossy-filter-sharpness-5.webp) -- ok -- from [`lossy-filter-sharpness-5.txt`](cases/lossy-filter-sharpness-5.txt)

Sharpness 5, the first level that quarters it.

This is the first level of the second shift.

### [`lossy-filter-simple-max.webp`](files/lossy-filter-simple-max.webp) -- ok -- from [`lossy-filter-simple-max.txt`](cases/lossy-filter-simple-max.txt)

The simple loop filter at level 63 and sharpness 7.

filter_type 1, and the sharpness clamp on the interior limit. cwebp picks its
own level and never gets near the top of the range.

## Lossy: quantizer

The frame quantizer index and the five deltas around it, one per plane and
coefficient kind, with clamps that are not all the same.

### [`lossy-quant-deltas-mirrored.webp`](files/lossy-quant-deltas-mirrored.webp) -- ok -- from [`lossy-quant-deltas-mirrored.txt`](cases/lossy-quant-deltas-mirrored.txt)

The five quantizer deltas at the ends lossy-quant-deltas does not use.

Each of the five 4-bit signed fields written at its other extreme, so between
the two files every one of them is seen at both -15 and +15.

### [`lossy-quant-deltas.webp`](files/lossy-quant-deltas.webp) -- ok -- from [`lossy-quant-deltas.txt`](cases/lossy-quant-deltas.txt)

All five quantizer deltas present, at the ends of their 4-bit range.

dqy1_dc, dqy2_dc, dqy2_ac, dquv_dc and dquv_ac at once. cwebp writes only the
two chroma deltas, and only small ones.

### [`lossy-quant-dequant-overflow.webp`](files/lossy-quant-dequant-overflow.webp) -- ok -- from [`lossy-quant-dequant-overflow.txt`](cases/lossy-quant-dequant-overflow.txt)

A coefficient of 2114 at the coarsest quantizer, so the dequantized value does
not fit the int16 it is stored in.

"out[kZigzag[n]] = VP8GetSigned(br, v) * dq[n > 0]" with v * dq of 600376
against an int16_t destination. Nothing an encoder can produce, and worth
watching under a sanitizer.

### [`lossy-quant-max.webp`](files/lossy-quant-max.webp) -- ok -- from [`lossy-quant-max.txt`](cases/lossy-quant-max.txt)

The frame quantizer at 127, the coarsest.

The last entry of both quantizer tables, and the top of the clip() range that
the delta cases push against.

### [`lossy-quant-min.webp`](files/lossy-quant-min.webp) -- ok -- from [`lossy-quant-min.txt`](cases/lossy-quant-min.txt)

The frame quantizer at 0, the finest the format allows.

kDcTable[0] and kAcTable[0] are both 4, and the Y2 AC quantizer hits the "if
(m->y2_mat[1] < 8) m->y2_mat[1] = 8" floor in VP8ParseQuant().

### [`lossy-quant-uv-dc-clamp.webp`](files/lossy-quant-uv-dc-clamp.webp) -- ok -- from [`lossy-quant-uv-dc-clamp.txt`](cases/lossy-quant-uv-dc-clamp.txt)

A chroma DC quantizer index pushed past 117, where it is clamped rather than at
127.

The odd "clip(q + dquv_dc, 117)" in VP8ParseQuant(), a limit the four other
planes do not have. base 110 plus 15 lands on 125, so the clamp is what makes
the difference.

## Lossy: coefficient probabilities

The 1056 probabilities that drive the coefficient coder, each one optionally
replaced in the frame header, plus the skip probability.

### [`lossy-proba-all-updated.webp`](files/lossy-proba-all-updated.webp) -- ok -- from [`lossy-proba-all-updated.txt`](cases/lossy-proba-all-updated.txt)

Every one of the 1056 coefficient probabilities updated.

Every one of the 1056 flags set, each followed by a raw byte, so partition 0
holds nothing but probability updates. cwebp writes a handful at most.

### [`lossy-proba-one-update.webp`](files/lossy-proba-one-update.webp) -- ok -- from [`lossy-proba-one-update.txt`](cases/lossy-proba-one-update.txt)

A single coefficient probability updated, the other 1055 left alone.

One 1 among the update flags of VP8ParseProba(), on the [i4-AC][band 0][ctx 0]
EOB probability, which the coefficients below then use.

### [`lossy-proba-refresh-and-skip-zero.webp`](files/lossy-proba-refresh-and-skip-zero.webp) -- ok -- from [`lossy-proba-refresh-and-skip-zero.txt`](cases/lossy-proba-refresh-and-skip-zero.txt)

The entropy-refresh bit set, and a skip probability of 0.

refresh_entropy_probs is read and dropped by libwebp, so no other file sets it.
A prob_skip_false of 0 says no macroblock is skipped while two are, which is
the most expensive way the flag can be coded and the bottom of its range.

### [`lossy-proba-skip-extremes.webp`](files/lossy-proba-skip-extremes.webp) -- ok -- from [`lossy-proba-skip-extremes.txt`](cases/lossy-proba-skip-extremes.txt)

A skip probability of 255 with nothing skipped, and the flag itself written
out.

use_skip_proba with a probability that says every macroblock should be skipped
while none is, which is the most expensive way to code it.

### [`lossy-proba-zero.webp`](files/lossy-proba-zero.webp) -- ok -- from [`lossy-proba-zero.txt`](cases/lossy-proba-zero.txt)

Coefficient probabilities of 0 and of 255, the ends of the range.

A probability of 0 makes the boolean split 0, so the "bit is 0" branch
renormalizes from an empty range. Legal, and never emitted.

## Lossy: prediction modes

The 16x16 and 4x4 luma modes and the chroma modes, and the neighbour-indexed
probability table the 4x4 modes are coded with.

### [`lossy-mode-i16-all-four.webp`](files/lossy-mode-i16-all-four.webp) -- ok -- from [`lossy-mode-i16-all-four.txt`](cases/lossy-mode-i16-all-four.txt)

The four 16x16 luma modes, one per macroblock.

DC_PRED, V_PRED, H_PRED and TM_PRED through the hardcoded tree at probabilities
156, 128 and 163, and all four 16x16 reconstructions.

### [`lossy-mode-i4-all-ten.webp`](files/lossy-mode-i4-all-ten.webp) -- ok -- from [`lossy-mode-i4-all-ten.txt`](cases/lossy-mode-i4-all-ten.txt)

All ten 4x4 luma modes inside one macroblock, twice over.

Every leaf of the B_PRED tree, and every 4x4 predictor including the ones that
need the four pixels above and to the right.

### [`lossy-mode-i4-context.webp`](files/lossy-mode-i4-context.webp) -- ok -- from [`lossy-mode-i4-context.txt`](cases/lossy-mode-i4-context.txt)

Four B_PRED macroblocks whose 4x4 modes walk the [above][left] probability
table.

kBModesProba is indexed by the two neighbouring modes, so this is the only way
to reach entries other than [B_DC][B_DC]. The top row of one macroblock is the
context of the one below it.

### [`lossy-mode-mixed.webp`](files/lossy-mode-mixed.webp) -- ok -- from [`lossy-mode-mixed.txt`](cases/lossy-mode-mixed.txt)

16x16 and 4x4 macroblocks alternating, in both directions.

A 16x16 macroblock writes its mode into all four of its neighbours' 4x4
contexts, so this is what checks that the two mode paths agree on the context
they leave behind.

### [`lossy-mode-uv-all-four.webp`](files/lossy-mode-uv-all-four.webp) -- ok -- from [`lossy-mode-uv-all-four.txt`](cases/lossy-mode-uv-all-four.txt)

The four chroma modes, one per macroblock.

The uvmode tree at probabilities 142, 114 and 183, and the 8x8 chroma
predictors.

## Lossy: coefficients

The token coder of section 13: magnitudes and their escape categories, end-of-
block, zero runs, and the four coefficient types. The band sweeps below share a
trick: the three token classes drive the context of the next position -- a zero
gives 0, a +-1 gives 1, anything larger gives 2 -- so a block of each, walked
to position 15, reads every band at that class, and placing the blocks so their
neighbour contexts are 0, 1 and 2 in turn is the only way to reach band 0,
which is never a token's successor.

### [`lossy-coeff-all-types.webp`](files/lossy-coeff-all-types.webp) -- ok -- from [`lossy-coeff-all-types.txt`](cases/lossy-coeff-all-types.txt)

All four coefficient types in one macroblock, and both luma types across two.

The Y2 block (type 1), luma after Y2 starting at position 1 (type 0), chroma
(type 2) and luma with its own DC (type 3). No single macroblock can reach all
four, since types 0 and 3 are exclusive.

### [`lossy-coeff-bands-chroma.webp`](files/lossy-coeff-bands-chroma.webp) -- ok -- from [`lossy-coeff-bands-chroma.txt`](cases/lossy-coeff-bands-chroma.txt)

The same sweep across the four blocks of a chroma plane.

Type 2 (chroma). The 2x2 layout means block 3 is the only one with both a left
and an above neighbour, so it is the only way to read the chroma band 0 at
context 2.

### [`lossy-coeff-bands-i16.webp`](files/lossy-coeff-bands-i16.webp) -- ok -- from [`lossy-coeff-bands-i16.txt`](cases/lossy-coeff-bands-i16.txt)

The same sweep for the two block kinds a 16x16 macroblock has: the Y2 block and
the luma blocks that follow it.

Types 1 (Y2) and 0 (luma after Y2). The luma blocks start at position 1, so
band 0 is unreachable for them and band 1 is what their neighbour context
selects. One Y2 block per macroblock means the four macroblocks are what give
it contexts 0, 1, 1 and 2.

### [`lossy-coeff-bands-i4.webp`](files/lossy-coeff-bands-i4.webp) -- ok -- from [`lossy-coeff-bands-i4.txt`](cases/lossy-coeff-bands-i4.txt)

Three 4x4 luma blocks that sweep every coefficient band, one per context.

Type 3 (luma with its own DC).

### [`lossy-coeff-cat3.webp`](files/lossy-coeff-cat3.webp) -- ok -- from [`lossy-coeff-cat3.txt`](cases/lossy-coeff-cat3.txt)

Category-3 coefficients: 11 to 18, three extra bits each.

The first escape category, kCat3 = {173, 148, 140}, at both ends of its range.

### [`lossy-coeff-cat4.webp`](files/lossy-coeff-cat4.webp) -- ok -- from [`lossy-coeff-cat4.txt`](cases/lossy-coeff-cat4.txt)

Category-4 coefficients: 19 to 34, four extra bits.

kCat4, and the p[9] branch that separates category 4 from category 3.

### [`lossy-coeff-cat5.webp`](files/lossy-coeff-cat5.webp) -- ok -- from [`lossy-coeff-cat5.txt`](cases/lossy-coeff-cat5.txt)

Category-5 coefficients: 35 to 66, five extra bits.

kCat5, reached through p[8] = 1 and p[10] = 0, which is a different pair of
probabilities from the lower categories.

### [`lossy-coeff-cat6-max.webp`](files/lossy-coeff-cat6-max.webp) -- ok -- from [`lossy-coeff-cat6-max.txt`](cases/lossy-coeff-cat6-max.txt)

The largest coefficient the format can encode: 2114.

3 + (8 << 3) + 2047, every one of kCat6's eleven extra bits set. One more would
wrap round inside the same eleven bits.

### [`lossy-coeff-cat6.webp`](files/lossy-coeff-cat6.webp) -- ok -- from [`lossy-coeff-cat6.txt`](cases/lossy-coeff-cat6.txt)

Category-6 coefficients: 67 upwards, eleven extra bits.

kCat6, the longest escape. 67 is the first value it can hold.

### [`lossy-coeff-context.webp`](files/lossy-coeff-context.webp) -- ok -- from [`lossy-coeff-context.txt`](cases/lossy-coeff-context.txt)

Neighbouring blocks with and without coefficients, so that every context value
from 0 to 2 is used.

The "ctx = left + top" that picks one of the three probability sets. Context 2
needs both neighbours non-empty, which only happens a few blocks into a
macroblock.

### [`lossy-coeff-empty-blocks.webp`](files/lossy-coeff-empty-blocks.webp) -- ok -- from [`lossy-coeff-empty-blocks.txt`](cases/lossy-coeff-empty-blocks.txt)

Every one of the 25 blocks empty, but the macroblock not skipped.

An end-of-block at the very first position of every block. Without a skip
probability in the frame there is no other way to say it, and ParseResiduals()
still runs in full.

### [`lossy-coeff-full-block.webp`](files/lossy-coeff-full-block.webp) -- ok -- from [`lossy-coeff-full-block.txt`](cases/lossy-coeff-full-block.txt)

A block with all sixteen coefficients non-zero, so the loop ends by running out
of positions rather than on an end-of-block.

The "n == 16" exit of GetCoeffs(), which is the only way out that does not read
an end-of-block bit, and every band from 0 to 7.

### [`lossy-coeff-medium-magnitudes.webp`](files/lossy-coeff-medium-magnitudes.webp) -- ok -- from [`lossy-coeff-medium-magnitudes.txt`](cases/lossy-coeff-medium-magnitudes.txt)

Coefficients of 5 through 10, coded with the fixed probabilities 159, 165 and
145.

The two branches under p[6]: 5 and 6 through probability 159, then 7 to 10
through 165 and 145. Those three constants appear nowhere else.

### [`lossy-coeff-small-magnitudes.webp`](files/lossy-coeff-small-magnitudes.webp) -- ok -- from [`lossy-coeff-small-magnitudes.txt`](cases/lossy-coeff-small-magnitudes.txt)

Coefficients of 1, 2, 3 and 4, the magnitudes with their own tree branches.

The v == 1 shortcut, then the p[4]/p[5] pair of GetLargeValue() that separates
2 from 3 and 4.

### [`lossy-coeff-wht-full.webp`](files/lossy-coeff-wht-full.webp) -- ok -- from [`lossy-coeff-wht-full.txt`](cases/lossy-coeff-wht-full.txt)

A Y2 block with more than one coefficient, next to one with only a DC.

The two halves of the Y2 branch in ParseResiduals(): "nz > 1" runs the full
VP8TransformWHT, while a lone DC takes the inlined "(dc[0] + 3) >> 3" shortcut.

### [`lossy-coeff-zero-runs.webp`](files/lossy-coeff-zero-runs.webp) -- ok -- from [`lossy-coeff-zero-runs.txt`](cases/lossy-coeff-zero-runs.txt)

Single coefficients at positions 15, 12, 8 and 1, each behind a run of zeros.

The inner "sequence of zero coeffs" loop of GetCoeffs(), which walks the band
table without reading an end-of-block bit, and reaches position 15 in band 7.

## Lossy: skipped macroblocks

The per-macroblock skip flag, which drops the residual entirely and clears the
neighbouring non-zero flags -- almost all of them.

### [`lossy-skip-all.webp`](files/lossy-skip-all.webp) -- ok -- from [`lossy-skip-all.txt`](cases/lossy-skip-all.txt)

Every macroblock skipped.

The skip branch of VP8DecodeMB(), which clears the neighbour flags without
reading a single coefficient. Every token partition holds nothing but its own
padding.

### [`lossy-skip-i4x4-nz-dc.webp`](files/lossy-skip-i4x4-nz-dc.webp) -- ok -- from [`lossy-skip-i4x4-nz-dc.txt`](cases/lossy-skip-i4x4-nz-dc.txt)

A skipped 4x4 macroblock between two 16x16 ones that both carry a Y2 block.

VP8DecodeMB() clears nz_dc only when the skipped macroblock is not 4x4-coded,
so the second 16x16 macroblock sees the Y2 context left behind by the first.
Reproducing that quirk is the whole point.

### [`lossy-skip-mixed.webp`](files/lossy-skip-mixed.webp) -- ok -- from [`lossy-skip-mixed.txt`](cases/lossy-skip-mixed.txt)

Skipped and coded macroblocks alternating, with a skip probability of 1.

The most expensive coding of the skip flag: the probability says almost nothing
is skipped while half of it is. Also checks that a skipped macroblock leaves
its neighbours' contexts cleared.

## Lossy: token partitions

A lossy frame may carry 1, 2, 4 or 8 token partitions, macroblock row r being
read from partition r & (n - 1). cwebp does not expose config.partitions and
libwebp forces it back to 1 whenever the token path is used (webp_enc.c:124),
so none of this is reachable through the tools.

### [`lossy-parts-2-wrap.webp`](files/lossy-parts-2-wrap.webp) -- ok -- from [`lossy-parts-2-wrap.txt`](cases/lossy-parts-2-wrap.txt)

Four rows over two partitions, so each partition holds two non-adjacent rows.

The "mb_y & num_parts_minus_one" wrap. Each bit reader is left in the middle of
a row and resumed two rows later.

### [`lossy-parts-8-rows.webp`](files/lossy-parts-8-rows.webp) -- ok -- from [`lossy-parts-8-rows.txt`](cases/lossy-parts-8-rows.txt)

Eight macroblock rows over eight token partitions, one row each, every row
different.

Row r is read from partition r & 7, so a decoder that got the mapping wrong
would still parse but decode the rows in the wrong order. The pixel hash is
what catches that.

### [`lossy-parts-last-empty.webp`](files/lossy-parts-last-empty.webp) -- reject -- from [`lossy-parts-last-empty.txt`](cases/lossy-parts-last-empty.txt)

Four partitions whose declared sizes leave nothing for the last one.

The last partition is not declared anywhere; it gets whatever is left, which
the clamp on the third size has already reduced to nothing. ParsePartitions()
notices that part_start is no longer inside the buffer and refuses the frame
rather than handing out an empty reader.

### [`lossy-parts-size-past-end.webp`](files/lossy-parts-size-past-end.webp) -- reject -- from [`lossy-parts-size-past-end.txt`](cases/lossy-parts-size-past-end.txt)

Four partitions, the first declaring 16 MB of data.

The "if (psize > size_left) psize = size_left" clamp. Partition 0 swallows the
rest of the frame, so part_start reaches the end of the buffer and
ParsePartitions() returns NOT_ENOUGH_DATA: the frame is refused there, before a
single macroblock is decoded.

### [`lossy-parts-table-too-small.webp`](files/lossy-parts-table-too-small.webp) -- reject -- from [`lossy-parts-table-too-small.txt`](cases/lossy-parts-table-too-small.txt)

Eight partitions declared in a frame with only ten bytes left for the twenty-
one-byte size table.

The "size < 3 * last_part" test in ParsePartitions(), the one failure there
that cannot be papered over: the sizes themselves are unreadable, so there is
nothing to clamp. Confirmed to reach parts_size_table_truncated and no further.

## Lossy: truncation

Frames that stop early, at each of the places the decoder can notice: inside
partition 0, inside the macroblock modes, and inside the token data.

### [`lossy-truncated-header.webp`](files/lossy-truncated-header.webp) -- reject -- from [`lossy-truncated-header.txt`](cases/lossy-truncated-header.txt)

Partition 0 cut to two bytes, which is enough for the segment header and not
for the filter header.

The "cannot parse filter header" exit of VP8GetHeaders(). ParseFilterHeader()
returns !br->eof, and this is the only file that makes it the one to fail: one
byte less and the segment header goes first, one more and the failure moves to
the macroblock modes.

### [`lossy-truncated-modes.webp`](files/lossy-truncated-modes.webp) -- reject -- from [`lossy-truncated-modes.txt`](cases/lossy-truncated-modes.txt)

Partition 0 long enough for the whole frame header and not for the macroblock
modes that follow it.

"Premature end-of-partition0 encountered", out of the !dec->br.eof that
VP8ParseIntraModeRow() checks once per macroblock row rather than once per
macroblock. The token partitions are untouched, so this is the mode data
failing on its own.

### [`lossy-truncated-short-modes.webp`](files/lossy-truncated-short-modes.webp) -- ok -- from [`lossy-truncated-short-modes.txt`](cases/lossy-truncated-short-modes.txt)

Mode data for 15 macroblocks in a frame whose dimensions call for 16.

The missing macroblock is decoded out of partition 0's padding without the
reader ever running out, so the frame is accepted and the last macroblock
decodes to whatever the padding happens to say. Nothing checks that the mode
data is as long as the frame claims to be. Reading the file back gives 16
macroblocks, so it is the one case here that cannot be disassembled and
reassembled unchanged.

### [`lossy-truncated-tokens.webp`](files/lossy-truncated-tokens.webp) -- reject -- from [`lossy-truncated-tokens.txt`](cases/lossy-truncated-tokens.txt)

Partition 0 intact, the token partition cut in half.

"Premature end-of-file encountered" out of VP8DecodeMB(), the other truncation
path. The modes all parse, so the decoder gets several rows in before it fails.

## Lossy: partition sizes, from real encodes

The four files behind these are genuine encoder output, made through the
encoder API rather than cwebp, and the broken ones rewrite the raw size table
that follows partition 0. They carry a whole frame of real coefficients, which
the assembled cases do not.

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

### [`lossy-combo-all-features.webp`](files/lossy-combo-all-features.webp) -- ok -- from [`lossy-combo-all-features.txt`](cases/lossy-combo-all-features.txt)

Every optional tool switched on in one frame at once.

Segmentation with a map, the loop filter with per-segment strengths and mode
deltas, four token partitions, the skip flag, a probability update, and both
macroblock types. Each of those has a file of its own that isolates it; this is
the one that makes them interact, and it is what covers the pairs of features
that meet nowhere else.

---

257 files, 57387 bytes total. Rebuild with `generate.py`: it assembles
everything in `cases/` through `webp_asm.py`, which hands each case to
`vp8l_asm.py` or `vp8_asm.py`, and those to `vp8l.py` and `vp8.py`.
