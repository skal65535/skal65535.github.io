# WebP stress bitstreams

Small WebP files that exercise corners of the format a normal encoder never
emits, one layer of it at a time.

**Contents:** [What to run](#what-to-run) | [The code](#the-code) |
[The data](#the-data) | [The bitstreams](#the-bitstreams) |
[Using them](#using-them) | [How they were verified](#how-they-were-verified) |
[What is not covered](#what-is-not-covered) | [License](#license)

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

For the files one at a time -- what each one is, and which decoder path it
was written to reach -- see **[`BITSTREAMS.md`](BITSTREAMS.md)**;
**[`REACHES.md`](REACHES.md)** is the same set indexed the other way round,
by the path rather than by the file.

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
| [`BITSTREAMS.md`](BITSTREAMS.md) | Every file with the line its case calls itself, grouped. |
| [`REACHES.md`](REACHES.md) | The same set the other way round: every decoder path the probes measure, and which files reach it. |
| [`HOWTO.md`](HOWTO.md) | How to write a case, read a real file back into one, and add one here. |
| [`SYNTAX.md`](SYNTAX.md) | The whole case syntax, generated from `src/grammar.py`. |
| [`expected.txt`](expected.txt) | Name and expected verdict, one line per file. |
| [`hashes.txt`](hashes.txt) | SHA-256 of each decoding file's `-pam` output, so a silent change in decoded pixels fails too. |
| [`coverage.txt`](coverage.txt) | Which decoder path each file actually reached. |
| [`refs.txt`](refs.txt) | What each line a note points at said when it was written. |
| [`COPYING`](COPYING) | BSD 3-clause, the same as libwebp. |

## The bitstreams

The files themselves are in **[`files/`](files)**, each with its size and
expected verdict, and the text they are assembled from is in
**[`cases/`](cases)**.

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

**[`BITSTREAMS.md`](BITSTREAMS.md)** takes those groups a file at a time, with
the line each case calls itself, and **[`REACHES.md`](REACHES.md)** turns them
round -- every decoder path the probes measure, and which files reach it, which
is the question someone holding a decoder actually has. Both are generated.

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
  scaling knob, and then through every entry point libwebp exports. As of
  8/26 that is 61% of regions and 68% of lines from the files alone,
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
