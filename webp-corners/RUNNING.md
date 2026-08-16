# Running the suite

How to point the suite at a decoder, what it checks, and what a decoder that
is not libwebp has to do to run it. [`README.md`](README.md) says what the
suite is.

**Contents:**

* [The four roles](#the-four-roles)
* [What the check is](#what-the-check-is)
* [Running another decoder](#running-another-decoder)
* [Every knob](#every-knob)
* [The scripts](#the-scripts)
* [The files it reads and writes](#the-files-it-reads-and-writes)

## The four roles

A WebP file is read by more than one kind of decoder, and each sees a
different part of it. Every file carries a verdict per role, so a decoder is
only ever judged on what it is for.

**Still image decoder.** Reads one image out of a container. Refuses
anything that declares animation.

**Animation decoder.** Reads the frames, composes them over a canvas, and
returns each one. It is the only role that opens the animated files at all.

**Container parser.** Walks the chunks without decoding an image. It reaches
errors a decoder never gets to, and it accepts files a decoder rejects.

**Incremental decoder.** The still decoder fed a growing buffer. It must
reach the same verdict as the one-shot decoder. Where it does not,
`expected.txt` records both.

## What the check is

    DWEBP=/path/to/dwebp ./check.sh

For each file, `check.sh` decodes it, compares the outcome against the
verdict in `expected.txt`, and for a file that must decode, hashes the pixels
and compares against `hashes.txt`.

The hash is the SHA-256 of the decoded image written as
[PAM](https://netpbm.sourceforge.net/doc/pam.html), which is this header:

    P7
    WIDTH <w>
    HEIGHT <h>
    DEPTH 4
    MAXVAL 255
    TUPLTYPE RGB_ALPHA
    ENDHDR

followed by 8-bit RGBA rows, top to bottom, alpha not premultiplied. Every
line ends in one newline and there is no padding anywhere. For an animation
the hash covers every frame written that way, concatenated in order. Any
decoder that can produce those bytes can reproduce the hash.

## Running another decoder

`check.sh` drives libwebp's tools because that is what the environment names.
A decoder that is not libwebp needs an adapter that does three things:

1. Decode a file and report success or failure. Failure must be a reported
   error, never a crash, an out-of-bounds read, or a partial image returned
   as success. Under a sanitizer is where this is worth doing.
2. For a file whose verdict is `ok`, write the decoded pixels in the form
   above and hash them.
3. Read `expected.txt` for the verdict of the role being tested.

`expected.txt` is `name|still|slow|animation|container|incremental`, one line
per file, empty where a role has nothing to say. `hashes.txt` is `name
sha256`. Both are plain text, and they are the whole contract.

## Every knob

Nothing here looks for a decoder on its own account. Every tool is named in
the environment, so the thing under test is always the one you meant. A
missing tool is reported and skipped, never silently passed over.

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

## The scripts

The first two run the files through a decoder. The rest rebuild the corpus or
check the tools that write it.

| file | what it is |
| --- | --- |
| [`check.sh`](check.sh) | Decodes every file and checks the verdict and the pixels. Drives `$DWEBP`, and `$ANIM_DUMP` and `$WEBPINFO` for the animations. The one to run. |
| [`asan_sweep.sh`](asan_sweep.sh) | The same files in 14 output modes, against a sanitizer build named by `$ASAN_DWEBP`. |
| [`coverage.sh`](coverage.sh) | Measures how much of libwebp the suite reaches, from an instrumented build in a throwaway worktree. Reports the three passes the README tabulates. |
| [`generate.py`](generate.py) | Assembles `files/` from `cases/`. Writes every generated page and `expected.txt`, and refuses to finish when a link, a claim or an example no longer holds. |
| [`make_hashes.sh`](make_hashes.sh) | Rewrites `hashes.txt`, once the new output is known to be right. |
| [`make_coverage.sh`](make_coverage.sh) | Rewrites `coverage.txt`: which construct each file reaches, measured rather than claimed. |
| [`vp8_selftest.py`](vp8_selftest.py) | Checks the assemblers against real encodes. Only needed if you change them. |

## The files it reads and writes

| file | what it is |
| --- | --- |
| [`expected.txt`](expected.txt) | The verdict per role, one line per file. The contract. |
| [`hashes.txt`](hashes.txt) | The SHA-256 of the decoded pixels, for every file that must decode. A silent change in output fails too. |
| [`webp-corners.tgz`](webp-corners.tgz) | Every bitstream in one file, for taking them without the repository around them. |
| [`files`](files) | The bitstreams themselves, each with its size and verdict. |
| [`cases`](cases) | The text each one is assembled from. |
| [`coverage.txt`](coverage.txt) | Which construct each file was measured to reach. |
| [`refs.txt`](refs.txt) | What each line a note cites said when it was written. |
| [`COPYING`](COPYING) | BSD 3-clause, the same as libwebp. |
