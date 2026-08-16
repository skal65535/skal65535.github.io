# WebP stress bitstreams

A test suite for WebP decoders. Each file targets one construct of the
format: a container chunk, a header field, a Huffman code, a back-reference,
an animation frame. No encoder emits these files. They are written field by
field from text, so a file can say what an encoder cannot.

Every file states what a decoder must do with it. Any decoder can be held to
that, not only the reference one.

**Contents:**

* [What a decoder must do](#what-a-decoder-must-do)
* [What the suite covers](#what-the-suite-covers)
* [Running them](#running-them)
* [What it reaches in libwebp](#what-it-reaches-in-libwebp)
* [Limits](#limits)
* [Writing a case](#writing-a-case)
* [License](#license)

## What a decoder must do

Every file carries a verdict.

**ok** -- the decoder must decode the file, and must produce the pixels
recorded in `hashes.txt`.

**reject** -- the decoder must refuse the file and report an error. It must
not crash, read out of bounds, or return a partial image as success.

A verdict belongs to a decoder role. A still decoder refuses every animation
before it reads a frame, and that is correct; for those files the animation
decoder's verdict is the one that describes the file. A container parser
reports errors an image decoder never reaches. `expected.txt` carries one
column per role.

A verdict names no status code. The format says what is malformed. How a
decoder reports it is its own business.

The field names here are the specifications' own:
[RFC 6386](https://www.rfc-editor.org/rfc/rfc6386.html) for the lossy
bitstream, [RFC 9649](https://www.rfc-editor.org/rfc/rfc9649.html) for the
container.

## What the suite covers

**The RIFF container.** The VP8X chunk and the canvas it declares. Optional
chunks a decoder steps over by their declared length. Headers that lie about
what follows them.

**Lossy frames (VP8).** Every field of the frame header. Segmentation, loop
filter and quantizer records. The token coder out to its escape categories.
One, two, four and eight token partitions.

**Lossless images (VP8L).** Huffman codes and the code-length code that
describes them. Colour caches, back-references and the four transforms. The
entropy image that changes codes mid-row.

**Alpha planes.** The plane stored a byte per pixel, through each of the four
filters. The plane compressed by the lossless coder, in the 8-bit mode only an
alpha chunk reaches.

**Animation.** Frame position, duration, disposal and blending, composed over a
canvas one frame at a time.

[`BITSTREAMS.md`](BITSTREAMS.md) lists every file and what it is for.
[`REACHES.md`](REACHES.md) indexes them by the construct they exercise, which
is the way round to read it when a decoder has just failed one.

## Running them

    DWEBP=/path/to/dwebp ./check.sh

`check.sh` decodes every file, checks the verdict, and compares the decoded
pixels against `hashes.txt`. `asan_sweep.sh` runs the same files through a
sanitizer build in 14 output modes.

Both drive libwebp's tools. [`RUNNING.md`](RUNNING.md) is the whole of it:
the other decoders each file needs, what the pixel hash covers, and what a
decoder that is not libwebp has to do to run the suite.

Take the bitstreams alone from **[`webp-corners.tgz`](webp-corners.tgz)**,
19 kB, or the directory with its scripts:

    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/skal65535/skal65535.github.io.git
    cd skal65535.github.io
    git sparse-checkout set webp-corners

## What it reaches in libwebp

The suite is graded by measurement rather than by inspection. `coverage.sh`
builds libwebp with instrumentation and reports `src/dec` and `src/demux`
three times over, at 0be8ddd1:

| driven by | regions | lines | branches |
| --- | ---: | ---: | ---: |
| The bitstreams alone, as `check.sh` runs them | 61% | 68% | 55% |
| The same files, through every output and scaling option | 68% | 80% | 63% |
| The same files, through every decoding entry point | 81% | 92% | 77% |

The first row is what a bitstream controls. The rest is what a caller
controls: output conversion, rescaling, allocation failure. Separating the
two says whether a gap belongs to the suite or to code no file can reach.

## Limits

**Inter frames.** A WebP file carries a key frame. One file checks that a
decoder refuses an inter frame, and nothing here goes further.

**Encoding.** Nothing here tests an encoder or a muxer.

**Partial input.** Every tool hands the decoder a whole file. A decoder that
is fed a growing buffer is checked for the same verdict, not for its
behaviour at every byte boundary.

**Fields a decoder may ignore.** The profile and the entropy-refresh bit are
written at both ends of their range. A decoder that starts acting on either
fails a pixel hash rather than passing unnoticed.

## Writing a case

A case is a text file naming bitstream fields, one per line, and nothing
validates it on the way out. [`HOWTO.md`](HOWTO.md) is the walk-through and
[`SYNTAX.md`](SYNTAX.md) the reference:

    ./src/webp_asm.py cases/alph-raw-filter-gradient.txt /tmp/out.webp
    ./src/webp_dis.py --check some-animation.webp

The second reads a real `.webp` back into case text, which is the shortest
route to a file that is almost valid.

## License

BSD 3-clause, the same as libwebp. See [`COPYING`](COPYING). That covers the
generators, the scripts and the bitstreams in `files/` alike.
