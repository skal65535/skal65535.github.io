# Writing a case

A bitstream here is a text file in [`cases/`](cases), assembled into bytes.
This is how to write one, how to read an existing `.webp` back into one, and
what a program generating them needs to know. [`SYNTAX.md`](SYNTAX.md) is
the reference for every keyword; this is the part that is not a table.

## The shortest path

A case is field names, one per line. Save this as `my.txt`:

```case
lossless
width 4
height 4
argb 0xff112233 x16
```

and assemble it:

    ./src/webp_asm.py my.txt /tmp/out.webp

That is the whole loop. `webp_asm.py` reads the case, picks the assembler
that owns it, and writes the file; nothing else is needed to make a
bitstream.

A lossy frame instead, since a case saying `lossless` is a VP8L image and
anything else is a VP8 frame:

```case
width 32
height 32
yac_qi 30
```

Both of those leave almost everything unsaid. **Every field has a default**,
so a case says only what it is about -- the lossy one above has 4
macroblocks it never mentions, filled in with default modes and no
coefficients.

## Nothing is checked on the way

This is the point of the whole thing, and the one rule to keep in mind:

**A value too big for its field loses its top bits rather than being
refused.** `cache_bits 15` writes 15 into a 4-bit field that a decoder only
accepts 1..11 in. `magic 0x00` writes a signature byte no decoder will take.
That is how a stress case is made, and it is why the ranges in `SYNTAX.md`
are the width of the *bitstream field*, not what libwebp accepts.

The handful of things the assemblers do refuse are the ones they could not
write at all: a symbol a declared code has no entry for, a tile list that is
not the length the transform implies.

## The four layers

A case can describe any of them, and says nothing about the ones it does not
mention.

**The image** -- a VP8 frame in RFC 6386's field names, or a VP8L image.
This is what the two examples above are.

**The container**, in RFC 9649's names. A case that says nothing here gets a
plain `RIFF....WEBP` around its one image chunk; say something and you get
the chunk list you asked for:

```case
chunks VP8X VP8
alpha 1
icc_profile 1
payload ICCP 00112233
width 32
height 32
yac_qi 30
```

`chunks` is the spine: the fourccs to write, in order. Listing one twice is
allowed, and so is listing one nothing else in the case mentions -- that is
how an unknown chunk is made. `payload FOURCC HEX` fills in a chunk this
repository has no builder for, and overrides the builder when it does.

**An animation.** `frame` opens a block, and everything after it belongs to
that frame until the next block:

```case
frame
frame_duration 100
lossless
width 16
height 16
argb 0xffff0000 x256
frame
frame_x 4
frame_duration 100
lossless
width 8
height 8
argb 0xff00ff00 x64
```

A file with frames in it defaults to `VP8X ANIM ANMF ANMF`, the animation
flag set, and a canvas the frames fit in, so the case above says only what
it is choosing. Every one of those is a field like any other: set
`animation 0` and you get the frames without the flag.

**A compressed alpha plane.** `alph_plane` opens the other kind of block: an
alpha plane is a lossless image stream with its five-byte header left off,
and its green channel is the alpha. It attaches to the frame above it, or to
the file when there is none:

```case
chunks VP8X ALPH VP8
alpha 1
width 16
height 16
yac_qi 30
alph_plane
pixels 200 x256
```

Blocks are the only nesting the syntax has, and they are here because the
format nests here.

## Going the other way

Any `.webp` can be read back into a case, which is usually a better start
than a blank file:

    ./src/webp_dis.py some-animation.webp
    ./src/webp_dis.py --check some-animation.webp

`--check` reassembles what it read and compares the bytes, so it says
whether the case it printed really is that file. Take a real encode, read it
into text, change one line, assemble: that is the shortest route to a file
that is *almost* valid, which is usually what is wanted.

`src/vp8_dis.py` and `src/vp8l_dis.py` do the same for a bare frame or image
if the container is not interesting.

## Adding one to the corpus

A case in `cases/` needs a keyed header. `note`, `expect` and `exercises`
are required:

    # note: what this file is, in a sentence
    # expect: reject
    # exercises: which decoder path it reaches, and why that matters

then:

1. `python3 generate.py` -- assembles it into `files/`, and rebuilds
   `README.md`, `expected.txt` and the indexes.
2. `./check.sh` -- decodes it and says whether `expect` was right. Getting
   this wrong is normal; the file usually fails for a different reason than
   the one intended.
3. `LIBWEBP=... ./make_coverage.sh` -- records which decoder paths it
   actually reached. **Write the `exercises` line from that output, not from
   reading the source.** Several notes here were wrong until this was run.
4. `./make_hashes.sh` -- once the decoded output is known to be right.
5. `LIBWEBP=... ./coverage.sh` -- optional, and the test of whether the case
   earned its place: it says how much of `src/dec` and `src/demux` the whole
   corpus reaches. One that moves nothing is covering ground another file
   already holds.

The other header keys are optional and each says something no one can infer:
`roundtrip: no` for a case no disassembler can reproduce, `anim` and `info`
for the animation decoder's and webpinfo's verdicts, `incremental` for the
streaming decoder's when it differs, `unique` naming a probe the case claims
to be the only file reaching, and `slow` for one that allocates a gigabyte.

## For a program writing cases

`./src/grammar.py` prints the whole grammar as JSON -- every keyword, how
many values it takes, the kind and range of each, the enums, the constants.
**Read that rather than `SYNTAX.md`**, which is generated from it.

Two things it cannot tell you:

**A case that reaches nothing new is not worth having.** The corpus is
graded by what `coverage.txt` says each file reached, not by how strange it
looks. A random walk over the field ranges mostly produces files that are
rejected in the first hundred bytes, which one existing case already covers.

**Say one thing at a time.** Almost every file here differs from a valid one
in a single field, so when a decoder does something surprising the file
names the reason. A case with six broken fields tells you nothing about
which one mattered.

## What will bite a generator

These cost me time, and none of them is visible in the grammar:

* **A code with one symbol must be written with length 1**, not 0. The
  decoder then makes it zero bits wide via `BuildHuffmanTable()`'s
  single-value shortcut; length 0 is an empty code and a rejection.
* **A group whose five codes are all single-symbol costs no bits at all.**
  The decoder takes the whole pixel from the group and reads nothing, so
  writing pixel data for it changes no byte.
* **The code-length stream must cover the whole alphabet** or set
  `max_symbol`; the decoder loops until it has as many lengths as symbols.
* **Palette pixels are packed, and the case writes the packed bytes.** With
  3 colours the index is 2 bits and one byte holds four pixels, so a 16-wide
  image is 4 packed pixels per row, not 16.
* **`pixels` and `argb` are not a choice.** The decoder reads red, blue and
  alpha only when those three codes are not all single-symbol, so a case
  whose pixels vary in one of them must spell whole pixels with `argb`.
* **Distance 1 is plane code 2.** `kCodeToPlane[0]` is 0x18, which is a
  distance of one row, not one pixel.
* **A 16x16-mode macroblock's luma position 0 belongs to `y2`** and the
  assembler refuses a coefficient there.
* **`-` is not `0`.** The RFC splits an optional value into an update flag, a
  magnitude and a sign; `-` writes the flag alone, `0` writes the flag and a
  zero. Two different bitstreams.
* **An animation frame's offsets are stored halved.** `frame_x 4` puts the
  frame at x = 8; odd pixel offsets cannot be expressed.

## What runs it

[`README.md`](README.md) has the whole list, but the two that answer "does
my decoder survive this?" are `check.sh` and `asan_sweep.sh`, and both take
the decoder to use from the environment rather than finding one for you.
