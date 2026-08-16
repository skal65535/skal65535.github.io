# webp-corners: the code

What the scripts one directory up are built out of.
[`../generate.py`](../generate.py) drives all of it. Each assembler and
disassembler is also a command of its own, run from the directory above,
where `cases/` is. [`../HOWTO.md`](../HOWTO.md) says what to do with them.
This says how they fit together.

Three layers. A case only ever touches the top one.

**`webp_asm.py`** reads the case. It splits the container directives from
the image ones and hands the image to the assembler that owns it: a case
saying `lossless` is a VP8L image, anything else a lossy VP8 frame. It wraps
the result in RIFF. It also owns the two places the format nests. A `frame`
block is an animation frame with an image of its own. An `alph_plane` block
is a compressed alpha plane, which is a lossless image stream with its
header left off.

**`vp8l_asm.py`** and **`vp8_asm.py`** turn the text into bitstream fields.
Their docstrings are the format: every keyword, its default, and what it
writes. Nothing is validated or clamped. A value too big for its field loses
its top bits, which is usually the point.

**`vp8l.py`** and **`vp8.py`** do the bit-level work: the boolean coder,
canonical Huffman codes, prefix codes, the sub-image streams. They validate
nothing either.

`vp8_dis.py`, `vp8l_dis.py` and `webp_dis.py` go the other way. Disassemble
a file, reassemble from its own text, compare the bytes: that is how
[`../vp8_selftest.py`](../vp8_selftest.py) checks the writers against real
encodes rather than against themselves. `--check` does it for any file you
point it at. The first two read one chunk. `webp_dis.py` reads a whole file,
which is the only way to read an animation or an alpha plane.

`grammar.py` holds every keyword and the range of every value.
[`../SYNTAX.md`](../SYNTAX.md) is generated from it, so the reference cannot
drift from the code. A program writing cases should read `grammar.py`
directly: `./grammar.py` prints the same thing as JSON.

## The files

| file | what it is |
| --- | --- |
| [`vp8l.py`](vp8l.py) | VP8L lossless bitstream writer: bit packing, canonical Huffman codes, prefix coding, sub-images. |
| [`vp8l_asm.py`](vp8l_asm.py) | Assembles a lossless image from a text case. Its docstring is the format. |
| [`vp8.py`](vp8.py) | VP8 lossy bitstream writer: the boolean coder, the frame header, the mode trees, the coefficients. |
| [`vp8_asm.py`](vp8_asm.py) | Assembles a lossy frame from a text case, in RFC 6386's field names. Its docstring is the format. |
| [`webp_asm.py`](webp_asm.py) | Wraps either in a RIFF container, in RFC 9649's field names, and picks which assembler a case belongs to. |
| [`vp8_dis.py`](vp8_dis.py) | The other direction for a lossy frame. `--check` round trips one against libwebp. |
| [`vp8l_dis.py`](vp8l_dis.py) | The other direction for a lossless image, the same way. |
| [`webp_dis.py`](webp_dis.py) | The other direction for a whole file: chunks, animation frames and alpha planes, delegating each image to one of those two. |
| [`grammar.py`](grammar.py) | Every keyword and the range of every value, as data. `SYNTAX.md` is generated from it. |
| [`vp8_tables.py`](vp8_tables.py) | The VP8 constant tables, extracted from libwebp. |
| [`make_vp8_tables.py`](make_vp8_tables.py) | Extracts them, so they are never retyped. |
| [`lossy_parts.py`](lossy_parts.py) | The multi-partition lossy cases, patched from `sources/`. |
| [`make_partition_sources.c`](make_partition_sources.c) | Rebuilds `sources/`: cwebp cannot emit more than one token partition. |
| [`probes.py`](probes.py) | The `fprintf` probes `make_coverage.sh` patches in. |
| [`api_sweep.c`](api_sweep.c) | Every decoding entry point libwebp exports, for `coverage.sh`: the incremental decoder fed a few bytes at a time, caller-allocated buffers, the colorspaces dwebp cannot ask for, the demuxer's iterators. |
| [`check_refs.py`](check_refs.py) | Checks that the source lines the notes point at still say what the notes claim. |
