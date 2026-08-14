# webp-torture: the code

Everything the scripts one directory up are built out of. Nothing here is
run directly to produce the corpus -- [`../generate.py`](../generate.py)
does that -- though each assembler doubles as a command that turns one case
into one `.webp`. Run them from the directory above, where `cases/` is:

    ./src/webp_asm.py cases/alph-raw-filter-gradient.txt /tmp/out.webp
    ./src/vp8l_asm.py cases/codelen-depth-15.txt /tmp/out.webp
    ./src/vp8_asm.py cases/lossy-coeff-cat6.txt /tmp/out.webp
    ./src/vp8_dis.py some-photo.webp

Three layers, and a case only ever touches the top one:

* **`webp_asm.py`** reads the case, splits the container directives from the
  image ones, and hands the image to whichever assembler owns it: a case
  saying `lossless` is a VP8L image, anything else a lossy VP8 frame. It
  then wraps the result in RIFF.
* **`vp8l_asm.py`** and **`vp8_asm.py`** turn the text into the fields of a
  bitstream. Their docstrings are the format: every keyword, its default and
  what it writes. Nothing is validated or clamped -- a value too big for its
  field loses its top bits, which is usually the point.
* **`vp8l.py`** and **`vp8.py`** do the bit-level work: the boolean coder,
  canonical Huffman codes, prefix codes, the sub-image streams. They
  validate nothing either.

`vp8_dis.py` goes the other way, lossy only, and is what
[`../vp8_selftest.py`](../vp8_selftest.py) uses to check the writer against
real encodes rather than against itself. There is no VP8L disassembler yet.

`grammar.py` is the third thing a case touches, though not at assembly time:
it holds every keyword and the range of every value, and
[`../SYNTAX.md`](../SYNTAX.md) is generated from it, so the reference cannot
drift from the code.

## The files

| file | what it is |
| --- | --- |
| [`vp8l.py`](vp8l.py) | VP8L lossless bitstream writer: bit packing, canonical Huffman codes, prefix coding, sub-images. |
| [`vp8l_asm.py`](vp8l_asm.py) | Assembles a lossless image from a text case. Its docstring is the format. |
| [`vp8.py`](vp8.py) | VP8 lossy bitstream writer: the boolean coder, the frame header, the mode trees, the coefficients. |
| [`vp8_asm.py`](vp8_asm.py) | Assembles a lossy frame from a text case, in RFC 6386's field names. Its docstring is the format. |
| [`webp_asm.py`](webp_asm.py) | Wraps either in a RIFF container, in RFC 9649's field names, and picks which assembler a case belongs to. |
| [`vp8_dis.py`](vp8_dis.py) | The other direction: a lossy .webp back into that text. `--check` round trips one against libwebp. |
| [`grammar.py`](grammar.py) | Every keyword and the range of every value, as data. `SYNTAX.md` is generated from it. |
| [`vp8_tables.py`](vp8_tables.py) | The VP8 constant tables, extracted from libwebp. |
| [`make_vp8_tables.py`](make_vp8_tables.py) | Extracts them, so they are never retyped. |
| [`lossy_parts.py`](lossy_parts.py) | The multi-partition lossy cases, patched from `sources/`. |
| [`make_partition_sources.c`](make_partition_sources.c) | Rebuilds `sources/`: cwebp cannot emit more than one token partition. |
| [`probes.py`](probes.py) | The `fprintf` probes `make_coverage.sh` patches in. |
