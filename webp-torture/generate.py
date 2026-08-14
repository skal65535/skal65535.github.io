#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.
"""Generates the torture bitstreams. Run: python3 generate.py [outdir]

Every file but the multi-partition ones comes from a text case in cases/,
which carries its own note; this assembles them into files/ and writes
everything that is derived from them -- expected.txt, README.md, SYNTAX.md,
src/README.md and an index per directory. It also refuses to finish if a
link does not resolve, a 'unique:' claim disagrees with coverage.txt, or an
example in HOWTO.md does not assemble.
"""

import glob
import os
import re
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'src'))

import grammar
import lossy_parts
import vp8_asm
import webp_asm

SLOW = set()      # the cases that allocate enough memory to be worth skipping

# A row is (name, expect, note, exercises, size, anim, info, incremental,
# unique). The four after the size are what one decoder cannot say:
#   anim         the animation decoder's verdict, empty for a still. dwebp
#                refuses every animated file before looking at a frame, so
#                'expect' says nothing about those beyond that.
#   info         webpinfo's, worth recording where it disagrees since it
#                reads the same container to stricter rules.
#   incremental  the streaming decoder's, and only when it differs from
#                'expect', which is the whole reason to write it down.
#   unique       probes the case claims to be the only file reaching.
ANIM, INFO, INCR, UNIQUE = 5, 6, 7, 8


def verdict(row):
    """What a decoder must do with one file, in the words of the table.

    webpinfo is named only when it disagrees: it walks the container without
    decoding, so on its own it says less than the other two.
    """
    if row[INCR]:
        return '%s, incremental %s' % (row[1], row[INCR])
    if not row[ANIM]:
        return row[1]
    out = '%s, anim_dump %s' % (row[1], row[ANIM])
    if row[INFO] and row[INFO] != row[ANIM]:
        out += ', webpinfo %s' % row[INFO]
    return out


def wanted(row):
    """The verdict that is about the file rather than about dwebp."""
    return row[ANIM] or row[1]


GROUPS = [
    ('lossless-', 'Whole lossless images',
     'The two ends of the format rather than one corner of it: an image with '
     'nothing optional in it at all, and one with everything. Between them '
     'they are what the rest of the lossless files here are a departure '
     'from.'),
    ('simple-', 'Simple codes',
     'The 1-or-2-symbol shorthand a Huffman code can take. Its symbols are '
     'read as raw 8-bit values and are never checked against the alphabet '
     'size, so this is where a stream can say things an encoder cannot.'),
    ('codelen-', 'The code-length code',
     'The Huffman code that describes the lengths of another Huffman code, '
     'plus its repeat escapes (16, 17, 18) and the optional max_symbol '
     'field. cwebp only ever emits a narrow slice of this.'),
    ('meta-', 'Meta Huffman / entropy image',
     'The sub-image that picks one of several code groups per tile, and the '
     'remapping the decoder does when the group count looks implausible.'),
    ('cache-', 'Color cache', 'Size bounds, and cache-index literals.'),
    ('subimage-', 'Sub-images',
     'A lossless file carries whole image streams inside itself: one for '
     'each transform that needs a per-tile parameter, and one for the '
     'entropy image. Each is read by the same DecodeImageStream() as the '
     'outer image, minus the transforms and the entropy image it is not '
     'allowed to have of its own -- so each has a color cache and five '
     'Huffman codes that a file can say something about, and that cwebp '
     'always writes the same dull way.'),
    ('transform-palette-', 'Palette packing',
     'Index width follows the palette size, and the map is padded out to the '
     'packing capacity with black.'),
    ('transform-', 'Transforms',
     'Presence, repetition and tile sizes.'),
    ('lz77-', 'Back-references', 'Copy lengths and distances.'),
    ('predictor-', 'Predictor modes',
     'The per-tile predictor index. It is read as a 4-bit field, so all 16 '
     'values are reachable, but the format only defines 14 of them.'),
    ('header-', 'Frame header',
     'The 14-bit dimension fields and the version escape.'),
    ('container-', 'The RIFF container',
     'The layer above the image: the RIFF header, the extended-format VP8X '
     'chunk and its canvas size, the optional chunks a decoder must step '
     'over by their declared length alone, and the padding rule that makes '
     'an odd-sized one even on disk. Everything here is read by '
     'webp_dec.c before the frame is looked at.'),
    ('anim-', 'Animation',
     'A sequence of frames rather than one image: an ANIM chunk with the '
     'loop count, then an ANMF per frame carrying its own position, '
     'duration, disposal and blending, and its own image chunks. None of it '
     'is reachable through dwebp, which refuses any file claiming animation '
     'before looking at a frame, so these are checked with anim_dump '
     'instead -- the demuxer of demux.c, the composition of anim_decode.c, '
     'and one decode per frame. The verdict quoted for each is that one; '
     'every file here is still a reject to a still decoder.'),
    ('alph-', 'The alpha chunk',
     'ALPH carries the alpha plane beside a lossy frame: a header byte of '
     'four two-bit fields, then the plane itself, either stored as it is or '
     'compressed with the lossless coder in its 8-bit mode. That mode is a '
     'separate path through vp8l_dec.c from the one every VP8L image here '
     'takes, and an alpha chunk is the only thing that reaches it -- whether '
     'it sits beside a still frame or inside an animation frame. Each of the '
     'four filters has a routine of its own in dsp/filters.c, and the same '
     'stored bytes come out as four different planes, so the pixel hash is '
     'what tells those apart. A compressed plane is a lossless image stream '
     'with its header left off, so the alph-plane cases write one from text '
     'and can break each of the four conditions the 8-bit mode asks for.'),
    ('lossy-frame-', 'Lossy: frame tag and picture header',
     'The ten uncompressed bytes every lossy frame starts with: the profile, '
     'the visibility and key-frame bits, the length of partition 0, the '
     'start code and the two 14-bit dimensions.'),
    ('lossy-segment-', 'Lossy: segmentation',
     'Up to four segments, each with its own quantizer and loop-filter '
     'strength, and a per-macroblock map saying which is which. cwebp uses '
     'the feature but only ever writes absolute values, and always writes '
     'the map and the data together.'),
    ('lossy-filter-', 'Lossy: loop filter',
     'The in-loop deblocking filter: simple or normal, its level and '
     'sharpness, and the per-reference and per-mode deltas. '
     'PrecomputeFilterStrengths() shifts the interior limit right by one for '
     'sharpness 1 to 4 and by two for 5 to 7, then clamps it to 9 - '
     'sharpness, which is what the sharpness files sit either side of.'),
    ('lossy-quant-', 'Lossy: quantizer',
     'The frame quantizer index and the five deltas around it, one per plane '
     'and coefficient kind, with clamps that are not all the same.'),
    ('lossy-proba-', 'Lossy: coefficient probabilities',
     'The 1056 probabilities that drive the coefficient coder, each one '
     'optionally replaced in the frame header, plus the skip probability.'),
    ('lossy-mode-', 'Lossy: prediction modes',
     'The 16x16 and 4x4 luma modes and the chroma modes, and the '
     'neighbour-indexed probability table the 4x4 modes are coded with.'),
    ('lossy-coeff-', 'Lossy: coefficients',
     'The token coder of section 13: magnitudes and their escape categories, '
     'end-of-block, zero runs, and the four coefficient types. The band '
     'sweeps below share a trick: the three token classes drive the context '
     'of the next position -- a zero gives 0, a +-1 gives 1, anything larger '
     'gives 2 -- so a block of each, walked to position 15, reads every band '
     'at that class, and placing the blocks so their neighbour contexts are '
     "0, 1 and 2 in turn is the only way to reach band 0, which is never a "
     "token's successor."),
    ('lossy-skip-', 'Lossy: skipped macroblocks',
     'The per-macroblock skip flag, which drops the residual entirely and '
     'clears the neighbouring non-zero flags -- almost all of them.'),
    ('lossy-parts-', 'Lossy: token partitions',
     'A lossy frame may carry 1, 2, 4 or 8 token partitions, macroblock row '
     'r being read from partition r & (n - 1). cwebp does not expose '
     'config.partitions and libwebp forces it back to 1 whenever the token '
     'path is used (webp_enc.c:124), so none of this is reachable through '
     'the tools.'),
    ('lossy-truncated-', 'Lossy: truncation',
     'Frames that stop early, at each of the places the decoder can notice: '
     'inside partition 0, inside the macroblock modes, and inside the token '
     'data.'),
    ('lossy-', 'Lossy: partition sizes, from real encodes',
     'The four files behind these are genuine encoder output, made through '
     'the encoder API rather than cwebp, and the broken ones rewrite the raw '
     'size table that follows partition 0. They carry a whole frame of real '
     'coefficients, which the assembled cases do not.'),
]

README_HEAD = """# WebP torture bitstreams

Small WebP files that exercise corners of the format a normal encoder never
emits, one layer of it at a time.

**They are written, not captured.** Each one is a text case in
[`cases/`](cases) naming bitstream fields, one per line, under the names the
specification gives them, and an assembler in [`src/`](src) turns the case
into bytes. Nothing is checked on the way, so a case can say what no encoder
would -- and the file it produces is readable back as the text that
describes it. [`HOWTO.md`](HOWTO.md) is how to write one;
[`SYNTAX.md`](SYNTAX.md) is the reference, generated from
[`src/grammar.py`](src/grammar.py).

* **%(vp8l)d lossless (VP8L) streams**
  ([`vp8l_asm.py`](src/vp8l_asm.py)): the header, the transforms, the
  Huffman codes down to the code-length stream that describes them, and the
  pixel data.
* **%(lossy)d lossy VP8 frames** ([`vp8_asm.py`](src/vp8_asm.py)), under the
  field names [RFC 6386](https://www.rfc-editor.org/rfc/rfc6386.html) gives
  them. Seven of them are the exception to the paragraph above: a frame may
  carry up to eight token partitions and cwebp emits only one, so
  [`make_partition_sources.c`](src/make_partition_sources.c) makes four
  through the encoder API into [`sources/`](sources), and
  [`lossy_parts.py`](src/lossy_parts.py) turns those into seven -- the four
  as they are, plus three with the partition-size table rewritten.
* **%(container)d RIFF containers** ([`webp_asm.py`](src/webp_asm.py)), in
  [RFC 9649](https://www.rfc-editor.org/rfc/rfc9649.html)'s names: the
  extended-format VP8X chunk, the optional chunks a decoder must step over,
  and sizes that lie about what is behind them.
* **%(alpha)d alpha chunks**, where the plane is either stored one byte per
  pixel or compressed with the lossless coder in its 8-bit mode -- a
  different path through the decoder from the one every VP8L file here
  takes. A compressed plane is a lossless image stream without a header, so
  `vp8l_asm.py` writes those too.
* **%(anim)d animations** ([`webp_asm.py`](src/webp_asm.py) again): an ANIM
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

A file read by more than one decoder carries more than one verdict, and
they do not always agree. An animation reads `reject, anim_dump ok`: dwebp
returns UNSUPPORTED_FEATURE for anything claiming animation before it looks
at a frame, so that half says nothing about the file and the second half is
the one that does. Where `webpinfo` or the incremental decoder disagrees
with the rest, it is named too, and `check.sh` holds all of them to it.

%(files)s
%(code)s
## Using them

Every file is one click away from this page, but the corpus is one
directory of a much larger repository, so to take the whole thing at once
ask git for just that directory:

    git clone --depth 1 --filter=blob:none --sparse \\
        https://github.com/skal65535/skal65535.github.io.git
    cd skal65535.github.io
    git sparse-checkout set webp-torture

That fetches about 2MB instead of the ~90MB the rest of the site comes to.
Run the scripts above from `webp-torture/`.

To write one of your own, or read a real encode back into a case,
[`HOWTO.md`](HOWTO.md) is the walk-through and [`SYNTAX.md`](SYNTAX.md) the
reference. The short version:

    ./src/webp_asm.py cases/alph-raw-filter-gradient.txt /tmp/out.webp
    ./src/webp_dis.py --check some-animation.webp

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
the three disassemblers in `src/` read a file back into the text the
assemblers take, so a real encode can be disassembled, reassembled and
compared byte for byte. `vp8_selftest.py` runs that over `sources/`, over a
spread of images it asks cwebp to encode both ways, over animations it asks
webpmux to build, and over the corpus itself; it also writes every
coefficient magnitude the format allows, and pairs of frames that say the
same thing two different ways and must decode alike. A case that cannot
survive the trip says `roundtrip: no`, and the selftest fails a case that
says so and then survives it anyway.

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
"""


def wrap(text, indent=''):
    return '\n'.join(textwrap.wrap(text, 79, initial_indent=indent,
                                   subsequent_indent=indent)) + '\n'


def build_index(rows, groups):
    """A compact table: one row per group, with the ok/reject split."""
    out = ['| Group | Files | must decode | must be rejected |',
           '| --- | ---: | ---: | ---: |']
    seen = set()
    for prefix, title, _ in groups:
        group = [r for r in rows
                 if r[0].startswith(prefix) and r[0] not in seen]
        seen.update(r[0] for r in group)
        if not group:
            continue
        ok = sum(1 for r in group if wanted(r) == 'ok')
        out.append('| %s | %d | %d | %d |' %
                   (title, len(group), ok, len(group) - ok))
    ok = sum(1 for r in rows if wanted(r) == 'ok')
    out.append('| **total** | **%d** | **%d** | **%d** |' %
               (len(rows), ok, len(rows) - ok))
    return '\n'.join(out) + '\n'


INDEX_STYLE = """<!doctype html>
<meta charset="utf-8">
<title>webp-torture bitstreams</title>
<style>
 body { font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 54rem;
        padding: 0 1rem; }
 table { border-collapse: collapse; width: 100%%; }
 th, td { text-align: left; padding: .25rem .6rem; border-bottom: 1px solid #ddd; }
 td.n { text-align: right; font-variant-numeric: tabular-nums; }
 .reject { color: #a33; }
 code { font-size: 90%%; }
</style>
"""

FILES_INDEX_HEAD = INDEX_STYLE + """<h1>webp-torture bitstreams</h1>
<p>%(count)d WebP files that exercise corners of the format a normal encoder
never emits, most of them assembled from the text in
<a href="../cases/">cases/</a>. See the <a href="../">notes</a> for what each
one targets. <b>reject</b> means a conforming decoder must refuse the file.</p>
<table>
<tr><th>file</th><th>bytes</th><th>expected</th></tr>
"""

SOURCES_INDEX_HEAD = INDEX_STYLE + """<h1>webp-torture sources</h1>
<p>Real encoder output, not assembled: %(count)d lossy frames carrying 1, 2, 4
and 8 token partitions, made through the encoder API by
<a href="../src/make_partition_sources.c"><code>make_partition_sources.c</code></a>
because cwebp emits only one. Seven files in
<a href="../files/">files/</a> come from these: these four as they are, and
three more with the partition-size table rewritten to lie about what follows
it. <a href="../">The notes</a> say what each one does.</p>
<table>
<tr><th>file</th><th>bytes</th></tr>
"""

CASES_INDEX_HEAD = INDEX_STYLE + """<h1>webp-torture cases</h1>
<p>%(count)d text cases, each assembled into the .webp of the same name in
<a href="../files/">files/</a>. A case names the fields the specification
names &mdash; RFC 6386 for the lossy frame, RFC 9649 for the container
&mdash; and carries its own note on what it is for; the
<a href="../">notes</a> have the full write-up. <b>reject</b> means a
conforming decoder must refuse it.</p>
<table>
<tr><th>case</th><th>expected</th><th>what it is</th></tr>
"""


def write_files_index(outdir, rows):
    """files/index.html -- GitHub Pages serves no directory listing."""
    lines = [FILES_INDEX_HEAD % {'count': len(rows)}]
    for row in sorted(rows):
        lines.append('<tr><td><a href="%s.webp"><code>%s.webp</code></a></td>'
                     '<td class="n">%d</td><td class="%s">%s</td></tr>'
                     % (row[0], row[0], row[4],
                        'reject' if wanted(row) == 'reject' else 'ok',
                        verdict(row)))
    lines.append('</table>')
    with open(os.path.join(outdir, 'files', 'index.html'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_sources_index(outdir):
    """sources/index.html, so the directory can be linked like the others."""
    names = sorted(f for f in os.listdir(os.path.join(outdir, 'sources'))
                   if f.endswith('.webp'))
    lines = [SOURCES_INDEX_HEAD % {'count': len(names)}]
    for name in names:
        size = os.path.getsize(os.path.join(outdir, 'sources', name))
        lines.append('<tr><td><a href="%s"><code>%s</code></a></td>'
                     '<td class="n">%d</td></tr>' % (name, name, size))
    lines.append('</table>')
    with open(os.path.join(outdir, 'sources', 'index.html'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_cases_index(outdir, rows):
    """cases/index.html -- same reason, and the README links cases/."""
    lines = [CASES_INDEX_HEAD]
    n = 0
    for row in sorted(rows):
        name = row[0]
        if not os.path.exists(os.path.join(outdir, 'cases',
                                           name + '.txt')):
            continue                    # patched from sources/, not assembled
        n += 1
        lines.append('<tr><td><a href="%s.txt"><code>%s</code></a></td>'
                     '<td class="%s">%s</td><td>%s</td></tr>'
                     % (name, name,
                        'reject' if wanted(row) == 'reject' else 'ok',
                        verdict(row), html_escape(row[2])))
    lines.append('</table>')
    lines[0] = CASES_INDEX_HEAD % {'count': n}
    with open(os.path.join(outdir, 'cases', 'index.html'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def check_links(outdir):
    """Every relative link in the generated docs resolves.

    GitHub Pages serves no directory listing, so a linked directory needs an
    index of its own; Jekyll renders a README.md into one, which is how this
    page is served at all.
    """
    for doc, base in (('README.md', '.'), ('src/README.md', 'src'),
                      ('SYNTAX.md', '.'), ('HOWTO.md', '.')):
        with open(os.path.join(outdir, doc)) as f:
            text = f.read()
        for target in re.findall(r'\]\(([^)#][^)]*)\)', text):
            if target.startswith(('http', 'mailto')):
                continue
            path = os.path.normpath(os.path.join(outdir, base, target))
            assert os.path.exists(path), '%s links %s' % (doc, target)
            if os.path.isdir(path):
                assert any(os.path.exists(os.path.join(path, n))
                           for n in ('index.html', 'README.md')), \
                    '%s links %s/, which has no index' % (doc, target)


def check_howto(outdir):
    """Every worked example in HOWTO.md, assembled.

    A document telling someone how to write a case is worth nothing if its
    examples do not assemble, and prose is exactly where that rots. The
    ```case blocks are run rather than read; so are the case files its
    commands name.
    """
    path = os.path.join(outdir, 'HOWTO.md')
    with open(path) as f:
        text = f.read()
    blocks = re.findall(r'(?m)^```case\n(.*?)^```', text, re.S)
    assert blocks, 'HOWTO.md has no ```case examples left to check'
    for n, block in enumerate(blocks, 1):
        try:
            webp_asm.assemble_text(block)
        except vp8_asm.AsmError as e:
            raise AssertionError('HOWTO.md example %d does not assemble: %s'
                                 % (n, e))
    for named in set(re.findall(r'cases/[\w-]+\.txt', text)):
        assert os.path.exists(os.path.join(outdir, named)), \
            'HOWTO.md names %s, which does not exist' % named
    # named in backticks, as `slow` or as `roundtrip: no`
    missing = sorted(k for k in vp8_asm.HEADER_KEYS
                     if not re.search('`%s[`:]' % k, text))
    assert not missing, 'HOWTO.md never mentions the header key(s) %s' \
        % ' '.join(missing)
    return len(blocks)


def check_unique(outdir, rows):
    """Every '# unique:' claim, against what coverage.txt measured.

    A note saying a file is the only one to reach some path is the kind of
    claim nothing else here checks, and the kind most likely to stop being
    true the next time a case is added. Naming the probe makes it an
    assertion instead of a remark.
    """
    path = os.path.join(outdir, 'coverage.txt')
    claims = [r for r in rows if r[UNIQUE]]
    if not claims:
        return
    assert os.path.exists(path), 'a case claims a probe, but there is no ' \
        'coverage.txt to check it against; run make_coverage.sh'
    reached, measured = {}, set()
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        name, _, tags = line.partition(' ')
        measured.add(name)
        for tag in tags.split():
            reached.setdefault(tag, set()).add(name)
    # A case coverage.txt has never seen would make every claim below look
    # true, so say that rather than trusting it.
    stale = sorted({r[0] for r in rows} - measured)
    assert not stale, 'coverage.txt predates %s; rerun make_coverage.sh' \
        % ' '.join(stale[:3])
    for row in claims:
        for probe in row[UNIQUE].split():
            who = reached.get(probe, set())
            assert who == {row[0]}, \
                '%s says it is the only file reaching %s, but coverage.txt ' \
                'says %s' % (row[0], probe,
                             ' '.join(sorted(who)) if who else 'no file does')


def html_escape(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


# What lives where: the things you run stay at the top, the code they are
# built out of sits in src/. Paths are relative to this directory.
# The two that answer "does my decoder survive this?" come first; the rest
# are for changing the corpus, not for using it.
RUN = [
    ('check.sh', 'Decodes every file and checks the verdict and the pixels, '
                 'through `dwebp` or -- for the animations -- `$ANIM_DUMP` '
                 'and `$WEBPINFO`. The one to run.'),
    ('asan_sweep.sh', 'The same, in 14 output modes under a sanitizer build. '
                      'Point `$ASAN_DWEBP` at one.'),
    ('generate.py', 'Rebuilds `files/` from `cases/`, and writes '
                    '`expected.txt`, this README, `SYNTAX.md`, '
                    '`src/README.md`, and an `index.html` for each directory that needs one.'),
    ('make_hashes.sh', 'Rewrites `hashes.txt`, once the new output is known '
                       'to be right.'),
    ('make_coverage.sh', 'Rebuilds `coverage.txt` in a throwaway worktree.'),
    ('vp8_selftest.py', 'Checks the assemblers themselves, not the corpus: '
                        'only needed if you change them.'),
]

SRC = [
    ('src/vp8l.py', 'VP8L lossless bitstream writer: bit packing, canonical '
                    'Huffman codes, prefix coding, sub-images.'),
    ('src/vp8l_asm.py', 'Assembles a lossless image from a text case. Its '
                        'docstring is the format.'),
    ('src/vp8.py', 'VP8 lossy bitstream writer: the boolean coder, the frame '
                   'header, the mode trees, the coefficients.'),
    ('src/vp8_asm.py', 'Assembles a lossy frame from a text case, in RFC '
                       "6386's field names. Its docstring is the format."),
    ('src/webp_asm.py', 'Wraps either in a RIFF container, in RFC 9649\'s '
                        'field names, and picks which assembler a case '
                        'belongs to.'),
    ('src/vp8_dis.py', 'The other direction for a lossy frame. `--check` '
                       'round trips one against libwebp.'),
    ('src/vp8l_dis.py', 'The other direction for a lossless image, the same '
                        'way.'),
    ('src/webp_dis.py', 'The other direction for a whole file: chunks, '
                        'animation frames and alpha planes, delegating each '
                        'image to one of those two.'),
    ('src/grammar.py', 'Every keyword and the range of every value, as data. '
                       '`SYNTAX.md` is generated from it.'),
    ('src/vp8_tables.py', 'The VP8 constant tables, extracted from libwebp.'),
    ('src/make_vp8_tables.py', 'Extracts them, so they are never retyped.'),
    ('src/lossy_parts.py', 'The multi-partition lossy cases, patched from '
                           '`sources/`.'),
    ('src/make_partition_sources.c', 'Rebuilds `sources/`: cwebp cannot emit '
                                     'more than one token partition.'),
    ('src/probes.py', 'The `fprintf` probes `make_coverage.sh` patches in.'),
    ('src/check_refs.py', 'Checks that the source lines the notes point at '
                          'still say what the notes claim.'),
]

DATA = [
    ('HOWTO.md', 'How to write a case, read a real file back into one, and '
                 'add one here.'),
    ('SYNTAX.md', 'The whole case syntax, generated from `src/grammar.py`.'),
    ('expected.txt', 'Name and expected verdict, one line per file.'),
    ('hashes.txt', "SHA-256 of each decoding file's `-pam` output, so a "
                   'silent change in decoded pixels fails too.'),
    ('coverage.txt', 'Which decoder path each file actually reached.'),
    ('refs.txt', 'What each line a note points at said when it was written.'),
    ('COPYING', 'BSD 3-clause, the same as libwebp.'),
]


def code_table(outdir, entries, title, intro=None, strip=''):
    """A linked table of files, each one asserted to exist. 'strip' is the
    directory the README being written lives in, so its links are relative
    to it rather than to the top."""
    out = ['## %s\n' % title]
    if intro:
        out.append(wrap(intro))
    out += ['| file | what it is |', '| --- | --- |']
    for name, what in entries:
        assert os.path.exists(os.path.join(outdir, name)), name
        link = name[len(strip):] if name.startswith(strip) else name
        out.append('| [`%s`](%s) | %s |' % (link, link, what))
    return '\n'.join(out) + '\n'


def build_code_list(outdir):
    """The three tables that go into the top-level README."""
    return '\n'.join((
        code_table(outdir, RUN, 'What to run',
                   'The first two are the point: they run every file here '
                   'through a decoder and say whether it behaved. The rest '
                   'rebuild the corpus or check the tools that write it.'),
        code_table(outdir, SRC, 'The code',
                   'The layers underneath, in **[`src/`](src)**, which has '
                   'its own README describing how they fit together.'),
        code_table(outdir, DATA, 'The data')))


SYNTAX_HEAD = """# The case syntax

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

The header keys are %(header)s. `expect` is `ok` or `reject`; `slow` marks
the one file that allocates a gigabyte; `roundtrip: no` says the case cannot
be read back by whichever of the three disassemblers owns it; `anim` is
the same verdict from the
animation decoder, for the files a still one refuses on sight; `info` is
webpinfo's, which is a second reader of the container and not always of the
same opinion; `incremental` is the streaming decoder's, and is written down
only where it differs from `expect`, which is the whole reason to write it.
`unique` names probes the case claims to be the only file reaching, and
`generate.py` refuses to build if `coverage.txt` disagrees.

Which assembler owns a case follows from its keywords: a case saying
`lossless` is a VP8L image, anything else a lossy VP8 frame, and container
keywords may be added to either. Two keywords open a block -- `frame` and
`alph_plane` -- and everything after one belongs to it until the next block
or the end of the case; that is the only nesting there is.

"""

VALUE_NOTE = """
A value is written as a plain number -- `12`, `0x0c`, `0b1100` all work --
except where the table says otherwise. `-` in place of a number means the
field is *absent*, which is not the same as zero: it writes the flag alone.
"""


def value_range(v):
    """One value's type and range, for the table."""
    kind = v['kind']
    if kind == 'opt':
        return value_range(v['of']) + ', or `-`'
    if kind in ('uint', 'sint'):
        return '`%d`..`%d`' % (v['min'], v['max'])
    if kind == 'enum':
        return ', '.join('`%s`' % n for n in v['names'])
    return {'hex': 'hex digits', 'token': 'a word'}[kind]


def syntax_rows(keywords, scope):
    out = ['| keyword | values | range | what it is |',
           '| --- | --- | --- | --- |']
    for name in sorted(keywords):
        entry = keywords[name]
        if entry['scope'] != scope:
            continue
        arity = entry['arity']
        kinds = ' / '.join(sorted({value_range(v) for v in entry['values']}))
        out.append('| `%s` | %s | %s | %s |'
                   % (name, 'any' if arity == 'list' else arity,
                      kinds or '--', entry['doc'] or ''))
    return '\n'.join(out) + '\n'


SYNTAX_SCOPES = [
    ('image', 'The lossless image (VP8L)',
     'Written in the order the format carries them: the header, the '
     'transforms, the color cache and entropy image, the Huffman codes, then '
     'the pixel data.'),
    ('frame', 'The lossy frame (VP8)',
     "RFC 6386's field names, section by section, so a case reads against "
     'the specification rather than against the decoder.'),
    ('frame, image', 'Both',
     'The two image formats spell these the same way.'),
    ('macroblock', 'Inside a macroblock',
     '`macroblock` opens one; these fill it in. Macroblocks are laid out in '
     'raster order, and any the frame still needs at the end are added with '
     'default everything.'),
    ('group', 'Inside a group of Huffman codes',
     'A lossless image has one group per entry of its entropy image, and '
     'each group holds five codes: green, red, blue, alpha and dist.'),
    ('container', 'The RIFF container (RFC 9649)',
     'A case that says nothing here gets a plain `RIFF....WEBP` around its '
     'one image chunk. A fourcc is padded to four characters, so `VP8` means '
     "`'VP8 '`. A payload spelled out with `payload` replaces whatever this "
     'would otherwise have built for that chunk.'),
    ('animation', 'Animation (ANIM and ANMF)',
     '`frame` opens a block, and everything after it belongs to that frame '
     'until the next block or the end of the case: its ANMF header fields, '
     'its chunk list, and the image it carries. A file with frames in it '
     'defaults to `VP8X ANIM ANMF...` with the animation flag set and a '
     'canvas the frames fit in, so a case says only what it is changing.'),
]


def build_syntax(outdir):
    """SYNTAX.md, from src/grammar.py rather than from prose."""
    g = grammar.build()
    keys = ', '.join('`%s`' % k for k in sorted(g['header_keys']))
    out = [SYNTAX_HEAD % {'header': keys}, VALUE_NOTE]
    for scope, title, blurb in SYNTAX_SCOPES:
        out.append('## %s\n' % title)
        out.append(wrap(blurb))
        out.append(syntax_rows(g['keywords'], scope))
    out.append('## The forms a Huffman code takes\n')
    out.append(wrap(
        "`code NAME ...` names one of %s, and the word after it picks what "
        'the rest of the line means. A code the case says nothing about '
        'covers whatever the pixel data asks of it.'
        % ', '.join('`%s`' % n for n in g['enums']['code'])))
    out.append('| form | values | range | what it is |')
    out.append('| --- | --- | --- | --- |')
    for name in sorted(g['code_forms']):
        f = g['code_forms'][name]
        kinds = ' / '.join(sorted({value_range(v) for v in f['values']}))
        out.append('| `code NAME %s` | %s | %s | %s |'
                   % (name, 'any' if f['arity'] == 'list' else f['arity'],
                      kinds or '--', f['doc']))
    out.append('\n## The items a pixel list takes\n')
    out.append(wrap(
        '`pixels` lists symbols of the green code, in order. `argb` spells '
        'whole pixels instead, and the two append to the same stream.'))
    out.append('| item | what it is |')
    out.append('| --- | --- |')
    for name in sorted(g['pixel_items']):
        out.append('| `%s` | %s |' % (name, g['pixel_items'][name]['doc']))
    out.append('\n## Constants\n')
    out.append('| name | value |')
    out.append('| --- | --- |')
    for name in sorted(g['constants']):
        out.append('| `%s` | %d |' % (name, g['constants'][name]))
    with open(os.path.join(outdir, 'SYNTAX.md'), 'w') as f:
        f.write(re.sub(r'\n{3,}', '\n\n', '\n'.join(out)) + '\n')


SRC_README = """# webp-torture: the code

Everything the scripts one directory up are built out of. Nothing here is
run directly to produce the corpus -- [`../generate.py`](../generate.py)
does that -- though each assembler and disassembler doubles as a command of
its own, run from the directory above where `cases/` is.
[`../HOWTO.md`](../HOWTO.md) is what to do with them; this is how they fit
together.

Three layers, and a case only ever touches the top one:

* **`webp_asm.py`** reads the case, splits the container directives from the
  image ones, and hands the image to whichever assembler owns it: a case
  saying `lossless` is a VP8L image, anything else a lossy VP8 frame. It
  then wraps the result in RIFF. It also owns the two places the format
  nests: a `frame` block is an animation frame with an image of its own, and
  an `alph_plane` block is a compressed alpha plane, which is a lossless
  image stream with its header left off.
* **`vp8l_asm.py`** and **`vp8_asm.py`** turn the text into the fields of a
  bitstream. Their docstrings are the format: every keyword, its default and
  what it writes. Nothing is validated or clamped -- a value too big for its
  field loses its top bits, which is usually the point.
* **`vp8l.py`** and **`vp8.py`** do the bit-level work: the boolean coder,
  canonical Huffman codes, prefix codes, the sub-image streams. They
  validate nothing either.

`vp8_dis.py`, `vp8l_dis.py` and `webp_dis.py` go the other way, and are
what [`../vp8_selftest.py`](../vp8_selftest.py) uses to check the writers
against real encodes rather than against themselves: disassemble a file,
reassemble from its own text, compare the bytes. `--check` does exactly that
for any file you point it at. The first two read one chunk; `webp_dis.py`
reads a whole file, which is the only way an animation or an alpha plane can
be read at all.

`grammar.py` is the third thing a case touches, though not at assembly time:
it holds every keyword and the range of every value, and
[`../SYNTAX.md`](../SYNTAX.md) is generated from it, so the reference cannot
drift from the code.

%(files)s"""


def write_src_readme(outdir):
    text = SRC_README % {
        'files': code_table(outdir, SRC, 'The files', strip='src/')}
    with open(os.path.join(outdir, 'src', 'README.md'), 'w') as f:
        f.write(text)


def build_cases(outdir):
    """Assembles every cases/*.txt into files/, and returns its row."""
    rows = []
    for path in sorted(glob.glob(os.path.join(outdir, 'cases',
                                              '*.txt'))):
        name = os.path.basename(path)[:-len('.txt')]
        with open(path) as f:
            text = f.read()
        fields = vp8_asm.parse_header(text, path)
        if fields.get('slow') == 'yes':
            SLOW.add(name)
        data = webp_asm.assemble_text(text)
        with open(os.path.join(outdir, 'files', name + '.webp'), 'wb') as f:
            f.write(data)
        rows.append((name, fields['expect'], fields['note'],
                     fields['exercises'], len(data), fields['anim'],
                     fields['info'], fields['incremental'],
                     fields['unique']))
        print('%-40s %-7s %5d bytes' % (name, fields['expect'], len(data)))
    return rows


def build_file_list(index):
    """What the corpus holds, per group. Every file is one click from its own
    note further down, and listed with its size in files/."""
    return '## The bitstreams\n\n' + wrap(
        'The whole set lives in **[`files/`](files)**, which lists each one '
        'with its size and expected verdict. Every note below links straight '
        'to the file it is about.') + '\n' + index


def heading(outdir, row):
    """One file's heading, with a link to the case text when there is one."""
    name = row[0]
    src = os.path.join('cases', name + '.txt')
    out = '### [`%s.webp`](files/%s.webp) -- %s' % (name, name, verdict(row))
    if os.path.exists(os.path.join(outdir, src)):
        out += ' -- from [`%s.txt`](%s)' % (name, src)
    return out + '\n'


def write_readme(outdir, rows):
    used = set()
    kinds = {'lossy': 0, 'container': 0, 'alpha': 0, 'anim': 0, 'vp8l': 0}
    for r in rows:
        for prefix, kind in (('lossy-', 'lossy'), ('container-', 'container'),
                             ('alph-', 'alpha'), ('anim-', 'anim')):
            if r[0].startswith(prefix):
                kinds[kind] += 1
                break
        else:
            kinds['vp8l'] += 1
    index = build_index(rows, GROUPS)
    lines = [README_HEAD % dict(kinds,
                                files=build_file_list(index),
                                code=build_code_list(outdir))]
    for prefix, title, blurb in GROUPS + [(None, 'Other', None)]:
        group = [r for r in rows if r[0] not in used and
                 (prefix is None or r[0].startswith(prefix))]
        if not group:
            continue
        lines.append('## %s\n\n%s' % (title, wrap(blurb)) if blurb
                     else '## %s\n' % title)
        for row in group:
            used.add(row[0])
            lines.append(heading(outdir, row))
            lines.append(wrap(row[2]))
            lines.append(wrap(row[3]))
    total = sum(r[4] for r in rows)
    lines.append('---\n')
    lines.append(wrap(
        '%d files, %d bytes total. Rebuild with `generate.py`: it assembles '
        'everything in `cases/` through `webp_asm.py`, which hands each case '
        'to `vp8l_asm.py` or `vp8_asm.py`, and those to `vp8l.py` and '
        '`vp8.py`.' % (len(rows), total)))
    write_files_index(outdir, rows)
    write_cases_index(outdir, rows)
    write_sources_index(outdir)
    write_src_readme(outdir)
    build_syntax(outdir)
    text = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines))
    with open(os.path.join(outdir, 'README.md'), 'w') as f:
        f.write(text)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    files = os.path.join(outdir, 'files')
    os.makedirs(files, exist_ok=True)
    for stale in os.listdir(files) if os.path.isdir(files) else []:
        os.remove(os.path.join(files, stale))
    rows = lossy_parts.build(outdir) + build_cases(outdir)
    with open(os.path.join(outdir, 'expected.txt'), 'w') as f:
        for row in rows:
            f.write('%s|%s|%s|%s|%s|%s\n'
                    % (row[0], row[1], 'slow' if row[0] in SLOW else '',
                       row[ANIM], row[INFO], row[INCR]))
    write_readme(outdir, rows)
    check_links(outdir)
    check_unique(outdir, rows)
    print('%d HOWTO.md examples assemble' % check_howto(outdir))
    return rows


if __name__ == '__main__':
    main()
