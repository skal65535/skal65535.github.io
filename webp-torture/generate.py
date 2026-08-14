#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.
"""Generates the torture bitstreams. Run: python3 generate.py [outdir]

Every file but the multi-partition ones comes from a text case in cases/,
which carries its own note; this collects them into files/, expected.txt and
README.md. 'expect' is what the reference decoder is supposed to do with a
case: 'ok' or 'reject'.
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


GROUPS = [
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
    ('alph-', 'The alpha chunk',
     'ALPH carries the alpha plane beside a lossy frame: a header byte of '
     'four two-bit fields, then the plane itself, either stored as it is or '
     'compressed with the lossless coder in its 8-bit mode. That mode is a '
     'separate path through vp8l_dec.c from the one every VP8L file here '
     'takes, and these are the only files that reach it.'),
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
     'sharpness, and the per-reference and per-mode deltas.'),
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
     'end-of-block, zero runs, and the four coefficient types.'),
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
     'size table that follows partition 0. They carry 256 macroblocks of '
     'real coefficients, which the assembled cases do not.'),
]

README_HEAD = """# WebP torture bitstreams

Small WebP files that exercise corners of the format a normal encoder never
emits, one layer of it at a time:

* **%(vp8l)d lossless (VP8L) streams**, assembled by
  [`vp8l_asm.py`](src/vp8l_asm.py) from a text description of the
  bitstream: the header, the transforms, the Huffman codes down to the
  code-length stream that describes them, and the pixel data.
* **%(lossy)d lossy VP8 frames**. All but seven are assembled the same way by
  [`vp8_asm.py`](src/vp8_asm.py), under the field names
  [RFC 6386](https://www.rfc-editor.org/rfc/rfc6386.html) gives them. The
  other seven start from an encoder-API call
  ([`make_partition_sources.c`](src/make_partition_sources.c)) and are patched
  by [`lossy_parts.py`](src/lossy_parts.py).
* **%(container)d RIFF containers**, wrapped by
  [`webp_asm.py`](src/webp_asm.py) in
  [RFC 9649](https://www.rfc-editor.org/rfc/rfc9649.html)'s names: the
  extended-format VP8X chunk, the optional chunks a decoder must step over,
  and sizes that lie about what is behind them.
* **%(alpha)d alpha chunks**, where the plane is either stored one byte per
  pixel or compressed with the lossless coder in its 8-bit mode -- a
  different path through the decoder from the one every VP8L file here
  takes.

A case is a text file and the notes below are its own: each one carries what
it is, what the decoder should do with it, and which path that answer comes
from.

Each entry says what the reference decoder is expected to do:

* **ok** -- must decode, and must keep decoding to the same pixels. Several
  are not something cwebp can produce, so nothing else pins the behaviour.
* **reject** -- must fail cleanly and report a status, with no crash and no
  out-of-bounds access. Which status varies: a malformed Huffman code gives
  BITSTREAM_ERROR, a short partition table gives NOT_ENOUGH_DATA.

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
Then, from `webp-torture/`:

    ./check.sh              # verdict + decoded-pixel hash for every file
    ./asan_sweep.sh         # 14 decode modes, under a sanitizer build
    ./vp8_selftest.py       # checks the lossy writer against libwebp
    ./make_coverage.sh      # regenerate coverage.txt
    ./make_hashes.sh        # regenerate hashes.txt, once the output is right
    python3 generate.py     # rebuild files/, expected.txt and this README

A case is a text file, one field per line under the name the specification
gives it -- RFC 6386 for the lossy frame, RFC 9649 for the container -- so a
case reads against the format rather than against the decoder that happens
to be under test. Every field has a default, so a case says only what it is
about, and nothing is validated or clamped: a value too big for its field
loses its top bits, which is usually the point.

`src/webp_asm.py` assembles any of them. It hands the image part to
`src/vp8l_asm.py` if the case says `lossless` and to `src/vp8_asm.py` if it
does not; use any of the three directly, or read an existing lossy frame
back out as text to start from:

    ./src/webp_asm.py cases/alph-raw-filter-gradient.txt /tmp/out.webp
    ./src/vp8_asm.py cases/lossy-coeff-cat6.txt /tmp/out.webp
    ./src/vp8l_asm.py cases/codelen-depth-15.txt /tmp/out.webp
    ./src/vp8_dis.py some-photo.webp

[`SYNTAX.md`](SYNTAX.md) is the whole vocabulary in one place, generated
from [`src/grammar.py`](src/grammar.py); `./src/grammar.py` prints the same
thing as JSON, which is what a generator should read rather than the prose.
Each tool's docstring is the reference for the fields it owns:
`vp8l_asm.py` for the lossless image, `vp8_asm.py` for the lossy frame, and
`webp_asm.py` for the container and the alpha chunk. [`src/`](src) has a
README of its own saying how the three layers fit together.

`files/` is pure output and is wiped on every rebuild. The four lossy encodes
the multi-partition cases are patched from live in `sources/` --
[1](sources/lossy-1-partitions.webp), [2](sources/lossy-2-partitions.webp),
[4](sources/lossy-4-partitions.webp), [8](sources/lossy-8-partitions.webp)
partitions -- and are themselves rebuilt by `make_partition_sources.c`.

`check.sh`, `make_hashes.sh` and `vp8_selftest.py` honour `$DWEBP` and
`asan_sweep.sh` honours `$ASAN_DWEBP`, so all of them can be pointed at any
build, or at another decoder implementation; they fall back to whatever
`dwebp` is on `$PATH`. `make_coverage.sh` and `make_vp8_tables.py` need
`$LIBWEBP` set to a libwebp git checkout. `SKIP_SLOW=1` skips the one file
that allocates a gigabyte.

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

The lossy writer is checked a second way, against libwebp rather than against
itself: `vp8_dis.py` reads a frame back into the same text `vp8_asm.py`
assembles, so a real cwebp encode can be disassembled, reassembled and
compared byte for byte. Every file in `sources/` survives that, as do encodes
from 1x1 to 128x128 across the whole quality range -- 596 macroblocks, all
four 16x16 modes, all ten 4x4 modes and coefficients in every escape
category. `vp8_selftest.py` runs it, along with every coefficient magnitude
up to the format's largest and a handful of frames that say the same thing
two different ways and must decode alike.

What the corpus reaches is measured rather than assumed, and the measurement
is what says where to add files next. As it stands: every field of the lossy
frame header is written at both ends of its range; all 93 reachable
(coefficient type, band, context) probability cells are read, which is the
whole grid bar the three that no bitstream can select; all 28 pairs of
optional tools appear together in some frame; and two probes out of 95 are
unreached, the magic-byte and version tests inside `ReadImageInfo()`, both of
which `VP8LCheckSignature()` has already made by the time they run.

## What is not covered

Animation. There is no real ANIM or ANMF chunk, only the VP8X flag that
claims one, and nothing here goes near the demux API that would be needed to
walk a sequence of frames. No inter frames either, which libwebp refuses
outright, so there is little to pin beyond the one file that checks it does.

The two compressed alpha planes are cwebp output pasted in, so that path has
two points in it rather than a swept range: one that reaches the lossless
decoder's 8-bit loop and one that misses it. `vp8l_asm.py` could write them
-- an alpha plane is a VP8L image whose green channel carries the values --
and then they could be malformed like every other lossless case here.

Within a lossy key frame, what is left is what the decoder does not read:
the profile selects no reconstruction filter, and the entropy-refresh bit is
parsed and dropped. Both are written anyway, so a decoder that started
acting on either would fail a pixel hash rather than pass unnoticed.

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
        ok = sum(1 for r in group if r[1] == 'ok')
        out.append('| %s | %d | %d | %d |' %
                   (title, len(group), ok, len(group) - ok))
    ok = sum(1 for r in rows if r[1] == 'ok')
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
    for name, expect, _note, _exercises, size in sorted(rows):
        lines.append('<tr><td><a href="%s.webp"><code>%s.webp</code></a></td>'
                     '<td class="n">%d</td><td class="%s">%s</td></tr>'
                     % (name, name, size,
                        'reject' if expect == 'reject' else 'ok', expect))
    lines.append('</table>')
    with open(os.path.join(outdir, 'files', 'index.html'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_cases_index(outdir, rows):
    """cases/index.html -- same reason, and the README links cases/."""
    lines = [CASES_INDEX_HEAD]
    n = 0
    for name, expect, note, _exercises, _size in sorted(rows):
        if not os.path.exists(os.path.join(outdir, 'cases',
                                           name + '.txt')):
            continue                    # patched from sources/, not assembled
        n += 1
        lines.append('<tr><td><a href="%s.txt"><code>%s</code></a></td>'
                     '<td class="%s">%s</td><td>%s</td></tr>'
                     % (name, name, 'reject' if expect == 'reject' else 'ok',
                        expect, html_escape(note)))
    lines.append('</table>')
    lines[0] = CASES_INDEX_HEAD % {'count': n}
    with open(os.path.join(outdir, 'cases', 'index.html'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def html_escape(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


# What lives where: the things you run stay at the top, the code they are
# built out of sits in src/. Paths are relative to this directory.
RUN = [
    ('generate.py', 'Assembles every case in `cases/` and produces '
                    '`expected.txt`, this README, `src/README.md` and the two '
                    '`index.html` listings.'),
    ('check.sh', 'Decodes every file; checks the verdict and the pixels.'),
    ('make_hashes.sh', 'Rewrites `hashes.txt` when the new output is known '
                       'to be right.'),
    ('make_coverage.sh', 'Rebuilds `coverage.txt` in a throwaway worktree.'),
    ('asan_sweep.sh', 'Decodes every file in 14 modes, under a sanitizer '
                      'build.'),
    ('vp8_selftest.py', 'Round trips real encodes through the lossy writer '
                        'and reader, and checks what cwebp cannot emit '
                        'against dwebp.'),
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
    ('src/vp8_dis.py', 'The other direction: a lossy .webp back into that '
                       'text. `--check` round trips one against libwebp.'),
    ('src/grammar.py', 'Every keyword and the range of every value, as data. '
                       '`SYNTAX.md` is generated from it.'),
    ('src/vp8_tables.py', 'The VP8 constant tables, extracted from libwebp.'),
    ('src/make_vp8_tables.py', 'Extracts them, so they are never retyped.'),
    ('src/lossy_parts.py', 'The multi-partition lossy cases, patched from '
                           '`sources/`.'),
    ('src/make_partition_sources.c', 'Rebuilds `sources/`: cwebp cannot emit '
                                     'more than one token partition.'),
    ('src/probes.py', 'The `fprintf` probes `make_coverage.sh` patches in.'),
]

DATA = [
    ('SYNTAX.md', 'The whole case syntax, generated from `src/grammar.py`.'),
    ('expected.txt', 'Name and expected verdict, one line per file.'),
    ('hashes.txt', "SHA-256 of each decoding file's `-pam` output."),
    ('coverage.txt', 'Which decoder path each file actually reached.'),
    ('COPYING', 'BSD 3-clause, the same as libwebp.'),
]


def code_table(outdir, entries, title, intro=None):
    """A linked table of files, each one asserted to exist."""
    out = ['## %s\n' % title]
    if intro:
        out.append(wrap(intro))
    out += ['| file | what it is |', '| --- | --- |']
    for name, what in entries:
        assert os.path.exists(os.path.join(outdir, name)), name
        out.append('| [`%s`](%s) | %s |' % (name, name, what))
    return '\n'.join(out) + '\n'


def build_code_list(outdir):
    """The three tables that go into the top-level README."""
    return '\n'.join((
        code_table(outdir, RUN, 'What to run'),
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
be read back by `src/vp8_dis.py`.

Which assembler owns a case follows from its keywords: a case saying
`lossless` is a VP8L image, anything else a lossy VP8 frame, and container
keywords may be added to either.

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
     "`'VP8 '`."),
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

%(files)s"""


def write_src_readme(outdir):
    table = code_table(outdir, SRC, 'The files')
    text = SRC_README % {'files': table.replace('src/', '')}
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
                     fields['exercises'], len(data)))
        print('%-40s %-7s %5d bytes' % (name, fields['expect'], len(data)))
    return rows


def build_file_list(rows, groups, index):
    """A linked list of every bitstream, grouped, for the top of the README."""
    out = ['## The bitstreams\n',
           'Every name below links straight to the file. The whole set lives',
           'in **[`files/`](files/)**, which lists each one with its size and',
           'expected verdict; the notes further down say what each targets.\n',
           index]
    seen = set()
    for prefix, title, _ in groups:
        group = [r for r in rows if r[0].startswith(prefix) and r[0] not in seen]
        seen.update(r[0] for r in group)
        if not group:
            continue
        # pack the links by hand: textwrap would break them mid-URL
        line, block = '**%s** —' % title, []
        for r in group:
            link = ' [%s](files/%s.webp)' % (r[0], r[0])
            if len(line) + len(link) > 78 and line.strip():
                block.append(line)
                line = ' ' + link.lstrip()
            else:
                line += link
            line += ' ·'
        block.append(line.rstrip(' ·'))
        out.append('\n'.join(block) + '\n')
    return '\n'.join(out)


def heading(outdir, name, expect):
    """One file's heading, with a link to the case text when there is one."""
    src = os.path.join('cases', name + '.txt')
    out = '### [`%s.webp`](files/%s.webp) -- %s' % (name, name, expect)
    if os.path.exists(os.path.join(outdir, src)):
        out += ' -- from [`%s.txt`](%s)' % (name, src)
    return out + '\n'


def write_readme(outdir, rows):
    used = set()
    kinds = {'lossy': 0, 'container': 0, 'alpha': 0, 'vp8l': 0}
    for r in rows:
        for prefix, kind in (('lossy-', 'lossy'), ('container-', 'container'),
                             ('alph-', 'alpha')):
            if r[0].startswith(prefix):
                kinds[kind] += 1
                break
        else:
            kinds['vp8l'] += 1
    index = build_index(rows, GROUPS)
    lines = [README_HEAD % dict(kinds,
                                files=build_file_list(rows, GROUPS, index),
                                code=build_code_list(outdir))]
    for prefix, title, blurb in GROUPS + [(None, 'Other', None)]:
        group = [r for r in rows if r[0] not in used and
                 (prefix is None or r[0].startswith(prefix))]
        if not group:
            continue
        lines.append('## %s\n\n%s' % (title, wrap(blurb)) if blurb
                     else '## %s\n' % title)
        for name, expect, note, exercises, size in group:
            used.add(name)
            lines.append(heading(outdir, name, expect))
            lines.append(wrap(note))
            lines.append(wrap(exercises))
    total = sum(r[4] for r in rows)
    lines.append('---\n')
    lines.append(wrap(
        '%d files, %d bytes total. Rebuild with `generate.py`: it assembles '
        'everything in `cases/` through `webp_asm.py`, which hands each case '
        'to `vp8l_asm.py` or `vp8_asm.py`, and those to `vp8l.py` and '
        '`vp8.py`.' % (len(rows), total)))
    write_files_index(outdir, rows)
    write_cases_index(outdir, rows)
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
        for name, expect, _, _, _ in rows:
            f.write('%s|%s|%s\n' % (name, expect,
                                     'slow' if name in SLOW else ''))
    write_readme(outdir, rows)
    return rows


if __name__ == '__main__':
    main()
