#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.
"""Generates the stress bitstreams. Run: python3 generate.py [outdir]

Every file but the multi-partition ones comes from a text case in cases/,
which carries its own note; this assembles them into files/ and writes
everything that is derived from them -- expected.txt, README.md, SYNTAX.md,
src/README.md and an index per directory. It also refuses to finish if a
link does not resolve, a 'unique:' claim disagrees with coverage.txt, or an
example in HOWTO.md does not assemble.
"""

import glob
import gzip
import io
import os
import re
import sys
import tarfile
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'src'))

import grammar
import lossy_parts
import vp8_asm
import webp_asm

SLOW = set()      # the cases that allocate enough memory to be worth skipping

TARBALL = 'webp-corners.tgz'   # every bitstream, for a reader without git
TARDIR = 'webp-corners'        # the directory it unpacks into

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
     'One image with nothing optional in it, and one with all four '
     'transforms, a colour cache, an entropy image and every kind of pixel '
     'item. Every other lossless file here sets one field of those two to '
     'something an encoder would not.'),
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
     'entropy image. A sub-image is read as an image stream like any other, '
     'minus the transforms and the entropy image it may not have of its '
     'own. So each carries a colour cache and five Huffman codes a file can '
     'say something about. An encoder writes them all the same dull way.'),
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
     'an odd-sized one even on disk. All of it is read before the frame '
     'is.'),
    ('anim-', 'Animation',
     'An ANIM chunk carrying the loop count, then an ANMF per frame with '
     'its own position, duration, disposal and blending, and its own image '
     'chunks. Reading one takes a container walk, a frame composed over the '
     'canvas, and one image decode per frame.'),
    ('alph-', 'The alpha chunk',
     'ALPH carries the alpha plane beside a lossy frame: a header byte of '
     'four two-bit fields, then the plane itself, either stored as it is or '
     'compressed with the lossless coder in its 8-bit mode. Only an alpha '
     'chunk reaches that mode, beside a still frame or inside an animation '
     'frame alike. Each of the four filters turns the same stored bytes '
     'into a different plane, so the pixel hash is what tells them apart. A '
     'compressed plane is a lossless stream with its header left off. The '
     'alph-plane files write one from text, and break each of the four '
     'conditions the 8-bit mode asks for in turn.'),
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
     'sharpness, and the per-reference and per-mode deltas. Sharpness moves '
     'the interior limit in two steps, at 1 and at 5, and the sharpness '
     'files sit either side of both.'),
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
     'A lossy frame may carry 1, 2, 4 or 8 token partitions. Macroblock row '
     'r is read from partition r & (n - 1). No encoder here emits more than '
     'one, so nothing but a written bitstream reaches the rest.'),
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

README_HEAD = """# WebP stress bitstreams

**[`%(tarball_name)s`](%(tarball_name)s)**: a test suite of bitstreams to
exercise decoders with all WebP features.

Each file targets one construct of the format: a container chunk, a header
field, a Huffman code, a back-reference, an animation frame. No encoder emits
these files. They are written field by field from text, so a file can say
what an encoder cannot.

Every file says what a compliant decoder must do with it: decode it to the
pixels in `hashes.txt`, or refuse it without crashing.
[`RUNNING.md`](RUNNING.md) is the contract in full.

**Contents:**

%(toc)s
## What the suite covers

Every layer of the format, in the names its specification uses:
[RFC 6386](https://www.rfc-editor.org/rfc/rfc6386.html) for the lossy
bitstream, [RFC 9649](https://www.rfc-editor.org/rfc/rfc9649.html) for the
container.

%(cover)s
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

The tarball above is the bitstreams alone. To get the scripts with them,
take the directory:

    git clone --depth 1 --filter=blob:none --sparse \\
        https://github.com/skal65535/skal65535.github.io.git
    cd skal65535.github.io
    git sparse-checkout set webp-corners

## What it reaches in libwebp

The suite is graded by measurement rather than by inspection. `coverage.sh`
builds libwebp with instrumentation and reports `src/dec` and `src/demux`
three times over, at %(at)s:

%(coverage)s
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
"""


TOC_MARK = '<!--contents-->'


def wrap(text, indent=''):
    # never at a hyphen: every other name here has one in it, and a file name
    # broken over two lines takes the markdown around it with it
    return '\n'.join(textwrap.wrap(text, 79, initial_indent=indent,
                                   subsequent_indent=indent,
                                   break_on_hyphens=False)) + '\n'




def slug(title):
    """A heading's anchor, as both renderers derive it from the heading text.

    Only while the heading stays alphabetic: past that the two slugify by
    rules they do not share, which is what check_toc() holds them to.
    """
    return title.lower().replace(' ', '-')


def build_toc(text):
    """The page's own sections, as a list of links.

    Read off the rendered page rather than kept in a list beside it, so a
    section that is added, renamed or dropped cannot leave this behind.
    """
    return '\n'.join('* [%s](#%s)' % (h, slug(h))
                     for h in re.findall(r'^## (.+)$', text, re.M)) + '\n'


# What each family of files is for, in the order the README lists them. The
# counts are substituted, so these are wrapped here rather than written out:
# '80' and '148' are not the same width.
COVERAGE = [
    ('container', 'The RIFF container',
     'The VP8X chunk and the canvas it declares. Optional chunks a decoder '
     'steps over by their declared length. Headers that lie about what '
     'follows them.'),
    ('lossy', 'Lossy frames (VP8)',
     'Every field of the frame header. Segmentation, loop filter and '
     'quantizer records. The token coder out to its escape categories. One, '
     'two, four and eight token partitions.'),
    ('vp8l', 'Lossless images (VP8L)',
     'Huffman codes and the code-length code that describes them. Colour '
     'caches, back-references and the four transforms. The entropy image '
     'that changes codes mid-row.'),
    ('alpha', 'Alpha planes',
     'The plane stored a byte per pixel, through each of the four filters. '
     'The plane compressed by the lossless coder, in the 8-bit mode only an '
     'alpha chunk reaches.'),
    ('anim', 'Animation',
     'Frame position, duration, disposal and blending, composed over a '
     'canvas one frame at a time.'),
]

# The three passes coverage.sh reports over libwebp's src/dec and src/demux,
# and the revision it measured. check_coverage() holds the page to these.
COVERAGE_AT = '0be8ddd1'
COVERAGE_RUNS = [
    ('The bitstreams alone, as `check.sh` runs them', 61.20, 68.12, 54.97),
    ('The same files, through every output and scaling option', 68.00, 79.63,
     62.91),
    ('The same files, through every decoding entry point', 81.15, 91.87,
     77.10),
]


# Every knob, and what reads it. check_env() asserts this covers each
# variable the scripts actually take from the environment.
ENV = [
    ('DWEBP', 'The decoder under test -- `check.sh`, `make_hashes.sh`, '
              '`vp8_selftest.py`. Defaults to whatever `dwebp` is on '
              '`$PATH`.'),
    ('ANIM_DUMP', 'The animation decoder, which is the only thing that '
                  'opens the animated files. libwebp does not build it by '
                  'default: `cmake --build . --target anim_dump`.'),
    ('WEBPINFO', 'The container reader `check.sh` holds the animations to '
                 'as a second opinion. Ships with libwebp.'),
    ('CWEBP', 'The encoder, for the half of `vp8_selftest.py` that starts '
              'from real encodes rather than from this corpus.'),
    ('WEBPMUX', 'The muxer, likewise, for real animations.'),
    ('ASAN_DWEBP', 'A sanitizer build, for `asan_sweep.sh`.'),
    ('ASAN_ANIM_DUMP', 'The same for the animations. Looked for beside '
                       '`$ASAN_DWEBP` when unset, which is where the build '
                       'puts it.'),
    ('ASAN_OPTIONS', 'Passed through to those two; `detect_leaks=0` unless '
                     'you say otherwise.'),
    ('LIBWEBP', 'A libwebp git checkout, for `make_coverage.sh`, '
                '`coverage.sh` and `make_vp8_tables.py`.'),
    ('PROFDATA', '`llvm-profdata`, for `coverage.sh`. Taken from `$PATH` or '
                 '`xcrun` when unset.'),
    ('COV', '`llvm-cov`, likewise.'),
    ('SKIP_SLOW', 'Set it to skip the one file that allocates a gigabyte.'),
]


def build_env():
    out = ['| variable | what for |', '| --- | --- |']
    for name, what in ENV:
        out.append('| `$%s` | %s |' % (name, what))
    return '\n'.join(out) + '\n'


def check_env(outdir):
    """Every variable the scripts read from the environment is documented.

    Taking a tool from the environment is how this corpus stays pointed at
    the decoder you meant, so a knob nobody wrote down is a knob nobody
    uses. Two were missing when this was written.
    """
    want = set()
    for path in glob.glob(os.path.join(outdir, '*.sh')) + \
            glob.glob(os.path.join(outdir, '*.py')) + \
            glob.glob(os.path.join(outdir, 'src', '*.py')):
        text = open(path).read()
        want |= set(re.findall(r'\$\{([A-Z_]+):-', text))
        want |= set(re.findall(r"environ\.get\('([A-Z_]+)'", text))
        want |= set(re.findall(r'\bif \[ -n "\$([A-Z_]+)"', text))
    with open(os.path.join(outdir, 'RUNNING.md')) as f:
        running = f.read()
    missing = sorted(v for v in want if '`$%s`' % v not in running)
    assert not missing, 'RUNNING.md documents no $%s' % ', $'.join(missing)
    return len(want)


def build_coverage():
    """What the suite covers, by layer of the format."""
    return '\n'.join(wrap('**%s.** %s' % (title, what), '') + '\n'
                     for _, title, what in COVERAGE)


def build_coverage_table():
    """What the suite reaches in libwebp, as measured."""
    out = ['| driven by | regions | lines | branches |',
           '| --- | ---: | ---: | ---: |']
    for what, regions, lines, branches in COVERAGE_RUNS:
        out.append('| %s | %.0f%% | %.0f%% | %.0f%% |'
                   % (what, regions, lines, branches))
    return '\n'.join(out) + '\n'




INDEX_STYLE = """<!doctype html>
<meta charset="utf-8">
<title>webp-corners</title>
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

FILES_INDEX_HEAD = INDEX_STYLE + """<h1>webp-corners: bitstreams</h1>
%(tarball)s
<p>%(count)d WebP files that exercise corners of the format a normal encoder
never emits, most of them assembled from the text in
<a href="../cases/">cases/</a>. See the <a href="../">notes</a> for what each
one targets. <b>reject</b> means a conforming decoder must refuse the file.</p>
<table>
<tr><th>file</th><th>bytes</th><th>expected</th></tr>
"""

SOURCES_INDEX_HEAD = INDEX_STYLE + """<h1>webp-corners: sources</h1>
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

CASES_INDEX_HEAD = INDEX_STYLE + """<h1>webp-corners: cases</h1>
%(tarball)s
<p>%(count)d text cases, each assembled into the .webp of the same name in
<a href="../files/">files/</a>. A case names the fields the specification
names &mdash; RFC 6386 for the lossy frame, RFC 9649 for the container
&mdash; and carries its own note on what it is for; the
<a href="../">notes</a> have the full write-up. <b>reject</b> means a
conforming decoder must refuse it.</p>
<table>
<tr><th>case</th><th>expected</th><th>what it is</th></tr>
"""


def tarball_line():
    """The download the directory indexes open with. Someone who has landed
    on a listing of 263 files wants all of them, not one."""
    return ('<p>Every bitstream in one file: '
            '<b><a href="../%s">%s</a></b>.</p>' % (TARBALL, TARBALL))


def write_files_index(outdir, rows):
    """files/index.html -- GitHub Pages serves no directory listing."""
    lines = [FILES_INDEX_HEAD % {'count': len(rows),
                                 'tarball': tarball_line()}]
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
    lines[0] = CASES_INDEX_HEAD % {'count': n,
                                   'tarball': tarball_line()}
    with open(os.path.join(outdir, 'cases', 'index.html'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def check_links(outdir):
    """Every relative link in the generated docs resolves.

    GitHub Pages serves no directory listing, so a linked directory needs an
    index of its own; Jekyll renders a README.md into one, which is how this
    page is served at all.
    """
    for doc, base in (('README.md', '.'), ('src/README.md', 'src'),
                      ('SYNTAX.md', '.'), ('HOWTO.md', '.'),
                      ('BITSTREAMS.md', '.'), ('REACHES.md', '.'),
                      ('RUNNING.md', '.')):
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


def check_details(outdir):
    """Every folded group is written the way both renderers need.

    kramdown, which GitHub Pages runs, passes a raw HTML block straight
    through unless it carries markdown="block": without it the table inside
    comes out as literal pipes. GitHub's own renderer drops the attribute and
    needs only the blank lines around the content. Neither one complains, and
    the two are checked in different places, so nothing but this notices.
    """
    n = 0
    for doc in ('README.md', 'BITSTREAMS.md', 'REACHES.md', 'HOWTO.md',
                'RUNNING.md'):
        with open(os.path.join(outdir, doc)) as f:
            text = f.read()
        for block in re.findall(r'<details.*?</details>', text, re.S):
            assert 'markdown="block"' in block, \
                '%s: a <details> without markdown="block"' % doc
            assert '</summary>\n\n' in block, \
                '%s: a <details> with no blank line under its summary' % doc
            assert block.endswith('\n\n</details>'), \
                '%s: a <details> closed with no blank line above it' % doc
            n += 1
    return n


def check_toc(outdir):
    """Each contents list names every section of its page and nothing else.

    Generated from the page, so the two can only differ by an anchor neither
    renderer would produce -- which is a heading carrying punctuation, since
    GitHub and kramdown strip it by different rules.
    """
    n = 0
    for doc in ('README.md', 'RUNNING.md', 'HOWTO.md'):
        with open(os.path.join(outdir, doc)) as f:
            text = f.read()
        heads = re.findall(r'^## (.+)$', text, re.M)
        for title in heads:
            assert re.match(r'^[A-Za-z0-9 ]+$', title), \
                '%s: "%s" needs an anchor the two renderers agree on' \
                % (doc, title)
        assert [(t, slug(t)) for t in heads] \
            == re.findall(r'\[([^]]+)\]\(#([^)]+)\)', text), \
            '%s: the contents list and the sections have drifted' % doc
        n += len(heads)
    return n


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
    ('check.sh', 'Decodes every file and checks the verdict and the pixels. '
                 'Drives `$DWEBP`, and `$ANIM_DUMP` and `$WEBPINFO` for the '
                 'animations. The one to run.'),
    ('asan_sweep.sh', 'The same files in 14 output modes, against a sanitizer '
                      'build named by `$ASAN_DWEBP`.'),
    ('coverage.sh', 'Measures how much of libwebp the suite reaches, from an '
                    'instrumented build in a throwaway worktree. Reports the '
                    'three passes the README tabulates.'),
    ('generate.py', 'Assembles `files/` from `cases/`. Writes every generated '
                    'page and `expected.txt`, and refuses to finish when a '
                    'link, a claim or an example no longer holds.'),
    ('make_hashes.sh', 'Rewrites `hashes.txt`, once the new output is known '
                       'to be right.'),
    ('make_coverage.sh', 'Rewrites `coverage.txt`: which construct each file '
                         'reaches, measured rather than claimed.'),
    ('vp8_selftest.py', 'Checks the assemblers against real encodes. Only '
                        'needed if you change them.'),
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
    ('src/api_sweep.c', 'Every decoding entry point libwebp exports, for '
                        '`coverage.sh`: the incremental decoder fed a few '
                        'bytes at a time, caller-allocated buffers, the '
                        'colorspaces dwebp cannot ask for, the demuxer\'s '
                        'iterators.'),
    ('src/check_refs.py', 'Checks that the source lines the notes point at '
                          'still say what the notes claim.'),
]

DATA = [
    ('expected.txt', 'The verdict per role, one line per file. The contract.'),
    ('hashes.txt', 'The SHA-256 of the decoded pixels, for every file that '
                   'must decode. A silent change in output fails too.'),
    (TARBALL, 'Every bitstream in one file, for taking them without the '
              'repository around them.'),
    ('files', 'The bitstreams themselves, each with its size and verdict.'),
    ('cases', 'The text each one is assembled from.'),
    ('coverage.txt', 'Which construct each file was measured to reach.'),
    ('refs.txt', 'What each line a note cites said when it was written.'),
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


RUNNING_HEAD = """# Running the suite

How to point the suite at a decoder, what it checks, and what a decoder that
is not libwebp has to do to run it. [`README.md`](README.md) says what the
suite is.

**Contents:**

%(toc)s
## The two verdicts

**ok** -- decode it, and produce the pixels in `hashes.txt`.

**reject** -- refuse it, and report an error. No crash, no out-of-bounds
read, no partial image returned as success.

A verdict names no status code. The format says what is malformed. How a
decoder reports it is its own business.

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

%(env)s
%(run)s
%(data)s"""


def build_running(outdir):
    """RUNNING.md: the contract, the scripts, and the environment."""
    return RUNNING_HEAD % dict(
        toc=TOC_MARK,
        env=build_env(),
        run=code_table(outdir, RUN, 'The scripts',
                       'The first two run the files through a decoder. The '
                       'rest rebuild the corpus or check the tools that '
                       'write it.'),
        data=code_table(outdir, DATA, 'The files it reads and writes'))


def write_running(outdir):
    body = build_running(outdir)
    text = re.sub(r'\n{3,}', '\n\n', body.replace(TOC_MARK, build_toc(body)))
    with open(os.path.join(outdir, 'RUNNING.md'), 'w') as f:
        f.write(text)


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
its top bits rather than being refused, which for a stress case is usually
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


SRC_README = """# webp-corners: the code

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




def cell(text):
    return text.replace('|', '\\|').rstrip('.')


def group_table(outdir, group):
    """One group's files: the bytes, the text they came from, the verdict,
    and the one line the case calls itself."""
    out = ['| file | | what it is |', '| --- | --- | --- |']
    for row in group:
        name = row[0]
        link = '[`%s`](files/%s.webp)' % (name, name)
        if os.path.exists(os.path.join(outdir, 'cases', name + '.txt')):
            link += ' [txt](cases/%s.txt)' % name
        out.append('| %s | %s | %s |' % (link, verdict(row), cell(row[2])))
    return '\n'.join(out) + '\n'


def build_feature_index(outdir, rows):
    """Every decoder path the probes measured, and what reaches it.

    The other way round from everything else here, and the question someone
    with a decoder actually has: not "what does this file do" but "which
    file tests this". Generated from coverage.txt, so it is measurement
    rather than a claim.
    """
    path = os.path.join(outdir, 'coverage.txt')
    assert os.path.exists(path), 'no coverage.txt; run make_coverage.sh'
    reached = {}
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        name, _, tags = line.partition(' ')
        for tag in tags.split():
            reached.setdefault(tag, []).append(name)
    out = ['# webp-corners: what reaches what\n']
    out.append(wrap(
        'Every decoder path `src/probes.py` measures, and the files that '
        'reach it. [`BITSTREAMS.md`](BITSTREAMS.md) answers the same '
        'question backwards, a file at a time, and the [notes](README.md) '
        'say what the corpus is. Generated from `coverage.txt`, so this is '
        'the measurement rather than a claim about it.'))
    out.append('\n| decoder path | files |')
    out.append('| --- | --- |')
    for tag in sorted(reached):
        names = sorted(reached[tag])
        shown = ', '.join('`%s`' % n for n in names[:6])
        if len(names) > 6:
            shown += ' +%d' % (len(names) - 6)
        out.append('| `%s` | %s |' % (tag, shown))
    return '\n'.join(out) + '\n'


def write_bitstreams(outdir, rows):
    """Every file, one line each, grouped -- the part that does not fit on a
    front page. The README carries the group counts and links here."""
    used = set()
    out = ['# webp-corners: the bitstreams\n']
    out.append(wrap(
        'One row per file: what the case calls itself, the verdict a decoder '
        'must reach, and a link to both the bytes in `files/` and the text '
        'they were assembled from in `cases/`. The case says the rest -- '
        'which decoder path it reaches and why that is worth a file. '
        '[`REACHES.md`](REACHES.md) indexes the same set the other way '
        'round, and the [notes](README.md) say what all of it is.'))
    out.append(wrap(
        'The groups are folded shut. Click one to open it. All %d at once '
        'are **[`%s`](%s)**.' % (len(rows), TARBALL, TARBALL)))
    for prefix, title, blurb in GROUPS + [(None, 'Other', None)]:
        group = [r for r in rows if r[0] not in used and
                 (prefix is None or r[0].startswith(prefix))]
        if not group:
            continue
        ok = sum(1 for r in group if wanted(r) == 'ok')
        # markdown="block" is what makes the table inside render on GitHub
        # Pages: kramdown otherwise passes the whole element through and the
        # rows come out as literal pipes. GitHub's own renderer ignores the
        # attribute and needs only the blank lines. Both were checked.
        out.append('<details markdown="block">')
        out.append('<summary><b>%s</b> -- %d files, %d ok, %d reject'
                   '</summary>\n' % (title, len(group), ok, len(group) - ok))
        if blurb:
            out.append(wrap(blurb) + '\n')
        used.update(r[0] for r in group)
        out.append(group_table(outdir, group))
        out.append('</details>\n')
    text = re.sub(r'\n{3,}', '\n\n', '\n'.join(out))
    with open(os.path.join(outdir, 'BITSTREAMS.md'), 'w') as f:
        f.write(text)


def write_tarball(outdir, rows):
    """Every bitstream in one file, for a reader who wants the corpus and
    not the repository around it.

    The headers are written rather than taken from the filesystem. Most of
    what tar records per entry is the machine that made it -- the mtimes,
    the uid, the order the directory happened to be read in -- and any of it
    would give a different blob for the same 263 files, which in a tracked
    file means a rewrite on every rebuild.
    """
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode='w') as tar:
        for row in sorted(rows, key=lambda r: r[0]):
            with open(os.path.join(outdir, 'files', row[0] + '.webp'),
                      'rb') as f:
                data = f.read()
            info = tarfile.TarInfo('%s/%s.webp' % (TARDIR, row[0]))
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            tar.addfile(info, io.BytesIO(data))
    path = os.path.join(outdir, TARBALL)
    with open(path, 'wb') as f:
        # mtime=0 for the same reason: gzip stamps one into its own header
        with gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as gz:
            gz.write(tar_bytes.getvalue())
    return os.path.getsize(path)


def check_tarball(outdir, rows):
    """The tarball holds the bitstreams and nothing else, byte for byte.

    It is the one thing here that leaves without the scripts that check it,
    so it is opened again once it is closed.
    """
    with tarfile.open(os.path.join(outdir, TARBALL)) as tar:
        got = {m.name: tar.extractfile(m).read() for m in tar}
    want = {}
    for row in rows:
        with open(os.path.join(outdir, 'files', row[0] + '.webp'), 'rb') as f:
            want['%s/%s.webp' % (TARDIR, row[0])] = f.read()
    assert got == want, '%s and files/ differ: %s' % (
        TARBALL, ' '.join(sorted(set(got) ^ set(want))[:3]) or 'same names, '
        'different bytes')
    # No page says how big it is, because at this size nobody has to care.
    # That holds while it stays a download nobody thinks about first.
    grew = os.path.getsize(os.path.join(outdir, TARBALL))
    assert grew < 1 << 20, '%s is %d bytes: big enough that the pages now ' \
        'owe the reader a size' % (TARBALL, grew)
    return len(got)


def write_readme(outdir, rows):
    write_tarball(outdir, rows)
    write_bitstreams(outdir, rows)
    write_running(outdir)
    with open(os.path.join(outdir, 'REACHES.md'), 'w') as f:
        f.write(re.sub(r'\n{3,}', '\n\n', build_feature_index(outdir, rows)))
    body = README_HEAD % dict(toc=TOC_MARK,
                              tarball_name=TARBALL,
                              at=COVERAGE_AT,
                              coverage=build_coverage_table(),
                              cover=build_coverage())
    # the sections come from the pieces above as much as from the template,
    # so the contents list is written once the page is whole
    lines = [body.replace(TOC_MARK, build_toc(body))]
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
    print('%d HOWTO.md examples assemble, %d environment knobs documented, '
          '%d folded groups render both ways, %d sections in the contents '
          'list, %d files in %s'
          % (check_howto(outdir), check_env(outdir), check_details(outdir),
             check_toc(outdir), check_tarball(outdir, rows), TARBALL))
    return rows


if __name__ == '__main__':
    main()
