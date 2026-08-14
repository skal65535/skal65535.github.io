#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.

"""Regenerates vp8_tables.py from a libwebp checkout.

    LIBWEBP=~/libwebp ./make_vp8_tables.py > vp8_tables.py

Each table is lifted out of the C initializer that defines it, so nothing
is ever retyped and a table that moves upstream shows up as a diff here.
The asserts at the end are the only values read by eye.
"""

import os
import re
import sys
import textwrap

LIBWEBP = os.environ.get('LIBWEBP')


def read(path):
    if not LIBWEBP or not os.path.isdir(os.path.join(LIBWEBP, 'src')):
        sys.exit('set $LIBWEBP to a libwebp checkout')
    with open(os.path.join(LIBWEBP, 'src', path)) as f:
        return f.read()


def strip_comments(s):
    s = re.sub(r'/\*.*?\*/', ' ', s, flags=re.S)
    s = re.sub(r'//[^\n]*', ' ', s)
    return s


def array_body(text, name):
    """The {...} initializer of the array called 'name'."""
    m = re.search(r'\b%s\s*\[' % re.escape(name), text)
    assert m, name
    i = text.index('=', m.end())
    i = text.index('{', i)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    raise AssertionError('unbalanced braces for %s' % name)


def ints(text, name):
    return [int(v) for v in
            re.findall(r'-?\d+', array_body(strip_comments(text), name))]


def reshape(flat, dims):
    for d in reversed(dims[1:]):
        assert len(flat) % d == 0, (len(flat), d)
        flat = [flat[i:i + d] for i in range(0, len(flat), d)]
    assert len(flat) == dims[0], (len(flat), dims)
    return flat


def fmt(v, indent=0, width=76):
    """Nested-list literal, packed."""
    if not isinstance(v, list):
        return str(v)
    if not isinstance(v[0], list):
        line = '[' + ', '.join(str(x) for x in v) + ']'
        return textwrap.fill(line, width=width,
                             initial_indent=' ' * indent,
                             subsequent_indent=' ' * (indent + 1),
                             break_long_words=False,
                             break_on_hyphens=False)[indent:]
    inner = [fmt(x, indent + 1, width) for x in v]
    return '[' + (',\n' + ' ' * (indent + 1)).join(inner) + ']'


def depth(v):
    return 1 + depth(v[0]) if isinstance(v, list) else 0


def emit(out, name, value, note):
    out.append('# %s' % note)
    if depth(value) >= 3:  # too deep to align under the name
        rows = ',\n    '.join(fmt(v, 4) for v in value)
        out.append('%s = [\n    %s,\n]' % (name, rows))
    else:
        out.append('%s = %s' % (name, fmt(value, len(name) + 3)))
    out.append('')


def main():
    dec = read('dec/tree_dec.c')
    vp8 = read('dec/vp8_dec.c')
    bw = read('utils/bit_writer_utils.c')

    out = ['# Copyright 2026 Skal (pascal.massimino@gmail.com). '
           'All Rights Reserved.',
           '#',
           '# Use of this source code is governed by a BSD-style license',
           '# that can be found in the COPYING file in the root of the source',
           '# tree.',
           '',
           '"""VP8 constant tables, extracted from the libwebp source.',
           '',
           'Generated, not retyped: every table below is the exact array of '
           'the same',
           'name in the C code, so a mismatch here can only come from libwebp '
           'itself.',
           '"""',
           '']

    emit(out, 'NORM', ints(bw, 'kNorm'),
         'kNorm[], src/utils/bit_writer_utils.c -- renorm shift per range.')
    emit(out, 'NEW_RANGE', ints(bw, 'kNewRange'),
         'kNewRange[], same file -- ((range + 1) << kNorm[range]) - 1.')
    emit(out, 'BANDS', ints(dec, 'kBands'),
         'kBands[], src/dec/tree_dec.c (== VP8EncBands[]). Last entry is a '
         'sentinel.')

    for n in (3, 4, 5, 6):
        v = ints(vp8, 'kCat%d' % n)
        assert v[-1] == 0, v
        emit(out, 'CAT%d' % n, v[:-1],
             'kCat%d[], src/dec/vp8_dec.c, without the 0 sentinel.' % n)
    out.append('# kCat3456[], indexed by the 2-bit category code.')
    out.append('CAT3456 = [CAT3, CAT4, CAT5, CAT6]')
    out.append('')

    emit(out, 'COEFFS_PROBA0',
         reshape(ints(dec, 'CoeffsProba0'), [4, 8, 3, 11]),
         'CoeffsProba0[NUM_TYPES][NUM_BANDS][NUM_CTX][NUM_PROBAS], '
         'src/dec/tree_dec.c.')
    emit(out, 'COEFFS_UPDATE_PROBA',
         reshape(ints(dec, 'CoeffsUpdateProba'), [4, 8, 3, 11]),
         'CoeffsUpdateProba[][][][], same file: the odds that a proba is '
         'updated.')
    emit(out, 'BMODES_PROBA',
         reshape(ints(dec, 'kBModesProba'), [10, 10, 9]),
         'kBModesProba[NUM_BMODES][NUM_BMODES][NUM_BMODES - 1], same file.')

    text = '\n'.join(out)
    sys.stdout.write(text)

    # -- checks against values read by eye from the C source --
    ns = {}
    exec(text, ns)
    assert ns['NORM'][0] == 7 and ns['NORM'][127] == 0
    assert ns['NEW_RANGE'][0] == 127 and ns['NEW_RANGE'][126] == 253
    assert len(ns['NORM']) == 128 and len(ns['NEW_RANGE']) == 128
    assert ns['BANDS'] == [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0]
    assert ns['CAT3'] == [173, 148, 140]
    assert ns['CAT6'] == [254, 254, 243, 230, 196, 177, 153, 140, 133, 130,
                          129]
    assert ns['COEFFS_PROBA0'][0][0][0] == [128] * 11
    assert ns['COEFFS_PROBA0'][0][1][0][:6] == [253, 136, 254, 255, 228, 219]
    assert ns['COEFFS_PROBA0'][3][7][2] == [238, 1, 255] + [128] * 8
    assert ns['COEFFS_UPDATE_PROBA'][0][1][0][0] == 176
    assert ns['COEFFS_UPDATE_PROBA'][3][7][1][0] == 254
    assert ns['BMODES_PROBA'][0][0] == [231, 120, 48, 89, 115, 113, 120, 152,
                                        112]
    assert ns['BMODES_PROBA'][9][9] == [112, 19, 12, 61, 195, 128, 48, 4, 24]
    print('# checks ok', file=sys.stderr)


if __name__ == '__main__':
    main()
