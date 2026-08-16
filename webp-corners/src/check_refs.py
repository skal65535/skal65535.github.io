#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.

"""Checks the 'vp8l_dec.c:172' references the notes make.

    LIBWEBP=... ./src/check_refs.py            check refs.txt
    LIBWEBP=... ./src/check_refs.py --write    rewrite it

A note naming a line is claiming something about that line, and the claim
rots the moment upstream moves. refs.txt records what each cited line said
when the note was written; this checks it still says it, against the same
checkout make_coverage.sh stamps into coverage.txt. Three of them were
already wrong when this was written -- off by two, one and fifty-four.

Same bargain as hashes.txt: the recorded text is only right because it was
looked at once, so rewrite it only when the new numbers are known good.
"""

import glob
import os
import re
import sys

REFS = 'refs.txt'


def references(here):
    """(case, file, line) for every source reference the text makes."""
    out = []
    for path in sorted(glob.glob(os.path.join(here, 'cases', '*.txt'))) + \
            [os.path.join(here, 'generate.py')]:
        with open(path) as f:
            text = f.read()
        for m in re.finditer(r'\b([a-z0-9_]+\.[ch]):(\d+)', text):
            out.append((os.path.basename(path), m.group(1), int(m.group(2))))
    return out


def cited_line(root, name, line):
    """What that file says there, or None if it does not go that far."""
    found = glob.glob(os.path.join(root, 'src', '*', name))
    if not found:
        return None
    with open(found[0]) as f:
        source = f.read().splitlines()
    return source[line - 1].strip() if line <= len(source) else None


def main(argv):
    root = os.environ.get('LIBWEBP', '')
    if not os.path.isdir(os.path.join(root, 'src')):
        print('set $LIBWEBP to a libwebp checkout', file=sys.stderr)
        return 2
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, REFS)
    rows = [(where, '%s:%d' % (name, line), cited_line(root, name, line))
            for where, name, line in references(here)]

    if '--write' in argv:
        with open(path, 'w') as f:
            f.write('# What each line a note points at said when it was\n'
                    '# written. Checked by src/check_refs.py, rewritten by\n'
                    '# make_coverage.sh against the revision coverage.txt\n'
                    '# stamps.\n')
            for where, at, text in rows:
                f.write('%s|%s|%s\n' % (where, at, text if text else ''))
        print('wrote %s (%d references)' % (REFS, len(rows)))
        return 0

    if not os.path.exists(path):
        print('no %s; run with --write' % REFS, file=sys.stderr)
        return 2
    want = {}
    with open(path) as f:
        for line in f:
            if not line.startswith('#'):
                where, at, text = line.rstrip('\n').split('|', 2)
                want[(where, at)] = text
    bad = 0
    for gone in sorted(set(want) - {(w, a) for w, a, _ in rows}):
        print('%s: %s is no longer referenced; rerun with --write'
              % gone, file=sys.stderr)
        bad += 1
    for where, at, text in rows:
        was = want.get((where, at))
        if was is None:
            print('%s: %s is new; rerun with --write' % (where, at),
                  file=sys.stderr)
        elif text is None:
            print('%s: %s no longer exists' % (where, at), file=sys.stderr)
        elif text != was:
            print('%s: %s now says\n    %s\n  but the note was written when '
                  'it said\n    %s' % (where, at, text, was), file=sys.stderr)
        else:
            continue
        bad += 1
    print('%d source references checked, %d moved' % (len(rows), bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
