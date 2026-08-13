#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.

"""Checks the lossy writer against libwebp, four ways.

    ./vp8_selftest.py [extra-real-encodes.webp...]

  round trip   every .webp in sources/ (and any named on the command line)
               is disassembled, reassembled from its own text, and compared
               byte for byte. Anything the two disagree on shows up within a
               byte or two, so this pins the whole syntax against files
               libwebp itself produced.
  cases        the same, starting from the text in cases/ instead. The
               ones meant to be refused often cannot be read back at
               all, which is the point of them; those are counted.
  levels       every coefficient magnitude from 1 to 2114, the largest the
               category-6 escape can hold, written and read back. Real
               encodes stop well short of that.
  pixels       pairs of frames that say the same thing two different ways
               must decode to the same image. This is what checks the parts
               cwebp never emits -- skip flags, loop-filter deltas, one
               segment pretending to be four -- against the decoder rather
               than against this file. Needs $DWEBP.
"""

import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

import vp8
import vp8_asm
import vp8_dis

DWEBP = os.environ.get('DWEBP', 'dwebp')


def round_trip(paths):
    """Real encodes: bytes -> text -> bytes must not move."""
    bad = done = 0
    for path in paths:
        with open(path, 'rb') as fp:
            data = fp.read()
        if b'VP8 ' not in data[:20]:
            continue                   # lossless, not ours
        done += 1
        bad += bool(vp8_dis.check(path, data))
    print('sources  %d files' % done)
    return bad


def case_round_trip():
    """Cases: text -> bytes -> text -> bytes must not move either.

    A case whose whole point is to be refused is often unreadable by
    anything, this disassembler included. Those are counted rather than
    failed, as are the ones that say '# roundtrip: no' because what they
    pin is invisible to a reader. Everything else has to survive.
    """
    bad = broken = skipped = 0
    paths = sorted(glob.glob('cases/*.txt'))
    for path in paths:
        with open(path) as fp:
            text = fp.read()
        head = vp8_asm.parse_header(text, path)
        if head['roundtrip'] == 'no':   # the case says it cannot be read back
            skipped += 1
            continue
        want = vp8_dis.vp8_chunk(vp8_asm.assemble_text(text))
        try:
            f, sizes = vp8_dis.parse(want)
            got = vp8_dis.vp8_chunk(
                vp8_asm.assemble_text(vp8_dis.dump(f, sizes)))
        except vp8_dis.Truncated:
            got = None
        if got == want:
            continue
        if head['expect'] == 'reject':
            broken += 1
            continue
        print('%s: reassembles to %s bytes, not %d'
              % (path, len(got) if got else 'unreadable', len(want)),
              file=sys.stderr)
        bad += 1
    print('cases    %d files, %d refused by the reader, %d unreadable by '
          'design' % (len(paths), broken, skipped))
    return bad


def levels():
    """Every magnitude a coefficient can hold, both signs, read back."""
    bad = 0
    for magnitude in range(1, 2115):
        for at, sign in ((0, 1), (15, -1), (7, -1)):
            want = [0] * 16
            want[at] = sign * magnitude
            f = vp8.Frame(16, 16)
            f.mbs = [vp8.MB()]
            f.mbs[0].ymode = vp8.B_PRED   # so luma position 0 is codable
            f.mbs[0].y[0] = list(want)
            got, _ = vp8_dis.parse(vp8.assemble(f), raw=False)
            if got.mbs[0].y[0] != want:
                print('level %d at %d: read back %s'
                      % (sign * magnitude, at, got.mbs[0].y[0]),
                      file=sys.stderr)
                bad += 1
    print('levels   1..2114, both signs, three positions')
    return bad


# Pairs of cases that must decode to the same pixels. The first says it the
# plain way, the second the way no encoder would.
EQUIVALENT = [
    ('a skipped macroblock and an empty one', """
        width 32
        height 32
        yac_qi 30
        macroblock
        intra_y_mode V_PRED
        intra_chroma_mode H_PRED
        coeffs y2 9
        macroblock count 3
     """, """
        width 32
        height 32
        yac_qi 30
        mb_no_skip_coeff 1
        prob_skip_false 200
        macroblock
        intra_y_mode V_PRED
        intra_chroma_mode H_PRED
        coeffs y2 9
        macroblock count 3
        mb_skip_coeff 1
     """),
    ('loop-filter deltas that update nothing', """
        width 32
        height 32
        loop_filter_level 20
        sharpness_level 3
        yac_qi 40
        macroblock
        intra_y_mode TM_PRED
        intra_chroma_mode TM_PRED
        coeffs y[5] 1:3 -2
     """, """
        width 32
        height 32
        loop_filter_level 20
        sharpness_level 3
        loop_filter_adj_enable 1
        mode_ref_lf_delta_update 1
        ref_frame_delta - - - -
        mb_mode_delta - - - -
        yac_qi 40
        macroblock
        intra_y_mode TM_PRED
        intra_chroma_mode TM_PRED
        coeffs y[5] 1:3 -2
     """),
    ('four segments that all quantize alike', """
        width 32
        height 32
        yac_qi 44
        macroblock
        coeffs y2 -20
        coeffs y[3] 1:2
     """, """
        width 32
        height 32
        segmentation_enabled 1
        update_mb_segmentation_map 1
        update_segment_feature_data 1
        segment_feature_mode 1
        quantizer_update_value 44 44 44 44
        lf_update_value 0 0 0 0
        segment_prob 128 128 128
        yac_qi 44
        macroblock
        coeffs y2 -20
        coeffs y[3] 1:2
     """),
    ('a probability update that changes nothing', """
        width 16
        height 16
        yac_qi 20
        macroblock
        intra_y_mode B_PRED
        coeffs y[0] 4 3 2 1
     """, """
        width 16
        height 16
        yac_qi 20
        coeff_prob 3 0 0 0 202
        macroblock
        intra_y_mode B_PRED
        coeffs y[0] 4 3 2 1
     """),
    ('the entropy-refresh bit libwebp ignores', """
        width 16
        height 16
        yac_qi 12
        macroblock
        intra_y_mode H_PRED
        intra_chroma_mode V_PRED
        coeffs u[1] 5
     """, """
        width 16
        height 16
        refresh_entropy_probs 1
        yac_qi 12
        macroblock
        intra_y_mode H_PRED
        intra_chroma_mode V_PRED
        coeffs u[1] 5
     """),
    ('the same rows split over four partitions', """
        width 64
        height 64
        yac_qi 25
        macroblock count 16
        coeffs y2 6 -2
     """, """
        width 64
        height 64
        log2_nbr_of_DCT_partitions 2
        yac_qi 25
        macroblock count 16
        coeffs y2 6 -2
     """),
]


def decode(data, tmp):
    """The decoded pixels, or None if dwebp refused the file."""
    path = os.path.join(tmp, 'case.webp')
    with open(path, 'wb') as fp:
        fp.write(data)
    out = os.path.join(tmp, 'case.pam')
    rc = subprocess.run([DWEBP, '-quiet', path, '-pam', '-o', out],
                        capture_output=True).returncode
    if rc != 0:
        return None
    with open(out, 'rb') as fp:
        return hashlib.sha256(fp.read()).hexdigest()


def pixels():
    """Frames that mean the same thing must decode to the same image."""
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for what, plain, roundabout in EQUIVALENT:
            a, b = (decode(vp8_asm.assemble_text(text), tmp)
                    for text in (plain, roundabout))
            if a is None or b is None:
                which = 'the plain one' if a is None else 'the other'
                print('%s: dwebp rejected %s' % (what, which),
                      file=sys.stderr)
                bad += 1
            elif a != b:
                print('%s: decodes differently (%s vs %s)'
                      % (what, a[:12], b[:12]), file=sys.stderr)
                bad += 1
    print('pixels   %d equivalences' % len(EQUIVALENT))
    return bad


def main(argv):
    os.chdir(os.path.dirname(os.path.abspath(argv[0])))
    bad = 0
    bad += round_trip(sorted(glob.glob('sources/*.webp')) + argv[1:])
    bad += case_round_trip()
    bad += levels()
    if shutil.which(DWEBP):
        bad += pixels()
    else:
        print('pixels   skipped, set $DWEBP to a dwebp binary')
    print('FAILED, %d checks did not pass' % bad if bad else 'all checks pass')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
