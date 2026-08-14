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
  encodes      a spread of images encoded both ways at every setting that
               changes what cwebp writes, each disassembled and reassembled.
               The only check starting from what libwebp wrote. Needs $CWEBP.
  cases        the same, starting from the text in cases/ instead, lossy
               then lossless. The
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
import struct
import os
import shutil
import subprocess
import sys
import tempfile
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'src'))

import vp8
import vp8_asm
import vp8_dis
import vp8l
import vp8l_asm
import vp8l_dis

DWEBP = os.environ.get('DWEBP', 'dwebp')


def image_chunk(data):
    """Which image chunk a file carries: b'VP8 ', b'VP8L', or None.

    Not a guess at the first bytes -- a lossy file with alpha puts VP8X and
    ALPH in front of its frame.
    """
    if data[:4] != b'RIFF':
        return b'VP8L' if data[:1] == bytes([vp8l.MAGIC]) else b'VP8 '
    at = 12
    while at + 8 <= len(data):
        tag = data[at:at + 4]
        if tag in (b'VP8 ', b'VP8L'):
            return tag
        size = int.from_bytes(data[at + 4:at + 8], 'little')
        at += 8 + size + (size & 1)
    return None


def round_trip(paths):
    """Real encodes: bytes -> text -> bytes must not move.

    Either kind: a lossy frame goes through vp8_dis.py, a lossless image
    through vp8l_dis.py, and both have to come back byte for byte.
    """
    bad = lossy = lossless = 0
    for path in paths:
        with open(path, 'rb') as fp:
            data = fp.read()
        tag = image_chunk(data)
        if tag == b'VP8 ':
            lossy += 1
            bad += bool(vp8_dis.check(path, data))
        elif tag == b'VP8L':
            lossless += 1
            bad += bool(vp8l_dis.check(path, data))
    print('sources  %d lossy, %d lossless' % (lossy, lossless))
    return bad


def png(path, w, h, pixels):
    """A minimal RGBA PNG, so the encoder has something to chew on."""
    raw = b''.join(b'\x00' + bytes(pixels[y * w * 4:(y + 1) * w * 4])
                   for y in range(h))

    def chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data +
                struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))

    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' +
                chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)) +
                chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def sample_images(where):
    """A dozen images an encoder treats differently: sizes down to 1x1, flat,
    gradient, noise, palettised, striped, and one with real alpha."""
    def flat(w, h, rgba):
        return list(rgba) * (w * h)

    out = []
    for w, h in ((1, 1), (1, 17), (17, 1), (32, 32), (64, 48), (129, 97),
                 (256, 256)):
        out.append(('grad-%dx%d' % (w, h), w, h,
                    [c for y in range(h) for x in range(w)
                     for c in ((x * 7) % 256, (y * 5) % 256, (x ^ y) % 256,
                               255)]))
    w = h = 64
    seed, noise = 12345, []
    for _ in range(w * h):
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        noise += [(seed >> 7) & 0xff, (seed >> 15) & 0xff,
                  (seed >> 3) & 0xff, 255]
    out.append(('noise', w, h, noise))
    out.append(('flat', w, h, flat(w, h, (30, 90, 200, 255))))
    out.append(('alpha', w, h,
                [c for y in range(h) for x in range(w)
                 for c in (x * 4 % 256, y * 4 % 256, 128, (x * y) % 256)]))
    out.append(('palette', w, h,
                [c for y in range(h) for x in range(w)
                 for i in [(x // 7 + y // 5) % 8]
                 for c in (i * 31 % 256, i * 57 % 256, i * 13 % 256, 255)]))
    out.append(('stripes', w, h,
                [c for y in range(h) for x in range(w)
                 for c in ((255, 0, 0, 255) if (x // 3) % 2
                           else (0, 0, 255, 255))]))
    paths = []
    for name, iw, ih, pixels in out:
        png(os.path.join(where, name + '.png'), iw, ih, pixels)
        paths.append(os.path.join(where, name + '.png'))
    return paths


# settings that change what cwebp writes, either way
LOSSLESS = (['-lossless'], ['-z', '0'], ['-z', '3'], ['-z', '9'],
            ['-lossless', '-m', '6', '-q', '100'],
            ['-lossless', '-near_lossless', '60'], ['-lossless', '-exact'])
LOSSY = (['-q', '0'], ['-q', '50'], ['-q', '95'], ['-q', '100'],
         ['-q', '75', '-m', '6'], ['-q', '75', '-segments', '1'],
         ['-q', '75', '-f', '0'])
SETTINGS = LOSSLESS + LOSSY


def real_encodes():
    """Encode a spread of images every way cwebp will, and round trip each.

    The only check on the assemblers that starts from something libwebp
    wrote rather than from this corpus. Needs $CWEBP.
    """
    cwebp = os.environ.get('CWEBP', 'cwebp')
    if shutil.which(cwebp) is None:
        print('encodes  skipped, no cwebp')
        return 0
    bad = done = lossy = 0
    tmp = tempfile.mkdtemp()
    try:
        for src in sample_images(tmp):
            for n, opts in enumerate(SETTINGS):
                out = '%s.%d.webp' % (src[:-4], n)
                if subprocess.call([cwebp, '-quiet'] + opts + [src, '-o', out],
                                   stderr=subprocess.DEVNULL):
                    continue
                done += 1
                with open(out, 'rb') as f:
                    data = f.read()
                if image_chunk(data) != b'VP8L':
                    lossy += 1
                    bad += bool(vp8_dis.check(out, data))
                    continue
                want = vp8l_dis.vp8l_chunk(data)
                try:
                    got = vp8l_dis.vp8l_chunk(
                        vp8l_asm.assemble_text(vp8l_dis.dump(want)))
                except (vp8l_dis.Truncated, vp8_asm.AsmError) as e:
                    print('%s: %s' % (os.path.basename(out), e),
                          file=sys.stderr)
                    bad += 1
                    continue
                if got != want:
                    print('%s: reassembles to %d bytes, not %d'
                          % (os.path.basename(out), len(got), len(want)),
                          file=sys.stderr)
                    bad += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('encodes  %d from cwebp, %d lossy and %d lossless, over %d images'
          % (done, lossy, done - lossy, len(SETTINGS) and
             done // len(SETTINGS)))
    return bad


def lossless_round_trip():
    """The lossless cases, the same way round.

    A case meant to be refused often cannot be read back at all -- that is
    what it is for -- so those are counted rather than failed. Left out are
    the one file that decodes to a gigabyte, and the cases marked
    '# roundtrip: no', which for a lossless one means it is really a
    container case: an animation frame and an alpha plane both carry a
    lossless image, and vp8l_dis.py reads a VP8L chunk, not a file.
    """
    bad = broken = skipped = 0
    done = 0
    for path in sorted(glob.glob('cases/*.txt')):
        with open(path) as fp:
            text = fp.read()
        if not vp8l_asm.is_lossless(text):
            continue
        head = vp8_asm.parse_header(text, path)
        if head.get('slow') == 'yes' or head['roundtrip'] == 'no':
            skipped += 1
            continue
        done += 1
        want = vp8l_dis.vp8l_chunk(vp8l_asm.assemble_text(text))
        try:
            got = vp8l_dis.vp8l_chunk(
                vp8l_asm.assemble_text(vp8l_dis.dump(want)))
        except (vp8l_dis.Truncated, vp8_asm.AsmError):
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
    print('lossless %d cases, %d refused by the reader, %d not a bare image'
          % (done, broken, skipped))
    return bad


def case_round_trip():
    """Cases: text -> bytes -> text -> bytes must not move either.

    A case whose whole point is to be refused is often unreadable by
    anything, this disassembler included. Those are counted rather than
    failed, as are the ones that say '# roundtrip: no' because what they
    pin is invisible to a reader. Everything else has to survive.

    Lossless cases go through lossless_round_trip() instead.
    """
    bad = broken = skipped = 0
    paths = []
    for path in sorted(glob.glob('cases/*.txt')):
        with open(path) as fp:
            if not vp8l_asm.is_lossless(fp.read()):
                paths.append(path)
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
    bad += lossless_round_trip()
    bad += real_encodes()
    bad += levels()
    if shutil.which(DWEBP):
        bad += pixels()
    else:
        print('pixels   skipped, set $DWEBP to a dwebp binary')
    print('FAILED, %d checks did not pass' % bad if bad else 'all checks pass')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
