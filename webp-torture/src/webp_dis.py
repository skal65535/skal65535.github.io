#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.

"""Reads a whole .webp file back into the case text that describes it.

    ./webp_dis.py some.webp                 print the case
    ./webp_dis.py --check some.webp         .webp -> .txt -> .webp, byte for
                                            byte

The other direction from webp_asm.py, and the layer above vp8_dis.py and
vp8l_dis.py: it walks the RIFF chunks, writes what each one says, and hands
the image chunks to whichever of those two owns them. An animation comes
back as one 'frame' block per ANMF, a compressed alpha plane as an
'alph_plane' block.

What it will not read is a file that lies about itself -- a chunk whose
declared length runs past the end, a frame that overruns its own payload.
Those are most of what the corpus is for, and they say '# roundtrip: no'.
"""

import os
import sys

import vp8_asm
import vp8_dis
import vp8l_dis
import webp_asm

HEADER = ['# note: disassembled by webp_dis.py', '# expect: ok',
          '# exercises: whatever the file it came from does.',
          '# roundtrip: no']


class Truncated(Exception):
    pass


# What a file that cannot be read back raises, whichever layer gives up.
UNREADABLE = (Truncated, vp8_dis.Truncated, vp8l_dis.Truncated,
              vp8_asm.AsmError, IndexError)


def walk(data, at, end, loose=False):
    """(fourcc, payload) for each chunk between two offsets.

    'loose' stops at the first thing that is not a whole chunk and reports
    what was left over, which at file level is the 'trailing' directive.
    Inside a frame there is no such directive, so anything left is fatal.
    """
    out = []
    while at < end:
        size = int.from_bytes(data[at + 4:at + 8], 'little') \
            if at + 8 <= end else 0
        if at + 8 > end or at + 8 + size > end:
            if loose:
                return out, data[at:end]
            raise Truncated('a chunk at %d that does not fit' % at)
        out.append((data[at:at + 4], data[at + 8:at + 8 + size]))
        at += 8 + size + (size & 1)
    if at != end:
        if loose:
            return out, b''            # a pad byte the last chunk implies
        raise Truncated('a pad byte past the end')
    return out, b''


def name_of(tag):
    """The fourcc as a case writes it; webp_asm.fourcc() pads it back."""
    return tag.decode('latin1').rstrip() or tag.decode('latin1')


def image_lines(tag, payload):
    """The lines describing one image chunk, whichever format it is."""
    if tag == b'VP8L':
        text = vp8l_dis.dump(payload)
    else:
        frame, sizes = vp8_dis.parse(payload)
        text = vp8_dis.dump(frame, sizes)
    return [l for l in text.splitlines() if not l.startswith('#')]


def image_size(chunks):
    """The size of the first image in a chunk list, for the fields that
    default to it: the ANMF box, and the length of a stored alpha plane."""
    for chunk in chunks:
        if chunk.tag == b'VP8L':
            _, width, height, _, _, _ = vp8l_dis.parse(chunk.payload)
            return width, height
        if chunk.tag == b'VP8 ':
            frame = vp8_dis.parse(chunk.payload)[0]
            return frame.width, frame.height
    return 1, 1


class Chunk:
    """One chunk of the file, kept until its neighbours have been read.

    ALPH comes before the image whose size it needs, and an ANMF header is
    written before the frame it introduces, so nothing can be turned into
    text on the first pass.
    """

    def __init__(self, tag, payload):
        self.tag, self.payload = tag, payload
        self.frame = None              # ANMF: its own Chunk list


def group_frames(data):
    """The file's chunks, with each ANMF holding its own, and the leftovers.

    Both of the ways a file can disagree with itself at this level are
    directives: a RIFF header that does not match what follows it, and
    bytes after the last whole chunk.
    """
    if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
        raise Truncated('not a RIFF/WEBP file')
    riff_size = int.from_bytes(data[4:8], 'little')
    out = []
    chunks, trailing = walk(data, 12, len(data), loose=True)
    for tag, payload in chunks:
        chunk = Chunk(tag, payload)
        if tag == b'ANMF':
            if len(payload) < 16:
                raise Truncated('ANMF of %d bytes' % len(payload))
            chunk.frame = [Chunk(t, p) for t, p in
                           walk(payload, 16, len(payload))[0]]
        out.append(chunk)
    return out, riff_size, trailing


def alph_lines(chunk, size):
    """An ALPH chunk: its header byte, then the plane, however it is stored.

    A stored plane is spelled out; a compressed one is an image stream in
    its own right and comes back as a block.
    """
    head, data = chunk.payload[0], chunk.payload[1:]
    out, block = [], []
    for field, shift in webp_asm.ALPH_FIELDS.items():
        value = (head >> shift) & 3
        if value:
            out.append('%s %d' % (field, value))
    if head & 3:                       # compression method 1: lossless
        block = ['alph_plane'] + vp8l_dis.image_lines(
            vp8l_dis.parse_alpha(data, *size))
    else:
        out.append('alph_data %s' % data.hex())
    return out, block


def contents(chunks, size, handled=()):
    """The lines for a list of chunks, and the blocks that trail them.

    A fourcc listed twice carries the same bytes both times, so the image
    and the alpha plane are written once however often they are named.
    'handled' names the chunks the caller has already written itself; a
    frame handles none of them, so an ANMF inside one is just bytes.
    """
    out, blocks, done = [], [], set()
    for chunk in chunks:
        if chunk.tag in done or chunk.tag in handled:
            continue
        done.add(chunk.tag)
        if chunk.tag == b'ALPH':
            more, block = alph_lines(chunk, size)
            out += more
            blocks += block
        elif chunk.tag in (b'VP8 ', b'VP8L'):
            out += image_lines(chunk.tag, chunk.payload)
        else:
            out.append(('payload %s %s' % (name_of(chunk.tag),
                                           chunk.payload.hex())).rstrip())
    return out, blocks


def frame_lines(chunk):
    """One ANMF chunk: the header fields, then the chunks it holds."""
    head = chunk.payload[:16]
    fields = [int.from_bytes(head[3 * i:3 * i + 3], 'little')
              for i in range(5)]
    bits = head[15]
    out = ['frame']
    size = image_size(chunk.frame)
    for name, value in zip(webp_asm.ANMF_FIELDS, fields):
        want = {'frame_width_minus_one': size[0] - 1,
                'frame_height_minus_one': size[1] - 1}.get(name, 0)
        if value != want:
            out.append('%s %d' % (name, value))
    for name, shift in webp_asm.ANMF_BITS.items():
        value = (bits >> shift) & (0x3f if shift == 2 else 1)
        if value:
            out.append('%s %d' % (name, value))
    out.append('chunks ' + ' '.join(name_of(c.tag) for c in chunk.frame))
    more, blocks = contents(chunk.frame, size)
    return out + more + blocks


def dump(data):
    """The case text for a whole file."""
    chunks, riff_size, trailing = group_frames(data)
    size = image_size(chunks)
    frames = [c for c in chunks if c.tag == b'ANMF']
    out = list(HEADER)
    out.append('chunks ' + ' '.join(name_of(c.tag) for c in chunks))
    if riff_size + 8 != len(data):
        out.append('riff_size %d' % riff_size)
    if trailing:
        out.append('trailing %s' % trailing.hex())
    for chunk in chunks:
        # A chunk this file has a builder for is written as its fields, but
        # only when it is the length that builder would have produced;
        # anything else is spelled out and left alone.
        if chunk.tag == b'VP8X' and len(chunk.payload) == 10:
            flags = int.from_bytes(chunk.payload[:4], 'little')
            for name, bit in webp_asm.FLAGS.items():
                # a clear flag needs no line, except the animation one once
                # there are frames: the assembler would turn it back on
                if flags & bit or (name == 'animation' and frames):
                    out.append('%s %d' % (name, 1 if flags & bit else 0))
            if flags >> 8:
                out.append('vp8x_reserved 0x%x' % (flags >> 8))
            out.append('canvas_width_minus_one %d'
                       % int.from_bytes(chunk.payload[4:7], 'little'))
            out.append('canvas_height_minus_one %d'
                       % int.from_bytes(chunk.payload[7:10], 'little'))
        elif chunk.tag == b'ANIM' and len(chunk.payload) == 6:
            out.append('background_color 0x%08x'
                       % int.from_bytes(chunk.payload[:4], 'little'))
            out.append('loop_count %d'
                       % int.from_bytes(chunk.payload[4:6], 'little'))
        elif chunk.tag in (b'VP8X', b'ANIM'):
            out.append(('payload %s %s' % (name_of(chunk.tag),
                                           chunk.payload.hex())).rstrip())
    more, blocks = contents(chunks, size, (b'VP8X', b'ANIM', b'ANMF'))
    for chunk in frames:
        blocks += frame_lines(chunk)
    return '\n'.join(out + more + blocks) + '\n'


def check(path, data=None):
    """Reads a file back into text and reassembles it. True if it matched."""
    if data is None:
        with open(path, 'rb') as f:
            data = f.read()
    try:
        text = dump(data)
    except UNREADABLE as e:
        print('%-44s unreadable: %s' % (path, e))
        return False
    again = webp_asm.assemble_text(text)
    if again == data:
        print('%-44s round trip ok, %d bytes' % (path, len(data)))
        return True
    print('%-44s reassembles to %d bytes, not %d'
          % (path, len(again), len(data)), file=sys.stderr)
    return False


def main(argv):
    args = [a for a in argv[1:] if not a.startswith('--')]
    if not args:
        print(__doc__.strip().split('\n\n')[0], file=sys.stderr)
        print('usage: %s [--check] <file.webp>'
              % os.path.basename(argv[0]), file=sys.stderr)
        return 1
    if '--check' in argv:
        return 0 if all([check(a) for a in args]) else 1
    for path in args:
        with open(path, 'rb') as f:
            sys.stdout.write(dump(f.read()))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
