#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.

"""Wraps an assembled frame in a RIFF container, chunk by chunk.

    ./webp_asm.py cases/container-vp8x.txt files/container-vp8x.webp

The layer above vp8_asm.py and vp8l_asm.py, and the one that picks between
them: a case that says 'lossless' is a VP8L image, anything else a lossy VP8
frame. A case that says nothing about the container gets the plain
'RIFF....WEBP' plus its one image chunk; the directives below are for the
cases that need the container itself to be interesting. Names are RFC
9649's.

  chunks FOURCC...        the chunks to write, in order. VP8 -- or VP8L, in
                          a lossless case -- is the image the rest of the
                          case describes, and is what the list defaults to
  alpha 0                 the VP8X feature flags, by their RFC names
  animation 0
  icc_profile 0
  exif_metadata 0
  xmp_metadata 0
  vp8x_reserved N         the two reserved bits and the reserved byte, raw
  canvas_width_minus_one N
  canvas_height_minus_one N
  alph_compression 0      the four fields of the ALPH chunk's header byte
  alph_filtering 0        0 none, 1 horizontal, 2 vertical, 3 gradient
  alph_preprocessing 0    1 is level reduction
  alph_reserved 0
  alph_raw N              N bytes of alpha plane, uncompressed. Default is
                          the width x height the frame needs
  alph_data HEX           the bytes after the header byte, spelled out
  payload FOURCC HEX      the bytes of a chunk this file has no builder for
  riff_size N             what the RIFF header claims, when that should not
                          be what the file actually holds
  chunk_size FOURCC N     the same lie, for one chunk's own header
  trailing HEX            bytes after the last chunk

A fourcc is padded to four characters, so VP8 means 'VP8 '. Listing one
twice is allowed and is the point of a few cases; so is listing a fourcc
nothing else in the file mentions, which is how an unknown chunk is made.
An odd-sized payload is followed by the pad byte the format requires. It
is the declared size that decides, so 'chunk_size' takes the pad byte away
as well as changing what the header says.

Two keywords open a block, and everything after one belongs to it until the
next block or the end of the case. This is the only nesting the syntax has,
and it is here because the format nests here: an animation holds a whole
image per frame, and a compressed alpha plane is an image of its own.

  frame                   an ANMF chunk. The lines under it are its own
                          case: its image, the chunk list that carries it,
                          and the ANMF header fields
  frame_x 0               the offset field; the pixel offset is twice it
  frame_y 0
  frame_width_minus_one N   default is the image the frame carries
  frame_height_minus_one N
  frame_duration 0        in milliseconds, 24 bits
  disposal_method 0       1 clears the frame's area to the background
  blending_method 0       1 is 'do not blend'
  frame_reserved 0        the six bits above those two
  loop_count 0            the ANIM chunk, which frames need before them
  background_color N      ARGB, and only ever read back out again

  alph_plane              the ALPH payload as a lossless image stream: no
                          signature, no dimensions of its own, which is
                          what VP8LDecodeAlphaHeader() reads. The plane is
                          the green channel. Attaches to the frame above
                          it, or to the file when there is none

A file with frames defaults to 'VP8X ANIM ANMF...', one ANMF per frame, and
sets the animation flag; a frame defaults to the one image chunk it holds,
after an ALPH if it describes one. The canvas defaults to what the frames
cover. Every one of those is a field like any other, so a case can set the
flag without the chunks, write the chunks in the wrong order, or leave a
frame outside the canvas it claims to be in.
"""

import os
import sys

import vp8
import vp8_asm
import vp8l
import vp8l_asm

# RFC 9649's feature flags, from WebPFeatureFlags in src/webp/mux_types.h
FLAGS = {'animation': 0x02, 'xmp_metadata': 0x04, 'exif_metadata': 0x08,
         'alpha': 0x10, 'icc_profile': 0x20}
ALPH_FIELDS = {'alph_compression': 0, 'alph_filtering': 2,
               'alph_preprocessing': 4, 'alph_reserved': 6}
# The ANMF header: five 3-byte fields in the order they are written, then a
# byte holding these three, least-significant first.
ANMF_FIELDS = ['frame_x', 'frame_y', 'frame_width_minus_one',
               'frame_height_minus_one', 'frame_duration']
ANMF_BITS = {'disposal_method': 0, 'blending_method': 1, 'frame_reserved': 2}

# The directives every chunk list understands, wherever it sits.
CHUNK_DIRECTIVES = set(ALPH_FIELDS) | {
    'chunks', 'payload', 'chunk_size', 'alph_raw', 'alph_data'}
DIRECTIVES = CHUNK_DIRECTIVES | set(FLAGS) | set(ANMF_BITS) | \
    set(ANMF_FIELDS) | {
    'vp8x_reserved', 'riff_size', 'trailing', 'canvas_width_minus_one',
    'canvas_height_minus_one', 'loop_count', 'background_color'}
BLOCKS = ('frame', 'alph_plane')


def fourcc(name):
    return name.encode('ascii')[:4].ljust(4)


def le(value, n):
    return (value & ((1 << (8 * n)) - 1)).to_bytes(n, 'little')


class Chunks:
    """A list of chunks, and whatever fills each one in.

    The file and each of its animation frames are both this: a sequence of
    fourccs, one of which is an image, with the rest either built here or
    spelled out. What differs is only what wraps them.
    """

    def __init__(self):
        self.image = b'VP8 '           # the chunk the block's own case is
        self.chunks = None             # None until the case says otherwise
        self.payloads = {}             # fourcc -> bytes
        self.sizes = {}                # fourcc -> what its header claims
        self.alph = dict.fromkeys(ALPH_FIELDS, None)
        self.alph_raw = None           # how many plane bytes to generate
        self.alph_data = None          # or the bytes, spelled out
        self.alph_plane = None         # or the lossless stream that is them
        self.what = None               # the image, as its assembler saw it
        self.payload = b''             # and as bytes
        self.used = False              # did the case say anything here at all
        self.line = 0

    def fail(self, msg):
        raise vp8_asm.AsmError('line %d: %s' % (self.line, msg))

    def num(self, tok, what):
        try:
            return int(tok, 0)
        except ValueError:
            self.fail('bad %s %r' % (what, tok))

    def take(self, name, rest):
        """One chunk-list directive. False if it belongs to a layer above."""
        if name == 'chunks':
            self.chunks = rest
        elif name in ALPH_FIELDS:
            self.alph[name] = self.num(rest[0], name)
        elif name == 'alph_raw':
            self.alph_raw = self.num(rest[0], name) if rest else -1
        elif name == 'alph_data':
            self.alph_data = bytes.fromhex(rest[0]) if rest else b''
        elif name == 'payload':
            self.payloads[fourcc(rest[0])] = bytes.fromhex(
                rest[1]) if len(rest) > 1 else b''
        elif name == 'chunk_size':
            self.sizes[fourcc(rest[0])] = self.num(rest[1], name)
        else:
            return False
        return True

    def describes_alpha(self):
        return (self.alph_plane is not None or self.alph_data is not None or
                self.alph_raw is not None or
                any(v is not None for v in self.alph.values()))

    def alph_chunk(self):
        """The header byte, then the alpha plane.

        An uncompressed plane is stored as it is, one byte per pixel, so
        the default is exactly the width by height the frame calls for --
        which is the size ALPHInit() checks against. A plane written as an
        image stream is compressed by definition, so that is the default
        the header byte carries.
        """
        head = 0
        for name, shift in ALPH_FIELDS.items():
            value = self.alph[name]
            if value is None:
                value = 1 if (name == 'alph_compression' and
                              self.alph_plane is not None) else 0
            head |= (value & 3) << shift
        if self.alph_plane is not None:
            data = self.alph_plane
        elif self.alph_data is not None:
            data = self.alph_data
        else:
            n = self.alph_raw
            if n is None or n < 0:
                n = self.what.width * self.what.height
            data = bytes((i * 37 + 11) & 0xff for i in range(n))
        return bytes([head]) + data

    def default_chunks(self):
        return (['ALPH'] if self.describes_alpha() else []) + \
            [self.image.decode('ascii').rstrip()]

    def chunk_data(self, tag):
        """What one fourcc carries. A spelled-out payload wins over anything
        this file would have built for that chunk, which is the only way to
        write one the builders have no shape for."""
        if tag in self.payloads:
            return self.payloads[tag]
        if tag == self.image:
            return self.payload
        if tag == b'ALPH':
            return self.alph_chunk()
        return b''

    def emit(self):
        """Every chunk in turn, header, payload and pad byte."""
        out = bytearray()
        for name in (self.chunks if self.chunks is not None
                     else self.default_chunks()):
            tag = fourcc(name)
            data = self.chunk_data(tag)
            size = self.sizes.get(tag, len(data))
            out += tag + le(size, 4) + data
            if size & 1:
                out += b'\0'           # the pad byte an odd payload needs
        return bytes(out)


class Frame(Chunks):
    """One ANMF chunk: the header of RFC 9649 section 2.5.4, then chunks."""

    def __init__(self):
        super().__init__()
        self.fields = dict.fromkeys(ANMF_FIELDS, None)
        self.bits = dict.fromkeys(ANMF_BITS, 0)

    def take(self, name, rest):
        if name in self.fields:
            self.fields[name] = self.num(rest[0], name)
        elif name in self.bits:
            self.bits[name] = self.num(rest[0], name)
        else:
            return super().take(name, rest)
        return True

    def field(self, name):
        """One header field, with the image behind it as the default."""
        value = self.fields[name]
        if value is not None:
            return value
        if name == 'frame_width_minus_one':
            return self.what.width - 1
        if name == 'frame_height_minus_one':
            return self.what.height - 1
        return 0

    def extent(self):
        """The canvas this frame needs: past its offset, its own size."""
        return (2 * self.field('frame_x') + self.field('frame_width_minus_one')
                + 1,
                2 * self.field('frame_y') + self.field('frame_height_minus_one')
                + 1)

    def anmf(self):
        """The ANMF payload: the header fields, then the frame's chunks."""
        flags = 0
        for name, shift in ANMF_BITS.items():
            flags |= (self.bits[name] << shift) & 0xff
        return b''.join(le(self.field(name), 3) for name in ANMF_FIELDS) + \
            bytes([flags]) + self.emit()


class Container(Chunks):
    """The whole file: the RIFF header, then every chunk in turn."""

    def __init__(self):
        super().__init__()
        self.flags = 0
        self.flags_seen = set()        # so one flag's default outlives another
        self.reserved = 0
        self.canvas = [None, None]     # width - 1, height - 1
        self.riff_size = None
        self.trailing = b''
        self.loop_count = 0
        self.background_color = 0xffffffff
        self.frames = []
        self.next_frame = 0

    def take(self, name, rest):
        if name in FLAGS:
            self.flags_seen.add(name)
            if self.num(rest[0], name):
                self.flags |= FLAGS[name]
            else:
                self.flags &= ~FLAGS[name]
        elif name == 'vp8x_reserved':
            self.reserved = self.num(rest[0], name)
        elif name.startswith('canvas_'):
            self.canvas[0 if 'width' in name else 1] = self.num(rest[0], name)
        elif name == 'riff_size':
            self.riff_size = self.num(rest[0], name)
        elif name == 'trailing':
            self.trailing = bytes.fromhex(rest[0])
        elif name in ('loop_count', 'background_color'):
            setattr(self, name, self.num(rest[0], name))
        else:
            return super().take(name, rest)
        return True

    def default_chunks(self):
        if not self.frames:
            return super().default_chunks()
        return ['VP8X', 'ANIM'] + ['ANMF'] * len(self.frames)

    def canvas_size(self):
        """What the VP8X claims, less one: the frames' extent, or the image."""
        if self.frames:
            sizes = [f.extent() for f in self.frames]
            return max(w for w, _ in sizes) - 1, max(h for _, h in sizes) - 1
        return self.what.width - 1, self.what.height - 1

    def vp8x(self):
        """The 10-byte VP8X payload: flags, then the canvas size, less one."""
        flags = self.flags
        if self.frames and 'animation' not in self.flags_seen:
            flags |= FLAGS['animation']
        default = self.canvas_size()
        canvas = [d if c is None else c for c, d in zip(self.canvas, default)]
        return le(flags | (self.reserved << 8), 4) + \
            le(canvas[0], 3) + le(canvas[1], 3)

    def chunk_data(self, tag):
        if tag in self.payloads:
            return self.payloads[tag]
        if tag == b'VP8X':
            return self.vp8x()
        if tag == b'ANIM':
            return le(self.background_color, 4) + le(self.loop_count, 2)
        if tag == b'ANMF':
            if self.next_frame >= len(self.frames):
                raise vp8_asm.AsmError(
                    'chunks: ANMF number %d, but the case writes %d frames'
                    % (self.next_frame + 1, len(self.frames)))
            self.next_frame += 1
            return self.frames[self.next_frame - 1].anmf()
        return super().chunk_data(tag)

    def build(self):
        out = self.emit() + self.trailing
        size = self.riff_size
        if size is None:
            size = 4 + len(out)        # 'WEBP' plus the chunks
        return b'RIFF' + le(size, 4) + b'WEBP' + out


def split_blocks(text):
    """The case's preamble, then one (keyword, lines) per block it opens."""
    blocks = [(None, [])]
    for lineno, line in enumerate(text.splitlines(), 1):
        args = line.split('#')[0].split()
        if args and args[0] in BLOCKS:
            blocks.append((args[0], []))
        else:
            blocks[-1][1].append((lineno, line))
    return blocks


def fill(box, lines, header=True, size=None):
    """Feeds one block's lines to 'box', and assembles the image around them.

    The two kinds of line are told apart by the keyword alone, so a case
    reads as one list however many layers it is really describing. 'size'
    is what a block inherits from the one it hangs off; a line of its own
    overrides it, being read later.
    """
    image = ['width %d' % size[0], 'height %d' % size[1]] if size else []
    for lineno, line in lines:
        args = line.split('#')[0].split()
        if args and args[0] in DIRECTIVES:
            box.line = lineno
            box.used = True
            if not box.take(args[0], args[1:]):
                box.fail('%r belongs to another layer' % args[0])
        else:
            image.append(line)
    text = '\n'.join(image)
    lossless = vp8l_asm.is_lossless(text) or not header
    asm = (vp8l_asm if lossless else vp8_asm).Assembler()
    box.what = asm.feed(text)
    box.image = b'VP8L' if lossless else b'VP8 '
    box.payload = asm.finish() if header else asm.finish(header=False)
    return box


def assemble_text(text):
    """The .webp bytes for one case, container, frames and all."""
    blocks = split_blocks(text)
    box = Container()
    fill(box, blocks[0][1])
    where = box                        # what an alph_plane attaches to
    for kind, lines in blocks[1:]:
        if kind == 'frame':
            where = fill(Frame(), lines)
            box.frames.append(where)
        else:
            plane = fill(Chunks(), lines, header=False,
                         size=(where.what.width, where.what.height))
            where.alph_plane, where.used = plane.payload, True
    if not box.used and not box.frames:
        return (vp8l if box.image == b'VP8L' else vp8).wrap_webp(box.payload)
    return box.build()


def main(argv):
    if not 2 <= len(argv) <= 3:
        print('usage: %s <case.txt> [<out.webp>]'
              % os.path.basename(argv[0]), file=sys.stderr)
        return 1
    src = argv[1]
    dst = argv[2] if len(argv) > 2 else os.path.splitext(src)[0] + '.webp'
    with open(src) as f:
        text = f.read()
    try:
        data = assemble_text(text)
    except vp8_asm.AsmError as e:
        print('%s:%s' % (src, e), file=sys.stderr)
        return 1
    with open(dst, 'wb') as f:
        f.write(data)
    print('%-40s %5d bytes' % (dst, len(data)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
