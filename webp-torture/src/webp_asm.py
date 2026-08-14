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

DIRECTIVES = set(FLAGS) | set(ALPH_FIELDS) | {
    'chunks', 'vp8x_reserved', 'payload', 'riff_size', 'chunk_size',
    'trailing', 'canvas_width_minus_one', 'canvas_height_minus_one',
    'alph_raw', 'alph_data'}


def fourcc(name):
    return name.encode('ascii')[:4].ljust(4)


class Container:
    """The chunk list a case asks for, and the bytes it comes to."""

    def __init__(self, image='VP8'):
        self.image = fourcc(image)     # the chunk the rest of the case is
        self.chunks = [image]
        self.flags = 0
        self.reserved = 0
        self.canvas = [None, None]     # width - 1, height - 1
        self.payloads = {}             # fourcc -> bytes
        self.sizes = {}                # fourcc -> what its header claims
        self.riff_size = None
        self.trailing = b''
        self.alph = dict.fromkeys(ALPH_FIELDS, 0)
        self.alph_raw = None           # how many plane bytes to generate
        self.alph_data = None          # or the bytes, spelled out
        self.line = 0

    def fail(self, msg):
        raise vp8_asm.AsmError('line %d: %s' % (self.line, msg))

    def num(self, tok, what):
        try:
            return int(tok, 0)
        except ValueError:
            self.fail('bad %s %r' % (what, tok))

    def feed(self, lines):
        for lineno, line in lines:
            self.line = lineno
            args = line.split('#')[0].split()
            name, rest = args[0], args[1:]
            if name == 'chunks':
                self.chunks = rest
            elif name in FLAGS:
                if self.num(rest[0], name):
                    self.flags |= FLAGS[name]
                else:
                    self.flags &= ~FLAGS[name]
            elif name in ALPH_FIELDS:
                self.alph[name] = self.num(rest[0], name)
            elif name == 'alph_raw':
                self.alph_raw = self.num(rest[0], name) if rest else -1
            elif name == 'alph_data':
                self.alph_data = bytes.fromhex(rest[0]) if rest else b''
            elif name == 'vp8x_reserved':
                self.reserved = self.num(rest[0], name)
            elif name.startswith('canvas_'):
                self.canvas[0 if 'width' in name else 1] = \
                    self.num(rest[0], name)
            elif name == 'payload':
                self.payloads[fourcc(rest[0])] = bytes.fromhex(rest[1])
            elif name == 'chunk_size':
                self.sizes[fourcc(rest[0])] = self.num(rest[1], name)
            elif name == 'riff_size':
                self.riff_size = self.num(rest[0], name)
            elif name == 'trailing':
                self.trailing = bytes.fromhex(rest[0])

    def vp8x(self, frame):
        """The 10-byte VP8X payload: flags, then the canvas size, less one."""
        w, h = self.canvas
        if w is None:
            w = frame.width - 1
        if h is None:
            h = frame.height - 1
        return ((self.flags | (self.reserved << 8)) & 0xffffffff).to_bytes(
            4, 'little') + (w & 0xffffff).to_bytes(3, 'little') + \
            (h & 0xffffff).to_bytes(3, 'little')

    def alph_chunk(self, frame):
        """The header byte, then the alpha plane.

        An uncompressed plane is stored as it is, one byte per pixel, so
        the default is exactly the width by height the frame calls for --
        which is the size ALPHInit() checks against.
        """
        head = 0
        for name, shift in ALPH_FIELDS.items():
            head |= (self.alph[name] & 3) << shift
        if self.alph_data is not None:
            data = self.alph_data
        else:
            n = self.alph_raw
            if n is None or n < 0:
                n = frame.width * frame.height
            data = bytes((i * 37 + 11) & 0xff for i in range(n))
        return bytes([head]) + data

    def build(self, frame, image):
        """The whole file: the RIFF header, then every chunk in turn."""
        out = bytearray()
        for name in self.chunks:
            tag = fourcc(name)
            if tag == self.image:
                data = image
            elif tag == b'VP8X':
                data = self.vp8x(frame)
            elif tag == b'ALPH':
                data = self.alph_chunk(frame)
            else:
                data = self.payloads.get(tag, b'')
            size = self.sizes.get(tag, len(data))
            out += tag + (size & 0xffffffff).to_bytes(4, 'little') + data
            if size & 1:
                out += b'\0'           # the pad byte an odd payload needs
        out += self.trailing
        size = self.riff_size
        if size is None:
            size = 4 + len(out)        # 'WEBP' plus the chunks
        return (b'RIFF' + (size & 0xffffffff).to_bytes(4, 'little') +
                b'WEBP' + bytes(out))


def assemble_text(text):
    """The .webp bytes for one case, container and all."""
    container, frame = [], []
    for lineno, line in enumerate(text.splitlines(), 1):
        args = line.split('#')[0].split()
        if args and args[0] in DIRECTIVES:
            container.append((lineno, line))
        else:
            frame.append(line)
    lossless = vp8l_asm.is_lossless(text)
    asm = (vp8l_asm if lossless else vp8_asm).Assembler()
    what = asm.feed('\n'.join(frame))
    image = asm.finish()
    if not container:
        return (vp8l if lossless else vp8).wrap_webp(image)
    box = Container('VP8L' if lossless else 'VP8')
    box.feed(container)
    return box.build(what, image)


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
