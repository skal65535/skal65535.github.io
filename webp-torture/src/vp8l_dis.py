#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.

"""Reads a lossless .webp back into the text vp8l_asm.py assembles.

    ./src/vp8l_dis.py some-lossless.webp          write the case to stdout
    ./src/vp8l_dis.py --check some-lossless.webp  and reassemble it, byte
                                                  for byte

The other direction from vp8l_asm.py, and the only check on it that does
not go through this file's own idea of the format: disassemble a real
encode, reassemble from its own text, compare. Anything the two disagree
on shows up as a differing byte.

It parses what the decoder parses, not what an encoder meant, so it writes
the low-level form of everything -- the pixels of a sub-image rather than
'predictor_tiles', every Huffman code spelled out rather than left to be
derived. That is what makes the round trip exact: nothing is re-chosen on
the way back.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vp8l
import vp8l_asm


class Truncated(Exception):
    """The stream ended in the middle of something."""


class BitReader:
    """LSB-first, like VP8LReadBits()."""

    def __init__(self, data):
        self.data = data
        self.pos = 0                       # in bits

    def read(self, n):
        value = 0
        for i in range(n):
            at = self.pos + i
            if at >= 8 * len(self.data):
                raise Truncated('at bit %d of %d' % (at, 8 * len(self.data)))
            value |= ((self.data[at >> 3] >> (at & 7)) & 1) << i
        self.pos += n
        return value


class Decoder:
    """Reads symbols of one Huffman code, from its lengths."""

    def __init__(self, lengths):
        self.lengths = lengths
        used = [s for s, l in enumerate(lengths) if l]
        self.trivial = used[0] if len(used) == 1 else None
        self.map = {}
        for symbol, (length, code) in enumerate(
                zip(lengths, vp8l.canonical_codes(lengths))):
            if length:
                self.map[(length, code)] = symbol

    def read(self, br):
        if self.trivial is not None:
            return self.trivial            # zero bits wide
        code = 0
        for length in range(1, vp8l.MAX_ALLOWED_CODE_LENGTH + 1):
            code |= br.read(1) << (length - 1)
            if (length, code) in self.map:
                return self.map[(length, code)]
        raise Truncated('no symbol after %d bits'
                        % vp8l.MAX_ALLOWED_CODE_LENGTH)


class Code:
    """One Huffman code as the stream spelled it, so it can be re-spelled."""

    def __init__(self):
        self.simple = None                 # symbols of the simple form
        self.short = False                 # first symbol in 1 bit
        self.num_codes = None
        self.max_symbol = None
        self.cl_lengths = None
        self.codelen = None                # [(symbol, extra bits, value)]
        self.lengths = []
        self.decoder = None

    def lines(self, name, prefix=''):
        """The 'code NAME ...' lines that write this one back."""
        out = []
        if self.simple is not None:
            form = 'simple1 ' if self.short else ''
            out.append('%scode %s %s%s'
                       % (prefix, name, form,
                          ' '.join('0x%02x' % s for s in self.simple)))
            return out
        out.append('%scode %s num_codes %d' % (prefix, name, self.num_codes))
        out.append('%scode %s cl_lengths %s' % (prefix, name, spell(
            self.cl_lengths)))
        if self.max_symbol is not None:
            out.append('%scode %s max_symbol %d'
                       % (prefix, name, self.max_symbol))
        out.append('%scode %s codelen %s'
                   % (prefix, name, ' '.join(spell_codelen(self.codelen))))
        if any(self.lengths):
            out.append('%scode %s lengths %s' % (prefix, name, spell(
                self.lengths)))
        return out


def spell(lengths):
    """Non-zero entries of a length array, as 'symbol:length' pairs."""
    return ' '.join('%d:%d' % (s, l) for s, l in enumerate(lengths) if l)


def spell_codelen(symbols):
    """(symbol, bits, value) back into the '18x138' shorthand."""
    out = []
    for symbol, _bits, value in symbols:
        if symbol in vp8l_asm.REPEATS:
            out.append('%dx%d' % (symbol, value + vp8l_asm.REPEATS[symbol][1]))
        else:
            out.append(str(symbol))
    return out


def read_code(br, alphabet_size):
    """Mirrors ReadHuffmanCode(): the simple form, or the normal one."""
    code = Code()
    code.lengths = [0] * alphabet_size
    if br.read(1):                                  # simple
        num_symbols = br.read(1) + 1
        code.short = br.read(1) == 0
        code.simple = [br.read(1 if code.short else 8)]
        if num_symbols == 2:
            code.simple.append(br.read(8))
        for symbol in code.simple:
            if symbol < alphabet_size:
                code.lengths[symbol] = 1
    else:
        code.num_codes = br.read(4) + 4
        code.cl_lengths = [0] * vp8l.CODE_LENGTH_CODES
        for i in range(code.num_codes):
            code.cl_lengths[vp8l.CODE_LENGTH_CODE_ORDER[i]] = br.read(3)
        cl = Decoder(code.cl_lengths)
        left = alphabet_size
        if br.read(1):                              # explicit max_symbol
            length_nbits = 2 + 2 * br.read(3)
            code.max_symbol = 2 + br.read(length_nbits)
            left = code.max_symbol
        code.codelen = read_code_lengths(br, cl, code.lengths, left)
    if not any(code.lengths):
        raise Truncated('empty Huffman code')
    code.decoder = Decoder(code.lengths)
    return code


def read_code_lengths(br, cl, lengths, left):
    """Mirrors ReadHuffmanCodeLengths(), recording what it read."""
    out = []
    symbol, previous = 0, vp8l.DEFAULT_CODE_LENGTH
    while symbol < len(lengths):
        if left == 0:
            break
        left -= 1
        length = cl.read(br)
        if length < 16:
            out.append((length, 0, 0))
            lengths[symbol] = length
            symbol += 1
            if length:
                previous = length
            continue
        bits, base = vp8l_asm.REPEATS[length]
        extra = br.read(bits)
        out.append((length, bits, extra))
        repeat = extra + base
        if symbol + repeat > len(lengths):
            raise Truncated('repeat past the end of the alphabet')
        fill = previous if length == 16 else 0
        for _ in range(repeat):
            lengths[symbol] = fill
            symbol += 1
    return out


class Group:
    """The five codes of one Huffman group."""

    def __init__(self, br, cache_bits):
        self.codes = [read_code(br, vp8l.alphabet_size(i, cache_bits))
                      for i in range(5)]

    @property
    def trivial_literal(self):
        return all(c.decoder.trivial is not None for c in self.codes[1:4])

    @property
    def trivial_code(self):
        """No bits at all per pixel: every code is one symbol, green a
        literal. cf. is_trivial_code in ReadHuffmanCodes()."""
        return (self.trivial_literal and
                all(c.decoder.trivial is not None for c in self.codes) and
                self.codes[0].decoder.trivial < vp8l.NUM_LITERAL_CODES)

    def lines(self, prefix=''):
        out = []
        for name, code in zip(vp8l_asm.CODE_NAMES, self.codes):
            out += code.lines(name, prefix)
        return out


class Image:
    """One image stream, level 0 or not."""

    def __init__(self, br, xsize, ysize, level0):
        self.xsize, self.ysize = xsize, ysize
        self.transforms = []               # [(name, params)]
        self.cache_bits = None
        self.meta_bits = None
        self.meta = None                   # the entropy image, as an Image
        self.groups = []
        self.pixels = []                   # decoded ARGB, for a sub-image
        self.items = []                    # what the pixel data said

        if level0:
            while br.read(1):
                xsize = self.read_transform(br, xsize, ysize)
            self.xsize = xsize
        if br.read(1):
            self.cache_bits = br.read(4)
            if not 1 <= self.cache_bits <= vp8l.MAX_CACHE_BITS:
                raise Truncated('color cache of %d bits' % self.cache_bits)
        groups = 1
        if level0 and br.read(1):
            self.meta_bits = vp8l.MIN_HUFFMAN_BITS + br.read(
                vp8l.NUM_HUFFMAN_BITS)
            self.meta = Image(br, vp8l.sub_sample_size(xsize, self.meta_bits),
                              vp8l.sub_sample_size(ysize, self.meta_bits),
                              level0=False)
            if any(i[0] == 'cache' for i in self.meta.items):
                raise Truncated('entropy image using its color cache: its '
                                'pixels are the group indices, and what the '
                                'cache held is not modelled here')
            self.tiles = [(p >> 8) & 0xffff for p in self.meta.pixels]
            groups = max(self.tiles) + 1
        else:
            self.tiles = None
        self.groups = [Group(br, self.cache_bits or 0) for _ in range(groups)]
        self.read_pixels(br)

    def read_transform(self, br, xsize, ysize):
        """One transform; returns the width left behind."""
        kind = br.read(2)
        name = [n for n, v in vp8l_asm.TRANSFORMS.items() if v == kind][0]
        params = {}
        if name == 'color_indexing':
            count = br.read(8) + 1
            params['count'] = count
            params['image'] = Image(br, count, 1, level0=False)
            xsize = vp8l.sub_sample_size(xsize,
                                         vp8l.palette_bits(count))
        elif name != 'subtract_green':
            params['bits'] = vp8l.MIN_TRANSFORM_BITS + br.read(
                vp8l.NUM_TRANSFORM_BITS)
            params['image'] = Image(
                br, vp8l.sub_sample_size(xsize, params['bits']),
                vp8l.sub_sample_size(ysize, params['bits']), level0=False)
        self.transforms.append((name, params))
        return xsize

    def read_pixels(self, br):
        """The pixel data, as items and as the ARGB it comes to.

        Only the entropy image's values are ever read back, so a cache index
        contributes a placeholder rather than a modelled cache; parse()
        refuses the one case where that would matter.
        """
        total = self.xsize * self.ysize
        at = 0
        while at < total:
            group = self.groups[0] if self.tiles is None else self.groups[
                vp8l.tile_at(self.tiles, at, self.xsize, self.meta_bits)]
            if group.trivial_code:
                literal = group.codes[0].decoder.trivial
                self.pixels.append(self.arb(group) | (literal << 8))
                at += 1
                continue
            symbol = group.codes[0].decoder.read(br)
            if symbol < vp8l.NUM_LITERAL_CODES:
                if group.trivial_literal:
                    self.items.append(('green', symbol))
                    self.pixels.append(self.arb(group) | (symbol << 8))
                else:
                    red = group.codes[1].decoder.read(br)
                    blue = group.codes[2].decoder.read(br)
                    alpha = group.codes[3].decoder.read(br)
                    pixel = ((alpha << 24) | (red << 16) |
                             (symbol << 8) | blue)
                    self.items.append(('argb', pixel))
                    self.pixels.append(pixel)
                at += 1
            elif symbol < vp8l.CACHE_BASE:
                length = read_prefix(br, symbol - vp8l.NUM_LITERAL_CODES)
                plane = read_prefix(br, group.codes[4].decoder.read(br))
                self.items.append(('copy', length, plane))
                distance = vp8l.plane_code_to_distance(self.xsize, plane)
                if distance > len(self.pixels) or at + length > total:
                    raise Truncated('back-reference out of range')
                for _ in range(length):
                    self.pixels.append(self.pixels[-distance])
                at += length
            else:
                self.items.append(('cache', symbol - vp8l.CACHE_BASE))
                self.pixels.append(0)
                at += 1

    @staticmethod
    def arb(group):
        """literal_arb: the three literal codes' single symbols."""
        alpha = group.codes[3].decoder.trivial or 0
        red = group.codes[1].decoder.trivial or 0
        blue = group.codes[2].decoder.trivial or 0
        return (alpha << 24) | (red << 16) | blue


def read_prefix(br, symbol):
    """Inverse of GetCopyLength()/GetCopyDistance()."""
    if symbol < 4:
        return symbol + 1
    extra_bits = (symbol - 2) >> 1
    offset = (2 + (symbol & 1)) << extra_bits
    return offset + br.read(extra_bits) + 1


def vp8l_chunk(data):
    """The VP8L payload of a RIFF file, or the whole thing if it is bare."""
    if data[:4] != b'RIFF':
        return data
    at = 12
    while at + 8 <= len(data):
        size = int.from_bytes(data[at + 4:at + 8], 'little')
        if data[at:at + 4] == b'VP8L':
            return data[at + 8:at + 8 + size]
        at += 8 + size + (size & 1)
    raise Truncated('no VP8L chunk')


def parse(payload):
    """(magic, width, height, alpha, version, Image) of one VP8L stream."""
    br = BitReader(payload)
    magic = br.read(8)
    width = br.read(vp8l.IMAGE_SIZE_BITS) + 1
    height = br.read(vp8l.IMAGE_SIZE_BITS) + 1
    alpha = br.read(1)
    version = br.read(vp8l.VERSION_BITS)
    return magic, width, height, alpha, version, Image(br, width, height, True)


def dump(payload, note='disassembled by vp8l_dis.py'):
    """The case text for one VP8L stream."""
    magic, width, height, alpha, version, image = parse(payload)
    out = ['# note: %s' % note, '# expect: ok',
           '# exercises: whatever the file it came from does.',
           '# roundtrip: no', 'lossless']
    if magic != vp8l.MAGIC:
        out.append('magic 0x%02x' % magic)
    out += ['width %d' % width, 'height %d' % height]
    if version:
        out.append('version %d' % version)
    if alpha:
        out.append('alpha_is_used 1')
    if image.transforms:
        out.append('transforms ' + ' '.join(n for n, _ in image.transforms))
    for name, params in image.transforms:
        if 'bits' in params:
            out.append('%s_bits %d' % (name, params['bits']))
        if name == 'color_indexing':
            colors = expand_palette(params['image'].pixels)
            for at in range(0, len(colors), 5):
                out.append('palette_colors ' +
                           ' '.join('0x%08x' % c for c in colors[at:at + 5]))
        if 'image' in params:
            key = 'palette' if name == 'color_indexing' else name
            out += sub_lines(key, params['image'])
    if image.cache_bits is not None:
        out.append('cache_bits %d' % image.cache_bits)
    if image.meta_bits is not None:
        out.append('meta_bits %d' % image.meta_bits)
        out += sub_lines('meta', image.meta)
    for group in image.groups:
        out.append('group')
        out += group.lines()
    out += pixel_lines(image.items)
    return '\n'.join(out) + '\n'


def expand_palette(pixels):
    """Inverse of delta_palette(): ExpandColorMap() accumulates per byte."""
    colors = []
    previous = 0
    for pixel in pixels:
        colors.append(sum((((pixel >> at) + (previous >> at)) & 0xff) << at
                          for at in (0, 8, 16, 24)))
        previous = colors[-1]
    return colors


def sub_lines(key, image):
    """One sub-image: its cache, its five codes, and its own pixel data."""
    out = []
    if image.cache_bits is not None:
        out.append('subimage_cache_bits %s %d' % (key, image.cache_bits))
    out.append('subimage %s' % key)
    out += image.groups[0].lines()
    out += pixel_lines(image.items)
    return out


def pixel_lines(items, per_line=8):
    """'pixels' and 'argb' runs, in the order the stream had them."""
    out, run, kind = [], [], None

    def flush():
        if not run:
            return
        for at in range(0, len(run), per_line):
            out.append('%s %s' % (kind, ' '.join(run[at:at + per_line])))
        del run[:]

    for item in items:
        want = 'argb' if item[0] == 'argb' else 'pixels'
        if want != kind:
            flush()
            kind = want
        if item[0] == 'green':
            run.append('0x%02x' % item[1])
        elif item[0] == 'argb':
            run.append('0x%08x' % item[1])
        elif item[0] == 'cache':
            run.append('cache %d' % item[1])
        else:
            run.append('copy %d %d' % (item[1], item[2]))
    flush()
    return out


def check(path, data=None):
    """Disassemble, reassemble, compare. Returns 1 if the bytes moved."""
    if data is None:
        with open(path, 'rb') as f:
            data = f.read()
    want = vp8l_chunk(data)
    try:
        text = dump(want)
    except Truncated as e:
        print('%s: cannot read: %s' % (path, e), file=sys.stderr)
        return 1
    got = vp8l_chunk(vp8l_asm.assemble_text(text))
    if got == want:
        print('%-44s round trip ok, %d bytes' % (path, len(want)))
        return 0
    at = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b),
              min(len(got), len(want)))
    print('%s: reassembles to %d bytes, not %d; first difference at %d'
          % (path, len(got), len(want), at), file=sys.stderr)
    return 1


def main(argv):
    args = [a for a in argv[1:] if not a.startswith('--')]
    if not args:
        print(__doc__.strip().split('\n\n')[0], file=sys.stderr)
        print('usage: %s [--check] <file.webp>...'
              % os.path.basename(argv[0]), file=sys.stderr)
        return 1
    if '--check' in argv:
        return min(1, sum(check(path) for path in args))
    for path in args:
        with open(path, 'rb') as f:
            sys.stdout.write(dump(vp8l_chunk(f.read()),
                                  'disassembled from %s'
                                  % os.path.basename(path)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
