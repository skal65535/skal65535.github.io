#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.

"""Assembles a lossless VP8L .webp from a text description of the bitstream.

    ./vp8l_asm.py cases/cache-bits-11.txt files/cache-bits-11.webp

The lossless half of vp8_asm.py, and the same idea: a list of fields, one
per line, '#' starting a comment, every field with a default, so a case says
only what it is about. Nothing is validated or clamped: a value too big for
its field just loses its top bits, which is usually the point. A case is
marked lossless by the bare directive 'lossless'; without it the case is a
lossy frame and vp8_asm.py owns it.

Fields are written in the order the format carries them -- the header, the
transforms, the color cache and entropy image, the Huffman codes, then the
pixel data -- so a case reads in that order too.

  lossless                  this case is a VP8L image, not a VP8 frame
  magic 0x2f                the signature byte VP8LCheckSignature() tests
  width 1                   14 bits, as is height
  height 1
  version 0                 3 bits; anything but 0 is refused
  alpha_is_used 0
  transforms NAME...        the transforms, in the order written. A type may
                            legally appear once; listing one twice is what a
                            case is for. Names are predictor, cross_color,
                            subtract_green and color_indexing
  predictor_bits 2          log2 of the tile size, 2..9
  predictor_tiles M...      one predictor index per tile
  cross_color_bits 2
  cross_color_tiles A...    one multiplier triple per tile, as a color: the
                            three live in its green, red and blue bytes
  palette_colors ARGB...    color_indexing's palette, written as colors: the
                            stream carries the per-byte deltas of these
  cache_bits N              the color cache, absent by default. 0 means
                            "present, and zero bits wide", which is not the
                            same thing and is not legal either
  meta_bits N               the entropy image, absent by default
  meta_tiles G...           one Huffman-group index per tile
  subimage NAME             aims the 'code', 'pixels' and 'argb' lines that
                            follow at one of the sub-images: predictor,
                            cross_color, palette or meta. Each is an image
                            stream of its own, so it has its own five codes
                            and its own pixel data -- back-references and
                            all, which is the only way to say what a real
                            encoder writes there. 'group' aims them back at
                            the image itself
  subimage_cache_bits NAME N   and its own color cache, absent by default
  code NAME ...             one Huffman code of the group being filled in
  group                     opens another group of codes
  group_count N             how many groups to write, when that should differ
                            from what the entropy image asks for
  pixels ITEM...            what the green and distance codes encode
  argb V...                 whole pixels, spelled as colors: every literal
                            code carries one. 'V xN' repeats

Repeating a directive appends to it, so a long list of tiles or colors can
be laid out over as many lines as suits.

Each group holds five codes, written green, red, blue, alpha then dist, and
a case names the form it wants each one in:

  code green 0x20           simple form, one symbol
  code green 0x10 0x20      simple form, two symbols
  code green simple1 1      simple form, the first symbol in 1 bit, not 8
  code green lengths L...   the normal form, from an explicit length per
                            symbol, positional from symbol 0. 'N:L' jumps to
                            symbol N first; trailing zeros are implied
  code green codelen S...   the normal form, spelling out the code-length
                            symbol stream itself: a length 0..15, or 16xN to
                            repeat the previous length N times, or 17xN and
                            18xN for N zeros
  code green cl_lengths S:L...  the code-length code itself, as symbol:length
                            pairs; without it one is built from how often
                            each code-length symbol is used
  code green num_codes N    how many of the 19 code-length codes to declare
  code green max_symbol N   the optional early stop
  code green complex        the normal form, over whatever the pixels need

'lengths' and 'codelen' are two halves of one thing, and a case may give
either or both: 'lengths' is what the code *is*, and so what the pixel data
is encoded with, while 'codelen' is what the stream *says* it is. Those
agree in a well-formed file, which is why most cases give only one.

A code the case says nothing about covers exactly what the pixel data asks
of it, in whichever form fits. So green and dist follow 'pixels', and red,
blue and alpha fall back to a single symbol each -- 0x40, 0x60 and 0xff,
with green's 0x20 -- which costs no bits at all.

'pixels' lists symbols of the green code, in order:

  V                         a literal, or any green symbol; 'V xN' repeats it
  cache N                   the color-cache index N
  copy LENGTH PLANE         a back-reference, spelled as the length and the
                            plane code themselves; the symbols and their
                            extra bits follow from those

'argb' appends to the same stream, so the two interleave in the order the
file gives them. Which of the two a case wants follows the format: the
decoder reads the red, blue and alpha codes only when they do not all hold a
single symbol, so a bare green symbol is a whole pixel exactly when they do,
and 'argb' is needed as soon as one of them does not. Writing every pixel
with 'argb' and letting the codes fall out of the colors is what an ordinary
image looks like here.

Which group codes a pixel follows the entropy image, tile by tile, so a
'pixels' or 'argb' list crosses from one group's codes to another's exactly
where the tiles say it does. Without an entropy image there is one group and
the question does not arise.
"""

import os
import sys

import vp8l
import vp8_asm
from vp8_asm import AsmError

# The five codes of a group, in the order the stream carries them, and the
# symbol each falls back to when nothing asks anything of it.
CODE_NAMES = ('green', 'red', 'blue', 'alpha', 'dist')
CODE_DEFAULTS = (0x20, 0x40, 0x60, 0xff, 0)

# The four images carried inside a level-0 one. Each is a stream in its own
# right -- its own color cache and its own five codes -- but with no
# transforms and no entropy image of its own.
SUBIMAGES = ('predictor', 'cross_color', 'palette', 'meta')

TRANSFORMS = {'predictor': vp8l.PREDICTOR_TRANSFORM,
              'cross_color': vp8l.CROSS_COLOR_TRANSFORM,
              'subtract_green': vp8l.SUBTRACT_GREEN_TRANSFORM,
              'color_indexing': vp8l.COLOR_INDEXING_TRANSFORM}

# The repeat escapes of the code-length code: symbol -> (extra bits, the
# count an all-zero extra field stands for).
REPEATS = {16: (2, 3), 17: (3, 3), 18: (7, 11)}

# name -> attribute, for the fields that are one plain number.
NUM_FIELDS = {'width': 'width', 'height': 'height', 'version': 'version',
              'alpha_is_used': 'alpha', 'predictor_bits': 'predictor_bits',
              'cross_color_bits': 'cross_color_bits',
              'cache_bits': 'cache_bits', 'meta_bits': 'meta_bits',
              'group_count': 'group_count', 'magic': 'magic'}
# name -> attribute, for the fields that are a list a repeat appends to.
LIST_FIELDS = {'predictor_tiles': 'predictor_tiles',
               'cross_color_tiles': 'cross_color_tiles',
               'meta_tiles': 'meta_tiles',
               'palette_colors': 'palette_colors',
               'transforms': 'transforms'}


def unroll(items):
    """The item stream one pixel at a time, repeats spelled out."""
    for item in items:
        for _ in range(item[2] if item[0] in ('green', 'argb') else 1):
            yield (item[0], item[1], 1) if item[0] in ('green', 'argb') \
                else item


def symbols(item):
    """(code index, symbol, extra bits, extra value) for what one item
    writes. A back-reference is the only one carrying extra bits."""
    if item[0] == 'green':
        return [(0, item[1], 0, 0)]
    if item[0] == 'argb':
        return [(i, (item[1] >> at) & 0xff, 0, 0)
                for i, at in enumerate(vp8l.CHANNEL_SHIFTS)]
    if item[0] == 'cache':
        return [(0, vp8l.CACHE_BASE + item[1], 0, 0)]
    length, plane = (vp8l.prefix_code(item[1]), vp8l.prefix_code(item[2]))
    return [(0, vp8l.NUM_LITERAL_CODES + length[0], length[1], length[2]),
            (4, plane[0], plane[1], plane[2])]


def item_freqs(items, cache_bits=0):
    """What one pixel-item stream asks of the five codes of a group."""
    freqs = [[0] * vp8l.alphabet_size(i, cache_bits) for i in range(5)]
    for item in unroll(items):
        for index, symbol, _bits, _extra in symbols(item):
            table = freqs[index]
            if symbol >= len(table):
                raise AsmError('pixels: symbol %d, past the %d the alphabet '
                               'holds' % (symbol, len(table)))
            table[symbol] += 1
    return freqs


def literal_arb(huffs):
    """Whether red, blue and alpha are one symbol each, and which pixel
    that makes. cf. is_trivial_literal in ReadHuffmanCodes()."""
    if not all(h.trivial for h in huffs[1:4]):
        return False, 0
    arb = 0
    for huff, at in zip(huffs[1:4], vp8l.CHANNEL_SHIFTS[1:]):
        arb |= next(s for s, l in enumerate(huff.lengths) if l) << at
    return True, arb


def copy_into(pixels, length, plane_code, xsize):
    """Repeat what a back-reference points at, so later values are known."""
    distance = vp8l.plane_code_to_distance(xsize, plane_code)
    for _ in range(length):
        pixels.append(pixels[-distance] if distance <= len(pixels) else 0)


class Code:
    """One Huffman code of a group: how to write it, and what it encodes."""

    def __init__(self, index):
        self.index = index          # 0..4, which alphabet it is over
        self.simple = None          # symbols of the simple form
        self.short = False          # first symbol in 1 bit rather than 8
        self.lengths = None         # {symbol: length}
        self.codelen = None         # [(symbol, extra bits, extra value)]
        self.num_codes = None
        self.max_symbol = None
        self.cl_lengths = None      # the code-length code, spelled out
        self.derive = False         # 'complex': normal form, from the pixels

    @property
    def spelled(self):
        """Whether the case says what the code is, rather than the pixels."""
        return (self.lengths is not None or self.codelen is not None or
                self.num_codes is not None or self.max_symbol is not None or
                self.cl_lengths is not None)

    def write(self, bw, cache_bits, freqs):
        """Emits the code, and returns the Huffman that encodes its symbols."""
        size = vp8l.alphabet_size(self.index, cache_bits)
        if self.spelled:
            lengths = [0] * size
            for symbol, length in sorted((self.lengths or {}).items()):
                if symbol >= size:
                    raise AsmError('length of symbol %d, past the %d the '
                                   'alphabet holds' % (symbol, size))
                lengths[symbol] = length
            vp8l.write_complex_code(bw, lengths, use_repeats=True,
                                    num_codes=self.num_codes,
                                    max_symbol=self.max_symbol,
                                    raw_symbols=self.codelen,
                                    cl_lengths=self.cl_lengths)
            return vp8l.Huffman(lengths)
        if self.simple is not None:
            vp8l.write_simple_code(bw, len(self.simple), self.simple,
                                   use_8bit=not self.short)
            # Symbols past the alphabet are written to the stream but leave
            # no code behind, which is the whole point of those cases.
            lengths = [0] * size
            for symbol in self.simple:
                if symbol < size:
                    lengths[symbol] = 1
            return vp8l.Huffman(lengths)
        huff = vp8l.Huffman.from_freqs(freqs)
        if self.derive:
            vp8l.write_complex_code(bw, huff.lengths, use_repeats=True)
        else:
            vp8l.write_code(bw, huff)
        return huff


class Group:
    """The five codes one Huffman group holds."""

    def __init__(self):
        self.codes = [Code(i) for i in range(len(CODE_NAMES))]


class Image:
    """Everything a case says, before any of it is written."""

    def __init__(self):
        self.width = 1
        self.height = 1
        self.version = 0
        self.alpha = 0
        self.transforms = []
        self.predictor_bits = vp8l.MIN_TRANSFORM_BITS
        self.predictor_tiles = None
        self.cross_color_bits = vp8l.MIN_TRANSFORM_BITS
        self.cross_color_tiles = None
        self.palette_colors = None
        self.magic = vp8l.MAGIC
        self.sub_cache = dict.fromkeys(SUBIMAGES)     # name -> cache_bits
        self.sub_groups = {}                          # name -> Group
        self.sub_items = {}                           # name -> pixel items
        self.tiles = None            # group per tile, once meta is written
        self.cache_bits = None
        self.meta_bits = None
        self.meta_tiles = None
        self.group_count = None
        self.groups = []
        self.pixels = []

    def tile_groups(self):
        """The group index per tile: resolved from the entropy image once it
        is written, and whatever 'meta_tiles' said until then."""
        return self.tiles if self.tiles is not None else self.meta_tiles

    def walk(self, xsize):
        """(group, item) for every pixel, in order, repeats spelled out."""
        tiles = self.tile_groups()
        at = 0
        for item in unroll(self.pixels):
            group = 0 if self.meta_bits is None or not tiles else \
                vp8l.tile_at(tiles, at, xsize, self.meta_bits)
            yield group, item
            at += item[1] if item[0] == 'copy' else 1

    def freqs(self, xsize, num_groups):
        """What the pixel data asks of each group's five codes."""
        buckets = [[] for _ in range(max(num_groups, 1))]
        for group, item in self.walk(xsize):
            if group >= len(buckets):
                raise AsmError('pixels: the entropy image asks for group %d, '
                               'but %d are written' % (group, len(buckets)))
            buckets[group].append(item)
        return [item_freqs(b, self.cache_bits or 0) for b in buckets]

    def num_groups(self):
        if self.group_count is not None:
            return self.group_count
        tiles = self.tile_groups()
        return max(max(tiles) + 1 if tiles else 1, len(self.groups))


class Assembler:
    """Reads a case, writes the VP8L chunk it describes."""

    def __init__(self):
        self.img = Image()
        self.group = None
        self.items = self.img.pixels     # what 'pixels'/'argb' fill in
        self.line = 0

    def fail(self, msg):
        raise AsmError('line %d: %s' % (self.line, msg))

    def num(self, tok, what='value'):
        try:
            return int(tok, 0)
        except ValueError:
            self.fail('bad %s %r' % (what, tok))

    def one(self, args, what):
        return self.take(args, 1, what)[0]

    def take(self, args, n, what):
        if len(args) != n:
            self.fail('%s takes %d value(s), got %d' % (what, n, len(args)))
        return args

    # -- the codes of a group --------------------------------------------

    def current_group(self):
        if self.group is None:
            self.do_group([])
        return self.group

    def do_group(self, args):
        if args:
            self.fail('group takes no value')
        self.group = Group()
        self.img.groups.append(self.group)
        self.items = self.img.pixels

    def do_subimage(self, args):
        """Aims the 'code' and pixel lines that follow at one sub-image."""
        name = self.one(args, 'subimage')
        if name not in SUBIMAGES:
            self.fail('subimage names one of %s' % ', '.join(SUBIMAGES))
        self.group = self.img.sub_groups.setdefault(name, Group())
        self.items = self.img.sub_items.setdefault(name, [])

    def do_subimage_cache_bits(self, args):
        name, bits = self.take(args, 2, 'subimage_cache_bits')
        if name not in SUBIMAGES:
            self.fail('subimage_cache_bits names one of %s'
                      % ', '.join(SUBIMAGES))
        self.img.sub_cache[name] = self.num(bits, 'cache_bits')

    def do_code(self, args):
        if not args or args[0] not in CODE_NAMES:
            self.fail('code names one of %s' % ', '.join(CODE_NAMES))
        name, args = args[0], args[1:]
        code = self.current_group().codes[CODE_NAMES.index(name)]
        if not args:
            self.fail('code %s needs a value or a form' % name)
        form, rest = args[0], args[1:]
        if form == 'lengths':
            code.lengths = code.lengths or {}
            at = 0
            for tok in rest:
                if ':' in tok:
                    where, _, tok = tok.partition(':')
                    at = self.num(where, 'symbol')
                code.lengths[at] = self.num(tok, 'length')
                at += 1
        elif form == 'codelen':
            code.codelen = (code.codelen or []) + \
                [self.codelen_symbol(tok) for tok in rest]
        elif form == 'cl_lengths':
            code.cl_lengths = [0] * vp8l.CODE_LENGTH_CODES
            for tok in rest:
                where, _, value = tok.partition(':')
                code.cl_lengths[self.num(where, 'symbol')] = \
                    self.num(value, 'length')
        elif form in ('num_codes', 'max_symbol'):
            setattr(code, form, self.num(self.one(rest, form), form))
        elif form == 'complex':
            if rest:
                self.fail('code %s complex takes no value' % name)
            code.derive = True
        else:
            code.short = form == 'simple1'
            if code.short:
                args = rest
            if not 1 <= len(args) <= 2:
                self.fail('code %s takes one or two symbols, got %d'
                          % (name, len(args)))
            code.simple = [self.num(t, 'symbol') for t in args]

    def codelen_symbol(self, tok):
        """One entry of a code-length stream: a length, or an NxCOUNT run."""
        symbol, _, count = tok.partition('x')
        symbol = self.num(symbol, 'code-length symbol')
        if not count:
            return (symbol, 0, 0)
        if symbol not in REPEATS:
            self.fail('code-length symbol %d carries no count' % symbol)
        bits, base = REPEATS[symbol]
        extra = self.num(count, 'count') - base
        return (symbol, bits, extra & ((1 << bits) - 1))

    # -- the pixel data --------------------------------------------------

    def do_argb(self, args):
        """Whole pixels, spelled as colors: every literal code carries one."""
        i = 0
        while i < len(args):
            value, i = self.num(args[i], 'color'), i + 1
            repeat = 1
            if i < len(args) and args[i].startswith('x'):
                repeat, i = self.num(args[i][1:], 'repeat'), i + 1
            self.items.append(('argb', value, repeat))

    def do_pixels(self, args):
        i = 0
        while i < len(args):
            tok = args[i]
            i += 1
            if tok == 'copy':
                if i + 2 > len(args):
                    self.fail('copy needs a length and a plane code')
                self.items.append(('copy', self.num(args[i], 'length'),
                                   self.num(args[i + 1], 'plane code')))
                i += 2
            elif tok == 'cache':
                if i == len(args):
                    self.fail('cache needs an index')
                self.items.append(('cache', self.num(args[i], 'index')))
                i += 1
            else:
                repeat = 1
                if i < len(args) and args[i].startswith('x'):
                    repeat = self.num(args[i][1:], 'repeat')
                    i += 1
                self.items.append(
                    ('green', self.num(tok, 'symbol'), repeat))

    def do_lossless(self, args):
        if args:
            self.fail('lossless takes no value')

    # -- driving ---------------------------------------------------------

    def feed(self, text):
        for lineno, line in enumerate(text.splitlines(), 1):
            self.line = lineno
            args = line.split('#')[0].split()
            if not args:
                continue
            name, rest = args[0], args[1:]
            if name in NUM_FIELDS:
                setattr(self.img, NUM_FIELDS[name],
                        self.num(self.one(rest, name), name))
            elif name in LIST_FIELDS:
                attr = LIST_FIELDS[name]
                values = rest if name == 'transforms' else \
                    [self.num(t, name) for t in rest]
                setattr(self.img, attr, (getattr(self.img, attr) or []) +
                        values)
            else:
                handler = getattr(self, 'do_' + name, None)
                if handler is None:
                    self.fail('unknown field %r' % name)
                handler(rest)
        for name in self.img.transforms:
            if name not in TRANSFORMS:
                raise AsmError('unknown transform %r' % name)
        return self.img

    def tiles(self, given, xsize, bits, what, default=0):
        """One value per tile, with a default, checked against the count."""
        tw = vp8l.sub_sample_size(xsize, bits)
        th = vp8l.sub_sample_size(self.img.height, bits)
        if given is None:
            return [default] * (tw * th)
        if len(given) != tw * th:
            raise AsmError('%s has %d values, %dx%d tiles need %d'
                           % (what, len(given), tw, th, tw * th))
        return given

    def subimage(self, bw, name, pixels, xsize):
        """A complete non-level0 image stream, and the pixels it comes to.

        No transform bit and no meta-Huffman bit, but a color cache and five
        codes of its own, and its own pixel items when the case gave any.
        """
        cache_bits = self.img.sub_cache[name]
        group = self.img.sub_groups.get(name) or Group()
        items = self.img.sub_items.get(name) or \
            [('argb', p, 1) for p in pixels]
        if cache_bits is None:
            bw.put(0, 1)
        else:
            bw.put(1, 1)
            bw.put(cache_bits, 4)
        freqs = item_freqs(items, cache_bits or 0)
        huffs = []
        for i, code in enumerate(group.codes):
            want = freqs[i]
            if not any(want):
                want = list(want)
                want[CODE_DEFAULTS[i]] = 1
            huffs.append(code.write(bw, cache_bits or 0, want))
        return self.emit_items(bw, [huffs],
                               ((0, item) for item in unroll(items)),
                               xsize, name)

    def emit_items(self, bw, groups, walk, xsize, what):
        """Writes one pixel stream, and returns the pixels it decodes to.

        'walk' pairs each pixel with the group that codes it, which is how
        an entropy image changes codes part-way along a row.
        """
        literals = [literal_arb(huffs) for huffs in groups]
        out = []
        for n, item in walk:
            huffs, (trivial, arb) = groups[n], literals[n]
            if item[0] == 'green' and not trivial:
                raise AsmError('%s: symbol %d leaves red, blue and alpha '
                               'unsaid; spell the whole pixel with "argb"'
                               % (what, item[1]))
            for index, symbol, bits, extra in symbols(item):
                if 1 <= index <= 3 and trivial:
                    continue          # the decoder does not read those three
                self.emit(bw, huffs[index], symbol, what)
                if bits:
                    bw.put(extra, bits)
            if item[0] == 'copy':
                copy_into(out, item[1], item[2], xsize)
            elif item[0] == 'green':
                out.append(arb | (item[1] << 8))
            elif item[0] == 'argb':
                out.append(item[1])
            else:
                out.append(0)         # a cache index: only the index matters
        return out

    def emit(self, bw, huff, symbol, what):
        if not huff.trivial and not huff.lengths[symbol]:
            raise AsmError('%s: the code declared has no symbol %d to write '
                           'the data with' % (what, symbol))
        huff.emit_symbol(bw, symbol)

    def write_transforms(self, bw):
        """Every transform in turn; returns the width left behind."""
        img = self.img
        xsize = img.width
        for name in img.transforms:
            bw.put(1, 1)
            bw.put(TRANSFORMS[name], 2)
            if name == 'color_indexing':
                colors = img.palette_colors or [0xff000000, 0xff00ff00]
                bw.put(len(colors) - 1, 8)
                self.subimage(bw, 'palette', vp8l.delta_palette(colors),
                              len(colors))
                xsize = vp8l.sub_sample_size(xsize,
                                             vp8l.palette_bits(len(colors)))
            elif name == 'predictor':
                bw.put(img.predictor_bits - vp8l.MIN_TRANSFORM_BITS,
                       vp8l.NUM_TRANSFORM_BITS)
                modes = self.tiles(img.predictor_tiles, xsize,
                                   img.predictor_bits, 'predictor_tiles')
                # the predictor index is the green byte of its tile
                self.subimage(bw, 'predictor',
                              [0xff000000 | ((m & 0xff) << 8) for m in modes],
                              vp8l.sub_sample_size(xsize, img.predictor_bits))
            elif name == 'cross_color':
                bw.put(img.cross_color_bits - vp8l.MIN_TRANSFORM_BITS,
                       vp8l.NUM_TRANSFORM_BITS)
                self.subimage(
                    bw, 'cross_color',
                    self.tiles(img.cross_color_tiles, xsize,
                               img.cross_color_bits, 'cross_color_tiles',
                               default=0xff000000),
                    vp8l.sub_sample_size(xsize, img.cross_color_bits))
        bw.put(0, 1)                     # no more transforms
        return xsize

    def finish(self, header=True):
        """The assembled VP8L chunk.

        An alpha plane is the same stream without the five-byte header:
        VP8LDecodeAlphaHeader() knows the dimensions already and reads
        straight into DecodeImageStream().
        """
        img = self.img
        bw = vp8l.BitWriter()
        if header:
            vp8l.write_header(bw, img.width, img.height, img.alpha,
                              img.version, img.magic)
        xsize = self.write_transforms(bw)
        if img.cache_bits is None:
            bw.put(0, 1)
        else:
            bw.put(1, 1)
            bw.put(img.cache_bits, 4)
        if img.meta_bits is None:
            bw.put(0, 1)
        else:
            bw.put(1, 1)
            bw.put(img.meta_bits - vp8l.MIN_HUFFMAN_BITS,
                   vp8l.NUM_HUFFMAN_BITS)
            tiles = self.tiles(img.meta_tiles, xsize, img.meta_bits,
                               'meta_tiles')
            # the group index lives in the red (high) and green (low) bytes
            pixels = self.subimage(bw, 'meta',
                                   [((g >> 8) << 16) | ((g & 0xff) << 8)
                                    for g in tiles],
                                   vp8l.sub_sample_size(xsize, img.meta_bits))
            img.tiles = [(p >> 8) & 0xffff for p in pixels]
        count = img.num_groups()
        freqs = img.freqs(xsize, count)
        huffs = []
        for n in range(count):
            group = img.groups[n] if n < len(img.groups) else Group()
            built = []
            for i, code in enumerate(group.codes):
                # A code nothing asks anything of falls back to its one
                # default symbol, which costs no bits at all.
                want = freqs[n][i] if n < len(freqs) else \
                    [0] * vp8l.alphabet_size(i, img.cache_bits or 0)
                if not any(want):
                    want = list(want)
                    want[CODE_DEFAULTS[i]] = 1
                built.append(code.write(bw, img.cache_bits or 0, want))
            huffs.append(built)
        self.write_pixels(bw, huffs, xsize)
        return bw.to_bytes()

    def write_pixels(self, bw, huffs, xsize):
        """Level 0, where the entropy image may change codes mid-row."""
        if not self.img.pixels:
            return
        if not huffs:
            raise AsmError('pixels, but the case writes no Huffman codes')
        self.emit_items(bw, huffs, self.img.walk(xsize), xsize, 'pixels')


def is_lossless(text):
    """Whether a case describes a VP8L image rather than a VP8 frame."""
    return any(line.split('#')[0].split()[:1] == ['lossless']
               for line in text.splitlines())


def assemble_text(text):
    """The .webp bytes for one case."""
    asm = Assembler()
    try:
        asm.feed(text)
        return vp8l.wrap_webp(asm.finish())
    except vp8_asm.UNWRITABLE as e:
        raise vp8_asm.unwritable(e)


def main(argv):
    return vp8_asm.assemble_main(argv, assemble_text, __doc__)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
