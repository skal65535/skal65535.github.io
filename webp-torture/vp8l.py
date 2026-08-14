# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.

"""Minimal VP8L (WebP lossless) bitstream *writer*, for building torture cases.

Deliberately does no validation: the point is to emit streams that a normal
encoder cannot produce. Bit order and canonical-code layout mirror
src/utils/bit_writer_utils.c and src/utils/huffman_encode_utils.c.
"""

import struct

MAGIC = 0x2F
IMAGE_SIZE_BITS = 14
VERSION_BITS = 3

NUM_LITERAL_CODES = 256
NUM_LENGTH_CODES = 24
NUM_DISTANCE_CODES = 40
CODE_LENGTH_CODES = 19
MAX_ALLOWED_CODE_LENGTH = 15
DEFAULT_CODE_LENGTH = 8
MAX_CACHE_BITS = 11
MIN_HUFFMAN_BITS = 2
NUM_HUFFMAN_BITS = 3
MIN_TRANSFORM_BITS = 2
NUM_TRANSFORM_BITS = 3

PREDICTOR_TRANSFORM = 0
CROSS_COLOR_TRANSFORM = 1
SUBTRACT_GREEN_TRANSFORM = 2
COLOR_INDEXING_TRANSFORM = 3

# src/dec/vp8l_dec.c:63
CODE_LENGTH_CODE_ORDER = [17, 18, 0, 1, 2, 3, 4, 5, 16,
                          6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def alphabet_size(index, cache_bits=0):
    """Alphabet size of the 'index'th code of a meta code (GREEN..DIST)."""
    sizes = [NUM_LITERAL_CODES + NUM_LENGTH_CODES,
             NUM_LITERAL_CODES, NUM_LITERAL_CODES, NUM_LITERAL_CODES,
             NUM_DISTANCE_CODES]
    n = sizes[index]
    if index == 0 and cache_bits > 0:
        n += 1 << cache_bits
    return n


def sub_sample_size(size, sampling_bits):
    return (size + (1 << sampling_bits) - 1) >> sampling_bits


class BitWriter:
    """LSB-first, exactly like VP8LPutBits()."""

    def __init__(self):
        self.bits = []

    def put(self, value, n):
        for i in range(n):
            self.bits.append((value >> i) & 1)

    def to_bytes(self):
        out = bytearray((len(self.bits) + 7) // 8)
        for i, b in enumerate(self.bits):
            if b:
                out[i >> 3] |= 1 << (i & 7)
        return bytes(out)


def reverse_bits(value, n):
    r = 0
    for i in range(n):
        r = (r << 1) | ((value >> i) & 1)
    return r


def canonical_codes(lengths):
    """Canonical Huffman codes, bit-reversed.

    cf. ConvertBitDepthsToSymbols() in huffman_encode_utils.c.
    """
    depth_count = [0] * (MAX_ALLOWED_CODE_LENGTH + 1)
    for l in lengths:
        depth_count[l] += 1
    depth_count[0] = 0
    next_code = [0] * (MAX_ALLOWED_CODE_LENGTH + 1)
    code = 0
    for i in range(1, MAX_ALLOWED_CODE_LENGTH + 1):
        code = (code + depth_count[i - 1]) << 1
        next_code[i] = code
    codes = []
    for l in lengths:
        if l == 0:
            codes.append(0)
        else:
            codes.append(reverse_bits(next_code[l], l))
            next_code[l] += 1
    return codes


def huffman_lengths(freqs, max_len=15):
    """Plain Huffman code lengths from frequencies. Small alphabets only.

    Returns a list of lengths, 0 for unused symbols. A single used symbol gets
    length 1 -- that is what the bitstream must say; the decoder then makes the
    code 0 bits wide via BuildHuffmanTable()'s single-value shortcut.
    """
    used = [i for i, f in enumerate(freqs) if f > 0]
    lengths = [0] * len(freqs)
    if len(used) == 0:
        return lengths
    if len(used) == 1:
        lengths[used[0]] = 1
        return lengths
    nodes = [[freqs[i], [i]] for i in used]
    while len(nodes) > 1:
        nodes.sort(key=lambda n: (n[0], min(n[1])))
        a, b = nodes.pop(0), nodes.pop(0)
        for s in a[1] + b[1]:
            lengths[s] += 1
        nodes.append([a[0] + b[0], a[1] + b[1]])
    assert max(lengths) <= max_len, 'code too deep: %d' % max(lengths)
    return lengths


class Huffman:
    """One Huffman code, ready to be written and to encode symbols."""

    def __init__(self, lengths):
        self.lengths = list(lengths)
        self.codes = canonical_codes(self.lengths)
        self.trivial = sum(1 for l in self.lengths if l > 0) <= 1

    @classmethod
    def from_freqs(cls, freqs):
        return cls(huffman_lengths(freqs))

    @classmethod
    def single(cls, symbol, size):
        """A code with one symbol, decoded with 0 bits."""
        lengths = [0] * size
        lengths[symbol] = 1
        return cls(lengths)

    def emit_symbol(self, bw, symbol):
        n = self.lengths[symbol]
        assert n > 0 or self.trivial, 'symbol %d not in code' % symbol
        if n > 0 and not self.trivial:
            bw.put(self.codes[symbol], n)


def write_simple_code(bw, num_symbols, symbols, use_8bit=True):
    """The 'simple code' form. No range checking: that is the whole point."""
    bw.put(1, 1)                    # simple code marker
    bw.put(num_symbols - 1, 1)
    if use_8bit or symbols[0] > 1:
        bw.put(1, 1)                # first symbol is 8 bits
        bw.put(symbols[0], 8)
    else:
        bw.put(0, 1)                # first symbol is 1 bit
        bw.put(symbols[0], 1)
    if num_symbols == 2:
        bw.put(symbols[1], 8)


def code_length_symbols(lengths, use_repeats=False):
    """Turn a length array into the (symbol, extra_bits, extra_value) stream."""
    out = []
    if not use_repeats:
        for l in lengths:
            out.append((l, 0, 0))
        return out
    i = 0
    n = len(lengths)
    while i < n:
        l = lengths[i]
        run = 1
        while i + run < n and lengths[i + run] == l:
            run += 1
        if l == 0 and run >= 11:
            run = min(run, 138)
            out.append((18, 7, run - 11))
        elif l == 0 and run >= 3:
            out.append((17, 3, run - 3))
        elif l != 0 and run >= 4 and out:
            out.append((l, 0, 0))          # one literal first
            rest = min(run - 1, 6)
            out.append((16, 2, rest - 3))
            run = 1 + rest
        else:
            for _ in range(run):
                out.append((l, 0, 0))
        i += run
    return out


def write_complex_code(bw, lengths, use_repeats=False, num_codes=None,
                       max_symbol=None, raw_symbols=None):
    """The 'normal' form: a Huffman code over the code lengths themselves."""
    syms = raw_symbols if raw_symbols is not None else \
        code_length_symbols(lengths, use_repeats)
    freqs = [0] * CODE_LENGTH_CODES
    for s, _, _ in syms:
        freqs[s] += 1
    cl_lengths = huffman_lengths(freqs, max_len=7)
    cl = Huffman(cl_lengths)
    if num_codes is None:
        num_codes = CODE_LENGTH_CODES
        while (num_codes > 4 and
               cl_lengths[CODE_LENGTH_CODE_ORDER[num_codes - 1]] == 0):
            num_codes -= 1
    bw.put(0, 1)                    # not a simple code
    bw.put(num_codes - 4, 4)
    for i in range(num_codes):
        bw.put(cl_lengths[CODE_LENGTH_CODE_ORDER[i]], 3)
    if max_symbol is None:
        bw.put(0, 1)                # no explicit max_symbol
    else:
        bw.put(1, 1)
        length_nbits, raw = 2, 0
        while (1 << length_nbits) <= max_symbol - 2:
            raw += 1
            length_nbits += 2
        bw.put(raw, 3)
        bw.put(max_symbol - 2, length_nbits)
    for s, extra_bits, extra_val in syms:
        cl.emit_symbol(bw, s)
        if extra_bits:
            bw.put(extra_val, extra_bits)
    return cl


def write_code(bw, huff):
    """Emit a Huffman code, using the simple form when it fits."""
    used = [i for i, l in enumerate(huff.lengths) if l > 0]
    if len(used) == 0:
        write_simple_code(bw, 1, [0])
    elif len(used) == 1 and used[0] < 256:
        write_simple_code(bw, 1, [used[0]])
    elif len(used) == 2 and used[0] < 256 and used[1] < 256 and \
            huff.lengths[used[0]] == huff.lengths[used[1]]:
        write_simple_code(bw, 2, used)
    else:
        write_complex_code(bw, huff.lengths)


def write_subimage(bw, pixels):
    """A complete non-level0 image stream holding exactly 'pixels' (ARGB)."""
    # No transform bit and no meta-huffman bit here: DecodeImageStream() only
    # reads transforms at level 0, and ReadHuffmanCodes() short-circuits on
    # 'allow_recursion &&', so the meta bit is not in the stream either.
    bw.put(0, 1)  # no color cache
    greens = [(p >> 8) & 0xff for p in pixels]
    reds = [(p >> 16) & 0xff for p in pixels]
    blues = [p & 0xff for p in pixels]
    alphas = [(p >> 24) & 0xff for p in pixels]
    codes = []
    for vals, size in ((greens, alphabet_size(0)), (reds, 256),
                       (blues, 256), (alphas, 256)):
        freqs = [0] * size
        for v in vals:
            freqs[v] += 1
        codes.append(Huffman.from_freqs(freqs))
    codes.append(Huffman.single(0, NUM_DISTANCE_CODES))   # distances
    for c in codes:
        write_code(bw, c)
    trivial_literal = all(c.trivial for c in codes[1:4])
    for i in range(len(pixels)):
        codes[0].emit_symbol(bw, greens[i])
        if not trivial_literal:
            codes[1].emit_symbol(bw, reds[i])
            codes[2].emit_symbol(bw, blues[i])
            codes[3].emit_symbol(bw, alphas[i])


def delta_palette(colors):
    """ExpandColorMap() accumulates entries, so store per-byte deltas."""
    out = [colors[0]]
    for i in range(1, len(colors)):
        d = 0
        for shift in (0, 8, 16, 24):
            b = (((colors[i] >> shift) & 0xff) -
                 ((colors[i - 1] >> shift) & 0xff))
            d |= (b & 0xff) << shift
        out.append(d)
    return out


def palette_bits(num_colors):
    """How many indices a packed byte holds, as a log2 of the palette size."""
    return 0 if num_colors > 16 else 1 if num_colors > 4 else \
        2 if num_colors > 2 else 3


def write_header(bw, width, height, alpha=0, version=0):
    bw.put(MAGIC, 8)
    bw.put(width - 1, IMAGE_SIZE_BITS)
    bw.put(height - 1, IMAGE_SIZE_BITS)
    bw.put(alpha, 1)
    bw.put(version, VERSION_BITS)


def wrap_webp(vp8l_payload):
    """RIFF/WEBP container around a raw VP8L stream."""
    chunk = b'VP8L' + struct.pack('<I', len(vp8l_payload)) + vp8l_payload
    if len(vp8l_payload) & 1:
        chunk += b'\0'
    riff_size = 4 + len(chunk)
    return b'RIFF' + struct.pack('<I', riff_size) + b'WEBP' + chunk


# src/dec/vp8l_dec.c:67 -- extracted from the source, not retyped.
CODE_TO_PLANE = [
    24, 7, 23, 25, 40, 6, 39, 41, 22, 26, 38, 42, 56, 5, 55, 57, 21, 27,
    54, 58, 37, 43, 72, 4, 71, 73, 20, 28, 53, 59, 70, 74, 36, 44, 88, 69,
    75, 52, 60, 3, 87, 89, 19, 29, 86, 90, 35, 45, 68, 76, 85, 91, 51, 61,
    104, 2, 103, 105, 18, 30, 102, 106, 34, 46, 84, 92, 67, 77, 101, 107,
    50, 62, 120, 1, 119, 121, 83, 93, 17, 31, 100, 108, 66, 78, 118, 122,
    33, 47, 117, 123, 49, 63, 99, 109, 82, 94, 0, 116, 124, 65, 79, 16, 32,
    98, 110, 48, 115, 125, 81, 95, 64, 114, 126, 97, 111, 80, 113, 127, 96,
    112,
]
CODE_TO_PLANE_CODES = 120


def prefix_code(value):
    """Inverse of GetCopyDistance()/GetCopyLength().

    Returns (symbol, extra_bits, extra_value) such that the decoder recovers
    'value'. Lengths and distances share this encoding.
    """
    if value <= 4:
        return value - 1, 0, 0
    for symbol in range(4, NUM_DISTANCE_CODES):
        extra_bits = (symbol - 2) >> 1
        offset = (2 + (symbol & 1)) << extra_bits
        extra = value - offset - 1
        if 0 <= extra < (1 << extra_bits):
            return symbol, extra_bits, extra
    raise ValueError('no prefix code for %d' % value)


def plane_code_to_distance(xsize, plane_code):
    """Mirrors PlaneCodeToDistance() at src/dec/vp8l_dec.c:164."""
    if plane_code > CODE_TO_PLANE_CODES:
        return plane_code - CODE_TO_PLANE_CODES
    dist_code = CODE_TO_PLANE[plane_code - 1]
    yoffset = dist_code >> 4
    xoffset = 8 - (dist_code & 0xf)
    dist = yoffset * xsize + xoffset
    return dist if dist >= 1 else 1
