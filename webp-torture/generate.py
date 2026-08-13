#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.
"""Generates the torture bitstreams. Run: python3 generate.py [outdir]

Each case() call registers one file plus the note that goes into README.md.
'expect' is what the reference decoder is supposed to do: 'ok' or 'reject'.
"""

import glob
import os
import re
import sys
import textwrap

import lossy_parts
import vp8_asm
import webp_asm
from vp8l import (BitWriter, Huffman, alphabet_size, plane_code_to_distance,
                  prefix_code, sub_sample_size, write_complex_code,
                  write_header, write_simple_code, wrap_webp,
                  COLOR_INDEXING_TRANSFORM, CROSS_COLOR_TRANSFORM,
                  MIN_HUFFMAN_BITS, MIN_TRANSFORM_BITS, NUM_DISTANCE_CODES,
                  NUM_LENGTH_CODES, NUM_LITERAL_CODES, PREDICTOR_TRANSFORM,
                  SUBTRACT_GREEN_TRANSFORM)

CASES = []
SLOW = set()


def case(name, expect, note, exercises, slow=False):
    """Registers one file: 'expect' is 'ok' or 'reject'; 'slow' marks the
    cases that allocate enough memory to be worth skipping."""
    if slow:
        SLOW.add(name)

    def deco(fn):
        CASES.append((name, expect, note, exercises, fn))
        return fn
    return deco


# -----------------------------------------------------------------------------
# helpers


def trivial_codes(bw, green=0x20, red=0x40, blue=0x60, alpha=0xff, skip=()):
    """The five meta codes, each a 1-symbol simple code (0 bits/pixel)."""
    values = [green, red, blue, alpha, 0]
    for i, v in enumerate(values):
        if i in skip:
            continue
        write_simple_code(bw, 1, [v])


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


def end_transforms(bw, cache_bits=None, meta_precision=None):
    """Closes the transform list, then the color-cache and meta-Huffman
    fields. 'cache_bits' of 0 still means "cache present", which is how the
    invalid cases are built."""
    bw.put(0, 1)                                  # no (more) transforms
    if cache_bits is None:
        bw.put(0, 1)
    else:
        bw.put(1, 1)
        bw.put(cache_bits, 4)
    if meta_precision is None:
        bw.put(0, 1)
    else:
        bw.put(1, 1)
        bw.put(meta_precision - MIN_HUFFMAN_BITS, 3)


def head(bw, w=1, h=1, version=0, cache_bits=None, meta_precision=None):
    """A level-0 stream header with no transforms."""
    write_header(bw, w, h, version=version)
    end_transforms(bw, cache_bits, meta_precision)


def out(bw):
    return wrap_webp(bw.to_bytes())


# -----------------------------------------------------------------------------
# A. simple codes -- the form the encoder uses for <= 2 symbols


@case('simple-dist-2sym-first-oob', 'ok',
      'Distance code: simple form, 2 symbols, the first one 200 >= '
      'alphabet_size 40.',
      'ReadHuffmanCode() writes code_lengths[200] with alphabet_size 40; the '
      'code then has one symbol left and is accepted. Pins the behaviour '
      'CL 8256621 documents.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    write_simple_code(bw, 2, [200, 3])
    return out(bw)


@case('simple-dist-2sym-second-oob', 'ok',
      'Distance code: simple form, 2 symbols, the second one 200 >= 40.',
      'Same as above but the out-of-range symbol is the second 8-bit field.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    write_simple_code(bw, 2, [3, 200])
    return out(bw)


@case('simple-dist-2sym-both-oob', 'reject',
      'Distance code: both simple-form symbols out of range (200, 201).',
      'No symbol is left inside alphabet_size, so BuildHuffmanTable() sees an '
      'empty code and fails. Must stay a clean BITSTREAM_ERROR, not a crash.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    write_simple_code(bw, 2, [200, 201])
    return out(bw)


@case('simple-dist-1sym-oob', 'reject',
      'Distance code: simple form, single symbol 255, alphabet_size is 40.',
      'The single write lands past the logical alphabet but inside the shared '
      'max_alphabet_size buffer. Rejected because no symbol remains.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    write_simple_code(bw, 1, [255])
    return out(bw)


@case('simple-dist-sym-39-last-valid', 'ok',
      'Distance code: single symbol 39, the last in-range value.',
      'Boundary partner of simple-dist-sym-40-first-oob: 39 == '
      'NUM_DISTANCE_CODES - 1 must be accepted.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    write_simple_code(bw, 1, [39])
    return out(bw)


@case('simple-dist-sym-40-first-oob', 'reject',
      'Distance code: single symbol 40, the first out-of-range value.',
      'Exact boundary of the check that does not exist in ReadHuffmanCode(). '
      'If someone adds one, these two files pin where it goes.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    write_simple_code(bw, 1, [40])
    return out(bw)


@case('simple-green-1bit-symbol', 'ok',
      'Green code: simple form with first_symbol_len_code = 0, so the symbol '
      'is 1 bit wide.',
      'The short form of the simple code, only reachable when the symbol is 0 '
      'or 1. cwebp emits it rarely.')
def _():
    bw = BitWriter()
    head(bw)
    write_simple_code(bw, 1, [1], use_8bit=False)
    trivial_codes(bw, skip=(0,))
    return out(bw)


@case('simple-dist-2sym-duplicate', 'ok',
      'Distance code: simple form declaring 2 symbols that are the same (5, '
      '5).',
      'code_lengths[5] is written twice, so the code really has one symbol. '
      'BuildHuffmanTable() takes its single-value shortcut.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    write_simple_code(bw, 2, [5, 5])
    return out(bw)


@case('simple-green-2sym-1bit-each', 'ok',
      'Green code with two real symbols, so every pixel costs exactly 1 bit.',
      'The smallest non-trivial code. 4x1 pixels alternate between the two '
      'green values.')
def _():
    bw = BitWriter()
    head(bw, 4, 1)
    write_simple_code(bw, 2, [0x10, 0x20])
    trivial_codes(bw, skip=(0,))
    green = Huffman([1 if i in (0x10, 0x20) else 0
                     for i in range(alphabet_size(0))])
    for v in (0x10, 0x20, 0x20, 0x10):
        green.emit_symbol(bw, v)
    return out(bw)


# -----------------------------------------------------------------------------
# B. the code-length code -- the Huffman code that describes a Huffman code


@case('codelen-repeat16-no-previous', 'ok',
      'Code-length stream starting with code 16 (repeat previous), before any '
      'non-zero length was seen.',
      "Hits DEFAULT_CODE_LENGTH: 'prev_code_len' is still 8 at "
      'vp8l_dec.c:254, so the first symbols get length 8 out of nowhere.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    # 16 (extra 0) repeats the *default* length 8 over symbols 0..2, then an
    # explicit 8 and depths 1..6 complete the tree, then 30 zeros.
    syms = [(16, 2, 0), (8, 0, 0)] + [(d, 0, 0) for d in range(1, 7)]
    syms.append((18, 7, 19))                        # 11 + 19 = 30 zeros
    lengths = [8, 8, 8, 8, 1, 2, 3, 4, 5, 6] + [0] * (NUM_DISTANCE_CODES - 10)
    write_complex_code(bw, lengths, raw_symbols=syms)
    return out(bw)


@case('codelen-repeat18-138-zeros', 'ok',
      'Code-length stream using code 18 with its maximum run of 138 zeros.',
      'Longest repeat the format allows (11 + 127). Green alphabet is 280 '
      'symbols so two of them fit.')
def _():
    bw = BitWriter()
    head(bw)
    size = alphabet_size(0)
    lengths = [0] * size
    lengths[0] = 1
    lengths[277] = 1
    syms = [(1, 0, 0)]                  # symbol 0, length 1
    syms.append((18, 7, 127))           # 138 zeros
    syms.append((18, 7, 127))           # 138 zeros -> at 277
    syms.append((1, 0, 0))              # symbol 277, length 1
    syms += [(0, 0, 0), (0, 0, 0)]      # symbols 278, 279
    write_complex_code(bw, lengths, raw_symbols=syms)
    trivial_codes(bw, skip=(0,))
    green = Huffman(lengths)
    green.emit_symbol(bw, 0)
    return out(bw)


@case('codelen-repeat17-short-zeros', 'ok',
      'Code-length stream using code 17 (3..10 zeros) rather than 18.',
      'The short zero-run escape. Its extra field is 3 bits, offset 3.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    syms = [(1, 0, 0), (17, 3, 7),          # symbol 0, then 10 zeros
            (1, 0, 0),                      # symbol 11
            (17, 3, 7), (17, 3, 7), (17, 3, 5)]   # 10 + 10 + 8 = 28 zeros
    lengths = [1] + [0] * 10 + [1] + [0] * 28
    write_complex_code(bw, lengths, raw_symbols=syms)
    return out(bw)


@case('codelen-max-symbol-early-stop', 'ok',
      "Code-length stream with an explicit max_symbol far below the alphabet "
      'size.',
      'ReadHuffmanCodeLengths() breaks out at vp8l_dec.c:284 with most '
      'lengths still zero. Exercises the use_length branch cwebp never takes.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    lengths = [1, 1] + [0] * (NUM_DISTANCE_CODES - 2)
    syms = [(1, 0, 0), (1, 0, 0)]
    write_complex_code(bw, lengths, raw_symbols=syms, max_symbol=2)
    return out(bw)


@case('codelen-max-symbol-too-big', 'reject',
      'Explicit max_symbol greater than the alphabet size.',
      'Must be caught by the max_symbol > num_symbols test at '
      'vp8l_dec.c:273.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    lengths = [1, 1] + [0] * (NUM_DISTANCE_CODES - 2)
    syms = [(1, 0, 0), (1, 0, 0)]
    write_complex_code(bw, lengths, raw_symbols=syms, max_symbol=200)
    return out(bw)


@case('codelen-repeat-past-end', 'reject',
      'A repeat run that would write past the end of the alphabet.',
      'Must be caught by the symbol + repeat > num_symbols test at '
      'vp8l_dec.c:298.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    lengths = [1, 1] + [0] * (NUM_DISTANCE_CODES - 2)
    syms = [(1, 0, 0), (1, 0, 0), (18, 7, 127)]  # 138 zeros, only 38 left
    write_complex_code(bw, lengths, raw_symbols=syms)
    return out(bw)


@case('codelen-num-codes-4', 'ok',
      'Only 4 code-length codes declared, the minimum the 4-bit field allows.',
      'Restricts the code-length alphabet to {17, 18, 0, 1}, so lengths can '
      'only be 0 or 1 plus the two zero-run escapes.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    lengths = [1, 1] + [0] * (NUM_DISTANCE_CODES - 2)
    syms = [(1, 0, 0), (1, 0, 0), (18, 7, 27)]
    write_complex_code(bw, lengths, raw_symbols=syms, num_codes=4)
    return out(bw)


@case('codelen-num-codes-19', 'ok',
      'All 19 code-length codes declared.',
      'Maximum of the 4-bit num_codes field; every entry of '
      'kCodeLengthCodeOrder[] gets a 3-bit length.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    lengths = [1, 1] + [0] * (NUM_DISTANCE_CODES - 2)
    syms = [(1, 0, 0), (1, 0, 0), (18, 7, 27)]
    write_complex_code(bw, lengths, raw_symbols=syms, num_codes=19)
    return out(bw)


@case('codelen-depth-15', 'ok',
      'A green code containing a symbol of depth 15, MAX_ALLOWED_CODE_LENGTH.',
      'The deepest code the format allows; forces the two-level lookup in '
      'BuildHuffmanTable() past HUFFMAN_TABLE_BITS.')
def _():
    bw = BitWriter()
    head(bw)
    size = alphabet_size(0)
    # 15 symbols: depths 1,2,3,...,14,15,15 -> complete code
    lengths = [0] * size
    depths = list(range(1, 15)) + [15, 15]
    for i, d in enumerate(depths):
        lengths[i] = d
    write_complex_code(bw, lengths, use_repeats=True)
    trivial_codes(bw, skip=(0,))
    green = Huffman(lengths)
    green.emit_symbol(bw, 15)   # the depth-15 symbol
    return out(bw)


@case('codelen-single-symbol-complex-form', 'ok',
      'The complex form used to describe a code with exactly one symbol.',
      "Takes BuildHuffmanTable()'s offset[MAX_ALLOWED_CODE_LENGTH] == 1 "
      'shortcut, which makes the code 0 bits wide.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    lengths = [0] * NUM_DISTANCE_CODES
    lengths[7] = 1
    syms = [(0, 0, 0)] * 7 + [(1, 0, 0)] + [(18, 7, 21)]  # + 32 zeros = 40
    write_complex_code(bw, lengths, raw_symbols=syms)
    return out(bw)


@case('codelen-over-capacity', 'reject',
      'Three symbols of depth 1, more than the two codes of that length that '
      'exist.',
      'Caught early, by the count[len] > (1 << len) guard in '
      'BuildHuffmanTable(), before the tree walk runs.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    lengths = [1, 1, 1] + [0] * (NUM_DISTANCE_CODES - 3)
    syms = [(1, 0, 0)] * 3 + [(18, 7, 26)]   # + 37 zeros = 40
    write_complex_code(bw, lengths, raw_symbols=syms)
    return out(bw)


@case('codelen-oversubscribed', 'reject',
      'Lengths 1, 2, 2, 2: each length is individually possible, but together '
      'they over-subscribe the tree.',
      'Slips past the per-length capacity guard and is caught later, when '
      'num_open goes negative during the tree walk.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    lengths = [1, 2, 2, 2] + [0] * (NUM_DISTANCE_CODES - 4)
    syms = [(1, 0, 0), (2, 0, 0), (2, 0, 0), (2, 0, 0), (18, 7, 25)]
    write_complex_code(bw, lengths, raw_symbols=syms)
    return out(bw)


@case('codelen-two-level-table', 'ok',
      'A green code with depths up to 10, past the 8-bit root table.',
      'Forces BuildHuffmanTable() to allocate a second-level table and '
      'ReadSymbol() to take its two-step lookup.')
def _():
    bw = BitWriter()
    head(bw)
    size = alphabet_size(0)
    lengths = [0] * size
    depths = list(range(1, 10)) + [10, 10]      # 1..9 then two 10s: complete
    for i, d in enumerate(depths):
        lengths[i] = d
    write_complex_code(bw, lengths, use_repeats=True)
    trivial_codes(bw, skip=(0,))
    green = Huffman(lengths)
    green.emit_symbol(bw, 10)                   # one of the depth-10 symbols
    return out(bw)


@case('codelen-incomplete', 'reject',
      'A code whose lengths leave the tree incomplete (two symbols of depth '
      '2).',
      'Caught by the num_nodes != 2 * num_symbols - 1 test at the end of '
      'BuildHuffmanTable().')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    lengths = [2, 2] + [0] * (NUM_DISTANCE_CODES - 2)
    syms = [(2, 0, 0)] * 2 + [(18, 7, 27)]   # + 38 zeros = 40
    write_complex_code(bw, lengths, raw_symbols=syms)
    return out(bw)


@case('codelen-all-zero-lengths', 'reject',
      'A code-length stream that assigns length 0 to every symbol.',
      'Empty code. Different route to the same rejection as '
      'simple-dist-1sym-oob.')
def _():
    bw = BitWriter()
    head(bw)
    trivial_codes(bw, skip=(4,))
    lengths = [0] * NUM_DISTANCE_CODES
    syms = [(18, 7, 29)]  # 40 zeros
    write_complex_code(bw, lengths, raw_symbols=syms)
    return out(bw)


# -----------------------------------------------------------------------------
# C. meta Huffman: the entropy image that selects a code group per tile


def meta_case(bw, w, h, precision, groups):
    """Writes a level-0 stream whose entropy image assigns 'groups' per tile."""
    head(bw, w, h, meta_precision=precision)
    sw = sub_sample_size(w, precision)
    sh = sub_sample_size(h, precision)
    assert len(groups) == sw * sh, (len(groups), sw, sh)
    # group index lives in the red (high) and green (low) bytes
    pixels = [((g >> 8) << 16) | ((g & 0xff) << 8) for g in groups]
    write_subimage(bw, pixels)
    return max(groups) + 1


@case('meta-huffman-precision-min', 'ok',
      'Meta Huffman with the smallest tile size (precision 2, 4x4 pixels).',
      'MIN_HUFFMAN_BITS. A 16x16 image is split into 4x4 = 16 tiles, all '
      'pointing at group 0.')
def _():
    bw = BitWriter()
    n = meta_case(bw, 16, 16, 2, [0] * 16)
    for _g in range(n):
        trivial_codes(bw)
    return out(bw)


@case('meta-huffman-precision-max', 'ok',
      'Meta Huffman with the largest tile size (precision 9, 512x512 pixels).',
      'MAX_HUFFMAN_BITS. One tile covers the whole image, so the entropy '
      'image is 1x1.')
def _():
    bw = BitWriter()
    n = meta_case(bw, 8, 8, 9, [0])
    for _g in range(n):
        trivial_codes(bw)
    return out(bw)


@case('meta-huffman-two-groups', 'ok',
      'Two Huffman groups selected per tile by the entropy image.',
      'The left half of the image uses group 0 (green 0x20), the right half '
      'group 1 (green 0xd0).')
def _():
    bw = BitWriter()
    n = meta_case(bw, 8, 4, 2, [0, 1])
    assert n == 2
    trivial_codes(bw, green=0x20)
    trivial_codes(bw, green=0xd0)
    return out(bw)


@case('meta-huffman-sparse-groups', 'ok',
      'Entropy image referencing groups 0 and 900 only, leaving a 900-entry '
      'hole.',
      'num_htree_groups_max (901) exceeds the pixel count, so '
      'ReadHuffmanCodes() builds the mapping[] remap and the 899 unused '
      'groups take the "validate but do not store" branch.')
def _():
    bw = BitWriter()
    n = meta_case(bw, 8, 4, 2, [0, 900])
    assert n == 901
    for _g in range(n):
        trivial_codes(bw)
    return out(bw)


@case('meta-huffman-1001-groups', 'ok',
      'Entropy image whose highest group index is 1000, one past the '
      "decoder's arbitrary limit.",
      'Crosses the num_htree_groups_max > 1000 test at vp8l_dec.c:409, which '
      'forces the mapping[] path even when the count is plausible.')
def _():
    bw = BitWriter()
    n = meta_case(bw, 8, 4, 2, [0, 1000])
    assert n == 1001
    for _g in range(n):
        trivial_codes(bw)
    return out(bw)


# -----------------------------------------------------------------------------
# D. color cache


@case('cache-bits-1', 'ok',
      'Color cache with the minimum size, 1 bit (2 entries).',
      'Lower bound of the cache_bits >= 1 check in DecodeImageStream().')
def _():
    bw = BitWriter()
    head(bw, 1, 1, cache_bits=1)
    trivial_codes(bw)
    return out(bw)


@case('cache-bits-11', 'ok',
      'Color cache with the maximum size, 11 bits (2048 entries).',
      'MAX_CACHE_BITS. Also stretches the green alphabet to 280 + 2048 '
      'symbols.')
def _():
    bw = BitWriter()
    head(bw, 1, 1, cache_bits=11)
    trivial_codes(bw)
    return out(bw)


@case('cache-bits-0-invalid', 'reject',
      'Color cache flagged as present but with 0 bits.',
      'Must be rejected: the format reserves "no cache" for the flag bit, so '
      '0 is not a legal size.')
def _():
    bw = BitWriter()
    head(bw, 1, 1, cache_bits=0)
    trivial_codes(bw)
    return out(bw)


@case('cache-bits-12-invalid', 'reject',
      'Color cache with 12 bits, one past MAX_CACHE_BITS.',
      'Upper bound of the same check. The 4-bit field can hold up to 15.')
def _():
    bw = BitWriter()
    head(bw, 1, 1, cache_bits=12)
    trivial_codes(bw)
    return out(bw)


@case('cache-index-literal', 'ok',
      'A pixel coded as a color-cache index rather than as a literal.',
      'Green symbols >= NUM_LITERAL_CODES + NUM_LENGTH_CODES address the '
      'cache. Pixel 2 replays pixel 1 through cache slot 0.')
def _():
    bw = BitWriter()
    write_header(bw, 2, 1)
    bw.put(0, 1)
    bw.put(1, 1)
    bw.put(1, 4)              # 1-bit cache, 2 entries
    bw.put(0, 1)
    size = alphabet_size(0, cache_bits=1)
    cache_sym = NUM_LITERAL_CODES + NUM_LENGTH_CODES  # first cache index
    lengths = [0] * size
    lengths[0x20] = 1
    lengths[cache_sym] = 1
    write_complex_code(bw, lengths, use_repeats=True)
    trivial_codes(bw, skip=(0,))
    green = Huffman(lengths)
    green.emit_symbol(bw, 0x20)        # literal
    green.emit_symbol(bw, cache_sym)   # same pixel, via the cache
    return out(bw)


# -----------------------------------------------------------------------------
# E. transforms


@case('transform-all-four', 'ok',
      'All four transforms present in one stream.',
      'NUM_TRANSFORMS in a row: color-indexing, subtract-green, cross-color '
      'and predictor, each with its own sub-image.')
def _():
    bw = BitWriter()
    write_header(bw, 8, 8)
    # color indexing (must come first: it changes xsize)
    bw.put(1, 1)
    bw.put(COLOR_INDEXING_TRANSFORM, 2)
    bw.put(1, 8)                       # 2 colors
    write_subimage(bw, delta_palette([0xff000000, 0xff00ff00]))
    bw.put(1, 1)
    bw.put(SUBTRACT_GREEN_TRANSFORM, 2)
    bw.put(1, 1)
    bw.put(CROSS_COLOR_TRANSFORM, 2)
    bw.put(0, 3)                       # bits = MIN_TRANSFORM_BITS
    cw = sub_sample_size(8, MIN_TRANSFORM_BITS)
    ch = sub_sample_size(8, MIN_TRANSFORM_BITS)
    write_subimage(bw, [0xff000000] * (cw * ch))
    bw.put(1, 1)
    bw.put(PREDICTOR_TRANSFORM, 2)
    bw.put(0, 3)
    write_subimage(bw, [0xff000000] * (cw * ch))
    end_transforms(bw)
    trivial_codes(bw, green=0)
    return out(bw)


@case('transform-repeated', 'reject',
      'The subtract-green transform declared twice.',
      'Each transform type may appear once. Caught by the transforms_seen '
      'bitmask in ReadTransform().')
def _():
    bw = BitWriter()
    write_header(bw, 4, 4)
    bw.put(1, 1)
    bw.put(SUBTRACT_GREEN_TRANSFORM, 2)
    bw.put(1, 1)
    bw.put(SUBTRACT_GREEN_TRANSFORM, 2)
    end_transforms(bw)
    trivial_codes(bw)
    return out(bw)


@case('transform-palette-2-colors', 'ok',
      'Color-indexing transform with 2 colors, so 8 pixels are packed per '
      'byte.',
      'num_colors <= 2 gives bits = 3, the densest packing, and shrinks xsize '
      'to ceil(w / 8).')
def _():
    bw = BitWriter()
    write_header(bw, 8, 1)
    bw.put(1, 1)
    bw.put(COLOR_INDEXING_TRANSFORM, 2)
    bw.put(1, 8)                       # num_colors - 1
    write_subimage(bw, delta_palette([0xff0000ff, 0xff00ff00]))
    end_transforms(bw)
    # packed image is 1 pixel wide: green byte holds 8 one-bit indices
    trivial_codes(bw, green=0b10101010)
    return out(bw)


@case('transform-palette-256-colors', 'ok',
      'Color-indexing transform with the full 256-entry palette.',
      'MAX_PALETTE_SIZE, bits = 0 so there is no packing; also the largest '
      'value the 8-bit num_colors field can hold.')
def _():
    bw = BitWriter()
    write_header(bw, 4, 1)
    bw.put(1, 1)
    bw.put(COLOR_INDEXING_TRANSFORM, 2)
    bw.put(255, 8)
    write_subimage(bw, delta_palette(
        [0xff000000 | (i * 0x010101) for i in range(256)]))
    end_transforms(bw)
    trivial_codes(bw, green=200)
    return out(bw)


@case('transform-predictor-bits-max', 'ok',
      'Predictor transform with bits = 9, the maximum tile size.',
      'MIN_TRANSFORM_BITS + 7. The predictor sub-image is a single pixel for '
      'any image up to 512x512.')
def _():
    bw = BitWriter()
    write_header(bw, 8, 8)
    bw.put(1, 1)
    bw.put(PREDICTOR_TRANSFORM, 2)
    bw.put(7, 3)
    write_subimage(bw, [0xff000000])   # predictor 0 everywhere
    end_transforms(bw)
    trivial_codes(bw)
    return out(bw)


# -----------------------------------------------------------------------------
# F. LZ77 back-references


@case('lz77-distance-1-run', 'ok',
      'A single literal followed by a length-8 copy at distance 1.',
      'The degenerate overlapping copy: the copy loop reads bytes it has just '
      'written.')
def _():
    bw = BitWriter()
    head(bw, 10, 1)
    size = alphabet_size(0)
    length_sym = NUM_LITERAL_CODES + 6      # prefix 6 -> lengths 9..12
    lengths = [0] * size
    lengths[0x33] = 1
    lengths[length_sym] = 1
    write_complex_code(bw, lengths, use_repeats=True)
    trivial_codes(bw, skip=(0, 4))
    dist_lengths = [0] * NUM_DISTANCE_CODES
    dist_lengths[1] = 1                     # plane code 2 -> distance 1
    write_complex_code(bw, dist_lengths, use_repeats=True)
    green = Huffman(lengths)
    green.emit_symbol(bw, 0x33)             # one literal
    green.emit_symbol(bw, length_sym)
    bw.put(0, 2)                            # extra bits -> length 9
    # the distance code holds one symbol, so it is 0 bits wide
    return out(bw)


@case('lz77-max-length-symbol', 'ok',
      'A back-reference using length symbol 23, the largest the format '
      'defines.',
      'NUM_LENGTH_CODES - 1: 10 extra bits, copy lengths up to 4096. Here it '
      'copies 1200 pixels.')
def _():
    bw = BitWriter()
    write_header(bw, 3074, 1)               # 1 literal + a 3073-pixel copy
    end_transforms(bw)
    size = alphabet_size(0)
    length_sym = NUM_LITERAL_CODES + 23
    lengths = [0] * size
    lengths[0x44] = 1
    lengths[length_sym] = 1
    write_complex_code(bw, lengths, use_repeats=True)
    trivial_codes(bw, skip=(0, 4))
    dist_lengths = [0] * NUM_DISTANCE_CODES
    dist_lengths[1] = 1                     # plane code 2 -> distance 1
    write_complex_code(bw, dist_lengths, use_repeats=True)
    green = Huffman(lengths)
    green.emit_symbol(bw, 0x44)
    # symbol 23 -> extra_bits = (23 - 2) >> 1 = 10, offset = 3 << 10 = 3072,
    # so the copy length is 3072 + extra + 1.
    green.emit_symbol(bw, length_sym)
    bw.put(0, 10)                           # length 3073
    return out(bw)


# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# G. predictor modes


def predictor_case(bw, w, h, bits, modes):
    """A predictor transform whose sub-image assigns 'modes' per tile."""
    write_header(bw, w, h)
    bw.put(1, 1)
    bw.put(PREDICTOR_TRANSFORM, 2)
    bw.put(bits - MIN_TRANSFORM_BITS, 3)
    tw = sub_sample_size(w, bits)
    th = sub_sample_size(h, bits)
    assert len(modes) == tw * th, (len(modes), tw, th)
    write_subimage(bw, [0xff000000 | (m << 8) for m in modes])
    bw.put(0, 1)          # no further transform
    bw.put(0, 1)          # no color cache
    bw.put(0, 1)          # no meta huffman
    trivial_codes(bw, green=0x11, red=0x22, blue=0x33, alpha=0x44)


@case('predictor-all-16-modes', 'ok',
      'One tile per predictor index, 0 to 15, across a 64x4 image.',
      'The mode is read as ((pixel >> 8) & 0xf) at lossless.c:247, so all 16 '
      'indices are reachable even though the format defines only 0..13.')
def _():
    bw = BitWriter()
    predictor_case(bw, 64, 4, MIN_TRANSFORM_BITS, list(range(16)))
    return out(bw)


@case('predictor-mode-14-undefined', 'ok',
      'Every tile selects predictor 14, which the format does not define.',
      'Must decode, not crash: VP8LPredictorsAdd[14] is a padding sentinel '
      'pointing at PredictorAdd0_C (lossless.c:653), so the tile comes out as '
      'mode 0. Shrink the table to 14 entries and this is an out-of-bounds '
      'indirect call instead.')
def _():
    bw = BitWriter()
    predictor_case(bw, 16, 4, MIN_TRANSFORM_BITS, [14] * 4)
    return out(bw)


@case('predictor-mode-15-undefined', 'ok',
      'Every tile selects predictor 15, the other undefined index.',
      'Partner of predictor-mode-14-undefined and the largest value the '
      '4-bit mask can produce. Verified to decode identically to mode 14, '
      'i.e. both really do land on PredictorAdd0_C.')
def _():
    bw = BitWriter()
    predictor_case(bw, 16, 4, MIN_TRANSFORM_BITS, [15] * 4)
    return out(bw)


@case('predictor-mode-11-select', 'ok',
      'Predictor 11 (Select) over the whole image.',
      'The only predictor with a data-dependent branch, Select() at '
      'lossless.c:100.')
def _():
    bw = BitWriter()
    predictor_case(bw, 16, 4, MIN_TRANSFORM_BITS, [11] * 4)
    return out(bw)


@case('predictor-mode-13-clamp-half', 'ok',
      'Predictor 13 (ClampAddSubtractHalf) over the whole image.',
      'Exercises AddSubtractComponentHalf() and its Clip255(), the arithmetic '
      'most likely to differ between the C and SIMD paths.')
def _():
    bw = BitWriter()
    predictor_case(bw, 16, 4, MIN_TRANSFORM_BITS, [13] * 4)
    return out(bw)


@case('predictor-single-row', 'ok',
      'A predictor transform on a one-row image.',
      'Only the y_start == 0 shortcut runs (lossless.c:223): the first pixel '
      'takes mode 0 and the rest mode 1, so the tile modes are never read.')
def _():
    bw = BitWriter()
    predictor_case(bw, 16, 1, MIN_TRANSFORM_BITS, [7] * 4)
    return out(bw)


@case('predictor-tile-bits-min', 'ok',
      'Predictor tiles of 4x4 pixels, the smallest the format allows.',
      'MIN_TRANSFORM_BITS, so the sub-image is as large as it can get and the '
      'mode changes every four pixels.')
def _():
    bw = BitWriter()
    predictor_case(bw, 32, 8, MIN_TRANSFORM_BITS, [i % 14 for i in range(16)])
    return out(bw)


# -----------------------------------------------------------------------------
# H. back-reference distances


def lz77_case(bw, w, h, length, plane_code, n_literals=None,
              literal_green=0x33):
    """'n_literals' literals then one copy of 'length' at 'plane_code'."""
    if n_literals is None:
        n_literals = w * h - length
    head(bw, w, h)
    size = alphabet_size(0)
    lsym, lbits, lextra = prefix_code(length)
    length_sym = NUM_LITERAL_CODES + lsym
    lengths = [0] * size
    lengths[literal_green] = 1
    lengths[length_sym] = 1
    write_complex_code(bw, lengths, use_repeats=True)
    trivial_codes(bw, skip=(0, 4))
    dsym, dbits, dextra = prefix_code(plane_code)
    write_simple_code(bw, 1, [dsym])          # distance code: 0 bits wide
    green = Huffman(lengths)
    for _ in range(n_literals):
        green.emit_symbol(bw, literal_green)
    green.emit_symbol(bw, length_sym)
    if lbits:
        bw.put(lextra, lbits)
    if dbits:
        bw.put(dextra, dbits)


@case('lz77-plane-code-1', 'ok',
      'Back-reference with plane code 1, which means "the pixel directly '
      'above".',
      'kCodeToPlane[0] is 0x18: yoffset 1, xoffset 0, so the distance is a '
      'whole row rather than a small number.')
def _():
    bw = BitWriter()
    lz77_case(bw, 8, 2, length=8, plane_code=1)
    return out(bw)


@case('lz77-plane-code-clamped-to-1', 'ok',
      'Plane code 4 on a 1-pixel-wide image, where the 2-D offset computes to '
      '0.',
      'kCodeToPlane[3] is 0x19: yoffset 1, xoffset -1, so dist = xsize - 1 = '
      '0 and the "dist < 1 ? 1" clamp at vp8l_dec.c:173 fires. Only '
      'reachable at xsize 1.')
def _():
    bw = BitWriter()
    lz77_case(bw, 1, 8, length=7, plane_code=4)
    return out(bw)


@case('lz77-plane-code-120', 'ok',
      'Plane code 120, the last entry of the 2-D offset table.',
      'kCodeToPlane[119] is 0x70: yoffset 7, xoffset 8, so on a 16-wide image '
      'the distance is 120. Upper bound of the mapped range.')
def _():
    bw = BitWriter()
    assert plane_code_to_distance(16, 120) == 120
    lz77_case(bw, 16, 32, length=100, plane_code=120)
    return out(bw)


@case('lz77-distance-direct-121', 'ok',
      'Plane code 121, the first value past the table.',
      'Distances above CODE_TO_PLANE_CODES bypass the 2-D mapping entirely: '
      'the distance is plane_code - 120, so 121 means 1.')
def _():
    bw = BitWriter()
    assert plane_code_to_distance(16, 121) == 1
    lz77_case(bw, 16, 4, length=32, plane_code=121)
    return out(bw)


@case('lz77-distance-past-start', 'reject',
      'A back-reference pointing further back than the pixels decoded so far.',
      'One literal, then a copy at a distance of one whole row. Must be '
      'rejected rather than reading before the buffer.')
def _():
    bw = BitWriter()
    lz77_case(bw, 8, 2, length=8, plane_code=1, n_literals=1)
    return out(bw)


@case('lz77-length-past-end', 'reject',
      'A copy whose length runs past the last pixel of the image.',
      'Four literals then an 8-pixel copy in an 8-pixel image. Must be '
      'rejected rather than writing past the buffer.')
def _():
    bw = BitWriter()
    lz77_case(bw, 8, 1, length=8, plane_code=121, n_literals=4)
    return out(bw)


# -----------------------------------------------------------------------------
# I. palette packing


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
    return 0 if num_colors > 16 else 1 if num_colors > 4 else \
        2 if num_colors > 2 else 3


def palette_case(bw, w, colors, packed_greens):
    bits = palette_bits(len(colors))
    write_header(bw, w, 1)
    bw.put(1, 1)
    bw.put(COLOR_INDEXING_TRANSFORM, 2)
    bw.put(len(colors) - 1, 8)
    write_subimage(bw, delta_palette(colors))
    end_transforms(bw)
    assert len(packed_greens) == sub_sample_size(w, bits)
    size = alphabet_size(0)
    freqs = [0] * size
    for g in packed_greens:
        freqs[g] += 1
    green = Huffman.from_freqs(freqs)
    write_code(bw, green)
    trivial_codes(bw, skip=(0,))
    for g in packed_greens:
        green.emit_symbol(bw, g)


@case('transform-palette-3-colors', 'ok',
      'Palette of 3 colors, so indices are 2 bits and 4 pixels share a byte.',
      'num_colors in 3..4 selects bits = 2, the middle packing density. The '
      'byte 0xe4 holds indices 0, 1, 2, 3 least-significant first.')
def _():
    bw = BitWriter()
    palette_case(bw, 8, [0xff0000ff, 0xff00ff00, 0xffff0000], [0x24, 0x24])
    return out(bw)


@case('transform-palette-index-past-end', 'ok',
      'Palette of 3 colors addressed with index 3, which does not exist.',
      "Reads ExpandColorMap()'s black tail (vp8l_dec.c:1412): the map is "
      'padded out to the packing capacity of 4, so the pixel comes back '
      'transparent black instead of out of bounds.')
def _():
    bw = BitWriter()
    palette_case(bw, 8, [0xff0000ff, 0xff00ff00, 0xffff0000], [0xe4, 0xff])
    return out(bw)


@case('transform-palette-16-colors', 'ok',
      'Palette of 16 colors, so indices are 4 bits and 2 pixels share a byte.',
      'The bits = 1 packing, and the largest palette that still packs.')
def _():
    bw = BitWriter()
    colors = [0xff000000 | (i * 0x111111) for i in range(16)]
    palette_case(bw, 8, colors, [0x10, 0x32, 0x54, 0x76])
    return out(bw)


@case('transform-palette-1-color', 'ok',
      'Palette with a single color, the smallest the 8-bit field can express.',
      'bits = 3, so 8 pixels share a byte and the map is padded from 1 entry '
      'to 2. Index 1 is the black tail.')
def _():
    bw = BitWriter()
    palette_case(bw, 8, [0xff123456], [0x0f])
    return out(bw)


# -----------------------------------------------------------------------------
# J. frame header


@case('header-width-16384', 'ok',
      'Width 16384, one past WEBP_MAX_DIMENSION.',
      'The header stores width - 1 in 14 bits, so 16384 is expressible and '
      'the decoder accepts it. WEBP_MAX_DIMENSION (16383) is enforced only in '
      'the encoder, at webp_enc.c:347, so cwebp can never produce this.')
def _():
    bw = BitWriter()
    head(bw, 16384, 1)
    trivial_codes(bw)
    return out(bw)


@case('header-max-area-bomb', 'ok',
      '34 bytes declaring 16384x16384, every pixel one color.',
      'A bare VP8L stream has no area limit -- MAX_IMAGE_AREA is only checked '
      'in ParseVP8X (webp_dec.c:138) -- and single-symbol codes cost zero '
      'bits per pixel, so this decodes for real: 1.83GB peak RSS and 3.4s. '
      'The backstop is WEBP_MAX_ALLOCABLE_MEMORY (utils.c:185), nothing '
      'earlier.', slow=True)
def _():
    bw = BitWriter()
    head(bw, 16384, 16384)
    trivial_codes(bw)
    return out(bw)


@case('header-max-area-truncated', 'reject',
      'The same 16384x16384 header, cut off before the Huffman codes.',
      'Must fail on the missing data rather than allocating the gigabyte '
      'first. Partner of header-max-area-bomb: together they say where the '
      'allocation sits relative to the parse.')
def _():
    bw = BitWriter()
    head(bw, 16384, 16384)
    return out(bw)


@case('header-version-nonzero', 'reject',
      'Header with the 3-bit version field set to 1.',
      'Rejected by VP8LCheckSignature() at vp8l_dec.c:111, which tests '
      '(data[4] >> 5) before ReadImageInfo() ever runs. The version field is '
      "the format's only forward-compatibility escape.")
def _():
    bw = BitWriter()
    head(bw, 4, 4, version=1)
    trivial_codes(bw)
    return out(bw)


GROUPS = [
    ('simple-', 'Simple codes',
     'The 1-or-2-symbol shorthand a Huffman code can take. Its symbols are '
     'read as raw 8-bit values and are never checked against the alphabet '
     'size, so this is where a stream can say things an encoder cannot.'),
    ('codelen-', 'The code-length code',
     'The Huffman code that describes the lengths of another Huffman code, '
     'plus its repeat escapes (16, 17, 18) and the optional max_symbol '
     'field. cwebp only ever emits a narrow slice of this.'),
    ('meta-', 'Meta Huffman / entropy image',
     'The sub-image that picks one of several code groups per tile, and the '
     'remapping the decoder does when the group count looks implausible.'),
    ('cache-', 'Color cache', 'Size bounds, and cache-index literals.'),
    ('transform-palette-', 'Palette packing',
     'Index width follows the palette size, and the map is padded out to the '
     'packing capacity with black.'),
    ('transform-', 'Transforms',
     'Presence, repetition and tile sizes.'),
    ('lz77-', 'Back-references', 'Copy lengths and distances.'),
    ('predictor-', 'Predictor modes',
     'The per-tile predictor index. It is read as a 4-bit field, so all 16 '
     'values are reachable, but the format only defines 14 of them.'),
    ('header-', 'Frame header',
     'The 14-bit dimension fields and the version escape.'),
    ('container-', 'The RIFF container',
     'The layer above the image: the RIFF header, the extended-format VP8X '
     'chunk and its canvas size, the optional chunks a decoder must step '
     'over by their declared length alone, and the padding rule that makes '
     'an odd-sized one even on disk. Everything here is read by '
     'webp_dec.c before the frame is looked at.'),
    ('alph-', 'The alpha chunk',
     'ALPH carries the alpha plane beside a lossy frame: a header byte of '
     'four two-bit fields, then the plane itself, either stored as it is or '
     'compressed with the lossless coder in its 8-bit mode. That mode is a '
     'separate path through vp8l_dec.c from the one every VP8L file here '
     'takes, and these are the only files that reach it.'),
    ('lossy-frame-', 'Lossy: frame tag and picture header',
     'The ten uncompressed bytes every lossy frame starts with: the profile, '
     'the visibility and key-frame bits, the length of partition 0, the '
     'start code and the two 14-bit dimensions.'),
    ('lossy-segment-', 'Lossy: segmentation',
     'Up to four segments, each with its own quantizer and loop-filter '
     'strength, and a per-macroblock map saying which is which. cwebp uses '
     'the feature but only ever writes absolute values, and always writes '
     'the map and the data together.'),
    ('lossy-filter-', 'Lossy: loop filter',
     'The in-loop deblocking filter: simple or normal, its level and '
     'sharpness, and the per-reference and per-mode deltas.'),
    ('lossy-quant-', 'Lossy: quantizer',
     'The frame quantizer index and the five deltas around it, one per plane '
     'and coefficient kind, with clamps that are not all the same.'),
    ('lossy-proba-', 'Lossy: coefficient probabilities',
     'The 1056 probabilities that drive the coefficient coder, each one '
     'optionally replaced in the frame header, plus the skip probability.'),
    ('lossy-mode-', 'Lossy: prediction modes',
     'The 16x16 and 4x4 luma modes and the chroma modes, and the '
     'neighbour-indexed probability table the 4x4 modes are coded with.'),
    ('lossy-coeff-', 'Lossy: coefficients',
     'The token coder of section 13: magnitudes and their escape categories, '
     'end-of-block, zero runs, and the four coefficient types.'),
    ('lossy-skip-', 'Lossy: skipped macroblocks',
     'The per-macroblock skip flag, which drops the residual entirely and '
     'clears the neighbouring non-zero flags -- almost all of them.'),
    ('lossy-parts-', 'Lossy: token partitions',
     'A lossy frame may carry 1, 2, 4 or 8 token partitions, macroblock row '
     'r being read from partition r & (n - 1). cwebp does not expose '
     'config.partitions and libwebp forces it back to 1 whenever the token '
     'path is used (webp_enc.c:124), so none of this is reachable through '
     'the tools.'),
    ('lossy-truncated-', 'Lossy: truncation',
     'Frames that stop early, at each of the places the decoder can notice: '
     'inside partition 0, inside the macroblock modes, and inside the token '
     'data.'),
    ('lossy-', 'Lossy: partition sizes, from real encodes',
     'The four files behind these are genuine encoder output, made through '
     'the encoder API rather than cwebp, and the broken ones rewrite the raw '
     'size table that follows partition 0. They carry 256 macroblocks of '
     'real coefficients, which the assembled cases do not.'),
]

README_HEAD = """# WebP torture bitstreams

Small WebP files that exercise corners of the format a normal encoder never
emits, one layer of it at a time:

* **%(vp8l)d lossless (VP8L) streams**, written bit by bit by
  [`vp8l.py`](vp8l.py).
* **%(lossy)d lossy VP8 frames**. All but seven are assembled by
  [`vp8_asm.py`](vp8_asm.py) from a text description of the bitstream, one
  file per case in [`cases/`](cases), under the field names
  [RFC 6386](https://www.rfc-editor.org/rfc/rfc6386.html) gives them. The
  other seven start from an encoder-API call
  ([`make_partition_sources.c`](make_partition_sources.c)) and are patched
  by [`lossy_parts.py`](lossy_parts.py).
* **%(container)d RIFF containers**, wrapped by
  [`webp_asm.py`](webp_asm.py) in
  [RFC 9649](https://www.rfc-editor.org/rfc/rfc9649.html)'s names: the
  extended-format VP8X chunk, the optional chunks a decoder must step over,
  and sizes that lie about what is behind them.
* **%(alpha)d alpha chunks**, where the plane is either stored one byte per
  pixel or compressed with the lossless coder in its 8-bit mode -- a
  different path through the decoder from the one every VP8L file here
  takes.

A case is a text file and the notes below are its own: each one carries what
it is, what the decoder should do with it, and which path that answer comes
from.

Each entry says what the reference decoder is expected to do:

* **ok** -- must decode, and must keep decoding to the same pixels. Several
  are not something cwebp can produce, so nothing else pins the behaviour.
* **reject** -- must fail cleanly and report a status, with no crash and no
  out-of-bounds access. Which status varies: a malformed Huffman code gives
  BITSTREAM_ERROR, a short partition table gives NOT_ENOUGH_DATA.

%(files)s
%(code)s
## Using them

Every file is one click away from this page, but the corpus is one
directory of a much larger repository, so to take the whole thing at once
ask git for just that directory:

    git clone --depth 1 --filter=blob:none --sparse \\
        https://github.com/skal65535/skal65535.github.io.git
    cd skal65535.github.io
    git sparse-checkout set webp-torture

That fetches about 2MB instead of the ~90MB the rest of the site comes to.
Then, from `webp-torture/`:

    ./check.sh              # verdict + decoded-pixel hash for every file
    ./asan_sweep.sh         # 14 decode modes, under a sanitizer build
    ./vp8_selftest.py       # checks the lossy writer against libwebp
    ./make_coverage.sh      # regenerate coverage.txt
    ./make_hashes.sh        # regenerate hashes.txt, once the output is right
    python3 generate.py     # rebuild files/, expected.txt and this README

A case is a text file, one field per line under the name the specification
gives it -- RFC 6386 for the frame, RFC 9649 for the container -- so a case
reads against the format rather than against the decoder that happens to be
under test. `webp_asm.py` assembles any of them and hands the frame part to
`vp8_asm.py`; use either directly, or read an existing frame back out as
text to start from:

    ./webp_asm.py cases/alph-raw-filter-gradient.txt /tmp/out.webp
    ./vp8_asm.py cases/lossy-coeff-cat6.txt /tmp/out.webp
    ./vp8_dis.py some-photo.webp

Each tool's docstring is the reference for the fields it owns:
`vp8_asm.py` for the frame, `webp_asm.py` for the container and the alpha
chunk.

`files/` is pure output and is wiped on every rebuild. The four lossy encodes
the multi-partition cases are patched from live in `sources/` --
[1](sources/lossy-1-partitions.webp), [2](sources/lossy-2-partitions.webp),
[4](sources/lossy-4-partitions.webp), [8](sources/lossy-8-partitions.webp)
partitions -- and are themselves rebuilt by `make_partition_sources.c`.

`check.sh`, `make_hashes.sh` and `vp8_selftest.py` honour `$DWEBP` and
`asan_sweep.sh` honours `$ASAN_DWEBP`, so all of them can be pointed at any
build, or at another decoder implementation; they fall back to whatever
`dwebp` is on `$PATH`. `make_coverage.sh` and `make_vp8_tables.py` need
`$LIBWEBP` set to a libwebp git checkout. `SKIP_SLOW=1` skips the one file
that allocates a gigabyte.

`hashes.txt` holds the SHA-256 of each decoding file's `-pam` output, so the
suite catches a *silent* change in decoded pixels, not just a crash or a
changed verdict.

## How they were verified

Verdicts alone prove little -- a file can be rejected for the wrong reason,
and several of these were before being corrected. Each file was also run
against a decoder instrumented with probes on the exact lines the notes below
refer to; `coverage.txt` records which paths each file actually reached, and
`make_coverage.sh` regenerates it from `probes.py` in a throwaway worktree.
The notes are written from that output, not from reading the code.

The source line numbers they quote (`vp8l_dec.c:111` and friends) are only
meaningful against one revision of libwebp: the one recorded at the top of
`coverage.txt`, which `make_coverage.sh` stamps automatically. If those two
disagree, trust `coverage.txt`.

The lossy writer is checked a second way, against libwebp rather than against
itself: `vp8_dis.py` reads a frame back into the same text `vp8_asm.py`
assembles, so a real cwebp encode can be disassembled, reassembled and
compared byte for byte. Every file in `sources/` survives that, as do encodes
from 1x1 to 128x128 across the whole quality range -- 596 macroblocks, all
four 16x16 modes, all ten 4x4 modes and coefficients in every escape
category. `vp8_selftest.py` runs it, along with every coefficient magnitude
up to the format's largest and a handful of frames that say the same thing
two different ways and must decode alike.

What the corpus reaches is measured rather than assumed, and the measurement
is what says where to add files next. As it stands: every field of the lossy
frame header is written at both ends of its range; all 93 reachable
(coefficient type, band, context) probability cells are read, which is the
whole grid bar the three that no bitstream can select; all 28 pairs of
optional tools appear together in some frame; and one probe out of 88 is
unreached, a version check that `VP8LCheckSignature()` has already made by
the time it runs.

## What is not covered

Animation. There is no real ANIM or ANMF chunk, only the VP8X flag that
claims one, and nothing here goes near the demux API that would be needed to
walk a sequence of frames. No inter frames either, which libwebp refuses
outright, so there is little to pin beyond the one file that checks it does.

The two compressed alpha planes are cwebp output pasted in, so that path has
two points in it rather than a swept range: one that reaches the lossless
decoder's 8-bit loop and one that misses it. `vp8l.py` could generate them --
an alpha plane is a VP8L image whose green channel carries the values -- and
then they could be malformed like every other lossless case here.

Within a lossy key frame, what is left is what the decoder does not read:
the profile selects no reconstruction filter, and the entropy-refresh bit is
parsed and dropped. Both are written anyway, so a decoder that started
acting on either would fail a pixel hash rather than pass unnoticed.

## License

BSD 3-clause, the same as libwebp. See [`COPYING`](COPYING). That covers the
generators, the scripts and the bitstreams in `files/` alike.
"""


def wrap(text, indent=''):
    return '\n'.join(textwrap.wrap(text, 79, initial_indent=indent,
                                   subsequent_indent=indent)) + '\n'


def build_index(rows, groups):
    """A compact table: one row per group, with the ok/reject split."""
    out = ['| Group | Files | must decode | must be rejected |',
           '| --- | ---: | ---: | ---: |']
    seen = set()
    for prefix, title, _ in groups:
        group = [r for r in rows
                 if r[0].startswith(prefix) and r[0] not in seen]
        seen.update(r[0] for r in group)
        if not group:
            continue
        ok = sum(1 for r in group if r[1] == 'ok')
        out.append('| %s | %d | %d | %d |' %
                   (title, len(group), ok, len(group) - ok))
    ok = sum(1 for r in rows if r[1] == 'ok')
    out.append('| **total** | **%d** | **%d** | **%d** |' %
               (len(rows), ok, len(rows) - ok))
    return '\n'.join(out) + '\n'


INDEX_STYLE = """<!doctype html>
<meta charset="utf-8">
<title>webp-torture bitstreams</title>
<style>
 body { font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 54rem;
        padding: 0 1rem; }
 table { border-collapse: collapse; width: 100%%; }
 th, td { text-align: left; padding: .25rem .6rem; border-bottom: 1px solid #ddd; }
 td.n { text-align: right; font-variant-numeric: tabular-nums; }
 .reject { color: #a33; }
 code { font-size: 90%%; }
</style>
"""

FILES_INDEX_HEAD = INDEX_STYLE + """<h1>webp-torture bitstreams</h1>
<p>%(count)d WebP files that exercise corners of the format a normal encoder
never emits, most of them assembled from the text in
<a href="../cases/">cases/</a>. See the <a href="../">notes</a> for what each
one targets. <b>reject</b> means a conforming decoder must refuse the file.</p>
<table>
<tr><th>file</th><th>bytes</th><th>expected</th></tr>
"""

CASES_INDEX_HEAD = INDEX_STYLE + """<h1>webp-torture cases</h1>
<p>%(count)d text cases, each assembled into the .webp of the same name in
<a href="../files/">files/</a>. A case names the fields the specification
names &mdash; RFC 6386 for the frame, RFC 9649 for the container &mdash; and
carries its own note on what it is for; the <a href="../">notes</a> have the
full write-up. <b>reject</b> means a conforming decoder must refuse it.</p>
<table>
<tr><th>case</th><th>expected</th><th>what it is</th></tr>
"""


def write_files_index(outdir, rows):
    """files/index.html -- GitHub Pages serves no directory listing."""
    lines = [FILES_INDEX_HEAD % {'count': len(rows)}]
    for name, expect, _note, _exercises, size in sorted(rows):
        lines.append('<tr><td><a href="%s.webp"><code>%s.webp</code></a></td>'
                     '<td class="n">%d</td><td class="%s">%s</td></tr>'
                     % (name, name, size,
                        'reject' if expect == 'reject' else 'ok', expect))
    lines.append('</table>')
    with open(os.path.join(outdir, 'files', 'index.html'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_cases_index(outdir, rows):
    """cases/index.html -- same reason, and the README links cases/."""
    lines = [CASES_INDEX_HEAD]
    n = 0
    for name, expect, note, _exercises, _size in sorted(rows):
        if not os.path.exists(os.path.join(outdir, 'cases',
                                           name + '.txt')):
            continue                    # written by vp8l.py, not assembled
        n += 1
        lines.append('<tr><td><a href="%s.txt"><code>%s</code></a></td>'
                     '<td class="%s">%s</td><td>%s</td></tr>'
                     % (name, name, 'reject' if expect == 'reject' else 'ok',
                        expect, html_escape(note)))
    lines.append('</table>')
    lines[0] = CASES_INDEX_HEAD % {'count': n}
    with open(os.path.join(outdir, 'cases', 'index.html'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def html_escape(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


CODE = [
    ('vp8l.py', 'VP8L bitstream writer: bit packing, canonical Huffman codes, '
                'prefix coding, RIFF wrapping.'),
    ('generate.py', 'Writes the lossless cases, assembles the rest from '
                    '`cases/`, and produces `expected.txt`, this README '
                    'and the two `index.html` listings.'),
    ('vp8.py', 'VP8 lossy bitstream writer: the boolean coder, the frame '
               'header, the mode trees, the coefficients.'),
    ('vp8_asm.py', 'Assembles a lossy frame from a text case, in RFC 6386\'s '
                   'field names. Its docstring is the format.'),
    ('webp_asm.py', 'Wraps that frame in a RIFF container, in RFC 9649\'s '
                    'field names, for the cases that need one.'),
    ('vp8_dis.py', 'The other direction: a lossy .webp back into that text. '
                   '`--check` round trips one against libwebp.'),
    ('vp8_selftest.py', 'Round trips real encodes through both, and checks '
                        'what cwebp cannot emit against dwebp.'),
    ('vp8_tables.py', 'The VP8 constant tables, extracted from libwebp.'),
    ('make_vp8_tables.py', 'Extracts them, so they are never retyped.'),
    ('lossy_parts.py', 'The multi-partition lossy cases, patched from '
                       '`sources/`.'),
    ('make_partition_sources.c', 'Rebuilds `sources/`: cwebp cannot emit more '
                                 'than one token partition.'),
    ('check.sh', 'Decodes every file; checks the verdict and the pixels.'),
    ('make_hashes.sh', 'Rewrites `hashes.txt` when the new output is known '
                       'to be right.'),
    ('asan_sweep.sh', 'Decodes every file in 14 modes, under a sanitizer '
                      'build.'),
    ('probes.py', 'The `fprintf` probes `make_coverage.sh` patches in.'),
    ('make_coverage.sh', 'Rebuilds `coverage.txt` in a throwaway worktree.'),
    ('expected.txt', 'Name and expected verdict, one line per file.'),
    ('hashes.txt', "SHA-256 of each decoding file's `-pam` output."),
    ('coverage.txt', 'Which decoder path each file actually reached.'),
    ('COPYING', 'BSD 3-clause, the same as libwebp.'),
]


def build_code_list(outdir):
    """A linked table of the generators, scripts and data files."""
    out = ['## The code\n', '| file | what it is |', '| --- | --- |']
    for name, what in CODE:
        assert os.path.exists(os.path.join(outdir, name)), name
        out.append('| [`%s`](%s) | %s |' % (name, name, what))
    return '\n'.join(out) + '\n'


def build_cases(outdir):
    """Assembles every cases/*.txt into files/, and returns its row."""
    rows = []
    for path in sorted(glob.glob(os.path.join(outdir, 'cases',
                                              '*.txt'))):
        name = os.path.basename(path)[:-len('.txt')]
        with open(path) as f:
            text = f.read()
        fields = vp8_asm.parse_header(text, path)
        data = webp_asm.assemble_text(text)
        with open(os.path.join(outdir, 'files', name + '.webp'), 'wb') as f:
            f.write(data)
        rows.append((name, fields['expect'], fields['note'],
                     fields['exercises'], len(data)))
        print('%-40s %-7s %5d bytes' % (name, fields['expect'], len(data)))
    return rows


def build_file_list(rows, groups, index):
    """A linked list of every bitstream, grouped, for the top of the README."""
    out = ['## The bitstreams\n',
           'Every name below links straight to the file. The whole set lives',
           'in **[`files/`](files/)**, which lists each one with its size and',
           'expected verdict; the notes further down say what each targets.\n',
           index]
    seen = set()
    for prefix, title, _ in groups:
        group = [r for r in rows if r[0].startswith(prefix) and r[0] not in seen]
        seen.update(r[0] for r in group)
        if not group:
            continue
        # pack the links by hand: textwrap would break them mid-URL
        line, block = '**%s** —' % title, []
        for r in group:
            link = ' [%s](files/%s.webp)' % (r[0], r[0])
            if len(line) + len(link) > 78 and line.strip():
                block.append(line)
                line = ' ' + link.lstrip()
            else:
                line += link
            line += ' ·'
        block.append(line.rstrip(' ·'))
        out.append('\n'.join(block) + '\n')
    return '\n'.join(out)


def heading(outdir, name, expect):
    """One file's heading, with a link to the case text when there is one."""
    src = os.path.join('cases', name + '.txt')
    out = '### [`%s.webp`](files/%s.webp) -- %s' % (name, name, expect)
    if os.path.exists(os.path.join(outdir, src)):
        out += ' -- from [`%s.txt`](%s)' % (name, src)
    return out + '\n'


def write_readme(outdir, rows):
    used = set()
    kinds = {'lossy': 0, 'container': 0, 'alpha': 0, 'vp8l': 0}
    for r in rows:
        for prefix, kind in (('lossy-', 'lossy'), ('container-', 'container'),
                             ('alph-', 'alpha')):
            if r[0].startswith(prefix):
                kinds[kind] += 1
                break
        else:
            kinds['vp8l'] += 1
    index = build_index(rows, GROUPS)
    lines = [README_HEAD % dict(kinds,
                                files=build_file_list(rows, GROUPS, index),
                                code=build_code_list(outdir))]
    for prefix, title, blurb in GROUPS + [(None, 'Other', None)]:
        group = [r for r in rows if r[0] not in used and
                 (prefix is None or r[0].startswith(prefix))]
        if not group:
            continue
        lines.append('## %s\n\n%s' % (title, wrap(blurb)) if blurb
                     else '## %s\n' % title)
        for name, expect, note, exercises, size in group:
            used.add(name)
            lines.append(heading(outdir, name, expect))
            lines.append(wrap(note))
            lines.append(wrap(exercises))
    total = sum(r[4] for r in rows)
    lines.append('---\n')
    lines.append(wrap(
        '%d files, %d bytes total. Rebuild with `generate.py`: it writes the '
        'lossless cases itself with `vp8l.py`, and assembles everything in '
        '`cases/` through `webp_asm.py`, which hands the frame to '
        '`vp8_asm.py` and that to `vp8.py`.' % (len(rows), total)))
    write_files_index(outdir, rows)
    write_cases_index(outdir, rows)
    text = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines))
    with open(os.path.join(outdir, 'README.md'), 'w') as f:
        f.write(text)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    files = os.path.join(outdir, 'files')
    os.makedirs(files, exist_ok=True)
    for stale in os.listdir(files) if os.path.isdir(files) else []:
        os.remove(os.path.join(files, stale))
    rows = []
    failed = 0
    for name, expect, note, exercises, fn in CASES:
        try:
            data = fn()
        except Exception as e:  # noqa: BLE001
            print('FAILED to build %s: %s' % (name, e))
            failed += 1
            continue
        path = os.path.join(files, name + '.webp')
        with open(path, 'wb') as f:
            f.write(data)
        rows.append((name, expect, note, exercises, len(data)))
        print('%-40s %-7s %5d bytes' % (name, expect, len(data)))
    rows += lossy_parts.build(outdir)
    rows += build_cases(outdir)
    with open(os.path.join(outdir, 'expected.txt'), 'w') as f:
        for name, expect, _, _, _ in rows:
            f.write('%s|%s|%s\n' % (name, expect,
                                     'slow' if name in SLOW else ''))
    write_readme(outdir, rows)
    if failed:
        raise SystemExit('%d case(s) failed to build' % failed)
    return rows


if __name__ == '__main__':
    main()
