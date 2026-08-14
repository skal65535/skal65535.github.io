#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.

"""Every keyword a case can use, with the range of every value.

    ./src/grammar.py            the whole thing as JSON

The three assemblers own the keywords; this owns what a keyword's values
may be, which is the one thing their dictionaries do not carry. A field's
range is the width of the bitstream field behind it -- 'loop_filter_level'
is 6 bits because that is what write_filter_header() writes -- so this file
is the only place the two are stated together, and SYNTAX.md is generated
from it.

Ranges are what the *field* holds, not what a decoder accepts. Writing past
one is not refused, it just loses the top bits, so a generator that wants a
malformed file should go past them on purpose.

Each entry is:

  arity     how many values follow: a number, or 'list' for any count
  values    one entry per value position, each {'kind': ...}
  scope     what the keyword belongs to, and so where it may appear
  doc       one line

and a value is one of

  {'kind': 'uint',  'bits': n, 'min': 0, 'max': 2**n - 1}
  {'kind': 'sint',  'bits': n, 'min': -(2**n - 1), 'max': 2**n - 1}
  {'kind': 'enum',  'names': [...]}         a name, or the number behind it
  {'kind': 'hex'}                           an even number of hex digits
  {'kind': 'opt', 'of': {...}}              that, or '-' for "field absent"
  {'kind': 'token'}                         a bare word, listed in 'doc'
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vp8
import vp8_asm
import vp8l
import vp8l_asm
import webp_asm


def uint(bits):
    return {'kind': 'uint', 'bits': bits, 'min': 0, 'max': (1 << bits) - 1}


def sint(bits):
    """A magnitude of 'bits' plus a sign, as VP8PutSignedBits() writes it."""
    return {'kind': 'sint', 'bits': bits, 'min': -(1 << bits) + 1,
            'max': (1 << bits) - 1}


def enum(names):
    return {'kind': 'enum', 'names': sorted(names)}


def opt(of):
    return {'kind': 'opt', 'of': of}


HEX = {'kind': 'hex'}
TOKEN = {'kind': 'token'}

# --------------------------------------------------------------------------
# The lossy frame, RFC 6386's names. Widths are what vp8.py writes.

LOSSY = {
    # 9.1, the uncompressed frame tag
    'frame_type': (uint(1), '0 is a key frame, 1 an interframe'),
    'version': (uint(3), 'the profile'),
    'show_frame': (uint(1), ''),
    'width': (uint(14), ''),
    'height': (uint(14), ''),
    'horizontal_scale': (uint(2), 'the top 2 bits of the width field'),
    'vertical_scale': (uint(2), ''),
    'start_code': (HEX, "the three bytes after the tag, normally 9d012a"),
    # 9.2
    'color_space': (uint(1), ''),
    'clamping_type': (uint(1), ''),
    # 9.3, segmentation
    'segmentation_enabled': (uint(1), ''),
    'update_mb_segmentation_map': (uint(1), ''),
    'update_segment_feature_data': (uint(1), ''),
    'segment_feature_mode': (uint(1), '1 absolute, 0 delta'),
    'quantizer_update_value': (opt(sint(7)), 'one per segment'),
    'lf_update_value': (opt(sint(6)), 'one per segment'),
    'segment_prob': (opt(uint(8)), 'the three tree probabilities'),
    # 9.4, the loop filter
    'filter_type': (uint(1), '1 is the simple filter'),
    'loop_filter_level': (uint(6), ''),
    'sharpness_level': (uint(3), ''),
    'loop_filter_adj_enable': (uint(1), ''),
    'mode_ref_lf_delta_update': (uint(1), ''),
    'ref_frame_delta': (opt(sint(6)), 'one per reference frame'),
    'mb_mode_delta': (opt(sint(6)), 'one per mode class'),
    # 9.5
    'log2_nbr_of_DCT_partitions': (uint(2), 'so 1, 2, 4 or 8 partitions'),
    # 9.6, the quantizer
    'yac_qi': (uint(7), ''),
    'ydc_delta': (opt(sint(4)), ''),
    'y2dc_delta': (opt(sint(4)), ''),
    'y2ac_delta': (opt(sint(4)), ''),
    'uvdc_delta': (opt(sint(4)), ''),
    'uvac_delta': (opt(sint(4)), ''),
    # 9.8 - 9.10
    'refresh_entropy_probs': (uint(1), 'parsed and dropped by libwebp'),
    'mb_no_skip_coeff': (uint(1), ''),
    'prob_skip_false': (uint(8), ''),
}

LOSSY_ARITY = {'quantizer_update_value': 4, 'lf_update_value': 4,
               'segment_prob': 3, 'ref_frame_delta': 4, 'mb_mode_delta': 4}

# The keywords that are not one plain field.
LOSSY_EXTRA = {
    'coeff_prob': (4, [enum(['*']), enum(['*']), enum(['*']), enum(['*']),
                       uint(8)],
                   'frame',
                   'type, band, context and index, then the probability; '
                   "'*' stands for every value of that one"),
    'macroblock': (0, [], 'frame',
                   "opens one; 'macroblock count N' writes it N times"),
    'macroblock_count': (1, [uint(16)], 'frame',
                         'how many to write, when that should differ from '
                         'what width and height call for'),
    'segment_id': (1, [uint(2)], 'macroblock', ''),
    'mb_skip_coeff': (1, [uint(1)], 'macroblock',
                      'needs mb_no_skip_coeff signalled with it'),
    'intra_y_mode': (1, [enum(vp8_asm.YMODES)], 'macroblock',
                     'B_PRED makes the macroblock 4x4'),
    'intra_chroma_mode': (1, [enum(k for k in vp8_asm.YMODES if k != 'B')],
                          'macroblock', ''),
    'intra_b_mode': ('list', [enum(vp8_asm.BMODES)], 'macroblock',
                     '16 in all, over any number of lines'),
    'coeffs': ('list', [TOKEN, {'kind': 'sint', 'bits': 11, 'min': -2114,
                                'max': 2114}],
               'macroblock',
               'a block name -- y2, y[0]..y[15], u[0]..u[3], v[0]..v[3], or '
               "y[*] u[*] v[*] uv[*] -- then levels in zigzag order; 'N:V' "
               'jumps to position N'),
    'raw': ('list', [TOKEN, HEX], 'frame',
            "'raw part0 HEX' or 'raw token N HEX': bytes a partition's own "
            'syntax cannot produce'),
    'patch': ('list', [TOKEN, uint(24)], 'frame',
              'partition0_size N | part0_bytes N | part_size I N | '
              'truncate N | truncate tokens N -- rewrites what the frame '
              'says about itself once it is assembled'),
}

# --------------------------------------------------------------------------
# The lossless image. Widths are what vp8l.py writes.

LOSSLESS = {
    'lossless': (0, [], 'image', 'this case is a VP8L image, not a VP8 frame'),
    'magic': (1, [uint(8)], 'image',
              'the signature byte, 0x2f; anything else is refused'),
    'width': (1, [uint(14)], 'image', 'the field holds width - 1'),
    'height': (1, [uint(14)], 'image', ''),
    'version': (1, [uint(3)], 'image', 'anything but 0 is refused'),
    'alpha_is_used': (1, [uint(1)], 'image', 'a hint; libwebp drops it'),
    'transforms': ('list', [enum(vp8l_asm.TRANSFORMS)], 'image',
                   'in the order written. A type may legally appear once'),
    'predictor_bits': (1, [{'kind': 'uint', 'bits': 3, 'min': 2, 'max': 9}],
                       'image', 'log2 of the tile size'),
    'predictor_tiles': ('list', [uint(4)], 'image',
                        'one predictor index per tile, read as 4 bits'),
    'cross_color_bits': (1, [{'kind': 'uint', 'bits': 3, 'min': 2, 'max': 9}],
                         'image', ''),
    'cross_color_tiles': ('list', [uint(32)], 'image',
                          'one multiplier triple per tile, as a color'),
    'palette_colors': ('list', [uint(32)], 'image',
                       'up to 256; the stream carries their per-byte deltas'),
    'cache_bits': (1, [uint(4)], 'image',
                   'absent by default; 1..11 is the legal range'),
    'meta_bits': (1, [{'kind': 'uint', 'bits': 3, 'min': 2, 'max': 9}],
                  'image', 'the entropy image, absent by default'),
    'meta_tiles': ('list', [uint(16)], 'image',
                   'one Huffman-group index per tile'),
    'group': (0, [], 'image', 'opens another group of five codes'),
    'group_count': (1, [uint(16)], 'image',
                    'how many groups to write, when that should differ from '
                    'what the entropy image asks for'),
    'subimage': (1, [enum(vp8l_asm.SUBIMAGES)], 'image',
                 "aims the 'code' lines that follow at one sub-image"),
    'subimage_cache_bits': (2, [enum(vp8l_asm.SUBIMAGES), uint(4)], 'image',
                            "that sub-image's own color cache"),
    'code': ('list', [enum(vp8l_asm.CODE_NAMES), TOKEN], 'group',
             'one Huffman code; see the forms below'),
    'pixels': ('list', [TOKEN], 'image',
               "green-code symbols, 'cache N', or 'copy LENGTH PLANE'"),
    'argb': ('list', [uint(32)], 'image',
             "whole pixels; 'V xN' repeats one"),
}

# 'code NAME <form> ...' -- the second word picks what the rest means.
CODE_FORMS = {
    'simple': ('list', [uint(8)],
               'one or two symbols, written raw and never range-checked'),
    'simple1': (1, [uint(1)],
                'one symbol in a 1-bit field rather than 8'),
    'lengths': ('list', [uint(4)],
                "a length per symbol, positional from 0; 'N:L' jumps to "
                'symbol N, trailing zeros implied'),
    'codelen': ('list', [TOKEN],
                'the code-length stream itself: a length 0..15, or 16xN, '
                '17xN, 18xN for a run of N'),
    'cl_lengths': ('list', [uint(3)],
                   'the code-length code itself, as symbol:length pairs; '
                   'without it one is built from how often each code-length '
                   'symbol is used, which two encoders may do differently'),
    'num_codes': (1, [{'kind': 'uint', 'bits': 5, 'min': 4, 'max': 19}],
                  'how many of the 19 code-length codes to declare'),
    'max_symbol': (1, [uint(16)], 'the optional early stop'),
    'complex': (0, [], 'the normal form, over whatever the pixels need'),
}

PIXEL_ITEMS = {
    'V': ('a green-code symbol; V xN repeats it', [uint(16)]),
    'cache': ('the color-cache index N', [uint(11)]),
    'copy': ('a back-reference: the length, then the plane code',
             [uint(16), uint(16)]),
}

# --------------------------------------------------------------------------
# The RIFF container, RFC 9649's names.

CONTAINER = {
    'chunks': ('list', [TOKEN], 'container',
               'the fourccs to write, in order; listing one twice is allowed'),
    'riff_size': (1, [uint(32)], 'container',
                  'what the RIFF header claims, when that should not be what '
                  'the file holds'),
    'chunk_size': (2, [TOKEN, uint(32)], 'container', 'the same lie, per '
                   'chunk; it also decides the pad byte'),
    'payload': (2, [TOKEN, HEX], 'container',
                'the bytes of a chunk this file has no builder for'),
    'trailing': (1, [HEX], 'container', 'bytes after the last chunk'),
    'vp8x_reserved': (1, [uint(24)], 'container',
                      'the two reserved bits and the reserved byte, raw'),
    'canvas_width_minus_one': (1, [uint(24)], 'container', ''),
    'canvas_height_minus_one': (1, [uint(24)], 'container', ''),
    'alph_raw': (1, [uint(32)], 'container',
                 'that many bytes of uncompressed alpha plane'),
    'alph_data': (1, [HEX], 'container',
                  'the bytes after the ALPH header byte, spelled out'),
    'alph_plane': (0, [], 'container',
                   'opens a block: the ALPH payload as a lossless image '
                   'stream, with no signature of its own'),
}

# The animation chunks, and the frames they hold.
ANIMATION = {
    'frame': (0, [], 'animation',
              'opens a block: one ANMF chunk, with its own image and its own '
              'chunk list'),
    'loop_count': (1, [uint(16)], 'animation', '0 means forever'),
    'background_color': (1, [uint(32)], 'animation',
                         'ARGB; read back out again and never drawn'),
    'frame_x': (1, [uint(24)], 'animation',
                'the offset field: the pixel offset is twice it'),
    'frame_y': (1, [uint(24)], 'animation', ''),
    'frame_width_minus_one': (1, [uint(24)], 'animation',
                              "default is the frame's own image"),
    'frame_height_minus_one': (1, [uint(24)], 'animation', ''),
    'frame_duration': (1, [uint(24)], 'animation', 'milliseconds'),
    'disposal_method': (1, [uint(1)], 'animation',
                        "1 clears the frame's area to the background"),
    'blending_method': (1, [uint(1)], 'animation', "1 is 'do not blend'"),
    'frame_reserved': (1, [uint(6)], 'animation',
                       'the six bits above those two'),
}
CONTAINER.update(ANIMATION)
for _name in webp_asm.FLAGS:
    CONTAINER[_name] = (1, [uint(1)], 'container', 'a VP8X feature flag')
for _name in webp_asm.ALPH_FIELDS:
    CONTAINER[_name] = (1, [uint(2)], 'container', "an ALPH header field")


def build():
    """The whole grammar, as plain data."""
    keywords = {}

    def add(name, arity, values, scope, doc):
        keywords[name] = {'arity': arity, 'values': values, 'scope': scope,
                          'doc': doc}

    for name, (value, doc) in LOSSY.items():
        arity = LOSSY_ARITY.get(name, 1)
        add(name, arity, [value] * arity, 'frame', doc)
    for name, (arity, values, scope, doc) in LOSSY_EXTRA.items():
        add(name, arity, values, scope, doc)
    for name, (arity, values, scope, doc) in LOSSLESS.items():
        if name in keywords:            # width/height/version are both layers'
            keywords[name]['scope'] = 'frame, image'
            continue
        add(name, arity, values, scope, doc)
    for name, (arity, values, scope, doc) in CONTAINER.items():
        add(name, arity, values, scope, doc)

    return {
        'about': 'webp-corners case syntax. Ranges are what the bitstream '
                 'field holds, not what a decoder accepts.',
        'header_keys': {k: ('required' if v else 'optional')
                        for k, v in vp8_asm.HEADER_KEYS.items()},
        'keywords': keywords,
        'code_forms': {name: {'arity': arity, 'values': values, 'doc': doc}
                       for name, (arity, values, doc) in CODE_FORMS.items()},
        'pixel_items': {name: {'doc': doc, 'values': values}
                        for name, (doc, values) in PIXEL_ITEMS.items()},
        'enums': {
            'intra_y_mode': sorted(vp8_asm.YMODES),
            'intra_b_mode': sorted(vp8_asm.BMODES),
            'transforms': sorted(vp8l_asm.TRANSFORMS),
            'subimage': sorted(vp8l_asm.SUBIMAGES),
            'code': list(vp8l_asm.CODE_NAMES),
        },
        'constants': {
            'MAX_CACHE_BITS': vp8l.MAX_CACHE_BITS,
            'MAX_ALLOWED_CODE_LENGTH': vp8l.MAX_ALLOWED_CODE_LENGTH,
            'CODE_LENGTH_CODES': vp8l.CODE_LENGTH_CODES,
            'NUM_LITERAL_CODES': vp8l.NUM_LITERAL_CODES,
            'NUM_LENGTH_CODES': vp8l.NUM_LENGTH_CODES,
            'NUM_DISTANCE_CODES': vp8l.NUM_DISTANCE_CODES,
            'MIN_TRANSFORM_BITS': vp8l.MIN_TRANSFORM_BITS,
            'MIN_HUFFMAN_BITS': vp8l.MIN_HUFFMAN_BITS,
            'NUM_TYPES': vp8.NUM_TYPES,
            'NUM_BANDS': vp8.NUM_BANDS,
            'NUM_CTX': vp8.NUM_CTX,
            'NUM_PROBAS': vp8.NUM_PROBAS,
        },
    }


# Which module's docstring is the reference for a keyword of that scope.
OWNER = {'frame': vp8_asm, 'macroblock': vp8_asm, 'frame, image': vp8l_asm,
         'image': vp8l_asm, 'group': vp8l_asm, 'container': webp_asm,
         'animation': webp_asm}


def check():
    """The three places a keyword is written down have to agree: the
    assembler that takes it, the tables here, and the docstring that is that
    tool's own reference."""
    known = set(vp8_asm.FRAME_FIELDS) | set(vp8_asm.QUANT_DELTAS) | \
        set(vp8_asm.MB_FIELDS) | set(webp_asm.DIRECTIVES) | \
        set(webp_asm.BLOCKS) | \
        set(vp8l_asm.NUM_FIELDS) | set(vp8l_asm.LIST_FIELDS)
    for cls in (vp8_asm.Assembler, vp8l_asm.Assembler):
        known |= {n[3:] for n in dir(cls) if n.startswith('do_')}
    keywords = build()['keywords']
    bad = 0
    for what, names in (('not described here',
                         sorted(known - set(keywords))),
                        ('described here but unknown to the assemblers',
                         sorted(set(keywords) - known))):
        if names:
            print('%s: %s' % (what, ' '.join(names)), file=sys.stderr)
            bad += len(names)
    # a keyword is listed in the docstring of the tool that owns it
    undocumented = sorted(
        name for name, entry in keywords.items()
        if not re.search(r'(?m)^ *(\d+(\.\d+)? +)?%s\b' % re.escape(name),
                         OWNER[entry['scope']].__doc__))
    if undocumented:
        print('missing from their module docstring: %s'
              % ' '.join(undocumented), file=sys.stderr)
        bad += len(undocumented)
    return bad


def main(argv):
    if len(argv) > 1 and argv[1] == '--check':
        return 1 if check() else 0
    json.dump(build(), sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
