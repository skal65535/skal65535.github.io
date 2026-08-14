#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.

"""Assembles a lossy VP8 .webp from a text description of the bitstream.

    ./vp8_asm.py cases/lossy-coeff-cat3.txt files/lossy-coeff-cat3.webp

A case is a list of fields, one per line, '#' starting a comment. The names
are RFC 6386's, section by section, so a case reads against the spec rather
than against this file. Every field has a default, so a case says only what
it is about. Nothing is validated or clamped: a value too big for its field
just loses its top bits, which is usually the point.

  9.1  frame_type 0        0 is a key frame, 1 an interframe
       version 0           the 3-bit profile
       show_frame 1
       width 16            14 bits, with horizontal_scale in the top 2
       height 16
       horizontal_scale 0
       vertical_scale 0
       start_code 9d012a
  9.2  color_space 0
       clamping_type 0
  9.3  segmentation_enabled 0
       update_mb_segmentation_map 0
       update_segment_feature_data 0
       segment_feature_mode 1          1 absolute, 0 delta
       quantizer_update_value Q Q Q Q  one per segment
       lf_update_value F F F F
       segment_prob P P P
  9.4  filter_type 0                   1 is the simple filter
       loop_filter_level 0
       sharpness_level 0
       loop_filter_adj_enable 0
       mode_ref_lf_delta_update 0
       ref_frame_delta D D D D
       mb_mode_delta D D D D
  9.5  log2_nbr_of_DCT_partitions 0    so 1, 2, 4 or 8 partitions
  9.6  yac_qi 0
       ydc_delta D
       y2dc_delta D
       y2ac_delta D
       uvdc_delta D
       uvac_delta D
  9.8  refresh_entropy_probs 0
  9.9  coeff_prob TYPE BAND CTX INDEX VALUE     ('*' for all of one)
  9.10 mb_no_skip_coeff 0
       prob_skip_false P
  11   macroblock [count N]            opens one; count is not an RFC field
       segment_id 0
       mb_skip_coeff 0
       intra_y_mode DC_PRED
       intra_b_mode M M M M          16 in all, over any number
                                     of lines
       intra_chroma_mode DC_PRED
  13   coeffs BLOCK LEVEL...

'-' is the missing value: a field written as a bare 0 flag. It is not the
same as 0, which writes the flag *and* a zero value. The RFC splits those
into a _update flag, a _value and a _sign; one signed number stands for all
three here, and '-' clears the flag.

Two more directives have no RFC counterpart, because what they describe is
a frame lying about itself:

  macroblock_count N          how many to write, when that should differ
                              from what width and height call for
  raw       part0 HEX | token N HEX
  patch     partition0_size N | part0_bytes N | part_size I N
            truncate N | truncate tokens N

'macroblock_count' decides how much macroblock data is written at all;
'raw' appends bytes a partition's own syntax cannot produce; and 'patch'
rewrites what the frame said about itself once it is assembled, so the
declared sizes can be made to disagree with the data behind them.
Offsets there are within the VP8 chunk, not the RIFF file, which is rebuilt
around whatever is left. 'partition0_size' changes only what the frame tag
claims, while 'part0_bytes' shortens partition 0 for real and updates the
tag to match, leaving the token partitions where they are.

'macroblock' opens one; the fields of section 11 and the coefficients fill
in the one it opened. Macroblocks are laid out in raster order and any the
frame still needs at the end are added with default everything.

BLOCK is y2 for the block of luma DCs, y[0]..y[15], u[0]..u[3] or
v[0]..v[3] for the rest, and y[*], u[*], v[*] or uv[*] for all of a kind at
once. LEVELs are the quantized levels in coding (zigzag) order from
position 0, trailing zeros implied; 'N:LEVEL' jumps to position N first.
The luma blocks of a 16x16-mode macroblock start at position 1, and their
position 0 must be zero: that coefficient lives in y2.

Modes are the RFC's: DC_PRED, V_PRED, H_PRED, TM_PRED and B_PRED for
intra_y_mode and intra_chroma_mode, B_DC_PRED, B_TM_PRED, B_VE_PRED,
B_HE_PRED, B_RD_PRED, B_VR_PRED, B_LD_PRED, B_VL_PRED, B_HD_PRED and
B_HU_PRED for intra_b_mode. The _PRED may be left off, and a plain number
works too.
"""

import os
import re
import sys

import vp8

YMODES = {'DC': vp8.DC_PRED, 'V': vp8.V_PRED, 'H': vp8.H_PRED,
          'TM': vp8.TM_PRED, 'B': vp8.B_PRED}
BMODES = {'B_DC': vp8.B_DC_PRED, 'B_TM': vp8.B_TM_PRED, 'B_VE': vp8.B_VE_PRED,
          'B_HE': vp8.B_HE_PRED, 'B_RD': vp8.B_RD_PRED, 'B_VR': vp8.B_VR_PRED,
          'B_LD': vp8.B_LD_PRED, 'B_VL': vp8.B_VL_PRED, 'B_HD': vp8.B_HD_PRED,
          'B_HU': vp8.B_HU_PRED}

# RFC 6386 name -> (vp8.Frame attribute, how many values). 1 is a plain
# number; more than that is a list whose entries may be '-'.
FRAME_FIELDS = {
    'version': ('version', 1),
    'show_frame': ('show', 1),
    'width': ('width', 1),
    'height': ('height', 1),
    'horizontal_scale': ('xscale', 1),
    'vertical_scale': ('yscale', 1),
    'color_space': ('colorspace', 1),
    'clamping_type': ('clamp', 1),
    'segmentation_enabled': ('use_segment', 1),
    'update_mb_segmentation_map': ('update_map', 1),
    'update_segment_feature_data': ('update_data', 1),
    'segment_feature_mode': ('absolute_delta', 1),
    'quantizer_update_value': ('segment_quant', 4),
    'lf_update_value': ('segment_filter', 4),
    'segment_prob': ('segment_probs', 3),
    'filter_type': ('simple', 1),
    'loop_filter_level': ('filter_level', 1),
    'sharpness_level': ('sharpness', 1),
    'loop_filter_adj_enable': ('use_lf_delta', 1),
    'mode_ref_lf_delta_update': ('update_lf_delta', 1),
    'ref_frame_delta': ('ref_lf_delta', 4),
    'mb_mode_delta': ('mode_lf_delta', 4),
    'log2_nbr_of_DCT_partitions': ('parts_log2', 1),
    'yac_qi': ('base_q', 1),
    'refresh_entropy_probs': ('refresh_proba', 1),
    'mb_no_skip_coeff': ('use_skip_proba', 1),
    'prob_skip_false': ('skip_proba', 1),
    # Not an RFC field: how many macroblocks to write, when that should
    # disagree with the number width and height call for.
    'macroblock_count': ('num_mbs', 1),
}
# The RFC's names for the five deltas, against vp8.py's bitstream order.
QUANT_DELTAS = dict(zip(('ydc_delta', 'y2dc_delta', 'y2ac_delta',
                         'uvdc_delta', 'uvac_delta'), vp8.QUANT_DELTAS))
MB_FIELDS = {'segment_id': 'segment', 'mb_skip_coeff': 'skip'}


HEADER_KEYS = {'note': True, 'expect': True, 'exercises': True,
               'roundtrip': False, 'slow': False}   # value is 'required?'


class AsmError(Exception):
    pass


def parse_header(text, what='case'):
    """The keyed comment header a case file starts with.

    '# key: value', continued by the indented comment lines under it. The
    indent is what separates the two, so a continuation may itself contain
    a colon. Everything from the first line that is not a comment is the
    case itself.
    """
    fields, key = {}, None
    for line in text.splitlines():
        if not line.startswith('#'):
            break
        m = re.match(r'# ?(\w+):\s*(.*)$', line.rstrip())
        if m:
            key = m.group(1)
            if key not in HEADER_KEYS:
                raise AsmError('%s: unknown header key %r' % (what, key))
            fields[key] = m.group(2)
        elif key is not None:
            fields[key] += ' ' + line.lstrip('#').strip()
    for key, required in HEADER_KEYS.items():
        if required and key not in fields:
            raise AsmError('%s: no "# %s:" line' % (what, key))
    if fields['expect'] not in ('ok', 'reject'):
        raise AsmError('%s: expect is %r, not ok or reject'
                       % (what, fields['expect']))
    if fields.setdefault('roundtrip', 'yes') not in ('yes', 'no'):
        raise AsmError('%s: roundtrip is %r, not yes or no'
                       % (what, fields['roundtrip']))
    return fields


def block_names(mb):
    """(name, levels) for every block of a macroblock that carries them.

    The other half of Assembler.blocks(): this spells the names that one
    reads, so both directions of the syntax live here.
    """
    out = [('y2', mb.y2)] if not mb.is_i4x4 else []
    out += [('y[%d]' % i, b) for i, b in enumerate(mb.y)]
    out += [('u[%d]' % i, b) for i, b in enumerate(mb.u)]
    out += [('v[%d]' % i, b) for i, b in enumerate(mb.v)]
    return out


class Assembler:
    """Turns a case into a vp8.Frame, one line at a time."""

    def __init__(self):
        self.f = vp8.Frame()
        self.mb = None                 # the macroblock 'macroblock' opened
        self.mb_repeat = 1
        self.bmodes_seen = False
        self.patches = []
        self.line = 0

    # -- value parsing -------------------------------------------------------

    def fail(self, msg):
        raise AsmError('line %d: %s' % (self.line, msg))

    def num(self, tok, what='value'):
        try:
            return int(tok, 0)
        except ValueError:
            self.fail('bad %s %r' % (what, tok))

    def opt(self, tok, what='value'):
        """A number, or None for the '-' that means 'field not present'."""
        return None if tok == '-' else self.num(tok, what)

    def hexbytes(self, tok):
        try:
            return bytes.fromhex(tok)
        except ValueError:
            self.fail('%r is not an even number of hex digits' % tok)

    def mode(self, tok, table, what):
        name = tok.upper()
        if name.endswith('_PRED'):
            name = name[:-len('_PRED')]
        if name in table:
            return table[name]
        if tok.lstrip('-').isdigit():
            return int(tok)
        self.fail('unknown %s %r' % (what, tok))

    def one(self, args, what):
        """The single value a scalar field takes."""
        return self.take(args, 1, what)[0]

    def take(self, args, n, what):
        if len(args) != n:
            self.fail('%s needs %d values, got %d' % (what, n, len(args)))
        return args

    # -- fields --------------------------------------------------------------

    def set_frame_field(self, name, args):
        attr, n = FRAME_FIELDS[name]
        if n == 1:
            setattr(self.f, attr, self.num(self.one(args, name), name))
        else:
            setattr(self.f, attr,
                    [self.opt(t, name) for t in self.take(args, n, name)])

    def do_start_code(self, args):
        self.f.signature = self.hexbytes(self.one(args, 'start_code'))

    def do_frame_type(self, args):
        # The RFC's polarity: 0 is a key frame, which is all libwebp decodes.
        self.f.keyframe = 1 - self.num(self.one(args, 'frame_type'),
                                       'frame_type')

    def do_coeff_prob(self, args):
        args = self.take(args, 5, 'coeff_prob')
        v = self.num(args[4], 'probability')
        limits = ((vp8.NUM_TYPES, 'type'), (vp8.NUM_BANDS, 'band'),
                  (vp8.NUM_CTX, 'context'), (vp8.NUM_PROBAS, 'index'))
        picked = []
        for tok, (limit, what) in zip(args[:4], limits):
            if tok == '*':
                picked.append(range(limit))
                continue
            n = self.num(tok, what)
            if not 0 <= n < limit:
                self.fail('%s %d is outside 0..%d' % (what, n, limit - 1))
            picked.append([n])
        for t in picked[0]:
            for b in picked[1]:
                for c in picked[2]:
                    for i in picked[3]:
                        self.f.proba_update[(t, b, c, i)] = v

    # -- macroblocks ---------------------------------------------------------

    def do_macroblock(self, args):
        self.flush_mb()
        self.mb = vp8.MB()
        self.bmodes_seen = False
        if args:
            if args[0] != 'count':
                self.fail('macroblock takes only "count N", not %r' % args[0])
            self.mb_repeat = self.num(self.one(args[1:], 'count'), 'count')

    def current(self, what):
        if self.mb is None:
            self.fail('%s before any macroblock' % what)
        return self.mb

    def do_intra_y_mode(self, args):
        self.current('intra_y_mode').ymode = self.mode(
            self.one(args, 'intra_y_mode'), YMODES, 'intra_y_mode')

    def do_intra_chroma_mode(self, args):
        self.current('intra_chroma_mode').uvmode = self.mode(
            self.one(args, 'intra_chroma_mode'), YMODES,
            'intra_chroma_mode')

    def do_intra_b_mode(self, args):
        """The 16 modes, in as many lines as suits: four rows of four reads
        the way they are laid out."""
        mb = self.current('intra_b_mode')
        if not self.bmodes_seen:
            mb.bmodes, self.bmodes_seen = [], True
        mb.bmodes += [self.mode(t, BMODES, 'intra_b_mode') for t in args]
        if len(mb.bmodes) > 16:
            self.fail('intra_b_mode has %d modes, 16 is the most a '
                      'macroblock has' % len(mb.bmodes))

    def do_coeffs(self, args):
        self.current('coeffs')
        if not args:
            self.fail('coeffs needs a block name')
        blocks = self.blocks(args[0])
        levels = [0] * 16
        pos = 0
        for tok in args[1:]:
            if ':' in tok:
                at, _, tok = tok.partition(':')
                pos = self.num(at, 'position')
            if not 0 <= pos < 16:
                self.fail('position %d is outside 0..15' % pos)
            levels[pos] = self.num(tok, 'level')
            pos += 1
        for b in blocks:
            b[:] = levels

    def blocks(self, name):
        """The coefficient arrays a block name stands for."""
        mb = self.mb
        if name == 'y2':
            if mb.is_i4x4:
                self.fail('a B_PRED macroblock has no y2 block')
            return [mb.y2]
        kind, _, index = name.partition('[')
        arrays = {'y': mb.y, 'u': mb.u, 'v': mb.v, 'uv': mb.u + mb.v}
        if kind not in arrays or not index.endswith(']'):
            self.fail('unknown block %r' % name)
        index = index[:-1]
        if index == '*':
            return arrays[kind]
        if kind == 'uv' or not index.isdigit():
            self.fail('%r needs a plain block number, or *' % name)
        if int(index) >= len(arrays[kind]):
            self.fail('%s has only %d blocks' % (kind, len(arrays[kind])))
        return [arrays[kind][int(index)]]

    # -- lying about the frame -----------------------------------------------

    def do_raw(self, args):
        what = args[0] if args else ''
        if what == 'token':
            i, data = self.take(args[1:], 2, 'raw token')
            self.f.raw_tokens[self.num(i, 'partition')] = self.hexbytes(data)
        elif what == 'part0':
            self.f.raw_part0 += self.hexbytes(self.one(args[1:], 'raw part0'))
        else:
            self.fail('raw takes part0 or token N, not %r' % what)

    def do_patch(self, args):
        what = args[0] if args else ''
        if what == 'part_size':
            i, n = self.take(args[1:], 2, 'patch part_size')
            self.patches.append(('part_size', self.num(i, 'index'),
                                 self.num(n, 'size')))
        elif what == 'truncate' and args[1:2] == ['tokens']:
            n = self.num(self.one(args[2:], 'patch truncate tokens'),
                         'value')
            self.patches.append(('truncate_tokens', n))
        elif what in ('partition0_size', 'part0_bytes', 'truncate'):
            self.patches.append(
                (what, self.num(self.one(args[1:], 'patch ' + what), 'value')))
        else:
            self.fail('unknown patch %r' % what)

    # -- driving -------------------------------------------------------------

    def flush_mb(self):
        if self.mb is not None:
            if self.bmodes_seen and len(self.mb.bmodes) != 16:
                self.fail('intra_b_mode has %d modes, a macroblock needs 16'
                          % len(self.mb.bmodes))
            self.f.mbs += [self.mb] * self.mb_repeat
        self.mb, self.mb_repeat = None, 1

    def check_mbs(self):
        """The two things the bitstream has no way to say."""
        for n, mb in enumerate(self.f.mbs):
            if not mb.is_i4x4:
                for i, b in enumerate(mb.y):
                    if b[0]:
                        raise AsmError('macroblock %d: y[%d] position 0 '
                                       'belongs to y2 in a 16x16-mode '
                                       'macroblock' % (n, i))
            if mb.skip and not self.f.use_skip_proba:
                raise AsmError('macroblock %d: mb_skip_coeff needs '
                               'mb_no_skip_coeff to be signalled with' % n)

    def feed(self, text):
        for lineno, line in enumerate(text.splitlines(), 1):
            self.line = lineno
            args = line.split('#')[0].split()
            if not args:
                continue
            name, rest = args[0], args[1:]
            if name in FRAME_FIELDS:
                self.set_frame_field(name, rest)
            elif name in QUANT_DELTAS:
                self.f.dq[QUANT_DELTAS[name]] = \
                    self.opt(self.one(rest, name), name)
            elif name in MB_FIELDS:
                setattr(self.current(name), MB_FIELDS[name],
                        self.num(self.one(rest, name), name))
            else:
                handler = getattr(self, 'do_' + name, None)
                if handler is None:
                    self.fail('unknown field %r' % name)
                handler(rest)
        self.flush_mb()
        return self.f

    def finish(self):
        """The assembled VP8 chunk, patches and all."""
        f = self.f
        want = f.mb_count()
        f.mbs += [vp8.MB() for _ in range(want - len(f.mbs))]
        del f.mbs[want:]
        self.check_mbs()
        data = bytearray(vp8.assemble(f))
        for patch in self.patches:
            if patch[0] == 'partition0_size':
                vp8.set_partition0_size(data, patch[1])
            elif patch[0] == 'part_size':
                at = vp8.size_table_offset(data) + 3 * patch[1]
                data[at:at + 3] = (patch[2] & 0xffffff).to_bytes(3, 'little')
            elif patch[0] == 'truncate':
                del data[patch[1]:]
            elif patch[0] == 'part0_bytes':
                end = vp8.size_table_offset(data)
                keep = min(patch[1], end - vp8.FRAME_HEADER_SIZE)
                del data[vp8.FRAME_HEADER_SIZE + keep:end]
                vp8.set_partition0_size(data, keep)
            elif patch[0] == 'truncate_tokens':
                del data[vp8.size_table_offset(data) + patch[1]:]
        return bytes(data)


def assemble_text(text):
    """The .webp bytes for one case."""
    asm = Assembler()
    asm.feed(text)
    return vp8.wrap_webp(asm.finish())


def main(argv):
    if not 2 <= len(argv) <= 3:
        print(__doc__.strip().split('\n\n')[0], file=sys.stderr)
        print('usage: %s <case.txt> [<out.webp>]'
              % os.path.basename(argv[0]), file=sys.stderr)
        return 1
    src = argv[1]
    dst = argv[2] if len(argv) > 2 else os.path.splitext(src)[0] + '.webp'
    with open(src) as f:
        text = f.read()
    try:
        data = assemble_text(text)
    except AsmError as e:
        print('%s:%s' % (src, e), file=sys.stderr)
        return 1
    with open(dst, 'wb') as f:
        f.write(data)
    print('%-40s %5d bytes' % (dst, len(data)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
