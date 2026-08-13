#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.

"""Turns a lossy VP8 .webp back into the text vp8_asm.py assembles.

    ./vp8_dis.py picture.webp            # print the case
    ./vp8_dis.py --check picture.webp    # dump, reassemble, compare

--check is the verification: a file that survives a round trip byte for
byte pins every field, every probability and every coefficient, since the
boolean coder turns any disagreement into different bytes almost at once.
Run it over real cwebp output and the writer is checked against libwebp
rather than against itself.

The boolean decoder is a port of VP8GetBit()/VP8GetSigned()
(src/utils/bit_reader_inl_utils.h) and the parsing follows VP8GetHeaders(),
ParseIntraMode() and ParseResiduals(). It is meant for the small frames of
a torture corpus, not for speed.
"""

import itertools
import struct
import sys

import vp8
import vp8_asm
from vp8_tables import BANDS, BMODES_PROBA, CAT3456, COEFFS_UPDATE_PROBA


class Truncated(Exception):
    pass


class BoolReader:
    """The VP8 boolean decoder, one byte at a time."""

    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.range = 255 - 1           # the decoder keeps range - 1
        self.value = 0
        self.bits = -8
        self.eof = 0
        self._load()

    def _load(self):
        """VP8LoadFinalBytes(): past the end, feed zeroes and remember it."""
        if self.pos < len(self.data):
            self.value = (self.value << 8) | self.data[self.pos]
            self.pos += 1
            self.bits += 8
        elif not self.eof:
            self.value <<= 8
            self.bits += 8
            self.eof = 1
        else:
            self.bits = 0

    def get_bit(self, prob):
        if self.bits < 0:
            self._load()
        pos = self.bits
        split = (self.range * prob) >> 8
        bit = 1 if (self.value >> pos) > split else 0
        if bit:
            rng = self.range - split
            self.value -= (split + 1) << pos
        else:
            rng = split + 1
        shift = 8 - rng.bit_length()
        self.bits -= shift
        self.range = (rng << shift) - 1
        return bit

    def get_signed(self, v):
        """VP8GetSigned(): the sign of a coefficient, fused with its renorm."""
        if self.bits < 0:
            self._load()
        pos = self.bits
        split = self.range >> 1
        bit = 1 if (self.value >> pos) > split else 0
        self.bits -= 1
        if bit:
            self.range = (self.range - 1) | 1
            self.value -= (split + 1) << pos
        else:
            self.range |= 1
        return -v if bit else v

    def get_value(self, n):
        v = 0
        for i in reversed(range(n)):
            v |= self.get_bit(0x80) << i
        return v

    def get_flagged(self, n):
        return self.get_value(n) if self.get_bit(0x80) else None

    def get_flagged_signed(self, n):
        if not self.get_bit(0x80):
            return None
        v = self.get_value(n)
        return -v if self.get_bit(0x80) else v

    def check(self):
        if self.eof:
            raise Truncated('ran past the end of a partition')


def get_large_value(br, p):
    """GetLargeValue(), src/dec/vp8_dec.c."""
    if not br.get_bit(p[3]):
        if not br.get_bit(p[4]):
            return 2
        return 3 + br.get_bit(p[5])
    if not br.get_bit(p[6]):
        if not br.get_bit(p[7]):
            return 5 + br.get_bit(159)
        return 7 + 2 * br.get_bit(165) + br.get_bit(145)
    bit1 = br.get_bit(p[8])
    cat = 2 * bit1 + br.get_bit(p[9 + bit1])
    v = 0
    for prob in CAT3456[cat]:
        v += v + br.get_bit(prob)
    return v + 3 + (8 << cat)


def get_coeffs(br, probas, ctx, first):
    """GetCoeffsFast(): the levels in coding order, and the non-zero flag."""
    out = [0] * 16
    n = first
    p = probas[BANDS[n]][ctx]
    while n < 16:
        if not br.get_bit(p[0]):
            break
        while not br.get_bit(p[1]):
            n += 1
            if n == 16:
                return out, 1
            p = probas[BANDS[n]][0]
        nxt = probas[BANDS[n + 1]]
        if not br.get_bit(p[2]):
            v, p = 1, nxt[1]
        else:
            v, p = get_large_value(br, p), nxt[2]
        out[n] = br.get_signed(v)
        n += 1
    return out, int(n > first)


def parse(data, raw=True):
    """A vp8.Frame from a VP8 chunk payload, and its declared part sizes.

    'raw' re-encodes the frame to find the bytes its syntax does not
    account for. Callers that only want the parsed fields can skip it.
    """
    f = vp8.Frame()
    if len(data) < vp8.FRAME_HEADER_SIZE:
        raise Truncated('%d bytes is not even a frame header' % len(data))
    bits = int.from_bytes(data[0:3], 'little')
    f.keyframe = int(not (bits & 1))
    f.version = (bits >> 1) & 7
    f.show = (bits >> 4) & 1
    part0_size = bits >> 5
    f.signature = data[3:6]
    w, h = struct.unpack('<HH', data[6:10])
    f.width, f.xscale = w & 0x3fff, w >> 14
    f.height, f.yscale = h & 0x3fff, h >> 14
    if part0_size > len(data) - vp8.FRAME_HEADER_SIZE:
        raise Truncated('partition 0 claims %d bytes, %d are left'
                        % (part0_size, len(data) - vp8.FRAME_HEADER_SIZE))
    part0 = data[vp8.FRAME_HEADER_SIZE:vp8.FRAME_HEADER_SIZE + part0_size]
    rest = data[vp8.FRAME_HEADER_SIZE + part0_size:]

    br = BoolReader(part0)
    if f.keyframe:
        f.colorspace = br.get_bit(0x80)
        f.clamp = br.get_bit(0x80)
    parse_segment_header(br, f)
    parse_filter_header(br, f)
    f.parts_log2 = br.get_value(2)
    parse_quant(br, f)
    f.refresh_proba = br.get_bit(0x80)
    parse_probas(br, f)

    parts, sizes = split_partitions(rest, f.num_parts)
    readers = [BoolReader(p) for p in parts]
    parse_mbs(br, readers, f)
    br.check()
    for r in readers:
        r.check()
    if raw:
        f.token_bytes = set_raw_tails(f, part0, parts)
    return f, sizes


def set_raw_tails(f, part0, parts):
    """Bytes the writer would not have produced by itself, kept verbatim.

    The boolean coder pads to a byte, so what a partition ends with is not
    a function of what it says; anything past that is somebody's padding or
    somebody's garbage, and either way a round trip has to carry it.
    """
    mine0, mine = vp8.assemble_parts(f)
    if part0.startswith(mine0):
        f.raw_part0 = part0[len(mine0):]
    for i, (want, got) in enumerate(zip(parts, mine)):
        if want.startswith(got) and len(want) > len(got):
            f.raw_tokens[i] = want[len(got):]
    return mine


def split_partitions(rest, num_parts):
    """ParsePartitions(): the size table, then the token partitions."""
    last = num_parts - 1
    if len(rest) < 3 * last:
        raise Truncated('no room for the %d-entry partition size table' % last)
    sizes = [int.from_bytes(rest[3 * i:3 * i + 3], 'little')
             for i in range(last)]
    out = []
    at = 3 * last
    for size in sizes:
        size = min(size, len(rest) - at)
        out.append(rest[at:at + size])
        at += size
    out.append(rest[at:])
    return out, sizes


def parse_segment_header(br, f):
    f.use_segment = br.get_bit(0x80)
    if not f.use_segment:
        return
    f.update_map = br.get_bit(0x80)
    f.update_data = br.get_bit(0x80)
    if f.update_data:
        f.absolute_delta = br.get_bit(0x80)
        f.segment_quant = [br.get_flagged_signed(7)
                           for _ in range(vp8.NUM_MB_SEGMENTS)]
        f.segment_filter = [br.get_flagged_signed(6)
                            for _ in range(vp8.NUM_MB_SEGMENTS)]
    if f.update_map:
        f.segment_probs = [br.get_flagged(8)
                           for _ in range(vp8.MB_FEATURE_TREE_PROBS)]


def parse_filter_header(br, f):
    f.simple = br.get_bit(0x80)
    f.filter_level = br.get_value(6)
    f.sharpness = br.get_value(3)
    f.use_lf_delta = br.get_bit(0x80)
    if f.use_lf_delta:
        f.update_lf_delta = br.get_bit(0x80)
        if f.update_lf_delta:
            f.ref_lf_delta = [br.get_flagged_signed(6)
                              for _ in range(vp8.NUM_REF_LF_DELTAS)]
            f.mode_lf_delta = [br.get_flagged_signed(6)
                               for _ in range(vp8.NUM_MODE_LF_DELTAS)]


def parse_quant(br, f):
    f.base_q = br.get_value(7)
    for name in vp8.QUANT_DELTAS:
        f.dq[name] = br.get_flagged_signed(4)


def parse_probas(br, f):
    for t in range(vp8.NUM_TYPES):
        for b in range(vp8.NUM_BANDS):
            for c in range(vp8.NUM_CTX):
                update = COEFFS_UPDATE_PROBA[t][b][c]
                for p in range(vp8.NUM_PROBAS):
                    if br.get_bit(update[p]):
                        f.proba_update[(t, b, c, p)] = br.get_value(8)
    f.use_skip_proba = br.get_bit(0x80)
    if f.use_skip_proba:
        f.skip_proba = br.get_value(8)


def parse_mbs(br, readers, f):
    """Modes out of partition 0, coefficients out of the token partitions."""
    probas = f.probas()
    seg_probs = f.effective_segment_probs()
    nz = vp8.NzContext(f.mb_w)
    top = [vp8.B_DC_PRED] * (4 * f.mb_w)
    for mb_y in range(f.mb_h):
        left = [vp8.B_DC_PRED] * 4
        nz.start_row()
        row = [vp8.MB() for _ in range(f.mb_w)]
        for mb_x, mb in enumerate(row):     # VP8ParseIntraModeRow()
            if f.update_map:
                mb.segment = (br.get_bit(seg_probs[2]) + 2
                              if br.get_bit(seg_probs[0])
                              else br.get_bit(seg_probs[1]))
            if f.use_skip_proba:
                mb.skip = br.get_bit(f.skip_proba)
            if br.get_bit(145):
                mb.ymode = get_i16_mode(br)
                top[4 * mb_x:4 * mb_x + 4] = [mb.ymode] * 4
                left[:] = [mb.ymode] * 4
            else:
                mb.ymode = vp8.B_PRED
                for y in range(4):
                    mode = left[y]
                    for x in range(4):
                        mode = get_bmode(br, BMODES_PROBA[top[4 * mb_x + x]]
                                         [mode])
                        mb.bmodes[4 * y + x] = top[4 * mb_x + x] = mode
                    left[y] = mode
            mb.uvmode = get_uv_mode(br)
        for mb_x, mb in enumerate(row):
            parse_mb_residuals(readers[mb_y & (f.num_parts - 1)], f, mb, nz,
                               mb_x, probas)
        f.mbs += row


def get_bmode(br, p):
    """ParseIntraMode()'s 4x4 tree."""
    if not br.get_bit(p[0]):
        return vp8.B_DC_PRED
    if not br.get_bit(p[1]):
        return vp8.B_TM_PRED
    if not br.get_bit(p[2]):
        return vp8.B_VE_PRED
    if not br.get_bit(p[3]):
        if not br.get_bit(p[4]):
            return vp8.B_HE_PRED
        return vp8.B_VR_PRED if br.get_bit(p[5]) else vp8.B_RD_PRED
    if not br.get_bit(p[6]):
        return vp8.B_LD_PRED
    if not br.get_bit(p[7]):
        return vp8.B_VL_PRED
    return vp8.B_HU_PRED if br.get_bit(p[8]) else vp8.B_HD_PRED


def get_i16_mode(br):
    """ParseIntraMode()'s hardcoded 16x16 tree."""
    if br.get_bit(156):
        return vp8.TM_PRED if br.get_bit(128) else vp8.H_PRED
    return vp8.V_PRED if br.get_bit(163) else vp8.DC_PRED


def get_uv_mode(br):
    if not br.get_bit(142):
        return vp8.DC_PRED
    if not br.get_bit(114):
        return vp8.V_PRED
    return vp8.TM_PRED if br.get_bit(183) else vp8.H_PRED


def parse_mb_residuals(br, f, mb, nz, mb_x, probas):
    """ParseResiduals(), keeping the levels instead of dequantizing them."""
    if mb.skip and f.use_skip_proba:
        nz.skip_mb(mb_x, mb.is_i4x4)
        return
    if not mb.is_i4x4:
        ctx = nz.top_y2[mb_x] + nz.left_y2
        mb.y2, v = get_coeffs(br, probas[vp8.TYPE_Y2], ctx, 0)
        nz.top_y2[mb_x] = nz.left_y2 = v
        first, ac_type = 1, vp8.TYPE_Y_AFTER_Y2
    else:
        first, ac_type = 0, vp8.TYPE_Y_WITH_DC
    for y in range(4):
        for x in range(4):
            ctx = nz.top_y[mb_x][x] + nz.left_y[y]
            mb.y[4 * y + x], v = get_coeffs(br, probas[ac_type], ctx, first)
            nz.top_y[mb_x][x] = nz.left_y[y] = v
    for ch, blocks in enumerate((mb.u, mb.v)):
        for y in range(2):
            for x in range(2):
                ctx = nz.top_uv[mb_x][ch][x] + nz.left_uv[ch][y]
                blocks[2 * y + x], v = get_coeffs(br, probas[vp8.TYPE_UV],
                                                  ctx, 0)
                nz.top_uv[mb_x][ch][x] = nz.left_uv[ch][y] = v


# -- text output --------------------------------------------------------------

# The names vp8_asm accepts, read backwards, so the two directions cannot
# drift apart.
YMODE_NAMES = {v: k + '_PRED' for k, v in vp8_asm.YMODES.items()}
BMODE_NAMES = {v: k + '_PRED' for k, v in vp8_asm.BMODES.items()}


def opt(v):
    return '-' if v is None else str(v)


def line(out, name, value, default=None):
    """One 'field value' line, unless the value is the default."""
    if value != default:
        out.append('%s %s' % (name, value))


def opts(out, name, values):
    """One line of values that may each be absent."""
    out.append('%s %s' % (name, ' '.join(opt(v) for v in values)))


def dump(f, sizes):
    """The case text for a parsed frame, in RFC 6386's field names."""
    out = ['# disassembled by vp8_dis.py']
    line(out, 'frame_type', 1 - f.keyframe, 0)
    line(out, 'version', f.version, 0)
    line(out, 'show_frame', f.show, 1)
    out.append('width %d' % f.width)
    out.append('height %d' % f.height)
    line(out, 'horizontal_scale', f.xscale, 0)
    line(out, 'vertical_scale', f.yscale, 0)
    if f.signature != vp8.SIGNATURE:
        out.append('start_code %s' % f.signature.hex())
    line(out, 'color_space', f.colorspace, 0)
    line(out, 'clamping_type', f.clamp, 0)
    if f.use_segment:
        out.append('segmentation_enabled 1')
        line(out, 'update_mb_segmentation_map', f.update_map, 0)
        line(out, 'update_segment_feature_data', f.update_data, 0)
        if f.update_data:
            line(out, 'segment_feature_mode', f.absolute_delta, 1)
            opts(out, 'quantizer_update_value', f.segment_quant)
            opts(out, 'lf_update_value', f.segment_filter)
        if f.update_map:
            opts(out, 'segment_prob', f.segment_probs)
    line(out, 'filter_type', f.simple, 0)
    line(out, 'loop_filter_level', f.filter_level, 0)
    line(out, 'sharpness_level', f.sharpness, 0)
    if f.use_lf_delta:
        out.append('loop_filter_adj_enable 1')
        line(out, 'mode_ref_lf_delta_update', f.update_lf_delta, 0)
        if f.update_lf_delta:
            opts(out, 'ref_frame_delta', f.ref_lf_delta)
            opts(out, 'mb_mode_delta', f.mode_lf_delta)
    if f.parts_log2:
        out.append('log2_nbr_of_DCT_partitions %d  # %d partitions'
                   % (f.parts_log2, f.num_parts))
    line(out, 'yac_qi', f.base_q, 0)
    for name, key in vp8_asm.QUANT_DELTAS.items():
        line(out, name, opt(f.dq[key]), '-')
    line(out, 'refresh_entropy_probs', f.refresh_proba, 0)
    for (t, b, c, i), v in sorted(f.proba_update.items()):
        out.append('coeff_prob %d %d %d %2d %3d' % (t, b, c, i, v))
    if f.use_skip_proba:
        out.append('mb_no_skip_coeff 1')
        out.append('prob_skip_false %d' % f.skip_proba)
    out += dump_mbs(f)
    for what, data in ([('part0', f.raw_part0)] +
                       [('token %d' % i, d)
                        for i, d in sorted(f.raw_tokens.items())]):
        if data:
            out.append('raw %s %s' % (what, data.hex()))
    # Only a multi-partition frame has a size table that can be wrong, and
    # parse() has already worked out what the partitions really weigh.
    tokens = f.token_bytes if sizes else ()
    for i, (declared, part) in enumerate(zip(sizes, tokens)):
        real = len(part) + len(f.raw_tokens.get(i, b''))
        if declared != real:
            out.append('patch part_size %d %d  # really %d' % (i, declared,
                                                               real))
    return '\n'.join(out) + '\n'


def dump_mbs(f):
    """One macroblock per record, runs of identical ones coalesced."""
    out = []
    for mb, count in runs(f.mbs):
        out.append('macroblock' + (' count %d' % count if count > 1 else ''))
        line(out, 'segment_id', mb.segment, 0)
        if f.use_skip_proba:
            line(out, 'mb_skip_coeff', mb.skip, 0)
        line(out, 'intra_y_mode', YMODE_NAMES[mb.ymode], 'DC_PRED')
        if mb.is_i4x4:
            names = [BMODE_NAMES[m] for m in mb.bmodes]
            out += ['intra_b_mode ' + ' '.join(names[4 * i:4 * i + 4])
                    for i in range(4)]
        line(out, 'intra_chroma_mode', YMODE_NAMES[mb.uvmode], 'DC_PRED')
        if mb.skip and f.use_skip_proba:
            continue
        for name, levels in vp8_asm.block_names(mb):
            if any(levels):
                out.append('coeffs %-6s %s' % (name, dump_levels(levels)))
    return out


def dump_levels(levels):
    """The levels, jumping over runs of zeroes rather than spelling them."""
    out = []
    at = 0
    for i, v in enumerate(levels):
        if not v:
            continue
        out.append(('%d:%d' % (i, v)) if i > at else str(v))
        at = i + 1
    return ' '.join(out)


def runs(items):
    """(macroblock, count) for runs of macroblocks that say the same thing."""
    def says(mb):
        return (mb.segment, mb.skip, mb.ymode, mb.uvmode, mb.bmodes,
                mb.y2, mb.y, mb.u, mb.v)
    return [(next(g), 1 + sum(1 for _ in g))
            for _, g in itertools.groupby(items, key=says)]


def vp8_chunk(data):
    """The VP8 chunk payload of a RIFF file, or a bare frame as it is."""
    if data[0:4] != b'RIFF':
        return data
    at, size = vp8.find_vp8_chunk(data)
    return data[at:at + size]


def check(path, data=None):
    """Dump, assemble the dump, and compare with what we started from."""
    if data is None:
        with open(path, 'rb') as fp:
            data = fp.read()
    want = vp8_chunk(data)
    f, sizes = parse(want)
    text = dump(f, sizes)
    got = vp8_chunk(vp8_asm.assemble_text(text))
    if got == want:
        print('%-44s round trip ok, %d bytes' % (path, len(want)))
        return 0
    at = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b),
              min(len(got), len(want)))
    print('%s: differs at byte %d of %d (got %d bytes)'
          % (path, at, len(want), len(got)), file=sys.stderr)
    print('  want %s' % want[max(0, at - 4):at + 8].hex(' '), file=sys.stderr)
    print('  got  %s' % got[max(0, at - 4):at + 8].hex(' '), file=sys.stderr)
    return 1


def main(argv):
    flags = [a for a in argv[1:] if a.startswith('-')]
    args = [a for a in argv[1:] if not a.startswith('-')]
    if not args:
        print('usage: %s [--check] <file.webp>...' % argv[0], file=sys.stderr)
        return 1
    if '--check' in flags:
        return max(check(a) for a in args)
    for path in args:
        with open(path, 'rb') as fp:
            f, sizes = parse(vp8_chunk(fp.read()))
        sys.stdout.write(dump(f, sizes))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
