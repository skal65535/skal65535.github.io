# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.

"""Minimal VP8 (WebP lossy) bitstream *writer*, for building torture cases.

Deliberately does no validation: the point is to emit streams that a normal
encoder cannot produce. The boolean coder is a port of VP8BitWriter
(src/utils/bit_writer_utils.c) and the syntax below follows the decoder,
src/dec/vp8_dec.c and src/dec/tree_dec.c, section by section.

Only key frames are described. Everything the decoder reads has a field
here, and nothing clamps it: writing a 5-bit value into a 4-bit field just
drops the top bit, which is often exactly what a case wants.
"""

import struct

from vp8_tables import (BANDS, BMODES_PROBA, CAT3456, COEFFS_PROBA0,
                        COEFFS_UPDATE_PROBA, NEW_RANGE, NORM)

# intra prediction modes, src/dec/common_dec.h. The 4x4 numbering is the one
# that matters: the 16x16 modes are a subset of it, which is why a 16x16 mode
# can be stored as the 4x4 context of its neighbours.
(B_DC_PRED, B_TM_PRED, B_VE_PRED, B_HE_PRED, B_RD_PRED, B_VR_PRED, B_LD_PRED,
 B_VL_PRED, B_HD_PRED, B_HU_PRED) = range(10)
DC_PRED, V_PRED, H_PRED, TM_PRED = B_DC_PRED, B_VE_PRED, B_HE_PRED, B_TM_PRED
B_PRED = 10                     # not a mode number: 'use the 16 4x4 modes'

NUM_MB_SEGMENTS = 4
MB_FEATURE_TREE_PROBS = 3
NUM_REF_LF_DELTAS = 4
NUM_MODE_LF_DELTAS = 4
NUM_TYPES, NUM_BANDS, NUM_CTX, NUM_PROBAS = 4, 8, 3, 11

# coefficient types, i.e. the first index of the probability table
TYPE_Y_AFTER_Y2, TYPE_Y2, TYPE_UV, TYPE_Y_WITH_DC = 0, 1, 2, 3

SIGNATURE = b'\x9d\x01\x2a'
FRAME_HEADER_SIZE = 10

# The five quantizer deltas, in the order the frame header carries them.
QUANT_DELTAS = ('y1_dc', 'y2_dc', 'y2_ac', 'uv_dc', 'uv_ac')


class BoolWriter:
    """The VP8 boolean coder. A port of VP8BitWriter."""

    def __init__(self):
        self.range = 255 - 1
        self.value = 0
        self.run = 0
        self.nb_bits = -8
        self.buf = bytearray()

    def _flush(self):
        s = 8 + self.nb_bits
        bits = self.value >> s
        self.value -= bits << s
        self.nb_bits -= 8
        if (bits & 0xff) != 0xff:
            if bits & 0x100 and self.buf:    # carry over the pending 0xff's
                self.buf[-1] = (self.buf[-1] + 1) & 0xff
            if self.run > 0:
                self.buf.extend([0x00 if (bits & 0x100) else 0xff] * self.run)
                self.run = 0
            self.buf.append(bits & 0xff)
        else:
            self.run += 1                    # delay, a carry may still come

    def _renorm(self, shift):
        self.range = NEW_RANGE[self.range]
        self.value <<= shift
        self.nb_bits += shift
        if self.nb_bits > 0:
            self._flush()

    def put_bit(self, bit, prob):
        bit = 1 if bit else 0
        split = (self.range * prob) >> 8
        if bit:
            self.value += split + 1
            self.range -= split + 1
        else:
            self.range = split
        if self.range < 127:
            self._renorm(NORM[self.range])
        return bit

    def put_uniform(self, bit):
        """VP8PutBitUniform(): a bit with probability 1/2."""
        bit = 1 if bit else 0
        split = self.range >> 1
        if bit:
            self.value += split + 1
            self.range -= split + 1
        else:
            self.range = split
        if self.range < 127:
            self._renorm(1)
        return bit

    def put_bits(self, value, n):
        """n raw bits, most significant first. Extra bits are dropped."""
        for i in reversed(range(n)):
            self.put_uniform((value >> i) & 1)

    def put_flagged(self, value, n):
        """A flag, then n bits: the 'optional value' shape of the header.

        'value' is None for a bare 0 flag. Everything else is written even
        if the field cannot hold it.
        """
        if self.put_uniform(value is not None):
            self.put_bits(value, n)

    def put_flagged_signed(self, value, n):
        """VP8PutSignedBits(): a flag, then n magnitude bits, then a sign.

        Unlike the encoder's version, a 0 still writes a flag: 'present and
        zero' and 'absent' are different bitstreams, and only None is absent.
        """
        if self.put_uniform(value is not None):
            self.put_bits((abs(value) << 1) | (value < 0), n + 1)

    def finish(self):
        """VP8BitWriterFinish(): pad, flush, and hand back the bytes."""
        self.put_bits(0, 9 - self.nb_bits)
        self.nb_bits = 0
        self._flush()
        return bytes(self.buf)


def put_coeffs(bw, ctx, levels, first, probas):
    """One block of coefficients. A port of PutCoeffs(), src/enc/frame_enc.c.

    'levels' are the 16 quantized levels in coding (zigzag) order, 'probas'
    the [band][ctx][11] table of one coefficient type. Returns the block's
    non-zero flag, which is the neighbours' context.
    """
    last = -1
    for n in reversed(range(16)):
        if levels[n]:
            last = n
            break
    n = first
    # should be probas[BANDS[n]], but it is the same for n = 0 or 1
    p = probas[n][ctx]
    if not bw.put_bit(last >= 0, p[0]):
        return 0
    while n < 16:
        c = levels[n]
        n += 1
        sign = c < 0
        v = -c if sign else c
        if not bw.put_bit(v != 0, p[1]):
            p = probas[BANDS[n]][0]
            continue
        if not bw.put_bit(v > 1, p[2]):
            p = probas[BANDS[n]][1]
        else:
            put_large_value(bw, v, p)
            p = probas[BANDS[n]][2]
        bw.put_uniform(sign)
        if n == 16 or not bw.put_bit(n <= last, p[0]):
            return 1
    return 1


def put_large_value(bw, v, p):
    """A coefficient magnitude of 2 or more (section 13.2)."""
    if not bw.put_bit(v > 4, p[3]):
        if bw.put_bit(v != 2, p[4]):
            bw.put_bit(v == 4, p[5])
    elif not bw.put_bit(v > 10, p[6]):
        if not bw.put_bit(v > 6, p[7]):
            bw.put_bit(v == 6, 159)
        else:
            bw.put_bit(v >= 9, 165)
            bw.put_bit(not (v & 1), 145)
    else:                            # categories 3 to 6, 3 to 11 extra bits
        cat = 0
        while cat < 3 and v >= 3 + (8 << (cat + 1)):
            cat += 1
        bw.put_bit(cat >> 1, p[8])
        bw.put_bit(cat & 1, p[9] if cat < 2 else p[10])
        v -= 3 + (8 << cat)
        tab = CAT3456[cat]
        for i, prob in enumerate(tab):
            bw.put_bit((v >> (len(tab) - 1 - i)) & 1, prob)


def put_segment(bw, s, probs):
    """The segment id, a 2-level tree. PutSegment(), src/enc/tree_enc.c."""
    if bw.put_bit(s >= 2, probs[0]):
        bw.put_bit(s & 1, probs[2])
    else:
        bw.put_bit(s & 1, probs[1])


def put_i16_mode(bw, mode):
    if bw.put_bit(mode in (TM_PRED, H_PRED), 156):
        bw.put_bit(mode == TM_PRED, 128)
    else:
        bw.put_bit(mode == V_PRED, 163)


def put_i4_mode(bw, mode, probs):
    """One 4x4 mode, under the [above][left] probabilities."""
    if bw.put_bit(mode != B_DC_PRED, probs[0]):
        if bw.put_bit(mode != B_TM_PRED, probs[1]):
            if bw.put_bit(mode != B_VE_PRED, probs[2]):
                if not bw.put_bit(mode >= B_LD_PRED, probs[3]):
                    if bw.put_bit(mode != B_HE_PRED, probs[4]):
                        bw.put_bit(mode != B_RD_PRED, probs[5])
                elif bw.put_bit(mode != B_LD_PRED, probs[6]):
                    if bw.put_bit(mode != B_VL_PRED, probs[7]):
                        bw.put_bit(mode != B_HD_PRED, probs[8])


def put_uv_mode(bw, mode):
    if bw.put_bit(mode != DC_PRED, 142):
        if bw.put_bit(mode != V_PRED, 114):
            bw.put_bit(mode != H_PRED, 183)


class MB:
    """One macroblock: its modes, and 25 blocks of 16 coefficient levels."""

    def __init__(self):
        self.segment = 0
        self.skip = 0
        self.ymode = DC_PRED           # or B_PRED, and then bmodes[] is used
        self.bmodes = [B_DC_PRED] * 16
        self.uvmode = DC_PRED
        self.y2 = [0] * 16             # only written when ymode != B_PRED
        self.y = [[0] * 16 for _ in range(16)]
        self.u = [[0] * 16 for _ in range(4)]
        self.v = [[0] * 16 for _ in range(4)]

    @property
    def is_i4x4(self):
        return self.ymode == B_PRED


class Frame:
    """Every field of a key frame, with the defaults of a plain flat image."""

    def __init__(self, width=16, height=16):
        # frame tag and picture header (the 10 uncompressed bytes)
        self.width, self.height = width, height
        self.xscale, self.yscale = 0, 0
        self.version = 0               # 'profile' in the decoder, 3 bits
        self.show = 1
        self.keyframe = 1
        self.signature = SIGNATURE
        self.colorspace, self.clamp = 0, 0
        # segment header
        self.use_segment = 0
        self.update_map = 0
        self.update_data = 0
        self.absolute_delta = 1
        self.segment_quant = [None] * NUM_MB_SEGMENTS
        self.segment_filter = [None] * NUM_MB_SEGMENTS
        self.segment_probs = [None] * MB_FEATURE_TREE_PROBS
        # filter header
        self.simple = 0
        self.filter_level = 0
        self.sharpness = 0
        self.use_lf_delta = 0
        self.update_lf_delta = 0
        self.ref_lf_delta = [None] * NUM_REF_LF_DELTAS
        self.mode_lf_delta = [None] * NUM_MODE_LF_DELTAS
        # partitions, quantizer
        self.parts_log2 = 0
        self.base_q = 0
        self.dq = {'y1_dc': None, 'y2_dc': None, 'y2_ac': None,
                   'uv_dc': None, 'uv_ac': None}
        # probabilities
        self.refresh_proba = 0         # read and ignored by libwebp
        self.proba_update = {}         # (type, band, ctx, index) -> value
        self.use_skip_proba = 0
        self.skip_proba = 0
        # macroblocks
        self.mbs = []
        self.num_mbs = None            # None: as many as the size implies
        # raw appendices, for cases the syntax cannot express
        self.raw_part0 = b''
        self.raw_tokens = {}           # partition index -> bytes
        self.token_bytes = ()          # what vp8_dis re-encoded them to

    @property
    def mb_w(self):
        return (self.width + 15) >> 4

    @property
    def mb_h(self):
        return (self.height + 15) >> 4

    @property
    def num_parts(self):
        return 1 << self.parts_log2

    def mb_count(self):
        return self.mb_w * self.mb_h if self.num_mbs is None else self.num_mbs

    def probas(self):
        """The coefficient probabilities the decoder will end up with.

        Read-only: the defaults are handed back as they are when nothing
        overrides them, rather than copied 1056 entries at a time.
        """
        if not self.proba_update:
            return COEFFS_PROBA0
        out = [[[list(ctx) for ctx in band] for band in t]
               for t in COEFFS_PROBA0]
        for (t, b, c, p), v in self.proba_update.items():
            out[t][b][c][p] = v
        return out

    def effective_segment_probs(self):
        """255 is the reset value a missing update leaves behind."""
        return [255 if p is None else p for p in self.segment_probs]


def write_header(bw, f):
    """Partition 0 up to, but not including, the macroblock modes."""
    if f.keyframe:                     # the decoder skips these otherwise
        bw.put_uniform(f.colorspace)
        bw.put_uniform(f.clamp)
    write_segment_header(bw, f)
    write_filter_header(bw, f)
    bw.put_bits(f.parts_log2, 2)
    write_quant(bw, f)
    bw.put_uniform(f.refresh_proba)
    write_probas(bw, f)


def write_segment_header(bw, f):
    if not bw.put_uniform(f.use_segment):
        return
    bw.put_uniform(f.update_map)
    if bw.put_uniform(f.update_data):
        bw.put_uniform(f.absolute_delta)
        for q in f.segment_quant:
            bw.put_flagged_signed(q, 7)
        for s in f.segment_filter:
            bw.put_flagged_signed(s, 6)
    if f.update_map:
        for p in f.segment_probs:
            bw.put_flagged(p, 8)


def write_filter_header(bw, f):
    bw.put_uniform(f.simple)
    bw.put_bits(f.filter_level, 6)
    bw.put_bits(f.sharpness, 3)
    if bw.put_uniform(f.use_lf_delta):
        if bw.put_uniform(f.update_lf_delta):
            for d in f.ref_lf_delta:
                bw.put_flagged_signed(d, 6)
            for d in f.mode_lf_delta:
                bw.put_flagged_signed(d, 6)


def write_quant(bw, f):
    bw.put_bits(f.base_q, 7)
    for name in QUANT_DELTAS:
        bw.put_flagged_signed(f.dq[name], 4)


def write_probas(bw, f):
    """1056 update flags, then the skip probability."""
    updates = f.proba_update
    for t in range(NUM_TYPES):
        for b in range(NUM_BANDS):
            for c in range(NUM_CTX):
                update = COEFFS_UPDATE_PROBA[t][b][c]
                for p in range(NUM_PROBAS):
                    v = updates.get((t, b, c, p)) if updates else None
                    if bw.put_bit(v is not None, update[p]):
                        bw.put_bits(v, 8)
    if bw.put_uniform(f.use_skip_proba):
        bw.put_bits(f.skip_proba, 8)


def write_modes(bw, f):
    """Every macroblock's mode info, in raster order, into partition 0."""
    seg_probs = f.effective_segment_probs()
    top = [B_DC_PRED] * (4 * f.mb_w)   # dec->intra_t, kept across rows
    left = [B_DC_PRED] * 4             # dec->intra_l, reset per row
    for i in range(f.mb_count()):
        mb_x = i % f.mb_w
        if mb_x == 0:
            left = [B_DC_PRED] * 4
        mb = f.mbs[i]
        if f.update_map:
            put_segment(bw, mb.segment, seg_probs)
        if f.use_skip_proba:
            bw.put_bit(mb.skip, f.skip_proba)
        if bw.put_bit(not mb.is_i4x4, 145):
            put_i16_mode(bw, mb.ymode)
            # a 16x16 mode doubles as the 4x4 context of its neighbours
            top[4 * mb_x:4 * mb_x + 4] = [mb.ymode] * 4
            left[:] = [mb.ymode] * 4
        else:
            for y in range(4):
                mode = left[y]         # the block to the left of column 0
                for x in range(4):
                    probs = BMODES_PROBA[top[4 * mb_x + x]][mode]
                    mode = mb.bmodes[4 * y + x]
                    put_i4_mode(bw, mode, probs)
                    top[4 * mb_x + x] = mode
                left[y] = mode
        put_uv_mode(bw, mb.uvmode)


class NzContext:
    """The 'has non-zero coefficients' flags the coefficient contexts use.

    Top values live for the whole frame, left values are reset at the start
    of every macroblock row -- VP8InitScanline(), src/dec/vp8_dec.c.
    """

    def __init__(self, mb_w):
        self.top_y = [[0] * 4 for _ in range(mb_w)]
        self.top_uv = [[[0] * 2, [0] * 2] for _ in range(mb_w)]
        self.top_y2 = [0] * mb_w
        self.start_row()

    def start_row(self):
        self.left_y = [0] * 4
        self.left_uv = [[0] * 2, [0] * 2]
        self.left_y2 = 0

    def skip_mb(self, mb_x, is_i4x4):
        """VP8DecodeMB(): a skipped macroblock clears its neighbours' flags,
        but leaves the Y2 flag alone when it has no Y2 block to clear."""
        self.top_y[mb_x] = [0] * 4
        self.left_y = [0] * 4
        self.top_uv[mb_x] = [[0] * 2, [0] * 2]
        self.left_uv = [[0] * 2, [0] * 2]
        if not is_i4x4:
            self.top_y2[mb_x] = self.left_y2 = 0


def write_mb_residuals(bw, f, mb, nz, mb_x, probas):
    """One macroblock's coefficients. A port of CodeResiduals()."""
    if mb.skip and f.use_skip_proba:
        nz.skip_mb(mb_x, mb.is_i4x4)
        return
    if not mb.is_i4x4:
        ctx = nz.top_y2[mb_x] + nz.left_y2
        v = put_coeffs(bw, ctx, mb.y2, 0, probas[TYPE_Y2])
        nz.top_y2[mb_x] = nz.left_y2 = v
        first, ac_type = 1, TYPE_Y_AFTER_Y2
    else:
        first, ac_type = 0, TYPE_Y_WITH_DC
    for y in range(4):
        for x in range(4):
            ctx = nz.top_y[mb_x][x] + nz.left_y[y]
            v = put_coeffs(bw, ctx, mb.y[4 * y + x], first, probas[ac_type])
            nz.top_y[mb_x][x] = nz.left_y[y] = v
    for ch, blocks in enumerate((mb.u, mb.v)):
        for y in range(2):
            for x in range(2):
                ctx = nz.top_uv[mb_x][ch][x] + nz.left_uv[ch][y]
                v = put_coeffs(bw, ctx, blocks[2 * y + x], 0, probas[TYPE_UV])
                nz.top_uv[mb_x][ch][x] = nz.left_uv[ch][y] = v


def write_residuals(f, parts):
    """Every macroblock's coefficients, row r going to partition r % n."""
    nz = NzContext(f.mb_w)
    probas = f.probas()
    for i in range(f.mb_count()):
        mb_x, mb_y = i % f.mb_w, i // f.mb_w
        if mb_x == 0:
            nz.start_row()
        bw = parts[mb_y & (f.num_parts - 1)]
        write_mb_residuals(bw, f, f.mbs[i], nz, mb_x, probas)


def assemble_parts(f):
    """The bool-coded partitions on their own: partition 0, then the tokens."""
    assert len(f.mbs) == f.mb_count(), \
        'have %d macroblocks, need %d' % (len(f.mbs), f.mb_count())
    bw = BoolWriter()
    write_header(bw, f)
    write_modes(bw, f)
    parts = [BoolWriter() for _ in range(f.num_parts)]
    write_residuals(f, parts)
    return bw.finish(), [p.finish() for p in parts]


def assemble(f):
    """The whole VP8 chunk payload: header, partition 0, token partitions."""
    part0, tokens = assemble_parts(f)
    part0 += f.raw_part0
    tokens = [t + f.raw_tokens.get(i, b'') for i, t in enumerate(tokens)]

    bits = ((0 if f.keyframe else 1) | (f.version << 1) | (f.show << 4) |
            (len(part0) << 5))
    header = (bits & 0xffffff).to_bytes(3, 'little') + f.signature
    header += struct.pack('<HH', (f.width & 0x3fff) | (f.xscale << 14),
                          (f.height & 0x3fff) | (f.yscale << 14))
    sizes = b''.join((len(t) & 0xffffff).to_bytes(3, 'little')
                     for t in tokens[:-1])
    return header + part0 + sizes + b''.join(tokens)


def partition0_size(data):
    """The 19-bit length the frame tag claims for partition 0."""
    return int.from_bytes(data[0:3], 'little') >> 5


def set_partition0_size(data, size):
    """Rewrites that field in place, dropping whatever will not fit."""
    bits = int.from_bytes(data[0:3], 'little')
    data[0:3] = (((bits & 0x1f) | (size << 5)) & 0xffffff).to_bytes(3,
                                                                    'little')


def size_table_offset(data):
    """Where the 3-byte token partition sizes sit, per the frame tag."""
    return FRAME_HEADER_SIZE + partition0_size(data)


def find_vp8_chunk(data):
    """(offset, size) of the VP8 chunk payload in a RIFF file."""
    assert data[0:4] == b'RIFF' and data[8:12] == b'WEBP', 'not a RIFF/WEBP'
    pos = 12
    while pos + 8 <= len(data):
        size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        if data[pos:pos + 4] == b'VP8 ':
            return pos + 8, size
        pos += 8 + size + (size & 1)
    raise ValueError('no VP8 chunk')


def wrap_webp(payload):
    """RIFF/WEBP container around a raw VP8 frame."""
    chunk = b'VP8 ' + struct.pack('<I', len(payload)) + payload
    if len(payload) & 1:
        chunk += b'\0'
    return b'RIFF' + struct.pack('<I', 4 + len(chunk)) + b'WEBP' + chunk
