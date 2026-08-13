#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.
"""Builds the multi-partition (lossy VP8) cases.

The partition *count* lives inside the bool-coded partition 0, so it cannot be
patched byte-wise; these start from real encodes of 1, 2, 4 and 8 partitions
and rewrite the raw 3-byte-per-entry size table that follows partition 0.
cwebp cannot emit any of this: it does not expose config.partitions at all,
and libwebp forces num_parts back to 1 whenever the token path is used
(webp_enc.c:124).
"""

import os
import struct
import sys

NUM_PARTS = 8  # what the patched cases below are built from


def find_vp8_chunk(data):
    """Returns the offset of the VP8 chunk payload inside a RIFF file."""
    assert data[0:4] == b'RIFF' and data[8:12] == b'WEBP', 'not a RIFF/WEBP'
    pos = 12
    while pos + 8 <= len(data):
        tag = data[pos:pos + 4]
        size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        if tag == b'VP8 ':
            return pos + 8
        pos += 8 + size + (size & 1)
    raise ValueError('no VP8 chunk')


def size_table_offset(data):
    """Offset of the partition size table: past the 10-byte keyframe header
    and past partition 0, whose length is in the frame tag."""
    off = find_vp8_chunk(data)
    bits = data[off] | (data[off + 1] << 8) | (data[off + 2] << 16)
    return off + 10 + (bits >> 5)          # bits >> 5 is partition 0's size


def read_sizes(data, num_parts=NUM_PARTS):
    """The (num_parts - 1) declared partition sizes."""
    off = size_table_offset(data)
    return [int.from_bytes(data[off + 3 * i:off + 3 * i + 3], 'little')
            for i in range(num_parts - 1)]


def patch_sizes(data, sizes):
    """Rewrites the 3-byte little-endian partition sizes."""
    off = size_table_offset(data)
    out = bytearray(data)
    for i, s in enumerate(sizes):
        out[off + 3 * i:off + 3 * i + 3] = s.to_bytes(3, 'little')
    return bytes(out)


def cases(src):
    """The seven cases, as (name, expect, note, exercises, data).

    'src' maps a partition count to a real encode with that many partitions.
    """
    eight = src[NUM_PARTS]
    sizes = read_sizes(eight)
    return [
        ('lossy-1-partitions', 'ok',
         'A single token partition: the default, and the control for the '
         'others.',
         'Same encode settings as the 2/4/8 files, so a size or hash '
         'difference against them is entirely the partitioning.',
         src[1]),
        ('lossy-2-partitions', 'ok',
         'A plain 2-partition lossy frame.',
         'cwebp never emits this: config.partitions is API-only and is '
         'forced back to 1 for method >= 3 unless low_memory is set.',
         src[2]),
        ('lossy-4-partitions', 'ok',
         'A plain 4-partition lossy frame.',
         'Same, with the size table holding three entries.',
         src[4]),
        ('lossy-8-partitions', 'ok',
         'A plain 8-partition lossy frame, the maximum the 2-bit field '
         'allows.',
         'MAX_NUM_PARTITIONS. Seven 3-byte size-table entries, and eight '
         'independent bit-readers in the decoder.',
         eight),
        ('lossy-8-partitions-size-overflow', 'reject',
         'Eight partitions whose first declared size is 0xffffff, far past '
         'the data.',
         'Hits the "if (psize > size_left) psize = size_left" clamp in '
         'ParsePartitions(): partition 0 swallows the whole remainder and '
         'the other seven get zero-length readers.',
         patch_sizes(eight, [0xffffff] + sizes[1:])),
        ('lossy-8-partitions-zero-sizes', 'reject',
         'Eight partitions all declared as zero bytes long.',
         'Every token partition but the last is empty, so the last one is '
         'handed the whole remainder. Legal to parse, garbage to decode.',
         patch_sizes(eight, [0] * (NUM_PARTS - 1))),
        ('lossy-8-partitions-sizes-sum-past-end', 'reject',
         'Eight partitions whose declared sizes add up to more than the '
         'chunk holds.',
         'The clamp fires part-way through the loop, so later partitions '
         'get zero-length readers while earlier ones look valid.',
         patch_sizes(eight, [s * 4 + 64 for s in sizes])),
    ]


def build(outdir='.', srcdir=None):
    """Writes the cases into <outdir>/files and returns their README rows."""
    # The 1/2/4/8-partition sources live in sources/, not files/, so that
    # files/ stays pure output and can be wiped before a rebuild. Regenerate
    # them with make_partition_sources.c, which needs the encoder API.
    srcdir = srcdir or os.path.join(outdir, 'sources')
    src = {}
    for n in (1, 2, 4, NUM_PARTS):
        name = os.path.join(srcdir, 'lossy-%d-partitions.webp' % n)
        with open(name, 'rb') as f:
            src[n] = f.read()
    files = os.path.join(outdir, 'files')
    os.makedirs(files, exist_ok=True)
    rows = []
    for name, expect, note, exercises, data in cases(src):
        with open(os.path.join(files, name + '.webp'), 'wb') as f:
            f.write(data)
        rows.append((name, expect, note, exercises, len(data)))
        print('%-40s %-7s %5d bytes' % (name, expect, len(data)))
    return rows


if __name__ == '__main__':
    build(*sys.argv[1:3])
