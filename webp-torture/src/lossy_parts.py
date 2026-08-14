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
import sys

import vp8

NUM_PARTS = 8  # what the patched cases below are built from


def size_table_offset(data):
    """Offset of the partition size table inside a whole RIFF file."""
    off = vp8.find_vp8_chunk(data)[0]
    return off + vp8.size_table_offset(data[off:])


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
        rows.append((name, expect, note, exercises, len(data), '', '', '', ''))
        print('%-40s %-7s %5d bytes' % (name, expect, len(data)))
    return rows


if __name__ == '__main__':
    build(*sys.argv[1:3])
