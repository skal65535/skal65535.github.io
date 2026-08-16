#!/usr/bin/env python3
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.
"""Patches a libwebp source tree with fprintf probes, for coverage.txt.

Usage: python3 probes.py <path-to-libwebp-worktree>

Every probe sits on the exact line a README note refers to, so running the
corpus through the patched decoder shows which files really reach which path.
Use a throwaway worktree; this rewrites sources in place.

Some lines here run past 80 columns: they are verbatim libwebp source used as
match anchors, and wrapping them would stop them matching.
"""

import sys

MACROS = ('#include <stdio.h>\n'
          '#define PROBE(t) fprintf(stderr, "PROBE:%s\\n", t)\n'
          '#define PROBE1(f, v) fprintf(stderr, "PROBE:" f "\\n", v)\n'
          # For a probe on a per-pixel path: the output is collected into one
          # shell variable, and the file that decodes to a gigabyte would put
          # 268 million lines in it.
          '#define PROBE_ONCE(t) do { static int probed_;'
          ' if (!probed_) { probed_ = 1; PROBE(t); } } while (0)\n')

VP8L = [
    # simple codes
    # Anchored on the symbol read rather than on the comment that sits above
    # code_lengths[symbol] on one branch, so this applies to upstream main
    # too.
    ("""    int symbol = VP8LReadBits(br, (first_symbol_len_code == 0) ? 1 : 8);""",
     """    int symbol = VP8LReadBits(br, (first_symbol_len_code == 0) ? 1 : 8);
    if (symbol >= alphabet_size) PROBE("simple_sym1_oob");
    if (first_symbol_len_code == 0) PROBE("simple_sym_1bit");
    if (num_symbols == 1) PROBE("simple_1sym"); else PROBE("simple_2sym");"""),
    ("""      symbol = VP8LReadBits(br, 8);
      code_lengths[symbol] = 1;""",
     """      symbol = VP8LReadBits(br, 8);
      if (symbol >= alphabet_size) PROBE("simple_sym2_oob");
      code_lengths[symbol] = 1;"""),
    # complex form
    ("""    const int num_codes = VP8LReadBits(br, 4) + 4;""",
     """    const int num_codes = VP8LReadBits(br, 4) + 4;
    if (num_codes == 4) PROBE("num_codes_min");
    if (num_codes == 19) PROBE("num_codes_max");
    PROBE("complex_code");"""),
    ("""      const int use_prev = (code_len == kCodeLengthRepeatCode);""",
     """      const int use_prev = (code_len == kCodeLengthRepeatCode);
      if (code_len == 16) PROBE("codelen_16");
      if (code_len == 16 && prev_code_len == DEFAULT_CODE_LENGTH)
        PROBE("codelen_16_default_prev");
      if (code_len == 17) PROBE("codelen_17");
      if (code_len == 18) PROBE("codelen_18");"""),
    ("""      int repeat = VP8LReadBits(br, extra_bits) + repeat_offset;""",
     """      int repeat = VP8LReadBits(br, extra_bits) + repeat_offset;
      if (code_len == 18 && repeat == 138) PROBE("codelen_18_max_run");
      if (code_len == 17 && repeat == 10) PROBE("codelen_17_max_run");"""),
    ("""      if (symbol + repeat > num_symbols) {
        goto End;""",
     """      if (symbol + repeat > num_symbols) {
        PROBE("reject_repeat_past_end");
        goto End;"""),
    ("""    max_symbol = 2 + VP8LReadBits(br, length_nbits);
    if (max_symbol > num_symbols) {
      goto End;
    }""",
     """    max_symbol = 2 + VP8LReadBits(br, length_nbits);
    PROBE("use_length");
    if (max_symbol > num_symbols) {
      PROBE("reject_max_symbol_too_big");
      goto End;
    }"""),
    ("""    if (max_symbol-- == 0) break;""",
     """    if (max_symbol-- == 0) { PROBE("max_symbol_early_stop"); break; }"""),
    # header
    ("""  if (VP8LReadBits(br, VP8L_VERSION_BITS) != 0) return 0;""",
     """  if (*width == 16384) PROBE("header_width_16384");
  if ((uint64_t)*width * *height >= (1ull << 28)) PROBE("header_huge_area");
  if (VP8LReadBits(br, VP8L_VERSION_BITS) != 0) {
    PROBE("reject_version");
    return 0;
  }"""),
    ("""  return (size >= VP8L_FRAME_HEADER_SIZE && data[0] == VP8L_MAGIC_BYTE &&
          (data[4] >> 5) == 0);  // version""",
     """  if (size >= VP8L_FRAME_HEADER_SIZE && data[0] != VP8L_MAGIC_BYTE) {
    PROBE("reject_signature_magic");
  }
  if (size >= VP8L_FRAME_HEADER_SIZE && data[0] == VP8L_MAGIC_BYTE &&
      (data[4] >> 5) != 0) {
    PROBE("reject_signature_version");
  }
  return (size >= VP8L_FRAME_HEADER_SIZE && data[0] == VP8L_MAGIC_BYTE &&
          (data[4] >> 5) == 0);  // version"""),
    # the entropy image, consulted again part-way along a row
    ("""    if ((col & mask) == 0) {
      htree_group = GetHtreeGroupForPos(hdr, col, row);
    }""",
     """    if ((col & mask) == 0) {
      const HTreeGroup* const previous = htree_group;
      htree_group = GetHtreeGroupForPos(hdr, col, row);
      if (htree_group != previous) PROBE("meta_group_switch");
    }"""),
    # the signature byte, tested a second time as the stream is read
    ("""  if (VP8LReadBits(br, 8) != VP8L_MAGIC_BYTE) return 0;""",
     """  {
    const int magic = VP8LReadBits(br, 8);
    if (magic != VP8L_MAGIC_BYTE) { PROBE("reject_magic"); return 0; }
  }"""),
    ("""  *has_alpha = VP8LReadBits(br, 1);""",
     """  *has_alpha = VP8LReadBits(br, 1);
  if (*has_alpha) PROBE("alpha_is_used");"""),
    # sub-images: the same reader, one level down
    ("""  int color_cache_bits = 0;""",
     """  int color_cache_bits = 0;
  if (!is_level0) PROBE("subimage_stream");"""),
    # color cache
    ("""    color_cache_bits = VP8LReadBits(br, 4);
    ok = (color_cache_bits >= 1 && color_cache_bits <= MAX_CACHE_BITS);""",
     """    color_cache_bits = VP8LReadBits(br, 4);
    if (!is_level0) PROBE("subimage_cache");
    if (color_cache_bits == 1) PROBE("cache_bits_min");
    if (color_cache_bits == MAX_CACHE_BITS) PROBE("cache_bits_max");
    ok = (color_cache_bits >= 1 && color_cache_bits <= MAX_CACHE_BITS);
    if (!ok) PROBE("reject_cache_bits");"""),
    # transforms
    ("""  if (dec->transforms_seen & (1U << type)) {
    return 0;""",
     """  { static const char* kT[4] = {"tf_predictor", "tf_cross_color",
      "tf_subtract_green", "tf_color_indexing"}; PROBE(kT[type]); }
  if (dec->transforms_seen & (1U << type)) {
    PROBE("reject_transform_repeated");
    return 0;"""),
    ("""      transform->bits =
          MIN_TRANSFORM_BITS + VP8LReadBits(br, NUM_TRANSFORM_BITS);""",
     """      transform->bits =
          MIN_TRANSFORM_BITS + VP8LReadBits(br, NUM_TRANSFORM_BITS);
      if (transform->bits == MIN_TRANSFORM_BITS) PROBE("tf_bits_min");
      if (transform->bits == MIN_TRANSFORM_BITS + 7) PROBE("tf_bits_max");"""),
    ("""      const int num_colors = VP8LReadBits(br, 8) + 1;""",
     """      const int num_colors = VP8LReadBits(br, 8) + 1;
      if (num_colors == 1) PROBE("palette_1");
      if (num_colors == 2) PROBE("palette_2");
      if (num_colors == 3) PROBE("palette_3");
      if (num_colors == 16) PROBE("palette_16");
      if (num_colors == 256) PROBE("palette_256");"""),
    ("""    for (; i < 4 * final_num_colors; ++i) {
      new_data[i] = 0;  // black tail.""",
     """    if (i < 4 * final_num_colors) PROBE("palette_black_tail");
    for (; i < 4 * final_num_colors; ++i) {
      new_data[i] = 0;  // black tail."""),
    # meta huffman
    ("""    hdr->huffman_subsample_bits = huffman_precision;""",
     """    PROBE("meta_huffman");
    if (huffman_precision == MIN_HUFFMAN_BITS) PROBE("meta_precision_min");
    if (huffman_precision == MIN_HUFFMAN_BITS + 7) PROBE("meta_precision_max");
    hdr->huffman_subsample_bits = huffman_precision;"""),
    ("""      mapping = (int*)WebPSafeMalloc(num_htree_groups_max, sizeof(*mapping));""",
     """      PROBE("meta_mapping_path");
      if (num_htree_groups_max > 1000) PROBE("meta_groups_over_1000");
      mapping = (int*)WebPSafeMalloc(num_htree_groups_max, sizeof(*mapping));"""),
    ("""    if (mapping != NULL && mapping[i] == -1) {""",
     """    if (mapping != NULL && mapping[i] == -1) {
      PROBE("meta_unused_group");"""),
    # LZ77 and distances
    # Anchored on the line below it as well: the same "if (code <
    # NUM_LITERAL_CODES)" opens DecodeAlphaData a hundred lines earlier, and
    # matching that one put these three probes on a path no file here takes.
    ("""    if (code < NUM_LITERAL_CODES) {  // Literal
      if (htree_group->is_trivial_literal) {""",
     """    if (code >= NUM_LITERAL_CODES && code < NUM_LITERAL_CODES + NUM_LENGTH_CODES)
      PROBE("lz77_copy");
    if (code >= NUM_LITERAL_CODES + NUM_LENGTH_CODES) PROBE("cache_index");
    if (code == NUM_LITERAL_CODES + NUM_LENGTH_CODES - 1) PROBE("lz77_max_len_sym");
    if (code < NUM_LITERAL_CODES) {  // Literal
      if (htree_group->is_trivial_literal) {"""),
    # The 8-bit path, which only an ALPH chunk with a compressed plane
    # reaches: every VP8L file in the corpus goes through DecodeImageData.
    # Which of the three literal shapes a pixel takes: the whole ARGB from
    # the group with no bits read at all, the green symbol over a fixed
    # arb, or all four channels read one after another.
    ("""    if (htree_group->is_trivial_code) {
      *src = htree_group->literal_arb;""",
     """    if (htree_group->is_trivial_code) {
      PROBE_ONCE("literal_trivial_code");
      *src = htree_group->literal_arb;"""),
    ("""      if (htree_group->is_trivial_literal) {
        if (VP8LIsEndOfStream(br)) break;""",
     """      if (htree_group->is_trivial_literal) {
        PROBE_ONCE("literal_trivial_arb");
        if (VP8LIsEndOfStream(br)) break;"""),
    ("""        int red, blue, alpha;
        red = ReadSymbol(htree_group->htrees[RED], br);""",
     """        int red, blue, alpha;
        PROBE_ONCE("literal_four_channels");
        red = ReadSymbol(htree_group->htrees[RED], br);"""),
    ("""  assert(Is8bOptimizable(hdr));""",
     """  assert(Is8bOptimizable(hdr));
  PROBE("alpha_8b_data");
  if (mask == ~0) PROBE("alpha_8b_no_blocks"); else PROBE("alpha_8b_blocks");"""),
    ("""          CopyBlock8b(src, dist, length);
        } else {
          ok = 0;""",
     """          PROBE("alpha_8b_copy");
          CopyBlock8b(src, dist, length);
        } else {
          PROBE("alpha_8b_copy_oob");
          ok = 0;"""),
    # Which arm of each copy the back-reference takes. Both functions write
    # words rather than bytes where the distance lets them, so the same copy
    # is four different pieces of code depending on its distance, its length
    # and where it starts.
    ("""  while ((uintptr_t)dst & 3) {
    *dst++ = *src++;""",
     """  while ((uintptr_t)dst & 3) {
    PROBE_ONCE("copy8b_unaligned");
    *dst++ = *src++;"""),
    ("""  for (i <<= 2; i < length; ++i) {
    dst[i] = src[i];
  }""",
     """  for (i <<= 2; i < length; ++i) {
    PROBE_ONCE("copy8b_tail");
    dst[i] = src[i];
  }"""),
    ("""      case 1:
        pattern = src[0];""",
     """      case 1:
        PROBE("copy8b_dist_1");
        pattern = src[0];"""),
    ("""      case 2:
#if !defined(WORDS_BIGENDIAN)""",
     """      case 2:
        PROBE("copy8b_dist_2");
#if !defined(WORDS_BIGENDIAN)"""),
    ("""      case 4:
        WEBP_UNSAFE_MEMCPY(&pattern, src, sizeof(uint32_t));
        break;
      default:
        goto Copy;""",
     """      case 4:
        PROBE("copy8b_dist_4");
        WEBP_UNSAFE_MEMCPY(&pattern, src, sizeof(uint32_t));
        break;
      default:
        PROBE("copy8b_no_pattern");
        goto Copy;"""),
    ("""Copy:
  if (dist >= length) {  // no overlap -> use WEBP_UNSAFE_MEMCPY()
    WEBP_UNSAFE_MEMCPY(dst, src, length * sizeof(*dst));""",
     """Copy:
  if (dist >= length) {  // no overlap -> use WEBP_UNSAFE_MEMCPY()
    PROBE("copy8b_memcpy");
    WEBP_UNSAFE_MEMCPY(dst, src, length * sizeof(*dst));"""),
    ("""  if (length & 1) {  // Finish with left-over.
    dst[i << 1] = src[i << 1];""",
     """  if (length & 1) {  // Finish with left-over.
    PROBE("copy32b_tail");
    dst[i << 1] = src[i << 1];"""),
    ("""    } else {
      WEBP_UNSAFE_MEMCPY(&pattern, src, sizeof(pattern));
    }""",
     """    } else {
      PROBE("copy32b_dist_2");
      WEBP_UNSAFE_MEMCPY(&pattern, src, sizeof(pattern));
    }"""),
    ("""  } else if (dist >= length) {  // no overlap
    WEBP_UNSAFE_MEMCPY(dst, src, length * sizeof(*dst));
  } else {
    int i;
    for (i = 0; i < length; ++i) dst[i] = src[i];
  }""",
     """  } else if (dist >= length) {  // no overlap
    WEBP_UNSAFE_MEMCPY(dst, src, length * sizeof(*dst));
  } else {
    int i;
    PROBE("copy32b_overlap");
    for (i = 0; i < length; ++i) dst[i] = src[i];
  }"""),
    ("""  if (plane_code > CODE_TO_PLANE_CODES) {
    return plane_code - CODE_TO_PLANE_CODES;""",
     """  if (plane_code > CODE_TO_PLANE_CODES) {
    PROBE("dist_direct");
    return plane_code - CODE_TO_PLANE_CODES;"""),
    ("""    return (dist >= 1) ? dist : 1;  // dist<1 can happen if xsize is very small""",
     """    if (plane_code == 1) PROBE("dist_plane_code_1");
    if (plane_code == CODE_TO_PLANE_CODES) PROBE("dist_plane_code_120");
    if (dist < 1) PROBE("dist_clamped_to_1");
    return (dist >= 1) ? dist : 1;  // dist<1 can happen if xsize is very small"""),
]

HUFFMAN = [
    ("""  // Error, all code lengths are zeros.
  if (count[0] == code_lengths_size) {
    return 0;
  }""",
     """  // Error, all code lengths are zeros.
  if (count[0] == code_lengths_size) {
    PROBE("build_empty_code");
    return 0;
  }"""),
    ("""    if (count[len] > (1 << len)) {
      return 0;
    }""",
     """    if (count[len] > (1 << len)) {
      PROBE("reject_count_over_capacity");
      return 0;
    }"""),
    ("""  // Special case code with only one value.
  if (offset[MAX_ALLOWED_CODE_LENGTH] == 1) {""",
     """  if (count[MAX_ALLOWED_CODE_LENGTH] > 0) PROBE("build_depth_15");
  { int l; for (l = root_bits + 1; l <= MAX_ALLOWED_CODE_LENGTH; ++l)
      if (count[l] > 0) { PROBE("build_two_level"); break; } }
  // Special case code with only one value.
  if (offset[MAX_ALLOWED_CODE_LENGTH] == 1) {
    PROBE("build_single_value");"""),
    ("""      num_open -= count[len];
      if (num_open < 0) {
        return 0;
      }
      if (root_table == NULL) continue;""",
     """      num_open -= count[len];
      if (num_open < 0) {
        PROBE("reject_oversubscribed");
        return 0;
      }
      if (root_table == NULL) continue;"""),
    ("""    if (num_nodes != 2 * offset[MAX_ALLOWED_CODE_LENGTH] - 1) {
      return 0;
    }""",
     """    if (num_nodes != 2 * offset[MAX_ALLOWED_CODE_LENGTH] - 1) {
      PROBE("reject_incomplete");
      return 0;
    }"""),
]

VP8 = [
    ("""  dec->num_parts_minus_one = (1 << VP8GetValue(br, 2, "global-header")) - 1;
  last_part = dec->num_parts_minus_one;
  if (size < 3 * last_part) {""",
     """  dec->num_parts_minus_one = (1 << VP8GetValue(br, 2, "global-header")) - 1;
  last_part = dec->num_parts_minus_one;
  PROBE1("parts_%d", (int)last_part + 1);
  if (size < 3 * last_part) {
    PROBE("parts_size_table_truncated");"""),
    ("""    if (psize > size_left) psize = size_left;""",
     """    if (psize > size_left) { PROBE("parts_psize_clamped"); psize = size_left; }
    if (psize == 0) PROBE("parts_psize_zero");"""),
    ("""  if (part_start < buf_end) return VP8_STATUS_OK;""",
     """  if (part_start < buf_end) return VP8_STATUS_OK;
  PROBE("parts_no_data_left");"""),
    # Why a file was rejected. Every VP8-level failure goes through here, so
    # one probe names the reason instead of leaving it to be guessed at.
    ("""  if (dec->status == VP8_STATUS_OK) {
    dec->status = error;""",
     """  if (dec->status == VP8_STATUS_OK) {
    { static char b[80]; int i;
      snprintf(b, sizeof(b), "error:%s", msg);
      for (i = 0; b[i] != 0; ++i) if (b[i] == ' ') b[i] = '-';
      PROBE(b); }
    dec->status = error;"""),
    # The checks in VP8GetInfo() reject before any of that, by returning 0.
    ("""  if (!VP8CheckSignature(data + 3, data_size - 3)) {
    return 0;  // Wrong signature.""",
     """  if (!VP8CheckSignature(data + 3, data_size - 3)) {
    PROBE("info:bad-signature");
    return 0;  // Wrong signature."""),
    ("""    if (!key_frame) {  // Not a keyframe.
      return 0;""",
     """    if (!key_frame) {  // Not a keyframe.
      PROBE("info:not-a-keyframe");
      return 0;"""),
    ("""      return 0;  // unknown profile""",
     """      PROBE("info:unknown-profile");
      return 0;  // unknown profile"""),
    ("""      return 0;  // first frame is invisible!""",
     """      PROBE("info:not-displayable");
      return 0;  // first frame is invisible!"""),
    ("""      return 0;                         // inconsistent size information.""",
     """      PROBE("info:partition-length");
      return 0;                         // inconsistent size information."""),
    ("""      return 0;  // We don't support both width and height to be zero.""",
     """      PROBE("info:zero-dimension");
      return 0;  // We don't support both width and height to be zero."""),
    # Coefficient magnitudes: which branch of section 13.2 each one took.
    ("""    if (!VP8GetBit(br, p[4], "coeffs")) {
      v = 2;""",
     """    if (!VP8GetBit(br, p[4], "coeffs")) {
      PROBE("coeff_v2");
      v = 2;"""),
    ("""      v = 3 + VP8GetBit(br, p[5], "coeffs");""",
     """      v = 3 + VP8GetBit(br, p[5], "coeffs");
      PROBE("coeff_v3_4");"""),
    ("""        v = 5 + VP8GetBit(br, 159, "coeffs");""",
     """        v = 5 + VP8GetBit(br, 159, "coeffs");
        PROBE("coeff_v5_6");"""),
    ("""        v = 7 + 2 * VP8GetBit(br, 165, "coeffs");
        v += VP8GetBit(br, 145, "coeffs");""",
     """        v = 7 + 2 * VP8GetBit(br, 165, "coeffs");
        v += VP8GetBit(br, 145, "coeffs");
        PROBE("coeff_v7_10");"""),
    ("""      v += 3 + (8 << cat);""",
     """      v += 3 + (8 << cat);
      PROBE1("coeff_cat%d", cat + 3);
      if (v == 2114) PROBE("coeff_max");"""),
    # The Y2 block decides between the full inverse WHT and the shortcut.
    ("""    if (nz > 1) {  // more than just the DC -> perform the full transform""",
     """    if (nz == 0) PROBE("wht_empty");
    if (nz == 1) PROBE("wht_dc_only");
    if (nz > 1) PROBE("wht_full");
    if (nz > 1) {  // more than just the DC -> perform the full transform"""),
    # A skipped macroblock clears its neighbours' flags -- except the Y2 one,
    # which it leaves alone when there is no Y2 block to clear.
    ("""    left->nz = mb->nz = 0;
    if (!block->is_i4x4) {
      left->nz_dc = mb->nz_dc = 0;
    }""",
     """    PROBE("mb_skipped");
    left->nz = mb->nz = 0;
    if (!block->is_i4x4) {
      left->nz_dc = mb->nz_dc = 0;
    } else {
      PROBE("skip_i4x4_keeps_nz_dc");
    }"""),
]

TREE = [
    # Segment id, skip flag, and the 16x16-or-4x4 decision, per macroblock.
    ("""  block->is_i4x4 = !VP8GetBit(br, 145, "block-size");""",
     """  block->is_i4x4 = !VP8GetBit(br, 145, "block-size");
  if (dec->segment_hdr.update_map) {
    PROBE1("segment_id_%d", block->segment);
  }
  if (dec->use_skip_proba && block->skip) PROBE("mb_skip_flag");
  PROBE(block->is_i4x4 ? "mb_i4x4" : "mb_i16");"""),
    ("""    block->imodes[0] = ymode;""",
     """    PROBE1("ymode_%d", ymode);
    block->imodes[0] = ymode;"""),
    ("""        top[x] = ymode;""",
     """        PROBE1("bmode_%d", ymode);
        top[x] = ymode;"""),
    ("""  block->uvmode = !VP8GetBit(br, 142, "pred-modes-uv")   ? DC_PRED
                  : !VP8GetBit(br, 114, "pred-modes-uv") ? V_PRED
                  : VP8GetBit(br, 183, "pred-modes-uv")  ? TM_PRED
                                                         : H_PRED;""",
     """  block->uvmode = !VP8GetBit(br, 142, "pred-modes-uv")   ? DC_PRED
                  : !VP8GetBit(br, 114, "pred-modes-uv") ? V_PRED
                  : VP8GetBit(br, 183, "pred-modes-uv")  ? TM_PRED
                                                         : H_PRED;
  PROBE1("uvmode_%d", block->uvmode);"""),
    # Whether a coefficient probability was signalled, not merely different.
    ("""          const int v =
              VP8GetBit(br, CoeffsUpdateProba[t][b][c][p], "global-header")
                  ? VP8GetValue(br, 8, "global-header")
                  : CoeffsProba0[t][b][c][p];""",
     """          const int updated =
              VP8GetBit(br, CoeffsUpdateProba[t][b][c][p], "global-header");
          const int v = updated ? VP8GetValue(br, 8, "global-header")
                                : CoeffsProba0[t][b][c][p];
          if (updated) PROBE("proba_update");
          if (updated && v == 0) PROBE("proba_zero");
          if (updated && v == 255) PROBE("proba_255");"""),
    ("""  dec->use_skip_proba = VP8Get(br, "global-header");""",
     """  dec->use_skip_proba = VP8Get(br, "global-header");
  if (dec->use_skip_proba) PROBE("skip_proba");"""),
]

QUANT = [
    ("""  for (i = 0; i < NUM_MB_SEGMENTS; ++i) {
    int q;
    if (hdr->use_segment) {""",
     """  if (dqy1_dc | dqy2_dc | dqy2_ac | dquv_dc | dquv_ac) PROBE("quant_delta");
  if (base_q0 == 0) PROBE("quant_min");
  if (base_q0 == 127) PROBE("quant_max");
  for (i = 0; i < NUM_MB_SEGMENTS; ++i) {
    int q;
    if (hdr->use_segment) {
      PROBE(hdr->absolute_delta ? "quant_segment_absolute"
                                : "quant_segment_delta");"""),
    ("""      m->y1_mat[0] = kDcTable[clip(q + dqy1_dc, 127)];""",
     """      if (q + dqy1_dc > 127 || q + dqy1_dc < 0) PROBE("quant_y1_dc_clipped");
      m->y1_mat[0] = kDcTable[clip(q + dqy1_dc, 127)];"""),
    ("""      if (m->y2_mat[1] < 8) m->y2_mat[1] = 8;""",
     """      if (m->y2_mat[1] < 8) PROBE("quant_y2_ac_floor");
      if (m->y2_mat[1] < 8) m->y2_mat[1] = 8;"""),
    ("""      m->uv_mat[0] = kDcTable[clip(q + dquv_dc, 117)];""",
     """      if (q + dquv_dc > 117) PROBE("quant_uv_dc_clamped_at_117");
      m->uv_mat[0] = kDcTable[clip(q + dquv_dc, 117)];"""),
]

DSP = [
    ("""        const VP8LPredictorAddSubFunc pred_func =
            VP8LPredictorsAdd[((*pred_mode_src++) >> 8) & 0xf];""",
     """        const int pred_mode = ((*pred_mode_src) >> 8) & 0xf;
        const VP8LPredictorAddSubFunc pred_func =
            VP8LPredictorsAdd[((*pred_mode_src++) >> 8) & 0xf];
        PROBE1("pred_mode_%d", pred_mode);"""),
    ("""      ColorCodeToMultipliers(*pred++, &m);
      VP8LTransformColorInverse(&m, src, tile_width, dst);""",
     """      ColorCodeToMultipliers(*pred++, &m);
      if (m.green_to_red | m.green_to_blue | m.red_to_blue) {
        PROBE("cross_color_multipliers");
      }
      VP8LTransformColorInverse(&m, src, tile_width, dst);"""),
    ("""  if (y_start == 0) {  // First Row follows the L (mode=1) mode.""",
     """  if (y_start == 0) PROBE("pred_first_row");
  if (y_start == 0) {  // First Row follows the L (mode=1) mode."""),
]

# The demuxer, which dwebp does not link at all: every probe below is
# reachable only through anim_dump, and only for a file with an animation
# column in expected.txt.
DEMUX = [
    # a frame's own chunk list
    ("""      case MKFOURCC('A', 'L', 'P', 'H'):
        if (alpha_chunks == 0) {""",
     """      case MKFOURCC('A', 'L', 'P', 'H'):
        PROBE("demux_frame_alph");
        if (alpha_chunks == 0) {"""),
    ("""        if (alpha_chunks > 0) return PARSE_ERROR;  // VP8L has its own alpha""",
     """        if (alpha_chunks > 0) {
          PROBE("demux_vp8l_with_alph");
          return PARSE_ERROR;  // VP8L has its own alpha
        }"""),
    ("""      Done:
      default:""",
     """      Done:
        PROBE("demux_frame_chunk_done");
      default:"""),
    # the ANMF header
    ("""  if (actual_size < min_size) return PARSE_ERROR;""",
     """  if (actual_size < min_size) {
    PROBE("demux_frame_too_small");
    return PARSE_ERROR;
  }"""),
    ("""  if (frame->width * (uint64_t)frame->height >= MAX_IMAGE_AREA) {""",
     """  if (frame->width * (uint64_t)frame->height >= MAX_IMAGE_AREA) {
    PROBE("demux_frame_area_overflow");"""),
    ("""    if (status != PARSE_ERROR &&
        mem->start - start_offset > anmf_payload_size) {
      status = PARSE_ERROR;
    }""",
     """    if (status != PARSE_ERROR &&
        mem->start - start_offset > anmf_payload_size) {
      PROBE("demux_frame_overran_payload");
      status = PARSE_ERROR;
    }"""),
    ("""  if (status != PARSE_ERROR && is_animation && frame->frame_num > 0) {""",
     """  if (status != PARSE_ERROR && is_animation && frame->frame_num == 0) {
    PROBE("demux_frame_dropped");
  }
  if (status != PARSE_ERROR && is_animation && frame->frame_num > 0) {
    PROBE("demux_frame_added");"""),
    # the top-level walk
    ("""      case MKFOURCC('V', 'P', '8', 'X'): {
        return PARSE_ERROR;
      }""",
     """      case MKFOURCC('V', 'P', '8', 'X'): {
        PROBE("demux_second_vp8x");
        return PARSE_ERROR;
      }"""),
    ("""        if (anim_chunks > 0 || is_animation) return PARSE_ERROR;""",
     """        if (anim_chunks > 0 || is_animation) {
          PROBE("demux_image_in_animation");
          return PARSE_ERROR;
        }"""),
    ("""        if (chunk_size_padded < ANIM_CHUNK_SIZE) return PARSE_ERROR;""",
     """        if (chunk_size_padded < ANIM_CHUNK_SIZE) {
          PROBE("demux_anim_too_small");
          return PARSE_ERROR;
        }
        if (chunk_size_padded > ANIM_CHUNK_SIZE) PROBE("demux_anim_padded");"""),
    ("""        } else {
          store_chunk = 0;
          goto Skip;
        }""",
     """        } else {
          PROBE("demux_anim_duplicate");
          store_chunk = 0;
          goto Skip;
        }"""),
    ("""        if (anim_chunks == 0) return PARSE_ERROR;  // 'ANIM' precedes frames.""",
     """        if (anim_chunks == 0) {
          PROBE("demux_frame_before_anim");
          return PARSE_ERROR;  // 'ANIM' precedes frames.
        }"""),
    ("""      Skip:
      default: {""",
     """      Skip:
        if (!store_chunk) PROBE("demux_chunk_flagless");
      default: {"""),
    # validation, once the whole file has been walked
    ("""  if (dmux->loop_count < 0) return 0;""",
     """  if (dmux->loop_count < 0) { PROBE("demux_loop_negative"); return 0; }"""),
    ("""  if (dmux->state == WEBP_DEMUX_DONE && dmux->frames == NULL) return 0;""",
     """  if (dmux->state == WEBP_DEMUX_DONE && dmux->frames == NULL) {
    PROBE("demux_no_frames");
    return 0;
  }"""),
    ("""      if (!is_animation && f->frame_num > 1) return 0;""",
     """      if (!is_animation && f->frame_num > 1) {
        PROBE("demux_frames_without_flag");
        return 0;
      }"""),
    ("""        if (alpha->size == 0 && image->size == 0) return 0;""",
     """        if (alpha->size == 0 && image->size == 0) {
          PROBE("demux_frame_empty");
          return 0;
        }"""),
    ("""        if (alpha->size > 0 && alpha->offset > image->offset) {
          return 0;
        }""",
     """        if (alpha->size > 0 && alpha->offset > image->offset) {
          PROBE("demux_alph_after_image");
          return 0;
        }"""),
    ("""    if (frame->width + frame->x_offset > canvas_width) return 0;
    if (frame->height + frame->y_offset > canvas_height) return 0;""",
     """    PROBE("demux_bounds_inexact");
    if (frame->width + frame->x_offset > canvas_width) {
      PROBE("demux_frame_past_canvas");
      return 0;
    }
    if (frame->height + frame->y_offset > canvas_height) {
      PROBE("demux_frame_past_canvas");
      return 0;
    }"""),
    ("""  if (!allow_partial && partial) return NULL;""",
     """  if (!allow_partial && partial) { PROBE("demux_partial"); return NULL; }"""),
    ("""  if (state != NULL) *state = dmux->state;""",
     """  if (status == PARSE_ERROR && dmux->state == WEBP_DEMUX_PARSING_HEADER) {
    PROBE("demux_no_master_chunk");
  }
  if (state != NULL) *state = dmux->state;"""),
]

# Frame composition: what the demuxer hands over is one image per frame, and
# this is everything that turns those into a canvas.
ANIM = [
    ("""  dec->blend_func = (mode == MODE_RGBA || mode == MODE_BGRA)""",
     """  PROBE((mode == MODE_RGBA || mode == MODE_BGRA) ? "anim_blend_nonpremult"
                                                 : "anim_blend_premult");
  dec->blend_func = (mode == MODE_RGBA || mode == MODE_BGRA)"""),
    ("""  if (curr->frame_num == 1) {
    return 1;
  } else if ((!curr->has_alpha || curr->blend_method == WEBP_MUX_NO_BLEND) &&""",
     """  if (curr->frame_num == 1) {
    PROBE("anim_key_first_frame");
    return 1;
  } else if ((!curr->has_alpha || curr->blend_method == WEBP_MUX_NO_BLEND) &&"""),
    ("""                         canvas_height)) {
    return 1;
  } else {""",
     """                         canvas_height)) {
    PROBE("anim_key_opaque_full_frame");
    return 1;
  } else {
    PROBE("anim_key_from_prev_dispose");"""),
    ("""  if (is_key_frame) {
    if (!ZeroFillCanvas(dec->curr_frame, width, height)) {""",
     """  PROBE(is_key_frame ? "anim_key_frame" : "anim_inter_frame");
  if (is_key_frame) {
    if (!ZeroFillCanvas(dec->curr_frame, width, height)) {"""),
    ("""    if (dec->prev_iter.dispose_method == WEBP_MUX_DISPOSE_NONE) {
      int y;""",
     """    if (dec->prev_iter.dispose_method == WEBP_MUX_DISPOSE_NONE) {
      int y;
      PROBE("anim_blend_whole_rows");"""),
    ("""      assert(dec->prev_iter.dispose_method == WEBP_MUX_DISPOSE_BACKGROUND);""",
     """      assert(dec->prev_iter.dispose_method == WEBP_MUX_DISPOSE_BACKGROUND);
      PROBE("anim_blend_around_prev");"""),
    ("""    *left1 = src->x_offset;
    *width1 = src->width;
    return;""",
     """    PROBE("anim_range_disjoint");
    *left1 = src->x_offset;
    *width1 = src->width;
    return;"""),
    ("""  if (src->x_offset < dst->x_offset) {
    *left1 = src->x_offset;""",
     """  if (src->x_offset < dst->x_offset) {
    PROBE("anim_range_left");
    *left1 = src->x_offset;"""),
    ("""  if (src_max_x > dst_max_x) {
    *left2 = dst_max_x;""",
     """  if (src_max_x > dst_max_x) {
    PROBE("anim_range_right");
    *left2 = dst_max_x;"""),
    ("""  if (dec->prev_iter.dispose_method == WEBP_MUX_DISPOSE_BACKGROUND) {
    ZeroFillFrameRect(""",
     """  if (dec->prev_iter.dispose_method == WEBP_MUX_DISPOSE_BACKGROUND) {
    PROBE("anim_dispose_background");
    ZeroFillFrameRect("""),
]

FILES = [('src/dec/vp8l_dec.c', VP8L), ('src/utils/huffman_utils.c', HUFFMAN),
         ('src/dec/vp8_dec.c', VP8), ('src/dec/tree_dec.c', TREE),
         ('src/dec/quant_dec.c', QUANT), ('src/dsp/lossless.c', DSP),
         ('src/demux/demux.c', DEMUX), ('src/demux/anim_decode.c', ANIM)]


def main():
    root = sys.argv[1].rstrip('/')
    missing = 0
    for rel, patches in FILES:
        path = '%s/%s' % (root, rel)
        with open(path) as f:
            s = f.read()
        at = s.index('#include')
        s = s[:at] + MACROS + s[at:]
        for old, new in patches:
            # An anchor that matches twice is worse than one that matches
            # never: replace() would take the first, and a probe planted on
            # a path nothing runs looks exactly like a path nothing covers.
            found = s.count(old)
            if found != 1:
                print('%s anchor in %s:\n  %s'
                      % ('MISSING' if found == 0 else '%d MATCHES for' % found,
                         rel, old.strip()[:70]))
                missing += 1
                continue
            s = s.replace(old, new, 1)
        with open(path, 'w') as f:
            f.write(s)
    print('patched %d files, %d anchors missing' % (len(FILES), missing))
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
