# webp-corners: what reaches what

Every construct `src/probes.py` measures, and the files that reach it.
[`BITSTREAMS.md`](BITSTREAMS.md) answers the same question backwards, a file at
a time, and the [notes](README.md) say what the corpus is. Generated from
`coverage.txt`, so this is the measurement rather than a claim about it.

| construct | files |
| --- | --- |
| `alpha_8b_blocks` | `alph-plane-meta-huffman` |
| `alpha_8b_copy` | `alph-lossless-palette`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-plain`, `alph-plane-lz77-unaligned` |
| `alpha_8b_copy_oob` | `alph-plane-lz77-past-start` |
| `alpha_8b_data` | `alph-lossless-palette`, `alph-plane-filtered`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-past-start` +6 |
| `alpha_8b_no_blocks` | `alph-lossless-palette`, `alph-plane-filtered`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-past-start` +5 |
| `alpha_is_used` | `anim-alpha-flag-missing`, `anim-blend-none`, `anim-blend-over`, `anim-blend-ranges`, `anim-dispose-background`, `anim-dispose-blend-matrix` +2 |
| `anim_blend_around_prev` | `anim-blend-ranges` |
| `anim_blend_nonpremult` | `anim-alph-after-image`, `anim-alpha-flag-missing`, `anim-alpha-lossless-frame`, `anim-alpha-raw-frame`, `anim-anim-chunk-padded`, `anim-anim-chunk-short` +41 |
| `anim_blend_whole_rows` | `anim-alpha-flag-missing`, `anim-blend-over`, `anim-blend-ranges`, `anim-canvas-larger-than-frames`, `anim-dispose-background`, `anim-dispose-blend-matrix` +2 |
| `anim_dispose_background` | `anim-blend-ranges`, `anim-dispose-background`, `anim-dispose-blend-matrix`, `anim-frame-1x1` |
| `anim_inter_frame` | `anim-alpha-flag-missing`, `anim-blend-over`, `anim-blend-ranges`, `anim-canvas-larger-than-frames`, `anim-dispose-background`, `anim-dispose-blend-matrix` +2 |
| `anim_key_first_frame` | `anim-alpha-flag-missing`, `anim-alpha-lossless-frame`, `anim-alpha-raw-frame`, `anim-anim-chunk-padded`, `anim-background-color`, `anim-blend-none` +21 |
| `anim_key_frame` | `anim-alpha-flag-missing`, `anim-alpha-lossless-frame`, `anim-alpha-raw-frame`, `anim-anim-chunk-padded`, `anim-background-color`, `anim-blend-none` +21 |
| `anim_key_from_prev_dispose` | `anim-alpha-flag-missing`, `anim-blend-over`, `anim-blend-ranges`, `anim-canvas-larger-than-frames`, `anim-dispose-background`, `anim-dispose-blend-matrix` +2 |
| `anim_key_opaque_full_frame` | `anim-alpha-raw-frame`, `anim-blend-none`, `anim-dispose-background`, `anim-duplicate-anim`, `anim-duration-extremes`, `anim-eight-frames` +6 |
| `anim_range_disjoint` | `anim-blend-ranges` |
| `anim_range_left` | `anim-blend-ranges` |
| `anim_range_right` | `anim-blend-ranges` |
| `bmode_0` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +21 |
| `bmode_1` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +7 |
| `bmode_2` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-coeff-all-types`, `lossy-combo-all-features` +6 |
| `bmode_3` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +7 |
| `bmode_4` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +7 |
| `bmode_5` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-coeff-all-types`, `lossy-combo-all-features` +6 |
| `bmode_6` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +7 |
| `bmode_7` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-coeff-all-types`, `lossy-combo-all-features` +6 |
| `bmode_8` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +7 |
| `bmode_9` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +7 |
| `build_depth_15` | `codelen-depth-15` |
| `build_empty_code` | `codelen-all-zero-lengths`, `header-max-area-truncated`, `meta-huffman-groups-truncated`, `simple-dist-1sym-oob`, `simple-dist-2sym-both-oob`, `simple-dist-sym-40-first-oob` +1 |
| `build_single_value` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-lossless-truncated`, `alph-plane-cache`, `alph-plane-filtered` +109 |
| `build_two_level` | `codelen-depth-15`, `codelen-two-level-table` |
| `cache_bits_max` | `cache-bits-11`, `subimage-cache-palette-max` |
| `cache_bits_min` | `cache-bits-1`, `cache-index-literal`, `subimage-cache-predictor-min` |
| `cache_index` | `cache-index-literal`, `lossless-all-features` |
| `codelen_16` | `codelen-repeat16-no-previous` |
| `codelen_16_default_prev` | `codelen-repeat16-no-previous` |
| `codelen_17` | `alph-lossless-byte-flipped`, `alph-lossless-predictor`, `alph-lossless-truncated`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-past-start`, `alph-plane-lz77-plain` +2 |
| `codelen_17_max_run` | `alph-plane-lz77-past-start`, `codelen-repeat17-short-zeros`, `lz77-plane-code-120` |
| `codelen_18` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4` +31 |
| `codelen_18_max_run` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-past-start` +20 |
| `coeff_cat3` | `anim-lossy-frames`, `container-duplicate-image-chunk`, `container-metadata-chunks`, `container-odd-chunk-payload`, `container-trailing-bytes`, `container-unknown-chunk` +39 |
| `coeff_cat4` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +72 |
| `coeff_cat5` | `lossy-coeff-cat5` |
| `coeff_cat6` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-coeff-cat6`, `lossy-coeff-cat6-max` +1 |
| `coeff_max` | `lossy-coeff-cat6-max`, `lossy-quant-dequant-overflow` |
| `coeff_v2` | `container-duplicate-image-chunk`, `container-metadata-chunks`, `container-odd-chunk-payload`, `container-trailing-bytes`, `container-unknown-chunk`, `container-vp8x` +35 |
| `coeff_v3_4` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +65 |
| `coeff_v5_6` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +43 |
| `coeff_v7_10` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-coeff-all-types`, `lossy-coeff-full-block` +17 |
| `complex_code` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-lossless-truncated`, `alph-plane-cache`, `alph-plane-lz77` +51 |
| `copy32b_dist_2` | `lz77-distance-2-pattern` |
| `copy32b_overlap` | `lz77-distance-3-overlap` |
| `copy32b_tail` | `lz77-distance-2-pattern` |
| `copy8b_dist_1` | `alph-plane-lz77`, `alph-plane-lz77-unaligned` |
| `copy8b_dist_2` | `alph-plane-lz77-dist-2` |
| `copy8b_dist_4` | `alph-plane-lz77-dist-4` |
| `copy8b_memcpy` | `alph-plane-lz77-plain` |
| `copy8b_no_pattern` | `alph-plane-lz77-plain` |
| `copy8b_tail` | `alph-plane-lz77-unaligned` |
| `copy8b_unaligned` | `alph-plane-lz77-unaligned` |
| `cross_color_multipliers` | `transform-cross-color-multipliers` |
| `demux_alph_after_image` | `anim-alph-after-image` |
| `demux_anim_duplicate` | `anim-duplicate-anim` |
| `demux_anim_padded` | `anim-anim-chunk-padded` |
| `demux_anim_too_small` | `anim-anim-chunk-short` |
| `demux_bounds_inexact` | `anim-alpha-flag-missing`, `anim-alpha-lossless-frame`, `anim-alpha-raw-frame`, `anim-anim-chunk-padded`, `anim-background-color`, `anim-blend-none` +23 |
| `demux_chunk_flagless` | `anim-duplicate-anim`, `anim-metadata-without-flags` |
| `demux_frame_added` | `anim-alph-after-image`, `anim-alpha-flag-missing`, `anim-alpha-lossless-frame`, `anim-alpha-raw-frame`, `anim-anim-chunk-padded`, `anim-anmf-odd-size` +29 |
| `demux_frame_alph` | `anim-alph-after-image`, `anim-alpha-lossless-frame`, `anim-alpha-raw-frame`, `anim-frame-alpha-only`, `anim-vp8l-with-alph` |
| `demux_frame_area_overflow` | `anim-frame-area-overflow` |
| `demux_frame_before_anim` | `anim-no-anim-chunk` |
| `demux_frame_chunk_done` | `anim-image-chunk-beside-frames`, `anim-two-images-in-frame` |
| `demux_frame_dropped` | `anim-empty-frame`, `anim-empty-frame-alone`, `anim-nested-anmf` |
| `demux_frame_overran_payload` | `anim-anmf-size-short` |
| `demux_frame_past_canvas` | `anim-frame-image-past-canvas`, `anim-frame-past-canvas` |
| `demux_frame_too_small` | `anim-anmf-header-truncated` |
| `demux_image_in_animation` | `anim-image-chunk-beside-frames`, `anim-two-images-in-frame` |
| `demux_no_frames` | `anim-frames-without-flag` |
| `demux_partial` | `anim-riff-size-past-end` |
| `demux_second_vp8x` | `anim-second-vp8x` |
| `demux_vp8l_with_alph` | `anim-vp8l-with-alph` |
| `dist_clamped_to_1` | `lz77-plane-code-clamped-to-1` |
| `dist_direct` | `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-plain`, `alph-plane-lz77-unaligned`, `lz77-distance-2-pattern`, `lz77-distance-3-overlap` +2 |
| `dist_plane_code_1` | `lz77-distance-past-start`, `lz77-plane-code-1` |
| `dist_plane_code_120` | `lz77-plane-code-120` |
| `error:Alpha-decoder-initialization-failed.` | `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-truncated`, `alph-plane-oversubscribed`, `alph-preprocessing-invalid`, `alph-raw-short` +1 |
| `error:Could-not-decode-alpha-data.` | `alph-lossless-byte-flipped`, `alph-plane-lz77-past-start` |
| `error:Premature-end-of-file-encountered.` | `lossy-8-partitions-zero-sizes`, `lossy-truncated-tokens` |
| `error:Premature-end-of-partition0-encountered.` | `lossy-truncated-modes` |
| `error:cannot-parse-filter-header` | `lossy-truncated-header` |
| `error:cannot-parse-partitions` | `lossy-8-partitions-size-overflow`, `lossy-8-partitions-sizes-sum-past-end`, `lossy-parts-last-empty`, `lossy-parts-size-past-end`, `lossy-parts-table-too-small` |
| `error:cannot-parse-segment-header` | `lossy-frame-part0-empty` |
| `header_huge_area` | `header-max-area-bomb`, `header-max-area-truncated` |
| `header_width_16384` | `header-max-area-bomb`, `header-max-area-truncated`, `header-width-16384` |
| `info:bad-signature` | `alph-no-vp8x`, `anim-no-vp8x`, `lossy-frame-bad-start-code` |
| `info:not-a-keyframe` | `lossy-frame-interframe` |
| `info:not-displayable` | `lossy-frame-not-shown` |
| `info:partition-length` | `lossy-frame-part0-past-end` |
| `info:unknown-profile` | `lossy-frame-version-4`, `lossy-frame-version-7` |
| `info:zero-dimension` | `lossy-frame-zero-width` |
| `literal_four_channels` | `alph-lossless-byte-flipped`, `alph-lossless-predictor`, `lossless-plain`, `transform-palette-3-colors`, `transform-palette-index-past-end` |
| `literal_trivial_arb` | `codelen-depth-15`, `codelen-two-level-table`, `lz77-distance-2-pattern`, `lz77-distance-3-overlap`, `predictor-all-16-modes`, `predictor-tile-bits-min` +1 |
| `literal_trivial_code` | `alph-lossless-byte-flipped`, `alph-lossless-predictor`, `alph-lossless-truncated`, `alph-plane-cache`, `alph-plane-literals`, `alph-plane-two-transforms` +62 |
| `lz77_copy` | `alph-lossless-byte-flipped`, `alph-lossless-predictor`, `lossless-all-features`, `lz77-distance-1-run`, `lz77-distance-2-pattern`, `lz77-distance-3-overlap` +7 |
| `lz77_max_len_sym` | `lz77-max-length-symbol` |
| `max_symbol_early_stop` | `codelen-max-symbol-early-stop`, `subimage-code-max-symbol` |
| `mb_i16` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +91 |
| `mb_i4x4` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-coeff-all-types` +21 |
| `mb_skip_flag` | `lossy-combo-all-features`, `lossy-proba-refresh-and-skip-zero`, `lossy-skip-all`, `lossy-skip-i4x4-nz-dc`, `lossy-skip-mixed`, `lossy-truncated-short-modes` |
| `mb_skipped` | `lossy-combo-all-features`, `lossy-proba-refresh-and-skip-zero`, `lossy-skip-all`, `lossy-skip-i4x4-nz-dc`, `lossy-skip-mixed`, `lossy-truncated-short-modes` |
| `meta_group_switch` | `lossless-all-features`, `meta-huffman-1001-groups`, `meta-huffman-per-tile-data`, `meta-huffman-sparse-groups`, `meta-huffman-two-groups`, `subimage-cache-entropy-image` |
| `meta_groups_over_1000` | `meta-huffman-1001-groups` |
| `meta_huffman` | `alph-plane-meta-huffman`, `lossless-all-features`, `meta-huffman-1001-groups`, `meta-huffman-groups-truncated`, `meta-huffman-per-tile-data`, `meta-huffman-precision-max` +4 |
| `meta_mapping_path` | `meta-huffman-1001-groups`, `meta-huffman-sparse-groups` |
| `meta_precision_max` | `meta-huffman-precision-max` |
| `meta_precision_min` | `alph-plane-meta-huffman`, `lossless-all-features`, `meta-huffman-1001-groups`, `meta-huffman-groups-truncated`, `meta-huffman-per-tile-data`, `meta-huffman-precision-min` +3 |
| `meta_unused_group` | `meta-huffman-1001-groups`, `meta-huffman-sparse-groups` |
| `num_codes_max` | `codelen-depth-15`, `codelen-num-codes-19` |
| `num_codes_min` | `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-past-start`, `alph-plane-lz77-plain`, `alph-plane-lz77-unaligned` +26 |
| `palette_1` | `transform-palette-1-color` |
| `palette_16` | `transform-palette-16-colors` |
| `palette_2` | `alph-lossless-palette`, `subimage-cache-palette-max`, `transform-all-four`, `transform-palette-2-colors` |
| `palette_256` | `transform-palette-256-colors` |
| `palette_3` | `alph-plane-cache`, `alph-plane-filtered`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-past-start` +10 |
| `palette_black_tail` | `alph-plane-cache`, `alph-plane-filtered`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4`, `alph-plane-lz77-past-start` +11 |
| `parts_1` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +99 |
| `parts_2` | `lossy-2-partitions`, `lossy-parts-2-wrap` |
| `parts_4` | `lossy-4-partitions`, `lossy-combo-all-features`, `lossy-parts-last-empty`, `lossy-parts-size-past-end` |
| `parts_8` | `lossy-8-partitions`, `lossy-8-partitions-size-overflow`, `lossy-8-partitions-sizes-sum-past-end`, `lossy-8-partitions-zero-sizes`, `lossy-parts-8-rows`, `lossy-parts-table-too-small` |
| `parts_no_data_left` | `lossy-8-partitions-size-overflow`, `lossy-8-partitions-sizes-sum-past-end`, `lossy-parts-last-empty`, `lossy-parts-size-past-end` |
| `parts_psize_clamped` | `lossy-8-partitions-size-overflow`, `lossy-8-partitions-sizes-sum-past-end`, `lossy-parts-last-empty`, `lossy-parts-size-past-end` |
| `parts_psize_zero` | `lossy-8-partitions-size-overflow`, `lossy-8-partitions-sizes-sum-past-end`, `lossy-8-partitions-zero-sizes`, `lossy-parts-size-past-end` |
| `parts_size_table_truncated` | `lossy-parts-table-too-small` |
| `pred_first_row` | `alph-lossless-predictor`, `lossless-all-features`, `predictor-all-16-modes`, `predictor-mode-11-select`, `predictor-mode-13-clamp-half`, `predictor-mode-14-undefined` +8 |
| `pred_mode_0` | `lossless-all-features`, `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form`, `transform-predictor-bits-max` |
| `pred_mode_1` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form`, `subimage-code-max-symbol` |
| `pred_mode_10` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_11` | `predictor-all-16-modes`, `predictor-mode-11-select`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_12` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_13` | `predictor-all-16-modes`, `predictor-mode-13-clamp-half`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_14` | `predictor-all-16-modes`, `predictor-mode-14-undefined`, `subimage-code-complex-form` |
| `pred_mode_15` | `predictor-all-16-modes`, `predictor-mode-15-undefined`, `subimage-code-complex-form` |
| `pred_mode_2` | `alph-lossless-predictor`, `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-cache-predictor-min`, `subimage-code-complex-form` |
| `pred_mode_3` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_4` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_5` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_6` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_7` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_8` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `pred_mode_9` | `predictor-all-16-modes`, `predictor-tile-bits-min`, `subimage-code-complex-form` |
| `proba_255` | `lossy-proba-zero` |
| `proba_update` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-combo-all-features` +3 |
| `proba_zero` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-proba-zero` |
| `quant_delta` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-quant-deltas` +2 |
| `quant_max` | `lossy-quant-dequant-overflow`, `lossy-quant-max` |
| `quant_min` | `lossy-coeff-cat3`, `lossy-coeff-cat4`, `lossy-coeff-cat5`, `lossy-coeff-cat6`, `lossy-coeff-cat6-max`, `lossy-coeff-full-block` +4 |
| `quant_segment_absolute` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-combo-all-features` +6 |
| `quant_segment_delta` | `lossy-segment-delta-quantizers` |
| `quant_uv_dc_clamped_at_117` | `lossy-quant-dequant-overflow`, `lossy-quant-max`, `lossy-quant-uv-dc-clamp`, `lossy-segment-four-quantizers`, `lossy-segment-no-map`, `lossy-segment-quant-extremes` |
| `quant_y1_dc_clipped` | `lossy-segment-quant-extremes` |
| `quant_y2_ac_floor` | `lossy-coeff-cat3`, `lossy-coeff-cat4`, `lossy-coeff-cat5`, `lossy-coeff-cat6`, `lossy-coeff-cat6-max`, `lossy-coeff-full-block` +6 |
| `reject_cache_bits` | `cache-bits-0-invalid`, `cache-bits-12-invalid`, `subimage-cache-12-invalid`, `subimage-cache-zero-invalid` |
| `reject_count_over_capacity` | `alph-plane-oversubscribed`, `codelen-over-capacity` |
| `reject_incomplete` | `codelen-incomplete` |
| `reject_max_symbol_too_big` | `codelen-max-symbol-too-big` |
| `reject_oversubscribed` | `codelen-oversubscribed`, `subimage-code-oversubscribed` |
| `reject_repeat_past_end` | `codelen-repeat-past-end` |
| `reject_signature_magic` | `alph-no-vp8x`, `anim-no-vp8x`, `header-magic-wrong` |
| `reject_signature_version` | `header-version-max`, `header-version-nonzero` |
| `reject_transform_repeated` | `transform-repeated` |
| `segment_id_0` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-combo-all-features`, `lossy-segment-delta-quantizers` +5 |
| `segment_id_1` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-combo-all-features`, `lossy-segment-delta-quantizers` +5 |
| `segment_id_2` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-combo-all-features` +6 |
| `segment_id_3` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-combo-all-features` +6 |
| `simple_1sym` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-lossless-truncated`, `alph-plane-cache`, `alph-plane-filtered` +109 |
| `simple_2sym` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-plane-filtered`, `alph-plane-meta-huffman`, `alph-plane-nontrivial-red` +19 |
| `simple_sym1_oob` | `simple-dist-1sym-oob`, `simple-dist-2sym-both-oob`, `simple-dist-2sym-first-oob`, `simple-dist-sym-40-first-oob` |
| `simple_sym2_oob` | `simple-dist-2sym-both-oob`, `simple-dist-2sym-second-oob` |
| `simple_sym_1bit` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-lossless-truncated`, `simple-green-1bit-symbol` |
| `skip_i4x4_keeps_nz_dc` | `lossy-skip-i4x4-nz-dc` |
| `skip_proba` | `lossy-combo-all-features`, `lossy-proba-refresh-and-skip-zero`, `lossy-proba-skip-extremes`, `lossy-skip-all`, `lossy-skip-i4x4-nz-dc`, `lossy-skip-mixed` +1 |
| `subimage_cache` | `subimage-cache-12-invalid`, `subimage-cache-entropy-image`, `subimage-cache-palette-max`, `subimage-cache-predictor-min`, `subimage-cache-zero-invalid` |
| `subimage_stream` | `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor`, `alph-lossless-truncated`, `alph-plane-cache`, `alph-plane-filtered` +46 |
| `tf_bits_max` | `alph-lossless-byte-flipped`, `alph-lossless-predictor`, `alph-lossless-truncated`, `transform-cross-color-bits-max`, `transform-predictor-bits-max` |
| `tf_bits_min` | `lossless-all-features`, `predictor-all-16-modes`, `predictor-mode-11-select`, `predictor-mode-13-clamp-half`, `predictor-mode-14-undefined`, `predictor-mode-15-undefined` +11 |
| `tf_color_indexing` | `alph-lossless-palette`, `alph-plane-cache`, `alph-plane-filtered`, `alph-plane-lz77`, `alph-plane-lz77-dist-2`, `alph-plane-lz77-dist-4` +18 |
| `tf_cross_color` | `lossless-all-features`, `subimage-cache-zero-invalid`, `transform-all-four`, `transform-cross-color-bits-max`, `transform-cross-color-multipliers` |
| `tf_predictor` | `alph-lossless-byte-flipped`, `alph-lossless-predictor`, `alph-lossless-truncated`, `lossless-all-features`, `predictor-all-16-modes`, `predictor-mode-11-select` +13 |
| `tf_subtract_green` | `alph-plane-two-transforms`, `lossless-all-features`, `transform-all-four`, `transform-repeated` |
| `use_length` | `codelen-max-symbol-early-stop`, `codelen-max-symbol-too-big`, `subimage-code-max-symbol` |
| `uvmode_0` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +106 |
| `uvmode_1` | `lossy-coeff-all-types`, `lossy-combo-all-features`, `lossy-filter-lf-delta`, `lossy-filter-normal-max`, `lossy-filter-simple-max`, `lossy-mode-mixed` +5 |
| `uvmode_2` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-filter-lf-delta`, `lossy-filter-normal-max` +6 |
| `uvmode_3` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-combo-all-features` +18 |
| `wht_dc_only` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +79 |
| `wht_empty` | `anim-alpha-lossless-frame`, `anim-alpha-raw-frame`, `anim-lossy-frames`, `anim-mixed-formats`, `container-duplicate-image-chunk`, `container-metadata-chunks` +34 |
| `wht_full` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-coeff-all-types`, `lossy-coeff-bands-i16` +8 |
| `ymode_0` | `alph-after-image`, `alph-compression-invalid`, `alph-empty-payload`, `alph-lossless-byte-flipped`, `alph-lossless-palette`, `alph-lossless-predictor` +91 |
| `ymode_1` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-8-partitions-zero-sizes`, `lossy-combo-all-features` +12 |
| `ymode_2` | `anim-lossy-frames`, `container-duplicate-image-chunk`, `container-metadata-chunks`, `container-odd-chunk-payload`, `container-trailing-bytes`, `container-unknown-chunk` +33 |
| `ymode_3` | `lossy-1-partitions`, `lossy-2-partitions`, `lossy-4-partitions`, `lossy-8-partitions`, `lossy-filter-normal-max`, `lossy-filter-simple-max` +9 |
