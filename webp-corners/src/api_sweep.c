// Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree.
// -----------------------------------------------------------------------------
// Every decoding entry point libwebp exports, over each file named on the
// command line: the one-shot convenience calls, the incremental decoder fed
// a few bytes at a time, the caller-allocated variants, the colorspaces no
// command line tool asks for, the demuxer's iterators, and the animation
// decoder in both blending modes.
//
// Used by coverage.sh, and only for coverage: nothing here checks a verdict,
// it only asks what a bitstream can be made to reach. dwebp -incremental
// hands the decoder the whole buffer in one call, so on its own it never
// suspends mid-stream and the streaming paths stay dark.
#include <stdio.h>
#include <stdlib.h>

#include "webp/decode.h"
#include "webp/demux.h"

static void Free(void* p) { WebPFree(p); }

static void OneShot(const uint8_t* data, size_t size, int w, int h) {
  int W, H;
  uint8_t* u;
  Free(WebPDecodeRGBA(data, size, &W, &H));
  Free(WebPDecodeARGB(data, size, &W, &H));
  Free(WebPDecodeBGRA(data, size, &W, &H));
  Free(WebPDecodeRGB(data, size, &W, &H));
  Free(WebPDecodeBGR(data, size, &W, &H));
  {
    uint8_t *y = NULL, *uu = NULL, *v = NULL;
    int stride = 0, uv_stride = 0;
    y = WebPDecodeYUV(data, size, &W, &H, &uu, &v, &stride, &uv_stride);
    Free(y);
  }
  if (w <= 0 || h <= 0) return;
  // the caller-allocated forms, once with room and once one byte short
  // the *Into forms write into the caller's buffer and hand the same pointer
  // back, so nothing here is freed but the buffer itself
  u = (uint8_t*)malloc((size_t)w * h * 4);
  if (u != NULL) {
    const size_t big = (size_t)w * h * 4;
    (void)WebPDecodeRGBAInto(data, size, u, big, w * 4);
    (void)WebPDecodeBGRAInto(data, size, u, big, w * 4);
    (void)WebPDecodeRGBInto(data, size, u, big, w * 3);
    (void)WebPDecodeBGRInto(data, size, u, big, w * 3);
    (void)WebPDecodeARGBInto(data, size, u, big, w * 4);
    (void)WebPDecodeRGBAInto(data, size, u, big / 2, w * 4);  // too small
    {
      uint8_t* const uv = u + (size_t)w * h;
      (void)WebPDecodeYUVInto(data, size, u, (size_t)w * h, w, uv,
                              (size_t)w * h / 4, (w + 1) / 2,
                              uv + (size_t)w * h / 4, (size_t)w * h / 4,
                              (w + 1) / 2);
    }
    free(u);
  }
}

// feeds the file in 'pieces' slices, through WebPIAppend (the decoder keeps
// its own copy) or WebPIUpdate (the caller's buffer just grows)
static void Drip(WebPIDecoder* idec, const uint8_t* data, size_t size,
                 int pieces, int append) {
  const size_t step = (size + pieces - 1) / (pieces > 0 ? pieces : 1) + 1;
  size_t at = 0;
  if (idec == NULL) return;
  while (at < size) {
    const size_t n = (size - at < step) ? size - at : step;
    int last_y, w, h, stride, uv_stride;
    uint8_t *u, *v, *a;
    if (append) {
      if (WebPIAppend(idec, data + at, n) > VP8_STATUS_SUSPENDED) break;
    } else {
      if (WebPIUpdate(idec, data, at + n) > VP8_STATUS_SUSPENDED) break;
    }
    at += n;
    (void)WebPIDecGetRGB(idec, &last_y, &w, &h, &stride);
    (void)WebPIDecGetYUVA(idec, &last_y, &u, &v, &a, &w, &h, &stride,
                          &uv_stride, &stride);
    (void)WebPIDecodedArea(idec, &last_y, &w, &h, &stride);
  }
  WebPIDelete(idec);
}

static void Incremental(const uint8_t* data, size_t size, int w, int h) {
  WebPDecoderConfig config;
  int pieces;
  for (pieces = 1; pieces <= 64; pieces *= 8) {
    WebPDecBuffer out;
    if (!WebPInitDecBuffer(&out)) return;
    out.colorspace = MODE_RGBA;
    Drip(WebPINewDecoder(&out), data, size, pieces, 1);
    WebPFreeDecBuffer(&out);
    if (!WebPInitDecBuffer(&out)) return;
    out.colorspace = MODE_YUVA;
    Drip(WebPINewDecoder(&out), data, size, pieces, 0);
    WebPFreeDecBuffer(&out);
  }
  // the same, but with the output buffer supplied by the caller
  if (w > 0 && h > 0 && (long)w * h < (1 << 22)) {
    const size_t rgba_size = (size_t)w * h * 4;
    uint8_t* const buf = (uint8_t*)malloc(rgba_size);
    if (buf != NULL) {
      Drip(WebPINewRGB(MODE_RGBA, buf, rgba_size, w * 4), data, size, 8, 1);
      Drip(WebPINewRGB(MODE_BGR, buf, rgba_size, w * 3), data, size, 3, 0);
      {
        const size_t luma = (size_t)w * h;
        const size_t uv = ((size_t)(w + 1) / 2) * ((h + 1) / 2);
        uint8_t* const planes = (uint8_t*)malloc(luma * 2 + uv * 2);
        if (planes != NULL) {
          uint8_t* const pu = planes + luma;
          uint8_t* const pv = pu + uv;
          uint8_t* const pa = pv + uv;
          Drip(WebPINewYUVA(planes, luma, w, pu, uv, (w + 1) / 2, pv, uv,
                            (w + 1) / 2, pa, luma, w),
               data, size, 8, 1);
          Drip(WebPINewYUV(planes, luma, w, pu, uv, (w + 1) / 2, pv, uv,
                           (w + 1) / 2),
               data, size, 8, 1);
          free(planes);
        }
      }
      free(buf);
    }
  }
  // and with the options the config carries: scaling, cropping, flipping
  if (WebPInitDecoderConfig(&config)) {
    config.options.use_scaling = 1;
    config.options.scaled_width = 23;
    config.options.scaled_height = 17;
    config.options.flip = 1;
    config.output.colorspace = MODE_rgbA;
    (void)WebPValidateDecoderConfig(&config);
    Drip(WebPIDecode(NULL, 0, &config), data, size, 8, 1);
    WebPFreeDecBuffer(&config.output);
  }
  if (WebPInitDecoderConfig(&config)) {
    config.options.use_cropping = 1;
    config.options.crop_left = 2;
    config.options.crop_top = 2;
    config.options.crop_width = 5;
    config.options.crop_height = 5;
    config.options.dithering_strength = 100;
    config.options.alpha_dithering_strength = 100;
    config.options.use_threads = 1;
    config.output.colorspace = MODE_Argb;
    (void)WebPValidateDecoderConfig(&config);
    Drip(WebPIDecode(NULL, 0, &config), data, size, 8, 1);
    WebPFreeDecBuffer(&config.output);
  }
}

// the colorspaces no command-line tool asks for, and the "slow memory"
// output that makes the library decode into a scratch buffer and copy
static void Modes(const uint8_t* data, size_t size, int w, int h) {
  static const WEBP_CSP_MODE kModes[] = {MODE_RGBA_4444, MODE_RGB_565,
                                         MODE_rgbA_4444, MODE_bgrA};
  size_t i;
  for (i = 0; i < sizeof(kModes) / sizeof(kModes[0]); ++i) {
    WebPDecoderConfig config;
    if (!WebPInitDecoderConfig(&config)) return;
    config.output.colorspace = kModes[i];
    (void)WebPDecode(data, size, &config);
    WebPFreeDecBuffer(&config.output);
    if (!WebPInitDecoderConfig(&config)) return;
    config.output.colorspace = kModes[i];
    config.options.use_scaling = 1;
    config.options.scaled_width = 19;
    config.options.scaled_height = 11;
    Drip(WebPIDecode(NULL, 0, &config), data, size, 8, 1);
    WebPFreeDecBuffer(&config.output);
  }
  if (w <= 0 || h <= 0) return;
  {
    const size_t n = (size_t)w * h * 4;
    uint8_t* const buf = (uint8_t*)malloc(n);
    WebPDecoderConfig config;
    if (buf == NULL) return;
    if (WebPInitDecoderConfig(&config)) {
      config.output.colorspace = MODE_rgbA;  // premultiplied, as the check asks
      config.output.is_external_memory = 2;  // slow memory: decode, then copy
      config.output.u.RGBA.rgba = buf;
      config.output.u.RGBA.size = n;
      config.output.u.RGBA.stride = w * 4;
      (void)WebPDecode(data, size, &config);
    }
    if (WebPInitDecoderConfig(&config)) {
      config.output.colorspace = MODE_rgbA;
      config.output.is_external_memory = 2;
      config.output.u.RGBA.rgba = buf;
      config.output.u.RGBA.size = n;
      config.output.u.RGBA.stride = w * 4;
      config.options.flip = 1;
      Drip(WebPIDecode(NULL, 0, &config), data, size, 8, 1);
    }
    free(buf);
  }
}

static void Iterate(const WebPDemuxer* dmux) {
  WebPIterator iter;
  WebPChunkIterator chunk;
  int i;
  static const char* const kTags[] = {"ICCP", "EXIF", "XMP ", "ZZZZ"};
  for (i = 0; i <= WEBP_FF_BACKGROUND_COLOR; ++i) {
    (void)WebPDemuxGetI(dmux, (WebPFormatFeature)i);
  }
  if (WebPDemuxGetFrame(dmux, 1, &iter)) {
    do {
      (void)iter.fragment.size;
    } while (WebPDemuxNextFrame(&iter));
    while (WebPDemuxPrevFrame(&iter)) continue;
    WebPDemuxReleaseIterator(&iter);
  }
  if (WebPDemuxGetFrame(dmux, 0, &iter)) WebPDemuxReleaseIterator(&iter);
  for (i = 0; i < 4; ++i) {
    if (WebPDemuxGetChunk(dmux, kTags[i], 1, &chunk)) {
      while (WebPDemuxNextChunk(&chunk)) continue;
      while (WebPDemuxPrevChunk(&chunk)) continue;
      WebPDemuxReleaseChunkIterator(&chunk);
    }
  }
}

static void Demux(const uint8_t* data, size_t size) {
  WebPData wd = {data, size};
  WebPDemuxer* dmux = WebPDemux(&wd);
  if (dmux != NULL) {
    Iterate(dmux);
    WebPDemuxDelete(dmux);
  }
  // partial, at a few cut points: header only, half, all but one byte
  {
    const size_t cuts[] = {8, 20, size / 2, size > 0 ? size - 1 : 0};
    size_t i;
    for (i = 0; i < 4; ++i) {
      WebPDemuxState state;
      WebPData part = {data, cuts[i] < size ? cuts[i] : size};
      WebPDemuxer* const p = WebPDemuxPartial(&part, &state);
      if (p != NULL) {
        Iterate(p);
        WebPDemuxDelete(p);
      }
    }
  }
  // a headerless stream: what is left once the RIFF and chunk headers are
  // dropped is a bare VP8/VP8L frame for the simple files
  if (size > 20) {
    WebPData raw = {data + 20, size - 20};
    WebPDemuxer* const p = WebPDemux(&raw);
    if (p != NULL) {
      Iterate(p);
      WebPDemuxDelete(p);
    }
  }
}

static void Animate(const uint8_t* data, size_t size, int premult,
                    int threads) {
  WebPData wd = {data, size};
  WebPAnimDecoderOptions opts;
  WebPAnimDecoder* dec;
  if (!WebPAnimDecoderOptionsInit(&opts)) return;
  opts.color_mode = premult ? MODE_rgbA : MODE_BGRA;
  opts.use_threads = threads;
  dec = WebPAnimDecoderNew(&wd, &opts);
  if (dec == NULL) return;
  {
    WebPAnimInfo info;
    if (WebPAnimDecoderGetInfo(dec, &info)) {
      uint32_t loop;
      for (loop = 0; loop < 2; ++loop) {
        uint8_t* buf;
        int timestamp;
        while (WebPAnimDecoderHasMoreFrames(dec)) {
          if (!WebPAnimDecoderGetNext(dec, &buf, &timestamp)) break;
        }
        WebPAnimDecoderReset(dec);
      }
      (void)WebPAnimDecoderGetDemuxer(dec);
    }
  }
  WebPAnimDecoderDelete(dec);
}

int main(int argc, const char* argv[]) {
  int i;
  for (i = 1; i < argc; ++i) {
    FILE* const f = fopen(argv[i], "rb");
    uint8_t* data;
    size_t size;
    int w = 0, h = 0;
    WebPBitstreamFeatures features;
    if (f == NULL) continue;
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    data = (uint8_t*)malloc(size > 0 ? size : 1);
    if (data == NULL || fread(data, 1, size, f) != size) {
      fclose(f);
      free(data);
      continue;
    }
    fclose(f);
    (void)WebPGetInfo(data, size, &w, &h);
    (void)WebPGetFeatures(data, size, &features);
    if ((long)w * h > (1 << 22)) w = h = 0;  // no gigabyte allocations here
    OneShot(data, size, w, h);
    Incremental(data, size, w, h);
    Modes(data, size, w, h);
    Demux(data, size);
    Animate(data, size, 0, 0);
    Animate(data, size, 1, 1);
    free(data);
  }
  (void)WebPGetDecoderVersion();
  (void)WebPGetDemuxVersion();
  return 0;
}
