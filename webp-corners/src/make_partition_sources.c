// Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree.
// -----------------------------------------------------------------------------
// Regenerates sources/lossy-{1,2,4,8}-partitions.webp.
//
// config.partitions is API-only -- cwebp does not expose it -- and libwebp
// forces num_parts back to 1 whenever the token path is used, so low_memory
// is set here to keep the requested partition count (webp_enc.c:124).
//
// Build against a libwebp checkout, e.g.
//   cc -I$LIBWEBP -o mkparts make_partition_sources.c \
//      $LIBWEBP/build/libwebp.a $LIBWEBP/build/libsharpyuv.a -lm
//   ./mkparts input.ppm sources

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/webp/encode.h"

#define WIDTH 128
#define HEIGHT 128

// Reads a binary PPM of exactly WIDTHxHEIGHT, skipping its three header lines.
static int ReadPPM(const char* path, uint8_t* rgb, size_t size) {
  FILE* const f = fopen(path, "rb");
  int c, newlines = 0;
  if (f == NULL) {
    fprintf(stderr, "cannot open %s\n", path);
    return 0;
  }
  while (newlines < 3 && (c = fgetc(f)) != EOF) {
    if (c == '\n') ++newlines;
  }
  if (fread(rgb, 1, size, f) != size) {
    fprintf(stderr, "%s is not %dx%d\n", path, WIDTH, HEIGHT);
    fclose(f);
    return 0;
  }
  fclose(f);
  return 1;
}

static int WriteFile(const char* path, const uint8_t* data, size_t size) {
  FILE* const f = fopen(path, "wb");
  const int ok = (f != NULL) && (fwrite(data, 1, size, f) == size);
  if (f != NULL) fclose(f);
  if (!ok) fprintf(stderr, "cannot write %s\n", path);
  return ok;
}

// Encodes 'rgb' with 1 << log2_parts token partitions into 'outdir'.
static int EncodeWithPartitions(const uint8_t* rgb, int log2_parts,
                                const char* outdir) {
  WebPConfig config;
  WebPPicture pic;
  WebPMemoryWriter writer;
  char path[512];
  int ok;

  if (!WebPConfigInit(&config) || !WebPPictureInit(&pic)) return 0;
  config.partitions = log2_parts;
  config.low_memory = 1;  // otherwise num_parts is forced back to 1
  config.method = 4;
  config.quality = 80;
  if (!WebPValidateConfig(&config)) return 0;

  pic.width = WIDTH;
  pic.height = HEIGHT;
  if (!WebPPictureImportRGB(&pic, rgb, WIDTH * 3)) return 0;
  WebPMemoryWriterInit(&writer);
  pic.writer = WebPMemoryWrite;
  pic.custom_ptr = &writer;

  ok = WebPEncode(&config, &pic);
  if (!ok) fprintf(stderr, "encode failed, error %d\n", pic.error_code);
  if (ok) {
    snprintf(path, sizeof(path), "%s/lossy-%d-partitions.webp", outdir,
             1 << log2_parts);
    ok = WriteFile(path, writer.mem, writer.size);
    if (ok) printf("%-36s %6zu bytes\n", path, writer.size);
  }
  WebPMemoryWriterClear(&writer);
  WebPPictureFree(&pic);
  return ok;
}

int main(int argc, char** argv) {
  static uint8_t rgb[WIDTH * HEIGHT * 3];
  int log2_parts;
  if (argc != 3) {
    fprintf(stderr, "usage: %s <%dx%d.ppm> <outdir>\n", argv[0], WIDTH, HEIGHT);
    return 1;
  }
  if (!ReadPPM(argv[1], rgb, sizeof(rgb))) return 1;
  for (log2_parts = 0; log2_parts <= 3; ++log2_parts) {
    if (!EncodeWithPartitions(rgb, log2_parts, argv[2])) return 1;
  }
  return 0;
}
