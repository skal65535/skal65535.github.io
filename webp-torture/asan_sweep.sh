#!/bin/bash
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.
# Decodes every file in 13 output/scaling modes. Point ASAN_DWEBP at a
# sanitizer build. The 'slow' file allocates ~1.8GB, so it is decoded once
# rather than in every mode.
D=${ASAN_DWEBP:-$(command -v dwebp)}
if [ ! -x "$D" ]; then
  echo "set \$ASAN_DWEBP to a sanitizer-built dwebp" >&2
  exit 2
fi
export ASAN_OPTIONS=${ASAN_OPTIONS:-detect_leaks=0}
slow=$(awk -F'|' '$3=="slow"{print $1}' expected.txt)
bad=0; n=0
while read -r opt; do
  for f in files/*.webp; do
    skip=
    for s in $slow; do [ "$(basename "$f" .webp)" = "$s" ] && skip=1; done
    [ -n "$skip" ] && continue
    out=$($D -quiet $opt "$f" -o /dev/null 2>&1); n=$((n+1))
    echo "$out" | grep -q AddressSanitizer && {
      echo "ASAN $(basename "$f") [$opt]"; echo "$out" | head -3; bad=1; }
  done
done <<'OPTS'
-pam
-ppm
-pgm
-bmp
-alpha
-incremental -pam
-incremental -mt -pam
-mt -pam
-nofancy -pgm
-crop 0 0 1 1 -pam
-resize 5 7 -pam
-incremental -resize 3 3 -pam
-dither 100 -alpha_dither -pam
OPTS
for s in $slow; do
  out=$($D -quiet -pam "files/$s.webp" -o /dev/null 2>&1); n=$((n+1))
  echo "$out" | grep -q AddressSanitizer && { echo "ASAN $s"; bad=1; }
done
echo "asan sweep: $n decodes, failures=$bad"
exit $bad
