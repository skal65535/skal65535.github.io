#!/bin/bash
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.
# Rewrites hashes.txt: the SHA-256 of the decoded pixels of every file that
# is expected to decode. check.sh compares against it, so a silent change in
# decoder output fails even when the verdict does not move.
# One line per file, from whichever decoder its expected.txt row names: dwebp
# for a still, anim_dump for an animation, where the hash covers every frame.
# Run this only when the new output is known to be right.
DWEBP=${DWEBP:-$(command -v dwebp)}
ANIM_DUMP=${ANIM_DUMP:-$(command -v anim_dump)}
if [ ! -x "$DWEBP" ]; then
  echo "set \$DWEBP to a dwebp binary, or put one on \$PATH" >&2
  exit 2
fi
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
n=0
: > "$TMP/hashes"
while IFS='|' read -r name expect flag anim info; do
  [ -z "$name" ] && continue
  [ "$flag" = slow ] && continue          # too big to hash on every rebuild
  if [ -n "$anim" ]; then
    [ "$anim" != ok ] && continue
    if [ ! -x "$ANIM_DUMP" ]; then
      echo "set \$ANIM_DUMP: $name is an animation and dwebp cannot read" \
           "one" >&2
      exit 2
    fi
    rm -rf "$TMP/f"; mkdir -p "$TMP/f"
    if ! "$ANIM_DUMP" -folder "$TMP/f" -prefix f_ -pam \
         "$HERE/files/$name.webp" > /dev/null; then
      echo "FAILED to decode $name, which expected.txt says should decode" >&2
      exit 1
    fi
    sum=$(cat "$TMP"/f/f_*.pam | shasum -a 256 | awk '{print $1}')
  else
    [ "$expect" != ok ] && continue
    if ! "$DWEBP" -quiet "$HERE/files/$name.webp" -pam -o "$TMP/out.pam"; then
      echo "FAILED to decode $name, which expected.txt says should decode" >&2
      exit 1
    fi
    sum=$(shasum -a 256 < "$TMP/out.pam" | awk '{print $1}')
  fi
  echo "$name $sum" >> "$TMP/hashes"
  n=$((n+1))
done < "$HERE/expected.txt"
mv "$TMP/hashes" "$HERE/hashes.txt"
echo "wrote hashes.txt ($n files)"
