#!/bin/bash
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.
# Rewrites hashes.txt: the SHA-256 of the decoded pixels of every file that
# is expected to decode. check.sh compares against it, so a silent change in
# decoder output fails even when the verdict does not move.
# Run this only when the new output is known to be right.
DWEBP=${DWEBP:-$(command -v dwebp)}
if [ ! -x "$DWEBP" ]; then
  echo "set \$DWEBP to a dwebp binary, or put one on \$PATH" >&2
  exit 2
fi
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
n=0
: > "$TMP/hashes"
while IFS='|' read -r name expect flag; do
  [ -z "$name" ] && continue
  [ "$expect" != ok ] && continue
  [ "$flag" = slow ] && continue          # too big to hash on every rebuild
  if ! "$DWEBP" -quiet "$HERE/files/$name.webp" -pam -o "$TMP/out.pam"; then
    echo "FAILED to decode $name, which expected.txt says should decode" >&2
    exit 1
  fi
  echo "$name $(shasum -a 256 < "$TMP/out.pam" | awk '{print $1}')" \
    >> "$TMP/hashes"
  n=$((n+1))
done < "$HERE/expected.txt"
mv "$TMP/hashes" "$HERE/hashes.txt"
echo "wrote hashes.txt ($n files)"
