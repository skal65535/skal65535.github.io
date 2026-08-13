#!/bin/bash
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.
# Decodes every file, compares the verdict with the 'expect' column, and for
# files that must decode, compares the decoded pixels against hashes.txt.
# Files tagged 'slow' allocate over a gigabyte; skip them with SKIP_SLOW=1.
DWEBP=${DWEBP:-$(command -v dwebp)}
if [ ! -x "$DWEBP" ]; then
  echo "set \$DWEBP to a dwebp binary, or put one on \$PATH" >&2
  exit 2
fi
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fail=0; n=0; hashed=0
while IFS='|' read -r name expect flag; do
  [ -z "$name" ] && continue
  [ "$flag" = slow ] && [ -n "$SKIP_SLOW" ] && continue
  f=files/$name.webp
  msg=$($DWEBP -quiet "$f" -pam -o "$TMP/out.pam" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then got=ok; else got=reject; fi
  n=$((n+1))
  if [ "$got" != "$expect" ]; then
    echo "MISMATCH $name: expected $expect, got $got  ${msg}"; fail=1; continue
  fi
  # decoded-pixel regression, for the files that decode (except the huge one)
  if [ "$got" = ok ] && [ "$flag" != slow ] && [ -f hashes.txt ]; then
    want=$(grep "^$name " hashes.txt | awk '{print $2}')
    have=$(shasum -a 256 < "$TMP/out.pam" | awk '{print $1}')
    if [ -n "$want" ]; then
      hashed=$((hashed+1))
      if [ "$want" != "$have" ]; then
        echo "PIXELS CHANGED $name: $want -> $have"; fail=1
      fi
    else
      echo "NOTE no hash recorded for $name"
    fi
  fi
done < expected.txt
[ $fail -eq 0 ] &&
  echo "all $n files behave as expected ($hashed pixel hashes matched)"
exit $fail
