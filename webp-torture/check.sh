#!/bin/bash
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.
# Decodes every file, compares the verdict with the 'expect' column, and for
# files that must decode, compares the decoded pixels against hashes.txt.
# A file with a fourth column is an animation: dwebp refuses every one of
# those outright, so anim_dump decodes it instead and the hash covers all of
# its frames at once. Point $ANIM_DUMP at one; without it that half is
# skipped, and said so. A fifth column is webpinfo's verdict on the same
# file, which is a second reader of the container and not always of the same
# opinion; $WEBPINFO points at it.
# Files tagged 'slow' allocate over a gigabyte; skip them with SKIP_SLOW=1.
DWEBP=${DWEBP:-$(command -v dwebp)}
ANIM_DUMP=${ANIM_DUMP:-$(command -v anim_dump)}
WEBPINFO=${WEBPINFO:-$(command -v webpinfo)}
if [ ! -x "$DWEBP" ]; then
  echo "set \$DWEBP to a dwebp binary, or put one on \$PATH" >&2
  exit 2
fi
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fail=0; n=0; hashed=0; animated=0; skipped=0; walked=0
while IFS='|' read -r name expect flag anim info; do
  [ -z "$name" ] && continue
  [ "$flag" = slow ] && [ -n "$SKIP_SLOW" ] && continue
  f=files/$name.webp
  msg=$($DWEBP -quiet "$f" -pam -o "$TMP/out.pam" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then got=ok; else got=reject; fi
  n=$((n+1))
  if [ "$got" != "$expect" ]; then
    echo "MISMATCH $name: expected $expect, got $got  ${msg}"; fail=1; continue
  fi
  # decoded-pixel regression, for the files that decode (except the huge one).
  # One hash per file, from whichever decoder the row names.
  if [ "$got" = ok ] && [ -z "$anim" ] && [ "$flag" != slow ] &&
     [ -f hashes.txt ]; then
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
  # webpinfo walks the container without decoding anything, to stricter
  # rules than the demuxer's; where the two disagree the notes say so, and
  # this is what keeps that true
  if [ -n "$info" ] && [ -x "$WEBPINFO" ]; then
    msg=$($WEBPINFO -diag -quiet "$f" 2>&1); rc=$?
    if [ $rc -eq 0 ]; then got=ok; else got=reject; fi
    walked=$((walked+1))
    if [ "$got" != "$info" ]; then
      echo "MISMATCH $name: webpinfo expected $info, got $got  ${msg}"; fail=1
    fi
  fi
  # the demuxer and the animation decoder, which is everything dwebp cannot
  # reach: it refuses an animated file before looking at a single frame
  [ -z "$anim" ] && continue
  if [ ! -x "$ANIM_DUMP" ]; then skipped=$((skipped+1)); continue; fi
  rm -rf "$TMP/f"; mkdir -p "$TMP/f"
  msg=$($ANIM_DUMP -folder "$TMP/f" -prefix f_ -pam "$f" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then got=ok; else got=reject; fi
  animated=$((animated+1))
  if [ "$got" != "$anim" ]; then
    echo "MISMATCH $name: anim_dump expected $anim, got $got  ${msg}"; fail=1
  elif [ "$got" = ok ] && [ -f hashes.txt ]; then
    want=$(grep "^$name " hashes.txt | awk '{print $2}')
    have=$(cat "$TMP"/f/f_*.pam | shasum -a 256 | awk '{print $1}')
    if [ -z "$want" ]; then
      echo "NOTE no hash recorded for $name"
    elif [ "$want" != "$have" ]; then
      echo "FRAMES CHANGED $name: $want -> $have"; fail=1
    else
      hashed=$((hashed+1))
    fi
  fi
done < expected.txt
[ $skipped -gt 0 ] &&
  echo "NOTE \$ANIM_DUMP is not set, so $skipped animations went unchecked"
[ $fail -eq 0 ] &&
  echo "all $n files behave as expected ($hashed pixel hashes matched," \
       "$animated through the animation decoder, $walked walked by webpinfo)"
exit $fail
