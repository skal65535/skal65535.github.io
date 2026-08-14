#!/bin/bash
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.
# Decodes every file, compares the verdict with the 'expect' column, and for
# files that must decode, compares the decoded pixels against hashes.txt.
# Then the same file again through three more readers, since a decoder is
# never only one implementation:
#   -incremental   always, and it must agree with the one-shot decode on both
#                  the verdict and the pixels. Column 6 is the exception, for
#                  the file where the two are meant to differ.
#   anim_dump      column 4: dwebp refuses every animated file outright, so
#                  this is the only thing that reads one. The hash covers all
#                  of its frames at once. $ANIM_DUMP points at it.
#   webpinfo       column 5: walks the container without decoding, to
#                  stricter rules. $WEBPINFO points at it.
# A missing tool is reported and skipped rather than failing the run.
# Files tagged 'slow' allocate over a gigabyte; skip them with SKIP_SLOW=1.
DWEBP=${DWEBP:-$(command -v dwebp)}
ANIM_DUMP=${ANIM_DUMP:-$(command -v anim_dump)}
WEBPINFO=${WEBPINFO:-$(command -v webpinfo)}
if [ ! -x "$DWEBP" ]; then
  echo "set \$DWEBP to a dwebp binary, or put one on \$PATH" >&2
  exit 2
fi
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
# an exit status in the words expected.txt uses, and the hash it recorded
verdict() { if [ "$1" -eq 0 ]; then echo ok; else echo reject; fi; }
recorded() { [ -f hashes.txt ] && grep "^$1 " hashes.txt | awk '{print $2}'; }
fail=0; n=0; hashed=0; animated=0; skipped=0; walked=0; streamed=0
# 'rest' is deliberate: with one variable per column, adding a column makes
# the last one swallow it instead, silently. That cost an afternoon once.
while IFS='|' read -r name expect flag anim info incr rest; do
  [ -z "$name" ] && continue
  [ "$flag" = slow ] && [ -n "$SKIP_SLOW" ] && continue
  f=files/$name.webp
  msg=$($DWEBP -quiet "$f" -pam -o "$TMP/out.pam" 2>&1); got=$(verdict $?)
  n=$((n+1))
  if [ "$got" != "$expect" ]; then
    echo "MISMATCH $name: expected $expect, got $got  ${msg}"; fail=1; continue
  fi
  # decoded-pixel regression, for the files that decode (except the huge one).
  # One hash per file, from whichever decoder the row names.
  if [ "$got" = ok ] && [ -z "$anim" ] && [ "$flag" != slow ]; then
    want=$(recorded "$name")
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
  # the incremental decoder is a second implementation of the same bitstream,
  # and it must reach the same verdict and the same pixels. Where it does not
  # -- a RIFF size past the end of the file looks like data still arriving --
  # the case says so, and that disagreement is what gets pinned.
  if [ -z "$anim" ] && [ "$flag" != slow ]; then
    want=${incr:-$expect}
    msg=$($DWEBP -quiet -incremental "$f" -pam -o "$TMP/inc.pam" 2>&1)
    got=$(verdict $?)
    streamed=$((streamed+1))
    if [ "$got" != "$want" ]; then
      echo "MISMATCH $name: incremental expected $want, got $got  ${msg}"
      fail=1
    elif [ "$got" = ok ] && [ "$expect" = ok ] &&
         ! cmp -s "$TMP/out.pam" "$TMP/inc.pam"; then
      echo "INCREMENTAL PIXELS $name: differ from the one-shot decode"; fail=1
    fi
  fi
  # webpinfo walks the container without decoding anything, to stricter
  # rules than the demuxer's; where the two disagree the notes say so, and
  # this is what keeps that true
  if [ -n "$info" ] && [ -x "$WEBPINFO" ]; then
    msg=$($WEBPINFO -diag -quiet "$f" 2>&1); got=$(verdict $?)
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
  msg=$($ANIM_DUMP -folder "$TMP/f" -prefix f_ -pam "$f" 2>&1)
  got=$(verdict $?)
  animated=$((animated+1))
  if [ "$got" != "$anim" ]; then
    echo "MISMATCH $name: anim_dump expected $anim, got $got  ${msg}"; fail=1
  elif [ "$got" = ok ]; then
    want=$(recorded "$name")
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
       "$streamed decoded twice, $animated through the animation decoder," \
       "$walked walked by webpinfo)"
exit $fail
