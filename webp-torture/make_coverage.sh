#!/bin/bash
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.
# Regenerates coverage.txt: builds an instrumented dwebp in a throwaway git
# worktree, runs every file through it, and records the probes each one hits.
# The source tree is never modified.
set -e
if [ -z "$LIBWEBP" ] || [ ! -d "$LIBWEBP/.git" ]; then
  echo "set \$LIBWEBP to a libwebp git checkout" >&2
  exit 2
fi
WT=$(mktemp -d)/wt
HERE=$(cd "$(dirname "$0")" && pwd)

git -C "$LIBWEBP" worktree add -q --detach "$WT" HEAD
cleanup() { git -C "$LIBWEBP" worktree remove --force "$WT" >/dev/null 2>&1; }
trap cleanup EXIT

python3 "$HERE/src/probes.py" "$WT"
cmake -S "$WT" -B "$WT/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
make -C "$WT/build" -j8 dwebp >/dev/null

# the notes point at lines in this same checkout
LIBWEBP="$LIBWEBP" python3 "$HERE/src/check_refs.py" || \
  echo "  ^ fix the notes, then: LIBWEBP=... ./src/check_refs.py --write"

REV=$(git -C "$LIBWEBP" rev-parse --short HEAD)
{
  echo "# Decoder paths each file actually reaches."
  echo "#"
  echo "# Regenerate with ./make_coverage.sh, which applies src/probes.py to a"
  echo "# throwaway worktree. Every probe sits on the line a README note"
  echo "# refers to, so this is the evidence behind those notes: a file can"
  echo "# reject for the wrong reason, and several did before being fixed."
  echo "#"
  echo "# libwebp revision: $REV"
  echo
  for f in "$HERE"/files/*.webp; do
    n=$(basename "$f" .webp)
    tags=$("$WT/build/dwebp" -quiet "$f" -o /dev/null 2>&1 |
             grep '^PROBE:' | sed 's/PROBE://' | sort -u | tr '\n' ' ')
    printf "%-40s %s\n" "$n" "$tags"
  done
} > "$HERE/coverage.txt"
echo "wrote coverage.txt ($(grep -c . "$HERE/coverage.txt") lines)"
