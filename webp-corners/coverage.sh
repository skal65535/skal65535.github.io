#!/bin/bash
# Copyright 2026 Skal (pascal.massimino@gmail.com). All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the COPYING file in the root of the source
# tree.
# How much of libwebp's decoder these files reach, measured rather than
# claimed. Builds an instrumented libwebp in a throwaway git worktree, runs
# the corpus through it three ways, and reports src/dec and src/demux:
#   corpus     the four readers check.sh uses, on every file
#   +options   the same files through every dwebp output and scaling knob,
#              one-shot and incremental
#   +api       plus src/api_sweep.c, which calls the entry points no command
#              line tool does -- the incremental decoder fed a few bytes at a
#              time, caller-allocated buffers, the colorspaces dwebp cannot
#              ask for, the demuxer's iterators
# The steps are cumulative, and the gap between the first and the last is
# the point: what a bitstream alone cannot reach is the caller's business,
# not the file's.
# Needs clang and llvm-profdata/llvm-cov. The source tree is never modified.
set -e
if [ -z "$LIBWEBP" ] || [ ! -d "$LIBWEBP/.git" ]; then
  echo "set \$LIBWEBP to a libwebp git checkout" >&2
  exit 2
fi
PROFDATA=${PROFDATA:-$(command -v llvm-profdata || xcrun --find llvm-profdata)}
COV=${COV:-$(command -v llvm-cov || xcrun --find llvm-cov)}
if [ ! -x "$PROFDATA" ] || [ ! -x "$COV" ]; then
  echo "no llvm-profdata/llvm-cov; set \$PROFDATA and \$COV" >&2
  exit 2
fi
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(mktemp -d)/wt
TMP=$(mktemp -d)
git -C "$LIBWEBP" worktree add -q --detach "$WT" HEAD
cleanup() {
  git -C "$LIBWEBP" worktree remove --force "$WT" >/dev/null 2>&1
  rm -rf "$TMP"
}
trap cleanup EXIT
mkdir -p "$TMP/prof"

FLAGS="-fprofile-instr-generate -fcoverage-mapping -O0 -g"
cmake -S "$WT" -B "$WT/build" -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate" \
  -DWEBP_BUILD_ANIM_UTILS=ON -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_GIF2WEBP=OFF \
  -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_EXTRAS=OFF \
  -DWEBP_BUILD_WEBPMUX=OFF >/dev/null
make -C "$WT/build" -j8 >/dev/null
B=$WT/build
clang $FLAGS -I"$WT/src" -I"$WT" "$HERE/src/api_sweep.c" -o "$TMP/api_sweep" \
  "$B/libwebpdemux.a" "$B/libwebp.a" -lm

# every file but the one that allocates a gigabyte: it says nothing new about
# which lines run, and costs minutes at -O0
awk -F'|' '$1 != "" && $3 != "slow" {print "'"$HERE"'/files/" $1 ".webp"}' \
  "$HERE/expected.txt" > "$TMP/files"

# 1. the corpus as check.sh runs it: four readers, verdicts and pixels checked
export LLVM_PROFILE_FILE="$TMP/prof/corpus-%p-%m.profraw"
cd "$HERE"
DWEBP=$B/dwebp ANIM_DUMP=$B/anim_dump WEBPINFO=$B/webpinfo \
  ./check.sh > "$TMP/check.log" || { cat "$TMP/check.log"; exit 1; }
tail -1 "$TMP/check.log"

# 2. the same files through dwebp's own knobs. Verdicts are not checked here
# -- check.sh just did that -- only which code runs.
export LLVM_PROFILE_FILE="$TMP/prof/options-%p-%m.profraw"
opts=(
  "-pam" "-ppm" "-bmp" "-tiff" "-pgm" "-yuv" "-alpha" "-pam -nofancy"
  "-pam -dither 100 -alpha_dither" "-pam -nodither" "-pam -nofilter"
  "-pam -resize 37 23" "-pam -resize 400 300" "-pam -crop 1 1 5 5"
  "-pam -mt" "-pam -flip" "-pam -noasm" "-ppm -noasm -nofancy"
  "-yuv -resize 37 23" "-pgm -resize 37 23" "-yuv -crop 1 1 5 5"
  "-alpha -resize 37 23" "-yuv -nofancy -mt"
)
n=0
while read -r f; do
  for o in "${opts[@]}"; do      # unquoted on purpose: one flag per word
    "$B/dwebp" -quiet $o "$f" -o "$TMP/out" >/dev/null 2>&1 || true
    "$B/dwebp" -quiet -incremental $o "$f" -o "$TMP/out" >/dev/null 2>&1 || true
    n=$((n + 2))
  done
done < "$TMP/files"
echo "$n decodes through ${#opts[@]} option sets, one-shot and incremental"

# 3. the API surface no tool reaches
export LLVM_PROFILE_FILE="$TMP/prof/api-%p-%m.profraw"
xargs -n 32 "$TMP/api_sweep" < "$TMP/files"
echo "every entry point, over the same files"

# One report per cumulative step, over the bitstream and container code. The
# other end of the decoder -- output formats, rescaling, buffer allocation --
# answers to the caller's config rather than to the file, so it is left out.
step() {  # the profiles of one step and every step before it, merged
  local out=$1
  shift
  # printf is a builtin, and the run leaves tens of thousands of profiles:
  # anything that has to be exec'd with them as arguments blows past ARG_MAX
  printf '%s\n' "$@" > "$TMP/list"
  "$PROFDATA" merge -sparse -f "$TMP/list" -o "$TMP/$out.profdata"
}
report() {
  "$COV" report "$B/dwebp" -object "$B/anim_dump" -object "$B/webpinfo" \
    -object "$TMP/api_sweep" -instr-profile="$TMP/$1.profdata" \
    "$WT"/src/dec/*.c "$WT"/src/demux/*.c 2>/dev/null
}
step corpus "$TMP"/prof/corpus-*
step options "$TMP"/prof/corpus-* "$TMP"/prof/options-*
step api "$TMP"/prof/*.profraw
echo
echo "src/dec + src/demux at $(git -C "$LIBWEBP" describe --tags --always):"
for s in corpus options api; do
  printf '  %-9s ' "$s"
  report $s | tail -1 |
    awk '{printf "regions %7s  functions %7s  lines %7s  branches %7s\n",
                 $4, $7, $10, $13}'
done
echo
echo "per file, the corpus on its own:"
report corpus |
  awk '$1 ~ /\.c$/ {printf "  %-22s regions %7s  lines %7s  branches %7s\n",
                           $1, $4, $10, $13}'

# README.md tabulates these nine, so they are a claim like any other here:
# say when they have moved rather than letting the page drift away from the
# measurement. The table lives in generate.py, which writes the README.
round() {  # a percentage column of the report, to the nearest whole number
  report "$1" | tail -1 | awk -v c="$2" '{printf "%.0f", $c}'
}
echo
stale=0
for s in corpus options api; do
  row="| $(round $s 4)% | $(round $s 10)% | $(round $s 13)% |"
  if ! grep -qF -- "$row" "$HERE/README.md"; then
    echo "README.md no longer has the $s row: $row"
    stale=1
  fi
done
if [ $stale = 0 ]; then
  echo "README.md still tabulates all three passes."
else
  echo "  ^ fix COVERAGE_RUNS in generate.py, then rerun it"
fi
