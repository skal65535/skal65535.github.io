#!/bin/zsh
# publish.sh — The gallery as one self-contained folder: index.html (imports rewritten to ./common and
# ./gaussopt/src), common, stipple and gaussopt/src as committed (git's files, the working tree's copy; the page's
# modules among them), and the scenes; copied to DEST (rsync syntax). With it the app (the editor) at
# gaussopt/index.html: its page and its stylesheet, its ../common and ../stipple the gallery's own; no serve.py (the
# app finds none: no datasets served, no snapshots to the server).
#   zsh publish.sh SCENES_DIR DEST     e.g. zsh publish.sh ./scenes host:gallery
# SCENES_DIR: index.json and a folder per scene (scene.glb, photos.glb if any: README.md). Nothing else from
# SCENES_DIR is copied.
set -e
SCENES=${1:?SCENES_DIR}; DEST=${2:?DEST}
WEB=${0:A:h:h}
STAGE=$(mktemp -d -t gallery)
trap 'rm -rf $STAGE' EXIT  # the staged copy goes once copied, or when a step fails
mkdir -p $STAGE/scenes
sed -e "s#'\.\./common/#'./common/#g" -e "s#\"\.\./common/#\"./common/#g" -e "s#'\.\./gaussopt/src/#'./gaussopt/src/#g" $WEB/gallery/index.html > $STAGE/index.html
# the page's modules and the app, as committed
git -C $WEB ls-files common stipple gaussopt/index.html gaussopt/gaussopt.css gaussopt/src | while read f; do
  mkdir -p $STAGE/${f:h}; cp $WEB/$f $STAGE/$f
done
cp $SCENES/index.json $STAGE/scenes/
for id in $(node -e 'for (const s of JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"))) console.log(s.id)' $SCENES/index.json); do
  mkdir -p $STAGE/scenes/$id
  cp $SCENES/$id/scene.glb $STAGE/scenes/$id/
  if [[ -f $SCENES/$id/photos.glb ]]; then cp $SCENES/$id/photos.glb $STAGE/scenes/$id/; fi
done
echo "staged in $STAGE:"; (cd $STAGE && find . -type f | sort | sed 's/^/  /'); du -sh $STAGE
rsync -a $STAGE/ $DEST/  # no --delete: files already in DEST stay
echo "copied to $DEST"
