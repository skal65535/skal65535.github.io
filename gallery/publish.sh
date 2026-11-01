#!/bin/zsh
# publish.sh — The gallery as one self-contained folder: index.html (imports rewritten to ./common and
# ./gaussopt/src), the modules it imports (their closure), and the scenes; copied to DEST (rsync syntax). With it the
# app (the editor) at gaussopt/index.html: its page, its stylesheet, gaussopt/src, common and stipple as committed
# (git's files, the working tree's copy), its ../common and ../stipple the gallery's own; no serve.py (the app finds
# none: no datasets served, no snapshots to the server).
#   zsh publish.sh SCENES_DIR DEST     e.g. zsh publish.sh ./scenes host:gallery
# SCENES_DIR: index.json and a folder per scene (scene.glb, photos.glb if any: README.md). Nothing else from
# SCENES_DIR is copied.
set -e
SCENES=${1:?SCENES_DIR}; DEST=${2:?DEST}
WEB=${0:A:h:h}
STAGE=$(mktemp -d -t gallery)
mkdir -p $STAGE/scenes
sed -e "s#'\.\./common/#'./common/#g" -e "s#\"\.\./common/#\"./common/#g" -e "s#'\.\./gaussopt/src/#'./gaussopt/src/#g" $WEB/gallery/index.html > $STAGE/index.html
# the import closure of the page's modules, from gaussopt/web
node -e '
const fs = require("fs"), path = require("path"), web = process.argv[1], seen = new Set();
const page = fs.readFileSync(path.join(web, "gallery/index.html"), "utf8");
const visit = (rel) => {
  if (seen.has(rel)) return;
  seen.add(rel);
  if (!rel.endsWith(".js")) return;
  const src = fs.readFileSync(path.join(web, rel), "utf8");
  for (const m of src.matchAll(/(?:from|import)\s*\(?\s*["\x27](\.[^"\x27]+)["\x27]/g)) visit(path.normalize(path.join(path.dirname(rel), m[1])));
};
for (const m of page.matchAll(/(?:from|src=|href=)\s*["\x27]\.\.\/((?:common|gaussopt)\/[^"\x27]+)["\x27]/g)) visit(m[1]);
console.log([...seen].join("\n"));
' $WEB > $STAGE/files.txt
while read f; do mkdir -p $STAGE/${f:h}; cp $WEB/$f $STAGE/$f; done < $STAGE/files.txt
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
rsync -a --exclude files.txt $STAGE/ $DEST/  # no --delete: files already in DEST stay
echo "copied to $DEST"
