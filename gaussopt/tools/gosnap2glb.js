// gosnap2glb.js — Older snapshots (.gosnap) as scene .glb snapshots (snapshot.js gosnapToSceneGlb): FILE.gosnap gives
// FILE.scene.glb beside it, the same splats, octree, optimizer state, counters, settings and leveling; no cameras, points
// or background (a .gosnap has none: the page takes them from the dataset, as it did). The .gosnap is kept.
// --project DIR: every DIR/splats/*.gosnap, and DIR/project.json's latest snapshot then its .scene.glb.
//   deno run --allow-read --allow-write tools/gosnap2glb.js [--project DIR] [FILE.gosnap ...]
"use strict";

import { gosnapToSceneGlb, GOSNAP_EXT, SNAPSHOT_EXT } from '../src/snapshot.js';

const args = [...Deno.args], at = args.indexOf('--project'), project = at >= 0 ? args.splice(at, 2)[1] : null;
const files = [...args];
if (project) for (const e of Deno.readDirSync(`${project}/splats`)) if (e.isFile && e.name.endsWith(GOSNAP_EXT)) files.push(`${project}/splats/${e.name}`);
if (!files.length || args.some(a => !a.endsWith(GOSNAP_EXT))) {
  console.error('usage: gosnap2glb.js [--project DIR] [FILE.gosnap ...]');
  Deno.exit(2);
}

const glbName = (f) => f.slice(0, -GOSNAP_EXT.length) + SNAPSHOT_EXT;
for (const f of files.sort()) {
  const bytes = Deno.readFileSync(f), glb = gosnapToSceneGlb(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  Deno.writeFileSync(`${glbName(f)}.part`, glb);
  Deno.renameSync(`${glbName(f)}.part`, glbName(f));  // never half written
  console.log(`${f} (${(bytes.length / 1e6).toFixed(1)} MB) -> ${glbName(f)} (${(glb.length / 1e6).toFixed(1)} MB)`);
}
if (project) {
  const path = `${project}/project.json`, m = JSON.parse(Deno.readTextFileSync(path)), latest = m.phases?.train?.latest;
  if (latest?.endsWith(GOSNAP_EXT)) {
    m.phases.train.latest = glbName(latest);
    Deno.writeTextFileSync(`${path}.part`, JSON.stringify(m, null, 1));  // as project.js writes it
    Deno.renameSync(`${path}.part`, path);
    console.log(`${path}: latest ${latest} -> ${m.phases.train.latest}`);
  }
}
