// project.js — A project folder: its store (store.js) and its manifest, project.json. The layout
// mirrors video2splat.py's OUT/ so the folder works with COLMAP and with serve.py:
//   project.json       format, name, dates, source video, one entry per phase (options, counts),
//                      the page's settings (app.js settings()), the scene's global transform
//                      (transform: 4x4 row-major, the world to the leveled one; math.js levelTransform)
//   images/            the keyframes, distorted, as the SfM saw them (written by the worker)
//   sharpness.csv      every sampled frame: sharpness, motion, keyframe status
//   sparse/0/          the distorted model of images/ (cameras.bin with the SfM camera, e.g. OPENCV)
//   gs/                the trainer's dataset: undistorted images/ + sparse/ (PINHOLE)
//   splats/            snapshots (.gosnap), .ply and .octree.json checkpoints
//   cache/             the SfM session (features.gosfm, model.gosfm), to resume repairs
// The files are the truth: the manifest only records what made them. It is written last after
// each change, so an interrupted write leaves the previous one.
"use strict";

const MANIFEST = 'project.json';
const FORMAT = 'gaussopt-project/1';

export class Project {
  constructor(store, manifest) {
    this.store = store;
    this.manifest = manifest;
    this.queue = Promise.resolve();  // writes of the manifest, in order
  }

  // The project in `store`: its manifest, or a new one (written at the first update)
  static async open(store) {
    const blob = await store.read(MANIFEST);
    let m = null;
    if (blob) {
      m = JSON.parse(await blob.text());
      if (m.format !== FORMAT) throw new Error(`${store.name}/${MANIFEST}: not a Gauss-Opt project (format ${m.format})`);
    }
    const now = new Date().toISOString();
    return new Project(store, m ?? { format: FORMAT, name: store.name, created: now, updated: now, source: null, phases: {} });
  }

  get name() { return this.manifest.name; }
  get where() { return this.store.where; }
  get isNew() { return !this.manifest.source && Object.keys(this.manifest.phases).length === 0; }

  // A file into the folder
  write(path, data) { return this.store.write(path, data); }
  read(path) { return this.store.read(path); }
  list(dir) { return this.store.list(dir); }

  // Merges `patch` ({ source?, phases?: { name: {...} }, settings?, transform? }) into the manifest and writes it
  update({ source, phases = {}, settings, transform } = {}) {
    const m = this.manifest;
    if (source !== undefined) m.source = source;
    if (settings !== undefined) m.settings = settings;
    if (transform !== undefined) m.transform = transform;
    for (const [k, v] of Object.entries(phases)) m.phases[k] = { ...m.phases[k], ...v, at: new Date().toISOString() };
    m.updated = new Date().toISOString();
    const text = JSON.stringify(m, null, 1);
    this.queue = this.queue.then(() => this.store.write(MANIFEST, text));
    return this.queue;
  }
}
