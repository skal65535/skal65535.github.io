// project_panel.js — The project folder controls of the Project phase (src/project/): choose a
// folder (File System Access API, Chrome / Edge; the last one is offered again after a reload),
// a project on serve.py (any browser, local), or download everything as a .zip. A new folder is
// filled with what the page has; an existing project is reopened: its SfM session (repairs work
// without the video), its gs/ dataset and its latest snapshot.
"use strict";

import { mkGui } from '../../../common/controls.js';
import { DirStore, ServerStore, MemoryStore } from '../project/store.js';
import { Project } from '../project/project.js';
import { zipBlob } from '../project/zip.js';
import { loadColmapFromMemory } from '../colmap.js';

const IDB = 'gaussopt', IDB_STORE = 'handles', RECENT = 'recent-project';

// The last project folder's handle, kept in IndexedDB (handles survive reloads, not their permission)
function idb(mode, fn) {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(IDB_STORE);
    req.onerror = () => reject(req.error);
    req.onsuccess = () => {
      const tx = req.result.transaction(IDB_STORE, mode), r = fn(tx.objectStore(IDB_STORE));
      tx.oncomplete = () => resolve(r?.result);
      tx.onerror = () => reject(tx.error);
    };
  });
}
const saveRecent = (handle) => idb('readwrite', s => s.put(handle, RECENT)).catch(() => {});
const loadRecent = () => idb('readonly', s => s.get(RECENT)).catch(() => null);

// Tooltips (mkGui tips: by key, or by folder name)
const TIPS = {
  'Project folder': 'a folder where everything the project makes is written: keyframes, cameras, the SfM session, settings, snapshots',
  where: 'the folder this project is saved to',
  pick: 'choose (or create) a folder on this computer for the project; it is reopened from there later',
  recent: 'reopen the folder used last time (the browser asks for access again)',
  'On serve.py': 'project folders kept by serve.py (--projects), for browsers without folder access',
  make: 'a new project folder on serve.py, named after the video',
  zip: 'everything of the project as one .zip',
};

export class ProjectPanel {
  constructor(app, host) {
    this.app = app;
    this.state = { where: 'none: results stay in this page (saves are downloads)', server: '' };
    this.serverProjects = null;
    const gui = mkGui(host, { draggable: false, tips: TIPS });
    const f = gui.addFolder('Project folder');
    f.add(this.state, 'where').name('saved to').listen().disable();
    this.pickBtn = f.add({ pick: () => this.pickFolder() }, 'pick').name('Choose a folder…');
    if (!window.showDirectoryPicker) this.pickBtn.name('Choose a folder… (Chrome / Edge)').disable();
    this.recentBtn = f.add({ recent: () => this.reopenRecent() }, 'recent').name('Reopen the last folder');
    this.recentBtn.disable();
    this.serverFolder = f.addFolder('On serve.py', { collapsed: true });
    this.zipBtn = f.add({ zip: () => this.downloadZip() }, 'zip').name('Download as .zip');
    this.gui = gui;
    this._initRecent();
    this.refreshServer();
  }

  async _initRecent() {
    this.recent = window.showDirectoryPicker ? await loadRecent() : null;
    if (this.recent) this.recentBtn.name(`Reopen ${this.recent.name}/`).enable();
  }

  // serve.py's projects (its /projects.json): open one, or a new one named after the video
  async refreshServer() {
    this.serverProjects = await ServerStore.serverProjects();
    const f = this.serverFolder;
    f.clear();
    if (!this.serverProjects) {
      f.add(this.state, 'server').name('(serve.py not reachable)').disable();
      return;
    }
    f.add({ make: () => this.newOnServer() }, 'make').name('New project on serve.py…');
    for (const name of this.serverProjects) f.add({ [name]: () => this.attach(new ServerStore(name)) }, name).name(`Open ${name}`).tip(`reopen the project ${name} (serve.py's projects directory)`);
  }

  async pickFolder() {
    try {
      const root = await window.showDirectoryPicker({ mode: 'readwrite', id: 'gaussopt-project' });
      await saveRecent(root);
      this.recent = root;
      this.recentBtn.name(`Reopen ${root.name}/`).enable();
      await this.attach(new DirStore(root));
    } catch (e) {
      if (e.name !== 'AbortError') this.app.showError(`Project folder: ${e.message}`);
    }
  }

  async reopenRecent() {
    const store = new DirStore(this.recent);
    if (!await store.permitted(true)) { this.app.log(`no permission to write ${this.recent.name}/`, 'warn'); return; }
    await this.attach(store);
  }

  async newOnServer() {
    const name = prompt('Project name on serve.py', this._defaultName());
    if (!name) return;
    await this.attach(new ServerStore(name.trim()));
    this.refreshServer();
  }

  _defaultName() { return (this.app.sfm?.videoName ?? 'project').replace(/\.[^.]+$/, '').replace(/[^\w.-]+/g, '_'); }

  // Make `store` the page's project: an empty one is filled with what the page has, an existing one
  // is reopened (its video's results, unless the page already works on another video)
  async attach(store) {
    const app = this.app;
    try {
      const project = await Project.open(store);
      const video = app.sfm?.videoName, src = project.manifest.source?.name;
      if (!project.isNew && video && src && src !== video) {
        throw new Error(`${store.name} holds the results of ${src}, not ${video}: choose another folder`);
      }
      app.sfm?.setStore(store.describe());  // first: a store the worker cannot take leaves nothing attached
      app.project = project;
      this.state.where = project.where;
      app._setSnapshotsEnabled(app.serverOk);
      if (project.isNew) await this.fill(project);
      else await this.resume(project);
      app.log(`Project: ${project.where}`, 'ok');
    } catch (e) {
      app.showError(`Project: ${e.message}`);
    }
  }

  // What the page has, into a new project: the SfM session and its results (worker), the trainer's state
  async fill(project) {
    const app = this.app;
    if (app.sfm?.videoName) await project.update({ source: app.sfm.sourceInfo(), ...app.sfm.keyframesRecord() });
    if (app.sfm) await app.sfm.flush();
    if (app.trainer?.stepCount > 0) await app.writeSnapshotTo(project);
    await project.update({ settings: app.settings() });
  }

  // An existing project: the SfM session for repairs (when the page has none), then its gs/
  // dataset into the trainer, from its latest snapshot when there is one
  async resume(project) {
    const app = this.app, m = project.manifest;
    app.applySettings(m.settings);  // first: e.g. val_split and downsample apply to the load
    if (!app.sfm?.quality && await project.read('cache/features.gosfm')) {
      await app.openVideoPanel(null);
      app.sfm.restoreFromStore();
    }
    if (!await project.read('gs/sparse/cameras.bin')) return;
    const sparse = {};
    for (const f of ['cameras.bin', 'images.bin', 'points3D.bin']) sparse[f] = await (await project.read(`gs/sparse/${f}`))?.arrayBuffer();
    const colmap = loadColmapFromMemory(sparse, [], { valSplit: app.params.val_split });
    for (const im of colmap.images) {  // a folder's files are cheap to open; serve.py's are fetched by the trainer
      if (project.store.urlOf) im.imageUrl = project.store.urlOf(`gs/images/${im.name}`);
      else im.fileBlob = await project.read(`gs/images/${im.name}`);
    }
    colmap.fromSfm = !!m.phases.cameras;  // cameras made by the SfM here (repairs can update them), or not
    const latest = m.phases.train?.latest;
    const snap = latest ? await project.read(latest) : null;
    await app.loadProjectModel(colmap, snap && await snap.arrayBuffer(), latest);
    app.applySettings(m.settings);  // over the snapshot's own (the settings may have changed since)
    if (app.sfm && m.phases.cameras) app.sfm.markSent(m.phases.cameras.version);
  }

  // The project's files as a .zip download: a project in memory, filled as a new folder would be
  async downloadZip() {
    const app = this.app, store = new MemoryStore(this._defaultName());
    const keep = app.project;
    try {
      app.log(`Packing ${store.name}.zip…`);
      const project = await Project.open(store);
      app.project = project;
      if (app.sfm) app.sfm.setStore(store.describe());
      if (app.sfm?.videoName) await project.update({ source: app.sfm.sourceInfo() });
      const files = app.sfm ? await app.sfm.flush() : [];
      for (const [path, blob] of files ?? []) await store.write(path, blob);
      if (app.trainer?.stepCount > 0) await app.writeSnapshotTo(project);
      await project.update();
      const zip = await zipBlob([...store.files]);
      const a = document.createElement('a');
      a.href = URL.createObjectURL(zip);
      a.download = `${store.name}.zip`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 60000);
      app.log(`Downloaded ${store.name}.zip (${store.files.size} files, ${(zip.size / 1e6).toFixed(0)} MB)`, 'ok');
    } catch (e) {
      app.showError(`Project .zip: ${e.message}`);
    } finally {
      app.project = keep;
      app.sfm?.setStore(keep?.store.describe() ?? null);
    }
  }
}
