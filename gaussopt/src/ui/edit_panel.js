// edit_panel.js — The editor's floater removal (edit.js): its controls (the 'Edit' folder) and its box over the view.
// With 'click picks a cluster' on, a click on the view picks the splat under it and grows its cluster: shown in magenta
// (trainer.js setSelected), its size in the box, then Delete or Cancel. 'Find floater clusters' lists every cluster apart from
// the scene, the most visible first: the view goes to each (‹ ›), Delete or skip. 'Find isolated floaters' selects at
// once the splats of the groups smaller than that (thousands of lone splats), one Delete for all. The training pauses
// while the box is open, and goes on when it closes. 'Undo delete' puts the last deleted splats back.
import { SplatGrid, pickSplat, growCluster, clusterInfo, maxClusterSize, EDIT_DEFAULTS } from '../edit.js';
import { orbitAnglesFromCenter, clamp } from '../math.js';

export const EDIT_TIPS = {
  edit_pick: 'a click on the view picks the splat under it and the cluster grown from it, shown in magenta, for deletion',
  edit_reach: 'splats touch when either one\'s centre is within this many sigmas of the other, in its shape: ' +
              'higher joins more into a cluster, lower cuts a floater off what it touches',
  edit_max: 'the largest a floater may be, as a share of the splats: a cluster growing past it touches the scene',
  edit_min: 'the fewest splats a listed cluster has (1: every lone splat apart from the scene too, thousands of them)',
  edit_find: 'every cluster apart from the scene (the largest cluster), still apart at twice the reach, the most visible from ' +
             'the nearest camera first: the view goes to each',
  edit_isolated: 'every group of fewer splats than the smallest cluster (lone splats mostly) apart from the scene, still ' +
                 'apart at twice the reach: all selected at once, one Delete for all',
  edit_undo: 'the splats of the last deletion put back',
};

export class EditPanel {
  constructor(app, gui) {
    this.app = app;
    this.opts = { edit_pick: false, edit_reach: EDIT_DEFAULTS.reach, edit_max: EDIT_DEFAULTS.maxShare, edit_min: EDIT_DEFAULTS.minSplats };
    this.sel = null;       // { indices, info, attached, seed } shown
    this.list = null;      // the floaters found: { clusters, at, total, largest }
    this.undo = null;      // trainer.removeSplats' rows, for 'undo delete'
    this.resume = false;   // the training was on when the box opened
    this.busy = false;
    this.gen = 0;          // Find floater clusters' jobs: a result of an older one dropped
    const f = gui.addFolder('Edit', { collapsed: true });
    f.add(this.opts, 'edit_pick').name('click picks a cluster');
    f.add(this.opts, 'edit_reach', 0.5, 6, 0.25).name('cluster reach').onChange(() => this._regrow());
    f.add(this.opts, 'edit_max', 0.001, 0.1, 0.001).name('largest floater (share)').onChange(() => this._regrow());
    f.add(this.opts, 'edit_min', 1, 50, 1).name('smallest cluster (splats)').onChange(() => this.list && this._regrow());
    f.add({ edit_find: () => this.findFloaters() }, 'edit_find').name('Find floater clusters');
    f.add({ edit_isolated: () => this.findFloaters('isolated') }, 'edit_isolated').name('Find isolated floaters');
    this.undoCtrl = f.add({ edit_undo: () => this.undoDelete() }, 'edit_undo').name('Undo delete').disable();
    this._makeBox();
  }

  get picking() { return this.opts.edit_pick; }

  _makeBox() {
    const $ = (id) => document.getElementById(id);
    this.box = $('edit-box');
    $('edit-close').onclick = () => this.close();
    $('edit-prev').onclick = () => this._step(-1);
    $('edit-next').onclick = () => this._step(1);
    $('edit-fly').onclick = () => this.sel?.info && this.flyTo(this.sel.info);
    $('edit-delete').onclick = () => this.deleteSelection();
    this.$ = $;
  }

  _show(title, info, { canDelete = true } = {}) {
    this.$('edit-title').textContent = title;
    this.$('edit-info').textContent = info;
    const list = !!this.list?.clusters.length;
    this.$('edit-prev').hidden = this.$('edit-next').hidden = !list;
    this.$('edit-delete').disabled = !canDelete;
    this.$('edit-fly').disabled = !this.sel?.info || !!this.sel.isolated;
    this.box.style.display = 'block';
    this.app.drawGen++;
  }

  _pause() {
    const P = this.app.params;
    if (this.box.style.display !== 'block') this.resume = !!P.training;
    if (P.training) this.app.setTraining(false);
  }

  // The box closed, the selection cleared, the training on again if it was
  close() {
    this.gen++;  // a Find floater clusters running: its result dropped
    this._unselect();
    this.list = null;
    this.box.style.display = 'none';
    if (this.resume) this.app.setTraining(true);
    this.resume = false;
    this.app.drawGen++;
  }

  _unselect() {
    if (this.sel && !this.sel.attached) this.app.trainer?.setSelected(this.sel.indices, false);
    this.sel = null;
  }

  _select(indices, params, seed = -1) {
    this._unselect();
    const info = clusterInfo(params, indices);
    this.sel = { indices, info, seed, attached: false };
    this.app.trainer.setSelected(indices, true);
    return info;
  }

  _grid(params, count) {
    return new SplatGrid(params, count, { reach: this.opts.edit_reach, scaleMul: this.app.params.haze });
  }

  _describe(info) {
    const app = this.app, R = app.octree?.sceneRadius || 1;
    let near = Infinity;
    for (const im of app.colmap?.images ?? []) near = Math.min(near, Math.hypot(...[0, 1, 2].map(k => im.center[k] - info.center[k])));
    return `${info.indices.length} splats, ${(info.extent / R).toFixed(3)} x the scene radius across` +
           (isFinite(near) ? `, ${(near / R).toFixed(3)} from the nearest camera` : '');
  }

  // A click at canvas pixel (px, py): the cluster under it
  async clickAt(px, py) {
    const app = this.app, tr = app.trainer;
    if (!tr || !app.lastView || this.busy) return;
    this.busy = true;
    try {
      this._pause();
      const params = await tr.readSplats(), n = tr.count;
      const hit = pickSplat(params, n, { ...app.lastView, W: app.canvas.width, H: app.canvas.height }, px, py, { scaleMul: app.params.haze });
      this.list = null;
      if (!hit) { this._unselect(); this._show('No splat there', 'nothing drawn at that pixel', { canDelete: false }); return; }
      this._growFrom(hit.index, params, n);
    } finally { this.busy = false; }
  }

  _growFrom(seed, params, n) {
    const maxSize = maxClusterSize(n, { maxShare: this.opts.edit_max });
    const c = growCluster(this._grid(params, n), seed, maxSize);
    if (c.attached) {
      this._unselect();
      this.sel = { indices: c.indices, info: null, seed, attached: true };
      this._show('Attached to the scene', `the cluster grows past ${maxSize} splats: lower the cluster reach to cut it off`, { canDelete: false });
      return;
    }
    const info = this._select(c.indices, params, seed);
    this._show('Cluster', this._describe(info));
  }

  // The reach or the largest size changed: the clicked cluster grown again, or the floaters found again
  async _regrow() {
    if (this.busy || !this.app.trainer) return;
    if (this.list) return this.findFloaters();
    if (this.sel?.isolated) return this.findFloaters('isolated');
    if (this.sel?.seed >= 0) {
      this.busy = true;
      try {
        const tr = this.app.trainer, params = await tr.readSplats();
        this._growFrom(this.sel.seed, params, tr.count);
      } finally { this.busy = false; }
    }
  }

  // The floaters found off the page's thread (edit_worker.js; the page drawn meanwhile), the training paused: the
  // splats' indices hold. Closed meanwhile (or another scene): the result dropped. kind 'clusters': listed, one at a
  // time; 'isolated': the splats of the smaller groups, all selected.
  async findFloaters(kind = 'clusters') {
    const app = this.app, tr = app.trainer;
    if (!tr || this.busy) return;
    this.busy = true;
    try {
      this._pause();
      this._unselect();
      this.list = null;
      const gen = ++this.gen, t0 = performance.now(), params = await tr.readSplats(), n = tr.count;
      this._show(kind === 'isolated' ? 'Finding isolated floaters…' : 'Finding floater clusters…', `${n} splats, the page free meanwhile`, { canDelete: false });
      const res = await this._findInWorker({
        params: params.slice(), count: n, reach: this.opts.edit_reach, isolation: EDIT_DEFAULTS.isolation,
        scaleMul: app.params.haze, maxSize: maxClusterSize(n, { maxShare: this.opts.edit_max }), minSplats: this.opts.edit_min,
        cameras: (app.colmap?.images ?? []).map(im => [...im.center]), limit: 200, small: kind === 'isolated',
      });
      if (gen !== this.gen || tr !== app.trainer) return;
      if (kind === 'isolated') return this._showIsolated(res, params, t0);
      this.list = { ...res, at: 0 };
      app.log(`Floater clusters: ${res.total} of ${this.opts.edit_min} splats or more apart from the scene (${res.largest} splats; ${res.groups} clusters in all, ` +
              `${res.joined} more joining it at ${EDIT_DEFAULTS.isolation} x the reach) in ` +
              `${((performance.now() - t0) / 1000).toFixed(1)} s${res.total > res.clusters.length ? `, the ${res.clusters.length} most visible listed` : ''}`, 'info', 'edit');
      if (!res.clusters.length) { this._show('No floater clusters', `none of ${this.opts.edit_min} splats or more apart from the scene at this reach`, { canDelete: false }); return; }
      this._showListed();
    } catch (e) {
      app.log(`Find floater clusters failed: ${e.message}`, 'error', 'edit');
      if (this.box.style.display === 'block') this._show('Find floater clusters failed', e.message, { canDelete: false });
    } finally { this.busy = false; }
  }

  // One job of edit_worker.js (made once, kept): its result, or its error thrown
  _findInWorker(job) {
    if (!this.worker) {
      this.worker = new Worker(new URL('../edit_worker.js', import.meta.url), { type: 'module' });
      this.jobs = new Map();
      this.worker.onmessage = ({ data }) => {
        const j = this.jobs.get(data.id);
        this.jobs.delete(data.id);
        if (data.error) j?.reject(new Error(data.error)); else j?.resolve(data);
      };
      this.worker.onerror = (e) => {  // the worker itself failed (its module did not load): every job with it
        for (const j of this.jobs.values()) j.reject(new Error(e.message || 'the worker failed'));
        this.jobs.clear();
        this.worker = null;
      };
    }
    const id = (this.jobId = (this.jobId ?? 0) + 1);
    return new Promise((resolve, reject) => {
      this.jobs.set(id, { resolve, reject });
      this.worker.postMessage({ ...job, id }, [job.params.buffer]);
    });
  }

  // The isolated floaters, all selected (no view moved: they are everywhere)
  _showIsolated(res, params, t0) {
    const app = this.app, min = this.opts.edit_min, n = res.small.length;
    app.log(`Isolated floaters: ${n} splats in ${res.smallGroups} groups of fewer than ${min} apart from the scene in ` +
            `${((performance.now() - t0) / 1000).toFixed(1)} s`, 'info', 'edit');
    if (!n) { this._show('No isolated floaters', `no group of fewer than ${min} splats apart from the scene at this reach`, { canDelete: false }); return; }
    this._select(res.small, params);
    this.sel.isolated = true;
    this._show('Isolated floaters', `${n} splats in ${res.smallGroups} groups of fewer than ${min}, apart from the scene`);
  }

  _showListed(params = this.app.trainer.cpuParams) {
    const L = this.list, c = L.clusters[L.at];
    const info = this._select(c.indices, params);
    this._show(`Floater cluster ${L.at + 1} / ${L.clusters.length}`, this._describe(info));
    this.flyTo(info);
  }

  _step(d) {
    const L = this.list;
    if (!L?.clusters.length || this.busy) return;
    L.at = (L.at + d + L.clusters.length) % L.clusters.length;
    this._showListed();
  }

  // The view on the cluster: from the side of the nearest camera, its extent a third of the view or so
  flyTo(info) {
    const app = this.app, P = app.params, R = app.octree?.sceneRadius || 1;
    if (!info) return;
    let cam = null, best = Infinity;
    for (const im of app.colmap?.images ?? []) {
      const d = Math.hypot(...[0, 1, 2].map(k => im.center[k] - info.center[k]));
      if (d < best) { best = d; cam = im.center; }
    }
    const dir = cam ? [0, 1, 2].map(k => info.center[k] - cam[k]) : [0, 0, 1];
    const len = Math.hypot(...dir) || 1;
    const dist = clamp(3 * info.extent / Math.tan(P.fov * Math.PI / 360), 0.02 * R, cam ? Math.max(0.02 * R, best) : R);
    const eye = [0, 1, 2].map(k => info.center[k] - dir[k] / len * dist);
    const orb = orbitAnglesFromCenter(info.center, eye, app.worldUp);
    P.target = [...info.center];
    P.radius = orb.radius;
    P.phi = clamp(orb.phi, -89, 89);
    P.theta = orb.theta;
    if (P.camera_mode === 'demo') P.camera_mode = 'drone';
    app.flyAnim = null;
    P.snap_to_colmap = false;
    app.droneLive = false;  // the orbit's pose, the drone from it at the next input
    app.drawGen++;
  }

  async deleteSelection() {
    const app = this.app, tr = app.trainer, sel = this.sel;
    if (!tr || !sel || sel.attached || this.busy) return;
    this.busy = true;
    try {
      const n = sel.indices.length;
      const { removed, newIndex } = await tr.removeSplats(sel.indices);
      this.sel = null;
      this.undo = removed;
      this.undoCtrl.enable();
      app.editedAt(tr.stepCount);
      app.log(`Deleted ${n} splats (${tr.count} left)`, 'ok', 'edit');
      const L = this.list;
      if (!L) { this.close(); return; }
      // the other floaters' splats where they are now
      L.clusters.splice(L.at, 1);
      for (const c of L.clusters) c.indices = Int32Array.from(c.indices, i => newIndex[i]).filter(i => i >= 0).sort();
      L.clusters = L.clusters.filter(c => c.indices.length);
      if (!L.clusters.length) { this.list = null; this._show('No floater clusters left', `${n} splats deleted`, { canDelete: false }); return; }
      L.at = Math.min(L.at, L.clusters.length - 1);
      this._showListed();
    } finally { this.busy = false; }
  }

  async undoDelete() {
    const app = this.app, tr = app.trainer;
    if (!tr || !this.undo || this.busy) return;
    this.busy = true;
    try {
      this._unselect();
      this.list = null;
      const n = this.undo.count;
      await tr.restoreSplats(this.undo);
      this.undo = null;
      this.undoCtrl.disable();
      app.editedAt(tr.stepCount);
      app.log(`Put back ${n} splats (${tr.count} now)`, 'ok', 'edit');
      if (this.box.style.display === 'block') this.close();
      app.drawGen++;
    } finally { this.busy = false; }
  }

  // A new trainer (another scene): nothing selected, nothing to undo
  reset() {
    this.gen++;
    this.sel = null; this.list = null; this.undo = null;
    this.undoCtrl.disable();
    this.box.style.display = 'none';
    this.resume = false;
  }
}
