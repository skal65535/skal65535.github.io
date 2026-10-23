// store.js — Where a project folder lives, behind one interface (paths are relative, '/'-separated):
//   read(path) -> Blob | null, write(path, Blob | ArrayBuffer | string), list(dir) -> [{ name, kind }],
//   remove(path).
// DirStore: a folder the user picked (File System Access API, Chrome / Edge).
// ServerStore: serve.py's /project/<name>/<path> (GET / PUT / DELETE), any browser.
// MemoryStore: files in memory, the content of a project .zip (zip.js).
// A store is posted to a worker as describe()'s result and rebuilt there by storeFrom() (a
// MemoryStore as an empty one, whose files the worker posts back).
"use strict";

const parts = (path) => path.split('/').filter(Boolean);

export class DirStore {
  constructor(root) {
    this.kind = 'folder';
    this.root = root;
    this.name = root.name;
  }

  // Asks for write permission again (a handle restored after a reload has none); needs a click
  async permitted(ask = false) {
    const opts = { mode: 'readwrite' };
    if (await this.root.queryPermission?.(opts) === 'granted') return true;
    return ask && await this.root.requestPermission?.(opts) === 'granted';
  }

  get where() { return `folder ${this.name}/`; }

  async _dir(names, create) {
    let d = this.root;
    for (const n of names) d = await d.getDirectoryHandle(n, { create });
    return d;
  }

  async read(path) {
    const p = parts(path);
    try {
      return await (await (await this._dir(p.slice(0, -1), false)).getFileHandle(p.at(-1))).getFile();
    } catch (e) {
      if (e.name === 'NotFoundError' || e.name === 'TypeMismatchError') return null;
      throw e;
    }
  }

  async write(path, data) {
    const p = parts(path);
    const fh = await (await this._dir(p.slice(0, -1), true)).getFileHandle(p.at(-1), { create: true });
    const w = await fh.createWritable();  // written to a temporary file, swapped in on close
    try {
      await w.write(data);
      await w.close();
    } catch (e) {
      await w.abort().catch(() => {});
      throw e;
    }
  }

  async list(dir = '') {
    try {
      const out = [];
      for await (const [name, h] of (await this._dir(parts(dir), false)).entries()) out.push({ name, kind: h.kind });
      return out.sort((a, b) => a.name.localeCompare(b.name));
    } catch (e) {
      if (e.name === 'NotFoundError') return [];
      throw e;
    }
  }

  async remove(path) {
    const p = parts(path);
    try {
      await (await this._dir(p.slice(0, -1), false)).removeEntry(p.at(-1), { recursive: true });
    } catch (e) {
      if (e.name !== 'NotFoundError') throw e;
    }
  }

  describe() { return { type: 'dir', root: this.root }; }
}

export class ServerStore {
  constructor(name, base = '') {
    this.kind = 'server';
    this.name = name;
    this.base = `${base}/project/${encodeURIComponent(name)}`;
  }

  // Projects on serve.py (null: no project endpoint)
  static async serverProjects(base = '') {
    try {
      const r = await fetch(`${base}/projects.json`, { cache: 'no-store' });
      return r.ok ? (await r.json()).projects : null;
    } catch {
      return null;
    }
  }

  get where() { return `${this.name} (serve.py)`; }

  _url(path) { return `${this.base}/${parts(path).map(encodeURIComponent).join('/')}`; }

  // The URL of a file (e.g. for the trainer to fetch an image itself)
  urlOf(path) { return this._url(path); }

  async read(path) {
    const r = await fetch(`${this._url(path)}?optional`, { cache: 'no-store' });  // missing: 204, not a logged 404
    if (r.status === 204 || r.status === 404) return null;
    if (!r.ok) throw new Error(`GET ${path}: ${r.status} ${r.statusText}`);
    return r.blob();
  }

  async write(path, data) {
    const r = await fetch(this._url(path), { method: 'PUT', body: data });
    if (!r.ok) throw new Error(`PUT ${path}: ${await r.text().catch(() => r.statusText)}`);
  }

  async list(dir = '') {
    const r = await fetch(`${this._url(dir)}/?list`, { cache: 'no-store' });
    if (r.status === 404) return [];
    if (!r.ok) throw new Error(`list ${dir}: ${r.status} ${r.statusText}`);
    return (await r.json()).entries;
  }

  async remove(path) {
    const r = await fetch(this._url(path), { method: 'DELETE' });
    if (!r.ok && r.status !== 404) throw new Error(`DELETE ${path}: ${r.status} ${r.statusText}`);
  }

  describe() { return { type: 'server', name: this.name, base: this.base.slice(0, this.base.indexOf('/project/')) }; }
}

export class MemoryStore {
  constructor(name) {
    this.kind = 'memory';
    this.name = name;
    this.files = new Map();  // path -> Blob
  }

  get where() { return `${this.name}.zip`; }
  describe() { return { type: 'memory', name: this.name }; }

  async read(path) { return this.files.get(parts(path).join('/')) ?? null; }
  async write(path, data) { this.files.set(parts(path).join('/'), data instanceof Blob ? data : new Blob([data])); }
  async remove(path) { this.files.delete(parts(path).join('/')); }

  async list(dir = '') {
    const pre = parts(dir).join('/'), out = new Map();
    for (const p of this.files.keys()) {
      if (pre && !p.startsWith(`${pre}/`)) continue;
      const rest = parts(pre ? p.slice(pre.length + 1) : p);
      out.set(rest[0], rest.length > 1 ? 'directory' : 'file');
    }
    return [...out].map(([name, kind]) => ({ name, kind })).sort((a, b) => a.name.localeCompare(b.name));
  }
}

// The store a describe() result was made from (e.g. posted to a worker)
export function storeFrom(d) {
  if (!d) return null;
  if (d.type === 'memory') return new MemoryStore(d.name);
  return d.type === 'server' ? new ServerStore(d.name, d.base) : new DirStore(d.root);
}
