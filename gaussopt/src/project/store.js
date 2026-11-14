// store.js — Where a project folder lives, behind one interface (paths are relative, '/'-separated):
//   read(path) -> Blob | null, write(path, Blob | ArrayBuffer | string | [pieces]), list(dir) -> [{ name, kind }].
// [pieces]: a file's content in order (Uint8Array | Blob, project.js packPieces), written without making it one Blob.
// DirStore: a folder the user picked (File System Access API, Chrome / Edge).
// ServerStore: serve.py's /project/<name>/<path> (GET / PUT), any browser.
// MemoryStore: files in memory, the content of a project .zip (zip.js).
// A store is posted to a worker as describe()'s result and rebuilt there by storeFrom() (a
// MemoryStore as an empty one, whose files the worker posts back).
"use strict";

const parts = (path) => path.split('/').filter(Boolean);
const PIECE = 32 * 2 ** 20;  // ServerStore: the most bytes a request
const sizeOf = (pieces) => pieces.reduce((s, p) => s + (p.size ?? p.byteLength), 0);

export class DirStore {
  constructor(root) {
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
      for (const piece of Array.isArray(data) ? data : [data]) await w.write(piece);
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

  describe() { return { type: 'dir', root: this.root }; }
}

export class ServerStore {
  constructor(name, base = '') {
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

  // the body a Blob: an ArrayBuffer's bytes go into the browser's network events whole (a driving Playwright of
  // 150 MB of them gets a message too long for a string and stops), a Blob's do not. Pieces more than PIECE bytes in
  // all: PIECE at a time (serve.py ?append=), each request's bytes copied into one Uint8Array of PIECE at most, none
  // of the whole: Blobs of them add up in Chrome's blob storage, the pieces sent not freed in time
  // (ERR_BLOB_OUT_OF_MEMORY past ~500 MB), and PIECE bytes in the network events are fine
  async write(path, data) {
    if (Array.isArray(data) && sizeOf(data) > PIECE) return this._writePieces(path, data);
    await this._put(this._url(path), data instanceof Blob ? data : new Blob(Array.isArray(data) ? data : [data]), path);
  }

  async _put(url, body, path) {
    const r = await fetch(url, { method: 'PUT', body });
    if (!r.ok) throw new Error(`PUT ${path}: ${await r.text().catch(() => r.statusText)}`);
  }

  async _writePieces(path, pieces) {
    const total = sizeOf(pieces);
    let batch = [], size = 0, at = 0;
    const send = async () => {
      const body = new Uint8Array(size);
      let o = 0;
      for (const b of batch) {
        const u = b instanceof Blob ? new Uint8Array(await b.arrayBuffer()) : new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
        body.set(u, o); o += u.length;
      }
      await this._put(`${this._url(path)}?append=${at}${at + size === total ? '&last=1' : ''}`, body, path);
      at += size; batch = []; size = 0;
    };
    for (const p of pieces) {
      const n = p.size ?? p.byteLength;
      for (let k = 0; k < n;) {  // a piece of more than PIECE: in parts of it
        const m = Math.min(n - k, PIECE - size);
        batch.push(k === 0 && m === n ? p : p instanceof Blob ? p.slice(k, k + m) : p.subarray(k, k + m));
        size += m; k += m;
        if (size === PIECE) await send();
      }
    }
    if (size) await send();
  }

  async list(dir = '') {
    const r = await fetch(`${this._url(dir)}/?list`, { cache: 'no-store' });
    if (r.status === 404) return [];
    if (!r.ok) throw new Error(`list ${dir}: ${r.status} ${r.statusText}`);
    return (await r.json()).entries;
  }

  describe() { return { type: 'server', name: this.name, base: this.base.slice(0, this.base.indexOf('/project/')) }; }
}

export class MemoryStore {
  constructor(name) {
    this.name = name;
    this.files = new Map();  // path -> Blob
  }

  get where() { return `${this.name}.zip`; }
  describe() { return { type: 'memory', name: this.name }; }

  async read(path) { return this.files.get(parts(path).join('/')) ?? null; }
  async write(path, data) { this.files.set(parts(path).join('/'), data instanceof Blob ? data : new Blob(Array.isArray(data) ? data : [data])); }

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
