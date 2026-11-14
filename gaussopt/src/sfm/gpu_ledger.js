// gpu_ledger.js — The WebGPU buffers and textures a realm holds (a debug option: ?sfm_gpuLedger, the SfM worker's): bytes
// made, less those destroyed or collected, by the site that made them. Every createBuffer / createTexture from then on
// counted; the line every everyMs in the log (0: none, line() only).
"use strict";

// bytes a texel: the channels (r, rg, rgba) times the channel's bytes (8, 16, 32 bits); another format: 4
const texelBytes = (format) => {
  const m = /^(rgba|rg|r)(8|16|32)/.exec(format);
  return m ? m[1].length * Number(m[2]) / 8 : 4;
};
const sizeOfTexture = (d) => {
  const [w, h = 1, l = 1] = Array.isArray(d.size) ? d.size : [d.size.width, d.size.height ?? 1, d.size.depthOrArrayLayers ?? 1];
  return w * h * l * texelBytes(d.format) * (d.mipLevelCount > 1 ? 4 / 3 : 1);
};

// Returns { line(): the held bytes in MB, the largest four sites; held(): the bytes; stop(): no more lines } (null
// without WebGPU). The counting stays once installed (the prototypes' methods wrapped).
export function gpuLedger(log = () => {}, everyMs = 10000) {
  if (typeof GPUDevice === 'undefined') return null;
  const bySite = new Map(), held = new WeakMap();
  let gcBytes = 0;
  const drop = (h) => { h.state.done = true; const e = bySite.get(h.site); e.bytes -= h.bytes; e.n--; };
  const gone = new FinalizationRegistry((h) => { if (!h.state.done) { drop(h); gcBytes += h.bytes; } });
  // the caller of createBuffer / createTexture: [0] Error, [1] site, [2] add, [3] the wrapper, [4] it
  const site = () => (new Error().stack.split('\n')[4] ?? '').trim().replace(/^at /, '').replace(/https?:\/\/[^/]+\/[^ )]*\/src\//, '');
  const add = (obj, bytes) => {
    const s = site(), e = bySite.get(s) ?? { bytes: 0, n: 0 };
    e.bytes += bytes; e.n++; bySite.set(s, e);
    const h = { site: s, bytes, state: { done: false } };
    held.set(obj, h);
    gone.register(obj, h);
  };
  const wrap = (proto, make, sizeOf) => {
    const f = GPUDevice.prototype[make];
    GPUDevice.prototype[make] = function (d) { const o = f.call(this, d); add(o, sizeOf(d)); return o; };
    const destroy = proto.prototype.destroy;
    proto.prototype.destroy = function () {
      const h = held.get(this);
      if (h && !h.state.done) drop(h);
      return destroy.call(this);
    };
  };
  wrap(GPUBuffer, 'createBuffer', d => d.size);
  wrap(GPUTexture, 'createTexture', sizeOfTexture);
  const mb = (b) => (b / 2 ** 20).toFixed(0);
  const total = () => [...bySite.values()].reduce((s, e) => s + e.bytes, 0);
  const line = () => {
    const top = [...bySite.entries()].filter(([, e]) => e.bytes > 2 ** 20).sort((a, b) => b[1].bytes - a[1].bytes).slice(0, 4);
    return `[gpu ledger] ${mb(total())} MB held (${mb(gcBytes)} MB collected undestroyed so far)` +
           (top.length ? `: ${top.map(([s, e]) => `${mb(e.bytes)} MB in ${e.n} at ${s}`).join('; ')}` : '');
  };
  const timer = everyMs > 0 ? setInterval(() => log(line()), everyMs) : null;
  return { line, held: total, stop: () => clearInterval(timer) };
}
