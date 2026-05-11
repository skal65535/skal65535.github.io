// common/utils.js — shared helpers used by several demos.
//
// ES module. Add new helpers here rather than spinning up another file in
// common/. Currently exports:
//
//   makeRng        — seedable PRNG (mulberry32)
//   drawCircle     — 2D canvas circle (stroke or fill)
//   drawLine       — 2D canvas line
//   makeFPS        — EMA-smoothed frame-rate counter
//   onFileDrop     — drag-and-drop file handler on a DOM element
//   loadImageFile  — Promise wrapper around FileReader + Image
//   preventDefaults — convenience: stops propagation of an event

// Seedable PRNG. `next01()` returns a float in [0, 1); `nextSigned()` returns
// one in [-1, 1). The seed lives on `rng.seed` and can be reassigned mid-stream.
export function makeRng(seed = 91651088029) {
  const r = {
    seed,
    next01() {
      let t = r.seed += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    },
    nextSigned() { return r.next01() * 2 - 1; },
  };
  return r;
}

// 2D canvas helpers. Take the context explicitly so callers don't need a global.
export function drawCircle(ctx, x, y, color, radius, fill = false) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
  ctx.strokeStyle = ctx.fillStyle = color;
  fill ? ctx.fill() : ctx.stroke();
}
export function drawLine(ctx, x1, y1, x2, y2, color) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.stroke();
}

// EMA-smoothed FPS. Call .tick() once per frame; reads the current value
// from .value or as the return of .tick().
export function makeFPS({ weight = 0.8 } = {}) {
  let lastT = performance.now();
  let fps = 0;
  return {
    tick() {
      const cur = performance.now();
      if (cur > lastT) {
        const n = 1000 / (cur - lastT);
        fps = Math.round(fps * weight + n * (1 - weight));
        lastT = cur;
      }
      return fps;
    },
    get value() { return fps; },
  };
}

// Drag-and-drop. `onFile(File)` is invoked when a file is dropped on `area`.
// Adds the `highlightClass` to `area` during a drag-over (matches the CSS hook
// `#main-area.highlight` already in common/main.css).
export function onFileDrop(area, onFile, { highlightClass = 'highlight' } = {}) {
  const prevent = e => { e.preventDefault(); e.stopPropagation(); };
  ['dragenter','dragover','dragleave','drop'].forEach(n => area.addEventListener(n, prevent));
  ['dragenter','dragover'].forEach(n => area.addEventListener(n, () => area.classList.add(highlightClass)));
  ['dragleave','drop'].forEach(n => area.addEventListener(n, () => area.classList.remove(highlightClass)));
  area.addEventListener('drop', e => {
    const f = e.dataTransfer?.files?.[0];
    if (f) onFile(f);
  });
}

// Read a File as a data URL, then decode it into an HTMLImageElement.
// Resolves with the loaded image; rejects on read or decode failure.
export function loadImageFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('FileReader failed'));
    reader.onloadend = () => {
      const img = new Image();
      img.onload  = () => resolve(img);
      img.onerror = () => reject(new Error('Image decode failed'));
      img.src = reader.result;
    };
    try { reader.readAsDataURL(file); }
    catch (e) { reject(e); }
  });
}

export function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }
