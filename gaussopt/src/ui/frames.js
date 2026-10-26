// frames.js — The keyframe contact sheet over the 3D view (toolbar "▦ frames"): what the user judges
// in the Keyframes phase, when there is nothing 3D to show yet. Thumbnails come from the worker
// (small JPEGs); each is framed by its quality score once the cameras exist, the selection is
// outlined, and a click selects that keyframe (shift-click: a range) for the Cameras repairs.
"use strict";

const SCORE_CSS = { green: '#4ade80', amber: '#fbbf24', red: '#f87171' };

// p007 · 14 mm · portrait · 25 s: the photo's short name, its 35 mm equivalent focal (zoom included),
// its orientation as shot and its time from the first photo
function photoCaption({ name, time, labels: { focal35, zoom, orientation } }) {
  const lens = focal35 ? `${+focal35.toFixed(1)} mm${zoom !== 1 ? ` (${zoom}x)` : ''}` : zoom !== 1 ? `${zoom}x` : null;
  return [name.match(/^p\d+/)?.[0] ?? name, lens, orientation, `${Math.round(time)} s`].filter(Boolean).join(' · ');
}

export class FramesSheet {
  // onPick(names): the user selected keyframes
  constructor(host, button, onPick) {
    this.host = host;
    this.button = button;
    this.onPick = onPick;
    this.cells = new Map();  // name -> { el, url }
    this.order = [];
    this.anchor = null;      // name of the last click (shift-click selects up to here)
    button.addEventListener('click', () => this.toggle());
  }

  get shown() { return this.host.classList.contains('shown'); }

  toggle(on = !this.shown) {
    this.host.classList.toggle('shown', on);
    this.button.classList.toggle('on', on);
  }

  // frames: [{ name, time, blob }] in time order; replaces the sheet. aspect: their width / height.
  // A photo set's (DESIGN-sfm §13.3 P5) also have labels { focal35, zoom, orientation, lens }, used
  // (false: greyed, with the reason: another size, filtered out)
  set(frames, aspect = 16 / 9) {
    this.host.style.setProperty('--frame-aspect', aspect);
    for (const c of this.cells.values()) URL.revokeObjectURL(c.url);
    this.cells.clear();
    this.order = frames.map(f => f.name);
    this.host.replaceChildren(...frames.map(f => {
      const el = document.createElement('figure');
      const url = URL.createObjectURL(f.blob);
      const img = document.createElement('img');
      img.src = url;
      img.alt = f.name;
      img.loading = 'lazy';
      const cap = document.createElement('figcaption');
      cap.textContent = f.labels ? photoCaption(f) : `${f.name.replace(/\.jpg$/, '')} · ${f.time.toFixed(2)} s`;
      if (f.used === false) { el.classList.add('unused'); el.dataset.reason = f.reason ?? ''; }
      el.append(img, cap);
      if (f.used !== false) el.addEventListener('click', (e) => this._click(f.name, e.shiftKey));  // left-out photos are not keyframes
      this.cells.set(f.name, { el, url });
      return el;
    }));
    this.button.disabled = frames.length === 0;
  }

  // scores: Map name -> { score, reason } (keyframes the session knows)
  setScores(scores) {
    for (const [name, { el }] of this.cells) {
      const s = scores.get(name);
      el.style.borderColor = s ? SCORE_CSS[s.score] ?? '' : '';
      el.title = s ? `${name}: ${s.score}${s.reason ? ` — ${s.reason}` : ''}` : el.dataset.reason ? `${name}: not used (${el.dataset.reason})` : name;
    }
  }

  setSelected(names) {
    const sel = new Set(names);
    for (const [name, { el }] of this.cells) el.classList.toggle('selected', sel.has(name));
  }

  _click(name, range) {
    if (range && this.anchor) {
      const [a, b] = [this.order.indexOf(this.anchor), this.order.indexOf(name)].sort((x, y) => x - y);
      this.onPick(this.order.slice(a, b + 1));
      return;
    }
    this.anchor = name;
    this.onPick([name]);
  }
}
