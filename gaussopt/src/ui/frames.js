// frames.js — The keyframe contact sheet over the 3D view (toolbar "▦ frames"): what the user judges
// in the Keyframes phase, when there is nothing 3D to show yet. Thumbnails come from the worker
// (small JPEGs); each is framed by its quality score once the cameras exist, the selection is
// outlined, and a click selects that keyframe (shift-click: a range) for the Cameras repairs.
"use strict";

const SCORE_CSS = { green: '#4ade80', amber: '#fbbf24', red: '#f87171' };

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

  // frames: [{ name, time, blob }] in time order; replaces the sheet. aspect: their width / height
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
      cap.textContent = `${f.name.replace(/\.jpg$/, '')} · ${f.time.toFixed(2)} s`;
      el.append(img, cap);
      el.addEventListener('click', (e) => this._click(f.name, e.shiftKey));
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
      el.title = s ? `${name}: ${s.score}${s.reason ? ` — ${s.reason}` : ''}` : name;
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
