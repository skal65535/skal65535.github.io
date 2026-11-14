// view_switch.js — What the view area shows, one at a time (index.html #view-toolbar): the 3D view, the keyframes'
// contact sheet (frames.js) or the mapping played back (theatre.js). The training photo and the view's settings
// belong to the 3D view: shown in it only. Keys: 1 / 2 / 3 the three, Escape back to 3D.
"use strict";

export const VIEW_MODES = ['3d', 'frames', 'sfm'];
// a press of the photo button longer than this: held (shown while held), else a click (on until clicked again)
export const HOLD_MS = 250;

export class ViewSwitch {
  // buttons: { '3d', frames, sfm } the segment's buttons; layers: { frames, sfm } what covers the view (toggle(on),
  // their owner set to this: they close through it); inThreeD: the elements shown in 3D only; onSet(mode) after each
  constructor({ buttons, layers, inThreeD = [], onSet = () => {} }) {
    Object.assign(this, { buttons, layers, inThreeD, onSet, mode: '3d' });
    for (const l of Object.values(layers)) l.owner = this;
    for (const [m, b] of Object.entries(buttons)) b.addEventListener('click', () => this.set(m));
    this.set('3d');
  }

  // mode, or 3D for a mode without content (its button disabled); returns the mode shown
  set(mode) {
    if (!VIEW_MODES.includes(mode) || this.buttons[mode]?.disabled) mode = '3d';
    this.mode = mode;
    for (const [m, l] of Object.entries(this.layers)) l.toggle(m === mode);
    for (const [m, b] of Object.entries(this.buttons)) { b.classList.toggle('on', m === mode); b.setAttribute('aria-pressed', m === mode); }
    for (const el of this.inThreeD) el.hidden = mode !== '3d';
    this.onSet(mode);
    return mode;
  }

  // a layer closing itself (Escape in the theatre, the sheet emptied); one shown by its own toggle: hidden
  close(layer) { if (this.layers[this.mode] === layer) this.set('3d'); else layer.toggle(false); }

  // A key (KeyboardEvent.key lowercased); true: the switch's
  key(key) {
    const m = { 1: '3d', 2: 'frames', 3: 'sfm' }[key];
    if (m) { this.set(m); return true; }
    if (key === 'escape' && this.mode !== '3d') { this.set('3d'); return true; }
    return false;
  }
}

// The photo button: held, the photo while held; clicked, the photo until clicked again. onChange(on).
export class PhotoButton {
  constructor(button, onChange) {
    Object.assign(this, { button, onChange, latched: false, downT: 0 });
    button.addEventListener('pointerdown', e => {
      if (button.disabled) return;
      button.setPointerCapture(e.pointerId);
      this.downT = performance.now();
      this.onChange(true);
    });
    button.addEventListener('pointerup', () => {
      if (!this.downT) return;
      const click = performance.now() - this.downT < HOLD_MS;
      this.downT = 0;
      this.latched = click && !this.latched;
      this._show();
    });
    button.addEventListener('pointercancel', () => { this.downT = 0; this.latched = false; this._show(); });
  }

  get on() { return this.latched || this.downT > 0; }
  _show() { this.button.classList.toggle('on', this.latched); this.onChange(this.on); }
  // off (no photo to show, the view left)
  reset() { this.downT = 0; this.latched = false; this._show(); }
}
