// common/controls.js — minimal lil-gui-shaped UI built from plain HTML.
//
//   import { mkGui } from '../common/controls.js';
//   const gui = mkGui(container, { title: 'Stippling' });
//   gui.add(params, 'num_points', 500, 500000, 10).name('Max Points').onChange(cb);
//   gui.add(params, 'use_edges').name('Edge detection').onChange(cb);
//   gui.add(params, 'mode', { 'Auto': 0, 'Manual': 1 }).onChange(cb);
//   gui.add({ reset: () => resetSim() }, 'reset').name('Reset');
//   const sub = gui.addFolder('Camera');
//   sub.add(params.cam, 'fov', 5, 180, 1);
//   gui.add(params, 'fov', 5, 180, 1).tip('vertical field of view, in degrees');
//
// Tooltips: .tip(text) on a control or a folder, or mkGui(parent, { tips }) with a map from a
// control's key (or a folder's name) to its text, which also covers controls added later (the
// map goes to sub-folders). They show in one popup (installTips), which also takes over the
// `title` of any element of the page (its native tooltip is slow and unstyled).
//
// Layout: each control is a <div class="skal-row skal-row-{type}"> containing
// its own label/input/readout. The panel itself is a flex column. This keeps
// mobile reflow simple (rows can stack their internals) and lets the panel
// float anywhere on desktop without grid-track surprises.

// A page-wide hook after every control change, (obj, key, value): e.g. to keep the URL params in sync.
let changeHook = null;
export function onAnyChange(fn) { changeHook = fn; }

export function mkGui(parent, { title = '', collapsed = false, draggable = true, tips = null } = {}) {
  installTips();
  const root = document.createElement('div');
  root.className = 'skal-gui';
  if (collapsed) root.classList.add('collapsed');
  if (title) {
    const h = document.createElement('div');
    h.className = 'skal-gui-title';
    h.textContent = title;
    if (draggable) installDrag(root, h);
    else h.addEventListener('click', () => root.classList.toggle('collapsed'));
    root.appendChild(h);
  }
  const body = document.createElement('div');
  body.className = 'skal-gui-body';
  root.appendChild(body);
  parent.appendChild(root);
  return new GuiGroup(body, tips);
}

// The tooltip popup, once per page: shown after a short hover on an element with a `data-tip`
// (or a `title`, moved to `data-tip` on the way, so a title set later is followed too), under it
// (above near the bottom of the window), hidden on leave, press, scroll or key
const TIP_DELAY_MS = 350;
let tipEl = null;
export function installTips() {
  if (tipEl || typeof document === 'undefined') return;
  tipEl = document.createElement('div');
  tipEl.className = 'skal-tip';
  tipEl.setAttribute('role', 'tooltip');
  Object.assign(tipEl.style, {
    position: 'fixed', zIndex: 10000, maxWidth: '320px', padding: '6px 9px', borderRadius: '6px', pointerEvents: 'none',
    background: '#0b0c10f2', color: '#dde', border: '1px solid #3a3e50', boxShadow: '0 4px 14px #0008',
    font: '12px/1.4 system-ui, sans-serif', whiteSpace: 'pre-line', display: 'none',
  });
  document.body.appendChild(tipEl);
  let over = null, timer = 0;
  const hide = () => { clearTimeout(timer); timer = 0; tipEl.style.display = 'none'; };
  const show = () => {
    const text = over?.isConnected && over.dataset.tip;
    if (!text) return hide();
    tipEl.textContent = text;
    tipEl.style.display = 'block';
    const r = over.getBoundingClientRect(), w = tipEl.offsetWidth, h = tipEl.offsetHeight;
    const x = Math.max(4, Math.min(innerWidth - w - 4, r.left + Math.min(r.width, 160) / 2 - 12));
    const y = r.bottom + 6 + h < innerHeight ? r.bottom + 6 : Math.max(4, r.top - 6 - h);
    tipEl.style.left = `${x}px`;
    tipEl.style.top = `${y}px`;
  };
  document.addEventListener('pointermove', (e) => {
    if (e.pointerType === 'touch') return;
    const el = e.target.closest?.('[data-tip], [title]');
    if (el?.title) { el.dataset.tip = el.title; el.removeAttribute('title'); }
    if (el !== over) { hide(); over = el; if (el) timer = setTimeout(show, TIP_DELAY_MS); }
    else if (el && tipEl.style.display === 'block' && tipEl.textContent !== el.dataset.tip) show();  // a live tip (canvas)
  }, { passive: true });
  const reset = () => { hide(); over = null; };
  for (const type of ['pointerdown', 'keydown', 'scroll']) document.addEventListener(type, reset, { capture: true, passive: true });
  window.addEventListener('blur', reset);
  document.documentElement.addEventListener('pointerleave', reset);
}

// Pointer-event drag on the title bar. Click (no drag) toggles collapse.
// Listeners are attached to `document` during a drag — that survives the
// pointer leaving the handle (which happens immediately because the handle
// itself moves with the panel).
function installDrag(root, handle) {
  handle.addEventListener('pointerdown', ev => {
    if (ev.pointerType === 'mouse' && ev.button !== 0) return;
    const x0 = ev.clientX, y0 = ev.clientY;
    const l0 = root.offsetLeft, t0 = root.offsetTop;
    let moved = false;
    ev.preventDefault();

    const onMove = me => {
      const dx = me.clientX - x0, dy = me.clientY - y0;
      if (!moved && (dx * dx + dy * dy) < 25) return;        // 5px threshold
      moved = true;
      root.style.left   = (l0 + dx) + 'px';
      root.style.top    = (t0 + dy) + 'px';
      root.style.right  = 'auto';
      root.style.bottom = 'auto';
    };
    const onEnd = () => {
      document.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerup',   onEnd);
      document.removeEventListener('pointercancel', onEnd);
      if (!moved) root.classList.toggle('collapsed');
    };
    document.addEventListener('pointermove', onMove);
    document.addEventListener('pointerup',   onEnd);
    document.addEventListener('pointercancel', onEnd);
  });
}

class GuiGroup {
  constructor(host, tips = null, title = null) {
    this._host     = host;
    this._controls = [];   // direct children (controls and sub-groups)
    this._tips     = tips; // key or folder name -> tooltip, for the controls added here and below
    this._title    = title;
  }

  tip(text) { if (this._title) setTip(this._title, text); return this; }

  add(obj, key, a, b, c) {
    const v = obj[key];
    // 3rd-arg array → select with array values labelled by their string form.
    if (Array.isArray(a)) {
      const opts = Object.fromEntries(a.map(x => [String(x), x]));
      return this._mount(new SelectCtrl(obj, key, opts));
    }
    let ctrl;
    if (typeof v === 'function')                              ctrl = new ButtonCtrl(obj, key);
    else if (typeof v === 'boolean')                          ctrl = new ToggleCtrl(obj, key);
    else if (a !== undefined && typeof a === 'object' && a !== null)
                                                              ctrl = new SelectCtrl(obj, key, a);
    else if (typeof v === 'number' && typeof a === 'number')  ctrl = new SliderCtrl(obj, key, a, b ?? a + 1, c);
    else                                                      ctrl = new TextCtrl(obj, key);
    return this._mount(ctrl);
  }

  addColor(obj, key) { return this._mount(new ColorCtrl(obj, key)); }

  addFolder(name, { collapsed = false } = {}) {
    const wrap = document.createElement('div');
    wrap.className = 'skal-folder';
    if (collapsed) wrap.classList.add('collapsed');
    const t = document.createElement('div');
    t.className = 'skal-folder-title';
    t.textContent = name;
    t.addEventListener('click', () => wrap.classList.toggle('collapsed'));
    wrap.appendChild(t);
    const body = document.createElement('div');
    body.className = 'skal-folder-body';
    wrap.appendChild(body);
    this._host.appendChild(wrap);
    const group = new GuiGroup(body, this._tips, t);
    if (this._tips?.[name]) group.tip(this._tips[name]);
    this._controls.push(group);
    return group;
  }

  // lil-gui compatibility: walk all leaf controls under this group.
  controllersRecursive() {
    const out = [];
    for (const c of this._controls) {
      if (c instanceof GuiGroup) out.push(...c.controllersRecursive());
      else out.push(c);
    }
    return out;
  }

  // Remove every control and sub-group (e.g. to rebuild a select with new options).
  clear() {
    this._host.replaceChildren();
    this._controls = [];
    return this;
  }

  // Re-read every control's obj[key] (params changed outside the GUI).
  updateDisplay() {
    for (const c of this.controllersRecursive()) c.updateDisplay();
    return this;
  }

  _mount(ctrl) {
    ctrl._mount(this._host);
    if (this._tips?.[ctrl.property]) ctrl.tip(this._tips[ctrl.property]);
    this._controls.push(ctrl);
    return ctrl;
  }
}

// ── Control base ─────────────────────────────────────────────────────────────
class Ctrl {
  constructor(obj, key) {
    this._obj = obj;
    this._key = key;
    this._cb  = null;
    this._labelText = key;
    this._inputs = [];     // input elements to disable / poll
    this.property = key;   // lil-gui compatibility: callers find ctrls by this
  }
  name(text) { this._labelText = text; if (this._label) this._label.textContent = text; return this; }
  onChange(cb) { this._cb = cb; return this; }
  tip(text) { if (this._row_el) setTip(this._row_el, text); return this; }
  _fire() {
    if (this._cb) this._cb(this._obj[this._key]);
    changeHook?.(this._obj, this._key, this._obj[this._key]);
  }

  // Disable user interaction. Used by demos for read-only displays
  // (combine with .listen() so the displayed value still updates).
  // `enable(bool)` mirrors lil-gui: enable(false) ≡ disable().
  disable(yes = true) { return yes ? this._setDisabled(true)  : this._setDisabled(false); }
  enable (yes = true) { return yes ? this._setDisabled(false) : this._setDisabled(true);  }
  _setDisabled(d) {
    this._disabled = d;
    if (this._row_el) this._row_el.classList.toggle('skal-readonly', d);
    for (const el of this._inputs) el.disabled = d;
    return this;
  }

  // Poll obj[key] each frame and reflect it back into the UI. Used for
  // read-only displays (.listen().disable()) and for params changed
  // externally (e.g. autoplay updating a slider).
  listen() {
    if (this._polling) return this;
    this._polling = true;
    let last;
    const tick = () => {
      if (!this._row_el?.isConnected) return;
      const v = this._obj[this._key];
      if (v !== last) { this._refresh(v); last = v; }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    return this;
  }
  _refresh(_v) { /* subclasses override to push v back into their inputs */ }
  // lil-gui compatibility: show the current obj[key].
  updateDisplay() { this._refresh(this._obj[this._key]); return this; }

  _row(type) {
    const row = document.createElement('div');
    row.className = `skal-row skal-row-${type}`;
    this._row_el = row;
    return row;
  }
  _label_el() {
    const el = document.createElement('label');
    el.className = 'skal-row-label';
    el.textContent = this._labelText;
    return el;
  }
}

// ── Slider (number with min/max/step) ────────────────────────────────────────
class SliderCtrl extends Ctrl {
  constructor(obj, key, min, max, step) {
    super(obj, key);
    this._min = min; this._max = max;
    this._step = step ?? autoStep(min, max);
  }
  _mount(host) {
    const row = this._row('slider');
    this._label = this._label_el();
    this._read  = document.createElement('span');
    this._read.className = 'skal-readout';
    this._read.textContent = fmt(this._obj[this._key], this._step);

    this._range = document.createElement('input');
    this._range.type = 'range';
    this._range.min  = this._min;
    this._range.max  = this._max;
    this._range.step = this._step;
    this._range.value = this._obj[this._key];
    this._range.addEventListener('input', () => {
      const v = Number(this._range.value);
      this._obj[this._key] = v;
      this._read.textContent = fmt(v, this._step);
      this._fire();
    });
    this._inputs = [this._range];

    const header = document.createElement('div');
    header.className = 'skal-row-header';
    header.appendChild(this._label);
    header.appendChild(this._read);
    row.appendChild(header);
    row.appendChild(this._range);
    host.appendChild(row);
  }
  // lil-gui compatibility: change the range after creation (the value is not clamped).
  min(v) { this._min = v; if (this._range) this._range.min = v; return this.updateDisplay(); }
  max(v) { this._max = v; if (this._range) this._range.max = v; return this.updateDisplay(); }
  _refresh(v) {
    if (this._range) this._range.value = v;
    if (this._read)  this._read.textContent = fmt(v, this._step);
  }
}

// ── Toggle (boolean) ─────────────────────────────────────────────────────────
class ToggleCtrl extends Ctrl {
  _mount(host) {
    const row = this._row('toggle');
    this._label = this._label_el();
    this._cb_el = document.createElement('input');
    this._cb_el.type = 'checkbox';
    this._cb_el.checked = !!this._obj[this._key];
    this._cb_el.addEventListener('change', () => {
      this._obj[this._key] = this._cb_el.checked;
      this._fire();
    });
    this._inputs = [this._cb_el];
    row.appendChild(this._label);
    row.appendChild(this._cb_el);
    host.appendChild(row);
  }
  _refresh(v) { if (this._cb_el) this._cb_el.checked = !!v; }
}

// ── Select (named options object) ────────────────────────────────────────────
class SelectCtrl extends Ctrl {
  constructor(obj, key, options) { super(obj, key); this._options = options; }
  _mount(host) {
    const row = this._row('select');
    this._label = this._label_el();
    this._sel = document.createElement('select');
    for (const [name, val] of Object.entries(this._options)) {
      const o = document.createElement('option');
      o.value = JSON.stringify(val);
      o.textContent = name;
      if (sameVal(val, this._obj[this._key])) o.selected = true;
      this._sel.appendChild(o);
    }
    this._sel.addEventListener('change', () => {
      this._obj[this._key] = JSON.parse(this._sel.value);
      this._fire();
    });
    this._inputs = [this._sel];
    row.appendChild(this._label);
    row.appendChild(this._sel);
    host.appendChild(row);
  }
  _refresh(v) {
    if (!this._sel) return;
    const want = JSON.stringify(v);
    for (const o of this._sel.options) if (o.value === want) { this._sel.value = want; break; }
  }
}

// ── Button (calls the function on obj[key]) ──────────────────────────────────
class ButtonCtrl extends Ctrl {
  _mount(host) {
    const row = this._row('button');
    this._btn = document.createElement('button');
    this._btn.type = 'button';
    this._btn.className = 'skal-btn';
    this._btn.textContent = this._labelText;
    this._btn.addEventListener('click', () => this._obj[this._key]());
    this._label = this._btn;     // .name() retargets to the button text
    row.appendChild(this._btn);
    host.appendChild(row);
  }
  name(text) { this._labelText = text; if (this._btn) this._btn.textContent = text; return this; }
}

// ── Color picker ─────────────────────────────────────────────────────────────
class ColorCtrl extends Ctrl {
  _mount(host) {
    const row = this._row('color');
    this._label = this._label_el();
    this._inp = document.createElement('input');
    this._inp.type = 'color';
    this._inp.value = toHex(this._obj[this._key]);
    this._inp.addEventListener('input', () => {
      const v = this._inp.value;
      this._obj[this._key] = (typeof this._obj[this._key] === 'number')
                              ? parseInt(v.slice(1), 16) : v;
      this._fire();
    });
    this._inputs = [this._inp];
    row.appendChild(this._label);
    row.appendChild(this._inp);
    host.appendChild(row);
  }
  _refresh(v) { if (this._inp) this._inp.value = toHex(v); }
}

// ── Free-form text (fallback; also used for read-only displays) ──────────────
class TextCtrl extends Ctrl {
  _mount(host) {
    const row = this._row('text');
    this._label = this._label_el();
    this._inp = document.createElement('input');
    this._inp.type = 'text';
    this._inp.value = String(this._obj[this._key] ?? '');
    this._inp.addEventListener('change', () => {
      this._obj[this._key] = this._inp.value;
      this._fire();
    });
    this._inputs = [this._inp];
    row.appendChild(this._label);
    row.appendChild(this._inp);
    host.appendChild(row);
  }
  _refresh(v) { if (this._inp) this._inp.value = String(v ?? ''); }
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function setTip(el, text) { if (text) el.dataset.tip = text; else delete el.dataset.tip; }
function autoStep(min, max) {
  const span = Math.abs(max - min);
  if (span >= 100) return 1;
  if (span >= 10)  return 0.1;
  if (span >= 1)   return 0.01;
  return 0.001;
}
function fmt(v, step) {
  if (!Number.isFinite(v)) return String(v);
  const d = step >= 1 ? 0 : step >= 0.1 ? 1 : step >= 0.01 ? 2 : 3;
  return Number(v).toFixed(d);
}
function sameVal(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
function toHex(v) {
  if (typeof v === 'string') {
    return v.startsWith('#') ? v : '#' + v;
  }
  return '#' + (v >>> 0).toString(16).padStart(6, '0');
}
