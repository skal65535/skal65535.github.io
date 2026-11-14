// theatre.js — The SfM theatre (toolbar "◈ SfM"): the mapping played back over the 3D view, from its trace
// (sfm/trace.js), live while it runs or replayed after. Each keyframe placed, each window bundle adjustment, each
// loop closed and each piece merged or joined is drawn as it happened: the cameras along their path coloured by
// piece, a sample of the points, the event's frames lit, and a caption saying what the stage does and why.
// Optional: nothing is drawn while it is hidden. Its own canvas (2D), its own orbit view: the splats need not exist.
"use strict";

import { stateAt } from '../sfm/trace.js';
import { cross, normalize } from '../math.js';

const POSE = 12;
const PALETTE = ['#60a5fa', '#f472b6', '#34d399', '#fbbf24', '#a78bfa', '#f87171', '#22d3ee', '#a3e635', '#fb923c', '#e879f9'];
const segColor = (s) => PALETTE[((s % PALETTE.length) + PALETTE.length) % PALETTE.length];
const CACHE_EVERY = 256;  // a copy of the state every this many events: going back replays from the nearest
const MIN_SHARE = 0.07;   // of the timeline, each stage at least (the sequential mapping is most of the events)
const SCRUB = 1000;       // the scrubber's steps

// what each stage does, for the caption (DESIGN-sfm §5)
const STAGES = {
  map: ['Sequential mapping',
    'Two keyframes far enough apart seed a piece of map: their relative pose from the matched points (the essential ' +
    'matrix), the points triangulated. Each next keyframe is placed by its matches with the points already mapped ' +
    '(PnP), adds the points only it and its neighbours see, and a window bundle adjustment fits the last cameras and ' +
    'their points to every observation.'],
  loops: ['Loop closure',
    'The camera comes back over places it has seen. Pairs that look alike (a bag of visual words) are checked ' +
    'geometrically; a verified pair between two pieces merges them by a similarity (rotation, translation, scale). ' +
    'Within a piece it is a constraint: the pose graph bends the whole path to honour it, spreading the drift ' +
    'accumulated since.'],
  flowJoin: ['Turns in place',
    'A quadcopter that stops and turns sees new things from the same place: nothing to triangulate, so the map ' +
    'breaks. The optical flow through the turn gives the rotation; the scale comes from keyframes both pieces can ' +
    'place, else from the speed the drone flew at either side.'],
  refine: ['Refinement',
    'Window bundle adjustments glide along the whole path, twice, each window with the frames that see the same ' +
    'places (its loop partners), then the lens fitted once to the whole map.'],
  retriangulate: ['Retriangulation', 'With the final cameras, each point\'s track is extended to every keyframe that sees it, and new points triangulated.'],
  lateMerge: ['Late merges', 'The longer tracks may now tie pieces that could not be merged before.'],
  repair: ['Repairs', 'Stretches of poorly placed keyframes are taken out and placed again; a repair is kept only if the map fits better.'],
  undo: ['Undo', 'The last action undone: the cameras and points back as they were before it.'],
  end: ['Done', 'The cameras and points go to the trainer.'],
};
// what an event says (the caption's last line). A number not finite is saved as null (JSON): '?'
const fx = (x, d) => (Number.isFinite(x) ? x.toFixed(d) : '?'), ex = (x) => (Number.isFinite(x) ? x.toExponential(1) : '?');
const SAYS = {
  init: (i) => `piece ${i.segment} starts at keyframes ${i.i}-${i.j}: ${i.points} points, ${fx(i.angleDeg, 1)}° between their rays — its own origin and scale`,
  register: (i) => `keyframe ${i.img} placed`,
  ba: (i) => `window BA: ${i.free?.length ?? '?'} cameras free, ${i.observers - (i.free?.length ?? 0)} fixed, ${i.points} points`,
  segment: (i) => `piece ${i.segment} ends: ${i.size} keyframes (${i.first}..${i.last}); nothing more can be placed in it`,
  loops: (i) => `${i.pairs?.length ?? 0} loops verified within piece ${i.segment}`,
  posegraph: (i) => `pose graph of piece ${i.segment}: ${i.frames} cameras, ${i.loops} loops, cost ${ex(i.initialCost)} → ${ex(i.cost)}`,
  merge: (i) => `pieces ${i.a} and ${i.b} merged by ${i.inliers} shared points, scale ×${fx(i.scale, 2)}`,
  join: (i) => `turn in place at keyframes ${i.a}|${i.b} (${fx(i.turnDeg, 0)}°): piece ${i.from} joins ${i.into}, scale ×${fx(i.scale, 2)} by ${i.by === 'shared' ? 'keyframes placed in both' : 'the speeds'}`,
  unregister: (i) => `${i.frames?.length ?? 0} keyframes taken out`,
  tracks: (i) => `keyframes ${i.first}..${i.last}: ${i.created} new points`,
  undo: (i) => `${i.op ?? 'the last action'} undone`,
  end: () => 'done',
};
// the caption's line for event e (null: the event says nothing of its own)
export const eventSays = (e) => SAYS[e.kind]?.(e.info ?? {}) ?? null;
// a stage's [title, text]; a repair action has its own name
export const stageText = (name) => STAGES[name] ?? (name ? [`Repair: ${name}`, STAGES.repair[1]] : ['', '']);
// how long an event's frames stay lit (s), and in what
const FLASH = { register: [0.8, '#ffffff'], ba: [0.35, '#67e8f9'], merge: [1.6, '#fde047'], join: [1.6, '#fb923c'], posegraph: [1.4, '#86efac'], init: [1.2, '#ffffff'], tracks: [0.6, '#c4b5fd'] };

export class SfmTheatre {
  constructor(host, button) {
    this.host = host;
    this.button = button;
    host.innerHTML = `
      <canvas class="th-canvas"></canvas>
      <div class="th-caption"><div class="th-stage"></div><div class="th-text"></div><div class="th-says"></div></div>
      <div class="th-counts"></div>
      <div class="th-bar">
        <button class="th-play" title="play / pause (space)">▶</button>
        <div class="th-track"><input class="th-scrub" type="range" min="0" max="${SCRUB}" value="0"><div class="th-chapters"></div></div>
        <select class="th-speed" title="events a second">${[10, 30, 100, 300, 1000].map(v => `<option value="${v}"${v === 100 ? ' selected' : ''}>${v}/s</option>`).join('')}</select>
        <button class="th-live" title="follow the mapping as it happens">● live</button>
        <button class="th-fit" title="frame the cameras again (double-click the view)">⤢</button>
      </div>`;
    const q = (c) => host.querySelector(c);  // (keys: SfmTheatre.key, from the app's own handler)
    Object.assign(this, { canvas: q('.th-canvas'), stageEl: q('.th-stage'), textEl: q('.th-text'), saysEl: q('.th-says'), countsEl: q('.th-counts'),
                          playEl: q('.th-play'), scrubEl: q('.th-scrub'), chaptersEl: q('.th-chapters'), speedEl: q('.th-speed'), liveEl: q('.th-live') });
    this.ctx = this.canvas.getContext('2d');
    this.view = { yaw: 0.6, pitch: 0.5, dist: 10, target: [0, 0, 0], up: [0, -1, 0], fov: 0.9 };
    this.userMoved = false;
    this.reset();
    button?.addEventListener('click', () => this.toggle());  // (in the app: none, its ViewSwitch shows it)
    this.playEl.addEventListener('click', () => { this.playing = !this.playing; if (this.playing && this.cursor >= this.events.length - 1) this.seek(0); this.live = false; this._bar(); });
    this.liveEl.addEventListener('click', () => { this.live = !this.live; this.playing = false; this._bar(); });
    this.scrubEl.addEventListener('input', () => { this.live = this.playing = false; this.seek(this._eventAt(Number(this.scrubEl.value) / SCRUB)); this._bar(); });
    q('.th-fit').addEventListener('click', () => { this.userMoved = false; });
    this._mouse();
    new ResizeObserver(() => this._resize()).observe(this.canvas);
  }

  get shown() { return this.host.classList.contains('shown'); }

  // A key while shown (key: KeyboardEvent.key lowercased): space plays or pauses, the arrows step an event, Escape
  // closes. True: the theatre's (none of the view's flight keys reach the 3D view under it while shown)
  key(key) {
    if (!this.shown) return false;
    if (key === ' ') this.playEl.click();
    else if (key === 'escape') this.owner ? this.owner.close(this) : this.toggle(false);
    else if (key === 'arrowright' || key === 'arrowleft') { this.live = this.playing = false; this.seek(this.cursor + (key === 'arrowright' ? 1 : -1)); this._bar(); }
    return [' ', 'escape', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'e', 'q'].includes(key);
  }

  toggle(on = !this.shown) {
    if (on === this.shown) return;
    this.host.classList.toggle('shown', on);
    this.button?.classList.toggle('on', on);
    if (on) { this._resize(); this._loop(); }
  }

  // a new mapping (or none)
  reset() {
    this.events = [];
    this.cursor = -1;          // the last event applied
    this.state = { poses: new Float32Array(0), segments: new Int32Array(0) };
    this.shownPoses = null;    // what is drawn: eased towards state.poses
    this.cache = new Map();    // event index -> a copy of the state after it
    this.points = null;        // the last points checkpoint applied
    this.images = [];
    this.flashes = [];         // [{ frames, until, color, dur }]
    this.arcs = [];            // [{ a, b, until }]: loop pairs
    this.counts = { ba: 0, loops: 0, merges: 0, joins: 0 };
    // nothing mapped yet: what comes here, and where the keyframes are meanwhile
    this.stage = 'none';
    this.stageEl.textContent = 'Nothing mapped yet';
    this.textEl.textContent = 'The mapping plays here once the keyframes are extracted and matched; meanwhile their thumbnails come in ▦ frames (key 2).';
    this.saysEl.textContent = '';
    this.says = '';
    this.live = true;
    this.playing = false;
    this.chapters = [];
    this._bar();
  }

  // live: events as the worker sends them; followed when live
  push(events) {
    for (const e of events) {
      if (e.kind === 'stage') this.chapters.push([this.events.length, e.info.name]);
      this.events.push(e);
    }
    this._chapters();
  }

  // a whole recorded trace, from its start
  load(events) {
    this.reset();
    this.push(events);
    this.live = false;
    this.playing = true;
    this._bar();
  }

  // the state after event `to` (forward: applied in turn, with its effects; back: from the nearest copy)
  seek(to) {
    to = Math.max(-1, Math.min(this.events.length - 1, Math.round(to)));
    if (to < this.cursor) {
      let from = -1;
      for (const k of this.cache.keys()) if (k <= to && k > from) from = k;
      if (from >= 0) {
        const c = this.cache.get(from);
        this.state = { poses: c.poses.slice(), segments: c.segments.slice() };
        Object.assign(this, { points: c.points, counts: { ...c.counts }, images: c.images });
      } else {
        this.state = { poses: new Float32Array(0), segments: new Int32Array(0) };
        Object.assign(this, { points: null, counts: { ba: 0, loops: 0, merges: 0, joins: 0 }, images: [] });
      }
      this.cursor = from;
      this.flashes = []; this.arcs = [];
      this.shownPoses = null;  // drawn where they are (a renumbering between: the same index, another keyframe)
      while (this.cursor < to) this._apply(this.events[++this.cursor], false);
      this._caption(this.events[to]);
      return;
    }
    while (this.cursor < to) this._apply(this.events[++this.cursor], to - this.cursor < 40);
  }

  _apply(e, animate) {
    if (e.reset) { this.state = stateAt([e]); this.shownPoses = null; }
    if (e.images) this.images = e.images;
    const last = e.frames.length ? e.frames[e.frames.length - 1] + 1 : 0;
    if (last * POSE > this.state.poses.length) {
      const p = new Float32Array(last * POSE).fill(NaN), s = new Int32Array(last).fill(-1);
      p.set(this.state.poses); s.set(this.state.segments);
      this.state = { poses: p, segments: s };
    }
    for (let m = 0; m < e.frames.length; m++) {
      this.state.poses.set(e.poses.subarray(m * POSE, (m + 1) * POSE), e.frames[m] * POSE);
      this.state.segments[e.frames[m]] = e.segments[m];
    }
    if (e.points) this.points = e.points;
    if (e.kind === 'ba') this.counts.ba++;
    if (e.kind === 'loops') this.counts.loops += e.info.pairs.length;
    if (e.kind === 'merge') this.counts.merges++;
    if (e.kind === 'join') this.counts.joins++;
    if (animate) {
      const now = performance.now() / 1000, f = FLASH[e.kind];
      const lit = e.kind === 'ba' ? e.info.free : e.kind === 'register' ? [e.info.img] : e.kind === 'init' ? [e.info.i, e.info.j] :
                  e.kind === 'tracks' ? Array.from({ length: e.info.last - e.info.first + 1 }, (_, m) => e.info.first + m) : e.frames;
      if (f && lit.length) this.flashes.push({ frames: Int32Array.from(lit), until: now + f[0], dur: f[0], color: f[1] });
      if (e.kind === 'loops') for (const [a, b] of e.info.pairs) this.arcs.push({ a, b, until: now + 3 });
      if (e.kind === 'join') this.arcs.push({ a: e.info.a, b: e.info.b, until: now + 4, color: '#fb923c' });
    }
    this._caption(e);
    if (this.cursor > 0 && this.cursor % CACHE_EVERY === 0 && !this.cache.has(this.cursor))
      this.cache.set(this.cursor, { poses: this.state.poses.slice(), segments: this.state.segments.slice(), points: this.points, counts: { ...this.counts }, images: this.images });
  }

  _caption(e) {
    if (!e) return;
    // the stage: the last 'stage' event at or before the cursor
    let name = null;
    for (const [at, n] of this.chapters) if (at <= this.cursor) name = n;
    if (e.kind === 'end') name = 'end';
    if (name !== this.stage) {
      this.stage = name;
      const [title, text] = stageText(name);
      this.stageEl.textContent = title;
      this.textEl.textContent = text;
    }
    this.saysEl.textContent = eventSays(e) ?? this.saysEl.textContent;
  }

  _bar() {
    this.playEl.textContent = this.playing ? '❚❚' : '▶';
    this.liveEl.classList.toggle('on', this.live);
    this.scrubEl.value = Math.round(SCRUB * this._placeOf(Math.max(0, this.cursor)));
  }

  // The timeline, piecewise: each stage's events over a share of it, its count's but at least MIN_SHARE.
  // [[first event, its place 0..1]] by stage, then the end
  _timeline() {
    const N = this.events.length, starts = this.chapters.map(([at]) => at);
    if (!starts.length || starts[0] > 0) starts.unshift(0);
    const counts = starts.map((s, i) => (starts[i + 1] ?? N) - s), w = counts.map(c => Math.max(c / Math.max(1, N), MIN_SHARE));
    const sum = w.reduce((a, b) => a + b, 0);
    let x = 0;
    this.knots = starts.map((s, i) => { const k = [s, x]; x += w[i] / sum; return k; });
    this.knots.push([N - 1, 1]);
  }

  _placeOf(e) {
    const K = this.knots ?? [[0, 0], [Math.max(1, this.events.length - 1), 1]];
    for (let i = K.length - 2; i >= 0; i--) if (e >= K[i][0]) {
      const [e0, x0] = K[i], [e1, x1] = K[i + 1];
      return e1 > e0 ? x0 + (x1 - x0) * Math.min(1, (e - e0) / (e1 - e0)) : x0;
    }
    return 0;
  }

  _eventAt(x) {
    const K = this.knots ?? [[0, 0], [Math.max(1, this.events.length - 1), 1]];
    for (let i = K.length - 2; i >= 0; i--) if (x >= K[i][1]) {
      const [e0, x0] = K[i], [e1, x1] = K[i + 1];
      return x1 > x0 ? e0 + (e1 - e0) * Math.min(1, (x - x0) / (x1 - x0)) : e0;
    }
    return 0;
  }

  _chapters() {
    this._timeline();
    this.chaptersEl.replaceChildren(...this.chapters.map(([at, name], i) => {
      const el = document.createElement('span'), x = this._placeOf(at), next = i + 1 < this.chapters.length ? this._placeOf(this.chapters[i + 1][0]) : 1;
      el.style.left = `${(100 * x).toFixed(2)}%`;
      el.style.maxWidth = `${(100 * (next - x)).toFixed(2)}%`;  // its own span: clipped there
      el.title = stageText(name)[0];
      el.textContent = stageText(name)[0].split(' ')[0];
      el.addEventListener('click', () => { this.live = this.playing = false; this.seek(at); this._bar(); });
      return el;
    }));
  }

  // ── drawing ──────────────────────────────────────────────────────────────────

  _resize() {
    const r = this.canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(r.width * dpr));
    this.canvas.height = Math.max(1, Math.round(r.height * dpr));
  }

  _loop() {
    if (this._raf || !this.shown) return;
    let t = performance.now();
    const frame = (now) => {
      this._raf = null;
      if (!this.shown) return;
      this._raf = requestAnimationFrame(frame);  // first: a throw below (a bad event) leaves the next frame asked for
      const dt = Math.min(0.1, (now - t) / 1000);
      t = now;
      if (this.live && this.cursor < this.events.length - 1) this.seek(Math.min(this.events.length - 1, this.cursor + Math.max(1, (this.events.length - 1 - this.cursor) * 4 * dt)));
      else if (this.playing) {
        this._acc = (this._acc ?? 0) + Number(this.speedEl.value) * dt;
        const steps = Math.floor(this._acc);
        this._acc -= steps;
        if (steps) this.seek(this.cursor + steps);
        if (this.cursor >= this.events.length - 1) this.playing = false;
      }
      this._bar();
      this._draw(dt);
    };
    this._raf = requestAnimationFrame(frame);
  }

  // the drawn poses eased towards the state's: a merge or a pose graph slides the cameras over to where they go
  _ease(dt) {
    const target = this.state.poses;
    if (!this.shownPoses || this.shownPoses.length !== target.length) {
      const prev = this.shownPoses;
      this.shownPoses = target.slice();
      if (prev) for (let i = 0; i < Math.min(prev.length, target.length); i++) if (!Number.isNaN(prev[i]) && !Number.isNaN(target[i])) this.shownPoses[i] = prev[i];
    }
    const a = 1 - Math.exp(-dt / 0.18), P = this.shownPoses;
    for (let i = 0; i < target.length; i++) {
      const g = target[i];
      if (Number.isNaN(g) || Number.isNaN(P[i])) P[i] = g;
      else P[i] += (g - P[i]) * a;
    }
  }

  // the piece in focus (the last event's), and the view framed on it unless the user moved it
  _frame(dt) {
    const P = this.shownPoses, S = this.state.segments, n = S.length;
    const e = this.events[this.cursor];
    let focus = e?.info?.segment ?? e?.info?.into ?? (e?.frames.length ? e.segments[e.segments.length - 1] : null);
    if (focus === null || focus === undefined || focus < 0) focus = this.focus ?? -1;
    if (![...S].includes(focus)) {  // merged away: the largest piece
      const size = new Map();
      for (const s of S) if (s >= 0) size.set(s, (size.get(s) ?? 0) + 1);
      focus = [...size].sort((x, y) => y[1] - x[1])[0]?.[0] ?? -1;
    }
    this.focus = focus;
    if (this.userMoved) return;
    const xs = [[], [], []], up = [0, 0, 0];
    for (let k = 0; k < n; k++) {
      if (S[k] !== focus || Number.isNaN(P[k * POSE])) continue;
      for (let i = 0; i < 3; i++) { xs[i].push(P[k * POSE + 9 + i]); up[i] -= P[k * POSE + 3 + i]; }
    }
    if (!xs[0].length) return;
    const q = (v, f) => { const s = v.slice().sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(f * s.length))]; };
    const lo = xs.map(v => q(v, 0.02)), hi = xs.map(v => q(v, 0.98));
    const c = lo.map((l, i) => (l + hi[i]) / 2), r = Math.max(1e-6, Math.hypot(...hi.map((h, i) => h - lo[i])) / 2);
    const a = 1 - Math.exp(-dt / 0.5), v = this.view, ul = Math.hypot(...up) || 1;
    for (let i = 0; i < 3; i++) { v.target[i] += (c[i] - v.target[i]) * a; v.up[i] += (up[i] / ul - v.up[i]) * a; }
    v.dist += (2.4 * r - v.dist) * a;
  }

  // world -> pixel: [x, y, depth] or null behind the eye
  _projector() {
    const v = this.view, W = this.canvas.width, H = this.canvas.height;
    const u = normalize(v.up), ref = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const e1 = normalize(cross(ref, u)), e2 = cross(u, e1);  // a frame around up
    const dir = [0, 1, 2].map(i => Math.cos(v.pitch) * (Math.cos(v.yaw) * e1[i] + Math.sin(v.yaw) * e2[i]) + Math.sin(v.pitch) * u[i]);
    const eye = v.target.map((t, i) => t + v.dist * dir[i]);
    const f = normalize(dir.map(x => -x)), r = normalize(cross(f, u)), up = cross(r, f), focal = 0.5 * H / Math.tan(v.fov / 2);
    this.eye = eye;
    return (x, y, z) => {
      const d = [x - eye[0], y - eye[1], z - eye[2]];
      const zc = d[0] * f[0] + d[1] * f[1] + d[2] * f[2];
      if (zc <= 1e-6 * v.dist) return null;
      return [W / 2 + focal * (d[0] * r[0] + d[1] * r[1] + d[2] * r[2]) / zc, H / 2 - focal * (d[0] * up[0] + d[1] * up[1] + d[2] * up[2]) / zc, zc];
    };
  }

  _draw(dt) {
    const g = this.ctx, W = this.canvas.width, H = this.canvas.height, dpr = devicePixelRatio || 1;
    g.fillStyle = '#0d0e13';
    g.fillRect(0, 0, W, H);
    this._ease(dt);
    this._frame(dt);
    const P = this.shownPoses, S = this.state.segments, n = S.length, proj = this._projector(), focus = this.focus;
    const alpha = (s) => (s === focus ? 1 : 0.18);
    // the points: the focus piece's bright
    if (this.points) {
      const { X, segment } = this.points, sz = Math.max(1, 1.2 * dpr);
      for (const pass of [false, true]) {
        for (let i = 0; i < segment.length; i++) {
          if ((segment[i] === focus) !== pass) continue;
          const p = proj(X[3 * i], X[3 * i + 1], X[3 * i + 2]);
          if (!p) continue;
          g.fillStyle = pass ? 'rgba(203, 213, 225, 0.55)' : 'rgba(148, 163, 184, 0.12)';
          g.fillRect(p[0], p[1], sz, sz);
        }
      }
    }
    // the path of each piece, in time order
    const C = (k) => (Number.isNaN(P[k * POSE]) ? null : proj(P[k * POSE + 9], P[k * POSE + 10], P[k * POSE + 11]));
    const at = new Array(n);
    for (let k = 0; k < n; k++) at[k] = C(k);
    g.lineWidth = 1.5 * dpr;
    for (let k = 1; k < n; k++) {
      const a = at[k - 1], b = at[k];
      if (!a || !b || S[k] !== S[k - 1] || S[k] < 0) continue;
      g.strokeStyle = segColor(S[k]);
      g.globalAlpha = alpha(S[k]);
      g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
    }
    // the cameras: a small pyramid along each one's view, every one when few, else every few
    const scale = 0.012 * this.view.dist, every = Math.max(1, Math.ceil(n / 150));
    const lit = new Map(), now = performance.now() / 1000;
    this.flashes = this.flashes.filter(f => f.until > now);
    for (const f of this.flashes) for (const k of f.frames) lit.set(k, f);
    g.lineWidth = 1 * dpr;
    for (let k = 0; k < n; k++) {
      if (!at[k] || (k % every && !lit.has(k))) continue;
      const f = lit.get(k), o = k * POSE;
      const corner = (x, y) => proj(...[0, 1, 2].map(i => P[o + 9 + i] + scale * (P[o + i] * x * 0.8 + P[o + 3 + i] * y * 0.45 + P[o + 6 + i])));
      const cs = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
      if (cs.some(c => !c)) continue;
      g.globalAlpha = f ? 0.35 + 0.65 * Math.max(0, (f.until - now) / f.dur) : 0.7 * alpha(S[k]);
      g.strokeStyle = f ? f.color : segColor(S[k]);
      g.beginPath();
      for (const c of cs) { g.moveTo(at[k][0], at[k][1]); g.lineTo(c[0], c[1]); }
      g.moveTo(cs[3][0], cs[3][1]);
      for (const c of cs) g.lineTo(c[0], c[1]);
      g.stroke();
    }
    // the loops, as arcs between the two cameras, fading
    this.arcs = this.arcs.filter(a => a.until > now);
    g.lineWidth = 1.2 * dpr;
    for (const { a, b, until, color } of this.arcs) {
      const pa = at[a], pb = at[b];
      if (!pa || !pb) continue;
      g.globalAlpha = Math.min(1, until - now);
      g.strokeStyle = color ?? '#e879f9';
      const mx = (pa[0] + pb[0]) / 2, my = (pa[1] + pb[1]) / 2 - 0.25 * Math.hypot(pb[0] - pa[0], pb[1] - pa[1]);
      g.beginPath(); g.moveTo(pa[0], pa[1]); g.quadraticCurveTo(mx, my, pb[0], pb[1]); g.stroke();
    }
    g.globalAlpha = 1;
    this._counts();
  }

  _counts() {
    const S = this.state.segments, pieces = new Set();
    let posed = 0;
    for (const s of S) if (s >= 0) { posed++; pieces.add(s); }
    const n = this.images.length || S.length;
    const c = this.counts, pts = this.points?.live;
    this.countsEl.innerHTML = [
      `<b>${posed}</b>/${n} keyframes placed`, `<b>${pieces.size}</b> piece${pieces.size === 1 ? '' : 's'}`,
      pts !== undefined ? `<b>${pts.toLocaleString()}</b> points` : '', `<b>${c.ba}</b> window BAs`,
      c.loops ? `<b>${c.loops}</b> loops` : '', c.merges ? `<b>${c.merges}</b> merges` : '', c.joins ? `<b>${c.joins}</b> turns joined` : '',
      `<span class="th-dim">event ${Math.max(0, this.cursor + 1)}/${this.events.length}</span>`,
    ].filter(Boolean).join('<br>');
  }

  // orbit (drag), dolly (wheel), pan (shift- or right-drag); double-click frames the cameras again
  _mouse() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    c.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 }; c.setPointerCapture(e.pointerId); });
    c.addEventListener('pointerup', () => { drag = null; });
    c.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y, v = this.view;
      drag.x = e.clientX; drag.y = e.clientY;
      this.userMoved = true;
      if (!drag.pan) { v.yaw -= dx * 0.008; v.pitch = Math.max(-1.5, Math.min(1.5, v.pitch + dy * 0.008)); return; }
      const proj = this._projector(), p0 = proj(...v.target);
      if (!p0) return;
      const k = v.dist * 2 * Math.tan(v.fov / 2) / c.clientHeight, u = normalize(v.up), f = normalize(v.target.map((t, i) => t - this.eye[i])), r = normalize(cross(f, u)), up = cross(r, f);
      for (let i = 0; i < 3; i++) v.target[i] += k * (-dx * r[i] + dy * up[i]);
    });
    c.addEventListener('wheel', (e) => { e.preventDefault(); this.userMoved = true; this.view.dist *= Math.exp(e.deltaY * 0.0015); }, { passive: false });
    c.addEventListener('dblclick', () => { this.userMoved = false; });
  }
}
