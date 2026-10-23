// session.js — A repairable reconstruction (DESIGN-sfm §3, §6.3): the keyframes, their verified
// pairs, the mapper state, and an operation log in which every action keeps a snapshot for undo.
// Actions are local (a keyframe range, or two keyframes) and re-run the S2..S4 stages there:
//   refine     window BA over the range and its covisible frames
//   drop       remove keyframes, re-adjust around them
//   retry      re-match the range (wider window, looser ratio), re-register it into the map
//   redo       cut the range into a new segment, re-initialize and map it, merge it back (Sim(3))
//   insert     add keyframes (densify: the worker extracts them from the video), register them
//   forceLoop  match two keyframes, add a 'user-loop' pair, re-run loop closure and refinement
//   rejectLoops drop loop pairs and replay loop closure + refinement from the sequential map
//   setFocal   set (or re-estimate) the focal, optionally fixed, and re-run the reconstruction
//   repair     retry every amber / red stretch a re-match can mend, keeping only what improves the scores
// Keyframes are referred to by name (names sort in time order); indices shift when frames are
// dropped or inserted. Matching goes through `pairSource(todo [[nameA, nameB, type]], { match })`,
// which returns verified pairs [{ a, b, type, matches, F }] (GPU matching of the features in the
// worker, a lookup in the tests; options may carry the action's `rng`). Each action seeds its own
// RNGs, so a sequence of actions is reproducible.
"use strict";

import { makeRng } from '../math.js';
import { mapSequential, finish } from './reconstruct.js';
import { SequentialMapper } from './mapper.js';
import { closeLoops } from './loop.js';
import { keyframeQuality } from './quality.js';

// Key of an unordered keyframe pair by names
export const key = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);
// Pairs [{ i, j }] with the names of their keyframes as a, b
export const withNames = (pairs, images) => pairs.map(p => ({ ...p, a: images[p.i].name, b: images[p.j].name }));

export class SfmSession {
  // input: { camera, images: [{ name, keypoints, time? }] (time order), pairs: [{ i, j, type, matches }] }
  // mapping: mapper / reconstruct options (focal, hfovDeg, sweeps, ...); loops: S3 on / off
  constructor(input, { pairSource = null, mapping = {}, loops = true, window = 10, seed = 1, log = () => {}, progress = () => {} } = {}) {
    this.camera = input.camera;
    this.images = input.images;
    this.pairs = withNames(input.pairs, input.images);
    Object.assign(this, { pairSource, mapping, loopsOn: loops, window, seed, log, progress });
    this.ops = [];
    this.loops = [];             // loop edges by name, with their residual after the pose graph
    this.twoView = new Map();    // quality.js cache: two-view poses by keyframe names, for the current camera and pairs
    this.viewPsnr = null;        // Map(name -> validation PSNR), from the trainer
    this.rejected = new Set();   // loop pairs the user rejected (keys)
    this.afterSequential = null; // replay point for rejectLoop
  }

  // ── state ────────────────────────────────────────────────────────────────────

  _index() { return new Map(this.images.map((im, k) => [im.name, k])); }

  _indexed(pairs = this.pairs) {
    const ix = this._index();
    return pairs.map(p => ({ ...p, i: ix.get(p.a), j: ix.get(p.b) })).filter(p => p.i !== undefined && p.j !== undefined)
      .map(p => (p.i < p.j ? p : { ...p, i: p.j, j: p.i, a: p.b, b: p.a, matches: swapMatches(p.matches) }));
  }

  _input() { return { camera: this.camera, images: this.images, pairs: this._indexed() }; }
  _options() { return { ...this.mapping, loops: this.loopsOn, seed: this.seed, log: this.log, progress: this.progress }; }

  _loopsByName(stats) { return withNames(stats?.loops ?? [], this.images); }

  // Replace pairs with the same keyframes, add the others (verified pairs [{ a, b, type, matches }])
  _setPairs(found) {
    const keys = new Set(found.map(p => key(p.a, p.b)));
    this.pairs = [...this.pairs.filter(p => !keys.has(key(p.a, p.b))), ...found];
    for (const k of keys) this.twoView.delete(k);
  }

  // S2..S4 from scratch. proposeLoops(mapper) (optional): more verified pairs [{ i, j, type,
  // matches }] once the sequential map exists (geometric loop candidates, pipeline.js).
  async reconstruct({ proposeLoops = null } = {}) {
    const state = mapSequential(this._input(), this._options());
    this.mapper = state.mapper;
    if (proposeLoops && this.loopsOn) this._setPairs(withNames(await proposeLoops(this.mapper), this.images));
    this.afterSequential = { snapshot: this.mapper.snapshot(), images: this.images };
    const res = finish(this._input(), state, this._options());
    this.loops = this._loopsByName(res.stats.loops);
    return res;
  }

  // checkLoops: relocalize the loop edges on the map (quality.js; a second or so, for what the panel shows)
  quality({ checkLoops = false } = {}) {
    const camera = this.mapper.camera.params.join();
    if (camera !== this.twoViewCamera) { this.twoView.clear(); this.twoViewCamera = camera; }
    const ix = this._index();
    const loops = this.loops.map(l => ({ ...l, i: ix.get(l.a), j: ix.get(l.b) })).filter(l => l.i !== undefined && l.j !== undefined);
    return keyframeQuality(this.mapper, { loops, checkLoops, times: this.images.map((im, k) => im.time ?? k), cache: this.twoView, viewPsnr: this.viewPsnr });
  }

  // Saved state (project.js): keyframe list (without keypoints: the features hold them), pairs,
  // mapper state, loops, options. The undo history is not kept.
  toState() {
    const pairs = this._indexed(), { json: mapper, ...mapperArrays } = this.mapper.toState();
    return {
      json: { images: this.images.map(im => ({ name: im.name, time: im.time })), loops: this.loops, rejected: [...this.rejected],
              mapping: this.mapping, loopsOn: this.loopsOn, window: this.window, seed: this.seed,
              pairTypes: pairs.map(p => p.type), mapper },
      arrays: {
        pairIJ: Int32Array.from(pairs.flatMap(p => [p.i, p.j])),
        pairCount: Uint32Array.from(pairs.map(p => p.matches.length)),
        pairMatches: concat(pairs.map(p => p.matches)),
        ...mapperArrays,
      },
    };
  }

  // A session from toState(): keypoints of each keyframe come from `keypoints(name)`
  static fromState({ json, arrays }, keypoints, { pairSource = null, log = () => {}, progress = () => {} } = {}) {
    const images = json.images.map(im => ({ ...im, keypoints: keypoints(im.name) }));
    const pairs = [];
    for (let p = 0, o = 0; p < arrays.pairCount.length; p++) {
      const n = arrays.pairCount[p];
      pairs.push({ i: arrays.pairIJ[2 * p], j: arrays.pairIJ[2 * p + 1], type: json.pairTypes[p], matches: arrays.pairMatches.slice(o, o + n) });
      o += n;
    }
    const s = new SfmSession({ camera: json.mapper.camera, images, pairs },
                             { pairSource, mapping: json.mapping, loops: json.loopsOn, window: json.window, seed: json.seed, log, progress });
    s.mapper = new SequentialMapper(s._input(), { ...json.mapping, log, progress, pairTypes: [...new Set(json.pairTypes), 'seq'] });
    s.mapper.setState({ json: json.mapper, ...arrays });
    s.loops = json.loops;
    s.rejected = new Set(json.rejected);
    return s;
  }

  model() { return this.mapper.model(); }

  // ── operation log ────────────────────────────────────────────────────────────

  // Runs an action with a snapshot for undo; on failure the state is restored. Returns
  // { op, ...the action's result, before, after (score counts over the affected frames), quality }
  async _action(op, args, fn) {
    if (!this.mapper) throw new Error('no reconstruction yet');
    const snapshot = { mapper: this.mapper.snapshot(), images: this.images, pairs: this.pairs, loops: this.loops,
                       rejected: new Set(this.rejected), afterSequential: this.afterSequential, mapping: this.mapping };
    const before = this.quality();
    this.mapper.rng = makeRng(this.seed + 7919 * (this.ops.length + 1));
    const t = performance.now();
    let result;
    try {
      result = await fn();
    } catch (e) {
      this._restore(snapshot);
      throw e;
    }
    this.ops.push({ op, args, snapshot, seconds: (performance.now() - t) / 1000 });
    const after = this.quality();
    const affected = new Set(result.frames ?? []);
    const count = (q) => {
      const c = { green: 0, amber: 0, red: 0, missing: 0 };
      const byName = new Map(q.frames.map(f => [f.name, f]));
      for (const n of affected) byName.has(n) ? c[byName.get(n).score]++ : c.missing++;
      return c;
    };
    return { op, ...result, before: count(before), after: count(after), quality: after };
  }

  _restore(s) {
    this.mapper.restore(s.mapper);
    Object.assign(this, { images: s.images, pairs: s.pairs, loops: s.loops, rejected: s.rejected, afterSequential: s.afterSequential, mapping: s.mapping });
    this.twoView.clear();  // the pairs were others
  }

  // Undo the last action: returns its name, or null when there is none
  undo() {
    const last = this.ops.pop();
    if (!last) return null;
    this._restore(last.snapshot);
    return last.op;
  }

  // ── helpers for the actions ──────────────────────────────────────────────────

  _indices(names) {
    const ix = this._index();
    return [...new Set(names)].map(n => {
      if (!ix.has(n)) throw new Error(`unknown keyframe ${n}`);
      return ix.get(n);
    }).sort((a, b) => a - b);
  }

  // The segment the range belongs to: the most common one among registered frames that share
  // pairs with it from outside; -1 when nothing outside is registered
  _segmentAround(range) {
    const inRange = new Set(range), votes = new Map();
    for (const k of range) {
      for (const e of this.mapper.adj[k]) {
        if (inRange.has(e.other) || !this.mapper.poses[e.other]) continue;
        const s = this.mapper.segment[e.other];
        votes.set(s, (votes.get(s) ?? 0) + e.a.length);
      }
    }
    return [...votes].sort((a, b) => b[1] - a[1])[0]?.[0] ?? -1;
  }

  // Window BA over the range, its time neighbours and its covisible frames, then new tracks
  _adjust(range, { intrinsics = false, margin = this.window >> 1 } = {}) {
    const m = this.mapper;
    const lo = Math.max(0, range[0] - margin), hi = Math.min(m.n - 1, range.at(-1) + margin);
    const core = [];
    for (let k = lo; k <= hi; k++) if (m.poses[k]) core.push(k);
    if (!core.length) return;
    const segs = new Set(core.map(k => m.segment[k]));
    for (const seg of segs) {
      const frames = core.filter(k => m.segment[k] === seg);
      m.localBA(frames[0], { frames: [...frames, ...m.partners(frames, { minGap: this.window })], refineIntrinsics: intrinsics, iterations: 20 });
      for (const k of frames) m.triangulateNew(k);
    }
  }

  // Pairs of each frame in `range` with the frames within `window` (by time), as names
  _neighbourTodo(range, window) {
    const todo = new Map();
    for (const k of range) {
      for (let j = Math.max(0, k - window); j <= Math.min(this.mapper.n - 1, k + window); j++) {
        if (j === k) continue;
        const [a, b] = [this.images[Math.min(j, k)].name, this.images[Math.max(j, k)].name];
        if (!this.rejected.has(key(a, b))) todo.set(key(a, b), [a, b, 'seq']);
      }
    }
    return [...todo.values()];
  }

  async _match(todo, options = {}) {
    if (!this.pairSource) throw new Error('this action needs the keyframe features (pairSource)');
    return todo.length ? this.pairSource(todo, { ...options, rng: makeRng(this.seed + 104729 * (this.ops.length + 1)) }) : [];
  }

  // Segment merges over all pairs (Sim(3) + track fusion), the rest of the map untouched
  _merge() {
    const pairs = this._indexed();
    const stats = closeLoops(this.mapper, pairs, { rng: this.mapper.rng, log: this.log, mergeOnly: true });
    this.mapper.addPairs(pairs);
    return stats;
  }

  // S3 + S4 on the current state: loop closure over all pairs (the pose graph moves the whole
  // segment), then the refinement sweeps and retriangulation that clean up after it
  _closeAndRefine() {
    const res = finish(this._input(), { mapper: this.mapper, timings: {} }, this._options());
    this.loops = this._loopsByName(res.stats.loops);
  }

  // ── actions (DESIGN-sfm §6.3) ────────────────────────────────────────────────

  refine(names, { intrinsics = false } = {}) {
    return this._action('refine', { names, intrinsics }, () => {
      this._adjust(this._indices(names), { intrinsics, margin: 0 });
      return { frames: names };
    });
  }

  drop(names) {
    return this._action('drop', { names }, () => {
      const gone = new Set(names), old = this._indices(names);
      const keep = this.images.map((_, k) => k).filter(k => !gone.has(this.images[k].name));
      this.images = keep.map(k => this.images[k]);
      this.pairs = this.pairs.filter(p => !gone.has(p.a) && !gone.has(p.b));
      this.loops = this.loops.filter(l => !gone.has(l.a) && !gone.has(l.b));
      this.mapper.reindex(this.images, keep);
      // re-adjust around each dropped stretch (its position in the new list)
      const around = [...new Set(old.map(o => keep.findIndex(k => k > o)).map(k => (k < 0 ? this.images.length - 1 : k)))];
      for (const k of around) this._adjust([k]);
      return { frames: [...new Set(around.flatMap(k => [k - 1, k].filter(x => x >= 0)))].map(k => this.images[k].name), dropped: names.length };
    });
  }

  retry(names, { window = 2 * this.window, maxRatio = 0.9 } = {}) {
    return this._action('retry', { names, window, maxRatio }, async () => {
      const range = this._indices(names);
      const found = await this._match(this._neighbourTodo(range, window), { match: { maxRatio } });
      this._setPairs(found);
      this.mapper.replacePairs(this._indexed(found));
      const seg = this._segmentAround(range);
      if (seg < 0) throw new Error('retry: no registered keyframe next to the range (use redo)');
      this.mapper.unregister(range);
      const registered = this.mapper.grow(seg, new Set(range));
      this._adjust(range);
      return { frames: names, registered, pairs: found.length };
    });
  }

  // Retry of every stretch of amber / red keyframes a re-match can mend (not rotation-only frames,
  // whose poses are right, nor separate maps, which need redo), each kept only when the scores
  // improve (fewer red, or as many and more green) and undone otherwise. One undo reverts it all.
  repair({ passes = 1 } = {}) {
    return this._action('repair', { passes }, async () => {
      const ops = this.ops.length, frames = new Set(), mendable = (f) => f.score !== 'green' && !/rotation only|separate map/.test(f.reason);
      let counts = this.quality().counts, tried = 0, kept = 0;
      for (let pass = 0; pass < passes; pass++) {
        const runs = [];
        let run = null;
        for (const f of this.quality().frames) {
          if (!mendable(f)) run = null;
          else (run ?? (runs.push(run = []), run)).push(f.name);
        }
        let keptNow = 0;
        for (const [k, names] of runs.entries()) {
          this.progress('repair', k, runs.length);
          tried++;
          const what = `${names[0]}${names.length > 1 ? ` +${names.length - 1}` : ''}`;
          let r;
          try {
            r = await this.retry(names);
          } catch (e) {
            this.log(`repair: ${what}: ${e.message}`);
            continue;
          }
          const c = r.quality.counts, better = c.red < counts.red || (c.red === counts.red && c.green > counts.green);
          this.log(`repair: ${what}: ${JSON.stringify(counts)} -> ${JSON.stringify(c)}, ${better ? 'kept' : 'undone'}`);
          if (!better) { this.undo(); continue; }
          counts = c;
          keptNow++;
          for (const n of names) frames.add(n);
        }
        this.progress('repair', runs.length, runs.length);
        kept += keptNow;
        if (!keptNow) break;  // the next pass would try the same stretches on the same map
      }
      this.ops.length = ops;  // the kept retries are one action: undo goes back to before them all
      return { frames: [...frames], tried, kept };
    });
  }

  // A weak link inside the range splits it: new maps are started until no pair is left, all are
  // merged by loop closure, and what is still unregistered is registered into its neighbours' map
  redo(names) {
    return this._action('redo', { names }, () => {
      const range = this._indices(names), m = this.mapper;
      m.unregister(range);
      const left = new Set(range);
      let maps = 0;
      while (left.size >= 2 && m.initialize(left)) {
        m.grow(m.nextSegmentId - 1, left);
        for (const k of [...left]) if (m.poses[k]) left.delete(k);
        maps++;
      }
      if (!maps) throw new Error('redo: no keyframe pair of the range has the parallax to start a map (select a longer range)');
      const stats = this._merge();
      if (left.size) {
        const seg = this._segmentAround([...left]);
        if (seg >= 0) m.grow(seg, left);
      }
      this._adjust(range);
      const registered = range.filter(k => m.poses[k]).length;
      return { frames: names, registered, maps, merges: stats.merges, segment: m.segment[range[0]] };
    });
  }

  // images: [{ name, keypoints, time? }] not in the session yet (their features known to pairSource)
  insert(images) {
    return this._action('insert', { names: images.map(im => im.name) }, async () => {
      const ix = this._index();
      for (const im of images) if (ix.has(im.name)) throw new Error(`insert: ${im.name} is already a keyframe`);
      const merged = [...this.images, ...images].sort((p, q) => (p.name < q.name ? -1 : p.name > q.name ? 1 : 0));
      this.images = merged;
      this.mapper.reindex(merged, merged.map(im => (ix.has(im.name) ? ix.get(im.name) : -1)));
      const added = this._indices(images.map(im => im.name));
      const found = await this._match(this._neighbourTodo(added, this.window));
      this._setPairs(found);
      this.mapper.addPairs(this._indexed(found));
      const seg = this._segmentAround(added);
      if (seg < 0) throw new Error('insert: the new keyframes match no registered keyframe');
      const registered = this.mapper.grow(seg, new Set(added));
      this._adjust(added);
      return { frames: images.map(im => im.name), registered, pairs: found.length };
    });
  }

  forceLoop(a, b, { maxRatio = 0.9 } = {}) {
    return this._action('forceLoop', { a, b }, async () => {
      if (a === b) throw new Error('force loop: pick two different keyframes');
      const [lo, hi] = a < b ? [a, b] : [b, a];
      this.rejected.delete(key(lo, hi));
      const found = await this._match([[lo, hi, 'user-loop']], { match: { maxRatio } });
      if (!found.length) throw new Error(`force loop: ${lo} and ${hi} do not share enough verified matches`);
      this._setPairs(found);
      this._closeAndRefine();
      const used = this.loops.some(l => key(l.a, l.b) === key(lo, hi));
      const range = this._indices([lo, hi]);
      if (!used && this.mapper.segment[range[0]] === this.mapper.segment[range[1]]) {
        throw new Error(`force loop: ${found[0].matches.length / 2} matches, but no consistent 3D transform between the two views`);
      }
      for (const k of range) this._adjust([k]);
      return { frames: [lo, hi], matches: found[0].matches.length / 2 };
    });
  }

  // Replays S3 + S4 from the sequential map without the pair (local repairs since are lost: the
  // log says so); after drops / inserts, the whole reconstruction is re-run instead
  rejectLoop(a, b) { return this.rejectLoops([[a, b]]); }

  rejectLoops(pairs) {
    return this._action('rejectLoops', { pairs }, async () => {
      const keys = new Set(pairs.map(([a, b]) => key(a, b)));
      for (const k of keys) this.rejected.add(k);
      this.pairs = this.pairs.filter(p => !keys.has(key(p.a, p.b)));
      if (this.afterSequential?.images === this.images) {
        // a copy (snapshot() reads only what a snapshot holds): restore() hands its arrays to the
        // live mapper, and afterSequential stays the replay point (undo keeps a reference to it)
        this.mapper.restore(this.mapper.snapshot.call(this.afterSequential.snapshot));
        this._closeAndRefine();
        this.log('reject loop: loop closure and refinement replayed from the sequential map (later repairs are not kept)');
      } else {
        await this.reconstruct();
        this.log(`reject loop: ${this.afterSequential ? 'keyframes changed since the sequential map' : 'no sequential map in a restored session'}, reconstruction re-run`);
      }
      return { frames: [...new Set(pairs.flat())] };
    });
  }

  // The focal shapes the whole map, so the reconstruction is re-run with it (fixed or estimated);
  // local repairs since the first run are not kept
  setFocal(focal, { fix = true } = {}) {
    return this._action('setFocal', { focal, fix }, async () => {
      this.mapping = { ...this.mapping, focal: focal === 'auto' ? 'auto' : Number(focal),
                       refine: fix && focal !== 'auto' ? ['k1'] : ['f', 'k1'] };
      await this.reconstruct();
      this.log(`focal ${focal === 'auto' ? 'estimated' : `set to ${Number(focal).toFixed(1)} px${fix ? ', fixed' : ''}`}: reconstruction re-run`);
      return { frames: this.images.map(im => im.name), focal: this.mapper.camera.params[0] };
    });
  }
}

function concat(arrays) {
  const out = new Uint32Array(arrays.reduce((s, a) => s + a.length, 0));
  let o = 0;
  for (const a of arrays) { out.set(a, o); o += a.length; }
  return out;
}

function swapMatches(m) {
  const out = new m.constructor(m.length);
  for (let k = 0; k < m.length; k += 2) { out[k] = m[k + 1]; out[k + 1] = m[k]; }
  return out;
}
