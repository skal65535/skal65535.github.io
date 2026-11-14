// trace.js — A record of the mapping as it happens, for the SfM theatre (ui/theatre.js): its events in order
// (a keyframe registered, a window BA, a segment ended, a loop closed, two pieces merged or joined, a stage begun, a
// part of the retriangulation), each with the poses it changed. The poses are diffed against the last ones recorded
// after every event, so a change is caught wherever it came from; the event only says why. The points are sampled at
// checkpoints (every pointsEvery registrations, each stage's start, each part of the retriangulation and the end), at
// most maxPoints of them. Past maxEvents
// nothing more is recorded (full: true).
// The mapper calls record(mapper, kind, info) through its `trace` option (SequentialMapper.trace); null: none.
// An event: { kind, t (ms since the first), info, frames: Int32Array (the keyframes whose pose or segment changed),
// poses: Float32Array (12 a frame: R row-major, then the centre; NaN: no pose now), segments: Int32Array (-1: none),
// points?: { X: Float32Array (3 a point), segment: Int32Array, live (the map's points, all) }, images? (the first
// event's, and a reset's: [{ name, time }] a keyframe), reset?: true (the keyframes renumbered, e.g. one
// inserted by a repair: every pose is in this event, the earlier ones' indices are the old ones) }
"use strict";

import { poseCenter } from './camera.js';

const POSE = 12, NONE = [];
const CHECKPOINTS = new Set(['stage', 'segment', 'merge', 'join', 'posegraph', 'tracks', 'end']);

export class SfmTrace {
  constructor({ maxPoints = 8000, pointsEvery = 100, maxEvents = 200000, onEvent = null } = {}) {
    Object.assign(this, { maxPoints, pointsEvery, maxEvents, onEvent });
    this.events = [];
    this.full = false;
    this.t0 = null;
    this.n = 0;
    this.last = new Float32Array(0);  // the poses as last recorded
    this.lastSeg = new Int32Array(0);
    this.images = null;  // the keyframe list last recorded (the mapper's array), names: their names
    this.names = null;
    this.sinceCheckpoint = 0;
  }

  // A trace that goes on from recorded events (a project's, reopened): its next event holds only what changed since
  // the last of them
  static from(events, options = {}) {
    const tr = new SfmTrace(options);
    tr.events = events.slice();
    const last = events.at(-1);
    tr.t0 = performance.now() - (last ? last.t + 1 : 0);
    let names = null;
    for (const e of events) if (e.images) names = e.images.map(im => im.name);
    if (names) {
      const { poses, segments } = stateAt(events);
      tr._resize(names.length);
      tr.last.set(poses.subarray(0, Math.min(poses.length, tr.last.length)));
      tr.lastSeg.set(segments.subarray(0, Math.min(segments.length, tr.lastSeg.length)));
      tr.names = names;
    }
    return tr;
  }

  // the mapper's `trace` option
  get hook() { return (mapper, kind, info) => this.record(mapper, kind, info); }

  // other keyframes than those last recorded (a reindex: inserted, dropped; whatever their count)
  _renumbered(mapper) {
    const ims = mapper.images ?? NONE;
    if (ims === this.images) return false;
    const same = this.names?.length === ims.length && ims.every((im, k) => im.name === this.names[k]);
    this.images = ims;
    this.names = ims.map(im => im.name);
    return !same;
  }

  record(mapper, kind, info = {}) {
    if (this.events.length >= this.maxEvents) { this.full = true; return null; }
    const now = performance.now();
    this.t0 ??= now;
    const renumbered = this._renumbered(mapper) || mapper.n !== this.n || !this.events.length, reset = renumbered && this.events.length > 0;
    if (renumbered) this._resize(mapper.n);  // every frame compared afresh
    const frames = [], poses = [], segments = [], cur = new Float32Array(POSE);
    for (let k = 0; k < this.n; k++) {
      const P = mapper.poses[k], seg = P ? mapper.segment[k] : -1;
      if (P) poseRow(P, cur); else cur.fill(NaN);
      const at = k * POSE;
      let same = seg === this.lastSeg[k];
      for (let c = 0; same && c < POSE; c++) same = Object.is(cur[c], this.last[at + c]);
      if (same) continue;
      this.last.set(cur, at);
      this.lastSeg[k] = seg;
      frames.push(k); segments.push(seg);
      for (let c = 0; c < POSE; c++) poses.push(cur[c]);
    }
    const ev = { kind, t: now - this.t0, info, frames: Int32Array.from(frames), poses: Float32Array.from(poses), segments: Int32Array.from(segments) };
    if (reset) ev.reset = true;
    if (renumbered) ev.images = (mapper.images ?? []).map(im => ({ name: im.name, time: im.time ?? null }));
    if (kind === 'register') this.sinceCheckpoint++;
    if (CHECKPOINTS.has(kind) || this.sinceCheckpoint >= this.pointsEvery) {
      ev.points = this.samplePoints(mapper);
      this.sinceCheckpoint = 0;
    }
    this.events.push(ev);
    this.onEvent?.(ev);
    return ev;
  }

  // The live points whose id is a multiple of step (a power of 2, the fewest ids over maxPoints: the same points from
  // one checkpoint to the next whatever is deleted, and half of them when the map doubles), with the segment of
  // their first observer
  samplePoints(mapper) {
    let live = 0;
    for (const P of mapper.points) if (P) live++;
    let step = 1;
    while (mapper.points.length / step > this.maxPoints) step *= 2;
    const X = [], segment = [];
    for (let pid = 0; pid < mapper.points.length; pid += step) {
      const P = mapper.points[pid];
      if (!P) continue;
      X.push(P.X[0], P.X[1], P.X[2]);
      segment.push(mapper.segment[P.obs.keys().next().value] ?? -1);
    }
    return { X: Float32Array.from(X), segment: Int32Array.from(segment), live };
  }

  _resize(n) {
    this.n = n;
    this.last = new Float32Array(n * POSE).fill(NaN);
    this.lastSeg = new Int32Array(n).fill(-1);
  }
}

// R row-major and the centre, into out
function poseRow(P, out) {
  for (let c = 0; c < 9; c++) out[c] = P.R[c];
  out.set(poseCenter(P), 9);
}

// The poses and segments of every keyframe after events[0..upto] (the theatre's state at an event): { poses:
// Float32Array(n * 12), segments: Int32Array(n) }, n the largest frame index seen + 1 since the last reset
export function stateAt(events, upto = events.length - 1) {
  let from = 0;
  for (let e = upto; e > 0 && !from; e--) if (events[e].reset) from = e;
  let n = 0;
  for (let e = from; e <= upto; e++) { const f = events[e].frames; if (f.length) n = Math.max(n, f[f.length - 1] + 1); }
  const poses = new Float32Array(n * POSE).fill(NaN), segments = new Int32Array(n).fill(-1);
  for (let e = from; e <= upto; e++) {
    const { frames, poses: p, segments: s } = events[e];
    for (let m = 0; m < frames.length; m++) {
      poses.set(p.subarray(m * POSE, (m + 1) * POSE), frames[m] * POSE);
      segments[frames[m]] = s[m];
    }
  }
  return { poses, segments };
}

// The events as one buffer (a project's cache/trace.bin): a JSON head (meta, e.g. the model the trace ends at, and
// the events without their arrays, each array's place), then the arrays, each 4-byte aligned
const ARRAYS = [['frames', Int32Array], ['poses', Float32Array], ['segments', Int32Array]];
export function encodeTrace(events, meta = {}) {
  const parts = [];
  let at = 0;
  const place = (a) => { const r = [at, a.length]; parts.push(a); at += a.byteLength; return r; };
  const head = events.map(({ frames, poses, segments, points, ...rest }) => ({
    ...rest, frames: place(frames), poses: place(poses), segments: place(segments),
    ...(points ? { points: { X: place(points.X), segment: place(points.segment), live: points.live } } : {}),
  }));
  const json = new TextEncoder().encode(JSON.stringify({ format: 'gaussopt-trace/1', meta, events: head }));
  const pad = (4 - (json.length + 4) % 4) % 4, out = new Uint8Array(4 + json.length + pad + at);
  new DataView(out.buffer).setUint32(0, json.length + pad, true);
  out.set(json, 4);
  out.fill(0x20, 4 + json.length, 4 + json.length + pad);
  let o = 4 + json.length + pad;
  for (const a of parts) { out.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), o); o += a.byteLength; }
  return out.buffer;
}

// { meta, events }; throws on anything else than an encodeTrace buffer, whole
export function decodeTrace(buf) {
  const len = new DataView(buf).getUint32(0, true), head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, len)));
  if (head.format !== 'gaussopt-trace/1') throw new Error(`not an SfM trace (format ${head.format})`);
  const base = 4 + len, view = (T, [o, n]) => new T(buf, base + o, n);  // a view past the end throws (a truncated file)
  const events = head.events.map(e => ({
    ...e, ...Object.fromEntries(ARRAYS.map(([k, T]) => [k, view(T, e[k])])),
    ...(e.points ? { points: { X: view(Float32Array, e.points.X), segment: view(Int32Array, e.points.segment), live: e.points.live } } : {}),
  }));
  return { meta: head.meta ?? {}, events };
}
