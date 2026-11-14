// worker.js — The video -> cameras session (DESIGN-sfm §3, §5) in a Dedicated Worker, so the
// page stays responsive: S0 analysis and keyframe selection, S1 features, S2-S4 reconstruction,
// S5 undistortion and the COLMAP model. Stage results are cached: changing a later option
// re-runs from that stage only. Stop = the page terminates the worker.
// After a run the session (session.js) stays here for repairs (DESIGN-sfm §6.3).
// The analysis, keyframes and model are saved in OPFS per video (restored when it is opened again)
// and as a .gosfm project on request. With a project folder (src/project/store.js, message 'store'),
// the session goes to its cache/ instead, and the results to it as they are made: images/ (the
// keyframes, as extracted), sharpness.csv, sparse/0 (the distorted model), gs/ (the trainer's
// dataset). 'flush' writes everything the session has into a newly attached folder.
//   in:  { type: 'open', file } | { type: 'open', files (several videos of one place) } | { type: 'photos', files } | { type: 'project', file } | { type: 'run', options (until: 'keyframes' stops there) }
//        | { type: 'action', op, args }
//        | { type: 'undo' } | { type: 'send' } | { type: 'save' } | { type: 'psnr', values: [[name, dB]] }
//        | { type: 'photoFilter', filter: { lens?, zoom?, orientation?: [values] } | null }
//        | { type: 'store', store (describe()) | null } | { type: 'flush' } | { type: 'restoreStore' }
//   out: { type: 'video', info } | { type: 'log', text, level } | { type: 'progress', stage, done, total }
//        | { type: 'selection', ... } | { type: 'keyframes', count } | { type: 'thumbs', frames: [{ name, time, blob }], aspect (w / h),
//            append (true: the keyframes extracted since the last, the sheet's kept) }
//        | { type: 'quality', quality, action }
//        | { type: 'result', sparse, images, masks, summary }
//        | { type: 'restored' } | { type: 'project', blob, name } | { type: 'idle', status } | { type: 'error', message }
//        | { type: 'flushed', files: [[path, Blob]] (a memory store's) }
// Messages are handled one at a time, in order.
"use strict";

import { openVideo, sampleFrames, decodeFrames, grayOf, liftGray } from './video.js';
import { fitSize, resizeArea, selectFrames, zoomEvents, flowChains, zoomCameras, zoomPriors } from './frames.js';
import { OVERLAY, staticMask, dropMasked, maskAt, packMask, unpackMask } from './overlay.js';
import { GpuSift, SIFT_DEFAULTS, rootSift } from './sift.js';
import { GpuMatcher } from './match.js';
import { GpuBundleAdjuster } from './ba_gpu.js';
import { GpuUndistort, PointColors } from './undistort.js';
import { reconstructFromFeatures, featurePairSource } from './pipeline.js';
import { SfmTrace, encodeTrace, decodeTrace } from './trace.js';
import { SfmSession } from './session.js';
import { packProject, packPieces, unpackProject, mergeProjects, projectPart } from './project.js';
import { SESSION_DEFAULTS, FEATURE_KEYS, CAMERA_KEYS } from './defaults.js';
import { readExif, jpegSize } from './exif.js';
import { planPhotos, planMore, photoCameras, photoMapping, baseMapping } from './photos.js';
import { coverage, coverageAdvice } from './coverage.js';
import { datasetFromMapper, writeColmapModel } from './export.js';
import { xyOnly, mapperKeypoints, adaptFeatures } from './frontend.js';
import { PointTracker, packTrackPairs, unpackTrackPairs } from './track.js';
import { seconds } from './util.js';
import { hashString, makeRng } from '../math.js';
import { storeFrom } from '../project/store.js';

const post = (msg, transfer = []) => self.postMessage(msg, transfer);
const log = (text, level = '') => post({ type: 'log', text, level });
// debug: the GPU buffers and textures held, in the log every 10 s (the page's ?sfm_gpuLedger: the worker's ?gpuLedger)
if (new URLSearchParams(self.location.search).has('gpuLedger')) import('./gpu_ledger.js').then(m => m.gpuLedger(log));
const progress = (stage, done, total) => post({ type: 'progress', stage, done, total });
const same = (a, b, keys) => a && keys.every(k => a[k] === b[k]);
const perFrame = (feats) => Math.round(feats.reduce((s, f) => s + f.keypoints.length / 4, 0) / feats.length);  // mean features per frame
const newId = () => `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
const warnCapped = (capped, what) => {
  if (capped) log(`SIFT: ${capped} ${what}(s) had over ${SIFT_DEFAULTS.maxKeypoints} candidate keypoints: the ones kept depend on the GPU's order`, 'warn');
};

let video = null, gpu = null, current = null;  // current: { session, features, o, version } after a run
const keyframeName = (time) => `t${String(Math.round(time * 1000)).padStart(7, '0')}.jpg`;  // DESIGN-sfm §3
const cache = {};  // analysis, features: { options (that produced them), id, ... }; source: the video of a restored project

// Several videos of one place (open with files): per video { video, offset, and its own CLIP_KEYS and features (one
// video's keyframes, as keyframes() makes them) }, null: one video. Each is analysed, its keyframes picked, extracted,
// matched in sequence and tracked as a video alone; then the keyframes are one list (clipKeyframes), a video's
// `clip`, its times and names `offset` s on (its first frame CLIP_GAP s after the last of the one before), no sequential pair between
// two. useClip(c): the c-th is the one decoded (video), its keys in the cache.
let clips = null, clipAt = -1, offset = 0;
const CLIP_GAP = 10, CLIP_KEYS = ['analysis', 'motion', 'zoom', 'selection'];
function useClip(c) {
  if (clipAt === c) return;
  if (clipAt >= 0) for (const k of CLIP_KEYS) clips[clipAt][k] = cache[k];
  for (const k of CLIP_KEYS) cache[k] = clips[c][k];
  clipAt = c; video = clips[c].video; offset = clips[c].offset;
}
// a video's first and last frame times (its samples' pts: a cut may start before 0)
const spanOf = (v) => v.samples.reduce(([lo, hi], s) => [Math.min(lo, s.pts / v.timescale), Math.max(hi, s.pts / v.timescale)], [Infinity, -Infinity]);
// a video's key, in use or not
const clipCache = (c, k) => (c === clipAt ? cache[k] : clips[c][k]);
// the video whose keyframes a time (offset) is of
const clipOfTime = (t) => clips.findLastIndex(c => c.offset <= t + 1e-6);

function reset() {
  for (const k of Object.keys(cache)) delete cache[k];
  for (const c of clips ?? []) for (const k of [...CLIP_KEYS, 'features']) delete c[k];
  thumbs.clear();
  current = null;
}
// reset(), and nothing open: no video, photo set or OPFS key (a failed restore() keeps them)
function resetAll() {
  reset();
  video = storeKey = photos = clips = null;
  clipAt = -1; offset = 0;
}

async function devices() {
  if (gpu) return gpu;
  const sift = await GpuSift.create();
  gpu = { sift, matcher: await GpuMatcher.create(sift.device), undistort: await GpuUndistort.create(sift.device) };
  return gpu;
}

// Once: the hardware decoder failed and the video is decoded in software (video.js decodeFrames)
function noteDecoder() {
  if (video?.decoderFallback && !video.decoderNoted) {
    video.decoderNoted = true;
    log(`video: the hardware decoder failed (${video.decoderFallback}), decoding in software`, 'warn');
  }
}

// S0a pool: analysis workers (frames_worker.js), CHUNK consecutive frames per job
const CHUNK = 48;
let pool = null;
function analysisPool() {
  if (pool) return pool;
  const n = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 2));
  pool = Array.from({ length: n }, () => {
    const w = new Worker(new URL('./frames_worker.js', import.meta.url), { type: 'module' });
    w.busy = null;
    return w;
  });
  return pool;
}

// S0a: sharpness and motion of every sampled frame (one decode pass at <= sharpDim), the
// analysis spread over the pool while the decoder runs. With an on-screen display (overlay.js), its motion
// again in a second pass, its corners left out (masked; DESIGN-sfm §5.1.1): they do not move, and where they
// outnumber the scene's (thermal asphalt, a wall) the median motion is theirs, 0, and no keyframe is taken.
async function analyze(o) {
  const keys = ['sampleFps', 'sharpDim', 'flowDim'];
  if (same(cache.analysis?.options, o, keys)) return cache.analysis;
  const t = performance.now(), order = sampleFrames(video, o.sampleFps), [w, h] = fitSize(video.width, video.height, o.sharpDim);
  // the on-screen display's mask (overlay.js): every `every`-th frame at the grid's size
  const [gw, gh] = fitSize(w, h, OVERLAY.gridDim), every = Math.max(1, Math.ceil(order.length / OVERLAY.maxFrames)), grid = [];
  const { sharp, steps, times, zoom, flowH } = await analysisPass(order, w, h, o, null, (gray, i) => {
    if (i % every === 0) grid.push(gw === w && gh === h ? gray.slice() : resizeArea(gray, w, h, gw, gh));
  });
  log(`frames: ${order.length} sampled at ${Math.min(o.sampleFps, video.fps).toFixed(1)} fps, ` +
      `analyzed in ${seconds(t)} (${analysisPool().length} workers)`);
  const t1 = performance.now(), overlay = packMask(staticMask(grid, gw, gh));
  log(overlay.b64 ? `overlay: ${(100 * overlay.share).toFixed(1)}% of the frame masked (${overlay.regions} still regions with text or lines, ${grid.length} frames at ${gw} x ${gh}, ${seconds(t1)})`
                  : `overlay: none (${overlay.why})`);
  let masked = null;
  if (overlay.b64) {
    const t2 = performance.now(), m = await analysisPass(order, w, h, o, unpackMask(overlay));
    masked = { steps: m.steps, zoom: m.zoom, flowH: m.flowH };
    const path = (a) => a.reduce((p, v) => p + (Number.isFinite(v) ? v : 0), 0).toFixed(2);
    log(`overlay: the motion without its corners, ${path(m.steps)} diagonals in all (${path(steps)} with them), in ${seconds(t2)}`);
  }
  return cache.analysis = { options: { ...o }, order, sharp, steps, times, zoom, flowH, overlay, masked };
}

// One decode pass of the sampled frames `order` at w x h through the pool: { sharp, steps, times, zoom, flowH } per
// frame; mask: no corner tracked on it (frames_worker.js); onGray(gray, i): each frame's gray levels as decoded
async function analysisPass(order, w, h, o, mask, onGray = () => {}) {
  const sharp = new Array(order.length), steps = new Array(order.length), times = new Array(order.length), zoom = new Array(order.length).fill(null), flowH = new Array(order.length).fill(null);
  const workers = analysisPool(), options = { sharpDim: o.sharpDim, flowDim: o.flowDim };
  const position = new Map(order.map((k, i) => [k, i]));  // sample -> index in order
  let chunk = [], at = [], last = null, done = 0, failure = null;
  const free = () => workers.find(wk => !wk.busy) ?? Promise.race(workers.map(wk => wk.busy)).then(free);
  const submit = async (frames, at, overlap) => {
    const wk = await free();
    wk.busy = new Promise((resolve) => {
      wk.onmessage = ({ data }) => {
        data.sharpness.forEach((v, k) => { sharp[at[k]] = v; steps[at[k]] = data.motion[k]; zoom[at[k]] = data.zoom[k]; flowH[at[k]] = data.flowH[k]; });
        done += data.sharpness.length;
        progress('analyze', done, order.length);
        wk.busy = null;
        resolve();
      };
      wk.onerror = (e) => { failure = new Error(`frame analysis: ${e.message}`); wk.busy = null; resolve(); };
    });
    wk.postMessage({ frames, w, h, options, overlap, mask }, frames.map(f => f.buffer).filter((b, k) => k > 0 || !overlap));
  };
  await decodeFrames(video, new Set(order), w, h, async (ctx, k, time) => {
    const gray = grayOf(ctx.getImageData(0, 0, w, h).data, w * h);
    times[position.get(k)] = time;
    onGray(gray, position.get(k));
    chunk.push(gray);
    at.push(position.get(k));
    if (chunk.length === CHUNK) {
      const frames = last ? [last, ...chunk] : chunk;
      last = chunk.at(-1).slice();  // the next chunk's overlap frame (the original is transferred)
      await submit(frames, at, frames !== chunk);
      chunk = []; at = [];
    }
    if (failure) throw failure;
  });
  noteDecoder();
  if (chunk.length) await submit(last ? [last, ...chunk] : chunk, at, !!last);
  await Promise.all(workers.map(wk => wk.busy));
  if (failure) throw failure;
  const missing = order.length - times.filter(t => t !== undefined).length;
  if (missing) throw new Error(`${missing} of ${order.length} sampled frames not decoded (duplicate timestamps in the video?)`);
  return { sharp, steps, times, zoom, flowH };
}

// The analysis's steps for options o: without the on-screen display's corners when it is masked (and measured so:
// an analysis saved before has none)
const motionOf = (a, o) => (o.overlay !== false && a.masked ? { ...a, ...a.masked } : a);

// The zoom events of the analysed frames (frames.js zoomEvents; none in an analysis saved without its zoom steps)
function zoomOf(analysis, o) {
  const steps = (analysis.zoom ?? []).map(z => z && { scale: z[0], fit: z[1] });
  const z = zoomEvents(steps, analysis.times, o);
  if (z.events.length) {
    const t = (i) => analysis.times[i].toFixed(2);
    log(`zoom: ${z.events.map(e => `x${e.zoom.toFixed(2)} at ${t(e.from)}-${t(e.to)} s`).join(', ')} (image scale, walking included): ` +
        `no keyframe during them, a camera per stretch between them, its focal found when mapping`);
  }
  return z;
}

// S0b keyframe selection, shown on the panel's strip
function selection(analysis, o) {
  analysis = cache.motion = motionOf(analysis, o);
  const zoom = cache.zoom = zoomOf(analysis, o);
  const sel = selectFrames({ sharpness: analysis.sharp, motion: analysis.steps, times: analysis.times, span: zoom.span }, o);
  if (!clips) post({ type: 'selection', times: analysis.times, sharpness: analysis.sharp, status: sel.status, path: sel.path, threshold: sel.threshold });
  cache.selection = sel;
  save(sharpnessPath(), sharpnessCsv(analysis, sel));
  return sel;
}

// Several videos: their frames on the strip one after the other (selection's, the times offset)
function postClipsSelection() {
  const of = (k, f) => clips.flatMap((_, c) => (clipCache(c, k) ? f(clipCache(c, k), c) : []));
  post({ type: 'selection', times: of('motion', (a, c) => a.times.map(t => t + clips[c].offset)), sharpness: of('motion', a => a.sharp),
         status: of('selection', s => s.status), path: null, threshold: clipCache(0, 'selection')?.threshold ?? 0 });
}

// Every sampled frame (as video2splat.py's sharpness.csv, with times): the name it has as a keyframe; of several
// videos, the c-th's in sharpness-<c>.csv but the first's
const sharpnessPath = () => (clipAt > 0 ? `sharpness-${clipAt}.csv` : 'sharpness.csv');
function sharpnessCsv(a, sel) {
  const rows = a.times.map((t, i) => `${keyframeName(t + offset)},${(t + offset).toFixed(4)},${a.sharp[i].toFixed(2)},${a.steps[i]?.toFixed(5) ?? ''},${sel.status[i]}`);
  return ['frame,time,sharpness,motion,status', ...rows].join('\n') + '\n';
}

// S1 on video samples (decoded at w x h): per frame onFrame({ name, time, keypoints, descriptors, responses }, { name,
// blob (JPEG) }); dense: with denseFeatures at densePeak (adaptFeatures trims them)
async function extract(samples, w, h, o, onFrame, { dense = false } = {}) {
  const { sift } = await devices();
  const ov = overlayMask(o);
  let capped = 0, dropped = 0, total = 0;
  await decodeFrames(video, new Set(samples), w, h, async (ctx, k, at) => {
    const time = at + offset, name = keyframeName(time);
    const gray = grayOf(ctx.getImageData(0, 0, w, h).data, w * h);
    const jpeg = { name, blob: await ctx.canvas.convertToBlob({ type: 'image/jpeg', quality: o.jpegQuality }) };
    const f = await sift.extract(liftGray(gray, o.siftLift, true), w, h, dense ? { maxFeatures: Math.max(o.maxFeatures, o.denseFeatures), peak: Math.min(o.siftPeak, o.densePeak) }
                                                  : { maxFeatures: o.maxFeatures, peak: o.siftPeak });
    if (o.rootSift) f.descriptors = rootSift(f.descriptors);
    capped += f.capped;
    total += f.keypoints.length / 4;
    const { keypoints, descriptors, responses, dropped: d } = ov ? dropMasked(f.keypoints, f.descriptors, ov, w, h, f.responses) : f;
    dropped += d ?? 0;
    await onFrame({ name, time, keypoints, descriptors, responses }, jpeg);
  });
  noteDecoder();
  warnCapped(capped, 'frame');
  if (dropped) log(`overlay: ${dropped} of ${total} keypoints on it dropped (${(100 * dropped / total).toFixed(1)}%)`);
}

// The analysis's on-screen display mask when the option is on (overlay.js; null: none found, off, or a photo set)
const overlayMask = (o, a = cache.analysis) => (o.overlay !== false ? unpackMask(a?.overlay) : null);
// Every video's (several: each one's), the ones found
const overlayMasks = (o) => (clips ? clips.map((_, c) => overlayMask(o, clipCache(c, 'analysis'))) : [overlayMask(o)]).filter(Boolean);

// Small JPEGs of the keyframes for the page's contact sheet (src/ui/frames.js), made once per name
const THUMB_WIDTH = 320, THUMB_DECODES = 4;
const thumbs = new Map();
// A thumbnail's JPEG: bitmap (closed) drawn at [x, y, w, h] on a black w x h canvas (the bitmap's size by default)
async function thumbJpeg(bitmap, w = bitmap.width, h = bitmap.height, at = [0, 0, w, h]) {
  const canvas = new OffscreenCanvas(w, h), ctx = canvas.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(bitmap, ...at);
  bitmap.close();
  return canvas.convertToBlob({ type: 'image/jpeg', quality: 0.8 });
}
// The keyframes' thumbnails as they are extracted (a run's, before the tracks and the matching): a batch at most every
// THUMBS_MS, the first replacing the sheet; start(), then add(name, time, blob) per keyframe, end() the last batch
const THUMBS_MS = 500;
const liveThumbs = {
  on: false, first: true, batch: [], at: 0,
  start() { Object.assign(this, { on: true, first: true, batch: [], at: 0 }); },
  async add(name, time, blob, aspect) {
    if (!this.on) return;
    if (!thumbs.has(name)) thumbs.set(name, await thumbJpeg(await createImageBitmap(blob, { resizeWidth: THUMB_WIDTH, resizeQuality: 'medium' })));
    this.batch.push({ name, time, blob: thumbs.get(name) });
    this.aspect = aspect;
    if (performance.now() - this.at >= THUMBS_MS) this.flush();
  },
  flush() {
    if (this.batch.length) post({ type: 'thumbs', frames: this.batch, aspect: this.aspect, append: !this.first });
    Object.assign(this, { first: false, batch: [], at: performance.now() });
  },
  end() { if (this.on) this.flush(); this.on = false; },
};
async function postThumbs(features) {
  const times = new Map(features.feats.map(f => [f.name, f.time]));
  for (const { name, blob } of features.jpegs) {
    if (thumbs.has(name)) continue;
    thumbs.set(name, await thumbJpeg(await createImageBitmap(blob, { resizeWidth: THUMB_WIDTH, resizeQuality: 'medium' })));
  }
  const frames = features.jpegs.filter(j => times.has(j.name)).map(j => ({ name: j.name, time: times.get(j.name), blob: thumbs.get(j.name) }));
  post({ type: 'thumbs', frames: frames.sort((a, b) => a.time - b.time), aspect: features.width / features.height });
}

// A photo set's contact sheet (DESIGN-sfm §13.3 P5): every photo in capture order, upright as shot
// (not rotated as mapped: what the user took), contained in the set's aspect; with its labels, and
// whether it is used (with the reason when not: another size, filtered out). Thumbnails are made once,
// from the files.
async function postPhotoThumbs() {
  const { plan, files } = photos, [W, H] = plan.size, w = THUMB_WIDTH, h = Math.round(THUMB_WIDTH * H / W), t = performance.now();
  const made = plan.photos.filter(p => !thumbs.has(p.name)).length;
  const thumb = async (p) => {
    const bitmap = await createImageBitmap(files[p.item], { imageOrientation: 'from-image' });
    const s = Math.min(w / bitmap.width, h / bitmap.height), dw = bitmap.width * s, dh = bitmap.height * s;
    thumbs.set(p.name, await thumbJpeg(bitmap, w, h, [(w - dw) / 2, (h - dh) / 2, dw, dh]));
  };
  // THUMB_DECODES at a time: the browser decodes them on several threads
  const todo = plan.photos.filter(p => !thumbs.has(p.name));
  for (let k = 0; k < todo.length; k += THUMB_DECODES) await Promise.all(todo.slice(k, k + THUMB_DECODES).map(thumb));
  if (made) log(`photos: ${made} thumbnails in ${seconds(t)}`);
  const skipped = new Map(plan.skipped.map(s => [s.name, s.reason])), used = new Set(plan.used.map(p => p.name));
  post({ type: 'thumbs', aspect: W / H, frames: plan.photos.map(p => ({
    name: p.name, time: p.time, blob: thumbs.get(p.name), used: used.has(p.name), reason: skipped.get(p.name) ?? null,
    labels: { focal35: p.focal35, zoom: p.labels.zoom, orientation: p.labels.orientation, lens: p.labels.lens } })) });
}

// Sorted indices as runs, for the log: [3, 4, 5, 9] -> '3-5, 9'
const spans = (ks) => ks.reduce((out, k) => { const r = out.at(-1); if (r && k === r[1] + 1) r[1] = k; else out.push([k, k]); return out; }, [])
  .map(([a, b]) => (a === b ? `${a}` : `${a}-${b}`)).join(', ');

// S0b: the keyframes picked, as { sel, kept (their samples) }
function picks(o, analysis) {
  const sel = selection(analysis, o), kept = sel.kept.map(i => analysis.order[i]);
  const blurry = sel.status.filter(s => s === 'blurry').length;
  log(`keyframes: ${kept.length} kept (${blurry} blurry dropped, median threshold ${sel.threshold.toFixed(1)}` +
      (sel.path ? `; path ${(sel.path.at(-1) * o.spacing).toFixed(2)} diagonals${o.minRate ? `, at least ${o.minRate} spacings a second` : ''})` : ')'));
  if (kept.length < 3) throw new Error(`only ${kept.length} keyframes: lower the spacing or the blur threshold`);
  return { sel, kept };
}

// The features `f` made with options o of the keyframes `kept`, if they are
const sameFeatures = (f, o, kept) => same(f?.options, o, FEATURE_KEYS) && f.kept.join() === kept.join();

// S0b + S1: keyframes (second decode pass at full size): JPEG, gray -> SIFT
async function keyframes(o, analysis) {
  const { sel, kept } = picks(o, analysis);
  if (sameFeatures(cache.features, o, kept)) {
    // a fresh list: densify appends to the session's own (the JPEGs of added frames can stay)
    return videoKeyframes({ ...cache.features, feats: cache.features.feats.slice(0, kept.length) }, sel.kept);
  }
  cache.features = await keyframeFeatures(o, analysis, sel, kept);
  saveFeatures(cache.features);
  return videoKeyframes(cache.features, sel.kept);
}

// Several videos (clips): each one's keyframes as a video's alone (its features kept per video: a re-run extracts
// only a video whose keyframes changed), then one list, the j-th video's after the one before (joinClips)
async function clipKeyframes(o) {
  const parts = [];
  for (const [c, clip] of clips.entries()) {
    useClip(c);
    log(`video ${c + 1} of ${clips.length}: ${video.file.name}${offset ? ` (its keyframes at ${offset.toFixed(1)} s and on)` : ''}`);
    const analysis = await analyze(o), { sel, kept } = picks(o, analysis);
    if (!sameFeatures(clip.features, o, kept)) clip.features = await keyframeFeatures(o, analysis, sel, kept);
    const f = clip.features;  // its keyframes' own copies: zoomCameras and flowChains set their fields
    parts.push(videoKeyframes({ ...f, feats: f.feats.slice(0, kept.length).map(x => ({ ...x, clip: c })) }, sel.kept));
  }
  postClipsSelection();
  cache.features = joinClips(parts, o);
  saveFeatures(cache.features);
  return cache.features;
}

// The videos' keyframes (videoKeyframes', in order) as one list: their pairs and tracks re-indexed, their flow chains
// apart; a video's main zoom stretch is the first's camera 0 (the same drone and lens, its focal found from all), its
// other stretches cameras of their own (their image scale to its main one: zoomCameras)
function joinClips(parts, o) {
  const { width, height } = parts[0], feats = [], jpegs = [], trackPairs = [], seq = [], stats = {}, levels = [1];
  let chain = 0;
  for (const [c, p] of parts.entries()) {
    if (p.width !== width || p.height !== height) throw new Error(`video ${c + 1}: keyframes of ${p.width} x ${p.height}, the first's ${width} x ${height}: open videos of one size`);
    const base = feats.length, camera = (p.zoomLevels ?? [1]).map((r, k) => (k ? levels.push(r) - 1 : 0));
    let chains = 0;
    for (const f of p.feats) {
      if (f.camera !== undefined) f.camera = camera[f.camera];
      if (f.flowChain !== undefined) { chains = Math.max(chains, f.flowChain + 1); f.flowChain += chain; }
      feats.push(f);
    }
    chain += chains;
    jpegs.push(...p.jpegs);
    for (const q of p.trackPairs ?? []) trackPairs.push({ ...q, i: q.i + base, j: q.j + base });
    for (const q of p.seq?.pairs ?? []) seq.push({ ...q, i: q.i + base, j: q.j + base });
    for (const [k, v] of Object.entries(p.seq?.stats ?? {})) stats[k] = (stats[k] ?? 0) + v;
  }
  if (levels.length > 1) for (const f of feats) f.camera ??= 0;
  return { options: { ...o }, id: newId(), kept: parts.flatMap(p => p.kept), feats, jpegs, width, height,
           trackPairs: trackPairs.length ? trackPairs : null, seq: parts.every(p => p.seq) ? { pairs: seq.sort((a, b) => a.i - b.i || a.j - b.j), stats, window: o.window } : null,
           zoomLevels: levels.length > 1 ? levels : null, clips: clipsMeta() };
}

// What a project keeps of the videos: each one's name, duration, offset, its first and last frames' times (offset:
// start, end) and keyframes (the analysis samples)
const clipsMeta = () => clips.map(c => ({ name: c.video.file.name, duration: c.video.duration, offset: c.offset,
                                          start: c.offset + spanOf(c.video)[0], end: c.offset + spanOf(c.video)[1], kept: c.features?.kept ?? null }));

// S1 of the video in use: the keyframes `kept` (sel: their selection) as JPEGs and SIFT, their sequential pairs
// (adaptFeatures) and tracks
async function keyframeFeatures(o, analysis, sel, kept) {
  const [w, h] = fitSize(video.width, video.height, o.maxDim);
  const t = performance.now(), jpegs = [];
  let feats = [];
  await extract(kept, w, h, o, async (f, jpeg) => {
    feats.push(f);
    jpegs.push(jpeg);
    progress('features', feats.length, kept.length);
    await liveThumbs.add(f.name, f.time, jpeg.blob, w / h);
    await save(`images/${jpeg.name}`, jpeg.blob);
  }, { dense: o.adaptFeatures });
  liveThumbs.flush();
  // as many features as a stretch needs (frontend.js adaptFeatures): the sequential pairs matched here, before the
  // tracks, as the keyframes' keypoints may change
  let seq = null;
  if (o.adaptFeatures) {
    const ta = performance.now(), { matcher } = await devices();
    const a = await adaptFeatures(feats, matcher, { rng: makeRng(7919), maxFeatures: o.maxFeatures, peak: o.siftPeak, window: o.window, minLink: o.minLink,
                                                    onProgress: (d, n) => progress('match', d, n) });
    feats = a.feats; seq = { ...a.seq, window: o.window };
    log(`features: ${a.weak.length ? `${a.weak.length} weak links (${spans(a.weak)}): keyframes ${spans(a.denseFrames)} with up to ${Math.max(o.maxFeatures, o.denseFeatures)}` : 'no weak link'}; ` +
        `the sequential pairs matched in ${seconds(ta)}`);
  }
  log(`features: ${feats.length} keyframes at ${w} x ${h}, ${perFrame(feats)} per frame in ${seconds(t)}`);
  const trackPairs = o.track ? await trackPoints(analysis, sel.kept, feats, w, h, o) : null;
  // id: a saved model is restored only over the features it was built from
  return { options: { ...o }, id: newId(), kept, feats, jpegs, width: w, height: h, trackPairs, seq };
}

// The keyframes' keypoints tracked through every sampled frame between the first and the last (track.js, DESIGN-sfm
// §5.2.1; a third decode pass, at the keyframes' size): sets each keyframe's `tracked` (the keypoints added to it),
// returns the tracks' pairs. picks: the keyframes' analysis indices.
async function trackPoints(analysis, picks, feats, w, h, o) {
  const t = performance.now(), ov = overlayMask(o), tracker = new PointTracker(w, h);
  const keyframeOf = new Map(picks.map((i, k) => [analysis.order[i], k]));
  const wanted = new Set(analysis.order.slice(picks[0], picks.at(-1) + 1));
  const keep = ov ? (x, y) => !maskAt(ov, x, y, w, h) : null;
  let done = 0;
  await decodeFrames(video, wanted, w, h, async (ctx, sample) => {
    tracker.frame(grayOf(ctx.getImageData(0, 0, w, h).data, w * h));
    const k = keyframeOf.get(sample);
    if (k !== undefined) tracker.keyframe(k, feats[k].keypoints, keep);
    progress('track', ++done, wanted.size);
  });
  noteDecoder();
  const { added, pairs, tracks } = tracker.result(), lost = tracker.stats.lost;
  feats.forEach((f, k) => { f.tracked = added.get(k) ?? new Float32Array(0); });
  const long = tracks.reduce((s, c, v) => s + (v >= 3 ? c : 0), 0), more = feats.reduce((s, f) => s + f.tracked.length / 2, 0);
  log(`tracks: ${done} frames, ${tracks.reduce((s, c) => s + c, 0)} tracks (${long} through 3 keyframes or more), ${Math.round(more / feats.length)} ` +
      `keypoints added a keyframe, ${pairs.length} pairs; lost ${JSON.stringify(lost)} in ${seconds(t)}`);
  return pairs;
}

// The keyframes' cameras (zoomCameras) and flow homographies (flowChains), picks: their analysis indices
const videoKeyframes = (features, picks) => flowChains(zoomCameras(features, picks, cache.zoom, cache.analysis.times, log), picks,
                                                      cache.motion ?? cache.analysis, video.width, video.height);

// ── photo sets (DESIGN-sfm §13): photos instead of a video's keyframes ─────────────────────

let photos = null;  // { files, plan (photos.js planPhotos) } of the open photo set, null for a video

// The photo items of files: EXIF and size from the first 512 KB of each, no decoding
async function readPhotos(files) {
  const items = [];
  for (const file of files) {
    const head = await file.slice(0, 512 * 1024).arrayBuffer(), exif = readExif(head);
    const size = jpegSize(head, exif.orientation);
    if (!size) { log(`${file.name}: not a JPEG (or no frame header in its first 512 KB), left out`, 'warn'); continue; }
    items.push({ fileName: file.name, width: size[0], height: size[1], exif, file });
  }
  return items;
}

// The plan of dropped photos
async function openPhotos(files) {
  const items = await readPhotos(files);
  if (items.length < 2) throw new Error(`${items.length} usable photo(s): a photo set needs at least 2 JPEGs`);
  photos = { items, files: items.map(it => it.file), filter: null, plan: null };
  return planSet();
}

// The plan of the open photo set with its label filter (P8.5: { lens: [..], zoom: [..], orientation: [..] },
// null: every photo), logged; the 'video' info of the panel
function planSet() {
  const f = photos.filter, keep = f && ((l) => Object.entries(f).every(([k, vals]) => !vals?.length || vals.includes(l[k])));
  const plan = photos.plan = planPhotos(photos.items, { filter: keep });
  if (plan.used.length < 2) throw new Error(`${plan.used.length} photo(s) left by the filter: a photo set needs at least 2`);
  plan.cameras.forEach((g, k) => log(`photos: camera ${k}: ${plan.used.filter(p => p.camera === k).length} x ${g.key}${g.focal35 ? `, ${g.focal35} mm equivalent` : ''}`));
  const filtered = plan.skipped.filter(s => s.reason === 'filtered out');
  if (filtered.length) log(`photos: ${filtered.length} filtered out (${filtered.map(s => s.name.match(/^p\d+/)?.[0] ?? s.name).join(', ')})`);
  for (const s of plan.skipped) if (s.reason !== 'filtered out') log(`photos: ${s.name} left out: ${s.reason}`, 'warn');
  if (plan.used.some(p => p.rotate)) log(`photos: ${plan.used.filter(p => p.rotate).length} rotated by 90° to ${plan.size[0] > plan.size[1] ? 'landscape' : 'portrait'}`);
  const [width, height] = plan.size;
  const values = (k) => [...new Set(plan.photos.map(p => p.labels[k]))];
  return { photos: plan.used.length, total: plan.photos.length, name: `${plan.photos.length} photos`, width, height,
           groups: plan.groups.map(g => ({ key: g.key, count: g.count })), skipped: plan.skipped.length,
           labels: { lens: values('lens'), zoom: values('zoom'), orientation: values('orientation') }, filter: photos.filter };
}

// One photo, decoded upright (EXIF orientation), rotated as planned (90° clockwise), at w x h:
// { feat (for the session: name, time, keypoints, descriptors), jpeg: { name, blob }, capped }
// the time of each step, summed over a run (photoFeatures logs it)
const photoTimes = { decode: 0, draw: 0, gray: 0, jpeg: 0, sift: 0 };
async function photoFeature(p, file, w, h, o, sift, ctx, canvas) {
  let t = performance.now();
  const lap = (k) => { const n = performance.now(); photoTimes[k] += n - t; t = n; };
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  lap('decode');
  ctx.save();
  if (p.rotate) { ctx.translate(w, 0); ctx.rotate(Math.PI / 2); ctx.drawImage(bitmap, 0, 0, h, w); }
  else ctx.drawImage(bitmap, 0, 0, w, h);
  ctx.restore();
  bitmap.close();
  lap('draw');
  const gray = grayOf(ctx.getImageData(0, 0, w, h).data, w * h);
  lap('gray');
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: o.jpegQuality });
  lap('jpeg');
  const f = await sift.extract(liftGray(gray, o.siftLift, false), w, h, { maxFeatures: o.maxFeatures, peak: o.siftPeak });
  const { keypoints, capped } = f, descriptors = o.rootSift ? rootSift(f.descriptors) : f.descriptors;
  lap('sift');
  await save(`images/${p.name}`, blob);
  return { feat: { name: p.name, time: p.time, camera: p.camera, keypoints, descriptors }, jpeg: { name: p.name, blob }, capped };
}

// S1 for photos: every photo of the plan (photoFeature), scaled to maxDim, JPEG, SIFT
async function photoFeatures(o) {
  const { plan, files } = photos;
  const names = plan.used.map(p => p.name).join();
  if (same(cache.features?.options, o, FEATURE_KEYS) && cache.features.photos && cache.features.kept.join() === names) {
    return { ...cache.features, feats: cache.features.feats.slice(0, cache.features.kept.length) };
  }
  const { sift } = await devices();
  const [w, h] = fitSize(plan.size[0], plan.size[1], o.maxDim);
  // per photo, across label filters (P8.5): a filter change extracts only the photos not seen yet
  if (!same(cache.photoFeats?.options, o, FEATURE_KEYS) || cache.photoFeats.size.join() !== [w, h].join()) cache.photoFeats = { options: { ...o }, size: [w, h], byName: new Map() };
  const canvas = new OffscreenCanvas(w, h), ctx = canvas.getContext('2d', { willReadFrequently: true });
  const t = performance.now(), feats = [], jpegs = [];
  let capped = 0, extracted = 0;
  for (const p of plan.used) {
    let r = cache.photoFeats.byName.get(p.name);
    if (!r) {
      r = await photoFeature(p, files[p.item], w, h, o, sift, ctx, canvas);
      cache.photoFeats.byName.set(p.name, r);
      capped += r.capped;
      extracted++;
    }
    feats.push({ ...r.feat, camera: p.camera });  // the camera index depends on the filter
    jpegs.push(r.jpeg);
    progress('features', feats.length, plan.used.length);
  }
  warnCapped(capped, 'photo');
  log(`features: ${feats.length} photos at ${w} x ${h} (${extracted} extracted, ${feats.length - extracted} cached), ${perFrame(feats)} per photo in ${seconds(t)} ` +
      `(${Object.entries(photoTimes).map(([k, v]) => `${k} ${(v / 1000).toFixed(1)}`).join(', ')} s)`);
  for (const k in photoTimes) photoTimes[k] = 0;
  cache.features = { options: { ...o }, id: newId(), kept: feats.map(f => f.name), feats, jpegs, width: w, height: h, photos: true };
  saveFeatures(cache.features);
  return cache.features;
}

// Add photos (DESIGN-sfm §13.3 P4): the new ones of the set's camera and size (photos.js planMore),
// their features, then session.insert matched against every photo (the window spans the set) and,
// with PHOTO_MAPPING, the pieces apart regrown into the main one
async function addPhotos(files) {
  if (!photos) throw new Error('add photos: open a photo set first');
  const { session, features, o } = current;
  const items = await readPhotos(files);
  const more = planMore(photos.plan, items);
  for (const sk of more.skipped) log(`add photos: ${sk.name} left out: ${sk.reason}`, 'warn');
  if (!more.photos.length) throw new Error(`add photos: none of the ${files.length} photo(s) can join the set (the log says why)`);
  const { sift } = await devices(), [w, h] = [features.width, features.height];
  const canvas = new OffscreenCanvas(w, h), ctx = canvas.getContext('2d', { willReadFrequently: true });
  const added = [];
  for (const p of more.photos) {
    const it = items[p.item], entry = { ...p, item: photos.files.length };
    const r = await photoFeature(p, it.file, w, h, o, sift, ctx, canvas);
    photos.files.push(it.file);
    photos.items.push(it);
    photos.plan.photos.push(entry);
    photos.plan.used.push(entry);
    features.feats.push(r.feat);  // the session's pair source matches from this array
    if (cache.features && cache.features !== features && cache.features.feats !== features.feats) cache.features.feats.push(r.feat);  // a cached copy (photoFeatures)
    features.jpegs.push(r.jpeg);
    features.kept.push(r.feat.name);
    added.push({ name: r.feat.name, time: r.feat.time, keypoints: xyOnly(r.feat.keypoints) });
    progress('features', added.length, more.photos.length);
  }
  log(`add photos: ${added.map(a => a.name).join(', ')}`);
  await postPhotoThumbs();
  session.window = Math.max(1, session.images.length + added.length - 1);  // every pair, as run() matches a set
  const undo = { photos: photos.plan.photos.length - more.photos.length, used: photos.plan.used.length - more.photos.length,
                 files: photos.files.length - more.photos.length, kept: features.kept.length - more.photos.length };
  let r;
  try {
    r = await session.insert(added);
  } catch (e) {
    unaddPhotos(undo);  // the session is as it was (its action restores itself on error)
    throw e;
  }
  (photos.added ??= []).push(undo);
  await saveFeatures(features);  // before the model that refers to them
  return r;
}

// Back to the filter before the last photoFilter, and to the map made with it (if any): its photos'
// cameras for the trainer again
async function undoFilter() {
  const e = photos.filterUndo.pop();
  photos.filter = e.filter;
  const info = planSet();
  current = e.current;
  post({ type: 'video', info: { ...info, filterUndone: true } });
  await postPhotoThumbs();
  log(`undo: photo filter, back to ${e.filter ? JSON.stringify(e.filter) : 'all photos'}`, 'ok');
  if (!current) { post({ type: 'idle', status: `${info.photos} of ${info.total} photos: Run to map them` }); return; }
  current.version++;
  postQuality({ op: 'undo', undone: 'photoFilter' });
  await send();
}

// Back to the photo set before an addPhotos (its features stay, unused, as densify's)
function unaddPhotos(u) {
  photos.plan.photos.length = u.photos;
  photos.plan.used.length = u.used;
  photos.files.length = u.files;
  photos.items.length = u.files;
  current.features.kept.length = u.kept;
}

// ── persistence (DESIGN-sfm §4): OPFS per video, and .gosfm projects ────────────

let storeKey = null;  // OPFS file prefix of the open video (null: nothing saved, e.g. a dropped project)
let store = null;     // the project folder (src/project/store.js), null: none

// A file into the project folder, if any, said to the page (the stepper's saved marks); a failure is logged, the session goes on
async function save(path, data) {
  if (!store) return;
  try {
    await store.write(path, data);
    post({ type: 'saved', path, where: 'project' });
  } catch (e) {
    log(`project: could not write ${path} (${e.message})`, 'warn');
  }
}

async function opfs(name, blob = null) {
  try {
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle('gaussopt-sfm', { create: true });
    if (!blob) return await (await dir.getFileHandle(name)).getFile();
    const w = await (await dir.getFileHandle(name, { create: true })).createWritable();
    for (const piece of Array.isArray(blob) ? blob : [blob]) await w.write(piece);  // packPieces' pieces: one after the other
    await w.close();
  } catch (e) {
    if (blob) log(`could not save ${name} in the browser storage: ${e.message}`, 'warn');
    return null;
  }
}

// Analysis + keyframes (JPEGs, SIFT) of features `f` (the session's after a run: with densified
// frames): written once per keyframe set, and after a densify. pack: packProject (a Blob), or packPieces (its
// pieces, for a store: a container of a long video, or of several, is more than a Blob the browser can hold)
function featuresProject(f, pack = packProject) {
  const { analysis } = cache, parts = {};
  for (const x of f.feats) {
    parts[`kp/${x.name}`] = x.keypoints; parts[`desc/${x.name}`] = x.descriptors;
    if (x.tracked?.length) parts[`trk/${x.name}`] = x.tracked;
  }
  // the tracks' pairs: [i, j, matches] per pair, their matches one after the other
  if (f.trackPairs?.length) {
    const { pairs, matches } = packTrackPairs(f.trackPairs);
    parts['track/pairs'] = pairs;
    parts['track/matches'] = matches;
  }
  for (const j of f.jpegs) parts[`jpeg/${j.name}`] = j.blob;
  const source = video && !clips ? { name: video.file.name, duration: video.duration } : cache.source ?? null;
  // a photo set has no frame analysis (S0): its plan stands in for the source
  const part = (a) => a && (({ order, sharp, steps, times, zoom, flowH, options }) => ({ options, order, sharp, steps, times, zoom, flowH }))(a);
  // several videos: each one's (videos: f.clips, a keyframe's clip)
  const several = clips ? { videos: clipsMeta(), analyses: clips.map((_, c) => part(clipCache(c, 'analysis'))) } : f.clips ? { videos: f.clips, analyses: cache.analyses ?? null } : {};
  return pack({ video: source, analysis: clips ? null : part(analysis), ...several, photos: photos ? { plan: photos.plan } : cache.photoPlan ?? null,
                       features: { options: f.options, id: f.id, kept: f.kept, width: f.width, height: f.height,
                                   frames: f.feats.map(x => ({ name: x.name, time: x.time, ...(x.clip ? { clip: x.clip } : {}) })) } }, parts);
}

const modelProject = (saved = new Date().toISOString(), pack = packProject) => {
  const { json, arrays } = current.session.toState();
  return pack({ session: json, o: current.o, version: current.version, repaired: current.repaired !== false, features: current.features.id, saved }, arrays);
};

// The session's containers: in the project's cache/ with a project folder, else in OPFS per video (a failure is
// logged: the run goes on without them, some callers do not wait for them)
async function saveFeatures(f) {
  try {
    if (store) await save('cache/features.gosfm', featuresProject(f, packPieces));
    else if (storeKey && await opfs(`${storeKey}.features`, featuresProject(f, packPieces)) !== null) post({ type: 'saved', path: 'cache/features.gosfm', where: 'browser' });
  } catch (e) {
    log(`the keyframes' features are not saved (${e.message})`, 'warn');
  }
}
// with a project folder, the theatre's trace beside it, stamped with the model it ends at (restoreFrom replays it
// for that model only: none recorded for this one, the earlier trace.bin is another's)
async function saveModel() {
  if (!current) return;
  const saved = new Date().toISOString();
  if (store) {
    await save('cache/model.gosfm', modelProject(saved, packPieces));
    if (current.trace) await save('cache/trace.bin', encodeTrace(current.trace.events, { model: saved, full: current.trace.full }));
  } else if (storeKey && await opfs(`${storeKey}.model`, modelProject(saved, packPieces)) !== null) post({ type: 'saved', path: 'cache/model.gosfm', where: 'browser' });
}
// The saved containers [features, model] (null when missing): the project's, else this video's in OPFS
async function loadSaved() {
  const fp = await store?.read('cache/features.gosfm').catch(() => null);
  if (fp) return [fp, await store.read('cache/model.gosfm').catch(() => null)];
  return storeKey ? [await opfs(`${storeKey}.features`), await opfs(`${storeKey}.model`)] : [null, null];
}

// The saved session of the opened video or photo set, if any (a failure is logged): true when its model is restored
async function restoreSaved() {
  const [fp, mp] = await loadSaved();
  if (!fp) return false;
  return restore(await unpackProject(fp), mp && await unpackProject(mp)).catch(e => { log(e.message, 'warn'); return false; });
}

// What a saved features container was made from: { name, duration } of its video, or { name,
// photos } of a photo set (no duration)
function savedSource(json) {
  if (json.photos) return { name: `${json.photos.plan.used.length} photos`, photos: json.photos.plan.used.length, duration: 0 };
  if (json.videos) return { name: json.videos.map(v => v.name).join(' + '), duration: json.videos.at(-1).end, videos: json.videos };
  const v = json.video, times = json.analysis?.times ?? [];
  return { name: v?.name ?? '(unknown video)', duration: v?.duration ?? times.reduce((m, t) => Math.max(m, t), 0) };
}

// Caches and session from saved containers (views from unpackProject / projectPart). On failure
// nothing is restored.
async function restore(fp, mp) {
  try {
    return await restoreFrom(fp, mp);
  } catch (e) {
    reset();
    throw new Error(`the saved session cannot be restored (${e.message}): Run all starts over`);
  }
}

async function restoreFrom(fp, mp) {
  const a = fp.json.analysis, f = fp.json.features, feats = [], jpegs = [];
  cache.analysis = a;  // null for a photo set
  cache.source = fp.json.video;
  cache.photoPlan = fp.json.photos ?? null;
  cache.analyses = fp.json.analyses ?? null;  // several videos': kept for the project's next save when they are not open
  for (const { name, time, clip } of f.frames) {
    feats.push({ name, time, ...(clip ? { clip } : {}), keypoints: await fp.get(`kp/${name}`), descriptors: await fp.get(`desc/${name}`),
                 ...(fp.keys.includes(`trk/${name}`) ? { tracked: await fp.get(`trk/${name}`) } : {}) });
    if (fp.keys.includes(`jpeg/${name}`)) jpegs.push({ name, blob: new Blob([await (await fp.get(`jpeg/${name}`)).arrayBuffer()], { type: 'image/jpeg' }) });
  }
  let trackPairs = null;
  if (fp.keys.includes('track/pairs')) {
    trackPairs = unpackTrackPairs(await fp.get('track/pairs'), await fp.get('track/matches'));
  }
  cache.features = { options: f.options, id: f.id, kept: f.kept, feats, jpegs, width: f.width, height: f.height, trackPairs,
                     ...(fp.json.videos ? { clips: fp.json.videos } : {}) };
  const own = clips && fp.json.videos && restoreClips(fp.json, cache.features);
  post({ type: 'keyframes', count: feats.length });
  await postThumbs(cache.features);
  if (!mp) return false;
  if (mp.json.features !== f.id) { log('the saved model is of other keyframes (an interrupted run): not restored', 'warn'); return false; }
  const arrays = {};
  for (const k of mp.keys) arrays[k] = await mp.get(k);
  const { matcher } = await devices();
  const byName = new Map(feats.map(x => [x.name, x]));
  const keypoints = (n) => {
    if (!byName.has(n)) throw new Error(`keyframe ${n} of the model has no saved features`);
    return mapperKeypoints(byName.get(n));
  };
  const trace = await savedTrace(mp.json.saved);
  const session = SfmSession.fromState({ json: mp.json.session, arrays }, keypoints,
                                       { pairSource: featurePairSource(feats, matcher), log: (s) => log(`  ${s}`), progress, trace: trace?.hook });
  const o = { ...SESSION_DEFAULTS, ...mp.json.o };  // a model saved before an option existed: its default
  current = { session, features: cache.features, o, version: mp.json.version, repaired: mp.json.repaired !== false, trace };  // saved before 'repaired' was: after the repair
  if (a && !clips) selection(a, o);
  if (own) {  // each video's selection (its zooms, for densify), on the strip one after the other
    for (const [c, clip] of clips.entries()) if (clipCache(c, 'analysis')) { useClip(c); selection(cache.analysis, o); }
    postClipsSelection();
  }
  log(`restored the saved session (${mp.json.saved}): ${session.images.length} keyframes; Run all starts over`, 'ok');
  postQuality();
  post({ type: 'restored', repaired: current.repaired });
  return true;
}

// The project's videos open (the same names, in order): each one's analysis and features as clipKeyframes made them
// (its keyframes the first of its own in f, the densified after them; its tracks re-indexed), so that densify and a
// re-run go on from them. False when they are other videos.
function restoreClips({ videos, analyses }, f) {
  if (videos.length !== clips.length || videos.some((v, c) => v.name !== clips[c].video.file.name)) {
    log(`the project is of ${videos.map(v => v.name).join(' + ')}, not the open videos: their keyframes are not restored as theirs`, 'warn');
    return false;
  }
  let base = 0;
  for (const [c, v] of videos.entries()) {
    useClip(c);
    cache.analysis = analyses?.[c] ?? null;
    const n = v.kept?.length ?? 0, feats = f.feats.filter(x => (x.clip ?? 0) === c).slice(0, n), names = new Set(feats.map(x => x.name));
    if (n && feats.length === n) {
      const trackPairs = (f.trackPairs ?? []).filter(p => p.i >= base && p.j < base + n).map(p => ({ ...p, i: p.i - base, j: p.j - base }));
      clips[c].features = { options: f.options, id: f.id, kept: v.kept, feats, jpegs: f.jpegs.filter(j => names.has(j.name)), width: f.width, height: f.height,
                            trackPairs: trackPairs.length ? trackPairs : null, seq: null };
    }
    base += n;
  }
  return true;
}

// The trainer's points: seen by trainMinObs keyframes; by default 3 for a video, 2 for photos (a
// higher held-out PSNR on photo sets)
const trainMinObs = (o) => o.trainMinObs > 0 ? o.trainMinObs : photos ? 2 : 3;

// S5: undistorted JPEGs, point colors, COLMAP binary model of dataset ds (datasetFromMapper)
async function handOver(ds, features, o) {
  const t = performance.now(), { undistort } = await devices();
  const colors = new PointColors(ds), byName = new Map(features.jpegs.map(j => [j.name, j.blob]));
  const { width: w, height: h } = ds.camera, canvas = new OffscreenCanvas(w, h), ctx = canvas.getContext('2d', { willReadFrequently: true });
  const images = [];
  for (const im of ds.images) {
    const bitmap = await createImageBitmap(byName.get(im.name));
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    const c = im.cameraId - 1;  // its lens's camera (DESIGN-sfm §13 P8; one size for all)
    const out = await undistort.run(ctx.getImageData(0, 0, w, h).data, ds.distortedCameras[c], ds.cameras[c]);
    colors.add(im, out);
    ctx.putImageData(new ImageData(out, w, h), 0, 0);
    images.push({ name: im.name, blob: await canvas.convertToBlob({ type: 'image/jpeg', quality: o.jpegQuality }) });
    progress('undistort', images.length, ds.images.length);
  }
  colors.finish();
  // the on-screen display's mask per camera, undistorted as its keyframes (any coverage after the resampling: masked)
  const ovs = overlayMasks(o), masks = [];  // several videos: what any of their displays covers
  if (ovs.length) for (let c = 0; c < ds.cameras.length; c++) {
    const rgba = new Uint8ClampedArray(4 * w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (ovs.some(ov => maskAt(ov, x + 0.5, y + 0.5, w, h))) rgba.fill(255, 4 * (y * w + x), 4 * (y * w + x) + 4);
    const out = await undistort.run(rgba, ds.distortedCameras[c], ds.cameras[c]), data = new Uint8Array(w * h);
    for (let p = 0; p < w * h; p++) data[p] = out[4 * p] > 0 ? 1 : 0;
    masks.push({ camera: c + 1, w, h, data });
  }
  log(`hand-over: ${images.length} undistorted keyframes, ${ds.points.length} colored points${masks.length ? `, the overlay's mask` : ''} in ${seconds(t)}`);
  return { sparse: writeColmapModel(ds), images, masks };
}

// The cameras for the trainer from the session's current state; null when there is nothing to train
// on (no piece of 2 photos with points), the advice says what to shoot
async function send() {
  const t0 = performance.now(), { session, features, o } = current;
  if (!session.mapper.poses.some(Boolean)) return null;
  const ds = datasetFromMapper(session.mapper, { minObs: trainMinObs(o) });
  if (ds.images.length < 2 || !ds.points.length) return null;
  const { sparse, images, masks } = await handOver(ds, features, o);
  if (store) {  // the trainer's dataset, and the distorted model of the keyframes in images/ (same points)
    const raw = datasetFromMapper(session.mapper, { pinhole: false, minObs: trainMinObs(o) });  // the same points as ds
    raw.points.forEach((p, i) => { p.rgb = ds.points[i].rgb; });
    for (const [name, buf] of Object.entries(writeColmapModel(raw))) await save(`sparse/0/${name}`, buf);
    for (const [name, buf] of Object.entries(sparse)) await save(`gs/sparse/${name}`, buf);
    for (const { name, blob } of images) await save(`gs/images/${name}`, blob);
  }
  const model = session.model();
  const summary = { keyframes: session.images.length, registered: model.images.length, exported: ds.images.length,
                    segments: new Set(model.images.map(im => im.segment)).size, focal: model.focal, points: ds.points.length,
                    seconds: (performance.now() - t0) / 1000, version: current.version,
                    distorted: ds.distorted.params, ...(ds.cameras.length > 1 ? { distortedCameras: ds.distortedCameras.map(c => c.params) } : {}) };  // the photos depend on it (undistortion)
  post({ type: 'result', sparse, images, masks, summary }, [...Object.values(sparse), ...masks.map(m => m.data.buffer)]);
  return summary;
}

// Per-keyframe scores for the panel (quality.js, without the heavy fields)
function postQuality(action = null) {
  const q = current.session.quality({ checkLoops: true });
  const time = new Map(current.session.images.map(im => [im.name, im.time]));
  post({ type: 'quality', action, quality: {
    counts: q.counts, main: q.main, version: current.version,
    frames: q.frames.map(f => ({ name: f.name, time: time.get(f.name), score: f.score, reason: f.reason, segment: f.segment,
                                  registered: f.registered, support: f.support })),
    loops: q.loops.map(l => ({ a: l.a, b: l.b, inliers: l.inliers, rotDeg: l.rotDeg, reloc: l.reloc, suspect: l.suspect })),
    ...(photos ? { coverage: photoCoverage() } : {}),
  } });
}

// A photo set's shooting advice (coverage.js): what to shoot next, by short name (p07)
// (all the gaps for the log: all = true)
function photoCoverage(all = false) {
  const mapper = current.session.mapper, c = coverage(mapper);
  const name = (k) => mapper.images[k].name.match(/^p\d+/)?.[0] ?? mapper.images[k].name;
  return { inMain: c.inMain, total: c.total, seenBy: c.seenBy, advice: coverageAdvice(c, name, all ? { maxGaps: Infinity, maxPieces: Infinity } : {}) };
}

// The mapping's trace (trace.js) for the theatre: its events to the page, a batch at most every TRACE_MS; going
// on from `events` (a project's) when given
const TRACE_MS = 200;
function traceRecorder(events = null) {
  let queue = [], timer = null;
  const flush = () => { timer = null; if (queue.length) post({ type: 'trace', events: queue }); queue = []; };
  const onEvent = (e) => { queue.push(e); timer ??= setTimeout(flush, TRACE_MS); };
  return events ? SfmTrace.from(events, { onEvent }) : new SfmTrace({ onEvent });
}

// The project's trace of the model saved at `saved` (saveModel), replayed in the theatre and recording on; null when
// there is none, it is another model's, or it cannot be read (the session restores without it)
async function savedTrace(saved) {
  const blob = await store?.read('cache/trace.bin').catch(() => null);
  if (!blob) return null;
  try {
    const { meta, events } = decodeTrace(await blob.arrayBuffer());
    if (meta.model !== saved) { log('cache/trace.bin: the trace of another model, not replayed', 'warn'); return null; }
    post({ type: 'trace', replay: events });
    return traceRecorder(events);
  } catch (e) {
    log(`cache/trace.bin: unreadable (${e.message}), not replayed`, 'warn');
    return null;
  }
}

// What the session holds, in MB: each typed array counted once (the pairs' matches are the session's and the
// matching's, the mapper's adjacency a copy of them), the keyframe JPEGs by their blobs' sizes
function heldLine() {
  const { session, features } = current, seen = new Set();
  const mb = (arrays) => { let b = 0; for (const a of arrays) if (a && !seen.has(a)) { seen.add(a); b += a.byteLength; } return (b / 2 ** 20).toFixed(0); };
  const m = session.mapper, obs = m.points.reduce((s, P) => s + (P ? P.obs.size : 0), 0);
  return `held: descriptors ${mb(features.feats.map(f => f.descriptors))} MB, keypoints ${mb(features.feats.map(f => f.keypoints))} MB, ` +
    `pair matches ${mb(session.pairs.map(p => p.matches))} MB, the mapper's adjacency ${mb(m.adj.flatMap(l => l.flatMap(e => [e.a, e.b])))} MB, ` +
    `tracks ${mb((features.trackPairs ?? []).map(p => p.matches))} MB, keyframe JPEGs ${((features.jpegs ?? []).reduce((s, j) => s + (j.blob?.size ?? 0), 0) / 2 ** 20).toFixed(0)} MB; ` +
    `${m.points.filter(Boolean).length} points with ${obs} observations (a Map each), ${session.ops.length} undo snapshots${session.afterSequential ? ' and the sequential map\'s' : ''}`;
}

// The mapping's checkpoints (DESIGN-sfm §4): with a project folder, the session as it is every checkpointMs of the
// mapping (mapper.js checkpoint) in cache/mapping.gosfm, stamped with its keyframes' features and the camera options;
// a run on those keyframes with those options goes on from it (the matching and the frames registered kept)
const pick = (o, keys) => Object.fromEntries(keys.filter(k => k in o).map(k => [k, o[k]]));
async function saveCheckpoint(session, features, o) {
  const t = performance.now(), { json, arrays } = session.toState();
  await save('cache/mapping.gosfm', packPieces({ session: json, o, features: features.id, cameras: pick(o, CAMERA_KEYS), saved: new Date().toISOString() }, arrays));
  log(`mapping checkpoint: ${session.mapper.stats.registered}/${session.images.length} registered, saved in ${((performance.now() - t) / 1000).toFixed(1)} s`);
}

// A session from the project's mapping checkpoint for these keyframes and camera options, else null
async function mappingCheckpoint(features, o, matcher, trace) {
  const blob = await store?.read('cache/mapping.gosfm').catch(() => null);
  if (!blob) return null;
  try {
    const mp = await unpackProject(blob);
    if (mp.json.done) return null;  // the mapping it was of went to its end
    const want = JSON.stringify(pick(o, CAMERA_KEYS));
    if (mp.json.features !== features.id || JSON.stringify(mp.json.cameras) !== want) {
      log('cache/mapping.gosfm: a checkpoint of other keyframes or camera options, not resumed');
      return null;
    }
    const arrays = {};
    for (const k of mp.keys) arrays[k] = await mp.get(k);
    const byName = new Map(features.feats.map(x => [x.name, x]));
    const session = SfmSession.fromState({ json: mp.json.session, arrays }, (n) => mapperKeypoints(byName.get(n)),
                                         { pairSource: featurePairSource(features.feats, matcher), log: (s) => log(`  ${s}`), progress, trace: trace?.hook });
    log(`mapping checkpoint of ${mp.json.saved}: ${session.mapper.stats.registered}/${session.images.length} registered, the mapping goes on from it`, 'ok');
    return session;
  } catch (e) {
    log(`cache/mapping.gosfm: not resumed (${e.message})`, 'warn');
    return null;
  }
}

async function run(options) {
  const o = { ...SESSION_DEFAULTS, ...options };
  const t0 = performance.now();
  if (!photos) liveThumbs.start();
  const features = await (photos ? photoFeatures(o) : clips ? clipKeyframes(o) : analyze(o).then(a => keyframes(o, a))).finally(() => liveThumbs.end());
  post({ type: 'keyframes', count: features.feats.length });
  if (photos) await postPhotoThumbs(); else await postThumbs(features);
  if (o.until === 'keyframes') {
    post({ type: 'idle', status: `${features.feats.length} keyframes, ${perFrame(features.feats)} features each` });
    return;
  }
  const { matcher } = await devices();
  const mapping = photos ? photoMapping(photos.plan, o, features.width, features.height) : baseMapping(o);
  // a photo set: every pair matched (the window spans the set, DESIGN-sfm §13.3 P3). The loop
  // stage stays: it merges segments too
  const window = photos ? Math.max(1, features.feats.length - 1) : o.window, loops = o.loops;
  const cameras = photos ? photoCameras(photos.plan, features.width, features.height) : zoomPriors(features);
  if (!photos && cameras) mapping.scanFocal = true;
  post({ type: 'trace', reset: true });  // the theatre's last mapping is not this one's, recorded or not
  const trace = o.trace ? traceRecorder() : null;
  const resume = await mappingCheckpoint(features, o, matcher, trace);
  const { session } = await reconstructFromFeatures(features.feats, features.width, features.height, matcher,
    { window, loops, lateLoops: o.lateLoops, mapping, log, progress, trace: trace?.hook, ...(cameras ? { cameras } : {}), ...(features.trackPairs ? { trackPairs: features.trackPairs } : {}),
      ...(!photos && features.seq?.window === window ? { seq: features.seq } : {}),
      ...(store ? { checkpoint: (session) => saveCheckpoint(session, features, o) } : {}), ...(resume ? { resume } : {}) });
  if (store) await save('cache/mapping.gosfm', packProject({ done: true, features: features.id, saved: new Date().toISOString() }, {}));
  // version: bumped by every action and undo; repaired: the run's repair done
  current = { session, features, o, version: 0, repaired: false, trace };
  // repair: each stretch of keyframes scored below green re-matched, kept where the scores improve (session.repair);
  // not for photos (P4)
  if (!photos && o.autoRepair && session.quality().counts.green < session.images.length) {
    await saveModel();  // the map before its repair (hours on a long video): restored, the repair goes on from it
    trace?.record(session.mapper, 'stage', { name: 'repair' });
    const r = await session.repair();
    trace?.record(session.mapper, 'end');
    log(`repair: ${r.kept}/${r.tried} retries kept, scores ${JSON.stringify(r.quality.counts)} (undo reverts it)`, r.quality.counts.red ? 'warn' : 'ok');
  }
  current.repaired = true;
  log(heldLine());
  postQuality();
  if (photos) for (const a of photoCoverage(true).advice) log(`to shoot: ${a}`, 'warn');
  await saveModel();
  const summary = await send();
  if (!summary) {
    const m = session.model();
    log(`done: ${m.images.length}/${session.images.length} placed, no piece of 2 photos with points to train on${photos ? ': see the advice (Cameras)' : ''}`, 'warn');
    post({ type: 'idle', status: `nothing to train on (${m.images.length}/${session.images.length} placed)` });
    return;
  }
  log(`done: ${summary.exported} cameras (${summary.registered}/${summary.keyframes} registered, ${summary.segments} segment(s)), ` +
      `${summary.points} points, focal ${summary.focal.toFixed(1)} px, ${((performance.now() - t0) / 1000).toFixed(1)} s`, 'ok');
}

// Densify (DESIGN-sfm §6.3): between consecutive selected keyframes, the sharpest analyzed frame
// of the gap's middle half becomes a keyframe (decoded, JPEG, SIFT), then session.insert()
async function densify(names) {
  if (!video) throw new Error('densify: open the video too (frames are decoded from it)');
  const { session, features, o } = current;
  const times = names.map(n => session.images.find(im => im.name === n)?.time).filter(t => t !== undefined).sort((a, b) => a - b);
  if (times.length < 2) throw new Error('densify: select at least two keyframes');
  if (clips) {  // several videos: the frames of one, decoded from it
    const c = clipOfTime(times[0]);
    if (times.some(t => clipOfTime(t) !== c)) throw new Error('densify: select keyframes of one video');
    useClip(c);
  }
  const analysis = cache.analysis;
  if (!analysis) throw new Error('densify: run the keyframes of this video first (its frames are not analysed)');
  const taken = new Set(session.images.map(im => im.name)), wanted = new Set();
  for (let k = 1; k < times.length; k++) {
    const [a, b] = [times[k - 1], times[k]], q = (b - a) / 4;
    let best = -1;
    analysis.times.forEach((at, i) => {
      const t = at + offset;
      if (cache.zoom?.span[i] < 0) return;  // during a zoom
      if (t > a + q && t < b - q && !taken.has(keyframeName(t)) && (best < 0 || analysis.sharp[i] > analysis.sharp[best])) best = i;
    });
    if (best >= 0) wanted.add(analysis.order[best]);
  }
  if (!wanted.size) throw new Error('densify: no sampled frame left between these keyframes (raise the sample fps)');
  const added = [];
  await extract(wanted, features.width, features.height, o, async (f, jpeg) => {
    const camera = stretchCamera(f.name, session.images);
    if (camera !== undefined) f.camera = camera;
    if (clips) f.clip = clipAt;
    features.feats.push(f);
    features.jpegs.push(jpeg);
    added.push({ name: f.name, time: f.time, keypoints: xyOnly(f.keypoints), ...(camera !== undefined ? { camera } : {}), ...(f.clip ? { clip: f.clip } : {}) });
    await save(`images/${jpeg.name}`, jpeg.blob);
  });
  log(`densify: ${added.length} keyframes added between ${names.length} selected`);
  await postThumbs(features);
  const r = await session.insert(added);
  await saveFeatures(features);  // before the model that refers to them
  return r;
}

// The camera of the keyframes in the same stretch between zooms as the analysed frame `name`
// (zoomCameras); undefined without zooms
function stretchCamera(name, images) {
  const z = cache.zoom;
  if (!z?.events.length) return undefined;
  const index = new Map(cache.analysis.times.map((t, k) => [keyframeName(t + offset), k])), span = z.span[index.get(name)];
  return images.find(im => im.camera !== undefined && z.span[index.get(im.name)] === span)?.camera;
}

async function action(op, args) {
  if (!current) throw new Error('run the reconstruction first');
  const t = performance.now();
  current.trace?.record(current.session.mapper, 'stage', { name: op });
  const r = op === 'densify' ? await densify(...args) : op === 'addPhotos' ? await addPhotos(...args) : await current.session[op](...args);
  current.trace?.record(current.session.mapper, 'end');
  current.version++;
  if (op === 'repair') current.repaired = true;
  await saveModel();
  const changed = `${JSON.stringify(r.before)} -> ${JSON.stringify(r.after)}`;
  log(`${op}: ${r.frames?.length ?? 0} keyframes, scores ${changed} in ${seconds(t)}` +
      (r.registered !== undefined ? `, ${r.registered} registered` : '') + (r.tried !== undefined ? `, ${r.kept}/${r.tried} retries kept` : ''),
      r.after?.red ? 'warn' : 'ok');
  postQuality({ op, before: r.before, after: r.after });
}

// Everything the session has, into a project folder attached after it was made (or a .zip's
// memory store, whose files go back to the page)
async function flush() {
  const f = current?.features ?? cache.features;
  if (f) {
    for (const c of clips ? clips.keys() : [-1]) {
      if (c >= 0) useClip(c);
      if (cache.analysis && cache.selection) await save(sharpnessPath(), sharpnessCsv(cache.motion ?? cache.analysis, cache.selection));
    }
    for (const { name, blob } of f.jpegs) await save(`images/${name}`, blob);
    await saveFeatures(f);
  }
  if (current) {
    await saveModel();
    await send();  // sparse/0 and gs/, and a fresh result for the page
  }
  post({ type: 'flushed', files: store?.files ? [...store.files] : null });
}

async function handle(data) {
  if (data.type === 'photos') {  // a photo set (DESIGN-sfm §13): photos.js plans it, run() takes its photos as keyframes
    resetAll();
    const info = await openPhotos(data.files);
    const used = photos.plan.used.map(p => photos.files[p.item]);
    storeKey = hashString(`photos|${used.map(f => `${f.name}|${f.size}|${f.lastModified}`).join('|')}`).toString(16);
    post({ type: 'video', info });
    await postPhotoThumbs();
    await restoreSaved();
  } else if (data.type === 'photoFilter') {  // only some photos of the set (P8.5): re-planned; Run maps them (cached features)
    if (!photos) throw new Error('open a photo set first');
    (photos.filterUndo ??= []).push({ filter: photos.filter, current });  // undone like a repair (undoFilter)
    photos.filter = data.filter;
    const info = planSet();
    current = null;  // the map was of other photos
    post({ type: 'video', info });
    await postPhotoThumbs();
    post({ type: 'idle', status: `${info.photos} of ${info.total} photos (${data.filter ? 'filtered' : 'all'}): Run to map them` });
  } else if (data.type === 'open' && data.files?.length > 1) {  // several videos of one place, mapped as one
    resetAll();
    clips = [];
    for (const file of data.files) {
      const v = await openVideo(file), last = clips.at(-1);
      if (last && (v.width !== last.video.width || v.height !== last.video.height || v.rotation !== last.video.rotation)) {
        clips = null;
        throw new Error(`${file.name}: ${v.width} x ${v.height}, ${data.files[0].name} ${last.video.width} x ${last.video.height}: open videos of one size`);
      }
      // its first frame CLIP_GAP s after the last of the one before
      clips.push({ video: v, offset: last ? last.offset + spanOf(last.video)[1] + CLIP_GAP - spanOf(v)[0] : 0 });
    }
    useClip(0);
    const { codec, width, height, fps, rotation } = video, end = clips.at(-1);
    post({ type: 'video', info: { name: data.files.map(f => f.name).join(' + '), codec, width, height, fps, duration: end.offset + spanOf(end.video)[1], rotation,
                                  frames: clips.reduce((s, c) => s + c.video.samples.length, 0), videos: clipsMeta() } });
    storeKey = hashString(data.files.map(f => `${f.name}|${f.size}|${f.lastModified}`).join('|')).toString(16);
    await restoreSaved();
  } else if (data.type === 'open') {
    resetAll();
    const file = data.file ?? data.files[0];
    video = await openVideo(file);
    const { codec, width, height, fps, duration, rotation, samples } = video;
    post({ type: 'video', info: { name: file.name, codec, width, height, fps, duration, rotation, frames: samples.length } });
    storeKey = hashString(`${file.name}|${file.size}|${file.lastModified}`).toString(16);
    await restoreSaved();
  } else if (data.type === 'project') {  // a dropped .gosfm: repairs work, densify and re-runs need the video
    const blob = data.file, fp = await projectPart(blob, 'features'), mp = await projectPart(blob, 'model');
    if (!fp) throw new Error(`${data.file.name}: no features in this project`);
    resetAll();
    post({ type: 'video', info: { project: data.file.name, ...savedSource(fp.json) } });
    await restore(fp, mp);
  } else if (data.type === 'save') {
    if (!current) throw new Error('nothing to save yet');
    const project = await mergeProjects({ features: featuresProject(current.features), model: modelProject() });
    post({ type: 'project', blob: project, name: `${((clips?.[0].video ?? video)?.file.name ?? 'session').replace(/\.[^.]+$/, '')}.gosfm` });
  } else if (data.type === 'run') {
    if (!video && !photos) throw new Error('open the video or the photos (a project alone has no frames to re-run)');
    await run(data.options);
  } else if (data.type === 'action') {
    await action(data.op, data.args);
  } else if (data.type === 'undo') {
    // the map's own actions first; once it has none, the filter change that made it (P8.5)
    if (!current?.session.ops.length && photos?.filterUndo?.length) { await undoFilter(); return; }
    const op = current?.session.undo();
    if (!op) { post({ type: 'idle', status: 'nothing to undo' }); return; }
    if (op === 'insert' && photos?.added?.length) unaddPhotos(photos.added.pop());  // a photo set inserts by addPhotos only
    log(`undo: ${op}`, 'ok');
    current.trace?.record(current.session.mapper, 'undo', { op });
    current.version++;
    await saveModel();
    postQuality({ op: 'undo', undone: op });
  } else if (data.type === 'send') {
    if (!current) throw new Error('run the reconstruction first');
    if (!await send()) post({ type: 'idle', status: 'nothing to train on: no piece of 2 photos with points' });
  } else if (data.type === 'store') {
    store = storeFrom(data.store);
  } else if (data.type === 'restoreStore') {  // a project folder without its video: repairs, no re-runs
    const [fp, mp] = await loadSaved();
    if (!fp) throw new Error('this project has no saved session (cache/features.gosfm)');
    resetAll();
    const f = await unpackProject(fp);
    post({ type: 'video', info: { project: store.name, ...savedSource(f.json) } });
    await restore(f, mp && await unpackProject(mp));
  } else if (data.type === 'restoreSaved') {  // a project attached to the open video or photos: its session over them
    if (!await restoreSaved()) post({ type: 'idle', status: 'ready' });
  } else if (data.type === 'flush') {
    await flush();
  } else if (data.type === 'psnr') {
    if (!current) return;
    current.session.viewPsnr = new Map(data.values);
    postQuality();
  }
}

// one message at a time: a run waits for the open before it, a score update for the repair
let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(() => handle(data)).catch((e) => {
    if (e?.stack) log(`error at ${e.stack.split('\n').slice(1, 4).map(l => l.trim()).join(' < ')}`, 'warn');  // where, for reports
    post({ type: 'error', message: e?.message ?? String(e) });
  }).then(() => GpuBundleAdjuster.releaseShared());  // the solver's buffers not kept between actions (the training's GPU)
};
