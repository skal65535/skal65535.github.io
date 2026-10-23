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
//   in:  { type: 'open', file } | { type: 'project', file } | { type: 'run', options (until: 'keyframes' stops there) }
//        | { type: 'action', op, args }
//        | { type: 'undo' } | { type: 'send' } | { type: 'save' } | { type: 'psnr', values: [[name, dB]] }
//        | { type: 'store', store (describe()) | null } | { type: 'flush' } | { type: 'restoreStore' }
//   out: { type: 'video', info } | { type: 'log', text, level } | { type: 'progress', stage, done, total }
//        | { type: 'selection', ... } | { type: 'keyframes', count } | { type: 'thumbs', frames: [{ name, time, blob }], aspect (w / h) }
//        | { type: 'quality', quality, action }
//        | { type: 'result', sparse, images, summary }
//        | { type: 'restored' } | { type: 'project', blob, name } | { type: 'idle', status } | { type: 'error', message }
//        | { type: 'flushed', files: [[path, Blob]] (a memory store's) }
// Messages are handled one at a time, in order.
"use strict";

import { openVideo, sampleFrames, decodeFrames, grayOf } from './video.js';
import { fitSize, selectFrames } from './frames.js';
import { GpuSift, SIFT_DEFAULTS } from './sift.js';
import { GpuMatcher } from './match.js';
import { GpuUndistort, PointColors } from './undistort.js';
import { reconstructFromFeatures, featurePairSource } from './pipeline.js';
import { SfmSession } from './session.js';
import { packProject, unpackProject, mergeProjects, projectPart } from './project.js';
import { SESSION_DEFAULTS } from './defaults.js';
import { datasetFromMapper, writeColmapModel } from './export.js';
import { xyOnly } from './frontend.js';
import { hashString } from './util.js';
import { storeFrom } from '../project/store.js';

const post = (msg, transfer = []) => self.postMessage(msg, transfer);
const log = (text, level = '') => post({ type: 'log', text, level });
const progress = (stage, done, total) => post({ type: 'progress', stage, done, total });
const seconds = (t) => `${((performance.now() - t) / 1000).toFixed(1)} s`;
const same = (a, b, keys) => a && keys.every(k => a[k] === b[k]);

let video = null, gpu = null, current = null;  // current: { session, features, o, version } after a run
const keyframeName = (time) => `t${String(Math.round(time * 1000)).padStart(7, '0')}.jpg`;  // DESIGN-sfm §3
const cache = {};  // analysis, features: { options (that produced them), id, ... }; source: the video of a restored project

function reset() {
  for (const k of Object.keys(cache)) delete cache[k];
  thumbs.clear();
  current = null;
}

async function devices() {
  if (gpu) return gpu;
  const sift = await GpuSift.create();
  gpu = { sift, matcher: await GpuMatcher.create(sift.device), undistort: await GpuUndistort.create(sift.device) };
  return gpu;
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
// analysis spread over the pool while the decoder runs
async function analyze(o) {
  const keys = ['sampleFps', 'sharpDim', 'flowDim'];
  if (same(cache.analysis?.options, o, keys)) return cache.analysis;
  const t = performance.now(), order = sampleFrames(video, o.sampleFps), [w, h] = fitSize(video.width, video.height, o.sharpDim);
  const sharp = new Array(order.length), steps = new Array(order.length), times = new Array(order.length);
  const workers = analysisPool(), options = { sharpDim: o.sharpDim, flowDim: o.flowDim };
  const position = new Map(order.map((k, i) => [k, i]));  // sample -> index in order
  let chunk = [], at = [], last = null, done = 0, failure = null;
  const free = () => workers.find(wk => !wk.busy) ?? Promise.race(workers.map(wk => wk.busy)).then(free);
  const submit = async (frames, at, overlap) => {
    const wk = await free();
    wk.busy = new Promise((resolve) => {
      wk.onmessage = ({ data }) => {
        data.sharpness.forEach((v, k) => { sharp[at[k]] = v; steps[at[k]] = data.motion[k]; });
        done += data.sharpness.length;
        progress('analyze', done, order.length);
        wk.busy = null;
        resolve();
      };
      wk.onerror = (e) => { failure = new Error(`frame analysis: ${e.message}`); wk.busy = null; resolve(); };
    });
    wk.postMessage({ frames, w, h, options, overlap }, frames.map(f => f.buffer).filter((b, k) => k > 0 || !overlap));
  };
  await decodeFrames(video, new Set(order), w, h, async (ctx, k, time) => {
    const gray = grayOf(ctx.getImageData(0, 0, w, h).data, w * h);
    times[position.get(k)] = time;
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
  if (chunk.length) await submit(last ? [last, ...chunk] : chunk, at, !!last);
  await Promise.all(workers.map(wk => wk.busy));
  if (failure) throw failure;
  const missing = order.length - times.filter(t => t !== undefined).length;
  if (missing) throw new Error(`${missing} of ${order.length} sampled frames not decoded (duplicate timestamps in the video?)`);
  log(`frames: ${order.length} sampled at ${Math.min(o.sampleFps, video.fps).toFixed(1)} fps, ` +
      `analyzed in ${seconds(t)} (${workers.length} workers)`);
  return cache.analysis = { options: { ...o }, order, sharp, steps, times };
}

// S0b keyframe selection, shown on the panel's strip
function selection(analysis, o) {
  const sel = selectFrames({ sharpness: analysis.sharp, motion: analysis.steps, times: analysis.times }, o);
  post({ type: 'selection', times: analysis.times, sharpness: analysis.sharp, status: sel.status, path: sel.path, threshold: sel.threshold });
  cache.selection = sel;
  save('sharpness.csv', sharpnessCsv(analysis, sel));
  return sel;
}

// Every sampled frame (as video2splat.py's sharpness.csv, with times): the name it has as a keyframe
function sharpnessCsv(a, sel) {
  const rows = a.times.map((t, i) => `${keyframeName(t)},${t.toFixed(4)},${a.sharp[i].toFixed(2)},${a.steps[i]?.toFixed(5) ?? ''},${sel.status[i]}`);
  return ['frame,time,sharpness,motion,status', ...rows].join('\n') + '\n';
}

// S1 on video samples (decoded at w x h): per frame onFrame({ name, time, keypoints, descriptors }, { name, blob (JPEG) })
async function extract(samples, w, h, o, onFrame) {
  const { sift } = await devices();
  let capped = 0;
  await decodeFrames(video, new Set(samples), w, h, async (ctx, k, time) => {
    const name = keyframeName(time);
    const gray = grayOf(ctx.getImageData(0, 0, w, h).data, w * h);
    const jpeg = { name, blob: await ctx.canvas.convertToBlob({ type: 'image/jpeg', quality: o.jpegQuality }) };
    const { keypoints, descriptors, capped: c } = await sift.extract(gray, w, h, { maxFeatures: o.maxFeatures });
    capped += c;
    await onFrame({ name, time, keypoints, descriptors }, jpeg);
  });
  if (capped) log(`SIFT: ${capped} frame(s) had over ${SIFT_DEFAULTS.maxKeypoints} candidate keypoints: the ones kept depend on the GPU's order`, 'warn');
}

// Small JPEGs of the keyframes for the page's contact sheet (src/ui/frames.js), made once per name
const THUMB_WIDTH = 320;
const thumbs = new Map();
async function postThumbs(features) {
  const times = new Map(features.feats.map(f => [f.name, f.time]));
  for (const { name, blob } of features.jpegs) {
    if (thumbs.has(name)) continue;
    const bitmap = await createImageBitmap(blob, { resizeWidth: THUMB_WIDTH, resizeQuality: 'medium' });
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    canvas.getContext('2d').drawImage(bitmap, 0, 0);
    bitmap.close();
    thumbs.set(name, await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.8 }));
  }
  const frames = features.jpegs.filter(j => times.has(j.name)).map(j => ({ name: j.name, time: times.get(j.name), blob: thumbs.get(j.name) }));
  post({ type: 'thumbs', frames: frames.sort((a, b) => a.time - b.time), aspect: features.width / features.height });
}

// S0b + S1: keyframes (second decode pass at full size): JPEG, gray -> SIFT
async function keyframes(o, analysis) {
  const sel = selection(analysis, o), kept = sel.kept.map(i => analysis.order[i]);
  const blurry = sel.status.filter(s => s === 'blurry').length;
  log(`keyframes: ${kept.length} kept (${blurry} blurry dropped, threshold ${sel.threshold.toFixed(1)}` +
      (sel.path ? `; path ${(sel.path.at(-1) * o.spacing).toFixed(2)} diagonals)` : ')'));
  if (kept.length < 3) throw new Error(`only ${kept.length} keyframes: lower the spacing or the blur threshold`);
  const keys = ['maxDim', 'maxFeatures', 'jpegQuality'];
  if (same(cache.features?.options, o, keys) && cache.features.kept.join() === kept.join()) {
    // a fresh list: densify appends to the session's own (the JPEGs of added frames can stay)
    return { ...cache.features, feats: cache.features.feats.slice(0, kept.length) };
  }
  const [w, h] = fitSize(video.width, video.height, o.maxDim);
  const t = performance.now(), feats = [], jpegs = [];
  await extract(kept, w, h, o, async (f, jpeg) => {
    feats.push(f);
    jpegs.push(jpeg);
    progress('features', feats.length, kept.length);
    await save(`images/${jpeg.name}`, jpeg.blob);
  });
  const per = Math.round(feats.reduce((s, f) => s + f.keypoints.length / 4, 0) / feats.length);
  log(`features: ${feats.length} keyframes at ${w} x ${h}, ${per} per frame in ${seconds(t)}`);
  // id: a saved model is restored only over the features it was built from
  const id = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
  cache.features = { options: { ...o }, id, kept, feats, jpegs, width: w, height: h };
  saveFeatures(cache.features);
  return cache.features;
}

// ── persistence (DESIGN-sfm §4): OPFS per video, and .gosfm projects ────────────

let storeKey = null;  // OPFS file prefix of the open video (null: nothing saved, e.g. a dropped project)
let store = null;     // the project folder (src/project/store.js), null: none

// A file into the project folder, if any; a failure is logged, the session goes on
async function save(path, data) {
  if (!store) return;
  try {
    await store.write(path, data);
  } catch (e) {
    log(`project: could not write ${path} (${e.message})`, 'warn');
  }
}

async function opfs(name, blob = null) {
  try {
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle('gaussopt-sfm', { create: true });
    if (!blob) return await (await dir.getFileHandle(name)).getFile();
    const w = await (await dir.getFileHandle(name, { create: true })).createWritable();
    await w.write(blob);
    await w.close();
  } catch (e) {
    if (blob) log(`could not save ${name} in the browser storage: ${e.message}`, 'warn');
    return null;
  }
}

// Analysis + keyframes (JPEGs, SIFT) of features `f` (the session's after a run: with densified
// frames): written once per keyframe set, and after a densify
function featuresProject(f) {
  const { analysis } = cache, parts = {};
  for (const x of f.feats) { parts[`kp/${x.name}`] = x.keypoints; parts[`desc/${x.name}`] = x.descriptors; }
  for (const j of f.jpegs) parts[`jpeg/${j.name}`] = j.blob;
  const { order, sharp, steps, times, options } = analysis;
  const source = video ? { name: video.file.name, duration: video.duration } : cache.source ?? null;
  return packProject({ video: source, analysis: { options, order, sharp, steps, times },
                       features: { options: f.options, id: f.id, kept: f.kept, width: f.width, height: f.height, frames: f.feats.map(x => ({ name: x.name, time: x.time })) } }, parts);
}

const modelProject = () => {
  const { json, arrays } = current.session.toState();
  return packProject({ session: json, o: current.o, version: current.version, features: current.features.id, saved: new Date().toISOString() }, arrays);
};

// The session's containers: in the project's cache/ with a project folder, else in OPFS per video
async function saveFeatures(f) {
  if (store) await save('cache/features.gosfm', featuresProject(f));
  else if (storeKey) await opfs(`${storeKey}.features`, featuresProject(f));
}
async function saveModel() {
  if (!current) return;
  if (store) await save('cache/model.gosfm', modelProject());
  else if (storeKey) await opfs(`${storeKey}.model`, modelProject());
}
// The saved containers [features, model] (null when missing): the project's, else this video's in OPFS
async function loadSaved() {
  const fp = await store?.read('cache/features.gosfm').catch(() => null);
  if (fp) return [fp, await store.read('cache/model.gosfm').catch(() => null)];
  return storeKey ? [await opfs(`${storeKey}.features`), await opfs(`${storeKey}.model`)] : [null, null];
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
  cache.analysis = a;
  cache.source = fp.json.video;
  for (const { name, time } of f.frames) {
    feats.push({ name, time, keypoints: await fp.get(`kp/${name}`), descriptors: await fp.get(`desc/${name}`) });
    if (fp.keys.includes(`jpeg/${name}`)) jpegs.push({ name, blob: new Blob([await (await fp.get(`jpeg/${name}`)).arrayBuffer()], { type: 'image/jpeg' }) });
  }
  cache.features = { options: f.options, id: f.id, kept: f.kept, feats, jpegs, width: f.width, height: f.height };
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
    return xyOnly(byName.get(n).keypoints);
  };
  const session = SfmSession.fromState({ json: mp.json.session, arrays }, keypoints,
                                       { pairSource: featurePairSource(feats, matcher), log: (s) => log(`  ${s}`), progress });
  current = { session, features: cache.features, o: mp.json.o, version: mp.json.version };
  selection(a, mp.json.o);
  log(`restored the saved session (${mp.json.saved}): ${session.images.length} keyframes; Run all starts over`, 'ok');
  postQuality();
  post({ type: 'restored' });
  return true;
}

// S5: undistorted JPEGs, point colors, COLMAP binary model
async function handOver(mapper, features, o) {
  const t = performance.now(), { undistort } = await devices();
  const ds = datasetFromMapper(mapper);
  const colors = new PointColors(ds), byName = new Map(features.jpegs.map(j => [j.name, j.blob]));
  const { width: w, height: h } = ds.camera, canvas = new OffscreenCanvas(w, h), ctx = canvas.getContext('2d', { willReadFrequently: true });
  const images = [];
  for (const im of ds.images) {
    const bitmap = await createImageBitmap(byName.get(im.name));
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    const out = await undistort.run(ctx.getImageData(0, 0, w, h).data, ds.distorted, ds.camera);
    colors.add(im, out);
    ctx.putImageData(new ImageData(out, w, h), 0, 0);
    images.push({ name: im.name, blob: await canvas.convertToBlob({ type: 'image/jpeg', quality: o.jpegQuality }) });
    progress('undistort', images.length, ds.images.length);
  }
  colors.finish();
  log(`hand-over: ${images.length} undistorted keyframes, ${ds.points.length} colored points in ${seconds(t)}`);
  return { sparse: writeColmapModel(ds), images, ds };
}

// The cameras for the trainer from the session's current state
async function send() {
  const t0 = performance.now(), { session, features, o } = current;
  const { sparse, images, ds } = await handOver(session.mapper, features, o);
  if (store) {  // the trainer's dataset, and the distorted model of the keyframes in images/ (same points)
    const raw = datasetFromMapper(session.mapper, { pinhole: false });
    raw.points.forEach((p, i) => { p.rgb = ds.points[i].rgb; });
    for (const [name, buf] of Object.entries(writeColmapModel(raw))) await save(`sparse/0/${name}`, buf);
    for (const [name, buf] of Object.entries(sparse)) await save(`gs/sparse/${name}`, buf);
    for (const { name, blob } of images) await save(`gs/images/${name}`, blob);
  }
  const model = session.model();
  const summary = { keyframes: session.images.length, registered: model.images.length, exported: ds.images.length,
                    segments: new Set(model.images.map(im => im.segment)).size, focal: model.focal, points: ds.points.length,
                    seconds: (performance.now() - t0) / 1000, version: current.version,
                    distorted: ds.distorted.params };  // the photos depend on it (undistortion)
  post({ type: 'result', sparse, images, summary }, Object.values(sparse));
  return summary;
}

// Per-keyframe scores for the panel (quality.js, without the heavy fields)
function postQuality(action = null) {
  const q = current.session.quality();
  const time = new Map(current.session.images.map(im => [im.name, im.time]));
  post({ type: 'quality', action, quality: {
    counts: q.counts, main: q.main, version: current.version,
    frames: q.frames.map(f => ({ name: f.name, time: time.get(f.name), score: f.score, reason: f.reason, segment: f.segment,
                                  registered: f.registered, support: f.support })),
    loops: q.loops.map(l => ({ a: l.a, b: l.b, inliers: l.inliers, rotDeg: l.rotDeg, suspect: l.suspect })),
  } });
}

async function run(options) {
  const o = { ...SESSION_DEFAULTS, ...options };
  const t0 = performance.now();
  const analysis = await analyze(o);
  const features = await keyframes(o, analysis);
  post({ type: 'keyframes', count: features.feats.length });
  await postThumbs(features);
  if (o.until === 'keyframes') {
    post({ type: 'idle', status: `${features.feats.length} keyframes, ${Math.round(features.feats.reduce((s, f) => s + f.keypoints.length / 4, 0) / features.feats.length)} features each` });
    return;
  }
  const { matcher } = await devices();
  const mapping = { sweeps: o.sweeps, focal: o.focal, ...(o.hfovDeg > 0 ? { hfovDeg: o.hfovDeg } : {}) };
  const { session } = await reconstructFromFeatures(features.feats, features.width, features.height, matcher,
    { window: o.window, loops: o.loops, mapping, log, progress });
  current = { session, features, o, version: 0 };  // version: bumped by every action and undo
  if (o.autoRepair && session.quality().counts.green < session.images.length) {
    const r = await session.repair();
    log(`repair: ${r.kept}/${r.tried} retries kept, scores ${JSON.stringify(r.quality.counts)} (undo reverts it)`, r.quality.counts.red ? 'warn' : 'ok');
  }
  postQuality();
  await saveModel();
  const summary = await send();
  log(`done: ${summary.exported} cameras (${summary.registered}/${summary.keyframes} registered, ${summary.segments} segment(s)), ` +
      `${summary.points} points, focal ${summary.focal.toFixed(1)} px, ${((performance.now() - t0) / 1000).toFixed(1)} s`, 'ok');
}

// Densify (DESIGN-sfm §6.3): between consecutive selected keyframes, the sharpest analyzed frame
// of the gap's middle half becomes a keyframe (decoded, JPEG, SIFT), then session.insert()
async function densify(names) {
  if (!video) throw new Error('densify: open the video too (frames are decoded from it)');
  const { session, features, o } = current, analysis = cache.analysis;
  const times = names.map(n => session.images.find(im => im.name === n)?.time).filter(t => t !== undefined).sort((a, b) => a - b);
  if (times.length < 2) throw new Error('densify: select at least two keyframes');
  const taken = new Set(session.images.map(im => im.name)), wanted = new Set();
  for (let k = 1; k < times.length; k++) {
    const [a, b] = [times[k - 1], times[k]], q = (b - a) / 4;
    let best = -1;
    analysis.times.forEach((t, i) => {
      if (t > a + q && t < b - q && !taken.has(keyframeName(t)) && (best < 0 || analysis.sharp[i] > analysis.sharp[best])) best = i;
    });
    if (best >= 0) wanted.add(analysis.order[best]);
  }
  if (!wanted.size) throw new Error('densify: no sampled frame left between these keyframes (raise the sample fps)');
  const added = [];
  await extract(wanted, features.width, features.height, o, async (f, jpeg) => {
    features.feats.push(f);
    features.jpegs.push(jpeg);
    added.push({ name: f.name, time: f.time, keypoints: xyOnly(f.keypoints) });
    await save(`images/${jpeg.name}`, jpeg.blob);
  });
  log(`densify: ${added.length} keyframes added between ${names.length} selected`);
  await postThumbs(features);
  const r = await session.insert(added);
  await saveFeatures(features);  // before the model that refers to them
  return r;
}

async function action(op, args) {
  if (!current) throw new Error('run the reconstruction first');
  const t = performance.now();
  const r = op === 'densify' ? await densify(...args) : await current.session[op](...args);
  current.version++;
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
  if (cache.analysis && f) {
    if (cache.selection) await save('sharpness.csv', sharpnessCsv(cache.analysis, cache.selection));
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
  if (data.type === 'open') {
    reset();
    video = storeKey = null;
    video = await openVideo(data.file);
    const { codec, width, height, fps, duration, rotation, samples } = video;
    post({ type: 'video', info: { name: data.file.name, codec, width, height, fps, duration, rotation, frames: samples.length } });
    storeKey = hashString(`${data.file.name}|${data.file.size}|${data.file.lastModified}`).toString(16);
    const [fp, mp] = await loadSaved();
    if (fp) await restore(await unpackProject(fp), mp && await unpackProject(mp)).catch(e => log(e.message, 'warn'));
  } else if (data.type === 'project') {  // a dropped .gosfm: repairs work, densify and re-runs need the video
    const blob = data.file, fp = await projectPart(blob, 'features'), mp = await projectPart(blob, 'model');
    if (!fp) throw new Error(`${data.file.name}: no features in this project`);
    reset();
    video = storeKey = null;
    const v = fp.json.video, times = fp.json.analysis.times;
    post({ type: 'video', info: { project: data.file.name, name: v?.name ?? '(unknown video)', duration: v?.duration ?? times.reduce((m, t) => Math.max(m, t), 0) } });
    await restore(fp, mp);
  } else if (data.type === 'save') {
    if (!current) throw new Error('nothing to save yet');
    const project = await mergeProjects({ features: featuresProject(current.features), model: modelProject() });
    post({ type: 'project', blob: project, name: `${(video?.file.name ?? 'session').replace(/\.[^.]+$/, '')}.gosfm` });
  } else if (data.type === 'run') {
    if (!video) throw new Error('open the video (a project alone has no frames to re-run)');
    await run(data.options);
  } else if (data.type === 'action') {
    await action(data.op, data.args);
  } else if (data.type === 'undo') {
    const op = current?.session.undo();
    if (!op) { post({ type: 'idle', status: 'nothing to undo' }); return; }
    log(`undo: ${op}`, 'ok');
    current.version++;
    await saveModel();
    postQuality({ op: 'undo' });
  } else if (data.type === 'send') {
    if (!current) throw new Error('run the reconstruction first');
    await send();
  } else if (data.type === 'store') {
    store = storeFrom(data.store);
  } else if (data.type === 'restoreStore') {  // a project folder without its video: repairs, no re-runs
    const [fp, mp] = await loadSaved();
    if (!fp) throw new Error('this project has no saved session (cache/features.gosfm)');
    reset();
    video = storeKey = null;
    const f = await unpackProject(fp), v = f.json.video;
    post({ type: 'video', info: { project: store.name, name: v?.name ?? '(unknown video)', duration: v?.duration ?? f.json.analysis.times.reduce((m, t) => Math.max(m, t), 0) } });
    await restore(f, mp && await unpackProject(mp));
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
  queue = queue.then(() => handle(data)).catch((e) => post({ type: 'error', message: e?.message ?? String(e) }));
};
