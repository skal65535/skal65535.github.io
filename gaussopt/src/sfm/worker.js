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
//        | { type: 'photoFilter', filter: { lens?, zoom?, orientation?: [values] } | null }
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
import { SESSION_DEFAULTS, PHOTO_MAPPING } from './defaults.js';
import { readExif, jpegSize } from './exif.js';
import { planPhotos, planMore, focalFromExif, hfovDeg } from './photos.js';
import { coverage, coverageAdvice } from './coverage.js';
import { defaultCamera, withFocal } from './camera.js';
import { datasetFromMapper, writeColmapModel } from './export.js';
import { xyOnly } from './frontend.js';
import { hashString } from './util.js';
import { storeFrom } from '../project/store.js';

const post = (msg, transfer = []) => self.postMessage(msg, transfer);
const log = (text, level = '') => post({ type: 'log', text, level });
const progress = (stage, done, total) => post({ type: 'progress', stage, done, total });
const seconds = (t) => `${((performance.now() - t) / 1000).toFixed(1)} s`;
const same = (a, b, keys) => a && keys.every(k => a[k] === b[k]);
const FEATURE_KEYS = ['maxDim', 'maxFeatures', 'jpegQuality'];  // the options the features depend on
const perFrame = (feats) => Math.round(feats.reduce((s, f) => s + f.keypoints.length / 4, 0) / feats.length);  // mean features per frame
const newId = () => `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
const warnCapped = (capped, what) => {
  if (capped) log(`SIFT: ${capped} ${what}(s) had over ${SIFT_DEFAULTS.maxKeypoints} candidate keypoints: the ones kept depend on the GPU's order`, 'warn');
};

let video = null, gpu = null, current = null;  // current: { session, features, o, version } after a run
const keyframeName = (time) => `t${String(Math.round(time * 1000)).padStart(7, '0')}.jpg`;  // DESIGN-sfm §3
const cache = {};  // analysis, features: { options (that produced them), id, ... }; source: the video of a restored project

function reset() {
  for (const k of Object.keys(cache)) delete cache[k];
  thumbs.clear();
  current = null;
}
// reset(), and nothing open: no video, photo set or OPFS key (a failed restore() keeps them)
function resetAll() {
  reset();
  video = storeKey = photos = null;
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
  warnCapped(capped, 'frame');
}

// Small JPEGs of the keyframes for the page's contact sheet (src/ui/frames.js), made once per name
const THUMB_WIDTH = 320, THUMB_DECODES = 4;
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
    const canvas = new OffscreenCanvas(w, h), ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(bitmap, (w - dw) / 2, (h - dh) / 2, dw, dh);
    bitmap.close();
    thumbs.set(p.name, await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.8 }));
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

// S0b + S1: keyframes (second decode pass at full size): JPEG, gray -> SIFT
async function keyframes(o, analysis) {
  const sel = selection(analysis, o), kept = sel.kept.map(i => analysis.order[i]);
  const blurry = sel.status.filter(s => s === 'blurry').length;
  log(`keyframes: ${kept.length} kept (${blurry} blurry dropped, median threshold ${sel.threshold.toFixed(1)}` +
      (sel.path ? `; path ${(sel.path.at(-1) * o.spacing).toFixed(2)} diagonals)` : ')'));
  if (kept.length < 3) throw new Error(`only ${kept.length} keyframes: lower the spacing or the blur threshold`);
  if (same(cache.features?.options, o, FEATURE_KEYS) && cache.features.kept.join() === kept.join()) {
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
  log(`features: ${feats.length} keyframes at ${w} x ${h}, ${perFrame(feats)} per frame in ${seconds(t)}`);
  // id: a saved model is restored only over the features it was built from
  cache.features = { options: { ...o }, id: newId(), kept, feats, jpegs, width: w, height: h };
  saveFeatures(cache.features);
  return cache.features;
}

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
  const { keypoints, descriptors, capped } = await sift.extract(gray, w, h, { maxFeatures: o.maxFeatures });
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

// A camera model per lens and zoom when the set has several (P8.2), each from its EXIF focal (camera 0's
// is calibrated, the others keep their ratio to it, mapper.js calibrateFocal); null for one camera
function photoCameras(features) {
  const { cameras } = photos.plan, { width: w, height: h } = features;
  if (cameras.length < 2) return null;
  return cameras.map(g => {
    const f = focalFromExif(g.focal35, w, h), cam = defaultCamera('OPENCV', w, h);
    return f ? withFocal(cam, f) : cam;
  });
}

// The mapper options from the run's: a video's, and the start of a photo set's
const baseMapping = (o) => ({ sweeps: o.sweeps, focal: o.focal, ...(o.hfovDeg > 0 ? { hfovDeg: o.hfovDeg } : {}) });

// The mapper options of a photo set: PHOTO_MAPPING, and the EXIF focal as the fallback field of
// view when F does not observe the focal (P2; not as the start: it is 3-4% low on the Pixel 10)
function photoMapping(o, features) {
  const g = photos.plan.cameras[0];
  const f = focalFromExif(g?.focal35, features.width, features.height);
  return { ...PHOTO_MAPPING, ...baseMapping(o), ...(f ? { fallbackHfovDeg: hfovDeg(f, features.width) } : {}) };
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
  const source = video ? { name: video.file.name, duration: video.duration } : cache.source ?? null;
  // a photo set has no frame analysis (S0): its plan stands in for the source
  const a = analysis ? (({ order, sharp, steps, times, options }) => ({ options, order, sharp, steps, times }))(analysis) : null;
  return packProject({ video: source, analysis: a, photos: photos ? { plan: photos.plan } : cache.photoPlan ?? null,
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

// The saved session of the opened video or photo set, if any (a failure is logged)
async function restoreSaved() {
  const [fp, mp] = await loadSaved();
  if (fp) await restore(await unpackProject(fp), mp && await unpackProject(mp)).catch(e => log(e.message, 'warn'));
}

// What a saved features container was made from: { name, duration } of its video, or { name,
// photos } of a photo set (no duration)
function savedSource(json) {
  if (json.photos) return { name: `${json.photos.plan.used.length} photos`, photos: json.photos.plan.used.length, duration: 0 };
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
  if (a) selection(a, mp.json.o);
  log(`restored the saved session (${mp.json.saved}): ${session.images.length} keyframes; Run all starts over`, 'ok');
  postQuality();
  post({ type: 'restored' });
  return true;
}

// The trainer's points: seen by trainMinObs keyframes; by default 3 for a video, 2 for photos (photo
// sets: held-out PSNR higher in all 9 runs of cuisine, salon, minerals x 3 floater settings, +0.15 to
// +0.77 dB, photo_runs/logs/photo_preset_from_page_log.txt)
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
  log(`hand-over: ${images.length} undistorted keyframes, ${ds.points.length} colored points in ${seconds(t)}`);
  return { sparse: writeColmapModel(ds), images };
}

// The cameras for the trainer from the session's current state; null when there is nothing to train
// on (no piece of 2 photos with points: e.g. out_scene2, out_scene4), the advice says what to shoot
async function send() {
  const t0 = performance.now(), { session, features, o } = current;
  if (!session.mapper.poses.some(Boolean)) return null;
  const ds = datasetFromMapper(session.mapper, { minObs: trainMinObs(o) });
  if (ds.images.length < 2 || !ds.points.length) return null;
  const { sparse, images } = await handOver(ds, features, o);
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
  post({ type: 'result', sparse, images, summary }, Object.values(sparse));
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

async function run(options) {
  const o = { ...SESSION_DEFAULTS, ...options };
  const t0 = performance.now();
  const features = photos ? await photoFeatures(o) : await keyframes(o, await analyze(o));
  post({ type: 'keyframes', count: features.feats.length });
  if (photos) await postPhotoThumbs(); else await postThumbs(features);
  if (o.until === 'keyframes') {
    post({ type: 'idle', status: `${features.feats.length} keyframes, ${perFrame(features.feats)} features each` });
    return;
  }
  const { matcher } = await devices();
  const mapping = photos ? photoMapping(o, features) : baseMapping(o);
  // a photo set: every pair matched (the window spans the set, DESIGN-sfm §13.3 P3). The loop
  // stage stays: it merges segments too (cuisine without it, and without regrowSegments: 7 photos
  // in one segment, not 12)
  const window = photos ? Math.max(1, features.feats.length - 1) : o.window, loops = o.loops;
  const cameras = photos ? photoCameras(features) : null;
  const { session } = await reconstructFromFeatures(features.feats, features.width, features.height, matcher,
    { window, loops, mapping, log, progress, ...(cameras ? { cameras } : {}) });
  current = { session, features, o, version: 0 };  // version: bumped by every action and undo
  // repair all scores keyframes against their predecessor in time: not for photos (P4)
  if (!photos && o.autoRepair && session.quality().counts.green < session.images.length) {
    const r = await session.repair();
    log(`repair: ${r.kept}/${r.tried} retries kept, scores ${JSON.stringify(r.quality.counts)} (undo reverts it)`, r.quality.counts.red ? 'warn' : 'ok');
  }
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
  const r = op === 'densify' ? await densify(...args) : op === 'addPhotos' ? await addPhotos(...args) : await current.session[op](...args);
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
  if (f) {
    if (cache.analysis && cache.selection) await save('sharpness.csv', sharpnessCsv(cache.analysis, cache.selection));
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
  } else if (data.type === 'open') {
    resetAll();
    video = await openVideo(data.file);
    const { codec, width, height, fps, duration, rotation, samples } = video;
    post({ type: 'video', info: { name: data.file.name, codec, width, height, fps, duration, rotation, frames: samples.length } });
    storeKey = hashString(`${data.file.name}|${data.file.size}|${data.file.lastModified}`).toString(16);
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
    post({ type: 'project', blob: project, name: `${(video?.file.name ?? 'session').replace(/\.[^.]+$/, '')}.gosfm` });
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
  });
};
