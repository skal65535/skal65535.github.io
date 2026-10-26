// photos.js — A photo set as keyframes (DESIGN-sfm §13.3 P1, P2, P8), DOM-free: capture order,
// names, labels (lens, zoom, orientation, kept separately), which photos to rotate by 90°, and
// the cameras (one per device, lens and zoom). The worker decodes; this decides.
"use strict";

import { captureTime } from './exif.js';

const DIAGONAL_35MM = Math.hypot(36, 24);  // mm: the 35 mm equivalent focal is relative to it

// Labels of one photo: lens and zoom (from EXIF) decide its camera; orientation (as shown, after
// the EXIF orientation) is a label only, since portraits are rotated (P8)
function photoLabels(exif, width, height) {
  const lens = exif.lensModel || (exif.focalLength ? `${+exif.focalLength.toFixed(3)} mm` : 'unknown lens');
  const zoom = exif.digitalZoomRatio > 0 ? +exif.digitalZoomRatio.toFixed(2) : 1;
  return { device: [exif.make, exif.model].filter(Boolean).join(' ') || 'unknown camera', lens, zoom,
           orientation: width >= height ? 'landscape' : 'portrait' };
}

const cameraKey = (l) => `${l.device} | ${l.lens} | ${l.zoom}x`;

// items: [{ fileName, width, height (as shown, after the EXIF orientation), exif }]; filter(labels):
// the photos to use (the label filters of P8.5, e.g. only the ultrawide ones), default all.
// Returns {
//   photos: [{ item (index in items), name, time (s from the first), labels, key, rotate, camera }] in capture order,
//   groups: [{ key, labels, count, focal35 }] by size (one per device, lens and zoom),
//   cameras: the groups used, a camera model each (camera k = cameras[k]; camera 0 the largest),
//   chosen: cameras[0].key, used: the photos of one size once rotated (camera: their group's index in cameras),
//   size: [w, h] of those, skipped: [{ name, reason }], t0: the first capture time (ms) or null,
//   digits: the width of the names' number }
// Photos of the minority orientation of the whole set (filter or not) are rotated by 90°. focal35 is the group's
// 35 mm equivalent focal times its digital zoom (the Pixel 10 reports the lens's 24 mm at 2x zoom;
// COLMAP's focal of those photos is 1.98x the 1x ones, DESIGN-sfm §13.2).
export function planPhotos(items, { filter = null } = {}) {
  const order = items.map((it, k) => ({ it, k, t: captureTime(it.exif) }))
    .sort((a, b) => (a.t ?? Infinity) - (b.t ?? Infinity) || (a.it.fileName < b.it.fileName ? -1 : a.it.fileName > b.it.fileName ? 1 : 0));
  // at least 3 digits: photos added later (planMore) keep name order up to 999
  const t0 = order.find(o => o.t !== null)?.t ?? null, digits = Math.max(3, String(order.length).length);
  const photos = order.map(({ it, k, t }, rank) => {
    const labels = photoLabels(it.exif, it.width, it.height);
    const stem = it.fileName.replace(/\.[^.]*$/, '').replace(/[^A-Za-z0-9._-]+/g, '_');
    return { item: k, name: `p${String(rank + 1).padStart(digits, '0')}_${stem}.jpg`, time: t !== null && t0 !== null ? (t - t0) / 1000 : rank,
             labels, key: cameraKey(labels), rotate: false, camera: -1, width: it.width, height: it.height,
             focal35: it.exif.focalLength35mm > 0 ? it.exif.focalLength35mm * labels.zoom : null };
  });
  const byKey = new Map();
  for (const p of photos) { if (!byKey.has(p.key)) byKey.set(p.key, []); byKey.get(p.key).push(p); }
  const groups = [...byKey].map(([key, ps]) => ({ key, labels: { ...ps[0].labels, orientation: undefined }, count: ps.length, focal35: ps[0].focal35 }))
    .sort((a, b) => b.count - a.count);
  const wanted = photos.filter(p => !filter || filter(p.labels));
  // the majority orientation of the whole set (landscape on a tie), whatever the filter: a filter
  // change then rotates no photo differently, and their features stay cached (worker.js photoFeatures)
  const landscape = photos.filter(p => p.labels.orientation === 'landscape').length * 2 >= photos.length;
  for (const p of photos) p.rotate = (p.labels.orientation === 'landscape') !== landscape;
  const shown = (p) => (p.rotate ? [p.height, p.width] : [p.width, p.height]);
  // one size for all: the most common once rotated
  const sizes = new Map();
  for (const p of wanted) { const s = shown(p).join('x'); sizes.set(s, (sizes.get(s) ?? 0) + 1); }
  const size = [...sizes].sort((a, b) => b[1] - a[1])[0]?.[0].split('x').map(Number) ?? null;
  const skipped = [], used = [];
  for (const p of photos) {
    if (filter && !filter(p.labels)) skipped.push({ name: p.name, reason: 'filtered out' });
    else if (shown(p).join('x') !== size.join('x')) skipped.push({ name: p.name, reason: `size ${shown(p).join(' x ')}, not ${size.join(' x ')}` });
    else used.push(p);
  }
  const cameras = groups.filter(g => used.some(p => p.key === g.key));
  for (const p of used) p.camera = cameras.findIndex(g => g.key === p.key);
  return { photos, groups, cameras, chosen: cameras[0]?.key ?? null, used, size, skipped, t0, digits };
}

// Photos added to a planned set (Add photos, P4): items as planPhotos's. The ones of the set's cameras
// (a lens and zoom it has) and size (once rotated to its orientation) are named after the set's last, in capture order, with
// times from the set's first; the others, files already in the set (by name) and the ones past what
// the set's name width can number are skipped. Returns { photos (as plan.photos, item: index in
// items), skipped: [{ name, reason }] }.
export function planMore(plan, items) {
  const stems = new Set(plan.photos.map(p => p.name.replace(/^p\d+_/, '')));
  const digits = plan.digits ?? plan.photos[0].name.match(/^p(\d+)_/)[1].length;  // a plan saved before digits
  const landscape = plan.size[0] >= plan.size[1];
  const order = items.map((it, k) => ({ it, k, t: captureTime(it.exif) }))
    .sort((a, b) => (a.t ?? Infinity) - (b.t ?? Infinity) || (a.it.fileName < b.it.fileName ? -1 : a.it.fileName > b.it.fileName ? 1 : 0));
  const photos = [], skipped = [];
  let rank = plan.photos.length;
  for (const { it, k, t } of order) {
    const labels = photoLabels(it.exif, it.width, it.height), key = cameraKey(labels);
    const stem = it.fileName.replace(/\.[^.]*$/, '').replace(/[^A-Za-z0-9._-]+/g, '_');
    const rotate = (labels.orientation === 'landscape') !== landscape;
    const shown = rotate ? [it.height, it.width] : [it.width, it.height];
    if (stems.has(`${stem}.jpg`)) { skipped.push({ name: it.fileName, reason: 'already in the set' }); continue; }
    const camera = (plan.cameras ?? [{ key: plan.chosen }]).findIndex(g => g.key === key);
    if (camera < 0) { skipped.push({ name: it.fileName, reason: `another camera (${key}) than the set's` }); continue; }
    if (shown.join('x') !== plan.size.join('x')) { skipped.push({ name: it.fileName, reason: `size ${shown.join(' x ')}, not ${plan.size.join(' x ')}` }); continue; }
    if (String(rank + 1).length > digits) { skipped.push({ name: it.fileName, reason: `more than ${10 ** digits - 1} photos: start a new set` }); continue; }
    rank++;
    stems.add(`${stem}.jpg`);
    photos.push({ item: k, name: `p${String(rank).padStart(digits, '0')}_${stem}.jpg`,
                  time: t !== null && plan.t0 !== null ? (t - plan.t0) / 1000 : rank - 1,
                  labels, key, rotate, camera, width: it.width, height: it.height,
                  focal35: it.exif.focalLength35mm > 0 ? it.exif.focalLength35mm * labels.zoom : null });
  }
  return { photos, skipped };
}

// The chosen group's focal in pixels for an image `width` x `height` (as used), from its 35 mm
// equivalent, or null; and the horizontal field of view it implies (the mapper's fallback, P2)
export function focalFromExif(focal35, width, height) {
  return focal35 > 0 ? focal35 * Math.hypot(width, height) / DIAGONAL_35MM : null;
}
export const hfovDeg = (focalPx, width) => 2 * Math.atan(width / 2 / focalPx) * 180 / Math.PI;
