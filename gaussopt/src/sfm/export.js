// export.js — SfM result -> COLMAP sparse model (DESIGN-sfm §4): the cameras.bin / images.bin /
// points3D.bin (written by colmap.js writeColmapModel, re-exported here) that colmap.js and COLMAP read, for one segment, as a PINHOLE camera with the
// keypoints moved to undistorted pixel coordinates; the images themselves are undistorted with
// the same intrinsics by the caller (undistort.js on the GPU in the page, tools/undistort.py).
// The points are filtered for the trainer (DESIGN-sfm §5.5): observations within maxReprojPx,
// at least minObs of them, and a viewing angle of at least minAngleDeg between two of them.
"use strict";

import { intrinsics, imageFromNormalized, normalizedFromImage, project, poseCenter } from './camera.js';
import { triangulationAngle } from './geometry.js';

// mapper: SequentialMapper after reconstruct(); segment: id to export (default: the largest).
// pinhole: false keeps the SfM camera (e.g. OPENCV) and the keypoints' pixels as detected, the
// model of the distorted keyframes (a project's sparse/0, for COLMAP's own tools); the same points.
// A camera per lens (DESIGN-sfm §13 P8): image k uses camera cameraId (1-based), in `cameras`; a
// camera whose frames joined another (mapper.js scanFocalOf) is left out.
// Returns { camera, distorted (camera 1's, as below), cameras (PINHOLE, or the SfM cameras),
//           distortedCameras (the SfM cameras), images: [{ id, name, cameraId, R, t, points2D: [x, y, pid] }],
//           points: [{ id, X, rgb, error, track: [[imageId, point2DIdx]] }] }
export function datasetFromMapper(mapper, { segment = null, maxReprojPx = 2, minObs = 3, minAngleDeg = 1.5, pinhole = true } = {}) {
  const seg = segment ?? mapper.largestSegment();
  const all = mapper.cameras ?? [mapper.camera], live = all.map((_, c) => (mapper.merged?.[c] ?? c) === c);
  let j = 0;
  const sfmCams = all.filter((_, c) => live[c]), newIndex = live.map(l => (l ? j++ : -1));
  const camOf = (k) => newIndex[mapper.camOf?.[k] ?? 0];
  const cameras = sfmCams.map(cam => pinhole ? pinholeOf(cam) : { ...cam, params: [...cam.params] });

  const frames = [];
  for (let k = 0; k < mapper.n; k++) if (mapper.poses[k] && mapper.segment[k] === seg) frames.push(k);
  const imageId = new Map(frames.map((k, i) => [k, i + 1]));
  const images = frames.map(k => ({ id: imageId.get(k), name: mapper.images[k].name, cameraId: camOf(k) + 1, R: mapper.poses[k].R, t: mapper.poses[k].t, points2D: [] }));
  const centers = new Map(frames.map(k => [k, poseCenter(mapper.poses[k])]));
  const minAngle = minAngleDeg * Math.PI / 180;
  const points = [];
  mapper.points.forEach((P) => {
    if (!P || P.obs.size < minObs) return;
    const obs = [];
    for (const [img, kp] of P.obs) {
      if (!imageId.has(img)) return;  // points of this segment only
      const cam = sfmCams[camOf(img)];
      const [u, v] = mapper.pixel(img, kp), [pu, pv, z] = project(cam, mapper.poses[img], P.X);
      const err = z > 0 ? Math.hypot(pu - u, pv - v) : Infinity;
      if (err <= maxReprojPx) obs.push({ img, u, v, err });
    }
    if (obs.length < minObs || !wideEnough(obs.map(o => centers.get(o.img)), P.X, minAngle)) return;
    const id = points.length + 1, track = [];
    for (const { img, u, v } of obs) {
      const im = images[imageId.get(img) - 1];
      track.push([im.id, im.points2D.length]);
      if (!pinhole) { im.points2D.push([u, v, id]); continue; }
      const cam = sfmCams[camOf(img)], [fx, fy, cx, cy] = intrinsics(cameras[camOf(img)]);
      const [x, y] = normalizedFromImage(cam, u, v);
      im.points2D.push([fx * x + cx, fy * y + cy, id]);
    }
    points.push({ id, X: P.X, rgb: [128, 128, 128], error: obs.reduce((s, o) => s + o.err, 0) / obs.length, track });
  });
  const distortedCameras = sfmCams.map(c => ({ ...c, params: [...c.params] }));
  return { segment: seg, camera: cameras[0], distorted: distortedCameras[0], cameras, distortedCameras, images, points };
}

// The PINHOLE camera the keyframes are undistorted to: the same size and center, the focal scaled up
// (the frame cropped) just enough that every pixel of it comes from inside the distorted frame, as
// COLMAP's undistorter with blank_pixels = 0. A barrel lens (k1 < 0) keeps its focal. Uncropped,
// a k1 > 0 lens leaves black borders, which the trainer fits as dark splats.
export function pinholeOf(cam) {
  const [fx, fy, cx, cy] = intrinsics(cam), { width: w, height: h } = cam, n = 64, border = [];
  for (let k = 0; k <= n; k++) {
    const a = k / n;
    border.push([a * w, 0], [a * w, h], [0, a * h], [w, a * h]);
  }
  const inside = (s) => border.every(([u, v]) => {
    const [du, dv] = imageFromNormalized(cam, (u - cx) / (fx * s), (v - cy) / (fy * s));
    return du >= -0.5 && du <= w + 0.5 && dv >= -0.5 && dv <= h + 0.5;
  });
  let lo = 1, hi = 1;
  if (!inside(1)) {
    while (!inside(hi) && hi < 4) { lo = hi; hi *= 1.25; }  // 4: past it, the lens model is not to be trusted
    for (let it = 0; it < 40; it++) { const m = (lo + hi) / 2; if (inside(m)) hi = m; else lo = m; }
  }
  return { model: 'PINHOLE', width: w, height: h, params: [fx * hi, fy * hi, cx, cy] };
}

// Some two of the camera centers see X at least minAngle apart
function wideEnough(centers, X, minAngle) {
  for (let a = 0; a < centers.length; a++) for (let b = a + 1; b < centers.length; b++) if (triangulationAngle(centers[a], centers[b], X) >= minAngle) return true;
  return false;
}

export { writeColmapModel } from '../colmap.js';
