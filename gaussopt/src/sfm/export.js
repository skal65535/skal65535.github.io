// export.js — SfM result -> COLMAP sparse model (DESIGN-sfm §4): the cameras.bin / images.bin /
// points3D.bin that colmap.js (and COLMAP) read, for one segment, as a PINHOLE camera with the
// keypoints moved to undistorted pixel coordinates; the images themselves are undistorted with
// the same intrinsics by the caller (undistort.js on the GPU in the page, tools/undistort.py).
// The points are filtered for the trainer (DESIGN-sfm §5.5): observations within maxReprojPx,
// at least minObs of them, and a viewing angle of at least minAngleDeg between two of them.
"use strict";

import { mat3ToQuat } from '../math.js';
import { CAMERA_MODELS, intrinsics, normalizedFromImage, project, poseCenter } from './camera.js';
import { triangulationAngle } from './geometry.js';

// mapper: SequentialMapper after reconstruct(); segment: id to export (default: the largest).
// pinhole: false keeps the SfM camera (e.g. OPENCV) and the keypoints' pixels as detected, the
// model of the distorted keyframes (a project's sparse/0, for COLMAP's own tools); the same points.
// Returns { camera (PINHOLE, or the SfM camera), distorted (the SfM camera),
//           images: [{ id, name, R, t, points2D: [x, y, pid] }], points: [{ id, X, rgb, error, track: [[imageId, point2DIdx]] }] }
export function datasetFromMapper(mapper, { segment = null, maxReprojPx = 2, minObs = 3, minAngleDeg = 1.5, pinhole = true } = {}) {
  const counts = new Map();
  for (let k = 0; k < mapper.n; k++) if (mapper.poses[k]) counts.set(mapper.segment[k], (counts.get(mapper.segment[k]) ?? 0) + 1);
  const seg = segment ?? [...counts].sort((a, b) => b[1] - a[1])[0][0];
  const cam = mapper.camera;
  const [fx, fy, cx, cy] = intrinsics(cam);
  const camera = pinhole ? { model: 'PINHOLE', width: cam.width, height: cam.height, params: [fx, fy, cx, cy] }
                         : { ...cam, params: [...cam.params] };

  const frames = [];
  for (let k = 0; k < mapper.n; k++) if (mapper.poses[k] && mapper.segment[k] === seg) frames.push(k);
  const imageId = new Map(frames.map((k, i) => [k, i + 1]));
  const images = frames.map(k => ({ id: imageId.get(k), name: mapper.images[k].name, R: mapper.poses[k].R, t: mapper.poses[k].t, points2D: [] }));
  const centers = new Map(frames.map(k => [k, poseCenter(mapper.poses[k])]));
  const minAngle = minAngleDeg * Math.PI / 180;
  const points = [];
  mapper.points.forEach((P) => {
    if (!P || P.obs.size < minObs) return;
    const obs = [];
    for (const [img, kp] of P.obs) {
      if (!imageId.has(img)) return;  // points of this segment only
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
      const [x, y] = normalizedFromImage(cam, u, v);
      im.points2D.push([fx * x + cx, fy * y + cy, id]);
    }
    points.push({ id, X: P.X, rgb: [128, 128, 128], error: obs.reduce((s, o) => s + o.err, 0) / obs.length, track });
  });
  return { segment: seg, camera, distorted: { ...cam, params: [...cam.params] }, images, points };
}

// Some two of the camera centers see X at least minAngle apart
function wideEnough(centers, X, minAngle) {
  for (let a = 0; a < centers.length; a++) for (let b = a + 1; b < centers.length; b++) if (triangulationAngle(centers[a], centers[b], X) >= minAngle) return true;
  return false;
}

// COLMAP binary files (little endian) for a dataset of datasetFromMapper's shape
export function writeColmapModel({ camera, images, points }) {
  const enc = new TextEncoder();
  const cameras = new Writer();
  cameras.u64(1).i32(1).i32(CAMERA_MODELS[camera.model][0]).u64(camera.width).u64(camera.height);
  for (const p of camera.params) cameras.f64(p);

  const ims = new Writer();
  ims.u64(images.length);
  for (const im of images) {
    const [qw, qx, qy, qz] = mat3ToQuat(im.R);
    ims.i32(im.id).f64(qw).f64(qx).f64(qy).f64(qz).f64(im.t[0]).f64(im.t[1]).f64(im.t[2]).i32(1);
    ims.bytes(enc.encode(im.name)).u8(0).u64(im.points2D.length);
    for (const [x, y, pid] of im.points2D) ims.f64(x).f64(y).i64(pid);
  }

  const pts = new Writer();
  pts.u64(points.length);
  for (const p of points) {
    pts.u64(p.id).f64(p.X[0]).f64(p.X[1]).f64(p.X[2]).u8(p.rgb[0]).u8(p.rgb[1]).u8(p.rgb[2]).f64(p.error).u64(p.track.length);
    for (const [imgId, idx] of p.track) pts.i32(imgId).i32(idx);
  }
  return { 'cameras.bin': cameras.buffer(), 'images.bin': ims.buffer(), 'points3D.bin': pts.buffer() };
}

class Writer {
  constructor() { this.buf = new ArrayBuffer(1 << 16); this.dv = new DataView(this.buf); this.off = 0; }
  grow(n) {
    if (this.off + n <= this.buf.byteLength) return;
    const next = new ArrayBuffer(Math.max(2 * this.buf.byteLength, this.off + n));
    new Uint8Array(next).set(new Uint8Array(this.buf, 0, this.off));
    this.buf = next; this.dv = new DataView(next);
  }
  u8(v) { this.grow(1); this.dv.setUint8(this.off, v); this.off += 1; return this; }
  i32(v) { this.grow(4); this.dv.setInt32(this.off, v, true); this.off += 4; return this; }
  u64(v) { this.grow(8); this.dv.setBigUint64(this.off, BigInt(v), true); this.off += 8; return this; }
  i64(v) { this.grow(8); this.dv.setBigInt64(this.off, BigInt(v), true); this.off += 8; return this; }
  f64(v) { this.grow(8); this.dv.setFloat64(this.off, v, true); this.off += 8; return this; }
  bytes(a) { this.grow(a.length); new Uint8Array(this.buf, this.off, a.length).set(a); this.off += a.length; return this; }
  buffer() { return this.buf.slice(0, this.off); }
}
