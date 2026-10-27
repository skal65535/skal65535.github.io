// export.js round trip through Gauss-Opt's own COLMAP parsers (colmap.js)
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { reconstruct } from '../src/sfm/reconstruct.js';
import { datasetFromMapper, writeColmapModel, pinholeOf } from '../src/sfm/export.js';
import { parseCamerasBin, parseImagesBin, parsePoints3DBin } from '../src/colmap.js';
import { project, makeCamera, imageFromNormalized } from '../src/sfm/camera.js';
import { assert, assertClose } from './util.js';

// a barrel lens, and a pincushion one, whose PINHOLE camera is cropped (pinholeOf)
for (const k1 of [-0.08, 0.15]) Deno.test(`exported COLMAP model parses back with colmap.js and reprojects as PINHOLE (k1 = ${k1})`, () => {
  const scene = makeSyntheticScene({ numFrames: 20, maxFeatures: 800, numGround: 6000, numBoxes: 6, numFar: 500,
                                     params: [1100, 1100, 960, 540, k1, 0.01, 0, 0] });
  const { mapper } = reconstruct(scene.input, { sweeps: 1 });
  const ds = datasetFromMapper(mapper);
  const files = writeColmapModel(ds);
  const cameras = parseCamerasBin(files['cameras.bin']);
  const images = parseImagesBin(files['images.bin'], cameras);
  const points = parsePoints3DBin(files['points3D.bin']);
  assert(cameras.size === 1 && cameras.get(1).model === 'PINHOLE', 'one PINHOLE camera');
  assert(images.length === ds.images.length && points.count === ds.points.length, 'counts');
  const cam = cameras.get(1);
  const byName = new Map(ds.images.map(im => [im.name, im]));
  let n = 0, err = 0;
  for (const im of images) {
    const src = byName.get(im.name);
    assertClose(im.R, src.R, 1e-9, `${im.name} rotation`);
    assertClose(im.t, src.t, 1e-9, `${im.name} translation`);
    for (let o = 0; o < im.obs2D.length; o += 3) {  // undistorted keypoints reproject through the pinhole
      const i = points.pointIndexById.get(im.obs2D[o + 2]);
      const X = [0, 1, 2].map(k => points.positions[3 * i + k]);
      const q = [0, 1, 2].map(r => im.R[3 * r] * X[0] + im.R[3 * r + 1] * X[1] + im.R[3 * r + 2] * X[2] + im.t[r]);
      err += Math.hypot(cam.fx * q[0] / q[2] + cam.cx - im.obs2D[o], cam.fy * q[1] / q[2] + cam.cy - im.obs2D[o + 1]);
      n++;
    }
  }
  assert(n > 1000 && err / n < 1.5, `${n} observations, mean pinhole reprojection ${(err / n).toFixed(3)} px`);
});

Deno.test('the distorted model (pinhole: false) keeps the SfM camera and the detected pixels', () => {
  const scene = makeSyntheticScene({ numFrames: 20, maxFeatures: 800, numGround: 6000, numBoxes: 6, numFar: 500,
                                     params: [1100, 1100, 960, 540, -0.08, 0.01, 0, 0] });
  const { mapper } = reconstruct(scene.input, { sweeps: 1 });
  const ds = datasetFromMapper(mapper, { pinhole: false }), gs = datasetFromMapper(mapper);
  const files = writeColmapModel(ds);
  const cameras = parseCamerasBin(files['cameras.bin']);
  const images = parseImagesBin(files['images.bin'], cameras);
  const points = parsePoints3DBin(files['points3D.bin']);
  assert(cameras.get(1).model === 'OPENCV' && images.length === gs.images.length && points.count === gs.points.length,
         'an OPENCV camera, the same images and points as the PINHOLE model');
  let n = 0, err = 0;
  for (const im of ds.images) {
    for (const [u, v, pid] of im.points2D) {  // detected pixels reproject through the distorted camera
      const [pu, pv] = project(ds.camera, im, ds.points[pid - 1].X);
      err += Math.hypot(pu - u, pv - v);
      n++;
    }
  }
  assert(n > 1000 && err / n < 1.5, `${n} observations, mean reprojection ${(err / n).toFixed(3)} px`);
});

// A camera per lens (DESIGN-sfm §13 P8): two PINHOLE cameras, each image on its lens's, and the
// undistorted keypoints reproject through their own image's camera
Deno.test('two lenses: a PINHOLE camera each, images on theirs, keypoints reproject through it', () => {
  const scene = makeSyntheticScene({ numFrames: 24, maxFeatures: 800, numGround: 6000, numBoxes: 6, numFar: 500, turns: 0.6, overlap: 6,
    cameras: [[1100, 1100, 960, 540, -0.08, 0.01, 0, 0], [700, 700, 960, 540, -0.03, 0.004, 0, 0]],
    cameraOf: (k) => (k % 3 === 2 ? 1 : 0), priorFocals: [1070, 680] });
  const { mapper } = reconstruct(scene.input, { sweeps: 1, refine: ['f', 'k1', 'k2'] });
  const ds = datasetFromMapper(mapper);
  const files = writeColmapModel(ds);
  const cameras = parseCamerasBin(files['cameras.bin']);
  const images = parseImagesBin(files['images.bin'], cameras);
  const points = parsePoints3DBin(files['points3D.bin']);
  assert(cameras.size === 2 && [...cameras.values()].every(c => c.model === 'PINHOLE'), 'two PINHOLE cameras');
  const lens = new Map(scene.input.images.map(im => [im.name, im.camera + 1]));
  assert(images.every(im => im.cameraId === lens.get(im.name)), 'each image on its lens\'s camera');
  // (no focal check: from this scene's self-calibration, 9% high, both focals end ~5% high; the
  // mapper tests check the focals, photo_runs/NOTES.md)
  let n = 0, err = 0;
  for (const im of images) {
    const cam = im.camera;
    for (let o = 0; o < im.obs2D.length; o += 3) {
      const i = points.pointIndexById.get(im.obs2D[o + 2]);
      const X = [0, 1, 2].map(k => points.positions[3 * i + k]);
      const q = [0, 1, 2].map(r => im.R[3 * r] * X[0] + im.R[3 * r + 1] * X[1] + im.R[3 * r + 2] * X[2] + im.t[r]);
      err += Math.hypot(cam.fx * q[0] / q[2] + cam.cx - im.obs2D[o], cam.fy * q[1] / q[2] + cam.cy - im.obs2D[o + 1]);
      n++;
    }
  }
  assert(n > 1000 && err / n < 1.5, `${n} observations, mean pinhole reprojection ${(err / n).toFixed(3)} px`);
});

// The PINHOLE camera of a pincushion lens (desk.mov: k1 = 0.29, portrait) crops the frame just enough
// that no pixel of it falls outside the distorted keyframe (no black border); a barrel lens keeps its focal
Deno.test('pinholeOf: no pixel of the undistorted frame outside the distorted one, the least crop', () => {
  const inside = (cam, pin) => {
    const [fx, fy, cx, cy] = pin.params, { width: w, height: h } = cam;
    let ok = true;
    for (let k = 0; k <= 200; k++) for (const [u, v] of [[k * w / 200, 0], [k * w / 200, h], [0, k * h / 200], [w, k * h / 200]]) {
      const [du, dv] = imageFromNormalized(cam, (u - cx) / fx, (v - cy) / fy);
      ok &&= du >= -0.5 && du <= w + 0.5 && dv >= -0.5 && dv <= h + 0.5;
    }
    return ok;
  };
  const desk = makeCamera('OPENCV', 1080, 1920, [1706, 1706, 540, 960, 0.29, 0, 0, 0]);
  const pin = pinholeOf(desk), s = pin.params[0] / 1706;
  assert(inside(desk, pin), `desk: a border pixel outside the keyframe at scale ${s}`);
  const looser = { ...pin, params: [pin.params[0] * 0.99, pin.params[1] * 0.99, 540, 960] };
  assert(!inside(desk, looser) && s > 1 && s < 1.2, `desk: scale ${s} is not the least crop`);
  assert(pin.params[2] === 540 && pin.params[3] === 960, 'center kept');
  const barrel = makeCamera('OPENCV', 1920, 1080, [1100, 1100, 960, 540, -0.08, 0.01, 0, 0]);
  assert(pinholeOf(barrel).params[0] === 1100, `barrel lens: focal ${pinholeOf(barrel).params[0]}`);
});
