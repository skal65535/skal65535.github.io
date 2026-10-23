import { makeCamera, defaultCamera, distort, undistort, imageFromNormalized, normalizedFromImage, project,
         poseCenter, poseFromCenter, inFieldOfView } from '../src/sfm/camera.js';
import { assertClose, rng, randomRotation } from './util.js';

const CAMERAS = [
  makeCamera('OPENCV', 1920, 1080, [1230, 1231, 960, 540, 0.005, -0.002, 0.001, 0.0002]),  // jardin5
  makeCamera('OPENCV', 1920, 1080, [900, 900, 960, 540, -0.25, 0.08, 0.001, -0.0005]),     // wide lens
  makeCamera('SIMPLE_RADIAL', 1920, 1080, [1200, 960, 540, -0.1]),
  makeCamera('RADIAL', 1920, 1080, [1200, 960, 540, -0.1, 0.02]),
];

Deno.test('undistort inverts distort over the whole image', () => {
  for (const cam of CAMERAS) {
    for (let u = 0; u <= 1920; u += 160) {
      for (let v = 0; v <= 1080; v += 120) {
        const [x, y] = normalizedFromImage(cam, u, v);
        assertClose(imageFromNormalized(cam, x, y), [u, v], 1e-6, `${cam.model} pixel (${u}, ${v})`);
      }
    }
    assertClose(undistort(cam, ...distort(cam, 0.3, -0.2)), [0.3, -0.2], 1e-10, cam.model);
  }
});

Deno.test('project matches the pinhole model and poses round-trip through centers', () => {
  const r = rng(7);
  const cam = makeCamera('PINHOLE', 1920, 1080, [1000, 1100, 960, 540]);
  for (let i = 0; i < 50; i++) {
    const R = randomRotation(r), C = [r() * 10, r() * 10, r() * 10];
    const pose = poseFromCenter(R, C);
    assertClose(poseCenter(pose), C, 1e-10, 'center');
    const X = [0, 1, 2].map(k => C[k] + 5 * R[6 + k] + R[k]);  // 5 ahead, 1 right
    assertClose(project(cam, pose, X), [960 + 1000 / 5, 540, 5], 1e-9, 'projection');
  }
});

Deno.test("defaultCamera uses COLMAP's 1.2 max(w, h) focal", () => {
  assertClose(defaultCamera('OPENCV', 1920, 1080).params, [2304, 2304, 960, 540, 0, 0, 0, 0], 0);
  assertClose(defaultCamera('SIMPLE_RADIAL', 1280, 720).params, [1536, 640, 360, 0], 0);
});

Deno.test('inFieldOfView rejects points whose distorted projection folds back into the frame', () => {
  const cam = CAMERAS[0];
  const pose = poseFromCenter([1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]);
  const X = [4.8, 0, 1];  // 78 deg off-axis, the frame's edge is at 38 deg: jardin5's polynomial maps it to u ~ 1293
  const [u, v] = project(cam, pose, X);
  assertClose([u > 0 && u < 1920 && v > 0 && v < 1080 ? 1 : 0], [1], 0, 'polynomial folds it inside the frame');
  assertClose([inFieldOfView(cam, pose, X) ? 1 : 0, inFieldOfView(cam, pose, [0.5, 0.2, 1]) ? 1 : 0], [0, 1], 0, 'FOV test');
});
