// gallery_stack.e2e.js — The gallery's viewer (gallery/index.html) composites a stack of faint splats as the trainer
// sums it: 300 white splats of opacity 0.01, flat sheets one behind the other across the view, look like one sheet of
// their total opacity 1 - 0.99^300 (0.951), seen from the scene's camera. In an 8-bit canvas each faint splat's share of
// a nearly covered pixel ((1 - A) 0.01, under half a step once A is over 0.8) rounds to 0: the stack stops near 0.8,
// darker than the one sheet (the viewer's half-float canvas, viewer.js render.textureFormat). Two scenes written here
// (scene_glb.js), served in place of the gallery's (index.json and their scene.glb routed), each its center pixel.
//   cd e2e && node gallery_stack.e2e.js
import { dirname } from 'node:path';
import { APP, PORT, fail, run } from './harness.js';
import { writeSceneGlb } from '../src/scene_glb.js';
import { SPLAT_FLOATS, rgbToSh0 } from '../src/colmap.js';

const W = 640, H = 480, N = 300, A = 0.01;
const logit = (a) => Math.log(a / (1 - a));
// sheets at depths 2..3 in front of a camera at the origin looking down +z (OpenCV axes), 4 wide, 1 mm thick, white
function scene(opacities) {
  const n = opacities.length, params = new Float32Array(n * SPLAT_FLOATS);
  opacities.forEach((a, i) => params.set([0, 0, 2 + i / n, logit(a), Math.log(4), Math.log(4), Math.log(1e-3), 0, 1, 0, 0, 0,
                                         rgbToSh0(1), rgbToSh0(1), rgbToSh0(1), 0], i * SPLAT_FLOATS));
  const view = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const cameras = [{ name: 'c0', view, camera: { fx: 500, fy: 500, cx: W / 2, cy: H / 2, width: W, height: H }, zMin: 1, zMax: 4, isValidation: false }];
  return new Uint8Array(writeSceneGlb({ count: n, params, cameras, path: [0], upEstimate: [0, -1, 0] }));
}
const scenes = { 'stack-faint': scene(new Array(N).fill(A)), 'stack-one': scene([1 - (1 - A) ** N]) };

await run(async (page) => {
  await page.route('**/gallery/scenes/index.json', route => route.fulfill({
    json: Object.keys(scenes).map(id => ({ id, title: id, photos: false, about: '' })) }));
  for (const [id, bytes] of Object.entries(scenes))
    await page.route(`**/gallery/scenes/${id}/scene.glb`, route => route.fulfill({ body: Buffer.from(bytes), contentType: 'model/gltf-binary' }));
  const center = {};
  for (const id of Object.keys(scenes)) {
    await page.goto(`http://localhost:${PORT}/gallery/index.html?scene=${id}&mode=photo%20camera&w=${W}&h=${H}&canvas-only&cameras-hidden`);
    await page.waitForFunction(() => document.getElementById('scene-info')?.textContent.length > 0, null, { timeout: 60000 });
    await page.waitForTimeout(1500);
    const png = await page.locator('#main-canvas').screenshot();
    center[id] = await page.evaluate(async (b64) => {  // the canvas' center pixel, from its screenshot
      const img = new Image(); img.src = `data:image/png;base64,${b64}`; await img.decode();
      const c = new OffscreenCanvas(img.width, img.height), g = c.getContext('2d');
      g.drawImage(img, 0, 0);
      return Array.from(g.getImageData(img.width >> 1, img.height >> 1, 1, 1).data.slice(0, 3));
    }, png.toString('base64'));
  }
  const a = center['stack-faint'], b = center['stack-one'], off = Math.max(...a.map((x, i) => Math.abs(x - b[i])));
  if (!(Math.min(...b) > 100)) fail(`the one sheet's center ${b}: not drawn`);
  if (!(off <= 4)) fail(`${N} sheets of opacity ${A}: ${a}, one of their total opacity: ${b} (off by ${off} / 255)`);
  return `${N} sheets of opacity ${A}: ${a}; one sheet of ${(1 - (1 - A) ** N).toFixed(3)}: ${b}`;
}, { static: dirname(APP) });
