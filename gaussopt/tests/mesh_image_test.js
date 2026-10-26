// mesh_image.js mipLevels: down to 1 x 1, each texel the alpha-weighted mean of the 2 x 2 below it (the
// texels off a texture's faces, alpha 0, do not blur into the faces' colors), the plain mean when all four have alpha 0.
import { mipLevels } from '../src/mesh_image.js';
import { assert } from './util.js';

Deno.test('mipLevels: alpha-weighted 2 x 2 means down to 1 x 1', () => {
  // 4 x 2: left half red (alpha 255) next to blue off the faces (alpha 0); right half two fill colors, alpha 0
  const px = [[255, 0, 0, 255], [0, 0, 255, 0], [10, 20, 30, 0], [30, 40, 50, 0],
              [255, 0, 0, 255], [0, 0, 255, 0], [10, 20, 30, 0], [30, 40, 50, 0]];
  const levels = mipLevels({ data: new Uint8ClampedArray(px.flat()), width: 4, height: 2 });
  assert(levels.length === 3 && levels[1].width === 2 && levels[1].height === 1 && levels[2].width === 1, levels.map(l => `${l.width}x${l.height}`).join(' '));
  const l1 = Array.from(levels[1].data);
  assert(l1.slice(0, 4).join() === '255,0,0,255', `the faces' red, not mixed with blue: ${l1.slice(0, 4)}`);
  assert(l1.slice(4, 8).join() === '20,30,40,0', `plain mean off the faces: ${l1.slice(4, 8)}`);
  assert(Array.from(levels[2].data).join() === '255,0,0,255', `top: ${Array.from(levels[2].data)}`);
});
