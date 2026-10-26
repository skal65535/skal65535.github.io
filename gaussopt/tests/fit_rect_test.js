// The photo's box in the canvas (src/math.js fitRect): the 2D compare modes letterbox the photo in a
// canvas that fills the view, and the split wipe maps the mouse through the same box.
import { fitRect } from '../src/math.js';
import { assert } from './util.js';

Deno.test('fitRect: the largest box of the photo\'s aspect, centered', () => {
  const eq = (r, x, y, w, h) => assert(r.x === x && r.y === y && r.w === w && r.h === h, JSON.stringify(r));
  eq(fitRect(1920, 1080, 960, 540), 0, 0, 960, 540);    // same shape: the whole canvas
  eq(fitRect(1920, 1080, 960, 720), 0, 90, 960, 540);   // a taller canvas: bars above and below
  eq(fitRect(1080, 1920, 960, 540), 328, 0, 304, 540);  // a portrait photo: bars beside it
  eq(fitRect(4000, 3000, 640, 640), 0, 80, 640, 480);
});
