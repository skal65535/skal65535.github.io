// video.js liftGray: SIFT's input with the dark tones lifted ('log'), as it is otherwise ('none')
import { liftGray } from '../src/sfm/video.js';
import { SESSION_DEFAULTS, FEATURE_KEYS } from '../src/sfm/defaults.js';
import { assert } from './util.js';

Deno.test('liftGray: log lifts the darks (5% -> 23%), keeps black and white and the order; none leaves it', () => {
  const g = Uint8Array.from({ length: 256 }, (_, v) => v), l = liftGray(g, 'log');
  assert(l[0] === 0 && l[255] === 255, `ends ${l[0]}, ${l[255]}`);
  assert(l.every((x, k) => k === 0 || x >= l[k - 1]), 'not monotone');
  assert(l[13] === Math.round(255 * Math.log1p(20 * 13 / 255) / Math.log1p(20)) && l[13] >= 55, `13 -> ${l[13]}`);
  assert(l[13] - l[3] > 3 * (13 - 3), `the darks' contrast not lifted: ${l[3]}, ${l[13]}`);
  assert(liftGray(g, 'none') === g, 'none: the same array');
  // 'video': a video's frames lifted, a photo's as it is
  assert(liftGray(g, 'video', true)[13] === l[13] && liftGray(g, 'video', false) === g, "'video': lifted for a video only");
});

// jardin5 mapped from its video: the darkest tenth of the pixels 33 -> 58 SfM points a 1000, the held-out PSNR the same
// off by default: on jardin5's first 14 s the lift took the sunlit lawn's features (a lower level of grass there: the
// rendered surface > 15% deeper than COLMAP's points at 9.6% of them, 4.1% without), and 8192 features cleared the
// shadows better (photo_runs/NOTES.md)
Deno.test("siftLift none by default, a feature option (the features extracted again when it changes)", () => {
  assert(SESSION_DEFAULTS.siftLift === 'none', `siftLift ${SESSION_DEFAULTS.siftLift}`);
  assert(SESSION_DEFAULTS.maxFeatures === 8192, `maxFeatures ${SESSION_DEFAULTS.maxFeatures}`);
  assert(FEATURE_KEYS.includes('siftLift'), 'not in FEATURE_KEYS');
});
