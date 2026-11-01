// decode.e2e.js — src/sfm/video.js decodeFrames falls back to the software decoder when the hardware one fails:
// (1) a thermal camera's H.264 (DECODE_E2E_VIDEO, default ~/viewsec4.mp4) that Chrome's hardware decoder on macOS
// rejects at its 5th sample ('Decoding error.'): every sampled frame delivered once, the fallback recorded;
// (2) a video that decodes (OVERLAY_E2E_VIDEO, default ~/test-parking.mov) with its hardware decoder made to fail
// after 30 frames: the frames delivered before are not delivered again, none is missing.
//   cd e2e && node decode.e2e.js
import { copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { basename, join } from 'node:path';
import { HERE, fail, urlOf, appUrl, run } from './harness.js';

const cached = (src) => {
  if (!existsSync(src)) throw new Error(`no video ${src}`);
  const dir = join(HERE, '.cache', 'videos'), f = join(dir, basename(src));
  mkdirSync(dir, { recursive: true });
  if (!existsSync(f)) copyFileSync(src, f);
  return urlOf(f);
};
const THERMAL = cached(process.env.DECODE_E2E_VIDEO ?? `${process.env.HOME}/viewsec4.mp4`);
const PLAIN = cached(process.env.OVERLAY_E2E_VIDEO ?? `${process.env.HOME}/test-parking.mov`);

await run(async (page) => {
  await page.goto(appUrl(''));
  const r = await page.evaluate(async ({ THERMAL, PLAIN }) => {
    const { openVideo, sampleFrames, decodeFrames } = await import('/gaussopt/src/sfm/video.js');
    const decode = async (url, fps) => {
      const video = await openVideo(await (await fetch(url)).blob());
      const wanted = new Set(sampleFrames(video, fps)), seen = new Map();
      let error = null;
      await decodeFrames(video, wanted, 160, 120, async (ctx, k) => { seen.set(k, (seen.get(k) ?? 0) + 1); }).catch(e => { error = e.message; });
      return { wanted: wanted.size, got: seen.size, twice: [...seen.values()].filter(n => n > 1).length,
               missing: [...wanted].filter(k => !seen.has(k)).length, fallback: video.decoderFallback ?? null, accel: video.config.hardwareAcceleration, error };
    };
    const thermal = await decode(THERMAL, 2);
    // a hardware decoder that fails after 30 frames (the software one as it is)
    const Real = window.VideoDecoder;
    window.VideoDecoder = class extends Real {
      constructor(init) {
        let n = 0, self = null;
        super({ ...init, output: (f) => { init.output(f); if (self?.hw && ++n === 30) queueMicrotask(() => { try { self.close(); } catch {} init.error(new DOMException('injected failure', 'EncodingError')); }); } });
        self = this;
      }
      configure(c) { this.hw = c.hardwareAcceleration === 'prefer-hardware'; return super.configure(c); }
    };
    try { return { thermal, injected: await decode(PLAIN, 6) }; } finally { window.VideoDecoder = Real; }
  }, { THERMAL, PLAIN });
  const ok = (x) => !x.error && x.got === x.wanted && x.twice === 0 && x.missing === 0 && x.accel === 'prefer-software' && x.fallback;
  if (!ok(r.thermal)) fail(`thermal video: ${JSON.stringify(r.thermal)}`);
  if (!ok(r.injected)) fail(`a hardware decoder failing after 30 frames: ${JSON.stringify(r.injected)}`);
  return `thermal: ${r.thermal.got} / ${r.thermal.wanted} frames (the hardware decoder: '${r.thermal.fallback}'); a hardware decoder failing after 30 frames: ` +
         `${r.injected.got} / ${r.injected.wanted} frames, none twice`;
});
