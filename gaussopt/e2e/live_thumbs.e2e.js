// live_thumbs.e2e.js — The keyframes' thumbnails as they are extracted (worker.js liveThumbs): a video's run to its
// keyframes has thumbnails in the frames sheet while the status still says it is extracting them (keyframes + SIFT)
// or tracking the points, before the run ends; at the end one per keyframe, none twice. The SfM view before any
// mapping says what comes there (not blank).
//   cd e2e && node live_thumbs.e2e.js   (SFM_E2E_VIDEO, SFM_E2E_SECONDS: the clip, harness.js)
import { fail, clip, urlOf, appUrl, run } from './harness.js';

const video = clip();

await run(async (page) => {
  await page.goto(appUrl(`?video=${encodeURIComponent(urlOf(video))}&sfm_autoSend=0`));
  await page.waitForFunction(() => window.app?.sfm?.state.status === 'ready', null, { timeout: 120000, polling: 500 });
  const empty = await page.evaluate(() => document.querySelector('#sfm-theatre .th-stage').textContent);
  if (empty !== 'Nothing mapped yet') fail(`the SfM view before a mapping: '${empty}'`);
  await page.evaluate(() => {
    const app = window.app;
    window.seen = { extracting: 0, tracking: 0 };  // the most thumbnails shown during each stage
    window.poll = setInterval(() => {
      const s = app.sfm.state.status, n = app.frames.order.length;
      if (/^keyframes \+ SIFT/.test(s)) window.seen.extracting = Math.max(window.seen.extracting, n);
      if (/^tracking points/.test(s)) window.seen.tracking = Math.max(window.seen.tracking, n);
    }, 50);
    app.sfm.run('keyframes');
  });
  await page.waitForFunction(() => window.app.sfm.running, null, { timeout: 60000 });
  await page.waitForFunction(() => !window.app.sfm.running, null, { timeout: 15 * 60000, polling: 500 });
  const r = await page.evaluate(() => {
    clearInterval(window.poll);
    const app = window.app, order = app.frames.order;
    return { ...window.seen, shown: order.length, distinct: new Set(order).size, keyframes: app.sfm.keyframeCount, status: app.sfm.state.status };
  });
  console.log(JSON.stringify(r));
  if (!(r.keyframes > 0) || r.shown !== r.keyframes || r.distinct !== r.shown) fail(`at the end: ${r.shown} thumbnails (${r.distinct} distinct), ${r.keyframes} keyframes`);
  if (!(r.extracting > 0 || r.tracking > 0)) fail(`no thumbnail before the keyframes were done (extracting: ${r.extracting}, tracking: ${r.tracking})`);
  return `${r.extracting} thumbnails shown while extracting, ${r.tracking} while tracking, ${r.shown} at the end for ${r.keyframes} keyframes`;
});
