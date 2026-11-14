// flight.e2e.js — A DJI video's flight record through the page (src/sfm/flight.js): opened with ?video=, its djmd
// track is read (the fixes, the first one's position as tests/flight_test.js has it), the GPS track drawn once
// there are cameras: cameras made from the track itself (a similarity of it at keyframe times) give the track back
// within 1e-6 m, and the line buffer holds its segments (each vertex on the fitted track); with show_gps off, none.
//   cd e2e && node flight.e2e.js   (FLIGHT_E2E_VIDEO: a DJI video, default canyon2.mp4 in .cache/videos)
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { HERE, fail, urlOf, appUrl, run } from './harness.js';

const video = process.env.FLIGHT_E2E_VIDEO ?? join(HERE, '.cache', 'videos', 'canyon2.mp4');
if (!existsSync(video)) throw new Error(`no video ${video} (set FLIGHT_E2E_VIDEO)`);

await run(async (page) => {
  await page.goto(appUrl(`?video=${encodeURIComponent(urlOf(video))}`));
  await page.waitForFunction(() => window.app?.telemetry?.t.length > 0, null, { timeout: 120000, polling: 200 });
  const r = await page.evaluate(async () => {
    const app = window.app, tel = app.telemetry;
    const { setPose } = await import(new URL('src/colmap.js', location.href).href);
    const { positionAt } = await import(new URL('src/sfm/flight.js', location.href).href);
    // cameras on the track: a scene unit 20 m, turned about z, one keyframe a second
    const c = Math.cos(1), s = Math.sin(1), world = (p) => [(c * p[0] - s * p[1]) / 20, (s * p[0] + c * p[1]) / 20 + 1, p[2] / 20];
    const camera = { width: 640, height: 360, fx: 500, fy: 500, cx: 320, cy: 180 };
    const images = [];
    for (let ms = 1000; ms < 1000 * tel.t.at(-1); ms += 1000) {
      const im = setPose({ name: `t${String(ms).padStart(7, '0')}.jpg`, camera }, [1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]);
      im.center = world(positionAt(tel, ms / 1000));
      images.push(im);
    }
    // the GPS track's vertices in what the renderer writes into its line buffer (orange, render.js COL_GPS)
    const q = app.renderer.device.queue, write = q.writeBuffer.bind(q);
    let gps = null;
    q.writeBuffer = (buf, off, data, ...rest) => {
      if (buf.label === 'wireframes') {
        const u = new Uint32Array(data.buffer, data.byteOffset, data.byteLength / 4), f = new Float32Array(u.buffer, u.byteOffset, u.length);
        gps = [];
        for (let k = 0; k < u.length; k += 4) if (u[k + 3] === 0xff0099ff) gps.push([f[k], f[k + 1], f[k + 2]]);
      }
      return write(buf, off, data, ...rest);
    };
    const drawn = (on) => { gps = null; app.params.show_gps = on; app._refreshWireframes(); return gps; };
    app.colmap = { images, points3D: null };
    app.octree ??= { leaves: [], sceneRadius: 1 };  // no dataset: the wireframes need an octree, none drawn
    const off = drawn(false), on = drawn(true);
    q.writeBuffer = write;
    const fit = app.gpsFit.fit, far = on ? Math.max(...on.map((p, k) => { const t = fit.points[(k + 1) >> 1]; return Math.hypot(p[0] - t[0], p[1] - t[1], p[2] - t[2]); })) : null;
    return { fixes: tel.t.length, first: [tel.lat[0], tel.lon[0], tel.alt[0]], fit: fit && { ...fit, points: fit.points.length },
             off: off?.length ?? 0, on: on?.length ?? 0, far, log: app.logText('sfm').split('\n').filter(l => /GPS/.test(l)) };
  });
  console.log(JSON.stringify(r));
  if (!(r.fixes > 1000)) fail(`${r.fixes} fixes`);
  if (video.endsWith('canyon2.mp4') && (r.fixes !== 7709 || Math.abs(r.first[0] - 42.72580961) > 1e-8 || Math.abs(r.first[2] - 1711.39) > 1e-3)) fail(`canyon2: ${JSON.stringify(r)}`);
  if (!r.fit || r.fit.residual > 1e-6 || Math.abs(r.fit.scale - 1 / 20) > 1e-9) fail(`the fit: ${JSON.stringify(r.fit)}`);
  if (r.off !== 0 || r.on !== 2 * (r.fit.points - 1) || !(r.far < 1e-5)) fail(`the track's line: ${r.off} vertices off, ${r.on} on, ${r.fit.points} points, ${r.far} from them`);
  if (!r.log.some(l => /GPS track \(djmd\)/.test(l)) || !r.log.some(l => /fitted to \d+ cameras/.test(l))) fail(`log: ${r.log}`);
  if (!process.exitCode) console.log('flight.e2e: ok');
});
