// S0 (DESIGN-sfm §5.1): MP4 demuxing, and keyframe selection against video2splat.py `select`.
import { demux } from '../src/sfm/mp4.js';
import { resizeArea, sharpness, flowPyramid, goodFeatures, trackLK, motion, step, zoomEvents, selectFrames, analyzeFrame } from '../src/sfm/frames.js';
import { makeRng } from '../src/math.js';
import { assert, assertClose, SLOW, env, exists } from './util.js';

const HOME = env('HOME');
const VIDEO = env('SFM_VIDEO') ?? `${HOME}/gskal/brush/jardin5/jardin5.mp4`;
const SHARPNESS_CSV = env('SFM_SHARPNESS') ?? `${HOME}/gskal/brush/jardin5/out/sharpness.csv`;

async function denoFileReader(path) {
  const f = await Deno.open(path), size = (await f.stat()).size;
  return {
    size,
    read: async (offset, length) => {
      const b = new Uint8Array(Math.max(0, Math.min(length, size - offset)));
      await f.seek(offset, Deno.SeekMode.Start);
      for (let r = 0; r < b.length;) { const k = await f.read(b.subarray(r)); if (!k) break; r += k; }
      return b;
    },
    close: () => f.close(),
  };
}

// Smooth random texture, sampled at (x - dx, y - dy): a known subpixel translation; zoom: scaled
// by that about the image center first
function texture(w, h, dx = 0, dy = 0, seed = 3, zoom = 1) {
  const r = makeRng(seed), blobs = Array.from({ length: 300 }, () => [r() * w, r() * h, 2 + 6 * r(), 200 * (r() - 0.5)]);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 128;
      const sx = w / 2 + (x - dx - w / 2) / zoom, sy = h / 2 + (y - dy - h / 2) / zoom;
      for (const [cx, cy, s, a] of blobs) {
        const d2 = (sx - cx) ** 2 + (sy - cy) ** 2;
        if (d2 < 9 * s * s) v += a * Math.exp(-d2 / (2 * s * s));
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

Deno.test('area resize: integer factors average blocks, totals are preserved', () => {
  const r = makeRng(1), w = 12, h = 9, src = Float32Array.from({ length: w * h }, () => 255 * r());
  const out = resizeArea(src, w, h, 4, 3);
  assertClose(out[0], [0, 1, 2].flatMap(y => [0, 1, 2].map(x => src[y * w + x])).reduce((a, b) => a + b) / 9, 1e-3);
  const odd = resizeArea(src, w, h, 5, 4);  // fractional weights
  assertClose(odd.reduce((a, b) => a + b) * (w * h) / 20, src.reduce((a, b) => a + b), 1e-2, 'mean');
});

Deno.test('sharpness: variance of the Laplacian (flat 0, checkerboard 64 a^2)', () => {
  const w = 16, h = 16;
  assertClose(sharpness(new Float32Array(w * h).fill(9), w, h), 0, 1e-9);
  const a = 10, board = Float32Array.from({ length: w * h }, (_, k) => ((k % w) + Math.floor(k / w)) % 2 ? a : -a);
  // interior Laplacian is +-8a; reflect-101 borders keep the pattern, so the variance is (8a)^2
  assertClose(sharpness(board, w, h), 64 * a * a, 1e-6);
});

Deno.test('pyramidal LK recovers a known translation (subpixel and 25 px)', () => {
  const w = 320, h = 200, A = flowPyramid(texture(w, h), w, h);
  for (const [dx, dy] of [[0.37, -0.81], [25.2, 11.6]]) {
    const B = flowPyramid(texture(w, h, dx, dy), w, h);
    const pts = goodFeatures(A[0], { maxCorners: 200 });
    assert(pts.length > 100, `${pts.length} corners`);
    const tracked = trackLK(A, B, pts);
    const err = tracked.flatMap((q, k) => q ? [Math.hypot(q[0] - pts[k][0] - dx, q[1] - pts[k][1] - dy)] : []).sort((a, b) => a - b);
    const stay = pts.filter(([x, y]) => x + dx < w - 1 && y + dy < h - 1 && x + dx > 0 && y + dy > 0).length;  // still in the image
    assert(err.length > 0.9 * stay, `(${dx}, ${dy}): ${err.length} of ${stay} tracked`);
    assert(err[err.length >> 1] < 0.05, `(${dx}, ${dy}): median error ${err[err.length >> 1].toFixed(3)} px`);
    assertClose(motion(A, B), Math.hypot(dx, dy) / Math.hypot(w, h), 0.05 / Math.hypot(w, h) + 1e-4, 'motion');
  }
});

// A zoom scales every track alike: step() reads it from the similarity fitted to the tracks
Deno.test('zoom step: a known zoom of the image (x1.1, with a translation)', () => {
  const w = 320, h = 200, A = flowPyramid(texture(w, h), w, h);
  const B = flowPyramid(texture(w, h, 3.2, -1.7, 3, 1.1), w, h);
  const s = step(A, B), still = step(A, A);
  assert(Math.abs(s.scale - 1.1) < 0.003 && s.fit > 0.9, `scale ${s.scale.toFixed(4)}, fit ${s.fit.toFixed(2)}`);
  assert(Math.abs(still.scale - 1) < 1e-3 && still.fit > 0.99, `still: scale ${still.scale.toFixed(4)}, fit ${still.fit.toFixed(2)}`);
});

// zoomEvents (DESIGN-sfm §5.1): 4 s at 30 fps, walking (0.3% a frame); at 1 s a zoom in (x1.12 a frame for
// 5 frames, all tracks fit), at 2.5 s a forward flight's expansion (4% a frame, half the tracks fit: depth)
Deno.test('zoom events: a zoom in splits the video, walking and a forward flight do not', () => {
  const n = 120, times = Array.from({ length: n }, (_, i) => i / 30);
  const steps = times.map((_, i) => (i >= 30 && i < 35 ? { scale: 1.12, fit: 0.98 } : i >= 75 && i < 90 ? { scale: 1.04, fit: 0.5 }
                                                         : { scale: 1.003, fit: 0.97 }));
  const z = zoomEvents(steps, times);
  assert(z.events.length === 1 && z.events[0].from === 30 && z.events[0].to === 34, JSON.stringify(z.events));
  assertClose(z.events[0].zoom, 1.12 ** 5, 1e-9, 'zoom');
  assert(z.span[29] === 0 && z.span[30] === -1 && z.span[33] === -1 && z.span[34] === 1 && z.span[n - 1] === 1, `spans ${Array.from(z.span).join('')}`);
  assertClose(z.level[1], 1.12 ** 5, 1e-9, 'level');
  assert(zoomEvents(steps, times, { zoomRate: 0 }).events.length === 0, 'zoomRate 0: off');
  // a lost step (null) inside the zoom stays in its event; a zoom of 10% in all is not one (zoomMin)
  steps[32] = null;
  assert(zoomEvents(steps, times).events.length === 1, 'a lost step');
  const small = times.map((_, i) => (i >= 30 && i < 32 ? { scale: 1.045, fit: 0.98 } : { scale: 1.003, fit: 0.97 }));
  assert(zoomEvents(small, times).events.length === 0, 'under zoomMin');
  // frame selection never picks a frame inside a zoom
  const sel = selectFrames({ sharpness: times.map((_, i) => (i === 31 ? 1000 : 100)), motion: times.map(() => 0.05), times, span: z.span });
  assert(!sel.kept.some(i => i >= 30 && i < 34) && sel.status[31] === 'zoom', `kept ${sel.kept.filter(i => i >= 28 && i < 36)}`);
});

Deno.test('frame selection: one sharpest pick per spacing, blurry picks dropped (video2splat rules)', () => {
  const n = 100, r = makeRng(5);
  const blurred = new Set([49, 50, 51]);  // the whole central half of segment 10 (path 9.8-10.2): its pick is blurry
  const sharp = Array.from({ length: n }, (_, i) => blurred.has(i) ? 10 : 100 + 50 * r());
  const steps = Array.from({ length: n }, (_, i) => i === 0 ? 0 : i === 70 ? Infinity : 0.01);  // 5 frames per spacing; a jump at 70
  const sel = selectFrames({ sharpness: sharp, motion: steps }, { spacing: 0.05 });
  assertClose(sel.path[69], 69 / 5, 1e-9);
  assertClose(sel.path[70] - sel.path[69], 1, 1e-9, 'a lost track counts one spacing');
  for (const i of sel.kept) {
    const k = Math.floor(sel.path[i] + 0.5);
    const central = sel.path.some(t => Math.abs(t - k) <= 0.25);  // else the whole segment competes
    assert(!central || Math.abs(sel.path[i] - k) <= 0.25, `pick ${i} in the central half of segment ${k}`);
  }
  const gap = sel.kept.findIndex(i => i > 51);
  for (let k = 1; k < sel.kept.length; k++) {
    const d = sel.path[sel.kept[k]] - sel.path[sel.kept[k - 1]];
    if (k !== gap) assert(d >= 0.5 && d <= 1.5 + 1e-9, `picks ${sel.kept[k - 1]}, ${sel.kept[k]}: ${d} spacings apart`);
  }
  const dropped = [...blurred].filter(i => sel.status[i] === 'blurry');
  assert(dropped.length === 1 && [...blurred].every(i => sel.status[i] !== 'kept'), `segment 10: ${[...blurred].map(i => sel.status[i])}`);
  const byTime = selectFrames({ sharpness: sharp, motion: steps, times: sharp.map((_, i) => i / 10) }, { spacing: 0, keepFps: 2 });
  assert(byTime.kept.length === 20 && byTime.kept.every(i => byTime.kept.filter(j => Math.floor(j / 5) === Math.floor(i / 5)).length === 1), 'one per 0.5 s');
});

Deno.test('frame selection: a dull stretch is kept, a blur dip inside it is dropped (local threshold)', () => {
  // 40 segments of 5 frames; the first half 8x less textured (jardin7's start), segment 10 blurred
  const n = 200, r = makeRng(7);
  const sharp = Array.from({ length: n }, (_, i) => (i < 100 ? 100 : 800) * (0.9 + 0.2 * r()) * (i >= 48 && i <= 52 ? 0.1 : 1));
  const steps = Array.from({ length: n }, (_, i) => i === 0 ? 0 : 0.01);
  const sel = selectFrames({ sharpness: sharp, motion: steps }, { spacing: 0.05 });
  const dull = sel.kept.filter(i => i < 100).length, blurred = [48, 49, 50, 51, 52].filter(i => sel.status[i] === 'blurry');
  assert(dull >= 18, `dull half: ${dull} keyframes`);
  assert(blurred.length === 1 && [48, 49, 50, 51, 52].every(i => sel.status[i] !== 'kept'), `the blurred segment: ${[48, 49, 50, 51, 52].map(i => sel.status[i])}`);
});

Deno.test({ name: 'MP4 demuxer: jardin5 sample table (SFM_VIDEO)', ignore: !exists(VIDEO), fn: async () => {
  const reader = await denoFileReader(VIDEO);
  try {
    const d = await demux(reader);
    assert(d.codec.startsWith('avc1.') && d.description?.length > 8, `codec ${d.codec}`);
    assert(d.codedWidth === 1920 && d.codedHeight === 1080, `${d.codedWidth} x ${d.codedHeight}`);
    const pts = d.samples.map(s => s.pts).sort((a, b) => a - b);
    assert(pts[0] === 0 && pts.every((t, k) => k === 0 || t > pts[k - 1]), 'presentation times start at 0, unique');
    assert(d.samples[0].key && d.samples.some(s => !s.key), 'key frames from stss');
    // the first sample starts with an AVCC length prefix that fits in the sample
    const head = await reader.read(d.samples[0].offset, 4);
    assert(new DataView(head.buffer).getUint32(0) < d.samples[0].size, 'NAL length prefix');
    if (VIDEO.endsWith('jardin5.mp4')) assert(d.samples.length === 3723 && Math.abs(d.duration - 124.1) < 0.1, `${d.samples.length} samples, ${d.duration} s`);
  } finally {
    reader.close();
  }
} });

const VP9_VIDEO = env('SFM_VP9_VIDEO') ?? `${HOME}/jardin10.mp4`;

// jardin10.mp4: profile 0 in the frames, 1 in its vpcC (ffprobe agrees with the frames)
Deno.test({ name: 'MP4 demuxer: VP9 codec string, profile from the frame header (SFM_VP9_VIDEO)', ignore: !exists(VP9_VIDEO), fn: async () => {
  const reader = await denoFileReader(VP9_VIDEO);
  try {
    const d = await demux(reader);
    assert(/^vp09\.0[0-3]\.\d\d\.(08|10|12)$/.test(d.codec) && !d.description, `codec ${d.codec}`);
    assert(d.samples[0].key && d.samples.some(s => !s.key) && d.samples.every(s => s.size > 0), 'key frames, sizes');
    if (VP9_VIDEO.endsWith('jardin10.mp4')) assert(d.codec === 'vp09.00.40.08' && d.samples.length === 4852, `${d.codec}, ${d.samples.length} samples`);
  } finally {
    reader.close();
  }
} });

// Gray frames of a video at <= 1000 px through ffmpeg (area scaling, as cv2.INTER_AREA)
async function* ffmpegGray(path, w, h) {
  const p = new Deno.Command('ffmpeg', { args: ['-v', 'quiet', '-i', path, '-vf', `scale=${w}:${h}:flags=area,format=gray`, '-f', 'rawvideo', '-'], stdout: 'piped' }).spawn();
  const size = w * h;
  let buf = new Uint8Array(0);
  for await (const chunk of p.stdout) {
    const next = new Uint8Array(buf.length + chunk.length);
    next.set(buf); next.set(chunk, buf.length); buf = next;
    let o = 0;
    for (; o + size <= buf.length; o += size) yield buf.slice(o, o + size);
    buf = buf.slice(o);
  }
  await p.status;
}

// video2splat.py decoded with ffmpeg to JPEG (q 2) and measured with OpenCV; here the frames come
// raw from ffmpeg, so sharpness differs by a near-constant factor and picks can move by a frame.
Deno.test({ name: 'frame selection on jardin5 matches video2splat.py --spacing 0.05 (SFM_SHARPNESS)', ignore: !SLOW || !exists(VIDEO) || !exists(SHARPNESS_CSV), fn: async () => {
  const sharp = [], steps = [0];
  let prev = null;
  for await (const g of ffmpegGray(VIDEO, 1000, 562)) {
    const a = analyzeFrame(g, 1000, 562);
    sharp.push(a.sharpness);
    if (prev) steps.push(motion(prev, a.flow));
    prev = a.flow;
  }
  const rows = (await Deno.readTextFile(SHARPNESS_CSV)).trim().split('\n').slice(1).map(l => l.split(','));
  const n = Math.min(rows.length, sharp.length);
  const ratio = rows.slice(0, n).map((r, i) => sharp[i] / Number(r[1])).sort((a, b) => a - b);
  const sel = selectFrames({ sharpness: sharp, motion: steps }, { spacing: 0.05 });
  const theirs = rows.flatMap((r, i) => r[3] === 'kept' ? [i] : []);
  const rel = [];
  for (let k = 1; k < theirs.length; k++) {
    const [a, b] = [theirs[k - 1], theirs[k]];
    if (b < n && rows[b][2] !== '') rel.push(0.05 * (sel.path[b] - sel.path[a]) / Number(rows[b][2]));
  }
  rel.sort((a, b) => a - b);
  console.log(`  ${n} frames: sharpness ratio ${ratio[n >> 1].toFixed(3)} (p5 ${ratio[Math.floor(0.05 * n)].toFixed(3)}, p95 ${ratio[Math.floor(0.95 * n)].toFixed(3)}); ` +
              `kept ${sel.kept.length} vs ${theirs.length}; path between their picks ${rel[rel.length >> 1].toFixed(3)} x`);
  assert(ratio[Math.floor(0.95 * n)] / ratio[Math.floor(0.05 * n)] < 1.05, 'sharpness proportional');
  assert(Math.abs(sel.kept.length / theirs.length - 1) < 0.02, 'kept count');
  assert(Math.abs(rel[rel.length >> 1] - 1) < 0.01, 'path length');
} });
