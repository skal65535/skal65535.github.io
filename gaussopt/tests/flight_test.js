// A drone video's flight record (src/sfm/flight.js): the djmd protobuf, the track in the cameras' world, and a
// real DJI file (SFM_DJI_VIDEO) against an independent decode (ffprobe's packets, a Python wire parser)
import { pbFields, djmdFix, readTelemetry, alignTrack, positionAt, keyframeTime } from '../src/sfm/flight.js';
import { setPose } from '../src/colmap.js';

const assert = (c, m) => { if (!c) throw new Error(m); };
const HOME = Deno.env.get('HOME');
const exists = (p) => { try { Deno.statSync(p); return true; } catch { return false; } };

// protobuf encoding, for the tests
const varint = (x) => { x = BigInt.asUintN(64, BigInt(x)); const o = []; do { let c = Number(x & 0x7fn); x >>= 7n; if (x) c |= 0x80; o.push(c); } while (x); return o; };
const msg = (...fields) => Uint8Array.from(fields.flat());
const vField = (fn, x) => [...varint(fn * 8), ...varint(x)];
const dField = (fn, x) => { const b = new Uint8Array(8); new DataView(b.buffer).setFloat64(0, x, true); return [...varint(fn * 8 + 1), ...b]; };
const sub = (fn, m) => [...varint(fn * 8 + 2), ...varint(m.length), ...m];

Deno.test('pbFields: varints (a negative int64 as 10 bytes), doubles, floats, nested messages', () => {
  const f32 = new Uint8Array(4); new DataView(f32.buffer).setFloat32(0, 59.94, true);
  const m = msg(vField(1, 1920), vField(2, -1329), dField(3, 0.7457), [...varint(4 * 8 + 5), ...f32], sub(5, msg(vField(1, 7))));
  const f = pbFields(m);
  assert(f.get(1) === 1920 && Math.abs(f.get(3) - 0.7457) < 1e-15 && Math.abs(f.get(4) - 59.94) < 1e-5, [...f].map(String).join(' '));
  assert(typeof f.get(2) === 'bigint' && BigInt.asIntN(64, f.get(2)) === -1329n, `${f.get(2)}`);
  assert(pbFields(f.get(5)).get(1) === 7, 'nested');
});

Deno.test('djmdFix: position, altitude, attitudes; a sample without a fix is none', () => {
  const lat = 42.72541686 * Math.PI / 180, lon = 0.54610699 * Math.PI / 180;
  const drone = msg(sub(3, msg(vField(1, -10), vField(2, 34), vField(3, -1329))), sub(4, msg(sub(1, msg(dField(2, lat), dField(3, lon))), vField(2, 1650890))));
  const frame = msg(sub(4, drone), sub(5, msg(sub(2, msg(vField(1, -180), vField(2, 25), vField(3, -1104))))));
  const f = djmdFix(msg(sub(3, frame)));
  assert(Math.abs(f.lat - 42.72541686) < 1e-9 && Math.abs(f.lon - 0.54610699) < 1e-9 && f.alt === 1650.89, JSON.stringify(f));
  assert(f.attitude.join() === '-1,3.4,-132.9' && f.gimbal.join() === '-18,2.5,-110.4', JSON.stringify(f));
  assert(djmdFix(msg(sub(1, msg(vField(1, 1))))) === null, 'a header-only sample');
});

Deno.test('alignTrack: cameras a similarity of the track at their times give the track back in their world', () => {
  const t = [], enu = [];
  for (let k = 0; k <= 600; k++) { const s = k / 10; t.push(s); enu.push([3 * s, 20 * Math.sin(s / 9), 0.2 * s]); }
  const tel = { t, enu };
  const s = 0.05, c = Math.cos(0.7), si = Math.sin(0.7), R = [c, -si, 0, si, c, 0, 0, 0, 1], T = [1, -2, 0.5];
  const world = (p) => [0, 1, 2].map(r => s * (R[3 * r] * p[0] + R[3 * r + 1] * p[1] + R[3 * r + 2] * p[2]) + T[r]);
  const images = [];
  for (let ms = 500; ms < 60000; ms += 750) {
    const im = setPose({ name: `t${String(ms).padStart(7, '0')}.jpg` }, [1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0]);
    im.center = world(positionAt(tel, ms / 1000));
    images.push(im);
  }
  images[10].center = images[10].center.map(x => x + 1);  // a camera 20 m off: left out by the refit
  const a = alignTrack(tel, images);
  assert(a.cameras === images.length && Math.abs(a.scale - s) < 1e-9 && a.residual < 1e-6, JSON.stringify({ ...a, points: a.points.length }));
  const p = a.points[300], q = world(enu[300]);
  assert(Math.hypot(...p.map((x, k) => x - q[k])) < 1e-9, `${p} vs ${q}`);
  assert(keyframeTime('gs/images/t0012345.jpg') === 12.345 && keyframeTime('p001_x.jpg') === null, 'keyframe times');
  // a hover sets no scale
  assert(alignTrack({ t, enu: enu.map(() => [0.1, 0, 0]) }, images) === null, 'a hover');
});

// canyon2.mp4 (a DJI NEO master, 128.6 s at 59.94 fps): reference values from ffprobe's djmd packets parsed in Python
// (photo_runs' djmd.py)
const DJI = Deno.env.get('SFM_DJI_VIDEO') ?? `${HOME}/gskal/brush/gaussopt/web/gaussopt/e2e/.cache/videos/canyon2.mp4`;
Deno.test({ name: 'readTelemetry: a DJI NEO video (SFM_DJI_VIDEO)', ignore: !exists(DJI), fn: async () => {
  const file = await Deno.open(DJI);
  let queue = Promise.resolve();  // one seek and read at a time (readTelemetry reads a batch at once)
  const readAt = async (o, n) => { const b = new Uint8Array(n); await file.seek(o, Deno.SeekMode.Start); let r = 0; while (r < n) r += await file.read(b.subarray(r)); return b; };
  const reader = { size: (await file.stat()).size, read: (o, n) => (queue = queue.then(() => readAt(o, n))) };
  try {
    const tel = await readTelemetry(reader);
    if (!DJI.endsWith('canyon2.mp4')) { assert(tel?.t.length > 0, 'no fixes'); return; }
    const near = (a, b, e) => Math.abs(a - b) <= e;
    assert(tel.t.length === 7709, `${tel.t.length} fixes`);
    const k = 3999, last = tel.t.length - 1;
    assert(near(tel.lat[0], 42.72580961, 1e-8) && near(tel.lon[0], 0.54436366, 1e-8) && near(tel.alt[0], 1711.39, 1e-3), `first ${tel.lat[0]} ${tel.lon[0]} ${tel.alt[0]}`);
    assert(near(tel.t[k], 66.7167, 1e-4) && near(tel.lat[k], 42.72374654, 1e-8) && tel.attitude[k].join() === '-15.8,1.2,-148.9' && tel.gimbal[k].join() === '9.8,0.8,-124.5',
           `${k}: ${tel.t[k]} ${tel.lat[k]} ${tel.attitude[k]} ${tel.gimbal[k]}`);
    assert(near(tel.t[last], 128.5951, 1e-4) && near(tel.alt[last], 1648.69, 1e-3), `last ${tel.t[last]} ${tel.alt[last]}`);
  } finally { file.close(); }
} });
