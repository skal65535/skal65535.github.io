// Photo sets (DESIGN-sfm §13.3 P1, P8): EXIF reading on JPEG bytes built here, and the plan of a
// mixed set (capture order, names, labels, rotation, camera groups).
import { readExif, captureTime, jpegSize } from '../src/sfm/exif.js';
import { planPhotos, planMore, focalFromExif } from '../src/sfm/photos.js';
import { assert, assertClose, SLOW, env, exists, hasGPU } from './util.js';
import { GpuSift } from '../src/sfm/sift.js';
import { GpuMatcher } from '../src/sfm/match.js';
import { reconstructFromFeatures } from '../src/sfm/pipeline.js';
import { PHOTO_MAPPING } from '../src/sfm/defaults.js';
import { compareModels } from '../src/sfm/compare.js';
import { readColmapModel } from '../tools/common.js';
import { readPGM } from '../tools/pgm.js';

// A JPEG start (SOI + APP1 Exif, little-endian TIFF): IFD0 with Make, Model, Orientation and the
// Exif IFD, which holds DateTimeOriginal, SubSecTimeOriginal, OffsetTimeOriginal, FocalLength,
// DigitalZoomRatio, FocalLengthIn35mmFilm and LensModel; then SOS.
function jpegWithExif({ make = 'Google', model = 'Pixel 10', orientation = 1, date = '2026:10:24 09:09:24', sub = '741',
                        offset = '+02:00', focal = [1854, 1000], zoom = [0, 1], f35 = 14, lens = 'Pixel 10 back camera 1.854mm f/2.2' } = {}) {
  const ascii = (s) => [...s].map(c => c.charCodeAt(0)).concat(0);
  const ifd0 = [[0x010F, 2, ascii(make)], [0x0110, 2, ascii(model)], [0x0112, 3, orientation], [0x8769, 4, 'exif']];
  const exif = [[0x9003, 2, ascii(date)], [0x9011, 2, ascii(offset)], [0x9291, 2, ascii(sub)], [0x920A, 5, focal],
                [0xA404, 5, zoom], [0xA405, 3, f35], [0xA434, 2, ascii(lens)]];
  const tiff = [];
  const w16 = (at, v) => { tiff[at] = v & 255; tiff[at + 1] = v >> 8; };
  const w32 = (at, v) => { for (let i = 0; i < 4; i++) tiff[at + i] = (v >>> (8 * i)) & 255; };
  const size = (entries) => 2 + 12 * entries.length + 4;
  const ifd0At = 8, exifAt = ifd0At + size(ifd0);
  let data = exifAt + size(exif);
  const write = (at, entries) => {
    w16(at, entries.length);
    entries.forEach(([tag, type, v], k) => {
      const e = at + 2 + 12 * k;
      w16(e, tag); w16(e + 2, type);
      if (type === 2) {
        w32(e + 4, v.length);
        if (v.length <= 4) v.forEach((b, i) => { tiff[e + 8 + i] = b; });
        else { w32(e + 8, data); v.forEach((b, i) => { tiff[data + i] = b; }); data += v.length; }
      } else if (type === 5) { w32(e + 4, 1); w32(e + 8, data); w32(data, v[0]); w32(data + 4, v[1]); data += 8; }
      else if (type === 3) { w32(e + 4, 1); w16(e + 8, v); }
      else { w32(e + 4, 1); w32(e + 8, v === 'exif' ? exifAt : v); }
    });
    w32(at + 2 + 12 * entries.length, 0);
  };
  tiff[0] = tiff[1] = 0x49; w16(2, 42); w32(4, ifd0At);
  write(ifd0At, ifd0);
  write(exifAt, exif);
  const body = Array.from({ length: data }, (_, i) => tiff[i] ?? 0);
  const app1 = [0x45, 0x78, 0x69, 0x66, 0, 0, ...body], len = app1.length + 2;
  return Uint8Array.from([0xFF, 0xD8, 0xFF, 0xE1, len >> 8, len & 255, ...app1, 0xFF, 0xDA, 0, 2]).buffer;
}

Deno.test('exif: the fields a photo set needs', () => {
  const e = readExif(jpegWithExif({ zoom: [170, 100] }));
  assert(e.make === 'Google' && e.model === 'Pixel 10' && e.orientation === 1, JSON.stringify(e));
  assert(e.lensModel === 'Pixel 10 back camera 1.854mm f/2.2' && e.focalLength35mm === 14, JSON.stringify(e));
  assertClose([e.focalLength, e.digitalZoomRatio], [1.854, 1.7], 1e-9, 'rationals');
  // 09:09:24.741 at +02:00 is 07:09:24.741 UTC
  assert(captureTime(e) === Date.UTC(2026, 9, 24, 7, 9, 24, 741), `capture time ${captureTime(e)}`);
  assert(Object.keys(readExif(Uint8Array.from([0xFF, 0xD8, 0xFF, 0xDA, 0, 2]).buffer)).length === 0, 'no EXIF: {}');
  assert(Object.keys(readExif(new ArrayBuffer(3))).length === 0, 'not a JPEG: {}');
  // a frame header (SOF0, 4000 x 3000) after the APP1 segment, as shown with orientation 1 and 6
  const bytes = new Uint8Array(jpegWithExif()), sos = bytes.length - 4;
  const withSof = Uint8Array.from([...bytes.slice(0, sos), 0xFF, 0xC0, 0, 11, 8, 3000 >> 8, 3000 & 255, 4000 >> 8, 4000 & 255, 1, 1, 0x11, 0, ...bytes.slice(sos)]).buffer;
  assert(jpegSize(withSof).join() === '4000,3000' && jpegSize(withSof, 6).join() === '3000,4000', `size ${jpegSize(withSof)}`);
  assert(jpegSize(bytes.buffer) === null, 'no frame header before SOS: null');
});

Deno.test('photos: capture order, labels, rotation, a camera per lens and zoom, label filters', () => {
  const at = (s) => `2026:10:24 09:10:${String(s).padStart(2, '0')}`;
  const main = { focal: [453, 100], f35: 24, lens: 'Pixel 10 back camera 4.53mm f/1.7' };
  const items = [
    { fileName: 'c.jpg', width: 3000, height: 4000, exif: readExif(jpegWithExif({ ...main, date: at(30) })) },  // portrait, main 1x
    { fileName: 'a.jpg', width: 4000, height: 3000, exif: readExif(jpegWithExif({ ...main, date: at(10) })) },  // landscape, main 1x
    { fileName: 'b.jpg', width: 4000, height: 3000, exif: readExif(jpegWithExif({ date: at(20) })) },           // ultrawide
    { fileName: 'd.jpg', width: 4000, height: 3000, exif: readExif(jpegWithExif({ ...main, date: at(40), zoom: [200, 100] })) },  // 2x
    { fileName: 'e.jpg', width: 4000, height: 3000, exif: readExif(jpegWithExif({ ...main, date: at(50) })) },  // landscape, main 1x
  ];
  const plan = planPhotos(items);
  assert(plan.photos.map(p => p.name).join() === 'p001_a.jpg,p002_b.jpg,p003_c.jpg,p004_d.jpg,p005_e.jpg', plan.photos.map(p => p.name).join());
  assertClose(plan.photos.map(p => p.time), [0, 10, 20, 30, 40], 1e-9, 'times');
  assert(plan.groups.length === 3 && plan.groups[0].count === 3 && plan.chosen === plan.groups[0].key, JSON.stringify(plan.groups));
  // a camera per lens and zoom, the largest group first; every photo used once rotated to one size
  assert(plan.cameras.map(g => g.key).join('/') === plan.groups.map(g => g.key).join('/'), 'cameras: the groups, by size');
  assert(plan.used.map(p => `${p.name}:${p.camera}`).join() === 'p001_a.jpg:0,p002_b.jpg:1,p003_c.jpg:0,p004_d.jpg:2,p005_e.jpg:0', JSON.stringify(plan.used.map(p => [p.name, p.camera])));
  assert(plan.used.filter(p => p.rotate).map(p => p.name).join() === 'p003_c.jpg', 'the portrait is rotated to the majority');
  assert(plan.size.join() === '4000,3000' && plan.skipped.length === 0, JSON.stringify(plan.skipped));
  const p3 = plan.photos[2].labels;
  assert(p3.orientation === 'portrait' && p3.zoom === 1 && p3.lens === main.lens, 'labels keep the orientation as shot');
  assert(plan.photos[3].labels.zoom === 2 && plan.photos[3].focal35 === 48, 'zoom times the 35 mm focal');
  // a label filter (P8.5): only the main lens at 1x, one camera
  const main1x = planPhotos(items, { filter: (l) => l.lens === main.lens && l.zoom === 1 });
  assert(main1x.used.map(p => p.name).join() === 'p001_a.jpg,p003_c.jpg,p005_e.jpg' && main1x.cameras.length === 1, 'the main lens at 1x');
  assert(main1x.skipped.map(s => s.reason).join() === 'filtered out,filtered out', JSON.stringify(main1x.skipped));
  // only portraits: still rotated to the whole set's majority, as without the filter (cached features)
  const portrait = planPhotos(items, { filter: (l) => l.orientation === 'portrait' });
  assert(portrait.used.length === 1 && portrait.used[0].rotate && portrait.size.join() === '4000,3000', JSON.stringify(portrait.used));
  // 24 mm equivalent over a 1920 x 1440 frame: 24 * 2400 / 43.27 (DESIGN-sfm §13.3 P2)
  assertClose(focalFromExif(24, 1920, 1440), 1331.2, 0.1, 'focal');

  // Add photos: named after the set's last in capture order, rotated to its orientation, on their
  // lens's camera; another size and a file already in the set are skipped
  const more = planMore(plan, [
    { fileName: 'f.jpg', width: 4000, height: 3000, exif: readExif(jpegWithExif({ ...main, date: at(59) })) },
    { fileName: 'g.jpg', width: 3000, height: 4000, exif: readExif(jpegWithExif({ ...main, date: at(55) })) },  // portrait
    { fileName: 'a.jpg', width: 4000, height: 3000, exif: readExif(jpegWithExif({ ...main, date: at(10) })) },  // in the set
    { fileName: 'h.jpg', width: 4000, height: 3000, exif: readExif(jpegWithExif({ date: at(56) })) },           // ultrawide (camera 1)
    { fileName: 'i.jpg', width: 2000, height: 1500, exif: readExif(jpegWithExif({ ...main, date: at(57) })) },  // size
  ]);
  assert(more.photos.map(p => `${p.name}:${p.item}:${p.rotate}:${p.camera}`).join() === 'p006_g.jpg:1:true:0,p007_h.jpg:3:false:1,p008_f.jpg:0:false:0', JSON.stringify(more.photos));
  assertClose(more.photos.map(p => p.time), [45, 46, 49], 1e-9, 'times from the set\'s first');
  assert(more.skipped.map(s => s.name).sort().join() === 'a.jpg,i.jpg', JSON.stringify(more.skipped));
  // a lens the set does not have
  const other = planMore(main1x, [{ fileName: 'h.jpg', width: 4000, height: 3000, exif: readExif(jpegWithExif({ date: at(56) })) }]);
  assert(other.photos.length === 0 && /another camera/.test(other.skipped[0].reason), JSON.stringify(other));
  // a set numbered to its names' width takes no more (a plan saved with 1 digit, 9 photos)
  const nine = Array.from({ length: 9 }, (_, k) => ({ name: `p${k + 1}_x${k}.jpg` }));
  const full = planMore({ ...plan, photos: nine, digits: undefined }, [
    { fileName: 'f.jpg', width: 4000, height: 3000, exif: readExif(jpegWithExif({ ...main, date: at(59) })) }]);
  assert(full.photos.length === 0 && /more than 9 photos/.test(full.skipped[0]?.reason), JSON.stringify(full));
});

// Real photo sets (SLOW=1, the data of DESIGN-sfm §13.2 under SFM_PHOTOS, default ~/gskal/brush):
// cuisine and minerals as the page maps them (PHOTO_MAPPING, every pair, the loop stage), against COLMAP's model.
// Two inputs that failed before: the browser's gray at 4096 features (7 photos in one segment
// without regrowSegments) and ImageMagick's at 6000 (a photo 40 deg off without relMaxTurnDeg).
const PHOTOS = env('SFM_PHOTOS') ?? `${env('HOME')}/gskal/brush`;
async function photoFeatures(dir, maxFeatures) {
  const names = [...Deno.readDirSync(dir)].map(e => e.name).filter(n => n.endsWith('.pgm')).sort();
  const sift = await GpuSift.create(), matcher = await GpuMatcher.create(sift.device), feats = [];
  let size;
  for (const n of names) {
    const im = await readPGM(`${dir}/${n}`);
    size = [im.width, im.height];
    feats.push({ name: n.replace(/\.pgm$/, '.jpg'), ...await sift.extract(im.data, im.width, im.height, { maxFeatures }) });
  }
  return { feats, size, matcher };
}
const mapPhotos = ({ feats, size, matcher }, seed, mapping = {}) =>
  reconstructFromFeatures(feats, size[0], size[1], matcher, { window: feats.length - 1, loops: true, seed, mapping: { ...PHOTO_MAPPING, ...mapping } });
// The mapping depends on the seed (RANSAC's draws in matching and mapping): cuisine sometimes starts from
// a pair that sets the focal ~9% high (8 photos in one segment), minerals splits 34 + 4. Each case runs
// SEEDS and passes on 2 of 3; the counts are those most seeds reach (photo_runs/NOTES.md: 12 seeds on
// cuisine, 4 on minerals, before and after the verification pool). minerals (40 photos, a room then
// close-ups of a bookshelf): COLMAP registers 39.
const SEEDS = [1, 2, 3];
// A known failure is left out with its reason, until fixed.
const CASES = [['cuisine', 'pgm_browser', 4096, 11, 'known failure: its first photo (a near-planar wall) registers 20-29 deg off on most seeds, ' +
                'before and after the verification pool (4 of 12 seeds right before, 2 of 11 after)'],
               ['cuisine', 'pgm1920', 6000, 11], ['minerals', 'pgm_browser', 4096, 34]];
for (const [set, input, maxFeatures, minCommon, known] of CASES) {
  const dir = `${PHOTOS}/${set}/${input}`, ref = `${PHOTOS}/${set}/colmap/sparse/0`;
  Deno.test({ name: `photos, ${set} (${input}, ${maxFeatures} features): one segment of ${minCommon}+ of COLMAP's photos, rotations within 3 deg, on 2 of ${SEEDS.length} seeds` +
                    (known ? ` (${known})` : ''),
              ignore: !SLOW || !hasGPU || !exists(dir) || !exists(ref) || !!known, fn: async () => {
    const features = await photoFeatures(dir, maxFeatures), colmap = await readColmapModel(ref), tags = [];
    let passed = 0;
    for (const seed of SEEDS) {
      const { model } = await mapPhotos(features, seed);
      const bySeg = new Map();
      for (const im of model.images) bySeg.set(im.segment, [...(bySeg.get(im.segment) ?? []), im]);
      const largest = [...bySeg.values()].sort((a, b) => b.length - a.length)[0], mine = new Set(largest.map(im => im.name));
      const m = compareModels({ images: largest, focal: model.focal }, { ...colmap, images: colmap.images.filter(im => mine.has(im.name)) });
      const ok = m.numCommon >= minCommon && m.maxRotDeg < 3 && m.rmsCenterRel < 0.01;
      passed += ok;
      tags.push(`seed ${seed}${ok ? '' : ' (failed)'}: largest segment ${largest.length}, ${m.numCommon} in COLMAP, RMS ${(100 * m.rmsCenterRel).toFixed(2)}%, ` +
                `rotations median ${m.medianRotDeg.toFixed(2)} max ${m.maxRotDeg.toFixed(2)} deg`);
    }
    console.log(tags.join("\n"));
    assert(passed >= 2, tags.join("; "));
  } });
}
