// edit.e2e.js — The editor's floater removal (ui/edit_panel.js, edit.js, trainer.js removeSplats / restoreSplats) in
// the app on the synthetic room (MESH_E2E_ROOM) served by serve.py: after 1030 steps, a blob of 27 opaque splats put in
// the air before a camera, where it touches nothing (restoreSplats). 'Find floater clusters' runs off the page's thread (its
// frames go on; closed meanwhile, its result dropped) and lists it, those 27 exactly; the view goes to it; Delete
// takes them out (the count 27 less, none left there, the others' splats, Adam state and moments kept, the page
// dirty though saved before); Undo puts them
// back, and the box closed, the training goes on (it was on when the list opened). A click on the view at the blob
// picks them (flags 1 on the GPU, magenta on screen), Cancel clears them and the training stays off (off when the box
// opened); 20 more steps train after the edits.
//   cd e2e && PORT=8795 node edit.e2e.js
import { fail, appUrl, run, room, appReady } from './harness.js';

const ROOM = room();

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=2'));
  await appReady(page);
  const shot = async () => {
    const png = (await page.locator('#main-canvas').screenshot()).toString('base64');
    return page.evaluate(async (png) => Array.from(await (await import('./e2e/page_util.js')).pngPixels(png)), png);
  };
  const frames = () => page.evaluate(() => new Promise(r => setTimeout(r, 800)));
  // the blob: 27 splats 0.015 x R apart, r 0.01 x R
  const setup = await page.evaluate(async () => {
    const { train } = await import('./e2e/page_util.js');
    const app = window.app, tr = app.trainer;
    // the room trained behind black: a random background leaves its air no splat-free spot for the blob (the
    // clearance below under 0.06 R as often as not)
    app.params.random_bg = false;
    await train(app, 1030);  // 30 past a rebalance: the moments not zeroed
    // in the air: of the points 0.1 to 0.6 x R before the first 8 cameras, the one furthest from the splats (each
    // splat's distance less 3 of its r)
    const R = app.octree.sceneRadius, sp = await tr.readSplats();
    let c = null, clear = -Infinity;
    for (const im of tr.images.slice(0, 8)) for (let t = 0.1; t <= 0.6; t += 0.05) {
      const q = [0, 1, 2].map(k => im.center[k] + t * R * im.view[2 + 4 * k]);
      let m = Infinity;
      for (let i = 0; i < tr.count && m > clear; i++) {
        const r = Math.exp(Math.max(sp[i * 16 + 4], sp[i * 16 + 5], sp[i * 16 + 6]));
        m = Math.min(m, Math.hypot(sp[i * 16] - q[0], sp[i * 16 + 1] - q[1], sp[i * 16 + 2] - q[2]) - 3 * r);
      }
      if (m > clear) { clear = m; c = q; }
    }
    const n = 27, params = new Float32Array(n * 16), pos = [];
    for (let i = 0; i < n; i++) {
      const p = [c[0] + ((i % 3) - 1) * 0.015 * R, c[1] + (((i / 3) | 0) % 3 - 1) * 0.015 * R, c[2] + (((i / 9) | 0) - 1) * 0.015 * R];
      params.set([...p, 2.2, ...Array(3).fill(Math.log(0.01 * R)), 0, 1, 0, 0, 0, -1, 1.5, -1, 0], i * 16);  // green
      pos.push(p);
    }
    const before = tr.count;
    await tr.restoreSplats({ count: n, params, moments: new Float32Array(n * 16), adamM: new Float32Array(n * 16), adamV: new Float32Array(n * 16), opGrad: null });
    window.__blob = pos;
    return { before, after: tr.count, R, clear: clear / R };
  });
  if (setup.after !== setup.before + 27) fail(`the blob put in: ${setup.before} -> ${setup.after} splats`);
  if (!(setup.clear > 0.06)) fail(`no room in the air for the blob: ${setup.clear.toFixed(3)} x R clear`);

  // the blob's splats now (their positions bit for bit)
  const blobIdx = (params) => page.evaluate(() => {
    const tr = window.app.trainer, p = tr.cpuParams, key = (x, y, z) => `${Math.fround(x)},${Math.fround(y)},${Math.fround(z)}`;
    const want = new Set(window.__blob.map(q => key(...q))), out = [];
    for (let i = 0; i < tr.count; i++) if (want.has(key(p[i * 16], p[i * 16 + 1], p[i * 16 + 2]))) out.push(i);
    return out;
  });

  // Find floater clusters closed (×) while it runs: its result dropped, the box stays closed
  const dropped = await page.evaluate(async () => {
    const ed = window.app.edit, job = ed.findFloaters();
    await new Promise(r => setTimeout(r, 50));
    document.getElementById('edit-close').click();
    await job;
    return { list: !!ed.list, box: getComputedStyle(document.getElementById('edit-box')).display };
  });
  if (dropped.list || dropped.box !== 'none') fail(`closed while finding: a list ${dropped.list}, the box ${dropped.box}`);

  // Find floater clusters: the blob listed, those 27; Delete
  const found = await page.evaluate(async () => {
    const app = window.app, ed = app.edit;
    app.setTraining(true);
    // off the page's thread: the page's frames go on meanwhile (the longest gap between two), 'Finding' shown
    const title = () => document.getElementById('edit-title').textContent;
    let frames = 0, gap = 0, last = performance.now(), running = true, finding = false;
    const tick = () => { const t = performance.now(); gap = Math.max(gap, t - last); last = t; frames++; if (running) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
    const t0 = performance.now(), job = ed.findFloaters();
    await new Promise(r => setTimeout(r, 50));
    finding = title().startsWith('Finding');
    await job;
    running = false;
    const took = performance.now() - t0;
    const L = ed.list;
    const p = app.trainer.cpuParams, key = (i) => `${p[i * 16]},${p[i * 16 + 1]},${p[i * 16 + 2]}`;
    const blob = new Set(window.__blob.map(q => q.map(Math.fround).join()));
    const at = L.clusters.findIndex(c => c.indices.length === 27 && [...c.indices].every(i => blob.has(key(i))));
    // when not listed: the cluster grown from one of the blob's splats
    let grown = null;
    if (at < 0) {
      const E = await import('./src/edit.js'), b = [...Array(app.trainer.count).keys()].filter(i => blob.has(key(i)));
      const g = E.growCluster(new E.SplatGrid(p, app.trainer.count, { reach: ed.opts.edit_reach }), b[0], 5000);
      grown = { blob: b.length, size: g.indices.length, attached: g.attached, total: L.total, masses: L.clusters.slice(0, 3).map(c => c.mass) };
    }
    return { listed: L.clusters.length, at, grown, took, frames, gap, finding, training: app.params.training, box: getComputedStyle(document.getElementById('edit-box')).display };
  });
  if (found.at < 0) fail(`the blob not among the ${found.listed} floaters listed: ${JSON.stringify(found.grown)}`);
  if (found.training || found.box !== 'block') fail(`while listed: training ${found.training}, the box ${found.box}`);
  // a search long enough to tell (the page's thread would be held for all of it), the frames on meanwhile
  if (!found.finding || found.took < 400 || found.gap > Math.max(250, 0.4 * found.took)) {
    fail(`Find floater clusters: 'Finding' shown ${found.finding}, ${found.took.toFixed(0)} ms, ${found.frames} frames meanwhile, the longest gap ${found.gap.toFixed(0)} ms`);
  }
  const del = await page.evaluate(async (at) => {
    const app = window.app, ed = app.edit, tr = app.trainer, n0 = tr.count;
    ed.list.at = at; ed._showListed();
    const sel = ed.sel.indices.length;
    // each kept splat's row (but its leaf id, slot 7, and its flags, 15: the next floater shown), Adam m and v and
    // moments, before and after, through newIndex
    const { splatState } = await import('./e2e/page_util.js'), s0 = await splatState(tr, true);
    const nonzero = [1, 2, 3].map(a => s0[a].reduce((c, x) => c + (x !== 0), 0));  // Adam m, v, the moments
    app._saved(tr.stepCount);  // as saved: the deletion alone makes the page dirty
    const clean = !app.dirty;
    const remove = tr.removeSplats.bind(tr);
    let newIndex = null;
    tr.removeSplats = async (x) => { const r = await remove(x); newIndex = r.newIndex; return r; };
    await ed.deleteSelection();
    tr.removeSplats = remove;
    const s1 = await splatState(tr, true), F = [16, 16, 16, 16];
    let kept = 0, moved = 0, first = null;
    for (let i = 0; i < n0; i++) {
      const j = newIndex[i];
      if (j < 0) continue;
      kept++;
      for (let a = 0; a < 4; a++) for (let k = 0; k < F[a]; k++) {
        if (a === 0 && (k === 7 || k === 15)) continue;
        if (!Object.is(s0[a][i * F[a] + k], s1[a][j * F[a] + k])) { moved++; first ??= { i, j, a, k, was: s0[a][i * F[a] + k], now: s1[a][j * F[a] + k] }; break; }
      }
    }
    return { n0, n1: tr.count, sel, kept, moved, first, nonzero, clean, dirty: app.dirty, undo: !!ed.undo };
  }, found.at);
  if (del.sel !== 27 || del.n1 !== del.n0 - 27) fail(`Delete: ${del.sel} selected, ${del.n0} -> ${del.n1} splats`);
  if (del.kept !== del.n1 || del.moved) fail(`Delete: ${del.kept} kept of ${del.n1}, ${del.moved} with their splat, Adam state or moments changed: ${JSON.stringify(del.first)}`);
  if ((await blobIdx()).length) fail('Delete: some of the blob left');
  if (!del.clean || !del.dirty || !del.undo) fail(`Delete: clean before ${del.clean}, dirty after ${del.dirty}, undo ${del.undo}`);
  if (del.nonzero.some(c => c < del.n0)) fail(`before Delete, non-zero values of Adam m, v and the moments: ${del.nonzero} (a test of their carry needs them)`);

  // Undo: the blob back
  // (the box closed: the training on again, as when the list opened; stopped here, before a step moves the blob)
  const undo = await page.evaluate(async () => {
    const app = window.app, ed = app.edit;
    await ed.undoDelete();
    const resumed = app.params.training;
    app.setTraining(false);
    return { n: app.trainer.count, undo: !!ed.undo, resumed };
  });
  if (undo.n !== del.n0 || undo.undo) fail(`Undo: ${undo.n} splats (want ${del.n0}), undo still ${undo.undo}`);
  if (!undo.resumed) fail('Undo closed the box: the training not on again');
  if ((await blobIdx()).length !== 27) fail(`Undo: ${(await blobIdx()).length} of the blob back`);

  // Find isolated floaters with the smallest cluster at 30: the blob's 27 among the splats selected at once, all of
  // them deleted in one Delete, and put back
  const isoBefore = await blobIdx();
  const iso = await page.evaluate(async () => {
    const app = window.app, ed = app.edit, tr = app.trainer, n0 = tr.count;
    ed.opts.edit_min = 30;
    await ed.findFloaters('isolated');
    const title = document.getElementById('edit-title').textContent, sel = ed.sel ? [...ed.sel.indices] : [];
    await ed.deleteSelection();
    const n1 = tr.count;
    await ed.undoDelete();
    ed.opts.edit_min = 3;
    return { n0, n1, n2: tr.count, title, sel, box: getComputedStyle(document.getElementById('edit-box')).display, training: app.params.training };
  });
  const isoBlob = await blobIdx();
  if (iso.title !== 'Isolated floaters' || iso.n1 !== iso.n0 - iso.sel.length || iso.n2 !== iso.n0 || iso.box !== 'none' || iso.training) {
    fail(`isolated floaters: '${iso.title}', ${iso.sel.length} selected, ${iso.n0} -> ${iso.n1} -> ${iso.n2} splats, the box ${iso.box}, training ${iso.training}`);
  }
  if (isoBlob.length !== 27) fail(`isolated floaters: ${isoBlob.length} of the blob back`);
  if (!isoBefore.every(i => iso.sel.includes(i))) fail(`isolated floaters: the blob's 27 not all selected (${isoBefore.filter(i => iso.sel.includes(i)).length})`);

  // a click on the blob: the view on it (flyTo), the canvas centre clicked
  await page.evaluate(() => {
    const app = window.app, ed = app.edit, p = window.__blob, c = [0, 1, 2].map(k => p.reduce((a, q) => a + q[k], 0) / p.length);
    ed.flyTo({ center: c, extent: 0.05 * app.octree.sceneRadius, indices: [] });
  });
  await frames();
  const plain = await shot();
  const box = await page.locator('#main-canvas').boundingBox();
  await page.evaluate(() => { window.app.edit.opts.edit_pick = true; });
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  await page.waitForFunction(() => window.app.edit.sel && !window.app.edit.busy, null, { timeout: 30000 });
  await frames();
  const picked = await page.evaluate(async () => {
    const ed = window.app.edit, tr = window.app.trainer, p = await tr.readSplats();
    let flagged = 0;
    for (let i = 0; i < tr.count; i++) if (p[i * 16 + 15] === 1) flagged++;
    return { sel: [...ed.sel.indices], attached: ed.sel.attached, flagged, training: window.app.params.training };
  });
  const blob = await blobIdx();
  if (picked.attached || picked.sel.join() !== blob.join() || picked.flagged !== 27) {
    fail(`click: ${picked.sel.length} selected (the blob's ${blob.length}: ${picked.sel.join() === blob.join()}), attached ${picked.attached}, ${picked.flagged} flagged on the GPU`);
  }
  // magenta where the blob is drawn: in the middle tenth of the view, the share of the pixels more red and blue
  // than green
  const tinted = await shot();
  const magenta = (px) => {
    const w = Math.round(box.width), h = Math.round(px.length / 4 / w);
    let n = 0, all = 0;
    for (let y = Math.round(0.45 * h); y < 0.55 * h; y++) for (let x = Math.round(0.45 * w); x < 0.55 * w; x++, all++) {
      const i = 4 * (y * w + x);
      if (px[i] > px[i + 1] + 80 && px[i + 2] > px[i + 1] + 60) n++;
    }
    return n / all;
  };
  const m0 = magenta(plain), m1 = magenta(tinted);
  if (!(m1 > m0 + 0.2)) fail(`selection not shown: magenta pixels ${(100 * m0).toFixed(2)}% -> ${(100 * m1).toFixed(2)}%`);
  // Cancel (×): the flags 0, the training still off (off when the box opened); 20 steps more
  const after = await page.evaluate(async () => {
    const app = window.app, tr = app.trainer;
    document.getElementById('edit-close').click();
    const p = await tr.readSplats();
    let flagged = 0;
    for (let i = 0; i < tr.count; i++) if (p[i * 16 + 15] !== 0) flagged++;
    const training = app.params.training;
    app.edit.opts.edit_pick = false;
    const { train } = await import('./e2e/page_util.js');
    const s0 = tr.stepCount;
    await train(app, 20);
    return { flagged, training, steps: tr.stepCount - s0, finite: Number.isFinite(tr.lastPsnr) };
  });
  if (after.flagged || after.training) fail(`Cancel: ${after.flagged} flagged, training ${after.training} (off when the box opened)`);
  if (after.steps < 20 || !after.finite) fail(`after the edits: ${after.steps} steps, PSNR finite ${after.finite}`);
  return `isolated (smallest 30): ${iso.sel.length} splats deleted at once and put back; found off the page's thread in ${(found.took / 1000).toFixed(1)} s (${found.frames} frames meanwhile, the longest gap ` +
         `${found.gap.toFixed(0)} ms); the blob 1 of ${found.listed} floaters, deleted (${del.n0} -> ${del.n1}, the others' splats, Adam state and moments kept) and put back; clicked: its 27 splats, ` +
         `magenta ${(100 * m0).toFixed(0)}% -> ${(100 * m1).toFixed(0)}% of the view's middle; cancelled, 20 more steps`;
}, { serverArgs: [ROOM] });
