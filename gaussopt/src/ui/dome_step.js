// dome_step.js — The Dome step (src/dome.js): a dome behind the splats from the training photos when the sky covers
// enough of them, given to the trainer (setDome: fixed while the splats train) and to the view (render.js env_render),
// with its preview in the stepper's Dome panel. Or a scene's dome (loadSceneGlb), or one made elsewhere (train.mjs DOME=).
"use strict";

import { buildDome, fillDome } from '../dome.js';
import { ENV_N } from '../env.js';

const VIEW_W = 256;  // the photos, at most this wide (the sky is smooth)
// the fills (params.js dome_fill, dome.js DOME_FILLS): label -> value; and the stepper's word for each
export const DOME_FILL_LABELS = { 'sky only': 'unseen', 'panorama': 'seen', 'sky everywhere': 'all' };
const FILL_NAME = Object.fromEntries(Object.entries(DOME_FILL_LABELS).map(([k, v]) => [v, k]));

export class DomeStep {
  constructor(app) {
    this.app = app;
    this.dome    = null;   // buildDome's result, or a scene's: { worth, farShare, n, cube, latlong?, ... }
    this.running = false;
    this.buf     = null;   // the dome for the view (render.js env_render: a vec4f a texel), when it is worth it
  }

  // Its state for the stepper (ui/phases.js)
  get state() {
    const d = this.dome;
    return { mode: this.app.params.dome, running: this.running,
             result: d ? { worth: d.worth, farShare: d.farShare, fill: d.fill ? FILL_NAME[d.fill] : null } : null };
  }

  // The training views' photos (the trainer's, at most VIEW_W wide), their cameras and the world's up into dome.js
  async build() {
    const app = this.app, tr = app.trainer;
    if (!tr || !app.gtAvailable || this.running) return;
    this.running = true;
    app.shell?.refresh();
    try {
      const t0 = performance.now(), ids = tr.trainViews.length ? tr.trainViews : tr.images.map((_, i) => i), views = [];
      for (const i of ids) {
        const v = await viewOf(tr, i);
        if (app.trainer !== tr) return;
        if (v) views.push(v);
      }
      const d = await buildDome(views, app.worldUp, { fill: app.params.dome_fill }, () => new Promise(r => setTimeout(r, 0)));
      if (app.trainer !== tr) return;
      this.set(d);
      const pct = (x) => `${(100 * x).toFixed(1)}%`;
      app.log(`Dome: the sky ${pct(d.farShare)} of ${views.length} photos' pixels (${pct(d.seen)} of the directions seen; exposure ` +
              `${d.gains[0].toFixed(2)}-${d.gains[1].toFixed(2)}): ${d.worth ? `a dome of ${d.n}² a face, ${FILL_NAME[d.fill]}` : 'no dome'}, ` +
              `${((performance.now() - t0) / 1000).toFixed(1)} s`, d.worth ? 'ok' : undefined);
    } catch (e) {
      app.showError(`Dome: ${e.message}`);
    } finally {
      this.running = false;
      app.shell?.refresh();
    }
  }

  // The dome on (a build() result worth it, or a scene's), off (null, or not worth it): the trainer's, the view's,
  // the preview
  set(d) {
    const app = this.app, on = !!d?.worth;
    this.dome = d;
    if (app.trainer && (on || app.trainer.domeN)) app.trainer.setDome(on ? d.cube : null, on ? d.n : 0);
    this.buf?.destroy();
    this.buf = null;
    if (on) {
      const f32 = new Float32Array(4 * 6 * d.n * d.n);
      for (let t = 0; t < 6 * d.n * d.n; t++) for (let c = 0; c < 3; c++) f32[4 * t + c] = d.cube[3 * t + c] / 255;
      this.buf = app.gpu.device.createBuffer({ size: f32.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: 'dome' });
      app.gpu.device.queue.writeBuffer(this.buf, 0, f32);
    }
    this._drawPreview();
    app.drawGen++;
    app.shell?.refresh();
  }

  // The dome made from the photos filled again with params.js dome_fill (dome.js fillDome: its maps, no photos): at
  // once, also while the splats train (they were trained with the other one until then)
  refill() {
    const d = this.dome, fill = this.app.params.dome_fill;
    if (!d?.maps || this.running || d.fill === fill) return;
    const { cube, filled } = fillDome(d.maps, fill, d.n);
    this.set({ ...d, fill, cube, latlong: { ...d.latlong, filled } });
    this.app.log(`Dome: filled again, ${FILL_NAME[fill]}`);
  }

  // The scene's background for a .glb (scene_glb.js env): the dome, else the learned one (bg_lr), else none
  async sceneEnv() {
    const d = this.dome, tr = this.app.trainer;
    if (d?.worth) return { n: d.n, rgb: d.cube, ...(d.farShare !== null && { sky: d.farShare }) };
    return this.app.params.bg_lr > 0 && tr ? { n: ENV_N, rgb: await tr.readEnv() } : null;
  }

  // The picture: the sky found (magenta: seen, not sky; black: unseen) over the dome made from it
  _drawPreview() {
    const canvas = document.getElementById('dome-preview'), stats = document.getElementById('dome-stats'), d = this.dome;
    if (!canvas || !stats) return;
    const ll = d?.latlong;
    canvas.hidden = !ll;
    stats.textContent = !d ? '' : d.farShare === null ? `A dome from the scene (${d.n}² texels a face).`
      : !ll ? `Sky: ${(100 * d.farShare).toFixed(1)}% of the photos' pixels; a dome made elsewhere (${d.n}² texels a face).`  // train.mjs DOME=
      : `Sky: ${(100 * d.farShare).toFixed(1)}% of the photos' pixels (a dome from 1%), ${(100 * d.seen).toFixed(0)}% of the directions seen, ` +
        `exposure ${d.gains[0].toFixed(2)}-${d.gains[1].toFixed(2)}. ${d.worth ? `The dome: ${d.n}² texels a face, fixed while the splats train.` : 'No dome.'} ` +
        `Above: the sky found (magenta: seen, not sky; black: never seen); below: the dome, ${FILL_NAME[d.fill] ?? d.fill}` +
        ({ unseen: ' (black: left to the splats).', seen: ' (what the cameras saw, the rest filled in).', all: ' (the sky spread everywhere).' }[d.fill] ?? '.');
    if (!ll) return;
    canvas.width = ll.W;
    canvas.height = 2 * ll.H;
    const img = new ImageData(ll.W, 2 * ll.H);
    [ll.observed, ll.filled].forEach((rgb, half) => {
      for (let i = 0; i < ll.W * ll.H; i++) {
        const o = 4 * (half * ll.W * ll.H + i);
        img.data[o] = rgb[3 * i]; img.data[o + 1] = rgb[3 * i + 1]; img.data[o + 2] = rgb[3 * i + 2]; img.data[o + 3] = 255;
      }
    });
    canvas.getContext('2d').putImageData(img, 0, 0);
  }
}

// Training view i as dome.js wants it: rgba (alpha 0: the camera's on-screen display) at most VIEW_W wide (a box
// filter), its intrinsics at that size, R (world to camera)
async function viewOf(tr, i) {
  const r = await tr.readView(i, ['gt']);
  if (!r) return null;
  const px = r.gt, W = tr.trainW, H = tr.trainH, s = Math.max(1, Math.ceil(W / VIEW_W)), w = Math.floor(W / s), h = Math.floor(H / s);
  const rgba = new Uint8Array(4 * w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const acc = [0, 0, 0];
    let alpha = 255;
    for (let dy = 0; dy < s; dy++) for (let dx = 0; dx < s; dx++) {
      const o = (y * s + dy) * tr.rowPitch + 4 * (x * s + dx);
      for (let c = 0; c < 3; c++) acc[c] += px[o + c];
      alpha = Math.min(alpha, px[o + 3]);
    }
    for (let c = 0; c < 3; c++) rgba[4 * (y * w + x) + c] = Math.round(acc[c] / (s * s));
    rgba[4 * (y * w + x) + 3] = alpha;
  }
  const k = r.k;
  return { R: tr.images[i].R, fx: k.fx / s, fy: k.fy / s, cx: k.cx / s, cy: k.cy / s, w, h, rgba };
}
