// telemetry.js — The side canvas: the training PSNR curve, the held-out PSNR and the octree / rebalance stats.
"use strict";

import { clamp } from '../math.js';

// Each sample's mean PSNR over the samples of the last `passSteps` steps (history: { step, psnr }, steps rising)
export function passMean(history, passSteps) {
  let lo = 0, sum = 0;
  return history.map((h, i) => {
    sum += h.psnr;
    while (history[lo].step <= h.step - passSteps) sum -= history[lo++].psnr;
    return sum / (i - lo + 1);
  });
}

// Octree / rebalance stats, `statsText`, and the train PSNR curve (by sample index) with held-out validation PSNR
// (orange points, by step). Each sample is one training view's PSNR, the views in a fixed round robin, so the raw
// curve (faint) repeats how hard each view is, once a pass; the bold line is the mean over the last `passSteps` steps
// (one pass over the training views; 0: the raw curve alone).
export function drawTelemetryPanel(canvas, history, octree, statsText = '', valHistory = [], passSteps = 0) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  // its size on screen (CSS pixels; the canvas's own when not laid out), its pixels at the device's density
  const W = canvas.clientWidth || canvas.width, H = canvas.clientHeight || canvas.height, dpr = canvas.clientWidth ? devicePixelRatio || 1 : 1;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.lineWidth = 1;
  ctx.fillStyle = '#121318';
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = '#2a2d3a';
  ctx.strokeRect(0.5, 0.5, W - 1, H - 1);

  ctx.font = W < 500 ? '9px monospace' : '12px monospace';  // a phone's pane: the header lines fit more
  ctx.fillStyle = '#e0e6f0';
  const s = octree ? octree.lastStats : { split: 0, cloned: 0, pruned: 0, floaters: 0, merged: 0, teleported: 0 };
  ctx.fillText(`Octree Leaves: ${octree ? octree.leaves.length : 0} | Rebalance: +${s.split} split, +${s.cloned} clone, ` +
               `-${s.pruned} prune, -${s.floaters} floaters, -${s.merged} merge${s.teleported ? `, ↻${s.teleported} mcmc` : ''}`, 12, 18);
  if (statsText) {
    ctx.fillStyle = '#8ef';
    ctx.fillText(statsText, 12, 36);
  }
  if (!history || history.length < 2) return;

  const padL = 44, padR = 16, padT = 46, padB = 20;
  const pw = W - padL - padR, ph = H - padT - padB;
  const s0 = history[0].step, s1 = history[history.length - 1].step;
  const vals = (valHistory || []).filter(v => v.step >= s0 && v.step <= s1);
  let minPsnr = 10, maxPsnr = 35;
  for (const h of [...history, ...vals]) {
    if (h.psnr < minPsnr) minPsnr = Math.floor(h.psnr - 1);
    if (h.psnr > maxPsnr) maxPsnr = Math.ceil(h.psnr + 1);
  }
  const yOf = (psnr) => padT + ph * (1 - clamp((psnr - minPsnr) / Math.max(1, maxPsnr - minPsnr), 0, 1));

  ctx.strokeStyle = '#252834';
  ctx.fillStyle = '#889';
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = padT + (ph * i) / 4;
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + pw, y);
    ctx.fillText(`${(maxPsnr - ((maxPsnr - minPsnr) * i) / 4).toFixed(1)}dB`, 4, y + 4);
  }
  ctx.stroke();

  const xOf = (i) => padL + (i / (history.length - 1)) * pw;
  ctx.strokeStyle = passSteps > 1 ? '#4af5' : '#4af';
  ctx.lineWidth = passSteps > 1 ? 1 : 1.8;
  ctx.beginPath();
  history.forEach((h, i) => ctx.lineTo(xOf(i), yOf(h.psnr)));
  ctx.stroke();
  if (passSteps > 1) {
    ctx.strokeStyle = '#4af';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    passMean(history, passSteps).forEach((m, i) => ctx.lineTo(xOf(i), yOf(m)));
    ctx.stroke();
  }

  if (vals.length > 0 && s1 > s0) {
    ctx.strokeStyle = ctx.fillStyle = '#fa4';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (const v of vals) {
      const x = padL + ((v.step - s0) / (s1 - s0)) * pw, y = yOf(v.psnr);
      ctx.lineTo(x, y);
      ctx.fillRect(x - 2, y - 2, 4, 4);
    }
    ctx.stroke();
    ctx.fillText(`val ${vals[vals.length - 1].psnr.toFixed(2)} dB`, W - padR - 90, padT - 4);
  }
}
