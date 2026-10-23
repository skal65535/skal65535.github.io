// video.js — S0 decoding in the browser (DESIGN-sfm §5.1, §5.7): WebCodecs VideoDecoder over the
// samples of mp4.js, frames in presentation order, drawn upright (track rotation applied) into an
// OffscreenCanvas at the requested size. Runs in a worker or on the page.
"use strict";

import { demux, blobReader } from './mp4.js';

// { ...demux(), file, width, height (displayed, after rotation), fps, config (VideoDecoderConfig) }
export async function openVideo(file) {
  const info = await demux(blobReader(file));
  const config = { codec: info.codec, codedWidth: info.codedWidth, codedHeight: info.codedHeight, hardwareAcceleration: 'prefer-hardware' };
  if (info.description) config.description = info.description;
  if (typeof VideoDecoder === 'undefined') throw new Error('this browser has no WebCodecs VideoDecoder');
  const supported = async () => (await VideoDecoder.isConfigSupported(config).catch(() => null))?.supported;
  if (!await supported()) delete config.hardwareAcceleration;  // e.g. VP9 without a hardware decoder: software
  if (!await supported()) {
    throw new Error(`cannot decode ${info.codec} here (HEVC needs a hardware decoder); transcode to H.264, ` +
                    `e.g. ffmpeg -i in.mov -c:v libx264 -crf 16 out.mp4, or use the COLMAP path`);
  }
  const swap = info.rotation === 90 || info.rotation === 270;
  const fps = info.samples.length / info.duration;
  return { ...info, file, config, fps,
           width: swap ? info.codedHeight : info.codedWidth, height: swap ? info.codedWidth : info.codedHeight };
}

// Presentation-order sample indices kept when sampling at `sampleFps` (every frame when
// sampleFps >= the video rate), as ffmpeg's fps filter: the first frame of each 1/sampleFps slot
export function sampleFrames(video, sampleFps) {
  const order = video.samples.map((s, k) => k).sort((a, b) => video.samples[a].pts - video.samples[b].pts);
  if (!(sampleFps > 0) || sampleFps >= video.fps - 1e-3) return order;
  const out = [], t0 = video.samples[order[0]].pts;
  let slot = -1;
  for (const k of order) {
    const s = Math.floor((video.samples[k].pts - t0) / video.timescale * sampleFps + 1e-6);
    if (s > slot) { out.push(k); slot = s; }
  }
  return out;
}

// Decode the track from the GOP before the first wanted sample (open GOPs reference it) to the
// last; for every sample index in `wanted` (a Set), in presentation order, awaits
// onFrame(canvas2d, sampleIndex, seconds) with the frame drawn at width x height.
export async function decodeFrames(video, wanted, width, height, onFrame) {
  // GPU-backed canvas: willReadFrequently would read every decoded frame back at full size and
  // scale it on the CPU (6x slower in Chrome); getImageData reads the scaled frame only
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  const { samples, timescale, rotation } = video;
  const byTime = new Map(samples.map((s, k) => [Math.round(s.pts * 1e6 / timescale), k]));
  const ready = [];
  let failure = null;
  const decoder = new VideoDecoder({ output: (frame) => ready.push(frame), error: (e) => { failure = e; } });
  decoder.configure(video.config);

  const drain = async () => {
    while (ready.length) {
      const frame = ready.shift();
      const k = byTime.get(frame.timestamp);
      try {
        if (k !== undefined && wanted.has(k)) {
          ctx.save();
          ctx.translate(width / 2, height / 2);
          ctx.rotate(rotation * Math.PI / 180);
          const [dw, dh] = rotation % 180 ? [height, width] : [width, height];
          ctx.drawImage(frame, -dw / 2, -dh / 2, dw, dh);
          ctx.restore();
          await onFrame(ctx, k, frame.timestamp / 1e6);
        }
      } finally {
        frame.close();
      }
    }
  };

  let first = samples.length - 1, last = -Infinity, syncs = 0;
  for (const k of wanted) { first = Math.min(first, k); last = Math.max(last, samples[k].dts); }
  while (first > 0 && !(samples[first].key && ++syncs === 2)) first--;  // the sync sample before first's
  // keep a few chunks in flight; read the file in ~8 MB spans of consecutive samples
  for (let k = first; k < samples.length;) {
    if (samples[k].dts > last && samples[k].key) break;  // past the last wanted frame (next GOP)
    let end = k + 1;
    while (end < samples.length && samples[end].offset === samples[end - 1].offset + samples[end - 1].size &&
           samples[end].offset + samples[end].size - samples[k].offset < 8 << 20) end++;
    const span = new Uint8Array(await video.file.slice(samples[k].offset, samples[end - 1].offset + samples[end - 1].size).arrayBuffer());
    for (let j = k; j < end; j++) {
      const s = samples[j];
      decoder.decode(new EncodedVideoChunk({ type: s.key ? 'key' : 'delta', timestamp: Math.round(s.pts * 1e6 / timescale),
                                             data: span.subarray(s.offset - samples[k].offset, s.offset - samples[k].offset + s.size) }));
      while (decoder.decodeQueueSize > 8 && !failure) await new Promise(r => setTimeout(r, 1));
      await drain();
      if (failure) throw failure;
    }
    k = end;
  }
  await decoder.flush().catch((e) => { failure ??= e; });
  await drain();
  decoder.close();
  if (failure) throw failure;
}

// Luma (BT.601) of RGBA pixels
export function grayOf(rgba, n) {
  const g = new Uint8Array(n);
  for (let k = 0, o = 0; k < n; k++, o += 4) g[k] = (77 * rgba[o] + 150 * rgba[o + 1] + 29 * rgba[o + 2] + 128) >> 8;
  return g;
}
