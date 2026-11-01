// page_util.js — What the e2e tests run inside the app's page, imported in page.evaluate:
//   const { train, photoIsRender } = await import('./e2e/page_util.js');
// (from the gallery's page: '/gaussopt/e2e/page_util.js')

// `n` more training steps of the app (stop_after), within `timeout` ms
export async function train(app, n, timeout = 300000) {
  const tr = app.trainer, P = app.params;
  P.stop_after = n;
  app.setTraining(true);
  const end = tr.stepCount + n;
  for (const t0 = Date.now(); tr.stepCount < end || P.training; await new Promise(res => setTimeout(res, 100)))
    if (Date.now() - t0 > timeout) throw new Error(`stuck at step ${tr.stepCount} of ${end}, training ${P.training}`);
  await tr._drainLossRead();
}

// Each of `views`' photos becomes its render at its pose (map(hat, i): the photo made of the render instead)
export async function photoIsRender(tr, views, map = (hat) => hat) {
  for (const [i, v] of views.entries()) {
    const { hat } = await tr.readView(v);
    tr.device.queue.writeTexture({ texture: await tr.ensureGtTexture(v) }, map(hat, i), { bytesPerRow: tr.rowPitch }, [tr.trainW, tr.trainH]);
  }
}

// Copies of the splats, Adam's m and v (and the moments with `moments`), to compare with changedCount
export async function splatState(tr, moments = false) {
  const srcs = [{ buf: tr.splatsBuf, bytes: tr.count * 64 }, { buf: tr.adamMBuf, bytes: tr.count * 64 }, { buf: tr.adamVBuf, bytes: tr.count * 64 }];
  if (moments) srcs.push({ buf: tr.momentsBuf, bytes: tr.count * 48 });
  return (await tr._readback('state', srcs)).map(a => a.slice());
}

// The values of each array of s1 not bit for bit those of s0
export const changedCount = (s0, s1) => s0.map((a, i) => a.reduce((c, x, j) => c + (Object.is(x, s1[i][j]) ? 0 : 1), 0));

// A PNG's RGBA bytes (a Uint8ClampedArray), from its base64
export async function pngPixels(b64) {
  const im = await createImageBitmap(await (await fetch(`data:image/png;base64,${b64}`)).blob());
  const c = new OffscreenCanvas(im.width, im.height), g = c.getContext('2d');
  g.drawImage(im, 0, 0);
  return g.getImageData(0, 0, im.width, im.height).data;
}
