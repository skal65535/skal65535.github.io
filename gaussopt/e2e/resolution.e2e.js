// resolution.e2e.js — A second pass on larger images (trainer.js setDownsample, app.js: params.downsample changed):
// the synthetic room (MESH_E2E_ROOM, as mesh.e2e.js) trained 300 steps at ds 4, then ds set to 2: the training
// resolution doubles, the splats and the step count stay, training goes on at the new size (the loss's PSNR is
// measured over its pixels) and the held-out views are scored at it.
//   cd e2e && node resolution.e2e.js
import { fail, appUrl, run, room, appReady } from './harness.js';

const ROOM = room();

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=4'));
  await appReady(page);
  const r = await page.evaluate(async () => {
    const app = window.app, tr = app.trainer, P = app.params;
    const { train: trainOf } = await import('./e2e/page_util.js'), train = (n) => trainOf(app, n, 120000);
    await train(300);
    const a = { W: tr.trainW, H: tr.trainH, step: tr.stepCount, count: tr.count, val: (await tr.evaluateValidation())?.psnr };
    const params0 = tr.cpuParams.slice(0, 16 * 100);
    P.downsample = 2;
    await new Promise(res => setTimeout(res, 500));  // a frame: the app applies it
    await tr.syncSplatsFromGpu();
    const same = tr.cpuParams.slice(0, 16 * 100).every((v, i) => v === params0[i]);
    const b = { W: tr.trainW, H: tr.trainH, step: tr.stepCount, count: tr.count, same, val: (await tr.evaluateValidation())?.psnr };
    await train(100);
    const gt = tr.gtTextures.get(tr.trainViews[0]);
    return { a, b, c: { step: tr.stepCount, psnr: tr.lastPsnr, gt: gt ? [gt.width, gt.height] : null, val: (await tr.evaluateValidation())?.psnr },
             log: app.logText('train').split('\n').filter(l => /Training resolution/.test(l)) };
  });
  const { a, b, c } = r;
  if (!(b.W === 2 * a.W && b.H === 2 * a.H)) fail(`ds 4 -> 2: training at ${b.W} x ${b.H}, was ${a.W} x ${a.H}`);
  if (!(b.step === a.step && b.count === a.count && b.same)) fail(`ds 4 -> 2: step ${a.step} -> ${b.step}, splats ${a.count} -> ${b.count}, the splats ${b.same ? 'kept' : 'changed'}`);
  if (!(c.step === b.step + 100 && c.psnr > 0 && c.gt?.[0] === b.W && c.gt?.[1] === b.H)) fail(`100 steps at ds 2: step ${c.step}, loss PSNR ${c.psnr}, ground truth ${c.gt}`);
  if (!(Math.abs(b.val - a.val) < 3 && c.val > 0)) fail(`held out: ${a.val} at ds 4, ${b.val} at ds 2 before training on, ${c.val} after`);
  if (r.log.length !== 1) fail(`the change logged ${r.log.length} times`);
  return `${a.W} x ${a.H} @ ${a.step} (held out ${a.val.toFixed(2)} dB) -> ${b.W} x ${b.H}, the ${b.count} splats kept (${b.val.toFixed(2)} dB at the new size); ` +
         `100 more steps there: held out ${c.val.toFixed(2)} dB, ground truth ${c.gt.join(' x ')}`;
}, { serverArgs: [ROOM] });
