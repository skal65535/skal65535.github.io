// The train PSNR curve's mean over a pass (render.js passMean): each sample is one view's PSNR, the views in a
// round robin, so the raw curve repeats the views' difficulty once a pass; the mean over a pass is flat.
import { passMean } from '../src/render.js';
import { assert, assertClose } from './util.js';

Deno.test('telemetry: the mean over a pass of a round robin of views is flat', () => {
  const views = [18, 22, 25, 20, 16, 23, 21], n = views.length;  // a PSNR per view
  const history = [];
  for (let step = 0; step < 2000; step += 5) history.push({ step, psnr: views[step % n] });  // a sample every 5 steps
  const m = passMean(history, n * 5);  // 5 x 7 steps: every view seen once in the window (5 and 7 coprime)
  const mean = views.reduce((a, b) => a + b, 0) / n;
  for (let i = n; i < m.length; i++) assertClose(m[i], mean, 1e-9, `sample ${i}`);
  assert(m[0] === views[0], 'the first sample alone');
  const raw = history.map(h => h.psnr);
  assert(Math.max(...raw) - Math.min(...raw) === 9, 'the raw curve swings by the views\' spread');
});
