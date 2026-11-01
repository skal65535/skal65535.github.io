// stipple_worker.js — Stipple seeds off the page's thread (stipple.js seedsFromView, E4): the trainer reads a view
// back some steps before a rebalance (GaussOptTrainer._prepareSeeds) and the seeds are ready when it comes, instead
// of the page computing them meanwhile (~300 ms at 720 x 960, the GPU idle). In: seedsFromView's args; out: its
// result + ms, or { error }, with the job's id.
import { seedsFromView } from './stipple.js';

self.onmessage = ({ data }) => {
  try {
    const t0 = performance.now(), s = seedsFromView(data);
    self.postMessage({ ...s, ms: performance.now() - t0, id: data.id }, [s.params.buffer]);
  } catch (e) {
    self.postMessage({ error: e.message, id: data.id });
  }
};
