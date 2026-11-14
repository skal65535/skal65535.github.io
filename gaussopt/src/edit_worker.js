// edit_worker.js — Find floater clusters off the page's thread (edit.js findClusters, ui/edit_panel.js): seconds on 300k
// splats, the page drawn meanwhile. In: { id, params (count rows, transferred), count, reach, isolation (x reach),
// scaleMul, maxSize, minSplats, small, cameras, limit }; out: findClusters' result + ms (the indices transferred), or
// { error }, with the job's id.
import { SplatGrid, findClusters } from './edit.js';

self.onmessage = ({ data: d }) => {
  try {
    const t0 = performance.now(), opts = { scaleMul: d.scaleMul };
    const grid = new SplatGrid(d.params, d.count, { ...opts, reach: d.reach });
    const iso = new SplatGrid(d.params, d.count, { ...opts, reach: d.isolation * d.reach });
    const res = findClusters(grid, d.maxSize, { isolation: iso, cameras: d.cameras, limit: d.limit, minSplats: d.minSplats, small: d.small });
    const moved = res.clusters.map(c => c.indices.buffer);
    if (res.small) moved.push(res.small.buffer);
    self.postMessage({ ...res, ms: performance.now() - t0, id: d.id }, moved);
  } catch (e) {
    self.postMessage({ error: e.message, id: d.id });
  }
};
