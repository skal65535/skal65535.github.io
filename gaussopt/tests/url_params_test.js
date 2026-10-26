// url.js urlAfterChange (names: params.js urlNamesOf): a UI change rewrites a param given in the URL (by name, alias or sfm_ name),
// leaves the others as typed, and adds nothing
import { urlNamesOf, URL_FLAGS } from '../src/params.js';
import { urlAfterChange } from '../src/url.js';
import { assert } from './util.js';

const eq = (a, b, what) => assert(a === b, `${what}: ${JSON.stringify(a)}, expected ${JSON.stringify(b)}`);

Deno.test('urlAfterChange: only params in the URL, by alias or name, the rest as typed', () => {
  const q = '?dataset=/datasets/jardin1_gs&ds=2&show_octree=0&train&max_splats=50000';
  eq(urlAfterChange(q, urlNamesOf('downsample'), 4, URL_FLAGS), '?dataset=/datasets/jardin1_gs&ds=4&show_octree=0&train&max_splats=50000', 'alias');
  eq(urlAfterChange(q, urlNamesOf('show_octree'), true, URL_FLAGS), '?dataset=/datasets/jardin1_gs&ds=2&show_octree=1&train&max_splats=50000', 'boolean by name');
  eq(urlAfterChange(q, urlNamesOf('training'), false, URL_FLAGS), '?dataset=/datasets/jardin1_gs&ds=2&show_octree=0&max_splats=50000', 'flag off');
  eq(urlAfterChange(q, urlNamesOf('training'), true, URL_FLAGS), null, 'flag on, already there');
  eq(urlAfterChange(q, urlNamesOf('haze'), 1.5, URL_FLAGS), null, 'not in the URL');
  eq(urlAfterChange(q, urlNamesOf('dataset_url'), '/x', URL_FLAGS), null, "dataset: _syncUrl's");
  eq(urlAfterChange('?training=1', urlNamesOf('training'), false, URL_FLAGS), '?training=0', 'by name: not a flag');
  eq(urlAfterChange('?target=0,0,0', urlNamesOf('target'), [1, 2.5, 3], URL_FLAGS), '?target=1,2.5,3', 'array');
  eq(urlAfterChange('?sfm_spacing=0.03&photos=/a/b.jpg', ['sfm_spacing'], 0.05), '?sfm_spacing=0.05&photos=/a/b.jpg', 'SfM panel');
  eq(urlAfterChange('?train', urlNamesOf('training'), false, URL_FLAGS), '', 'the last param');
});
