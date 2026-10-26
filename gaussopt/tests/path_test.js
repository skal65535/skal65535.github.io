// path.js (DESIGN-sfm §13.3 P7): the shortest open path through the cameras, exact and 2-opt
import { shortestPath, stepCosts, orderCost, pathStats } from '../src/path.js';
import { makeRng } from '../src/math.js';
import { assert } from './util.js';

const view = (x, y = 0, z = 0, fwd = [0, 1, 0]) => ({ center: [x, y, z], forward: fwd });

Deno.test('path: cameras along a line, shuffled, come back in line order', () => {
  const xs = [3, 0, 7, 1, 5, 2, 6, 4], views = xs.map(x => view(x));
  const order = shortestPath(views, { rotWeight: 0 }).map(i => xs[i]);
  const up = order.join() === '0,1,2,3,4,5,6,7', down = order.join() === '7,6,5,4,3,2,1,0';
  assert(up || down, order.join());
  assert(Math.abs(pathStats(shortestPath(views, { rotWeight: 0 }), views).length - 7) < 1e-9, 'length 7');
});

// every permutation of 7 views: the exact path is the cheapest
Deno.test('path: exact (Held-Karp) equals brute force; 2-opt is never better and close', () => {
  const rng = makeRng(3);
  const rand = (n) => Array.from({ length: n }, () => view(10 * rng(), 10 * rng(), rng(), [rng() - 0.5, 1, rng() - 0.5]));
  const views = rand(7), { C } = stepCosts(views, { rotWeight: 1 }), n = 7;
  let best = Infinity;
  const perm = (a, k) => {
    if (k === a.length) { best = Math.min(best, orderCost(a, C, n)); return; }
    for (let i = k; i < a.length; i++) { [a[k], a[i]] = [a[i], a[k]]; perm(a, k + 1); [a[k], a[i]] = [a[i], a[k]]; }
  };
  perm([...Array(n).keys()], 0);
  const exact = orderCost(shortestPath(views, { rotWeight: 1 }), C, n);
  assert(Math.abs(exact - best) < 1e-9, `exact ${exact}, brute force ${best}`);
  // 16 views: exact; the same views through 2-opt (17+ uses it): a copy with a far 17th view
  const v16 = rand(16), e16 = stepCosts(v16, { rotWeight: 1 });
  const exact16 = orderCost(shortestPath(v16, { rotWeight: 1 }), e16.C, 16);
  const v17 = [...v16, view(1e4, 0, 0)], e17 = stepCosts(v17, { rotWeight: 1 });
  const o17 = shortestPath(v17, { rotWeight: 1 }), heur = orderCost(o17.filter(i => i < 16), e16.C, 16);
  assert(o17.length === 17 && new Set(o17).size === 17, 'a permutation');
  assert(heur >= exact16 - 1e-9 && heur <= 1.05 * exact16, `2-opt ${heur.toFixed(3)}, exact ${exact16.toFixed(3)}`);
  assert(e17.scale > 0, 'scale');
});

// two rows of cameras facing opposite ways: by distance alone the path zigzags between the rows;
// with the turn weighted it walks one row, then the other (one turn)
Deno.test('path: the turn weight keeps the cameras facing one way together', () => {
  const views = [];
  for (let k = 0; k < 6; k++) { views.push(view(k, 0, 0, [0, 1, 0])); views.push(view(k + 0.5, 0.2, 0, [0, -1, 0])); }
  const turns = (o) => o.slice(1).filter((v, k) => views[v].forward[1] !== views[o[k]].forward[1]).length;
  const plain = shortestPath(views, { rotWeight: 0 }), weighted = shortestPath(views, { rotWeight: 2 });
  assert(turns(plain) > 2 && turns(weighted) === 1, `turns: distance only ${turns(plain)}, weighted ${turns(weighted)}`);
});
