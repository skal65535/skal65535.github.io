// coverage.js — What to shoot next, after a photo set is mapped (DESIGN-sfm §13.3 P4, P6), DOM-free:
// the photos that did not join the main segment and their strongest link, the steps along the
// capture order where two photos are too far apart to match well, and how many photos see each
// point. SHOOTING holds the guide's numbers, measured on cuisine and salon (DESIGN-sfm §13.2).
"use strict";

import { poseCenter } from './camera.js';

// Two photos match well (>= minLinkMatches F inliers) when the view turns <= maxTurnDeg between them
// and the camera moves <= maxStep of the scene depth: on cuisine and salon (COLMAP's poses), 27 of the
// 32 pairs within both had >= 50 inliers, and 7 of the 126 pairs turning more than 60 deg
const SHOOTING = { maxTurnDeg: 30, maxStep: 0.5, minLinkMatches: 50 };
const START_MATCHES = 100;  // matches of the initial pair (mapper.js initializeWith)

const axisOf = (R) => [R[6], R[7], R[8]];  // the viewing direction in the world (row 3 of R)
const turnDeg = (Ra, Rb) => {
  const a = axisOf(Ra), b = axisOf(Rb);
  return Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * 180 / Math.PI;
};

// mapper: SequentialMapper after reconstruct(), its images in capture order (photos.js names).
// Returns {
//   main: segment id (the largest), inMain: photos in it, total,
//   apart: [{ photos, registered, link: { photo, other, matches } | null }]  the pieces outside the main segment
//          (a segment, or one unregistered photo) and their strongest pair into it (photo in the piece),
//   gaps: [{ a, b, matches, turnDeg, step }]  consecutive main photos (capture order) linked by < minLinkMatches,
//          with the turn and the step (centre distance over the median depth of a's points) between them,
//   seenBy: { two, threePlus } fractions of the main segment's points (the trainer keeps 3+, export.js),
//   strongest: { a, b, matches } the pair with the most matches, or null }
export function coverage(mapper, o = SHOOTING) {
  const main = mapper.largestSegment();
  const isMain = (k) => !!mapper.poses[k] && mapper.segment[k] === main;
  const matches = (i, j) => mapper.adj[i].find(e => e.other === j)?.a.length ?? 0;

  // the pieces apart (a segment, or one unregistered photo) and their strongest pair into the main one
  const pieces = new Map();
  for (let k = 0; k < mapper.n; k++) {
    if (isMain(k)) continue;
    const key = mapper.poses[k] ? `s${mapper.segment[k]}` : `u${k}`;
    if (!pieces.has(key)) pieces.set(key, { photos: [], registered: !!mapper.poses[k], link: null });
    const piece = pieces.get(key);
    piece.photos.push(k);
    for (const e of mapper.adj[k]) {
      if (isMain(e.other) && e.a.length > (piece.link?.matches ?? 0)) piece.link = { photo: k, other: e.other, matches: e.a.length };
    }
  }
  const apart = [...pieces.values()];

  // median depth of each main photo's points, for the step
  const depth = new Map(), centers = new Map();
  for (let k = 0; k < mapper.n; k++) if (isMain(k)) { depth.set(k, []); centers.set(k, poseCenter(mapper.poses[k])); }
  let two = 0, threePlus = 0;
  for (const P of mapper.points) {
    if (!P) continue;
    let n = 0;
    for (const img of P.obs.keys()) {
      if (!depth.has(img)) continue;
      n++;
      const { R, t } = mapper.poses[img];
      depth.get(img).push(R[6] * P.X[0] + R[7] * P.X[1] + R[8] * P.X[2] + t[2]);
    }
    if (n === 2) two++;
    else if (n >= 3) threePlus++;
  }
  const median = (v) => { const s = [...v].sort((a, b) => a - b); return s.length ? s[s.length >> 1] : NaN; };

  const gaps = [], order = [...depth.keys()].sort((a, b) => a - b);
  for (let i = 1; i < order.length; i++) {
    const a = order[i - 1], b = order[i], m = matches(a, b);
    if (m >= o.minLinkMatches) continue;
    const [ca, cb] = [centers.get(a), centers.get(b)];
    gaps.push({ a, b, matches: m, turnDeg: turnDeg(mapper.poses[a].R, mapper.poses[b].R),
                step: Math.hypot(ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]) / median(depth.get(a)) });
  }
  // the strongest pair of all (when nothing could start a map)
  let strongest = null;
  for (let i = 0; i < mapper.n; i++) for (const e of mapper.adj[i]) if (e.other > i && e.a.length > (strongest?.matches ?? 0)) strongest = { a: i, b: e.other, matches: e.a.length };
  const pts = two + threePlus;
  return { main, inMain: order.length, total: mapper.n, apart, gaps, strongest,
           seenBy: { two: pts ? two / pts : 0, threePlus: pts ? threePlus / pts : 0 } };
}

// The advice as sentences; name(k): a photo's short name (e.g. 'p07'); o: the guide's numbers; maxGaps,
// maxPieces: the steps and pieces apart listed
export function coverageAdvice(c, name, { o = SHOOTING, maxGaps = 3, maxPieces = 5 } = {}) {
  const out = [];
  if (c.inMain === 0) {  // no map at all: a start needs START_MATCHES on one pair (mapper.js initializeWith)
    const s = c.strongest;
    out.push(s ? `no two photos could start a map: the strongest pair, ${name(s.a)}–${name(s.b)}, has ${s.matches} matches (a start needs ${START_MATCHES}, with parallax); ` +
                 `retake the scene in smaller steps (turn <= ${o.maxTurnDeg}°, move <= ${o.maxStep}× the distance)`
               : 'no two photos match: retake the scene with the same subject in view, in small steps');
    return out;
  }
  for (const { photos, registered, link } of c.apart.slice(0, maxPieces)) {
    const who = photos.map(name).join(', '), many = photos.length > 1;
    const what = !registered ? 'did not join' : many ? `are a piece of their own (${photos.length} photos)` : 'is a piece of its own';
    if (!link) out.push(`${who} ${what}, with no match in the main piece: shoot from between the main piece and ${many ? 'them' : 'it'}`);
    else out.push(`${who} ${what}: the best link to the main piece is ${name(link.photo)}–${name(link.other)} ` +
                  `(${link.matches} matches); add a photo between those two, turning <= ${o.maxTurnDeg}°`);
  }
  if (c.apart.length > maxPieces) out.push(`${c.apart.slice(maxPieces).reduce((n, p) => n + p.photos.length, 0)} more photos apart (the log lists them)`);
  // the worst gaps first (turn and step relative to their limits), maxGaps of them
  const over = (g) => Math.max(g.turnDeg / o.maxTurnDeg, g.step / o.maxStep);
  const gaps = c.gaps.filter(g => over(g) > 1).sort((a, b) => over(b) - over(a));
  for (const g of gaps.slice(0, maxGaps)) {
    const why = [g.turnDeg > o.maxTurnDeg && `turned ${g.turnDeg.toFixed(0)}°`, g.step > o.maxStep && `moved ${g.step.toFixed(1)}× the scene depth`].filter(Boolean);
    out.push(`between ${name(g.a)} and ${name(g.b)} the camera ${why.join(' and ')} (${g.matches} matches): a photo in between would link them`);
  }
  if (gaps.length > maxGaps) out.push(`${gaps.length - maxGaps} more such step(s) between photos taken one after the other (the log lists them)`);
  if (c.inMain >= 2 && c.seenBy.threePlus < 0.5) {
    out.push(`${(100 * c.seenBy.two).toFixed(0)}% of the points are seen by only 2 photos: more overlap (each part of the scene in 3+ photos) trains better`);
  }
  return out;
}
