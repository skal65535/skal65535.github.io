// coverage.js (DESIGN-sfm §13.3 P4, P6): the shooting advice after mapping a photo set, on
// synthetic scenes where the missing link is known.
import { makeSyntheticScene } from '../src/sfm/synth.js';
import { SequentialMapper } from '../src/sfm/mapper.js';
import { coverage, coverageAdvice } from '../src/sfm/coverage.js';
import { assert } from './util.js';

const BASE = { maxFeatures: 1000, numGround: 8000, numBoxes: 8, numFar: 1000 };
const name = (k) => `p${String(k + 1).padStart(2, '0')}`;

// Two halves joined by a pair of 10 matches (5-6): mapped without registerRelative, the map splits
// and the second half is one piece apart, its link 5-6 named
Deno.test('coverage: a split map names the piece apart and its link', () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 12, turns: 0.5, overlap: 11, minLoopMatches: 1 });
  const pairs = [];
  for (const p of scene.input.pairs) {
    if ((p.i < 6) === (p.j < 6)) pairs.push(p);
    else if (p.i === 5 && p.j === 6) pairs.push({ ...p, matches: p.matches.slice(0, 20) });
  }
  const mapper = new SequentialMapper({ ...scene.input, pairs }, { relativeRegistration: false });
  mapper.run();
  const c = coverage(mapper), advice = coverageAdvice(c, name);
  assert(c.inMain === 6 && c.apart.length === 1 && c.apart[0].photos.length === 6, `${c.inMain} in main, apart ${JSON.stringify(c.apart)}`);
  const { link } = c.apart[0];
  assert(link && link.matches === 10 && [5, 6].includes(link.photo) && [5, 6].includes(link.other), `link ${JSON.stringify(link)}`);
  assert(advice.some(s => s.includes('a piece of their own (6 photos)') && s.includes(`best link to the main piece is ${name(link.photo)}–${name(link.other)} (10 matches)`)),
         advice.join('\n'));
  assert(c.gaps.length === 0, `gaps ${JSON.stringify(c.gaps)}`);
});

// A full turn in 10 photos (about 40 deg apart), pair 3-4 unmatched: the map holds through 2-4 and 3-5,
// and the step 3-4 is reported with its turn
Deno.test('coverage: an unmatched step past maxTurnDeg is a gap', () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 10, turns: 1, overlap: 9 });
  const pairs = scene.input.pairs.filter(p => !(p.i === 3 && p.j === 4));
  const mapper = new SequentialMapper({ ...scene.input, pairs }, { relativeRegistration: true });
  mapper.run();
  const c = coverage(mapper), advice = coverageAdvice(c, name);
  assert(c.inMain === 10, `${c.inMain} in main`);
  assert(c.gaps.length === 1 && c.gaps[0].a === 3 && c.gaps[0].b === 4 && c.gaps[0].matches === 0, JSON.stringify(c.gaps));
  const [A, B] = [scene.truth.images[3].R, scene.truth.images[4].R];
  const truth = Math.acos(A[6] * B[6] + A[7] * B[7] + A[8] * B[8]) * 180 / Math.PI;  // the orbit sways: not 40 exactly
  assert(truth > 30 && Math.abs(c.gaps[0].turnDeg - truth) < 0.5, `turn ${c.gaps[0].turnDeg}, truth ${truth}`);
  assert(advice.some(s => s.startsWith(`between p04 and p05 the camera turned ${truth.toFixed(0)}°`)), advice.join('\n'));
});

// Two unmatched steps past maxTurnDeg, maxGaps 1: the worse one is listed, the other counted
Deno.test('coverage: the advice lists the worst gaps first, maxGaps of them', () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 10, turns: 1, overlap: 9 });
  const pairs = scene.input.pairs.filter(p => !(p.i === 3 && p.j === 4) && !(p.i === 1 && p.j === 2));  // turns 48 and 42 deg
  const mapper = new SequentialMapper({ ...scene.input, pairs }, { relativeRegistration: true });
  mapper.run();
  const c = coverage(mapper);
  assert(c.inMain === 10 && c.gaps.length === 2, `${c.inMain} in main, gaps ${JSON.stringify(c.gaps)}`);
  const worst = [...c.gaps].sort((a, b) => b.turnDeg - a.turnDeg)[0];  // both step < maxStep here: the turn decides
  assert(c.gaps.every(g => g.step < 0.5 && g.turnDeg > 30), JSON.stringify(c.gaps));
  const advice = coverageAdvice(c, name, { maxGaps: 1 });
  const listed = advice.filter(s => s.startsWith('between'));
  assert(listed.length === 1 && listed[0].startsWith(`between ${name(worst.a)} and ${name(worst.b)}`), advice.join('\n'));
  assert(advice.some(s => s.startsWith('1 more such step')), advice.join('\n'));
});

// Nothing placed: one sentence, the strongest pair and what a start needs
Deno.test('coverage: with no map, the advice names the strongest pair', () => {
  const scene = makeSyntheticScene({ ...BASE, numFrames: 6 });
  const pairs = scene.input.pairs.map(p => ({ ...p, matches: p.matches.slice(0, 2 * (p.i === 1 && p.j === 2 ? 60 : 20)) }));
  const mapper = new SequentialMapper({ ...scene.input, pairs });
  mapper.run();
  const c = coverage(mapper), advice = coverageAdvice(c, name);
  assert(c.inMain === 0 && advice.length === 1 && advice[0].includes('the strongest pair, p02–p03, has 60 matches (a start needs 100'), advice.join('\n'));
});
