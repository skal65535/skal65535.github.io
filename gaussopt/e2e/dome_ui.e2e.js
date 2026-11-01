// dome_ui.e2e.js — The Dome step in the page (ui/dome_step.js, src/dome.js, ui/phases.js), on a video's COLMAP folder
// with sky. The Dome step's "Find the sky": a dome (the sky over 5%
// of the photos' pixels), in the trainer (setDome, 256 texels a face), in the view (the 'background (dome)' toggle)
// and in a saved scene (sceneEnv); 'dome only' (the dome alone, no splats nor mesh); the stepper's summary; the fill
// changed in the panel (panorama: the seen directions their color, not black), filled again at once. Then with dome=auto, Train: the dome first, from the photos
// before any step, then the training (it fails when the training starts without it).
//   cd e2e && node dome_ui.e2e.js   (env: DOME_E2E_DATASET, a COLMAP folder with sky; SHOTS: a folder for screenshots)
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { fail, appUrl, run, appReady } from './harness.js';

const DATASET = process.env.DOME_E2E_DATASET ?? `${process.env.HOME}/gskal/brush/gare-st-lazare/gs`;
if (!existsSync(join(DATASET, 'sparse'))) throw new Error(`no COLMAP folder in ${DATASET} (set DOME_E2E_DATASET)`);

await run(async (page) => {
  await page.goto(appUrl('?dataset=/dataset&ds=4'));
  await appReady(page, 300000);
  const a = await page.evaluate(async () => {
    const app = window.app, P = app.params;
    P.dome = 'auto';
    app.shell.show('dome');
    const before = [...document.querySelectorAll('#phase-foot button')].map(b => b.textContent);
    await app.domeStep.build();
    const d = app.domeStep.dome, states = app.shell.states;
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    const shown = app.renderer.envBuf === app.domeStep.buf && !!app.domeStep.buf;
    P.show_dome = false;
    await new Promise(r => setTimeout(r, 300));
    const hidden = app.renderer.envBuf === null;
    P.show_dome = true;
    // dome only: the view draws the dome alone (no splats, no mesh), the 'background (dome)' toggle off or not
    const rs = app.renderer.renderScene, drawn = [];
    app.renderer.renderScene = function (o) { drawn.push({ splats: o.numVisible, mesh: o.meshMode, dome: this.envBuf === app.domeStep.buf }); return rs.call(this, o); };
    const view = async (only, bg) => { P.dome_only = only; P.show_dome = bg; drawn.length = 0; await new Promise(r => setTimeout(r, 300)); return drawn.at(-1); };
    P.mesh_view = 1;  // the mesh over the splats (none here: what the view asks of the renderer)
    const only = [await view(true, true), await view(true, false), await view(false, true)];
    app.renderer.renderScene = rs;
    P.mesh_view = 0;
    const env = await app.domeStep.sceneEnv();
    // a dome from elsewhere (train.mjs DOME=: its sky share, no map nor gains; a scene's: neither), then the step's back
    const given = [];
    for (const farShare of [0.05, null]) {
      try { app.domeStep.set({ worth: true, farShare, n: d.n, cube: d.cube, latlong: null }); given.push(document.getElementById('dome-stats').textContent); }
      catch (e) { given.push(`threw: ${e.message}`); }
    }
    app.domeStep.set(d);
    // the fill changed in the step's panel: filled again at once (no new dome made), in the trainer, the preview, the stepper
    const sets = [], sd = app.trainer.setDome;
    app.trainer.setDome = function (cube, n) { sets.push(cube); return sd.call(this, cube, n); };
    const pick = (label) => {
      const row = [...document.querySelectorAll('#dome-gui .skal-row')].find(r => r.querySelector('.skal-row-label')?.textContent.trim() === 'fill');
      const sel = row?.querySelector('select');
      if (!sel) return false;
      sel.value = [...sel.options].find(o => o.textContent === label)?.value ?? '';
      sel.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    };
    const blackSeen = (dd) => {  // seen texels black in the filled map (magenta in the observed one)
      const { W, H, observed, filled } = dd.latlong;
      let n = 0;
      for (let s = 0; s < W * H; s++) if (observed[3 * s] === 204 && observed[3 * s + 1] === 0 && !filled[3 * s] && !filled[3 * s + 1] && !filled[3 * s + 2]) n++;
      return n;
    };
    const refill = { found: pick('panorama'), fill: app.domeStep.dome.fill, running: app.domeStep.running, same: app.domeStep.dome.farShare === d.farShare,
                     trainer: sets.length === 1 && sets[0] === app.domeStep.dome.cube && sets[0] !== d.cube, black: [blackSeen(d), blackSeen(app.domeStep.dome)],
                     stats: document.getElementById('dome-stats').textContent, summary: app.shell.states.dome.summary };
    pick('sky only');
    refill.back = app.domeStep.dome.fill;
    app.trainer.setDome = sd;
    app.domeStep.set(d);
    return { before, worth: d?.worth, farShare: d?.farShare, n: d?.n, domeN: app.trainer.domeN, summary: states.dome.summary,
             after: [...document.querySelectorAll('#phase-foot button')].map(b => b.textContent), shown, hidden,
             env: env && { n: env.n, bytes: env.rgb.length, sky: env.sky }, preview: !document.getElementById('dome-preview').hidden, given, only, refill };
  });
  if (process.env.SHOTS) await page.screenshot({ path: join(process.env.SHOTS, 'dome_step.png') });
  if (!a.before.includes('Find the sky')) fail(`the Dome step's buttons: ${a.before.join(' | ')}`);
  if (!(a.worth && a.farShare > 0.05 && a.n === 256 && a.domeN === 256)) fail(`the dome: ${JSON.stringify(a)}`);
  if (!(a.summary.startsWith('sky ') && a.after.includes('No dome') && a.preview)) fail(`the Dome step after it: ${JSON.stringify(a)}`);
  if (!(a.given[0].startsWith('Sky: 5.0%') && a.given[1].startsWith('A dome from the scene'))) fail(`a dome from a file, from a scene: ${a.given.join(' / ')}`);
  const r = a.refill;
  if (!(r.found && r.fill === 'seen' && !r.running && r.same && r.trainer && r.black[0] > 1000 && r.black[1] < r.black[0] / 100 &&
        r.stats.includes('panorama') && r.summary.endsWith(', panorama') && r.back === 'unseen'))
    fail(`the fill changed to panorama, then sky only: ${JSON.stringify(r)}`);
  if (!(a.only.slice(0, 2).every(d => d?.splats === 0 && d.mesh === 0 && d.dome) && a.only[2]?.splats > 0 && a.only[2].mesh === 1 && a.only[2].dome))
    fail(`dome only (on, on with the background off, off): ${JSON.stringify(a.only)}`);
  if (!(a.shown && a.hidden)) fail(`the view's background: drawn ${a.shown}, hidden by the toggle ${a.hidden}`);
  if (!(a.env?.n === 256 && a.env.bytes === 18 * 256 * 256 && a.env.sky === a.farShare)) fail(`the scene's background: ${JSON.stringify(a.env)}`);

  // dome=auto: Train makes the dome first, then trains
  await page.goto(appUrl('?dataset=/dataset&ds=4&dome=auto&stop_after=20'));
  await appReady(page, 300000);
  const b = await page.evaluate(async () => {
    const app = window.app, tr = app.trainer;
    app.setTraining(true);
    const waited = { running: app.domeStep.running, training: app.params.training, step: tr.stepCount };
    for (const t0 = Date.now(); !(tr.stepCount >= 20 && !app.params.training); await new Promise(r => setTimeout(r, 200)))
      if (Date.now() - t0 > 300000) throw new Error(`no 20 steps: step ${tr.stepCount}, dome ${!!app.domeStep.dome}, running ${app.domeStep.running}`);
    return { waited, worth: app.domeStep.dome?.worth, domeN: tr.domeN, step: tr.stepCount };
  });
  if (process.env.SHOTS) await page.screenshot({ path: join(process.env.SHOTS, 'dome_auto.png') });
  if (!(b.waited.running && !b.waited.training && b.waited.step === 0)) fail(`dome=auto, Train: ${JSON.stringify(b.waited)} (the dome first)`);
  if (!(b.worth && b.domeN === 256 && b.step >= 20)) fail(`dome=auto: ${JSON.stringify(b)}`);
  return `the Dome step: the sky ${(100 * a.farShare).toFixed(1)}% of the photos' pixels, a dome of ${a.n}² a face in the trainer, the view ` +
         `(hidden by its toggle, alone with 'dome only') and the scene's background; panorama: ${a.refill.black[1]} seen texels black ` +
         `(${a.refill.black[0]} with the sky only); '${a.summary}' in the stepper. dome=auto: Train waited for it, then ${b.step} steps`;
}, { serverArgs: [DATASET] });
