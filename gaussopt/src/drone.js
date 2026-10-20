// drone.js — "Drone fly" free camera: a small angle-mode quadcopter model.
//
//   mouse drag  : stick (pitch / roll targets); release = stick centered = self-leveling
//   Up / Down   : forward / backward (pitch stick)
//   Left / Right: yaw
//   E / Q       : climb / descend, released = altitude hold
//   Space       : re-center (level, brake, turn the heading towards the scene center)
//   wheel       : camera gimbal tilt
//
// The drone flies like a camera drone in angle mode: tilting accelerates it horizontally,
// thrust is compensated for tilt (no altitude loss in turns), banking adds a coordinated yaw,
// and linear drag gives a terminal speed. All distances scale with the scene radius.
"use strict";

import { clamp, wrapPi, approach, dot, cross, sub, madd, normalize, rotate, orbitBasis, viewFromAxes } from './math.js';

const DEG = Math.PI / 180;
export class Drone {
  constructor() {
    this.pos = [0, 0, 0];
    this.vel = [0, 0, 0];
    this.yaw = 0; this.pitch = 0; this.roll = 0;  // radians; pitch > 0 = nose down, roll > 0 = right wing down
    this.yawRate = 0;
    this.throttle = 0;   // smoothed [-1, 1]
    this.gimbal = 0;     // camera tilt below the body axis (radians, > 0 = looking down)
    this.stick = [0, 0]; // [roll, pitch] in the unit disc
    this.stickF = [0, 0]; // low-passed stick actually flown
    this.keys = new Set();  // pressed keys, lowercase KeyboardEvent.key ('arrowup', 'e', ' ', ...)
    this.recenter = 0;   // seconds of re-centering autopilot left
    this.up = [0, -1, 0];
    this.maxTilt = 30 * DEG;
  }

  // Start from a camera pose (view matrix + center): heading and gimbal come from the
  // view direction (3rd row of the rotation), the body starts level.
  resetFromView(view, center, up) {
    this.up = normalize(up);
    const { U, e1, e2 } = orbitBasis(this.up);
    const f = [view[2], view[6], view[10]];
    this.pos = [...center];
    this.vel = [0, 0, 0];
    this.yaw = Math.atan2(dot(f, e2), dot(f, e1));
    this.pitch = 0; this.roll = 0; this.yawRate = 0; this.throttle = 0;
    this.gimbal = clamp(Math.asin(clamp(-dot(f, U), -1, 1)), -30 * DEG, 85 * DEG);
    this.recenter = 0;
    this.stickF = [0, 0];
  }

  // World-space body axes: forward f, right r, down d (OpenCV camera axes, as buildOrbitViewMatrix)
  _axes(extraPitch = 0) {
    const { U, e1, e2 } = orbitBasis(this.up);
    let f = madd(madd([0, 0, 0], e1, Math.cos(this.yaw)), e2, Math.sin(this.yaw));
    let r = normalize(cross(f, U));
    f = rotate(f, r, -(this.pitch + extraPitch));  // nose down around the right axis
    r = rotate(r, f, this.roll);                   // right wing down
    const d = cross(f, r);
    return { f, r, d };
  }

  update(dt, { sceneCenter, sceneRadius }) {
    dt = clamp(dt, 0, 0.05);
    const R = sceneRadius || 1;
    const g = 1.2 * R;           // "gravity" (scene radii / s^2)
    const drag = 1.4;            // 1/s: terminal horizontal speed ~ g tan(maxTilt) / drag ~ 0.5 R/s
    const U = this.up;
    const k = this.keys;

    // Inputs. The stick is low-passed: quick to follow the mouse, slow to re-center on
    // release, so self-leveling eases out instead of snapping back.
    // Up / Down push the pitch stick (forward / backward) on top of the mouse.
    const keyPitch = 0.7 * ((k.has('arrowup') ? 1 : 0) - (k.has('arrowdown') ? 1 : 0));
    const stick = [this.stick[0], clamp(this.stick[1] + keyPitch, -1, 1)];
    const held = stick[0] !== 0 || stick[1] !== 0;
    const sRate = held ? 6 : 1.6;
    this.stickF[0] = approach(this.stickF[0], stick[0], sRate, dt);
    this.stickF[1] = approach(this.stickF[1], stick[1], sRate, dt);
    let tgtRoll  = this.stickF[0] * this.maxTilt;
    let tgtPitch = this.stickF[1] * this.maxTilt;
    let thr = (k.has('e') ? 1 : 0) - (k.has('q') ? 1 : 0);  // climb / descend
    // yaw > 0 turns left (right-handed around up, right = f x up)
    let yawCmd = ((k.has('arrowleft') ? 1 : 0) - (k.has('arrowright') ? 1 : 0)) * 90 * DEG;
    let extraDrag = 0;
    if (k.has(' ')) this.recenter = Math.max(this.recenter, 2.0);
    if (this.recenter > 0) {
      this.recenter -= dt;
      tgtRoll = 0; tgtPitch = 0; thr = 0;
      this.stickF[0] = this.stickF[1] = 0;
      const { e1, e2 } = orbitBasis(U);
      const toC = sub(sceneCenter, this.pos);
      const want = Math.atan2(dot(toC, e2), dot(toC, e1));
      yawCmd = clamp(wrapPi(want - this.yaw) * 2, -75 * DEG, 75 * DEG);
      this.gimbal = approach(this.gimbal, 0, 1.2, dt);
      extraDrag = 1.2;
    }

    // Angle-mode attitude: first-order approach to the stick targets
    this.roll  = approach(this.roll,  tgtRoll,  4, dt);
    this.pitch = approach(this.pitch, tgtPitch, 4, dt);
    // coordinated turn: banking yaws towards the lower wing
    const yawTarget = yawCmd - 1.6 * Math.sin(this.roll);
    this.yawRate = approach(this.yawRate, yawTarget, 3, dt);
    this.yaw = wrapPi(this.yaw + this.yawRate * dt);
    this.throttle = approach(this.throttle, thr, 4, dt);

    // Forces: tilt-compensated thrust along body up, gravity, drag (stronger vertically)
    const { d } = this._axes();
    const bodyUp = [-d[0], -d[1], -d[2]];
    const cosTilt = Math.max(0.5, Math.cos(this.roll) * Math.cos(this.pitch));
    const thrust = g * (1 + 0.8 * this.throttle) / cosTilt;
    const vUp = dot(this.vel, U);
    const vH = madd(this.vel, U, -vUp);
    let acc = madd([0, 0, 0], bodyUp, thrust);
    acc = madd(acc, U, -g);
    acc = madd(acc, vH, -(drag + extraDrag));
    acc = madd(acc, U, -(2.5 + extraDrag) * vUp);
    this.vel = madd(this.vel, acc, dt);
    this.pos = madd(this.pos, this.vel, dt);
  }

  // View matrix of the gimbaled camera
  view() {
    const { f, r, d } = this._axes(this.gimbal);
    return viewFromAxes(r, d, f, this.pos);
  }

  hud(sceneRadius) {
    const R = sceneRadius || 1;
    const speed = Math.hypot(this.vel[0], this.vel[1], this.vel[2]) / R;
    return `drone: ${speed.toFixed(2)} R/s, thr ${(this.throttle * 100).toFixed(0)}%, ` +
           `pitch ${(this.pitch / DEG).toFixed(0)}°, roll ${(this.roll / DEG).toFixed(0)}°, gimbal ${(this.gimbal / DEG).toFixed(0)}°`;
  }
}
