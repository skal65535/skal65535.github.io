#!/usr/bin/env python3
"""Undistort frames for a model exported by tools/map.js or tools/run_frames.js (--export) and color its points.

    undistort.py SOURCE_IMAGES_DIR OUT_DIR

Reads OUT_DIR/undistort.json (the SfM camera and the PINHOLE camera of the exported model),
writes OUT_DIR/images/<name> undistorted to the PINHOLE intrinsics (same size), and sets the
color of every point in OUT_DIR/sparse/points3D.bin to the median of its observations.
Needs cv2 and numpy.
"""

import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np


def opencv_camera(model, params):
    """COLMAP camera -> OpenCV K and distortion. COLMAP's pixel centers are at +0.5."""
    if model in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = params[:4]
    else:
        fx = fy = params[0]
        cx, cy = params[1:3]
    if model == "OPENCV":
        dist = params[4:8]
    elif model == "SIMPLE_RADIAL":
        dist = [params[3], 0, 0, 0]
    elif model == "RADIAL":
        dist = [params[3], params[4], 0, 0]
    else:
        dist = [0, 0, 0, 0]
    K = np.array([[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1]], dtype=np.float64)
    return K, np.array(dist, dtype=np.float64)


def read_points2d(path):
    """{image_id: (name, array of (x, y))} from images.bin"""
    data, off, out = path.read_bytes(), 8, {}
    for _ in range(struct.unpack_from("<Q", data, 0)[0]):
        image_id = struct.unpack_from("<i", data, off)[0]
        off += 64
        end = data.index(b"\0", off)
        name = data[off:end].decode()
        off = end + 1
        n = struct.unpack_from("<Q", data, off)[0]
        off += 8
        xy = np.frombuffer(data, dtype=np.float64, count=3 * n, offset=off).reshape(n, 3)[:, :2]
        off += 24 * n
        out[image_id] = (name, xy)
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    meta = json.loads((out / "undistort.json").read_text())
    K, dist = opencv_camera(meta["distorted"]["model"], meta["distorted"]["params"])
    P, _ = opencv_camera("PINHOLE", meta["pinhole"]["params"])
    obs = read_points2d(out / "sparse" / "images.bin")
    path = out / "sparse" / "points3D.bin"
    data = bytearray(path.read_bytes())
    tracks, off = [], 8  # (offset of the point, its track) per point
    for _ in range(struct.unpack_from("<Q", data, 0)[0]):
        track_len = struct.unpack_from("<Q", data, off + 43)[0]
        tracks.append((off, np.frombuffer(bytes(data[off + 51: off + 51 + 8 * track_len]), dtype=np.int32).reshape(-1, 2)))
        off += 51 + 8 * track_len
    wanted = {}  # image name -> [(point, observation index)]: frames are sampled as they go by, not kept
    for k, (_, track) in enumerate(tracks):
        for image_id, idx in track:
            wanted.setdefault(obs[int(image_id)][0], []).append((k, int(idx)))
    samples = [[] for _ in tracks]
    by_name = {name: xy for name, xy in obs.values()}

    (out / "images").mkdir(exist_ok=True)
    for name in meta["images"]:
        img = cv2.imread(str(src / name), cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"Cannot read {src / name}")
        und = cv2.undistort(img, K, dist, None, P)
        cv2.imwrite(str(out / "images" / name), und, [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = und.shape[:2]
        for k, idx in wanted.get(name, []):
            x, y = by_name[name][idx]  # the pixel containing (x, y), with COLMAP's centers at +0.5
            samples[k].append(und[min(max(int(np.floor(y)), 0), h - 1), min(max(int(np.floor(x)), 0), w - 1)].tolist())  # not a view: the frame goes
    print(f"undistorted {len(meta['images'])} images -> {out / 'images'}")

    # colors: median over each point's observations (points3D.bin rewritten in place)
    for (off, _), s in zip(tracks, samples):
        b, g, r = np.median(np.array(s), axis=0).astype(np.uint8)
        data[off + 32: off + 35] = bytes([r, g, b])
    path.write_bytes(bytes(data))
    print(f"colored {len(tracks)} points")

if __name__ == "__main__":
    main()
