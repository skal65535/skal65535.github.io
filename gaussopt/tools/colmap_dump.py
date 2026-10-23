#!/usr/bin/env python3
"""Dump a COLMAP database to the compact format read by src/sfm/dump.js (DESIGN-sfm §8.2).

    colmap_dump.py DATABASE OUT_DIR [--overlap 15] [--descriptors] [--limit N] [--raw-matches]

OUT_DIR gets:
    dump.json        camera (database prior), images in name (= time) order, pairs
    keypoints.f32    x, y per keypoint (COLMAP pixel convention), images concatenated
    matches.u32      keypoint index pairs (i-th image first) of each pair's verified inliers
    descriptors.u8   128 bytes per keypoint (only with --descriptors; large)
    raw.u32          matches before geometric verification (only with --raw-matches), same pair
                     layout as matches.u32 with pairs[].rawOffset / rawMatches

--limit N keeps the first N images (name order) and the pairs between them.

Pairs come from two_view_geometries (verified inliers). Image indices are positions in name
order, with i < j; F (row-major, when COLMAP stored one) satisfies x_j^T F x_i = 0 in pixels.
type is "seq" when j - i <= --overlap (what sequential matching pairs), "loop" otherwise
(vocabulary-tree loop detection), so tests can hold loop pairs back. Standard library only.
"""

import argparse
import json
import sqlite3
import struct
import sys
from array import array
from contextlib import nullcontext
from pathlib import Path

MAX_IMAGE_ID = 2147483647  # COLMAP kMaxNumImages: pair_id = id1 * MAX + id2, id1 < id2
CAMERA_MODELS = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL", 4: "OPENCV"}
CONFIGS = {0: "UNDEFINED", 1: "DEGENERATE", 2: "CALIBRATED", 3: "UNCALIBRATED", 4: "PLANAR",
           5: "PANORAMIC", 6: "PLANAR_OR_PANORAMIC", 7: "WATERMARK", 8: "MULTIPLE"}


def doubles(blob, n):
    return list(struct.unpack(f"<{n}d", blob)) if blob and len(blob) == 8 * n else None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("database", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--overlap", type=int, default=15,
                   help="sequential matching overlap used for the database (default: 15, as video2splat.py)")
    p.add_argument("--descriptors", action="store_true", help="also write descriptors.u8")
    p.add_argument("--limit", type=int, default=0, help="first N images only")
    p.add_argument("--raw-matches", action="store_true", help="also write raw.u32 (unverified matches)")
    a = p.parse_args()
    if not a.database.exists():
        sys.exit(f"No database {a.database}")
    a.out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f"file:{a.database}?mode=ro", uri=True)

    cams = db.execute("SELECT camera_id, model, width, height, params FROM cameras").fetchall()
    if len(cams) != 1:
        sys.exit(f"Expected a single camera, found {len(cams)}")
    _, model, width, height, params = cams[0]
    if model not in CAMERA_MODELS:
        sys.exit(f"Unsupported camera model {model}")
    camera = {"model": CAMERA_MODELS[model], "width": width, "height": height,
              "params": list(struct.unpack(f"<{len(params) // 8}d", params))}

    rows = db.execute("SELECT image_id, name FROM images ORDER BY name").fetchall()
    if a.limit:
        rows = rows[:a.limit]
    index = {image_id: i for i, (image_id, _) in enumerate(rows)}
    images, offset = [], 0
    with open(a.out / "keypoints.f32", "wb") as fk, \
         (open(a.out / "descriptors.u8", "wb") if a.descriptors else nullcontext()) as fd:
        for image_id, name in rows:
            n, cols, blob = db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?",
                                       (image_id,)).fetchone() or (0, 2, b"")
            kp = array("f")
            kp.frombytes(blob or b"")
            xy = array("f", bytes(8 * n))
            xy[0::2], xy[1::2] = kp[0::cols], kp[1::cols]
            xy.tofile(fk)
            if a.descriptors:
                d = db.execute("SELECT rows, cols, data FROM descriptors WHERE image_id = ?",
                               (image_id,)).fetchone()
                if not d or d[0] != n or d[1] != 128:
                    sys.exit(f"{name}: descriptors do not match keypoints")
                fd.write(d[2])
            images.append({"id": image_id, "name": name, "numKeypoints": n, "offset": offset})
            offset += n

    pairs, offset, counts = [], 0, {}
    with open(a.out / "matches.u32", "wb") as fm:
        for pair_id, n, blob, config, F in db.execute(
                "SELECT pair_id, rows, data, config, F FROM two_view_geometries WHERE rows > 0"):
            id1, id2 = divmod(pair_id, MAX_IMAGE_ID)
            if id1 not in index or id2 not in index:
                continue
            i, j = index[id1], index[id2]
            m = array("I")
            m.frombytes(blob)
            F = doubles(F, 9)
            if i > j:  # name order differs from id order: swap images, match columns, and F -> F^T
                i, j = j, i
                m[0::2], m[1::2] = m[1::2], m[0::2]
                F = F and [F[3 * c + r] for r in range(3) for c in range(3)]
            m.tofile(fm)
            kind = "seq" if j - i <= a.overlap else "loop"
            counts[kind] = counts.get(kind, 0) + 1
            pairs.append({"i": i, "j": j, "type": kind, "config": CONFIGS.get(config, str(config)),
                          "numMatches": n, "offset": offset, "F": F})
            offset += n
    pairs.sort(key=lambda q: (q["i"], q["j"]))
    if a.raw_matches:
        by_ij = {(q["i"], q["j"]): q for q in pairs}
        raw_off = 0
        with open(a.out / "raw.u32", "wb") as fr:
            for pair_id, n, blob in db.execute("SELECT pair_id, rows, data FROM matches WHERE rows > 0"):
                id1, id2 = divmod(pair_id, MAX_IMAGE_ID)
                if id1 not in index or id2 not in index:
                    continue
                i, j = index[id1], index[id2]
                m = array("I")
                m.frombytes(blob)
                if i > j:
                    i, j = j, i
                    m[0::2], m[1::2] = m[1::2], m[0::2]
                q = by_ij.get((i, j))
                if q is None:  # matched but not verified: keep it, with no inliers
                    q = {"i": i, "j": j, "type": "seq" if j - i <= a.overlap else "loop", "config": "NONE",
                         "numMatches": 0, "offset": offset, "F": None}
                    pairs.append(q)
                    by_ij[(i, j)] = q
                m.tofile(fr)
                q["rawOffset"], q["rawMatches"] = raw_off, n
                raw_off += n
        pairs.sort(key=lambda q: (q["i"], q["j"]))

    meta = {"version": 1, "source": "colmap", "database": str(a.database.resolve()),
            "overlap": a.overlap, "descriptors": a.descriptors,
            "camera": camera, "images": images, "pairs": pairs}
    (a.out / "dump.json").write_text(json.dumps(meta))
    print(f"{len(images)} images, {sum(im['numKeypoints'] for im in images)} keypoints, "
          f"{len(pairs)} pairs ({counts.get('seq', 0)} seq, {counts.get('loop', 0)} loop), "
          f"{offset} matches -> {a.out}")


if __name__ == "__main__":
    main()
