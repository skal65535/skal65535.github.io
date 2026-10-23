#!/usr/bin/env python3
"""Convert frames to 8-bit binary PGM for the Deno SIFT tools and tests (Deno has no JPEG decoder).

    gray.py SRC_DIR OUT_DIR [--limit N]

Images in name order; needs cv2.
"""

import argparse
from pathlib import Path

import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    names = sorted(f for f in a.src.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if a.limit:
        names = names[:a.limit]
    for f in names:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(a.out / (f.stem + ".pgm")), img)
    print(f"{len(names)} images -> {a.out}")


if __name__ == "__main__":
    main()
