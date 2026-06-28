"""Visual gate: skinny→fat strips per fixture punk, using the byte-exact
pure-Python forward pass + the same compositor the dataset used.

Usage: python3 verify/render_strip.py   (writes out/strip_<id>.png + contact sheet)
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from compose import composite  # noqa: E402
from verify.pure_forward import body_levels  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
FIXTURES = [0, 1, 2, 3, 4, 5, 117, 372, 635]
LEVELS = [0, 5, 10, 15, 20]
SCALE = 6


def strip(punk_id: int) -> Image.Image:
    tiles = []
    for lv in LEVELS:
        lvmap = np.frombuffer(body_levels(punk_id, lv), dtype=np.uint8)
        tiles.append(composite(punk_id, lvmap.reshape(32, 32), scale=SCALE))
    w, h = tiles[0].size
    out = Image.new("RGB", (w * len(tiles) + 4 * (len(tiles) - 1), h),
                    (19, 26, 30))
    for i, t in enumerate(tiles):
        out.paste(t, (i * (w + 4), 0))
    return out


def main():
    strips = []
    for pid in FIXTURES:
        s = strip(pid)
        s.save(os.path.join(OUT, f"strip_{pid}.png"))
        strips.append(s)
        print(f"strip_{pid}.png  ({'/'.join(str(l) for l in LEVELS)})")
    w, h = strips[0].size
    sheet = Image.new("RGB", (w, (h + 6) * len(strips)), (19, 26, 30))
    for i, s in enumerate(strips):
        sheet.paste(s, (0, i * (h + 6)))
    sheet.save(os.path.join(OUT, "strips_contact_sheet.png"))
    print("strips_contact_sheet.png")


if __name__ == "__main__":
    main()
