"""Generate the Fat Punks training dataset.

Output:
  data/bodies/{slim,chubby,fat,huge}/body_LLLL_NNNNN.png   (32x32 grayscale)
  data/labels.json                                          {filename: level}

Tier folders carry the coarse fatness label per the project spec; labels.json
carries the exact level (0..20) used for conditioning.
"""
import json, os, sys
import numpy as np
from PIL import Image
import spritegen

OUT = os.path.join(os.path.dirname(__file__), "data", "bodies")
TIERS = [("slim", 0, 5), ("chubby", 6, 10), ("fat", 11, 15), ("huge", 16, 20)]
PER_LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 280
SEED = 1337

def tier_of(level):
    for name, lo, hi in TIERS:
        if lo <= level <= hi:
            return name

def main():
    rng = np.random.default_rng(SEED)
    for name, _, _ in TIERS:
        os.makedirs(os.path.join(OUT, name), exist_ok=True)
    labels = {}
    n = 0
    for level in range(21):
        f = level / 20.0
        for k in range(PER_LEVEL):
            img = spritegen.draw_body(spritegen.Identity(rng), f)
            fn = f"body_{level:02d}_{k:05d}.png"
            Image.fromarray(spritegen.level_to_gray(img)).save(
                os.path.join(OUT, tier_of(level), fn))
            labels[fn] = level
            n += 1
    with open(os.path.join(os.path.dirname(__file__), "data", "labels.json"), "w") as fp:
        json.dump(labels, fp)
    print(f"wrote {n} images")

if __name__ == "__main__":
    main()
