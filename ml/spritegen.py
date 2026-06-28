"""Procedural fat-body sprite generator for Fat Punks training data. v2

Draws front-facing, headless cartoon fat BUSTS (no arms - punks are busts
cropped at the bottom, so the body is too) on a 32x32 grid, 5 shade levels:

    0 = background   1 = outline   2 = shadow   3 = base   4 = highlight

Parameterized by fatness f in [0,1] plus per-identity variation. For a FIXED
identity, every shape parameter is smooth + monotone in f, so one identity
grows believably slim -> enormous.

Geometry contract (mirrors compositing constants):
  body 32x32 pasted at canvas y=8; punk (24x24) at (4,0) on top; punk neck
  bottoms out at canvas row 23 == body row 15 around canvas cols 13..17.
"""

from __future__ import annotations
import math
import numpy as np

SIZE = 32
GRAY = np.array([0, 60, 130, 190, 250], dtype=np.uint8)
BG, OUT, SHA, BASE, HI = 0, 1, 2, 3, 4


class Identity:
    """Per-punk body identity: sampled once, reused across all fatness levels."""

    def __init__(self, rng: np.random.Generator):
        self.cx = 15.0 + rng.uniform(-0.5, 0.5)
        self.sh_base = rng.uniform(5.0, 6.2)         # slim shoulder half-width
        self.sh_gain = rng.uniform(2.6, 4.0)         # shoulder growth to f=1
        self.bel_gain = rng.uniform(5.6, 7.6)        # belly bulge growth to f=1
        self.bc_frac = rng.uniform(0.58, 0.72)       # belly center (of span)
        self.bsig_frac = rng.uniform(0.34, 0.44)     # belly sigma (of span)
        self.shoulder_rise = rng.uniform(2.2, 3.6)   # rows the top climbs w/ f
        self.s0_base = rng.uniform(12.2, 13.4)       # slim shoulder-top row
        self.asym = rng.uniform(-0.45, 0.45)
        self.roll_rows = sorted(rng.choice(np.arange(0, 100), 2, replace=False))
        self.shine = rng.random() > 0.12
        self.crease = rng.random() > 0.10
        self.has_rolls = rng.random() > 0.18
        self.crease_span = rng.uniform(0.30, 0.42)
        self.shine_x = rng.uniform(-4.2, -2.6)       # offset from cx


def profile(idn: Identity, f: float):
    """Per-row half-widths (left, right) + shoulder-top row + roll rows."""
    f = min(1.0, max(0.0, f))
    s0 = int(round(idn.s0_base - idn.shoulder_rise * f))
    neck_hw = 3.0
    span = (SIZE - 1) - s0
    sh_hw = idn.sh_base + idn.sh_gain * f            # shoulders at all f
    bA = idn.bel_gain * (f ** 1.15)                  # belly amplitude
    bc = s0 + span * idn.bc_frac                     # belly center row
    bsig = span * idn.bsig_frac

    wl = np.zeros(SIZE)
    wr = np.zeros(SIZE)
    prev_hw = 0.0
    for y in range(SIZE):
        if y < s0 - 6:
            continue
        if y < s0:
            wl[y] = wr[y] = neck_hw
            continue
        # shoulder envelope: rounded rise to sh_hw over ~4.5 rows, slight drift
        t = (y - s0) / 4.5
        e = math.sin(min(1.0, t) * math.pi / 2)
        hw = neck_hw + (sh_hw - neck_hw) * e
        hw += 0.5 * min(1.0, (y - s0) / max(1.0, bc - s0))   # gentle drift out
        # belly bulge
        hw += bA * math.exp(-(((y - bc) / bsig) ** 2))
        hw = min(14.3, hw)
        if y == SIZE - 1:                            # tiny crop-edge rounding
            hw = max(hw - 0.8, prev_hw - 1.0)
        prev_hw = hw
        a = idn.asym * min(1.0, 2 * (y - s0) / span) * (0.4 + 0.6 * f)
        wl[y] = min(14.4, hw + a)
        wr[y] = min(14.4, hw - a)

    # symmetric roll pinches: 1px in on both sides at fixed fractional rows
    rolls = []
    if f > 0.5 and idn.has_rolls:
        depth = 0.6 + 0.9 * (f - 0.5) / 0.5
        n_rolls = 1 if (f < 0.85 or idn.crease) else 2     # max 2 horiz features
        for rr in idn.roll_rows[:n_rolls]:
            ry = s0 + int(round(span * (0.30 + 0.40 * rr / 100)))
            ry = max(s0 + 5, min(SIZE - 6, ry))
            if any(abs(ry - p) < 4 for p in rolls):
                continue
            rolls.append(ry)
            for dy in (-1, 0, 1):
                y = ry + dy
                d = depth if dy == 0 else depth * 0.45
                wl[y] -= d
                wr[y] -= d
    return wl, wr, s0, rolls


def draw_body(idn: Identity, f: float) -> np.ndarray:
    img = np.zeros((SIZE, SIZE), dtype=np.uint8)
    wl, wr, s0, rolls = profile(idn, f)

    L = np.full(SIZE, -1)
    R = np.full(SIZE, -1)
    for y in range(SIZE):
        if wl[y] <= 0:
            continue
        l = max(0, int(round(idn.cx - wl[y])))
        r = min(SIZE - 1, int(round(idn.cx + wr[y])))
        if r <= l:
            continue
        L[y], R[y] = l, r
        img[y, l:r + 1] = BASE

    body = img > 0
    for y in range(SIZE):
        for x in range(SIZE):
            if not body[y, x]:
                continue
            if y == 0 or x == 0 or x == SIZE - 1:
                img[y, x] = OUT
                continue
            up = body[y - 1, x]
            dn = body[y + 1, x] if y < SIZE - 1 else True   # bottom = crop
            lf = body[y, x - 1]
            rt = body[y, x + 1]
            if not (up and dn and lf and rt):
                img[y, x] = OUT

    belly_peak = s0 + (SIZE - 1 - s0) * idn.bc_frac
    for y in range(SIZE):
        if L[y] < 0 or R[y] - L[y] < 4:
            continue
        l, r = L[y], R[y]
        wide = (r - l) > 13
        sh = 2 if wide else 1                        # shadow inside right edge
        for x in range(max(l + 1, r - sh), r):
            if img[y, x] == BASE:
                img[y, x] = SHA
        if s0 < y <= belly_peak + 2:                 # rim light inside left edge
            for x in range(l + 1, min(l + 2, r)):
                if img[y, x] == BASE:
                    img[y, x] = HI

    # under-pec crease: short, curved (ends dip), only on chubby+
    if idn.crease and f > 0.45:
        cy = s0 + 4 + int(round(1.5 * f))
        if 0 < cy < SIZE - 1 and L[cy] >= 0:
            span = int((R[cy] - L[cy]) * idn.crease_span)
            c = int(round(idn.cx))
            for x in range(c - span, c + span + 1):
                yy = cy + (1 if abs(x - c) >= span - 1 else 0)   # curved ends
                if 0 <= x < SIZE and img[yy, x] in (BASE, HI):
                    img[yy, x] = SHA

    # roll creases: continuous sagging curve between the two silhouette pinches
    for ry in rolls:
        if L[ry] < 0 or ry + 1 >= SIZE:
            continue
        l, r = L[ry], R[ry]
        third = max(2, (r - l) // 3)
        for x in range(l + 1, r):
            yy = ry + 1 if (l + third < x < r - third) else ry   # middle sags
            if img[yy, x] in (BASE, HI):
                img[yy, x] = SHA

    # belly shine: small patch clearly inside the body, left of center
    if idn.shine and f > 0.34:
        by = int(round(belly_peak)) - 2
        bx = int(round(idn.cx + idn.shine_x))
        w = 2
        h = 2 + (1 if f > 0.62 else 0)
        for yy in range(by, min(SIZE - 1, by + h)):
            for xx in range(bx, bx + w):
                if 0 <= xx < SIZE and L[yy] >= 0 and L[yy] + 2 < xx < R[yy] - 2 \
                   and img[yy, xx] == BASE:
                    img[yy, xx] = HI

    return img


def level_to_gray(img: np.ndarray) -> np.ndarray:
    return GRAY[img]


def sample(rng: np.random.Generator, f: float) -> np.ndarray:
    return draw_body(Identity(rng), f)
