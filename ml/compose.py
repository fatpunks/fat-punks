"""Compositing pipeline for Fat Punks — Python mirror of the on-chain renderer.

Canvas: 32 wide x 40 tall (rendered as SVG rects on-chain, PNG here).
  - body (32x32 level map) pasted at (0, BODY_Y=8)
  - punk (24x24 RGBA)     pasted at (PUNK_X=4, 0), drawn ON TOP

Skin sampling (must match Solidity byte-for-byte):
  probe punk pixel (x=9, y=23) — the lit lower-left neck pixel. Exhaustive
  check across all 10,000 punks: this coord is ALWAYS type skin/fur
  (#ead9d9/#dbb180/#ae8b61/#713f1d humans, #7da269 zombie, #352410 ape,
  #c8fbfb alien). Single deterministic read on-chain: byte offset (23*24+9)*4.

Shade ramp from skin (r,g,b), integer math only:
  outline   = c * 9  / 25
  shadow    = c * 18 / 25
  base      = c
  highlight = min(255, c * 5 / 4)
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from PIL import Image

CANVAS_W, CANVAS_H = 32, 40
BODY_Y = 8
PUNK_X = 4
BG_COLOR = (99, 133, 150)            # punk blue-gray #638596 (standardized)
SKIN_PROBES = [(9, 23)]  # universal skin pixel: verified across all 10,000 punks

PUNKS_SHEET_ENV = "FATPUNKS_PUNKS_SHEET"
DEFAULT_PUNKS_SHEET = Path(__file__).resolve().parent.parent / "refs" / "punks.png"

_sheet = None


def punks_sheet_path() -> Path:
    configured = os.environ.get(PUNKS_SHEET_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_PUNKS_SHEET


def punk_rgba(idx: int) -> np.ndarray:
    global _sheet
    if _sheet is None:
        path = punks_sheet_path()
        if not path.exists():
            raise FileNotFoundError(
                f"missing punk sheet: set {PUNKS_SHEET_ENV} or place refs/punks.png"
            )
        _sheet = np.array(Image.open(path).convert("RGBA"))
    r, c = divmod(idx, 100)
    return _sheet[r * 24:(r + 1) * 24, c * 24:(c + 1) * 24]


def sample_skin(punk: np.ndarray) -> tuple[int, int, int]:
    for (x, y) in SKIN_PROBES:
        px = punk[y, x]
        if px[3] > 0:
            return int(px[0]), int(px[1]), int(px[2])
    return (219, 177, 128)            # unreachable fallback


def skin_ramp(rgb) -> list[tuple[int, int, int]]:
    r, g, b = rgb

    def m(c, num, den):
        return min(255, c * num // den)
    return [
        (m(r, 9, 25), m(g, 9, 25), m(b, 9, 25)),     # 1 outline
        (m(r, 18, 25), m(g, 18, 25), m(b, 18, 25)),  # 2 shadow
        (r, g, b),                                   # 3 base
        (m(r, 5, 4), m(g, 5, 4), m(b, 5, 4)),        # 4 highlight
    ]


def composite(punk_idx: int, body_levels: np.ndarray,
              bg=BG_COLOR, scale: int = 1) -> Image.Image:
    """body_levels: 32x32 uint8 in {0..4}. Returns RGB image."""
    punk = punk_rgba(punk_idx)
    ramp = skin_ramp(sample_skin(punk))

    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:, :] = bg

    for y in range(32):
        for x in range(32):
            lv = body_levels[y, x]
            if lv > 0:
                canvas[BODY_Y + y, x] = ramp[lv - 1]

    for y in range(24):
        for x in range(24):
            px = punk[y, x]
            if px[3] > 0:
                canvas[y, PUNK_X + x] = px[:3]

    img = Image.fromarray(canvas)
    if scale > 1:
        img = img.resize((CANVAS_W * scale, CANVAS_H * scale), Image.NEAREST)
    return img
