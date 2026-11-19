"""Final placement pass for rendered text.

Both renderers (MIT's manga2eng for vertical source text, hlayout.py for horizontal) produce one RGBA layer per
region instead of painting onto the page. This pass sees all layers at once and fixes the two failure modes they
can't see individually:

  * text partly off the page (manga2eng centres text on an enlarged balloon estimate that can extend past the
    edge): shrink anything larger than the page, then shift boxes back inside;
  * text piled on other text (neighbouring regions enlarged into each other): push overlapping boxes apart along
    the axis of least overlap; if there isn't room, shrink the crowded boxes a little and try again.

Layers that are already inside the page and clear of each other are left exactly where they were.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

log = logging.getLogger("mlt.placement")

MARGIN = 4          # px kept between text and the page edge
GAP = 4             # px kept between two text boxes
SEPARATE_ITERS = 80
SHRINK_ROUNDS = 5
SHRINK_STEP = 0.9


@dataclass
class Layer:
    img: Image.Image   # RGBA, cropped to its visible pixels
    x: float           # top-left on the page
    y: float
    x0: float = 0.0    # original top-left, for reporting displacement
    y0: float = 0.0

    @property
    def w(self):
        return self.img.width

    @property
    def h(self):
        return self.img.height

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    def scale(self, f: float):
        cx, cy = self.cx, self.cy
        nw, nh = max(1, round(self.w * f)), max(1, round(self.h * f))
        self.img = self.img.resize((nw, nh), Image.LANCZOS)
        self.x, self.y = cx - nw / 2, cy - nh / 2


def make_layer(img: Image.Image, x: float, y: float) -> Layer | None:
    """Crop an RGBA image to its non-transparent pixels and return it as a Layer (None if fully transparent)."""
    img = img.convert("RGBA")
    bbox = img.getchannel("A").getbbox()
    if bbox is None:
        return None
    img = img.crop(bbox)
    return Layer(img, x + bbox[0], y + bbox[1], x + bbox[0], y + bbox[1])


def _clamp(l: Layer, W: int, H: int):
    l.x = min(max(l.x, MARGIN), W - MARGIN - l.w)
    l.y = min(max(l.y, MARGIN), H - MARGIN - l.h)


def _overlap(a: Layer, b: Layer):
    """(dx, dy) penetration depths including GAP, or None if a and b are clear of each other."""
    dx = min(a.x + a.w, b.x + b.w) - max(a.x, b.x) + GAP
    dy = min(a.y + a.h, b.y + b.h) - max(a.y, b.y) + GAP
    if dx <= 0 or dy <= 0:
        return None
    return dx, dy


def _separate(layers: list[Layer], W: int, H: int) -> bool:
    """Push overlapping layers apart. Returns True if everything ended up clear."""
    for _ in range(SEPARATE_ITERS):
        moved = False
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                a, b = layers[i], layers[j]
                ov = _overlap(a, b)
                if ov is None:
                    continue
                dx, dy = ov
                moved = True
                if dy <= dx:  # cheaper to separate vertically
                    s = 1 if b.cy >= a.cy else -1
                    a.y -= s * dy / 2
                    b.y += s * dy / 2
                else:
                    s = 1 if b.cx >= a.cx else -1
                    a.x -= s * dx / 2
                    b.x += s * dx / 2
                _clamp(a, W, H)
                _clamp(b, W, H)
        if not moved:
            return True
    return not any(_overlap(layers[i], layers[j]) for i in range(len(layers)) for j in range(i + 1, len(layers)))


def place(base: np.ndarray, layers: list[Layer]) -> np.ndarray:
    """Fit every layer inside the page, separate overlaps, and composite onto base (RGB array)."""
    layers = [l for l in layers if l is not None]
    if not layers:
        return base
    H, W = base.shape[:2]

    for l in layers:  # anything bigger than the page gets shrunk to fit
        f = min((W - 2 * MARGIN) / l.w, (H - 2 * MARGIN) / l.h, 1.0)
        if f < 1.0:
            l.scale(f)
        _clamp(l, W, H)

    clear = _separate(layers, W, H)
    rounds = 0
    while not clear and rounds < SHRINK_ROUNDS:
        crowded = {id(l) for i, l in enumerate(layers) for m in layers[i + 1:] if _overlap(l, m)} | \
                  {id(m) for i, l in enumerate(layers) for m in layers[i + 1:] if _overlap(l, m)}
        for l in layers:
            if id(l) in crowded:
                l.scale(SHRINK_STEP)
                _clamp(l, W, H)
        clear = _separate(layers, W, H)
        rounds += 1
    if not clear:
        log.warning("couldn't fully separate %d text boxes on a %dx%d page", len(layers), W, H)

    moved = [(l, ((l.x - l.x0) ** 2 + (l.y - l.y0) ** 2) ** 0.5) for l in layers]
    moved = [(l, d) for l, d in moved if d >= 1]
    if moved:
        log.info("placement: moved %d of %d text boxes (max %.0f px)%s", len(moved), len(layers),
                 max(d for _, d in moved), f", shrank in {rounds} round(s)" if rounds else "")

    page = Image.fromarray(np.ascontiguousarray(base)).convert("RGBA")
    for l in layers:
        page.alpha_composite(l.img, (round(l.x), round(l.y)))
    return np.asarray(page.convert("RGB"))


def composite(base: np.ndarray, layers: list[Layer]) -> np.ndarray:
    """Composite layers where they are, without any fixes (for before/after comparisons)."""
    page = Image.fromarray(np.ascontiguousarray(base)).convert("RGBA")
    for l in layers:
        if l is None:
            continue
        x, y = round(l.x), round(l.y)
        img = l.img
        # alpha_composite can't take negative offsets: crop the part that hangs off the top/left
        if x < 0 or y < 0:
            img = img.crop((max(0, -x), max(0, -y), img.width, img.height))
            x, y = max(0, x), max(0, y)
        if x < page.width and y < page.height:
            img = img.crop((0, 0, min(img.width, page.width - x), min(img.height, page.height - y)))
            page.alpha_composite(img, (x, y))
    return np.asarray(page.convert("RGB"))
