"""Typesetting for regions whose *Japanese* was written horizontally.

MIT's manga2eng renderer treats every region like a speech bubble: it enlarges the region by its aspect ratio (up
to 3x), finds a balloon-shaped mask around it and grows lines out from the mask's centroid. That's great for
vertical text in bubbles, but a wide, flat line of horizontal Japanese turns into a squarish English block that
spills over the bubble edge or onto the lines above/below it.

Here we keep the horizontal footprint instead: wrap the English to the region's width and pick the largest font
that fits the region's height. Only if it can't fit at a readable size do we grow vertically, and then only into
free space (never into another text region), before finally shrinking further.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .placement import make_layer

log = logging.getLogger("mlt.hlayout")

H_LINE_ASPECT = 1.5      # a detected Japanese text line this much wider than tall counts as horizontal
PAD_X, PAD_Y = 0.06, 0.12  # free padding around the original text box, as fractions of its width/height
LINE_SPACING = 1.12      # line pitch as a multiple of the font size
GROWTH_TIERS = (1.0, 1.6, 2.4, 3.2)  # allowed box height multiples, tried in order
BG_TOLERANCE = 40        # grey levels from the box's median that still count as the same background
WIDE_MAX = 1.5           # may widen up to this multiple of the original width, into free space only
MIN_FONT = 12


def is_source_horizontal(region) -> bool:
    """True if the region's detected Japanese lines are mostly wide-and-flat (area-weighted)."""
    h_area = v_area = 0.0
    for quad in region.lines:
        q = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
        w, h = np.ptp(q[:, 0]), np.ptp(q[:, 1])
        if w >= H_LINE_ASPECT * h:
            h_area += w * h
        elif h >= H_LINE_ASPECT * w:
            v_area += w * h
    return h_area > v_area


def _wrap(draw, text: str, font, max_w: float) -> list[str] | None:
    """Greedy word wrap. None if a single word is wider than max_w."""
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}" if cur else word
        if draw.textlength(cand, font=font) <= max_w:
            cur = cand
            continue
        if not cur or draw.textlength(word, font=font) > max_w:
            return None
        lines.append(cur)
        cur = word
    if cur:
        lines.append(cur)
    return lines


def _balance(draw, text, font, max_w, n_lines) -> list[str]:
    """Narrowest wrap width that still gives n_lines, so lines come out even instead of one long + an orphan."""
    lo, hi, best = 0.0, max_w, _wrap(draw, text, font, max_w)
    for _ in range(12):
        mid = (lo + hi) / 2
        lines = _wrap(draw, text, font, mid)
        if lines is not None and len(lines) <= n_lines:
            best, hi = lines, mid
        else:
            lo = mid
    return best


def _free_space(box, others, page_w, page_h):
    """(left, top, right, bottom) limits the box may grow to before touching another region."""
    x1, y1, x2, y2 = box
    left, top, right, bottom = 0, 0, page_w, page_h
    for ox1, oy1, ox2, oy2 in others:
        if not (ox2 <= x1 or ox1 >= x2):  # overlaps horizontally: limits vertical growth
            if oy2 <= y1:
                top = max(top, oy2 + 4)
            elif oy1 >= y2:
                bottom = min(bottom, oy1 - 4)
        if not (oy2 <= y1 or oy1 >= y2):  # overlaps vertically: limits horizontal growth
            if ox2 <= x1:
                left = max(left, ox2 + 4)
            elif ox1 >= x2:
                right = min(right, ox1 - 4)
    return left, top, right, bottom


def _background_limits(gray: np.ndarray, box, max_dx: int, max_dy: int):
    """(left, top, right, bottom) of the open background around a region: the connected area of roughly the
    colour of the (already inpainted) text box. Inside a speech bubble that's the bubble's interior, so growth
    stops at the outline; over art or screentone it's about the box itself, so no growth there.

    Uses the narrowest row/column of the band (10th percentile) so curved bubble edges aren't crossed."""
    page_h, page_w = gray.shape
    x1, y1, x2, y2 = box
    rx1, ry1 = max(0, x1 - max_dx), max(0, y1 - max_dy)
    rx2, ry2 = min(page_w, x2 + max_dx), min(page_h, y2 + max_dy)
    roi = gray[ry1:ry2, rx1:rx2].astype(np.int16)
    inner = gray[y1:y2, x1:x2]
    if inner.size == 0:
        return box
    ref = int(np.median(inner))
    n, labels = cv2.connectedComponents((np.abs(roi - ref) < BG_TOLERANCE).astype(np.uint8), connectivity=4)
    inner_labels = labels[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1]
    vals, counts = np.unique(inner_labels[inner_labels > 0], return_counts=True)
    if not len(vals) or counts.max() < 0.5 * inner_labels.size:  # box isn't mostly flat background
        return box
    comp = labels == vals[np.argmax(counts)]

    def run(lines):  # length of the leading True run in each 1-D array
        return np.array([np.argmin(l) if not l.all() else len(l) for l in lines])

    bx1, by1, bx2, by2 = x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1
    rows, cols = comp[by1:by2], comp[:, bx1:bx2].T
    right = x2 + int(np.percentile(run(rows[:, bx2:]), 10))
    left = x1 - int(np.percentile(run(rows[:, :bx1][:, ::-1]), 10))
    bottom = y2 + int(np.percentile(run(cols[:, by2:]), 10))
    top = y1 - int(np.percentile(run(cols[:, :by1][:, ::-1]), 10))
    return left, top, right, bottom


def _fit(draw, text, font_path, sizes, start_fs, floor_fs):
    """Largest font (start_fs..floor_fs) whose wrapped text fits, trying (width, height) boxes in order."""
    for box_w, max_h in sizes:
        for fs in range(start_fs, floor_fs - 1, -1):
            font = ImageFont.truetype(font_path, fs)
            lines = _wrap(draw, text, font, box_w)
            if lines is not None and len(lines) * fs * LINE_SPACING <= max_h:
                return fs, box_w
    return None


def _groups(plans):
    """Union neighbouring rows that started at about the same Japanese size (a stacked monologue/caption split
    into several regions), so they get one shared English size instead of each its own maximum."""
    parent = list(range(len(plans)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(plans):
        for j in range(i + 1, len(plans)):
            b = plans[j]
            (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = a["box"], b["box"]
            similar = abs(a["orig_fs"] - b["orig_fs"]) <= 0.2 * max(a["orig_fs"], b["orig_fs"])
            x_overlap = min(ax2, bx2) - max(ax1, bx1) > 0
            v_gap = max(by1 - ay2, ay1 - by2)
            if similar and x_overlap and v_gap < 1.5 * max(ay2 - ay1, by2 - by1):
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(plans))]


def render_horizontal(img: np.ndarray, regions, font_path: str, all_regions=None) -> list:
    """Lay out each region's translation in (roughly) its original horizontal footprint and return one
    placement.Layer per region; placement.place() composites them after fixing off-page/overlapping boxes.

    img: the inpainted page (used to find bubble interiors). all_regions: every region on the page (vertical ones
    too), so growth never runs onto another region."""
    if not regions:
        return []
    page_h, page_w = img.shape[:2]
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))  # text measurement only
    all_boxes = [tuple(int(v) for v in r.xyxy) for r in (all_regions or regions)]
    gray = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)

    def box_sizes(w, h, cx, cy, limits):
        left, top, right, bottom = limits
        # growth is symmetric so the text stays centred on where the Japanese was
        room_w = 2 * min(cx - left, right - cx)
        room_h = 2 * min(cy - top, bottom - cy)
        base_w, base_h = w * (1 + 2 * PAD_X), h * (1 + 2 * PAD_Y)
        widths = [base_w, max(base_w, min(w * WIDE_MAX, room_w))]
        heights = [max(base_h, min(base_h * t, room_h)) for t in GROWTH_TIERS]
        return [(bw, bh) for bh in heights for bw in widths]  # widen before growing taller

    # Pass 1: fit each region on its own.
    plans = []
    for r in regions:
        # manga2eng sets everything in caps (comic convention, and the comic fonts' lowercase is small caps)
        text = " ".join((r.translation or "").split()).upper()
        if not text:
            continue
        box = tuple(int(v) for v in r.xyxy)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        free = _free_space(box, [b for b in all_boxes if b != box], page_w, page_h)
        bgl = _background_limits(gray, box, int(w * (WIDE_MAX - 1) / 2) + 8, int(h * GROWTH_TIERS[-1] / 2) + 8)
        inside = (max(free[0], bgl[0]), max(free[1], bgl[1]), min(free[2], bgl[2]), min(free[3], bgl[3]))
        inner_sizes = box_sizes(w, h, cx, cy, inside)   # stays inside the bubble / flat background
        outer_sizes = box_sizes(w, h, cx, cy, free)     # only avoids other text regions
        orig_fs = int(r.font_size) if r.font_size and r.font_size > 0 else 32
        start_fs = max(MIN_FONT, min(orig_fs, 64))
        readable = max(MIN_FONT, int(orig_fs * 0.65))
        fit = (_fit(draw, text, font_path, inner_sizes, start_fs, readable)
               or _fit(draw, text, font_path, outer_sizes, start_fs, readable)
               or _fit(draw, text, font_path, outer_sizes[-1:], readable, MIN_FONT))
        sizes = outer_sizes
        if fit is None:
            log.warning("text overflows its region at %dpx: %r", MIN_FONT, text[:60])
            fit = (MIN_FONT, sizes[-1][0])
        plans.append({"r": r, "text": text, "box": box, "cx": cx, "cy": cy, "orig_fs": orig_fs,
                      "fs": fit[0], "box_w": fit[1]})

    # Pass 2: one size per group of stacked rows (a smaller font always still fits the box it was fitted to).
    groups = _groups(plans)
    for p, g in zip(plans, groups):
        p["fs"] = min(q["fs"] for q, gq in zip(plans, groups) if gq == g)

    layers = []
    for p in plans:
        fs = p["fs"]
        font = ImageFont.truetype(font_path, fs)
        lines = _wrap(draw, p["text"], font, p["box_w"]) or [p["text"]]
        if len(lines) > 1:
            lines = _balance(draw, p["text"], font, p["box_w"], len(lines))
        fg, bg = p["r"].get_font_colors()
        fg, bg = tuple(int(c) for c in fg), tuple(int(c) for c in bg)
        stroke = max(1, round(fs * 0.12))
        pitch = fs * LINE_SPACING
        cw = int(max(draw.textlength(l, font=font) for l in lines) + 4 * stroke + 8)
        ch = int(len(lines) * pitch + 4 * stroke + 8)
        canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        cd = ImageDraw.Draw(canvas)
        y = (ch - len(lines) * pitch) / 2
        for line in lines:
            cd.text((cw / 2, y + pitch / 2), line, font=font, fill=fg + (255,), anchor="mm",
                    stroke_width=stroke, stroke_fill=bg + (255,))
            y += pitch
        layer = make_layer(canvas, p["cx"] - cw / 2, p["cy"] - ch / 2)
        if layer is not None:
            layers.append(layer)
    return layers
