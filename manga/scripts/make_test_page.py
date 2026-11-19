"""Draw a synthetic manga-style page: panel borders, oval speech bubbles, vertical Japanese text.

It's only a smoke test for the pipeline (detection -> OCR -> translation -> inpaint -> render); real scans are
much harder. Usage: python scripts/make_test_page.py out.png [--variant 2]
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = Path(__file__).resolve().parent.parent / "third_party" / "mit" / "fonts" / "msgothic.ttc"

PAGES = {
    1: [  # (bubble center x, y, lines (right-to-left columns))
        (980, 300, ["ねえ、", "今日は", "一緒に帰ろうよ"]),
        (520, 330, ["ごめん、", "まだ仕事が", "残ってるんだ"]),
        (930, 1050, ["じゃあ", "待ってる", "からね！"]),
        (420, 1080, ["本当に", "いいの？"]),
    ],
    2: [
        (980, 300, ["遅く", "なって", "ごめんね"]),
        (500, 320, ["ずっと", "待ってた", "んだから"]),
        (900, 1060, ["お腹", "すいた", "でしょ？"]),
        (430, 1070, ["うん、", "ラーメン", "食べに行こう"]),
    ],
}


def draw_vertical(d, cx, cy, columns, font, size):
    gap = int(size * 1.25)
    longest = max(len(c) for c in columns)
    x0 = cx + (len(columns) - 1) * gap // 2
    y0 = cy - longest * size // 2
    for ci, col in enumerate(columns):  # first column on the right
        x = x0 - ci * gap
        for ri, ch in enumerate(col):
            if ch in "、。":
                d.text((x + size // 2, y0 + ri * size - size // 2), ch, font=font, fill="black")
            elif ch in "ー":
                d.text((x, y0 + ri * size), "｜", font=font, fill="black")
            else:
                d.text((x - size // 2, y0 + ri * size), ch, font=font, fill="black", anchor="la")


# Variant 3: horizontal Japanese. (kind, x0, y0, lines): "bubble" = wide oval, "box" = rectangular caption,
# "free" = lines over screentone with no bubble.
HORIZONTAL = [
    ("bubble", 700, 280, ["今日はとても楽しかったね、また来ようよ"]),
    ("box", 120, 900, ["その日、彼女は初めて", "自分の気持ちに気づいた。", "もう後戻りはできない。"]),
    ("free", 160, 1450, ["ドキドキが止まらない", "どうしよう…"]),
]


def draw_horizontal_page(d, W, size, font):
    for x in range(60, W - 60, 14):  # screentone-ish hatching behind the free text
        d.line([(x, 1380), (x + 60, 1700)], fill=(150, 150, 150), width=2)
    for kind, x0, y0, lines in HORIZONTAL:
        lh = int(size * 1.4)
        tw = max(len(l) for l in lines) * size
        th = len(lines) * lh
        if kind == "bubble":
            d.ellipse([x0 - tw // 2 - 90, y0 - th // 2 - 60, x0 + tw // 2 + 90, y0 + th // 2 + 60],
                      fill="white", outline="black", width=5)
            ox, oy = x0 - tw // 2, y0 - th // 2
        elif kind == "box":
            d.rectangle([x0 - 30, y0 - 30, x0 + tw + 30, y0 + th + 20], fill="white", outline="black", width=4)
            ox, oy = x0, y0
        else:
            ox, oy = x0, y0
        for i, line in enumerate(lines):
            if kind == "free":  # white outline so it reads over the hatching, like real SFX/monologue text
                d.text((ox, oy + i * lh), line, font=font, fill="black", stroke_width=4, stroke_fill="white")
            else:
                d.text((ox, oy + i * lh), line, font=font, fill="black")


# Variant 4: placement stress test. Vertical bubbles that bleed off the left/right/top page edges (common in real
# manga) and a tight cluster of overlapping bubbles in the middle.
EDGE_AND_CLUSTER = [
    (40, 420, ["こんなところで", "何してるの？"]),          # bleeds off the left edge
    (1370, 600, ["ずっと探してたんだよ", "どこ行ってたの"]),   # bleeds off the right edge
    (700, 30, ["おはよう", "ございます"]),                # bleeds off the top edge
    (560, 1250, ["待って", "行かないで"]),                # cluster of three overlapping bubbles
    (760, 1300, ["離して", "もう遅いの"]),
    (650, 1500, ["お願い", "話を聞いて"]),
]


def draw_edge_page(d, size, font):
    for cx, cy, cols in EDGE_AND_CLUSTER:
        longest = max(len(c) for c in cols)
        bw = len(cols) * int(size * 1.25) // 2 + 50
        bh = longest * size // 2 + 60
        d.ellipse([cx - bw, cy - bh, cx + bw, cy + bh], fill="white", outline="black", width=5)
        draw_vertical(d, cx, cy, cols, font, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--variant", type=int, default=1, choices=sorted(PAGES) + [3, 4])
    a = ap.parse_args()
    W, H = 1400, 2000
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    size = 44
    font = ImageFont.truetype(str(FONT), size)
    # four panels with a little hatching so it isn't a blank page
    for box in [(40, 40, W - 40, 760), (40, 800, W - 40, 1960)]:
        d.rectangle(box, outline="black", width=6)
    for x in range(60, W - 60, 24):
        d.line([(x, 700), (x + 40, 740)], fill=(90, 90, 90), width=2)
        d.line([(x, 1900), (x + 40, 1940)], fill=(90, 90, 90), width=2)
    if a.variant == 4:
        draw_edge_page(d, size, font)
        img.save(a.out)
        print(a.out)
        return
    if a.variant == 3:
        draw_horizontal_page(d, W, size, font)
        img.save(a.out)
        print(a.out)
        return
    for cx, cy, cols in PAGES[a.variant]:
        longest = max(len(c) for c in cols)
        bw = len(cols) * int(size * 1.25) // 2 + 70
        bh = longest * size // 2 + 80
        d.ellipse([cx - bw, cy - bh, cx + bw, cy + bh], fill="white", outline="black", width=5)
        draw_vertical(d, cx, cy, cols, font, size)
    img.save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
