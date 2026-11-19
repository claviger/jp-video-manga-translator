"""Run detection/OCR/inpaint/render on an image with a canned translator (no Tabby), print each region's geometry,
and save the result. For iterating on layout/rendering without spending LLM calls.

    .venv/Scripts/python -m scripts.render_debug samples/horiz.png [out.png] [--raw]
    --raw composites without placement fixes (off-page / overlap), for comparison.
"""
import asyncio
import sys
from pathlib import Path

from service.hlayout import is_source_horizontal

import service.pipeline as P
import service.placement as P_placement

CANNED = {
    "今日はとても楽しかったね、また来ようよ": "Today was so much fun. Let's come back again sometime.",
    "その日、彼女は初めて自分の気持ちに気づいた。もう後戻りはできない。":
        "That day, she realized her feelings for the first time. There was no turning back.",
    "ドキドキが止まらないどうしよう…": "My heart won't stop pounding... What do I do...",
    "俺はずっと前から": "For a long time now,", "お前のことが": "I've been...",
    "好きだったんだ": "in love with you.", "ちょっと待って！": "Wait a second!",
    "話を聞いて": "Hear me out", "お願いだから…": "Please...", "本当に違うの": "It's really not like that",
    "こんなところで何してるの？": "What on earth are you doing out here in a place like this?",
    "ずっと探してたんだよどこ行ってたの": "I've been looking for you all this time! Where did you go?",
    "おはようございます": "Good morning to you, everybody!",
    "待って行かないで": "Wait! Please don't go, don't leave me here!",
    "離してもう遅いの": "Let go of me! It's already too late for us!",
    "お願い話を聞いて": "Please, I'm begging you, just listen to me!",
}


def fake_translate(tabby, texts, prev):
    return [CANNED.get(t, f"[{len(t)} chars] " + "lorem ipsum " * max(1, len(t) // 6)) for t in texts]


async def main(src: Path, out: Path, raw: bool = False):
    P.translate_page = fake_translate
    if raw:  # composite without the placement fixes, for before/after comparison
        P.place = P_placement.composite
    pipe = P.Pipeline(None)
    orig = pipe.mt._dispatch_with_context

    async def spy(cfg, texts, ctx):
        res = await orig(cfg, texts, ctx)
        for r, t in zip(ctx.text_regions, texts):
            x, y, w, h = r.xywh
            print(f"region {x},{y} {w}x{h} aspect={w / max(h, 1):.2f} src_horizontal={is_source_horizontal(r)} lines={len(r.lines)} "
                  f"font={r.font_size} text={t!r}")
        return res

    pipe.mt._dispatch_with_context = spy
    data = src.read_bytes()
    png, _, n = await pipe.translate(data, P.cache.image_hash(data), "")
    out.write_bytes(png)
    print(f"{n} regions -> {out}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")
    logging.getLogger("mlt").setLevel(logging.INFO)
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:] if a != "--raw"]
    src = Path(args[0])
    asyncio.run(main(src, Path(args[1]) if len(args) > 1 else src.with_suffix(".dbg.png"), "--raw" in sys.argv))
