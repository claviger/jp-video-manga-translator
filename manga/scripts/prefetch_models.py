"""Download every model the manga pipeline uses, without needing the LLM or a GPU.

Usage: python -m scripts.prefetch_models [--force]

Fetches the text detector, OCR and inpainter that service/pipeline.py configures (into MANGA_MODEL_DIR, default
<manga root>/models) and pyhyphen's English hyphenation dictionary that MIT's renderer uses. Already-downloaded
files are kept. Prints how much it downloaded.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from service import config
from service.pipeline import build_config  # also puts third_party/mit on sys.path and stubs pydensecrf


def _dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _hyphen_dir() -> Path:
    from hyphen.dictools import DEFAULT_DICT_PATH  # pyhyphen keeps dictionaries in the user data dir
    return Path(DEFAULT_DICT_PATH)


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


async def fetch(force: bool) -> None:
    from manga_translator.detection import get_detector
    from manga_translator.inpainting import get_inpainter
    from manga_translator.ocr import get_ocr
    from manga_translator.utils.inference import ModelWrapper

    ModelWrapper._MODEL_DIR = str(config.MODEL_DIR)
    cfg = build_config()
    wanted = [
        ("detector", cfg.detector.detector, get_detector),
        ("ocr", cfg.ocr.ocr, get_ocr),
        ("inpainter", cfg.inpainter.inpainter, get_inpainter),
    ]
    for kind, key, get in wanted:
        model = get(key)
        if not isinstance(model, ModelWrapper):
            print(f"{kind} {key.value}: nothing to download")
            continue
        if model.is_downloaded() and not force:
            print(f"{kind} {key.value}: already present")
            continue
        print(f"{kind} {key.value}: downloading...", flush=True)
        await model.download(force=force)

    from manga_translator.rendering.text_render import select_hyphenator
    h = select_hyphenator("en")
    print("hyphenation dictionary (en): " + ("ok" if h is not None else "NOT available (text won't be hyphenated)"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dirs = [config.MODEL_DIR, _hyphen_dir()]
    before = sum(_dir_size(d) for d in dirs)
    try:
        asyncio.run(fetch(a.force))
    except Exception as e:  # noqa: BLE001 - installer-facing: one clear line
        print(f"model download failed: {e}", file=sys.stderr)
        return 1
    after = sum(_dir_size(d) for d in dirs)
    print(f"Downloaded {_human(max(0, after - before))}; models in {config.MODEL_DIR} total {_human(_dir_size(config.MODEL_DIR))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
