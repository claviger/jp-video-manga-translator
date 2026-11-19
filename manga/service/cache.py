"""Translated-page cache shared by the HTTP service and the CLI tools.

cache/<sha256 of source image bytes>.v<PIPELINE_VERSION>.png   translated page
cache/<sha256 of source image bytes>.v<PIPELINE_VERSION>.json  {"pairs": [[ja, en], ...], ...} for next-page context
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from . import config

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def paths(h: str) -> tuple[Path, Path]:
    base = config.CACHE_DIR / f"{h}.v{config.PIPELINE_VERSION}"
    return base.with_suffix(base.suffix + ".png"), base.with_suffix(base.suffix + ".json")


def get_png(h: str) -> bytes | None:
    p, _ = paths(h)
    return p.read_bytes() if p.exists() else None


def get_pairs(h: str | None) -> list[tuple[str, str]]:
    if not h or not _HEX64.match(h):
        return []
    _, jp = paths(h)
    if not jp.exists():
        return []
    try:
        return [tuple(p) for p in json.loads(jp.read_text(encoding="utf-8"))["pairs"]]
    except (ValueError, KeyError, TypeError):
        return []


def put(h: str, png: bytes, pairs, regions: int, seconds: float):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pp, jp = paths(h)
    # write-then-rename so a concurrent reader (service + CLI) never sees a half-written file
    for path, data in ((pp, png), (jp, json.dumps({"pairs": list(pairs), "regions": regions,
                                                   "seconds": round(seconds, 1)}, ensure_ascii=False).encode())):
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
