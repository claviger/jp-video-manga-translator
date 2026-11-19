"""Translate individual image files.

Usage: ./translate.sh page1.png [page2.png ...] [--local | --service] [--no-cache]
Pages are treated as consecutive: each page gets the previous page's translations as context.
Writes <name>.en.png next to each input. Like translate_archive, it uses the running service if there is one
(so the GPU models aren't loaded twice) and falls back to the in-process pipeline otherwise.
"""
import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from service import cache, config
from scripts.translate_archive import SERVICE, FatalError, LocalBackend, ServiceBackend


async def main(a) -> int:
    paths = [Path(p) for p in a.images]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        sys.exit("No such file: " + ", ".join(map(str, missing)))
    token = config.TOKEN_FILE.read_text().strip() if config.TOKEN_FILE.exists() else ""
    if not a.local and token and ServiceBackend.available(token):
        backend, how = ServiceBackend(token, a.no_cache), "running service"
    elif a.service:
        sys.exit(f"The local service isn't reachable on {SERVICE} (or the token in {config.TOKEN_FILE} is wrong).")
    else:
        backend, how = LocalBackend(a.no_cache), "in-process pipeline"
    print(f"{len(paths)} image(s)  [{how}]", flush=True)
    prev_hash, failed = "", 0
    try:
        for p in paths:
            t0 = time.time()
            data = p.read_bytes()
            h = cache.image_hash(data)
            try:
                png, hit = await backend.translate(data, prev_hash)
            except FatalError as e:
                print(f"stopping: {e}", file=sys.stderr)
                return 1
            except Exception as e:  # noqa: BLE001 - report and carry on with the next page
                print(f"{p.name}: FAILED: {e}", file=sys.stderr)
                failed += 1
                prev_hash = ""
                continue
            prev_hash = h
            out = p.with_suffix(".en.png")
            out.write_bytes(png)
            print(f"{p.name}: {'cached' if hit else f'{time.time() - t0:.1f}s'} -> {out}")
            for ja, en in cache.get_pairs(h) or []:
                print(f"   {ja}  =>  {en}")
    finally:
        backend.close()
    return 2 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--no-cache", action="store_true", help="re-translate even if cached")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--local", action="store_true", help="always run the pipeline in-process")
    g.add_argument("--service", action="store_true", help="require the running service")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    sys.exit(asyncio.run(main(a)))
