"""Translate every page in a comic archive (.zip/.cbz/.rar/.cbr) and write a translated .cbz (or .cbr).

    .venv/Scripts/python -m scripts.translate_archive "Some Book.cbr"
    .venv/Scripts/python -m scripts.translate_archive book.cbz -o "book (EN).cbz" --format jpg

Pages are processed in natural sort order and each page gets the previous page's translations as context.
Non-image members (ComicInfo.xml etc.) are copied through unchanged; pages that fail are kept untranslated
and listed at the end (exit code 2).

Backend: if the local service is running (127.0.0.1:8765, token in service/token.txt) pages go through it, so
the GPU models aren't loaded twice. Otherwise the pipeline runs in this process. Either way the translated-page
cache is shared, so re-running after an interruption skips finished pages.

Output format: writing RAR needs WinRAR's rar.exe (proprietary; 7-Zip can only read RAR). If rar.exe is found
and the input was .rar/.cbr, the output is .cbr; otherwise it's a .cbz.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image

from service import cache, config

log = logging.getLogger("mlt.archive")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
MAX_MEMBER_BYTES = 200 * 1024 * 1024      # a single page bigger than this is skipped (zip-bomb guard)
MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024  # stop reading an archive that inflates past this
SERVICE = f"http://127.0.0.1:{config.PORT}"
RAR_EXE_CANDIDATES = [r"C:\Program Files\WinRAR\Rar.exe", r"C:\Program Files (x86)\WinRAR\Rar.exe"]
SEVENZIP_CANDIDATES = [r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"]


# ---------------- archive reading ----------------

@dataclass
class Member:
    name: str          # sanitized relative posix path, used in the output archive
    data: bytes
    is_image: bool


def safe_name(raw: str) -> str | None:
    """Relative posix path with no drive, no leading slash and no '..' components, or None to skip it."""
    p = raw.replace("\\", "/")
    p = re.sub(r"^[A-Za-z]:", "", p).lstrip("/")
    parts = [x for x in p.split("/") if x not in ("", ".")]
    if not parts or any(x == ".." for x in parts):
        return None
    if parts[0] == "__MACOSX" or parts[-1].startswith("._") or parts[-1].lower() in ("thumbs.db", ".ds_store"):
        return None
    return "/".join(parts)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _find(cands, name):
    found = shutil.which(name)
    if found:
        return found
    return next((c for c in cands if os.path.exists(c)), None)


def open_archive(path: Path):
    """Detect by content, not extension: plenty of .cbr files are really zips and vice versa."""
    if zipfile.is_zipfile(path):
        return "zip", zipfile.ZipFile(path)
    import rarfile
    sevenzip = _find(SEVENZIP_CANDIDATES, "7z")
    if sevenzip:
        rarfile.SEVENZIP_TOOL = sevenzip
    if rarfile.is_rarfile(path):
        try:
            return "rar", rarfile.RarFile(path)
        except rarfile.RarCannotExec as e:
            sys.exit(f"Can't read RAR: no unrar/7-Zip found ({e}). Install 7-Zip.")
    sys.exit(f"{path} is neither a zip nor a rar archive.")


def _is_link(kind: str, info) -> bool:
    """Symlinks/hardlinks only make sense on extraction; we never extract, so drop them (with a log line)."""
    if kind == "rar":
        link = info.is_symlink() or getattr(info, "file_redir", None) is not None
    else:
        link = (info.external_attr >> 16) & 0o170000 == 0o120000  # S_IFLNK in the unix mode bits
    if link:
        log.info("skipping link %r", info.filename)
    return link


def read_members(path: Path) -> tuple[str, list[Member]]:
    kind, arc = open_archive(path)
    out, total, seen = [], 0, set()
    with arc:
        infos = [i for i in arc.infolist() if not i.is_dir() and not _is_link(kind, i)]
        for info in sorted(infos, key=lambda i: natural_key(i.filename)):
            name = safe_name(info.filename)
            if name is None:
                log.info("skipping %r", info.filename)
                continue
            if name in seen:
                log.warning("duplicate member %r, keeping the first", name)
                continue
            if info.file_size > MAX_MEMBER_BYTES:
                log.warning("skipping %s: %d bytes is over the per-file limit", name, info.file_size)
                continue
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                sys.exit(f"{path} inflates to more than {MAX_TOTAL_BYTES >> 30} GB; refusing.")
            with arc.open(info) as f:
                data = f.read(MAX_MEMBER_BYTES + 1)
            if len(data) > MAX_MEMBER_BYTES:  # header lied about the size
                log.warning("skipping %s: over the per-file limit", name)
                continue
            seen.add(name)
            out.append(Member(name, data, posixpath.splitext(name)[1].lower() in IMAGE_EXTS))
    return kind, out


# ---------------- translation backends ----------------

class ServiceBackend:
    """Use the already-running local service (shares its loaded models and cache)."""

    def __init__(self, token: str, no_cache: bool):
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"
        self.no_cache = no_cache

    @staticmethod
    def available(token: str) -> bool:
        try:
            return requests.get(f"{SERVICE}/v1/health", headers={"Authorization": f"Bearer {token}"},
                                timeout=3).ok
        except requests.RequestException:
            return False

    async def translate(self, data: bytes, prev_hash: str) -> tuple[bytes, bool]:
        headers = {"Content-Type": "application/octet-stream"}
        if prev_hash:
            headers["X-Prev-Hash"] = prev_hash
        if self.no_cache:
            headers["X-No-Cache"] = "1"
        r = await asyncio.to_thread(self.s.post, f"{SERVICE}/v1/translate", data=data, headers=headers, timeout=900)
        if r.status_code == 503:
            raise FatalError(r.json().get("error", "service unavailable"))
        if not r.ok:
            try:
                msg = r.json().get("error")
            except ValueError:
                msg = r.text[:200]
            raise RuntimeError(f"HTTP {r.status_code}: {msg}")
        return r.content, r.headers.get("X-Cache") == "hit"

    def close(self):
        pass


class LocalBackend:
    """Run the pipeline in this process (service not running)."""

    def __init__(self, no_cache: bool):
        from service.llm import LLM, LLMError
        try:
            self.llm = LLM()  # checks the LLM settings before the heavy pipeline import
        except LLMError as e:
            sys.exit(str(e))
        # Load the LLM before MIT's models take GPU memory (see service/server.py lifespan for why)
        try:
            self.llm.ensure_ready()
        except LLMError as e:
            print(f"LLM not ready yet ({e}); will retry on the first page", file=sys.stderr, flush=True)
        from service.pipeline import Pipeline
        self.pipe = Pipeline(self.llm)
        self.no_cache = no_cache

    async def translate(self, data: bytes, prev_hash: str) -> tuple[bytes, bool]:
        from service.llm import LLMError
        h = cache.image_hash(data)
        if not self.no_cache:
            png = cache.get_png(h)
            if png is not None:
                return png, True
        t0 = time.time()
        try:
            png, pairs, regions = await self.pipe.translate(data, h, prev_hash)
        except LLMError as e:
            raise FatalError(str(e)) from e
        cache.put(h, png, pairs, regions, time.time() - t0)
        return png, False

    def close(self):
        self.llm.unload_if_ours()


class FatalError(RuntimeError):
    """Stops the whole run (e.g. wrong model loaded, bad LLM settings) instead of failing every page."""


# ---------------- output ----------------

def encode(png: bytes, src_name: str, fmt: str, quality: int) -> tuple[bytes, str]:
    """Re-encode the translated PNG. Returns (bytes, output member name)."""
    stem, ext = posixpath.splitext(src_name)
    ext = ext.lower()
    target = {"keep": {".jpeg": "jpg", ".jpg": "jpg", ".webp": "webp"}.get(ext, "png"),
              "png": "png", "jpg": "jpg", "webp": "webp"}[fmt]
    if target == "png":
        return png, stem + ".png"
    img = Image.open(io.BytesIO(png))
    buf = io.BytesIO()
    if target == "jpg":
        img.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
        return buf.getvalue(), stem + (".jpeg" if ext == ".jpeg" else ".jpg")
    img.save(buf, "WEBP", quality=quality, method=4)
    return buf.getvalue(), stem + ".webp"


def unique(name: str, used: set[str]) -> str:
    stem, ext = posixpath.splitext(name)
    n, cand = 1, name
    while cand.lower() in used:
        n += 1
        cand = f"{stem}_{n}{ext}"
    used.add(cand.lower())
    return cand


def write_cbz(out: Path, entries: list[tuple[str, bytes, bool]]):
    tmp = out.with_name(out.name + ".part")
    with zipfile.ZipFile(tmp, "w") as z:
        for name, data, is_image in entries:
            # images are already compressed; deflate only the small text/xml members
            z.writestr(name, data, compress_type=zipfile.ZIP_STORED if is_image else zipfile.ZIP_DEFLATED)
    os.replace(tmp, out)


def write_cbr(out: Path, entries: list[tuple[str, bytes, bool]], rar_exe: str):
    with tempfile.TemporaryDirectory(prefix="mlt-cbr-") as d:
        for name, data, _ in entries:
            p = Path(d, *name.split("/"))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        tmp = out.with_name(out.name + ".part.rar")
        tmp.unlink(missing_ok=True)
        # a: add, -r: recurse, -ep1: paths relative to the folder, -m0: store (images are compressed), -idq: quiet
        subprocess.run([rar_exe, "a", "-r", "-ep1", "-m0", "-idq", str(tmp), d + os.sep + "*"], check=True)
        os.replace(tmp, out)


def default_output(src: Path, kind: str, rar_exe: str | None) -> Path:
    ext = ".cbr" if (kind == "rar" and rar_exe) else ".cbz"
    return src.with_name(f"{src.stem} (EN){ext}")


# ---------------- main ----------------

async def run(a) -> int:
    src = Path(a.archive)
    if not src.is_file():
        sys.exit(f"No such file: {src}")
    rar_exe = _find(RAR_EXE_CANDIDATES, "rar")
    kind, members = read_members(src)
    pages = [m for m in members if m.is_image]
    if not pages:
        sys.exit(f"No images found in {src}.")
    out = Path(a.output) if a.output else default_output(src, kind, rar_exe)
    if out.suffix.lower() == ".cbr" and not rar_exe:
        sys.exit("Writing .cbr needs WinRAR's rar.exe, which isn't installed. Use a .cbz output name instead.")
    if out.resolve() == src.resolve():
        sys.exit("Output would overwrite the input; pass a different -o.")
    if out.exists() and not a.force:
        sys.exit(f"{out} already exists (use --force to overwrite).")

    token = config.TOKEN_FILE.read_text().strip() if config.TOKEN_FILE.exists() else ""
    if not a.local and token and ServiceBackend.available(token):
        backend, how = ServiceBackend(token, a.no_cache), "running service"
    elif a.service:
        sys.exit(f"The local service isn't reachable on {SERVICE} (or the token in {config.TOKEN_FILE} is wrong).")
    else:
        backend, how = LocalBackend(a.no_cache), "in-process pipeline"
    print(f"{src.name}: {len(pages)} pages ({kind}), {len(members) - len(pages)} other files -> {out.name}  [{how}]",
          flush=True)

    # Pages are issued in order with up to --jobs in flight. The service (or in-process pipeline) overlaps one
    # page's detection/OCR with another's LLM call and still feeds each page the previous page's translations.
    sem = asyncio.Semaphore(max(1, a.jobs))
    results: dict[int, tuple] = {}
    failed, t_start, done, done_new = [], time.time(), 0, 0

    async def do_page(i: int, m: Member, prev_hash: str):
        nonlocal done, done_new
        t0 = time.time()
        try:
            png, hit = await backend.translate(m.data, prev_hash)
            data, name = await asyncio.to_thread(encode, png, m.name, a.format, a.quality)
            status = "cached" if hit else f"{time.time() - t0:.1f}s"
            done_new += 0 if hit else 1
        except FatalError:
            raise
        except Exception as e:  # keep going; this page stays untranslated
            failed.append((m.name, str(e)))
            data, name, status = m.data, m.name, f"FAILED ({e})"
        finally:
            sem.release()
        results[i] = (name, data)
        done += 1
        remaining = len(pages) - done
        eta = ""
        if done_new and remaining:
            eta = f", ~{(time.time() - t_start) / done * remaining / 60:.1f} min left"
        print(f"  [{done}/{len(pages)}] {m.name}: {status}{eta}", flush=True)

    tasks, prev_hash = [], ""
    try:
        for i, m in enumerate(members):
            if not m.is_image:
                continue
            await sem.acquire()
            for t in tasks:  # surface a fatal error (e.g. wrong LLM model) before issuing more pages
                if t.done() and t.exception():
                    raise t.exception()
            tasks.append(asyncio.create_task(do_page(i, m, prev_hash)))
            prev_hash = cache.image_hash(m.data)
            if isinstance(backend, ServiceBackend):
                await asyncio.sleep(0.3)  # keep arrival order at the service so it knows to wait for context
        await asyncio.gather(*tasks)
    except FatalError as e:
        for t in tasks:
            t.cancel()
        print(f"\nStopped: {e}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, asyncio.CancelledError):
        for t in tasks:
            t.cancel()
        print("\nInterrupted. Finished pages are cached; re-run to resume.", file=sys.stderr)
        return 130
    finally:
        await asyncio.to_thread(backend.close)

    entries, used = [], set()
    for i, m in enumerate(members):  # original order, non-image members in place
        if m.is_image:
            name, data = results[i]
            entries.append((unique(name, used), data, True))
        else:
            entries.append((unique(m.name, used), m.data, False))

    if out.suffix.lower() == ".cbr":
        write_cbr(out, entries, rar_exe)
    else:
        write_cbz(out, entries)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {(time.time() - t_start) / 60:.1f} min")
    if failed:
        print(f"{len(failed)} page(s) left untranslated:", file=sys.stderr)
        for name, err in failed:
            print(f"  {name}: {err}", file=sys.stderr)
        return 2
    return 0


def main():
    ap = argparse.ArgumentParser(description="Translate a .zip/.cbz/.rar/.cbr comic into English.")
    ap.add_argument("archive")
    ap.add_argument("-o", "--output", help='output file (default: "<name> (EN).cbz", or .cbr if rar.exe exists)')
    ap.add_argument("--format", choices=["keep", "png", "jpg", "webp"], default="keep",
                    help="page image format in the output (default: keep the source format; gif/bmp become png)")
    ap.add_argument("--quality", type=int, default=92, help="JPEG/WebP quality (default 92)")
    ap.add_argument("--no-cache", action="store_true", help="re-translate pages even if cached")
    ap.add_argument("-j", "--jobs", type=int, default=3,
                    help="pages in flight at once (default 3; 1 = strictly one page at a time)")
    ap.add_argument("--force", action="store_true", help="overwrite the output file if it exists")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--local", action="store_true", help="always run the pipeline in this process")
    g.add_argument("--service", action="store_true", help="require the running service")
    ap.add_argument("-v", "--verbose", action="store_true", help="show pipeline logs")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):  # Japanese file names on a cp1252 console
        stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if not a.verbose:
        logging.getLogger("mlt.archive").setLevel(logging.INFO)
        for noisy in ("manga_translator", "manga-translator"):
            logging.getLogger(noisy).setLevel(logging.ERROR)
    sys.exit(asyncio.run(run(a)))


if __name__ == "__main__":
    main()
