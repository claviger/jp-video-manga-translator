"""Local translation service for the browser extension.

Security model (any web page can make the browser send requests to 127.0.0.1, so loopback alone isn't enough):
  * binds 127.0.0.1 only, never 0.0.0.0;
  * every request needs `Authorization: Bearer <token>`; the token lives in service/token.txt (generated on first
    run) and is only given to the extension's background worker, never to content scripts or pages;
  * no CORS headers at all, so a page can't read responses, and because the Authorization header makes the
    request non-simple, a page's preflight fails before the real request is sent;
  * Host header must be 127.0.0.1/localhost:<port> (blocks DNS-rebinding);
  * requests carrying an Origin other than chrome-extension:// are rejected (set MANGA_EXTENSION_ID to pin it
    to one extension id);
  * uploads are size- and pixel-capped and only common raster formats are decoded.
"""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import re
import secrets
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image

from . import cache, config
from .llm import LLM, LLMError

log = logging.getLogger("mlt.server")

Image.MAX_IMAGE_PIXELS = config.MAX_PIXELS
EXTENSION_ID = config.EXTENSION_ID


def load_or_create_token() -> str:
    p = config.TOKEN_FILE
    if p.exists():
        tok = p.read_text().strip()
        if len(tok) >= 32:
            return tok
    tok = secrets.token_urlsafe(32)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(tok + "\n")
    log.info("Generated a new service token in %s", p)
    return tok


TOKEN = load_or_create_token()
ALLOWED_HOSTS = {f"127.0.0.1:{config.PORT}", f"localhost:{config.PORT}"}

state: dict = {}


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    from .pipeline import Pipeline  # heavy import (torch, MIT)

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    llm = LLM()
    state["llm"] = llm
    # Get the LLM ready (loaded, with the tabby/lmstudio backends) before MIT's models take GPU memory: on
    # Windows, TabbyAPI failed to finish loading a model ("WinError 1450: Insufficient system resources") while
    # MIT's detector/OCR/inpainter were resident, and loaded fine before them. Each page checks again anyway.
    try:
        await asyncio.to_thread(llm.ensure_ready)
    except LLMError as e:
        log.warning("LLM not ready yet (%s); will retry on the first page", e)
    state["pipe"] = Pipeline(llm)
    log.info("Service ready on http://%s:%d (token file: %s)", config.HOST, config.PORT, config.TOKEN_FILE)
    try:
        yield
    finally:
        await asyncio.to_thread(llm.unload_if_ours)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def guard(request: Request, call_next):
    if request.headers.get("host", "") not in ALLOWED_HOSTS:
        return JSONResponse({"error": "bad host"}, status_code=421)
    origin = request.headers.get("origin")
    if origin is not None:
        ok = origin == f"chrome-extension://{EXTENSION_ID}" if EXTENSION_ID else origin.startswith("chrome-extension://")
        if not ok:
            return JSONResponse({"error": "origin not allowed"}, status_code=403)
    auth = request.headers.get("authorization", "")
    if not (auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].strip().encode(), TOKEN.encode())):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/v1/health")
async def health():
    llm: LLM = state["llm"]
    loaded, err = await asyncio.to_thread(llm.status)
    return {"ok": True, "llm_backend": llm.backend, "llm_model": llm.model, "llm_loaded": loaded, "llm_error": err,
            "loaded_by_us": llm.loaded_by_us}


@app.post("/v1/translate")
async def translate(request: Request):
    cl = request.headers.get("content-length")
    if cl is not None and int(cl) > config.MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "image too large"}, status_code=413)
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > config.MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "image too large"}, status_code=413)
    if not body:
        return JSONResponse({"error": "empty body"}, status_code=400)
    data = bytes(body)
    h = cache.image_hash(data)
    headers = {"X-Image-Hash": h, "Cache-Control": "no-store"}

    cached = cache.get_png(h) if request.headers.get("x-no-cache") != "1" else None
    if cached is not None:
        headers["X-Cache"] = "hit"
        return Response(cached, media_type="image/png", headers=headers)

    prev_hash = request.headers.get("x-prev-hash", "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", prev_hash):
        prev_hash = ""
    t0 = time.time()
    try:
        png, pairs, regions = await state["pipe"].translate(data, h, prev_hash)
    except LLMError as e:
        log.error("%s", e)
        return JSONResponse({"error": str(e)}, status_code=503, headers=headers)
    except (Image.DecompressionBombError, Image.UnidentifiedImageError, OSError) as e:
        return JSONResponse({"error": f"unsupported or invalid image: {e}"}, status_code=415, headers=headers)
    dt = time.time() - t0
    log.info("translated %s: %d regions, %d pairs, prev %s, %.1fs", h[:12], regions, len(pairs), prev_hash[:12] or "-", dt)
    cache.put(h, png, pairs, regions, dt)
    headers.update({"X-Cache": "miss", "X-Regions": str(regions)})
    return Response(png, media_type="image/png", headers=headers)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        config.check_llm()
    except config.ConfigError as e:
        raise SystemExit(str(e))
    log.info("Settings: %s; LLM %s (%s) at %s, key %s", config.ENV_FILE or "no .env file", config.LLM_MODEL,
             config.LLM_BACKEND, config.LLM_BASE_URL, "set" if config.LLM_API_KEY else "not set")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info", access_log=False,
                server_header=False, proxy_headers=False)


if __name__ == "__main__":
    main()
