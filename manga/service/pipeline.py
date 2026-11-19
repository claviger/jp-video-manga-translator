"""Wraps zyddnys/manga-image-translator (MIT, vendored as a git submodule in third_party/mit) as a library.

We use its detection, OCR, mask refinement, inpainting and rendering, and replace its translation step with
our own (service/llm.py) by overriding MangaTranslator._dispatch_with_context. That method receives the page's
OCR'd texts already filtered and sorted into reading order and returns one translation per text, so the rest of
MIT's pipeline (punctuation fixes, rendering) runs unchanged.
"""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import contextvars
import heapq
import io
import itertools
import logging
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import cache, config
from .hlayout import is_source_horizontal, render_horizontal
from .placement import make_layer, place
from .llm import LLM, translate_page

log = logging.getLogger("mlt.pipeline")

if str(config.MIT_DIR) not in sys.path:
    sys.path.insert(0, str(config.MIT_DIR))


def _install_densecrf_fallback():
    """pydensecrf only ships as source and needs MSVC Build Tools on Windows. If it isn't importable, install a
    stub so MIT imports, and make refine_mask a pass-through (MIT dilates the mask right after anyway). Text-mask
    edges are a bit rougher without CRF; install VS Build Tools and
    `uv pip install git+https://github.com/zyddnys/pydensecrf.git@fix/python313-cython3` to get the real thing."""
    try:
        import pydensecrf.densecrf  # noqa: F401
        return False
    except ImportError:
        pass
    pkg = types.ModuleType("pydensecrf")
    pkg.__path__ = []
    pkg.densecrf = types.ModuleType("pydensecrf.densecrf")
    pkg.utils = types.ModuleType("pydensecrf.utils")
    pkg.utils.compute_unary = pkg.utils.unary_from_softmax = None
    sys.modules.update({"pydensecrf": pkg, "pydensecrf.densecrf": pkg.densecrf, "pydensecrf.utils": pkg.utils})
    return True


_CRF_STUBBED = _install_densecrf_fallback()

from manga_translator import Config, MangaTranslator  # noqa: E402
from manga_translator.config import (  # noqa: E402
    Detector, Inpainter, InpaintPrecision, Ocr, Renderer, Translator,
)

if _CRF_STUBBED:
    import cv2
    from manga_translator.mask_refinement import text_mask_utils

    def _refine_mask_passthrough(rgbimg, rawmask):
        if rawmask.ndim == 3:
            rawmask = rawmask[:, :, 0]
        return cv2.threshold(rawmask, 127, 255, cv2.THRESH_BINARY)[1]

    text_mask_utils.refine_mask = _refine_mask_passthrough
    log.warning("pydensecrf not installed: using pass-through mask refinement (see pipeline.py)")


def build_config() -> Config:
    cfg = Config()
    cfg.translator.translator = Translator.original  # placeholder; our override does the real translation
    cfg.translator.target_lang = "ENG"
    cfg.translator.enable_post_translation_check = False  # it would re-dispatch through MIT's own translators
    cfg.detector.detector = Detector.default
    cfg.detector.detection_size = 2048
    cfg.ocr.ocr = Ocr.ocr48px
    cfg.inpainter.inpainter = Inpainter.lama_large
    cfg.inpainter.inpainting_size = 2048
    cfg.inpainter.inpainting_precision = InpaintPrecision.bf16
    # manga2eng fits English into the bubble shape; "default" overflowed bubbles on the test page.
    cfg.render.renderer = Renderer.manga2Eng
    cfg.render.rtl = True
    cfg.render.disable_font_border = False
    return cfg


def _capture_manga2eng_layers(canvas, original, regions, font_path, line_spacing, disable_border):
    """Run MIT's manga2eng layout but capture each region's text as a Layer instead of pasting it.

    render_textblock_list_eng builds `img_pil = Image.fromarray(img)` and calls img_pil.paste(text, pos, mask=text)
    once per region. For the duration of the call we swap the module's `Image` for a shim whose fromarray returns a
    recorder for exactly that array (identity check); every other use of PIL in the module is untouched. Returns
    None if nothing was captured although there was text to render, so the caller can fall back."""
    from manga_translator.rendering import text_render, text_render_eng

    real_image = text_render_eng.Image
    layers = []

    class _Recorder:
        def __init__(self, pil):
            self._pil = pil

        def paste(self, im, box=None, mask=None):
            if box is not None and len(box) == 2:
                layer = make_layer(im, box[0], box[1])
                if layer is not None:
                    layers.append(layer)

        def __array__(self, dtype=None, copy=None):
            return np.asarray(self._pil, dtype=dtype)

        def __getattr__(self, name):
            return getattr(self._pil, name)

    class _ImageShim:
        def __getattr__(self, name):
            return getattr(real_image, name)

        def fromarray(self, arr, *a, **k):
            pil = real_image.fromarray(arr, *a, **k)
            return _Recorder(pil) if arr is canvas else pil

    text_render.set_font(font_path)
    text_render_eng.Image = _ImageShim()
    try:
        text_render_eng.render_textblock_list_eng(
            canvas, regions, line_spacing=line_spacing, size_tol=1.2, original_img=original,
            downscale_constraint=0.8, disable_font_border=disable_border)
    finally:
        text_render_eng.Image = real_image
    if not layers and any((r.translation or "").strip() for r in regions):
        return None
    return layers


@dataclass
class _PageJob:
    """Per-page state, carried in a ContextVar so concurrent pages don't share it."""
    own_hash: str
    prev_hash: str
    pairs: list = field(default_factory=list)


_job: contextvars.ContextVar[_PageJob] = contextvars.ContextVar("mlt_page_job")


class _PriorityLock:
    """Mutex where lower priority numbers go first (then FIFO). Pages coming back from the LLM (priority 0) get
    the GPU before new pages start detection (priority 1), so finished work doesn't pile up behind lookahead."""

    def __init__(self):
        self._locked = False
        self._waiters: list = []
        self._seq = itertools.count()

    async def acquire(self, priority: int):
        if not self._locked and not self._waiters:
            self._locked = True
            return
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (priority, next(self._seq), fut))
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()  # we were handed the lock as we got cancelled; pass it on
            else:
                self._waiters = [w for w in self._waiters if w[2] is not fut]
                heapq.heapify(self._waiters)
            raise

    def release(self):
        while self._waiters:
            _, _, fut = heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)  # ownership passes directly; _locked stays True
                return
        self._locked = False


class _Translator(MangaTranslator):
    """MangaTranslator with translation delegated to our LLM step, run so that pages overlap: a page holds the
    GPU for detection/OCR, releases it while its LLM call runs (another page's detection/OCR runs meanwhile),
    then takes it back with priority for inpainting/rendering. LLM calls stay in page order for context."""

    def __init__(self, llm: LLM, pipeline: "Pipeline", **kw):
        super().__init__(kw)
        self.llm = llm
        self.pipeline = pipeline

    async def _dispatch_with_context(self, config_: Config, texts: list[str], ctx):
        job = _job.get()
        pl = self.pipeline
        pl.gpu.release()
        try:
            # Off the GPU loop: waiting for the previous page and the LLM call happen in a worker thread, so the
            # hand-off from one page's LLM call to the next doesn't wait for whatever GPU phase is running.
            return await asyncio.get_running_loop().run_in_executor(pl.llm_pool, pl.llm_step, job, list(texts))
        finally:
            await pl.gpu.acquire(0)

    async def _run_text_rendering(self, config_: Config, ctx):
        """Render every region to its own layer, then fix off-page and overlapping text before compositing.

        Vertical source text: MIT's manga2eng layout (captured per region instead of pasted). Horizontal source
        text: hlayout.py. Then placement.place() shifts/shrinks boxes that leave the page or pile on each other."""
        regions = list(ctx.text_regions or [])
        if not regions:
            return ctx.img_inpainted
        horiz = [r for r in regions if abs(r.angle) <= 3 and is_source_horizontal(r)]
        vert = [r for r in regions if r not in horiz]
        layers = []
        if vert:
            captured = _capture_manga2eng_layers(ctx.img_inpainted, ctx.img_rgb, vert, self.font_path,
                                                 config_.render.line_spacing, config_.render.disable_font_border)
            if captured is None:  # capture didn't work (MIT internals changed?): plain manga2eng, no placement fixes
                log.warning("couldn't capture manga2eng layers; rendering without placement fixes")
                ctx.text_regions = vert
                try:
                    base = await super()._run_text_rendering(config_, ctx)
                finally:
                    ctx.text_regions = regions
                layers += render_horizontal(ctx.img_inpainted, horiz, self.font_path, regions)
                return place(base, [l for l in layers])
            layers += captured
        layers += render_horizontal(ctx.img_inpainted, horiz, self.font_path, regions)
        return place(ctx.img_inpainted, layers)


class Pipeline:
    """Pages run on a dedicated GPU thread/event loop (MIT's inference blocks whatever loop it runs on); LLM calls
    run in worker threads. translate() is safe to call concurrently from the caller's loop."""

    MAX_IN_FLIGHT = 3  # pages admitted at once; more just wait (bounds memory held by queued contexts)

    def __init__(self, llm: LLM):
        self.llm = llm
        self.cfg = build_config()
        self.mt = _Translator(
            llm,
            self,
            use_gpu=True,
            kernel_size=3,
            models_ttl=0,
            context_size=0,
            model_dir=str(config.MODEL_DIR),
            font_path=str(config.MIT_DIR / "fonts" / "anime_ace_3.ttf"),
        )
        self.gpu_loop = asyncio.new_event_loop()
        threading.Thread(target=self.gpu_loop.run_forever, name="mlt-gpu", daemon=True).start()
        self.gpu = _PriorityLock()                         # used only on gpu_loop
        self.admit = asyncio.Semaphore(self.MAX_IN_FLIGHT)  # used only on gpu_loop
        self.llm_pool = ThreadPoolExecutor(max_workers=self.MAX_IN_FLIGHT + 1, thread_name_prefix="mlt-llm")
        self._lock = threading.Lock()
        self.pending: dict[str, cf.Future] = {}  # own_hash -> future of this page's (ja, en) pairs
        self.recent: dict[str, list] = {}        # own_hash -> pairs of recently finished pages

    # ---- context hand-off between pages (thread-safe) ----

    def prev_pairs(self, prev_hash: str) -> list[tuple[str, str]]:
        """Previous page's pairs: block until it's translated if it's in flight, else recent memory, else cache."""
        if not prev_hash:
            return []
        with self._lock:
            fut = self.pending.get(prev_hash)
            recent = self.recent.get(prev_hash)
        if fut is not None:
            try:
                return fut.result(timeout=600)
            except cf.TimeoutError:
                log.warning("gave up waiting for previous page %s; translating without context", prev_hash[:12])
                return []
        return recent if recent is not None else cache.get_pairs(prev_hash)

    def resolve(self, own_hash: str, pairs):
        with self._lock:
            self.recent[own_hash] = list(pairs)
            while len(self.recent) > 64:
                self.recent.pop(next(iter(self.recent)))
            fut = self.pending.get(own_hash)
        if fut is not None and not fut.done():
            fut.set_result(list(pairs))

    def llm_step(self, job: "_PageJob", texts: list[str]) -> list[str]:
        prev = self.prev_pairs(job.prev_hash)
        out = translate_page(self.llm, texts, prev)
        job.pairs = [(ja, en) for ja, en in zip(texts, out) if en]
        self.resolve(job.own_hash, job.pairs)
        return out

    # ---- page processing ----

    async def _run_on_gpu_loop(self, img, job: "_PageJob"):
        _job.set(job)  # this coroutine runs as its own task on gpu_loop, so the var is per page
        async with self.admit:
            await self.gpu.acquire(1)
            try:
                return await self.mt.translate(img, self.cfg, skip_context_save=True)
            finally:
                self.gpu.release()

    async def translate(self, image_bytes: bytes, own_hash: str, prev_hash: str = ""):
        """Returns (png_bytes, pairs, regions); pairs are this page's (ja, en) for the next page's context.

        Safe to call concurrently: pages overlap (detection/OCR of one page runs during another's LLM call) and
        the previous page's translations are used as context even while that page is still in flight, provided
        it was submitted first."""
        img = Image.open(io.BytesIO(image_bytes), formats=["JPEG", "PNG", "WEBP", "GIF", "BMP"])
        img.load()  # first frame only for animated GIF/WebP
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        with self._lock:
            mine = own_hash not in self.pending
            if mine:
                self.pending[own_hash] = cf.Future()
        job = _PageJob(own_hash, prev_hash)
        try:
            ctx = await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self._run_on_gpu_loop(img, job),
                                                                             self.gpu_loop))
        finally:
            if mine:
                self.resolve(own_hash, job.pairs)  # no-op if already resolved; unblocks the next page on errors
                with self._lock:
                    self.pending.pop(own_hash, None)
        result = ctx.result if ctx.result is not None else img
        if isinstance(result, np.ndarray):
            result = Image.fromarray(result)
        buf = io.BytesIO()
        result.save(buf, format="PNG", optimize=False)
        regions = len(ctx.text_regions or [])
        return buf.getvalue(), job.pairs, regions
