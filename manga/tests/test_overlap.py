"""Scheduling test for Pipeline's page overlap, without the GPU or the LLM.

MIT's translate() is replaced by a stub with the same shape: blocking "GPU" work, then the translation hook
(_dispatch_with_context, the real one), then more blocking GPU work. translate_page is stubbed with a sleep.
Checks: every page gets the previous page's pairs as context (even while it's in flight), LLM calls happen in
page order, and overlapping beats running pages one at a time.

    .venv/Scripts/python -m tests.test_overlap
"""
import asyncio
import hashlib
import time
import types

import service.pipeline as P

GPU1, LLM, GPU2 = 0.75, 1.5, 0.75
llm_log = []


def fake_translate_page(llm, texts, prev):
    llm_log.append((texts[0], [en for _, en in prev]))
    time.sleep(LLM)
    return [f"EN({t})" for t in texts]


async def fake_mt_translate(self, img, cfg, skip_context_save=True):
    time.sleep(GPU1)  # detection + OCR (blocking, like real GPU calls)
    ctx = types.SimpleNamespace(text_regions=[1], result=None)
    await self._dispatch_with_context(cfg, [img.info["page"]], ctx)
    time.sleep(GPU2)  # inpainting + rendering
    ctx.result = img
    return ctx


def page_bytes(n):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (n, n, n)).save(buf, "PNG")
    return buf.getvalue()


async def run(n_pages, concurrency):
    llm_log.clear()
    P.translate_page = fake_translate_page
    P._Translator.translate = fake_mt_translate
    pipe = P.Pipeline(None)
    orig_open = P.Image.open

    def tagged_open(fp, formats=None):  # attach the page number so the stub can "OCR" it
        im = orig_open(fp, formats=formats)
        im.info["page"] = f"p{im.getpixel((0, 0))[0]}"
        return im

    P.Image.open = tagged_open
    pages = [page_bytes(i) for i in range(n_pages)]
    hashes = [hashlib.sha256(b).hexdigest() for b in pages]
    sem = asyncio.Semaphore(concurrency)

    async def one(i):
        try:
            return await pipe.translate(pages[i], hashes[i], hashes[i - 1] if i else "")
        finally:
            sem.release()

    t0 = time.time()
    tasks = []
    for i in range(n_pages):
        await sem.acquire()
        tasks.append(asyncio.create_task(one(i)))
    results = await asyncio.gather(*tasks)
    P.Image.open = orig_open
    return time.time() - t0, results


def main():
    n = 6
    seq_t, _ = asyncio.run(run(n, 1))
    seq_log = list(llm_log)
    par_t, results = asyncio.run(run(n, 3))
    par_log = list(llm_log)
    expect = [(f"p{i}", [f"EN(p{i - 1})"] if i else []) for i in range(n)]
    assert seq_log == expect, seq_log
    assert par_log == expect, f"context/order broken with overlap: {par_log}"
    assert [r[1] for r in results] == [[(f"p{i}", f"EN(p{i})")] for i in range(n)]
    print(f"one at a time: {seq_t:.1f}s ({seq_t / n:.2f}s/page)   overlapped x3: {par_t:.1f}s ({par_t / n:.2f}s/page)")
    print(f"ideal overlapped ~ GPU1 + n*max(LLM, GPU1+GPU2) + GPU2 = {GPU1 + n * max(LLM, GPU1 + GPU2) + GPU2:.1f}s")
    assert par_t < seq_t * 0.75, "overlap gave no real speedup"
    print("ok: context chained in order, LLM calls in page order, overlap faster")


if __name__ == "__main__":
    main()
