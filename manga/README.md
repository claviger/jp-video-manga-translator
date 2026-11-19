# Manga translator

A Sugoi-Manga-Translator-style Chrome extension and archive tool that run entirely on your own hardware. Click a button on a manga page and it gets replaced with an English version, typeset into the bubbles.

- **Detection, OCR, inpainting, typesetting:** [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator) (GPL-3.0), used as a library from a git submodule in `third_party/mit`.
- **Translation:** our own step against any OpenAI-compatible LLM server (TabbyAPI, LM Studio, llama.cpp, vLLM, ...), shared with the video subtitle pipeline. It uses the same adult-genre framing, the "you/I, not he/she" pronoun rule, one JSON batch per page, tolerant JSON parsing and per-id retry.
- **Extension:** MV3, with no static content scripts and host access granted per site.

## Setup

Installation, the translation model and `.env` settings are covered in the [main README](../README.md): `uv run install.py --manga` from the repo root.

Notes on how the installer sets this part up:
- **Python 3.12 only.** On 3.13, `editdistance==0.8.1` has no Windows wheel, and 3.14 has none for half the torch stack.
- **`pydensecrf` is skipped.** It only ships as source and needs MSVC Build Tools. Without it, `service/pipeline.py` swaps CRF mask refinement for a pass-through. If you have VS Build Tools, you can add it back with `uv pip install --python ../.venv-manga "pydensecrf @ git+https://github.com/zyddnys/pydensecrf.git@fix/python313-cython3"`.
- **Upstream's extra index line is dropped.** `rusty-manga-image-translator==0.12.1` is on PyPI, and the extra index doesn't have that version.
- **Models** live in `models/` (`MANGA_MODEL_DIR`). With `--no-models`, the first run downloads them.
- **A standalone `.venv` works too.** The wrapper scripts use `<manga folder>/.venv` if it exists, otherwise `../.venv-manga`.

## Running

```bash
./run_service.sh
```

- It listens on `127.0.0.1:8765`.
- On first start it writes a random token to `service/token.txt`.
- Settings come from `.env` (in this folder or the repo root, or `$LJT_ENV_FILE`) plus the environment, which wins. See the root `.env.example`. `${NAME}` in `.env` expands from the environment, so a key can stay in your shell profile.
- At startup it gets the LLM ready (and loaded, with the `tabby`/`lmstudio` backends) before its own models go on the GPU. On Windows, TabbyAPI failed to finish loading a model while these models were resident, with "WinError 1450: Insufficient system resources", and loaded fine before them.

Then load the extension:

1. `chrome://extensions` → Developer mode → Load unpacked → pick `extension/`.
2. Open the extension's Options, paste the contents of `service/token.txt`, and Save. It should say "Service reachable".
3. On a manga page, click the toolbar icon, then **Enable on this site**. Chrome asks for access to the page's origin and to the origins its images come from.
4. A **Translate** button appears on each large image. Click it to translate, then click again to toggle between Original and English. Shift-click re-translates and skips the cache. **Translate all pages in order** in the popup does the whole chapter in sequence, so each page gets the previous one as context.

### Whole archives (.zip / .cbz / .rar / .cbr)

```bash
path/to/manga/translate.sh "Some Book.cbr"
path/to/manga/translate.sh book.cbz -o "book (EN).cbz" --format jpg
```

`translate.sh` works from any directory (relative paths resolve against where you are), reads the same `.env` as the service, and picks the archive or single-image tool from the first file's extension. Put it on your PATH with an alias if you like: `alias mtl=path/to/manga/translate.sh`.

- **Order and context:** every image is translated in natural sort order (page2 before page10), and each page gets the previous page as context. The output is written next to the input as `<name> (EN).cbz`.
- **Non-image files:** `ComicInfo.xml` and the like are copied through unchanged.
- **Failed pages:** a page that fails stays untranslated and is listed at the end (exit code 2). If Tabby has the wrong model loaded, the run stops instead of failing every page.
- **Format detection:** archives are recognized by their content, not the extension, so a `.cbr` that's really a zip works.
- **Reading RAR:** needs 7-Zip (`C:\Program Files\7-Zip\7z.exe`, through the `rarfile` package).
- **Writing RAR:** needs WinRAR's `rar.exe`, which is proprietary, and 7-Zip can't create RAR. So the output is `.cbz` unless `rar.exe` is installed and the input was RAR.
- **Backend:** if the service is running, pages go through it, so the GPU models aren't loaded twice and nothing runs out of VRAM. Otherwise the pipeline runs in-process. `--local` and `--service` force one or the other.
- **Resume:** the cache is shared, so re-running after Ctrl-C or a crash skips the pages that are already done.
- `-j/--jobs N` pages in flight (default 3; see "Page overlap" below). `--format keep|png|jpg|webp` (default `keep`: jpg stays jpg, gif/bmp become png), `--quality 92`, `--no-cache`, `--force`, `-v` for pipeline logs.
- **Safety:** members are read into memory and never extracted to disk. Entries with `..`, absolute paths, symlinks and hardlinks, `__MACOSX` and `._*` junk are skipped. Pages over 200 MB, or archives that inflate past 8 GB, are refused.

### Single images

```bash
path/to/manga/translate.sh page1.png page2.png
```

That writes `page1.en.png` and so on, treating the pages as consecutive. Like the archive tool, it uses the running service if there is one and falls back to in-process otherwise (`--local`, `--service`, `--no-cache`, `-v`). `scripts/make_test_page.py` draws synthetic test pages.

## How it works

1. The content script finds `<img>` elements that are at least 300×400 natural and 180 px on screen. It only sends the image URL to the background worker.
2. The background worker fetches the image itself with `credentials: "include"`, so the site's cookies apply. It sets `Referer` to the page URL using a session-scoped declarativeNetRequest rule, because many CDNs check it. Then it POSTs the bytes to the service with the bearer token.
3. The service hashes the image (SHA-256) and returns the cached PNG if it already has one. Otherwise it runs MIT: detection (DBNet), 48px OCR, filtering and right-to-left reading-order sort, then **our** translation, then LaMa-large inpainting, then the `manga2eng` renderer.
4. Context: the extension remembers each page's hash and sends the previous page's hash in `X-Prev-Hash`. The service then puts that page's (ja, en) pairs into the prompt as context only.
5. The cache lives in `cache/<sha256>.v<N>.png` plus `.json`, which holds the pairs. Bump `PIPELINE_VERSION` in `service/config.py` when the output would change.

### Page overlap (bulk translation)

The archive tool and the extension's "Translate all" keep up to 3 pages in flight. Inside the pipeline, MIT runs on a dedicated GPU thread and each page holds the GPU for detection/OCR, releases it during its LLM call (another page's detection/OCR runs meanwhile), then takes it back with priority for inpainting/rendering. LLM calls stay in page order: a page waits for the previous page's translations even while that page is still in flight, so context is the same as one-at-a-time. Clients issue requests in page order, 0.3 s apart, so the service sees them in order. `tests/test_overlap.py` checks the ordering and the speedup with stubbed GPU/LLM stages (18.1 s -> 10.5 s for 6 pages at 0.75 + 1.5 + 0.75 s, which is the theoretical best).

### LLM backends (`LLM_BACKEND`)

- **openai** (default): calls `/chat/completions` and never loads or unloads anything. If `/models` lists models and `LLM_MODEL` isn't one of them, it stops with a clear error.
- **tabby** (TabbyAPI may be shared): if `LLM_MODEL` is loaded, it's used. If **another** model is loaded, the service returns 503 with an explanation and never swaps it out; `LLM_USE_LOADED_MODEL=1` translates with whatever is loaded. If **nothing** is loaded, it loads `LLM_MODEL` with `TABBY_LOAD_CTX` (16384) and `TABBY_CACHE_MODE` (8,8), and unloads it on clean shutdown only if it loaded it itself and it's still the loaded model. A hard kill skips the unload.
- **lmstudio**: if `LLM_MODEL` isn't loaded, runs `lms load <model> --gpu max -c 16384`. On shutdown it unloads only a model it loaded; other loaded models are never touched.
- `LLM_REASONING_EFFORT` is sent as `reasoning_effort` only when set (`none` turns off thinking on servers that support it).

With a 6bpw 24B-class model at 16k ctx and the MIT models both resident, VRAM sat at about 31.5 of 32.6 GB on a 5090, so there's not much headroom. If it gets tight: a smaller context, a smaller quant, or `inpainting_size` 1536 in `pipeline.py`.

Timing on the 5090:
- First page after start: about 20 s, mostly model loading.
- A warm page: about 7-8 s, of which about 2 s is the LLM.
- A cache hit: about 0.1 s.

## Security model

Any web page can make the browser send requests to `127.0.0.1`, so binding to loopback isn't enough on its own. Here's what's in place:

- **Service:** binds 127.0.0.1 only, and the bind address isn't configurable.
- **Service:** requires the bearer token, compared in constant time.
- **Service:** sends no CORS headers. The `Authorization` header makes a page's request non-simple, so its preflight fails before the real request goes out.
- **Service:** rejects any `Host` other than `127.0.0.1:8765` or `localhost:8765`, which blocks DNS rebinding.
- **Service:** rejects any `Origin` that isn't `chrome-extension://`. Set `MANGA_EXTENSION_ID` to pin a single extension id.
- **Service:** caps uploads at 40 MB and 40 MP, and decodes only JPEG, PNG, WebP, GIF and BMP.
- **Service:** has `/docs` and OpenAPI turned off.
- **Extension:** only the background worker has the token. Content scripts never see it, and there's no `externally_connectable`, so pages can't message the worker.
- **Extension:** buttons live in a closed shadow root, and the click handler ignores events with `isTrusted === false`. A page can't drive translations with synthetic clicks.
- **Extension:** host access is opt-in per site through `optional_host_permissions`. The only install-time host permission is `http://127.0.0.1:8765/*`.
- **Extension:** popup and options actions such as registering a site are only accepted from extension pages, not from content scripts.

Residual risks I know about:
- The token sits in plaintext in `service/token.txt` and in the extension's `chrome.storage.local`. Any local process running as me can read both, and that's within my threat model for this tool.
- The translated image goes into the page as a `data:` URL, so the page can read it. It's a translation of an image the page already had, so I think that's fine.
- `third_party/mit/result/log_*.txt` holds MIT's run logs, which include the OCR'd text. They stay local but accumulate.

## Known limitations

- **Text placement pass.** Both renderers produce one layer per region (manga2eng's paste is captured rather than painted), and `service/placement.py` then shrinks anything bigger than the page, shifts boxes back inside the page, and pushes overlapping boxes apart (shrinking crowded ones a little if there's no room). Pages without problems are untouched. Moved text can end up partly over a bubble outline or art; the white stroke keeps it readable. If the manga2eng capture ever stops working (upstream internals changed), it falls back to plain manga2eng and logs a warning. `make_test_page.py --variant 4` draws edge-bleeding bubbles and an overlapping cluster; `render_debug.py --raw` shows the page without the fixes.
- **Partial Japanese at page edges.** In bubbles cut off by the page edge, detection sometimes misses a partial column, so that Japanese isn't inpainted and shows behind the English.

- **Horizontal Japanese uses a different typesetter.** `manga2eng` is built for vertical text in bubbles, so `service/hlayout.py` handles regions whose detected Japanese lines are wide and flat. It fits the English into the original footprint (font as large as fits, widen into free space up to 1.5x, then grow taller only into free space) and gives stacked rows of the same original size one shared English size. Rotated regions still go to MIT. `scripts/render_debug.py` renders a page with canned translations (no LLM) for checking layout, and `make_test_page.py --variant 3` draws horizontal cases.

- **Canvas and tiled readers don't work.** Readers that draw pages on `<canvas>` (often scrambled tiles, as on many official JP/EN storefronts) expose no image URL. Reading the canvas back is blocked when it's cross-origin tainted, and the tiles are shuffled anyway. The content script detects large canvases and says so. Supporting them would need screenshot capture (`chrome.tabs.captureVisibleTab`) and per-viewport translation, which is a different design.
- `blob:` images are read in the content script and sent as bytes. That works for blob URLs the page created, but I haven't tested it on a real site.
- Sites whose own scripts keep resetting `img.src` (some lazy loaders) can undo the swap.
- A page CSP with a strict `img-src` that excludes `data:` would block the swapped image. I haven't hit this yet.
- Cookie and Referer handling for extension-initiated fetches is set up as described above but not tested against real CDNs. If a site's images fail with 403, check the service worker console.
- **OCR misses:** the 48px OCR dropped 「、」 on the test page, so the comma glyphs survive inpainting as stray ticks. Panel detection fails on simple layouts and falls back to simple sorting, which logs a warning but gives the right order there.
- MIT prints an "experimental Python version, switch to the Rust rewrite" banner. The Python version is what's maintained in the repo as of 2026-11. The Rust rewrite is worth checking later.

## Layout

```
service/       config.py, cache.py (shared page cache), hlayout.py (horizontal-text typesetting), placement.py (off-page/overlap fixes), llm.py (LLM client + prompt), pipeline.py (MIT wrapper), server.py (FastAPI)
extension/     manifest.json, background.js, content.js, popup.*, options.*
scripts/       translate_archive.py (cbz/cbr), translate_file.py (single images), prefetch_models.py (model download),
               make_test_page.py (synthetic pages), render_debug.py (layout check with canned translations)
tests/         harness.html: runs content.js with a stubbed chrome.runtime (serve repo root, open /tests/harness.html)
               test_overlap.py: page-overlap scheduling test (no GPU/LLM needed)
third_party/   mit (submodule, pinned)
```

License: GPL-3.0 if distributed, since it imports manga-image-translator. It's fine for personal use either way.
