# jp-video-manga-translator

Japanese-to-English translation and transcription pipeline for video and images (manga), running on your own hardware with a local LLM.

- **Video:** takes a Japanese video and gives you an MKV with English subtitles and the Japanese transcript as a second subtitle track. For adult content with mosaic censorship, it can optionally decensor the video first with [Jasna](https://github.com/Kruk2/jasna).
- **Manga:** a Chrome extension that puts a **Translate** button on manga pages and replaces each page with an English version typeset into the speech bubbles, plus a command-line tool for whole `.cbz`/`.zip`/`.cbr`/`.rar` archives.

Translation goes to any OpenAI-compatible LLM server you run yourself (TabbyAPI, LM Studio, llama.cpp, vLLM, Ollama, ...) or a hosted API. Nothing goes to a third-party service unless you point it at one. The one optional exception is the [existing subtitles](#existing-subtitles) lookup on OpenSubtitles, which only runs once you give it an API key.

> **Tested on:** Windows 11, git bash, an RTX 5090 (32 GB), NVIDIA driver 580+. Linux paths are written but untested. Other GPUs should work with the `cu126` PyTorch build but haven't been tried. Please open an issue if something breaks.

How it works:

0. Optional: look for existing subtitles on OpenSubtitles, sync them to the audio and check the sync. If they pass, they're muxed and steps 2-5 are skipped ([Existing subtitles](#existing-subtitles)).
1. Optional Jasna decensor.
2. Vocal separation (Mel-Band RoFormer), so speech is easier to hear over noise.
3. faster-whisper large-v3 on both the original and the vocals-only audio, each in three passes:
   1. A batched pass over the whole file.
   2. Re-decoding stretches where Whisper collapsed into one long junk line.
   3. Transcribing speech the first two passes missed, found with a VAD, on its own.

   Each audio version catches lines the other misses, so their results are merged.
4. Cleanup filters for Whisper's known hallucinations: YouTube outro phrases, counting runs, repetition loops and nonverbal (vocal sounds) lines, Korean lines, known filler characters, etc.
5. Smart LLM-based translation. A model reads the video title and entire transcript, writes a brief on the premise, the speakers and their genders, register and pronouns, recurring terms, words that keep getting misheard, etc. This is combined with a system prompt that tells it to expect ASR errors and infer likely meanings, translate each line on its own and never move words between lines, and to default to you/I/we rather than he/she when Japanese drops pronouns. This leads to much better video translation than naive Google Translate style word-by-word.
6. ffmpeg mux.

Timing on an RTX 5090, for a 2-hour video with the subtitles only: vocal separation ~3.5 min, transcription ~8 min per audio version, translation ~10 min with Artemis.

This pipeline will actually work on non-Japanese media (both video & comics), but not nearly as well -- while the translation LLM used knows basically every language,
none of the Whisper cleanup filters will work on other languages, and comic translation will have trouble with languages that aren't read right-to-left. It will probably
not outperform other, existing translation pipelines if used on media that's not in Japanese.

## Which parts do you need?

Everything is optional except Python tooling. Pick the row that matches what you want:

| You want | Install | You also need |
|---|---|---|
| English subtitles for videos | `--video` | an LLM server, ffmpeg |
| Japanese subtitles only (no translation) | `--video` | ffmpeg |
| Decensored + English-subtitled videos | `--video` | an LLM server, ffmpeg, Jasna |
| Manga translation in Chrome | `--manga` | an LLM server, Chrome |
| Manga archives (.cbz/.zip) | `--manga` | an LLM server (7-Zip for .cbr/.rar) |
| Everything | `--all` | all of the above |

- **No Jasna?** Just don't pass `--decensor`. Decensoring only happens when you ask for it.
- **No LLM yet?** `video.sh X.mp4 --asr` still writes the Japanese subtitles.
- **Already decensored elsewhere** (e.g. the Jasna GUI)? Pass the decensored file as the input, or the original plus `--restored decensored.mp4`.

## Install

### 1. Prerequisites

- **NVIDIA GPU and driver.** RTX 50-series cards need driver 580 or newer for the CUDA 13 PyTorch build. Older cards can use `--cuda cu126`. CPU-only works in principle but is very, very slow.
- **[Git](https://git-scm.com)** and **[uv](https://docs.astral.sh/uv/getting-started/installation/)**. uv installs Python 3.12 for you, so you don't need Python yourself.
- **ffmpeg** on PATH (video only). On Windows, `winget install Gyan.FFmpeg` works. On Linux, use your package manager.
- **7-Zip** (manga only, optional). It's needed to read `.cbr`/`.rar` archives: `winget install 7zip.7zip`. On Linux just use `.zip`/`.cbz` archives.
- About 15 GB of disk: two Python environments with CUDA PyTorch (~5 GB each) and ~5 GB of models. The translation model (see below) is not included in this; it is a dependency.

### 2. Clone and run the installer

```bash
git clone --recurse-submodules https://github.com/claviger/jp-video-manga-translator.git
cd jp-video-manga-translator
uv run install.py --all        # or --video, or --manga
```

The installer does the following, and is safe to re-run:

- **Video:** creates `.venv-video` with PyTorch, faster-whisper and audio-separator, and downloads the Whisper large-v3 model (~3 GB) and the vocal-separation model (~900 MB).
- **Manga:** checks out the [manga-image-translator](https://github.com/zyddnys/manga-image-translator) submodule, creates `.venv-manga`, and downloads its detection, OCR and inpainting models (~700 MB).
- **Config:** copies `.env.example` to `.env`, then lists whatever you still need to do.

Options:
- `--cuda cu130|cu126|cpu` picks the PyTorch build. The default picks one from your NVIDIA driver.
- `--no-models` skips the downloads; the models then download on first use.

The two environments use Python 3.12 on purpose:
- On 3.13, a manga dependency (`editdistance`) has no Windows wheel.
- On 3.14, much of the torch stack has no Windows wheels.
- `pydensecrf` is skipped because it needs the MSVC Build Tools to compile. The manga pipeline falls back to plain mask refinement without it.

### 3. Translation model

This project doesn't install, download or run the translation LLM. You bring an OpenAI-compatible server and tell `.env` where it is.

**Recommended model: [TheDrummer/Artemis-31B-v1.2](https://huggingface.co/TheDrummer/Artemis-31B-v1.2)**, a Gemma 4 31B fine-tune designed for adult creative writing. In side-by-side tests on video subtitles it beat Gemma4-31B, Qwen3.8-27B, and even dedicated Japanese translation models like Sugoi-32B. It was best at repairing speech-recognition mistakes and translating explicit language without softening it. For non-adult content any Gemma4-based model will work for translation and can be run on Bedrock, Runpod, etc. Smaller (9b) Gemma4 models run locally will also work though they will not translate as well as a 31B or large model. 

Ways to run it:

- **LM Studio, llama.cpp or Ollama:** use a GGUF quant, e.g. [bartowski/TheDrummer_Artemis-31B-v1.2-GGUF](https://huggingface.co/bartowski/TheDrummer_Artemis-31B-v1.2-GGUF). Q5_K_M is a good fit for 24-32 GB of VRAM.
- **TabbyAPI:** use an EXL3 quant of the same model (5-6 bpw for 32 GB VRAM.) Search Hugging Face for "Artemis-31B-v1.2 exl3".
- **vLLM or a hosted API:** any instruct model that handles Japanese; Gemma4 models work best. Point `LLM_BASE_URL` at it.

Things worth knowing:

- **VRAM:** a 31B model at 5-6 bits takes ~20-24 GB, and the vision/speech models want a few GB more. If you are offloading the translation LLM to the cloud, 8 GB RAM is adequate for the other local models used.
  - **Video:** the pipeline runs Jasna, Whisper and the LLM one after another, so on one 32 GB card that's fine. With `LLM_BACKEND=lmstudio` it also unloads the LLM during the GPU-heavy steps.
  - **Manga:** the service keeps its models loaded while the LLM is too. Artemis 6bpw at 16k context plus the manga models used ~31.5 of 32.6 GB, so a 5bpw quant leaves more headroom.
  - **Smaller GPU:** run the LLM on another machine and point `LLM_BASE_URL` there.
- **Thinking models** (Qwen3, DeepSeek and others): set `LLM_REASONING_EFFORT=none` if your server supports it. In our tests, thinking added about 2,300 tokens per 30-line batch on Qwen3 and made DeepSeek about 50x slower per batch, for at most a small quality gain.
- **Avoid role-play-tuned models:** despite being theoretically good at dialog, models with a built-in role-play system prompt (e.g. "Orthia" variants) get confused by bare subtitle lines.

### 4. Configure `.env`

Open `.env` (created by the installer from [.env.example](.env.example), which documents every setting). The ones that matter:

```ini
# The LLM server
LLM_BASE_URL=http://localhost:5000/v1   # up to and including /v1
LLM_MODEL=Artemis-31B-v1.2              # exactly as listed at <LLM_BASE_URL>/models
LLM_API_KEY=                            # empty if your server has no auth; ${MY_TOKEN} reads an env var
LLM_BACKEND=openai                      # openai | tabby | lmstudio (see below)

# Video decensoring (optional)
JASNA_EXE=C:\Tools\jasna\jasna.exe
JASNA_ARGS=--restoration-model-name ltx-4step
```

`LLM_BACKEND` decides whether the tools manage the model for you:

- `openai` (default): only sends requests. You load the model yourself. Works with every server.
- `tabby`: for a TabbyAPI server that others may share.
  - If `LLM_MODEL` is loaded, it's used.
  - If nothing is loaded, the tools load it and unload it again when they're done. This needs an admin key.
  - If a different model is loaded, the run stops instead of swapping it out. `LLM_USE_LOADED_MODEL=1` uses whatever is loaded instead.
- `lmstudio`: LM Studio on the same machine. The tools load `LLM_MODEL` fully onto the GPU with the `lms` CLI, which avoids LM Studio's default of putting part of a big model on the CPU (10x slower).

Real environment variables override `.env`. Set `LJT_ENV_FILE` to use a different file.

### 5. Jasna (only for decensoring)

[Jasna](https://github.com/Kruk2/jasna) is a separate program (AGPL-3.0) that restores mosaic-censored video. This project only calls its command-line interface.

1. **Download.** Get the latest release from [github.com/Kruk2/jasna/releases](https://github.com/Kruk2/jasna/releases).
   - On Windows with NVIDIA, that's `jasna-windows-<version>.7z.001` and `.002`. There are also Windows AMD and Linux NVIDIA/AMD packages.
   - Download all parts of the archive into one folder and open the `.001` file with 7-Zip to extract it.
2. **Point the tools at it.** Set `JASNA_EXE` in `.env` to the extracted `jasna.exe`.
3. **Check the options.** Run `jasna.exe --help` to confirm it works and to see every option.
4. **Download the restoration model.** Run Jasna once with the model you want, in its GUI or with `--restoration-model-name`, so it can download the model.
   - The LTX models are large, and Jasna offers to download them when they're missing.
   - LTX models are "supporter" models that need a Jasna supporter license. Check Jasna's README for which ones, and for its terms. This said, they are **much** better, like NCIS "enhance!" made real.
   - If you can't use LTX, set `JASNA_ARGS=--restoration-model-name basicvsrpp` for the fast, free BasicVSR++ model.

Choosing settings:

- **Default `JASNA_ARGS`:** `--restoration-model-name ltx-4step` matches the GUI's "LTX, Good quality".
  - On an RTX 5090, a 2-hour video with little mosaic took about 75 minutes.
  - A 3.9-hour, heavily mosaicked video was estimated at 11+ hours.
- **`--ltx-auto-split 30`:** for long videos with heavy mosaic. Mosaics at least 30% of the frame height get LTX, and smaller ones get BasicVSR++. That same 3.9-hour video took 1.4 hours.
- **Only decensor censored originals.**: Jasna looks for mosaics, and an earlier bad restoration isn't one. Running Jasna on a video that was already decensored badly by another tool or model will not fix it.
- **Auto-split output:** it keeps the source codec, and restored parts are re-encoded at a high bitrate. An H.264 source can therefore come out larger than it went in, while full LTX runs write HEVC and usually shrink the file. Re-encode afterwards if size matters.

### 6. Chrome extension (manga only)

1. Start the service and leave it running: `manga/run_service.sh`. It listens only on `127.0.0.1:8765`. On first start it writes a random access token to `manga/service/token.txt`.
2. In Chrome, open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked** and pick the `manga/extension` folder.
3. Open the extension's **Options**, paste the contents of `manga/service/token.txt`, and click **Save**. It should say "Service reachable".
4. On a manga site, click the extension's toolbar icon, then **Enable on this site**. Chrome asks for access to that site and the sites its images come from.
5. A **Translate** button appears on large images. Click it to translate the page, and click again to switch between the original and English. **Translate all pages in order** in the popup does a whole chapter, giving each page the previous one as context.

Optional: set `MANGA_EXTENSION_ID` in `.env` to the ID shown on `chrome://extensions`, so the service only accepts requests from your copy of the extension.

## Use

### Video

The video tool does a set of independent steps. Name any combination of them; with none, `--subs` is assumed.

| Step | What it does |
|---|---|
| `--decensor` | Jasna mosaic restoration (needs `JASNA_EXE`) |
| `--opensubs` | existing subtitles from OpenSubtitles, synced to the audio (needs `OPENSUBTITLES_API_KEY`; see below) |
| `--asr` | Japanese subtitles by speech recognition |
| `--translate` | English subtitles by LLM translation of the best Japanese source: this run's `--asr`, an earlier transcript of the file, Japanese subtitles from OpenSubtitles, or a Japanese text track already in the video |
| `--convert` | write the output as `--format mkv` (default) or `mp4`. `--reencode hevc\|h264\|av1` also re-encodes the video (NVENC if your ffmpeg has it, else the CPU), with `--cq` for quality. Either option implies `--convert`. |
| `--subs` | smart subtitles: OpenSubtitles if it has subtitles that sync well, otherwise `--asr` + `--translate` |

```bash
./video.sh movie.mkv                                # smart subtitles
./video.sh ABC-123.mp4 --decensor --subs            # decensor, then smart subtitles
./video.sh ABC-123.mp4 --decensor                   # decensor only -> ABC-123.decensored.mp4
./video.sh ABC-123.mp4 --asr --translate            # always transcribe and translate
./video.sh anime.mkv --translate                    # translate the Japanese track already in the file
./video.sh old.avi --reencode hevc                  # convert to an HEVC MKV, nothing else
./video.sh ABC-123.mp4 --decensor --asr --translate --reencode hevc --format mp4   # all of it at once
./video.sh --help                                   # every option, with examples
```

Inputs and outputs:

```bash
./video.sh "E:/incoming"                            # every video in the folder
./video.sh "E:/incoming" -r --ext mkv,mp4           # include subfolders; only .mkv and .mp4
./video.sh "E:/incoming" -r --out-dir "F:/done"     # write to F:/done, mirroring the folder structure
./video.sh "E:/incoming" -r --out-dir "F:/done" --flatten   # everything directly in F:/done
```

- **Inputs:** you can mix files and folders. Folders are scanned one level deep unless you pass `-r`, and files this tool wrote itself (`.subbed.mkv`, `.restored.mp4`, ...) are skipped.
- **Outputs:** written next to each input unless `--out-dir` is given.
  - With `--flatten`, two inputs that would produce the same output name are caught before anything starts.
  - A file whose output already exists is skipped, so you can re-run a batch after stopping it. `--overwrite` redoes them.
- **Batches:** one failed file doesn't stop the batch. A summary at the end lists what was done, skipped and failed, and the exit code is 1 if anything failed.
- **Windows without git bash:** use `video.cmd` the same way.

The output is `ABC-123.decensored.subbed.mkv`, or without `.decensored` / `.subbed` when those steps didn't happen. Video and audio are copied untouched unless you `--reencode`. Subtitle tracks, in order:
- **English (OpenSubtitles)**, if it was used. It's the default track.
- **English (machine translation)**. It's the default when there's no OpenSubtitles track.
- **Japanese (ASR)**, the cleaned transcript.
- A track in the audio's own language from OpenSubtitles, if one was found.

**Intermediate files** are written next to the output and reused if they're already there:
- `ABC-123.restored.mp4`: Jasna's output.
- `ABC-123.vocals.flac`: the separated vocals.
- `ABC-123.ja.srt`: the raw Japanese transcript.
- `ABC-123.en.srt`: the English subtitles.
- `ABC-123.dossier.txt`: the context brief the LLM writes before translating (setting, speakers, tone, words it seems to mishear). You can edit it.
- `ABC-123.en.opensubtitles.srt`: the synced OpenSubtitles download.

They're deleted once a file succeeds, leaving only the final video. When a file fails, they're kept, so running the same command again picks up where it stopped. `--keep-intermediates` always keeps them, and `--no-mux` makes the subtitle files themselves the output instead of a video.

### Existing subtitles

For movies, TV and anime, someone has often already written good subtitles. With `OPENSUBTITLES_API_KEY` set in `.env`, the video tool checks [OpenSubtitles.com](https://www.opensubtitles.com) first. It only falls back to speech recognition and the LLM when nothing usable turns up.

1. **Match.** It searches by the file's OpenSubtitles moviehash first, which only matches subtitles uploaded for this exact release. Then it searches by the title, year, season and episode that [guessit](https://github.com/guessit-io/guessit) reads from the filename, or by `--imdb tt1234567` if you pass one. Results for a different title or episode are thrown out. The rest are ranked by hash match, frame rate, release-name overlap with your file, download count and uploader trust. Filenames that look like a JAV product code (`ABC-123`) are only searched by hash, since guessit reads them as TV episodes.
2. **Sync.** The best candidate is downloaded and synced to the audio with [ffsubsync](https://github.com/smacke/ffsubsync), which fixes constant offsets and frame-rate mismatches.
3. **Check.** Silero VAD, a different speech detector from the one ffsubsync uses, finds speech in the audio. The sync passes when enough of the subtitle time falls on speech (`OPENSUBTITLES_MIN_OVERLAP`, default 40%) and that's clearly better than the same subtitles shifted by a few seconds (`OPENSUBTITLES_MIN_MARGIN`, default 8 points). The margin is what catches subtitles for the wrong film or cut. The numbers for each candidate are printed, before and after syncing.
4. **Fall back.** If the check fails, the next candidate is tried (up to `OPENSUBTITLES_MAX_TRIES`, default 3), then speech recognition and translation as usual.

How I calibrated the check, on The Lost Boys (1987) with its own embedded English SDH track as the known-good subtitles:

| Test | Speech overlap | Margin | Result |
|---|---|---|---|
| Correct subtitles | 60% | +19 | pass |
| Shifted 4.2 s, then synced | 60% | +19 | pass (ffsubsync found -4.14 s) |
| Stretched 25/23.976 fps, then synced | 60% | +19 | pass (found x0.959) |
| Another film's subtitles | 34% | +1 | rejected |

ffsubsync's own score didn't separate these (the wrong film scored highest), which is why the separate check exists. On a 2-hour film the whole step, minus downloads, takes under a minute.

If the audio stream is tagged with a language other than the one you asked for (e.g. `jpn` on an anime), subtitles in that language are fetched and synced too, as a second track. An earlier `.ja.srt` transcript of the file is added as **Japanese (ASR)**.

The lookup runs with `--subs` (the default) or `--opensubs`. `--asr --translate` skips it. `--subs-lang fr` (or `SUBS_LANG`) picks the language of the default track, and `--no-original-subs` skips the second track. The translation fallback always writes English.

Quotas: OpenSubtitles allows 5 downloads per IP per day with only an API key, and 20 per day with a free account (`OPENSUBTITLES_USERNAME` / `OPENSUBTITLES_PASSWORD`). Each candidate tried counts; searches don't. Counters reset at midnight UTC. The login token is cached in `.cache/` for its 24-hour lifetime, because OpenSubtitles limits logins to 30 an hour.

Get the API key from your OpenSubtitles profile, under **API consumers**. OpenSubtitles asks that every application use its own key.

Subtitle tracks already in your video are dropped from the output, since you're subtitling it because they weren't good enough. Pass `--keep-subs` to keep them after the new tracks, not as the default.

Limits: ffsubsync applies one offset to the whole file, so subtitles for a cut with scenes added or removed in the middle usually fail the check and fall back to transcription.

### Manga

- **In the browser:** see [Chrome extension](#6-chrome-extension-manga-only) above.
- **Archives:**
  ```bash
  manga/translate.sh "Some Book.cbz"                  # writes "Some Book (EN).cbz"
  manga/translate.sh "Some Book.cbr" --format jpg     # .cbr/.rar need 7-Zip
  manga/translate.sh page1.png page2.png              # single pages -> page1.en.png ...
  ```

The archive tool uses the running service if there is one, so the models aren't loaded twice. Without it, the tool runs the pipeline itself. Results are cached by image hash, so re-running after an interruption skips finished pages.

[manga/README.md](manga/README.md) covers how it works, the security model, and known limitations, such as canvas-based readers.

## Security notes

- **The manga service listens only on 127.0.0.1 and requires a random token.** Binding to localhost alone isn't enough, because any web page can make your browser send requests to `127.0.0.1`. So the service also:
  - sends no CORS headers;
  - rejects unexpected `Host` headers, which blocks DNS rebinding;
  - only accepts requests from Chrome extensions.
- **The extension asks for access one site at a time.** Its only install-time host permission is the local service.
- **Secrets:** `.env` is git-ignored, and keys can stay in your shell profile with `LLM_API_KEY=${MY_TOKEN}`.
- **What OpenSubtitles sees** (only when `OPENSUBTITLES_API_KEY` is set): your API key, your login if you set one, your IP, each file's moviehash, and the title, year, season and episode guessed from its filename (not the filename itself). That's enough for them to know what's in your library. Use `--asr --translate` instead of `--subs` for files you'd rather not look up. The cached login token in `.cache/opensubtitles-token.json` is plaintext, like the manga token.
- **The token is stored in plaintext**, in `manga/service/token.txt` and in the extension's storage. Any process running as you can read it.

## Repository layout

```
install.py        installer (stdlib only; run with `uv run install.py`)
.env.example      every setting, documented
video.sh / .cmd   video launcher
video/            pipeline.py (the whole video pipeline), requirements.txt
manga/            service/ (local FastAPI service), extension/ (MV3 Chrome extension), scripts/ (archive and
                  single-image tools, model prefetch), tests/, third_party/mit (manga-image-translator, submodule)
models/           downloaded speech/separation models (git-ignored; manga models go in manga/models)
```

## License

GPL-3.0 (see [LICENSE](LICENSE)), because the manga part imports [manga-image-translator](https://github.com/zyddnys/manga-image-translator), which is GPL-3.0. Jasna is a separate program under its own license (AGPL-3.0, with some proprietary supporter models) and isn't included here. Model weights you download may have their own licenses.
