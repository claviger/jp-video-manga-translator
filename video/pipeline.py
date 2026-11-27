#!/usr/bin/env python3
"""
Video processing for a whole collection. Independent steps; combine any of them in one command:
  --decensor   Jasna mosaic restoration
  --opensubs   existing subtitles from OpenSubtitles, synced to the file's audio (see opensubs.py)
  --asr        Japanese subtitles by speech recognition (faster-whisper with vocal separation)
  --translate  English subtitles by LLM translation of the best Japanese source available
  --convert    output container (--format mkv|mp4) and optional re-encode (--reencode hevc|h264|av1)
  --subs       smart subtitles: OpenSubtitles if it syncs well, else --asr + --translate (default with no step)
Inputs can be files or folders (-r recurses, --ext filters). Outputs go next to each input or to --out-dir
(mirrored structure, or --flatten). Run `video.sh --help` for examples.

Intermediates (decensored video, vocals, raw transcript <name>.ja.srt, context brief, synced subtitles) are
written next to the output and reused if present, deleted when a file succeeds (--keep-intermediates keeps them)
and kept when it fails, so re-running resumes. The raw ja.srt gets the cleanup filters on every load.
A trailing ".restored" in the input name is ignored when naming outputs (X.restored.mp4 -> X.ja.srt).

Settings come from the repo's .env file (see .env.example): LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, LLM_BACKEND,
JASNA_EXE, JASNA_ARGS and a few optional ones. Real environment variables override the file.
Needs ffmpeg on PATH; install.py sets up everything else.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import requests

# Git Bash/mintty isn't detected as a console, so Python falls back to cp1252 and chokes on Japanese
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def load_env_file():
    """Read KEY=VALUE lines from the first .env found ($LJT_ENV_FILE, then the repo root) into os.environ.
    Real environment variables win over the file. No dependency: '#' comments, optional quotes, blank lines,
    and ${NAME} expands from the environment (so a key can stay in a shell profile: LLM_API_KEY=${MY_TOKEN})."""
    for c in (os.environ.get("LJT_ENV_FILE"), REPO_ROOT / ".env"):
        if not c or not Path(c).is_file():
            continue
        for line in Path(c).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip().removeprefix("export ").strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            value = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), value)
            os.environ.setdefault(key, value)
        return Path(c)
    return None


ENV_FILE = load_env_file()

# ---------------- config (from .env; see .env.example) ----------------
JASNA_EXE = os.environ.get("JASNA_EXE", "")
# Default matches the Jasna GUI's "LTX, Good quality". Jasna 0.12 CLI names: ltx-basic = Basic, ltx = Standard,
# ltx-4step = Good, ltx-8step = High, ltx-undistilled = Top; basicvsrpp is the fast BasicVSR++ model.
# For long, heavily mosaicked videos add "--ltx-auto-split 30": big mosaics get LTX, small ones BasicVSR++
# (a 3.9 h film went from an 11 h estimate to 1.4 h on an RTX 5090).
JASNA_EXTRA_ARGS = shlex.split(os.environ.get("JASNA_ARGS", "--restoration-model-name ltx-4step"))

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")  # faster-whisper name or a local CTranslate2 folder
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16")  # float16 is needed on RTX 50-series (Blackwell)

# Translation: any OpenAI-compatible /chat/completions endpoint. LLM_BACKEND decides model management:
#   openai   - (default) only send requests; you load the model yourself. Works with anything.
#   tabby    - TabbyAPI, treated as shared: if LLM_MODEL is loaded it's used; if nothing is loaded it's loaded
#              (needs an admin key) and unloaded afterwards; if another model is loaded the run stops rather than
#              swap it out (LLM_USE_LOADED_MODEL=1 translates with whatever is loaded instead).
#   lmstudio - LM Studio on this machine: loads LLM_MODEL fully on the GPU through the `lms` CLI.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_BACKEND = os.environ.get("LLM_BACKEND", "openai").strip().lower()
USE_LOADED_MODEL = os.environ.get("LLM_USE_LOADED_MODEL", "") == "1"
TABBY_LOAD_CTX = int(os.environ.get("TABBY_LOAD_CTX", "16384"))
TABBY_CACHE_MODE = os.environ.get("TABBY_CACHE_MODE", "8,8")
# Sent as reasoning_effort when set. "none" turns thinking off on servers that honour it (LM Studio and several
# hosted APIs; Qwen3.x went from ~2300 reasoning tokens per batch to 0). Leave unset for servers that reject it.
REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "") or None
# Existing subtitles first (video/opensubs.py): with OPENSUBTITLES_API_KEY set, OpenSubtitles is searched for
# subtitles in SUBS_LANG before any speech recognition. The ASR + LLM path is the fallback, and it always writes
# English, whatever SUBS_LANG is.
SUBS_LANG = os.environ.get("SUBS_LANG", "en").strip().lower() or "en"
REMOTE_RETRIES = 4  # attempts per request on network errors, 429 and 5xx

# Context brief ("dossier"), adapted from NihonSub: before translating, the LLM reads the title and a sample of
# the transcript and writes a short brief on premise, speakers and tone, which goes into every batch's prompt.
DOSSIER_SAMPLE_LINES = 200
DOSSIER_PROMPT = (
    "You are preparing a brief for a translator who will write English subtitles for a Japanese adult video. "
    "You get the video title and an evenly spaced sample of its transcript. The transcript comes from speech "
    "recognition, so it has errors. Write a concise brief, under 250 words, plain text, covering:\n"
    "1. Premise and setting, in 1-2 sentences.\n"
    "2. Speakers: how many, their genders, any names or nicknames, and who usually speaks to whom. "
    "This decides whether a line is translated with 'you' or with 'he'/'she'.\n"
    "3. Register and tone of each speaker (polite, casual, teasing, dominant...), with the first-person pronouns "
    "and sentence endings they use.\n"
    "4. Recurring names, terms and running jokes, with their intended meaning.\n"
    "5. Words the speech recognition seems to mishear repeatedly, with the probable intended word. Only list a "
    "mishearing when the intended word is clear from several occurrences; otherwise leave it out, since a wrong "
    "correction here would mislead every line of the translation.\n"
    "State only what the sample supports; say so when something is uncertain."
)
# With LLM_BACKEND=lmstudio the pipeline loads the model itself with full GPU offload. LM Studio's JIT
# auto-offload put a 31B Q5_K_M model mostly on CPU (98k context): 87s per 10-line batch vs 7s on the GPU.
LMS_CONTEXT = int(os.environ.get("LMSTUDIO_LOAD_CTX", "16384"))  # plenty for a 30-line batch plus context
BATCH_SIZE = 30      # lines per translation request
CONTEXT_LINES = 6    # previous lines sent as context only

# Counting hallucination: Whisper emits "16個", "17個", ... at exactly 1.000s, back to back.
COUNT_RUN_MIN = 4          # min consecutive cues to call it a hallucinated run
COUNT_RUN_DUR_TOL = 0.02   # seconds of slack around the 1.000s duration

# No VAD: Silero VAD (even at threshold 0.25) heard only ~16 of 151 min of speech in a 2.5 h test film and silently
# dropped most of the dialogue. Whisper runs on the full audio; the text filters below remove what it invents.
# Batched decoding over <=30s chunks cut at quiet points (no audio dropped). On 15 min of test-film samples:
#   sequential, temperature fallback + hallucination_silence_threshold: 244s
#   sequential, temperature=0: 120s (41s without hallucination_silence_threshold)
#   batched (batch 8), temperature=0: 18s, similar coverage  -> ~3 min for a 2.5 h film
# kotoba-whisper-v2.0-faster segfaults with word_timestamps (its CTranslate2 conversion), and without them it
# merges lines into 15s+ blocks, so it is not usable here.
# Vocal separation (python-audio-separator, in the same venv; on Python 3.14 its dependency diffq-fixed has no
# Windows wheel). Transcribing the separated vocals as well and adding their lines where the original audio has
# none raised coverage of VAD-confirmed speech on two test stretches from 53% -> 65% and 49% -> 60%. Neither
# audio version alone is better: each finds lines the other misses.
SEPARATOR_EXE = Path(sys.executable).parent / ("audio-separator.exe" if os.name == "nt" else "audio-separator")
SEPARATOR_MODEL = "vocals_mel_band_roformer.ckpt"   # top vocals SDR (12.6) in audio-separator's catalog
SEPARATOR_MODEL_DIR = Path(os.environ.get("SEPARATOR_MODEL_DIR", REPO_ROOT / "models" / "separator"))
# Separate in 10-minute pieces. Whole-file separation of a 3.9 h film ran 62 GB of RAM out,
# and the pieces are joined without a crossfade, which Whisper doesn't notice.
SEPARATOR_CHUNK_SEC = 600

WHISPER_BATCH_SIZE = 8
WORD_GAP_SPLIT = 1.5          # split a segment into separate cues where its words are this far apart
GAPFILL_VAD_THRESHOLD = 0.5   # pass 3: Silero regions with no subtitle get transcribed alone
GAPFILL_PAD = 1.0             # seconds of context around each pass-3 region
COLLAPSE_MIN_DUR = 5.0       # a junk segment this long marks a collapsed window worth re-decoding
REDECODE_CHUNK_MAX = 10.0     # pass-2 chunk length for collapsed stretches
REDECODE_CHUNK_SEARCH = 3.0
# In 10s pieces Whisper's stock-phrase hallucinations are short, so the duration rule can't catch them
REDECODE_STOCK_PHRASES = ["おやすみなさい", "お疲れ様", "ありがとうございました", "ご視聴"]
WHISPER_CHUNK_MAX = 30.0     # seconds; Whisper's window
WHISPER_CHUNK_SEARCH = 8.0   # look for the quietest point in the last 8s of each chunk
WHISPER_PARAMS = dict(
    language="ja",
    task="transcribe",              # transcribe JA; translation is done by the LLM
    beam_size=5,
    temperature=0.0,                # no fallback re-decodes (the main cost on noisy/music stretches)
    vad_filter=False,
    without_timestamps=False,       # batched mode defaults to one segment per 30s chunk
    word_timestamps=True,           # cue times from the words, not 30s-window timestamp tokens
    condition_on_previous_text=False,     # reduces runaway repetition loops
)
# Drop segments whose mean word probability is below this. On test-film samples real lines scored 0.3-0.99;
# below 0.2 were near-silent guesses ("すごい" 0.01, "大体" 0.15).
MIN_WORD_PROB = 0.2
# Silero VAD recall on 113 confirmed test-film dialogue lines: th=0.5 39%, 0.25 53%, 0.1 59%, 0.02 70%.
# Measured with a recall harness; any replacement VAD should catch ~100% before being enabled.

# Classic Whisper hallucinations on silence/music in Japanese (YouTube outro phrases)
HALLUCINATIONS = ["ご視聴ありがとうございました", "チャンネル登録", "最後までご視聴",
                  "字幕は視聴者によって作成",  # fan-sub credit Whisper learned from YouTube
                  # vlog-outro lines Whisper invents on the separated vocals track
                  "一緒に歌ってみるのが楽しかった", "お腹いっぱいになったので",
                  # news-caption line it invents over intros ("about the earthquake when it struck")
                  "地震発生時"]
# Stock phrases Whisper stretches over a whole silent 30s window; real ones are short
LONG_STOCK_PHRASES = ["お疲れ様でした", "おやすみなさい"]
LONG_STOCK_MIN_DUR = 5.0  # with word timestamps a fake one lasts ~8s; a real one is said in ~1s
IDENTICAL_RUN_MAX = 3   # a 4th+ consecutive identical line is a loop: keep only the first
# Lines made only of moans/breaths are not dialogue
FILLER_CHARS = set("あぁんっアァンッはハふフうぅ〜ーぁ")
FILLER_KEEP = {"うん", "ううん", "うんうん"}  # actual words ("yeah", "no")
# On-screen time cap: 2s + 0.3s per character, at most 10s (stretched cues stayed up for minutes)
MAX_CUE_BASE, MAX_CUE_PER_CHAR, MAX_CUE_DUR = 2.0, 0.3, 10.0

SYSTEM_PROMPT = (
    "You translate Japanese video dialogue into natural, idiomatic English subtitles. "
    # Genre framing, as in Sugoi's own system prompt: picks the sexual sense of ambiguous words
    # (first A/B: 先走ってる -> "getting ahead of yourself", ちょっとしませんか -> "shall we stop for a bit?").
    "This is dialogue from a Japanese adult video. When a word, idiom or phrase has both an everyday and a sexual "
    "meaning, prefer the sexual meaning if the context allows it (e.g. イく = to cum, 出す = to cum/ejaculate, "
    "先走り = precum, 濡れる = to get wet). Translate explicit language explicitly, without softening or censoring "
    "it, but don't add sexual content to lines that are clearly ordinary conversation. "
    "Keep each line short enough to read on screen and preserve tone and register. "
    "The Japanese comes from speech recognition and may contain errors; infer the most likely meaning. "
    "Translate each numbered line on its own: a sentence may be split across lines, so keep each part with "
    "its own id and never move words from one id to another. "
    "Most lines are spoken directly to the other person on screen: when Japanese omits the subject or object, "
    "use \"you\"/\"I\"/\"we\", not \"he\"/\"him\"/\"she\", unless the line clearly refers to someone else. "
    'Return ONLY a JSON array like [{"id": 12, "en": "..."}], one object per input id, nothing else.'
)


def run(cmd):
    cmd = [str(c) for c in cmd]
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def free_vram_for_jasna():
    """Before the GPU-heavy steps (Jasna, separation, Whisper), get the LLM out of VRAM where that's ours to do.

    lmstudio: unload LM Studio's models (the pipeline reloads its model for translation).
    tabby: the server is treated as shared, so a loaded model is left alone; this only warns.
    openai: nothing is known about where the model runs, so nothing is done."""
    if LLM_BACKEND == "lmstudio":
        lms = shutil.which("lms")
        if lms:
            subprocess.run([lms, "unload", "--all"], check=False)
    elif LLM_BACKEND == "tabby" and LLM_BASE_URL:
        loaded = tabby_loaded_model(LLM_BASE_URL, os.environ.get("LLM_API_KEY", ""))
        if loaded:
            print(f"Note: TabbyAPI has {loaded} loaded and holding VRAM. Not unloading it, since Tabby is treated "
                  "as shared; if it's on this GPU, Jasna, vocal separation and Whisper may run short of memory.",
                  flush=True)


def decensor(src: Path, dst: Path):
    if not JASNA_EXE or not Path(JASNA_EXE).is_file():
        sys.exit("Decensoring needs Jasna: set JASNA_EXE in .env to jasna.exe (see README, 'Jasna'), or leave out "
                 "--decensor.")
    run([JASNA_EXE, "--input", src, "--output", dst, *JASNA_EXTRA_ARGS])
    if not dst.exists():
        sys.exit(f"Jasna finished but {dst} was not found. Check Jasna's log and output naming.")


# ---------------- SRT helpers ----------------
def ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def parse_ts(s: str) -> float:
    h, m, rest = s.strip().split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


def write_srt(cues, path: Path, key: str):
    with path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(cues, 1):
            f.write(f"{i}\n{ts(c['start'])} --> {ts(c['end'])}\n{c[key]}\n\n")


def read_srt(path: Path):
    cues = []
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        if len(lines) < 3 or " --> " not in lines[1]:
            continue
        start, end = lines[1].split(" --> ")
        cues.append({"start": parse_ts(start), "end": parse_ts(end), "ja": "\n".join(lines[2:])})
    return cues


# ---------------- transcript cleanup (runs on fresh and reused transcripts) ----------------
_NUM = re.compile(r"\d+")


def _count_key(text: str):
    """'17個' -> ('{}個', 17). None unless the line has exactly one number."""
    nums = _NUM.findall(text)
    if len(nums) != 1:
        return None
    return _NUM.sub("{}", text, count=1).strip(), int(nums[0])


def drop_counting_runs(cues):
    """Remove runs of incrementing-number lines that are each exactly ~1s long and back to back.

    Only the fixed-cadence part of a counting run is removed. Irregularly spaced counts may be real speech."""
    n, drop, i = len(cues), set(), 0
    while i < n:
        j = i
        while j + 1 < n:
            a, b = cues[j], cues[j + 1]
            ka, kb = _count_key(a["ja"]), _count_key(b["ja"])
            if not (ka and kb and ka[0] == kb[0] and kb[1] == ka[1] + 1):
                break
            if abs(b["start"] - a["end"]) > COUNT_RUN_DUR_TOL:
                break
            if any(abs((c["end"] - c["start"]) - 1.0) > COUNT_RUN_DUR_TOL for c in (a, b)):
                break
            j += 1
        if j - i + 1 >= COUNT_RUN_MIN:
            drop.update(range(i, j + 1))
            print(f"Dropping counting hallucination {ts(cues[i]['start'])}-{ts(cues[j]['end'])}: "
                  f"{cues[i]['ja']} .. {cues[j]['ja']} ({j - i + 1} cues)", flush=True)
        i = j + 1
    return [c for k, c in enumerate(cues) if k not in drop]


_JA = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]")
_HANGUL = re.compile(r"[가-힯ᄀ-ᇿ㄰-㆏]")
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z' ]*[A-Za-z]")
_PUNCT = re.compile(r"[\s、。,.!?！？…・~\-「」『』()（）♪♡❤]+")


def normalize_text(text: str) -> str:
    """Remove English words Whisper mixes into Japanese ("痛いかな switching to the music") and stray dashes."""
    if _JA.search(text):
        text = _LATIN_RUN.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip(" -")


def drop_reason(c) -> str:
    text = c["ja"]
    if not text:
        return "empty"
    if not _JA.search(text):
        return "no Japanese"         # "Defense", "Kick", "me"
    if _HANGUL.search(text):
        return "Korean"
    if any(h in text for h in HALLUCINATIONS):
        return "outro phrase"
    if c["end"] - c["start"] >= LONG_STOCK_MIN_DUR and any(p in text for p in LONG_STOCK_PHRASES):
        return "stretched stock phrase"
    bare = _PUNCT.sub("", text)
    if bare not in FILLER_KEEP and set(bare) <= FILLER_CHARS:
        return "moan only"
    if len(bare) >= 3 and len(set(bare)) == 1:
        return "moan only"           # ええええええ, んんんん
    return ""


def collapse_identical_runs(cues):
    out, run = [], 0
    for c in cues:
        if out and c["ja"] == out[-1]["ja"] and c["start"] - out[-1]["start"] < 2.0:
            out[-1]["end"] = max(out[-1]["end"], c["end"])  # same line emitted twice: merge
            continue
        run = run + 1 if out and c["ja"] == out[-1]["ja"] else 1
        if run <= IDENTICAL_RUN_MAX:
            out.append(c)
        elif run == IDENTICAL_RUN_MAX + 1:
            # loop detected: keep just the first of the run
            del out[len(out) - IDENTICAL_RUN_MAX + 1:]
    return out


def cap_durations(cues):
    for k, c in enumerate(cues):
        limit = min(MAX_CUE_DUR, MAX_CUE_BASE + MAX_CUE_PER_CHAR * len(c["ja"]))
        end = min(c["end"], c["start"] + limit)
        if k + 1 < len(cues):
            end = min(end, max(c["start"] + 0.5, cues[k + 1]["start"]))  # no overlap with the next line
        c["end"] = end
    return cues


def clean_cues(cues, verbose: bool = False):
    from collections import Counter

    reasons = Counter()
    kept = []
    for c in cues:
        c = {**c, "ja": normalize_text(c["ja"])}
        why = drop_reason(c)
        if why:
            reasons[why] += 1
            if verbose:
                print(f"  drop [{why}] {ts(c['start'])} {c['ja']}", flush=True)
        else:
            kept.append(c)
    n = len(kept)
    kept = collapse_identical_runs(kept)
    reasons["repeat loop"] += n - len(kept)
    n = len(kept)
    kept = drop_counting_runs(kept)  # needs the original 1.000s durations, so before cap_durations
    reasons["counting run"] += n - len(kept)
    if +reasons:
        print("Cleanup removed: " + ", ".join(f"{k} {v}" for k, v in (+reasons).most_common()), flush=True)
    return cap_durations(kept)


def coverage_report(cues, duration: float, bucket: int = 600):
    """Cues per 10 min. Empty stretches mean dialogue was missed (or the scene really is silent)."""
    print("Coverage (cues per 10 min):", flush=True)
    for b in range(0, int(duration) + 1, bucket):
        n = sum(1 for c in cues if b <= c["start"] < b + bucket)
        flag = "   <-- check" if n < 5 else ""
        print(f"  {ts(b)[:5]}  {n:4}  {'#' * min(n, 60)}{flag}", flush=True)


def media_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


# ---------------- transcription ----------------
def segments_to_cues(segments, offset: float = 0.0, verbose: bool = True, dropped: list | None = None):
    """Whisper segments -> cues. Only filters that need Whisper's scores live here; text filters are in clean_cues.
    Segments removed here are appended to `dropped` (if given) so collapsed stretches can be re-decoded."""
    cues = []
    for s in segments:
        text = s.text.strip()
        if not text:
            continue
        words = s.words or []
        wprob = sum(w.probability for w in words) / len(words) if words else 0.0
        cue = {"start": s.start + offset, "end": s.end + offset, "ja": text, "wprob": wprob}
        # avg_logprob/no_speech_prob/compression_ratio are per 30s decode window, shared by all its segments
        why = ("compression" if s.compression_ratio > 2.4 else          # repetitive garbage
               "no_speech" if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0 else
               "low word prob" if wprob < MIN_WORD_PROB else "")
        if why:
            if dropped is not None:
                dropped.append({**cue, "why": why})
            continue
        for part in split_at_word_gaps(cue, words, offset):
            cues.append(part)
            if verbose:
                print(f"[{ts(part['start'])}] {part['ja']}", flush=True)
    return cues


def split_at_word_gaps(cue, words, offset: float):
    """One Whisper segment can span a long pause ("あ、" ... 5s ... "もうダメです"); the cue then starts at the
    breath and the duration cap ends it before the line is said. Split wherever words are > WORD_GAP_SPLIT apart."""
    if len(words) < 2:
        return [cue]
    groups, cur = [], [words[0]]
    for w in words[1:]:
        if w.start - cur[-1].end > WORD_GAP_SPLIT:
            groups.append(cur)
            cur = []
        cur.append(w)
    groups.append(cur)
    if len(groups) == 1:
        return [cue]
    out = []
    for g in groups:
        text = "".join(w.word for w in g).strip()
        if text:
            out.append({"start": g[0].start + offset, "end": g[-1].end + offset, "ja": text,
                        "wprob": sum(w.probability for w in g) / len(g)})
    return out


def quiet_chunks(audio, sr: int = 16000, max_len: float = WHISPER_CHUNK_MAX, search: float = WHISPER_CHUNK_SEARCH):
    """Split audio into <=max_len chunks, cutting at the quietest 50ms frame near each boundary. Nothing is dropped."""
    import numpy as np

    frame, n = int(0.05 * sr), len(audio)
    energy = np.sqrt(np.add.reduceat(audio.astype(np.float64) ** 2, np.arange(0, n, frame)) / frame)
    chunks, s = [], 0
    while n - s > max_len * sr:
        lo = (s + int((max_len - search) * sr)) // frame
        hi = (s + int(max_len * sr)) // frame
        cut = (lo + int(np.argmin(energy[lo:hi]))) * frame
        chunks.append({"start": s / sr, "end": cut / sr})
        s = cut
    chunks.append({"start": s / sr, "end": n / sr})
    return chunks


def collapsed_spans(cues, dropped, min_dur: float = COLLAPSE_MIN_DUR, pad: float = 0.5):
    """Stretches where Whisper collapsed: a long segment that is a loop ("はぁはぁ…", "あーーー"), a stock phrase
    stretched over the window ("お疲れ様でした"), moans only, or one the score filters threw out. Each span runs
    between the neighbouring good cues, so re-decoding it can't duplicate them."""
    good = sorted((c for c in cues if not drop_reason({**c, "ja": normalize_text(c["ja"])})), key=lambda c: c["start"])
    bad = [c for c in cues if drop_reason({**c, "ja": normalize_text(c["ja"])})] + dropped
    raw = sorted((max(0.0, c["start"] - pad), c["end"] + pad) for c in bad if c["end"] - c["start"] >= min_dur)
    spans = []
    for s, e in raw:
        if spans and s <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    out = []
    for s, e in spans:
        prev_end = max((g["end"] for g in good if g["end"] <= s + pad), default=0.0)
        next_start = min((g["start"] for g in good if g["start"] >= e - pad), default=e)
        s, e = max(s, prev_end), min(e, next_start)
        if e - s >= 1.0:
            out.append((s, e))
    return out


def separate_vocals(media: Path, out: Path):
    """Write the vocals stem of `media` to `out` (FLAC) with audio-separator, unless it already exists."""
    if out.exists():
        print(f"Reusing separated vocals: {out}", flush=True)
        return
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "input.wav"
        run(["ffmpeg", "-v", "error", "-y", "-i", media, "-vn", "-ac", "2", "-ar", "44100", wav])
        run([SEPARATOR_EXE, wav, "-m", SEPARATOR_MODEL, "--model_file_dir", SEPARATOR_MODEL_DIR,
             "--output_dir", tmp, "--single_stem", "Vocals", "--output_format", "FLAC", "--log_level", "warning",
             "--chunk_duration", SEPARATOR_CHUNK_SEC])
        stems = [f for f in Path(tmp).glob("*.flac") if "vocals" in f.name.lower()]
        if not stems:
            sys.exit(f"audio-separator produced no vocals stem in {tmp}")
        shutil.move(str(stems[0]), out)
    print(f"Separated vocals in {time.time() - t0:.0f}s: {out}", flush=True)


def is_good(c) -> bool:
    return not drop_reason({**c, "ja": normalize_text(c["ja"])})


def transcribe(media: Path, vocals: Path | None = None):
    from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio

    model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type=WHISPER_COMPUTE)
    pipe = BatchedInferencePipeline(model)
    print("== Transcribing original audio", flush=True)
    cues = transcribe_audio(decode_audio(str(media), sampling_rate=16000), pipe)
    if vocals:
        print("== Transcribing separated vocals", flush=True)
        vcues = transcribe_audio(decode_audio(str(vocals), sampling_rate=16000), pipe)
        good = [c for c in cues if is_good(c)]
        add = [c for c in vcues if is_good(c)
               and not any(g["start"] < c["end"] and g["end"] > c["start"] for g in good)]
        print(f"Vocals track added {len(add)} lines where the original audio had none", flush=True)
        cues = sorted(cues + add, key=lambda c: c["start"])
    del model  # release VRAM before the LLM step
    return cues


def transcribe_audio(audio, pipe, sr: int = 16000):
    """Three passes over 16 kHz mono audio: batched 30s chunks, re-decode collapsed stretches, fill gaps."""
    t0 = time.time()
    segments, _info = pipe.transcribe(
        audio, batch_size=WHISPER_BATCH_SIZE, clip_timestamps=quiet_chunks(audio), **WHISPER_PARAMS)
    dropped = []
    cues = segments_to_cues(segments, dropped=dropped)
    print(f"Pass 1: {len(audio) / sr / 60:.0f} min of audio in {time.time() - t0:.0f}s", flush=True)

    # Pass 2: Whisper collapses 30s windows dominated by moans/panting into one junk segment, losing the dialogue
    # in them. Re-decoding just those stretches in ~10s pieces recovered most of it (test film: 75 usable lines
    # from 4 min of collapsed audio in a sample, e.g. それがいいのよ, お口閉じてあげてください).
    t1 = time.time()
    spans = collapsed_spans(cues, dropped)
    extra = []
    for s, e in spans:
        clip = audio[int(s * sr):int(e * sr)]
        segs, _ = pipe.transcribe(clip, batch_size=WHISPER_BATCH_SIZE, **WHISPER_PARAMS,
                                  clip_timestamps=quiet_chunks(clip, sr, REDECODE_CHUNK_MAX, REDECODE_CHUNK_SEARCH))
        extra += [c for c in segments_to_cues(segs, offset=s, verbose=False) if s <= c["start"] < e
                  # these stretches are moan-heavy by construction: stock phrases there are hallucinations
                  and not any(ph in c["ja"] for ph in REDECODE_STOCK_PHRASES)]
    print(f"Pass 2: re-decoded {len(spans)} collapsed stretches ({sum(e - s for s, e in spans) / 60:.1f} min) "
          f"in {time.time() - t1:.0f}s: +{len(extra)} raw lines", flush=True)
    cues = sorted(cues + extra, key=lambda c: c["start"])

    # Pass 3: Whisper also skips lines inside windows that otherwise decoded fine. Silero VAD misses a lot of
    # this audio but is rarely wrong when it does fire, so every Silero speech region with no subtitle is
    # transcribed on its own (test-film sample: ~1 in 4 such regions gave a real line, the rest moans/junk).
    t2 = time.time()
    regions = uncovered_speech(audio, cues, sr)
    good = [c for c in cues if not drop_reason({**c, "ja": normalize_text(c["ja"])})]
    fill = []
    for s, e in regions:
        a, b = max(0.0, s - GAPFILL_PAD), min(len(audio) / sr, e + GAPFILL_PAD)
        clip = audio[int(a * sr):int(b * sr)]
        segs, _ = pipe.transcribe(clip, batch_size=WHISPER_BATCH_SIZE, **WHISPER_PARAMS,
                                  clip_timestamps=quiet_chunks(clip, sr, REDECODE_CHUNK_MAX, REDECODE_CHUNK_SEARCH))
        for c in segments_to_cues(segs, offset=a, verbose=False):
            if (c["start"] < e and c["end"] > s and not any(ph in c["ja"] for ph in REDECODE_STOCK_PHRASES)
                    and not drop_reason({**c, "ja": normalize_text(c["ja"])})
                    and not any(g["start"] < c["end"] and g["end"] > c["start"] for g in good + fill)):
                fill.append(c)
    print(f"Pass 3: transcribed {len(regions)} unsubtitled speech regions alone in {time.time() - t2:.0f}s: "
          f"+{len(fill)} raw lines", flush=True)
    return sorted(cues + fill, key=lambda c: c["start"])


def uncovered_speech(audio, cues, sr: int = 16000, min_len: float = 0.6):
    """Silero VAD speech regions that no usable cue overlaps."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    good = [c for c in cues if not drop_reason({**c, "ja": normalize_text(c["ja"])})]
    regions = [(t["start"] / sr, t["end"] / sr) for t in
               get_speech_timestamps(audio, VadOptions(threshold=GAPFILL_VAD_THRESHOLD, min_silence_duration_ms=500))]
    return [(s, e) for s, e in regions if e - s >= min_len and
            not any(c["start"] < e + 0.3 and c["end"] > s - 0.3 for c in good)]


# ---------------- translation ----------------
def ensure_model_loaded(model: str):
    """LLM_BACKEND=lmstudio: load `model` alone, fully on the GPU, unless this pipeline already did so.
    This unloads whatever else LM Studio has loaded, so use the openai backend if you share LM Studio."""
    lms = shutil.which("lms")
    if not lms:
        print("Note: lms CLI not found; relying on LM Studio JIT loading, which may put part of the model on CPU.",
              flush=True)
        return
    ps = subprocess.run([lms, "ps", "--json"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        loaded = json.loads(ps.stdout or "[]")
    except json.JSONDecodeError:
        loaded = []
    if len(loaded) == 1 and loaded[0].get("identifier") == model and loaded[0].get("contextLength") == LMS_CONTEXT:
        return  # loaded by a previous run of this script
    subprocess.run([lms, "unload", "--all"], capture_output=True, check=False)
    print(f"Loading {model} fully on GPU (context {LMS_CONTEXT})", flush=True)
    r = subprocess.run(
        [lms, "load", model, "--gpu", "max", "-c", str(LMS_CONTEXT), "--parallel", "1", "--identifier", model, "-y"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        sys.exit(f"lms load {model} failed:\n{(r.stdout + r.stderr)[-1000:]}")


def _tabby_headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"} if key else {}


def tabby_loaded_model(base: str, key: str) -> str | None:
    """Id of the model TabbyAPI has loaded, or None (also None if Tabby can't be reached)."""
    try:
        r = requests.get(base.rstrip("/") + "/model", headers=_tabby_headers(key), timeout=10)
    except requests.RequestException:
        return None
    return r.json().get("id") if r.ok else None


def tabby_ensure_loaded(llm) -> bool:
    """LLM_BACKEND=tabby: make sure llm.model is usable. Returns True if this call loaded it, so the caller
    unloads it afterwards and leaves the shared server as it found it."""
    loaded = tabby_loaded_model(llm.base, llm.key)
    if loaded == llm.model:
        return False
    if loaded:
        if USE_LOADED_MODEL:
            print(f"Note: TabbyAPI has {loaded} loaded, not {llm.model}; using it (LLM_USE_LOADED_MODEL=1)",
                  flush=True)
            llm.model = loaded
            return False
        sys.exit(f"TabbyAPI has {loaded} loaded, not {llm.model}. Tabby is treated as shared, so this script won't "
                 f"swap models: load {llm.model} yourself, pass --model {loaded}, or set LLM_USE_LOADED_MODEL=1.")
    print(f"Loading {llm.model} on TabbyAPI (context {TABBY_LOAD_CTX}, KV cache {TABBY_CACHE_MODE})", flush=True)
    t0 = time.time()
    # The load endpoint streams progress events and needs an admin key.
    r = requests.post(llm.base.rstrip("/") + "/model/load", headers=_tabby_headers(llm.key),
                      json={"model_name": llm.model, "max_seq_len": TABBY_LOAD_CTX, "cache_size": TABBY_LOAD_CTX,
                            "cache_mode": TABBY_CACHE_MODE}, stream=True, timeout=900)
    body = "".join(chunk.decode("utf-8", "replace") for chunk in r.iter_content(chunk_size=None))
    if not r.ok or tabby_loaded_model(llm.base, llm.key) != llm.model:
        sys.exit(f"TabbyAPI couldn't load {llm.model} (HTTP {r.status_code}): {body[-500:]}")
    print(f"Loaded {llm.model} in {time.time() - t0:.0f}s", flush=True)
    return True


def tabby_unload(llm):
    r = requests.post(llm.base.rstrip("/") + "/model/unload", headers=_tabby_headers(llm.key), timeout=120)
    print(f"Unloaded {llm.model} from TabbyAPI" if r.ok else
          f"Note: TabbyAPI unload returned HTTP {r.status_code}: {r.text[:200]}", flush=True)


@dataclass
class LLM:
    base: str
    key: str
    model: str
    backend: str  # openai, tabby or lmstudio (see LLM_BACKEND above)

    @property
    def provider(self) -> str:  # label for log messages
        return {"tabby": "TabbyAPI", "lmstudio": "LM Studio"}.get(self.backend, self.base)


def make_llm(model: str | None) -> LLM:
    if LLM_BACKEND not in ("openai", "tabby", "lmstudio"):
        sys.exit(f"LLM_BACKEND={LLM_BACKEND!r}: use openai, tabby or lmstudio.")
    missing = [k for k, v in (("LLM_BASE_URL", LLM_BASE_URL), ("LLM_MODEL", model or LLM_MODEL)) if not v]
    if missing:
        where = f"in {ENV_FILE}" if ENV_FILE else f"in a .env file (copy {REPO_ROOT / '.env.example'} to .env)"
        sys.exit(f"Translation needs {' and '.join(missing)}: set {'them' if len(missing) > 1 else 'it'} {where}. "
                 "To make Japanese subtitles only, use --asr without --translate.")
    return LLM(LLM_BASE_URL, os.environ.get("LLM_API_KEY", ""), model or LLM_MODEL, LLM_BACKEND)


def chat(user_msg: str, llm: LLM, system: str) -> str:
    headers = {"Authorization": f"Bearer {llm.key}"} if llm.key else {}
    body = {
        "model": llm.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
    }
    if REASONING_EFFORT:
        body["reasoning_effort"] = REASONING_EFFORT
    url = llm.base.rstrip("/") + "/chat/completions"
    attempts = 1 if llm.backend == "lmstudio" else REMOTE_RETRIES
    for attempt in range(1, attempts + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=900)
        except requests.RequestException as e:
            if attempt == attempts:
                sys.exit(f"{llm.provider} request failed: {e}")
            print(f"  {llm.provider}: {type(e).__name__}, retrying ({attempt}/{attempts})", flush=True)
            time.sleep(2 ** attempt)
            continue
        if (r.status_code == 429 or r.status_code >= 500) and attempt < attempts:
            print(f"  {llm.provider}: HTTP {r.status_code}, retrying ({attempt}/{attempts})", flush=True)
            time.sleep(2 ** attempt)
            continue
        break
    if r.status_code == 401:
        sys.exit(f"{llm.provider} returned 401. Check LLM_API_KEY in .env.")
    if not r.ok:
        sys.exit(f"{llm.provider} returned {r.status_code} for model '{llm.model}': {r.text[:500]}\n"
                 f"Model ids are listed at {llm.base.rstrip('/')}/models")
    out = r.json()["choices"][0]["message"].get("content") or ""
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()  # strip inline reasoning blocks


def find_title(src: Path, base: str) -> str:
    """'[ABC-123] Some Title.mp4' next to ABC-123.restored.mp4 -> 'Some Title'."""
    for f in sorted(src.parent.glob(f"*{base}*")):
        if f.suffix.lower() not in (".mp4", ".mkv", ".avi", ".wmv", ".mov"):
            continue
        rest = f.stem.replace(f"[{base}]", "").replace(base, "").strip(" -_")
        if rest and not f.stem.startswith(base + "."):
            return rest
    return ""


def load_or_build_dossier(path: Path, cues, title: str, llm: LLM) -> str:
    if path.exists():
        print(f"Reusing context brief: {path}", flush=True)
        return path.read_text(encoding="utf-8").strip()
    step = max(1, len(cues) // DOSSIER_SAMPLE_LINES)
    sample = cues[::step][:DOSSIER_SAMPLE_LINES]
    user = (f"Video title: {title}\n\n" if title else "") + \
        "Transcript sample, evenly spaced across the whole video:\n" + \
        "\n".join(f"[{ts(c['start'])[:8]}] {c['ja']}" for c in sample)
    print(f"Writing context brief from {len(sample)} sample lines" + (f" + title '{title}'" if title else ""),
          flush=True)
    brief = chat(user, llm, DOSSIER_PROMPT)
    path.write_text(brief + "\n", encoding="utf-8")
    print(brief, flush=True)
    return brief


def translation_system_prompt(dossier: str | None) -> str:
    if not dossier:
        return SYSTEM_PROMPT
    return (SYSTEM_PROMPT + "\n\nBrief on this video (use it for speaker genders, who is being addressed, tone "
            "and recurring terms; it was written from a speech-recognition transcript and may be wrong in places):\n"
            + dossier + "\n\nRemember: return ONLY the JSON array.")


_JSON_ITEM = re.compile(r'\{\s*"id"\s*:\s*"?(\d+)"?\s*,\s*"en"\s*:\s*("(?:[^"\\]|\\.)*")\s*\}')


def parse_json_array(text: str):
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    # One malformed object (e.g. artemis writing {"id": 492, "en "...}) breaks the whole array: salvage the
    # well-formed objects one by one so only the broken ids get retried.
    items = []
    for m in _JSON_ITEM.finditer(text):
        try:
            items.append({"id": int(m.group(1)), "en": json.loads(m.group(2))})
        except json.JSONDecodeError:
            pass
    return items


def to_map(items):
    out = {}
    for item in items:
        try:
            out[int(item["id"])] = str(item["en"]).strip()
        except (KeyError, TypeError, ValueError):
            pass
    return out


def translate(cues, llm: LLM, batch_size: int, dossier: str | None = None):
    system = translation_system_prompt(dossier)
    t0 = time.time()
    for b in range(0, len(cues), batch_size):
        batch = list(enumerate(cues[b : b + batch_size], start=b))
        ctx = cues[max(0, b - CONTEXT_LINES) : b]
        msg = ""
        if ctx:
            msg += "Previous lines, for context only (do not translate):\n"
            msg += "\n".join(c["ja"] for c in ctx) + "\n\n"
        msg += "Translate these lines:\n" + "\n".join(f"{i}: {c['ja']}" for i, c in batch)

        reply = chat(msg, llm, system)
        got = to_map(parse_json_array(reply))
        if sum(1 for i, _ in batch if got.get(i)) < len(batch) / 2:  # badly mangled: retry the whole batch once
            print(f"  batch reply unusable ({len(reply)} chars: {reply[:300]!r}...), retrying batch", flush=True)
            reply = chat(msg, llm, system)
            got.update(to_map(parse_json_array(reply)))
            if sum(1 for i, _ in batch if got.get(i)) < len(batch) / 2:
                print(f"  retry unusable too ({len(reply)} chars: {reply[:300]!r}...)", flush=True)
        for i, c in batch:
            if not got.get(i):  # model skipped or mangled this id: retry it alone
                print(f"  retrying id {i} alone", flush=True)
                retry = to_map(parse_json_array(chat(f"Translate these lines:\n{i}: {c['ja']}", llm, system)))
                got[i] = retry.get(i) or f"[untranslated] {c['ja']}"
            c["en"] = got[i]
        print(f"translated {min(b + batch_size, len(cues))}/{len(cues)} ({time.time() - t0:.0f}s)", flush=True)


# ---------------- mux / convert ----------------
TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".wmv", ".mov", ".m4v", ".ts", ".m2ts", ".mts", ".webm", ".flv", ".mpg",
              ".mpeg", ".vob", ".3gp"}
# Stems of files this tool writes; skipped when scanning folders so a second run doesn't process its own output
OUTPUT_STEM_SUFFIXES = (".restored", ".decensored", ".subbed", ".converted")
PARTIAL = ".ljt-partial"  # marks an output still being written

# --reencode: (NVENC encoder, its args, software fallback, its args). The CQ/CRF value is the last arg of each.
# The NVENC HEVC settings re-encoded a 3.9 h H.264 auto-split restore from 7.6 GB to 4.6 GB in 15 min (RTX 5090).
ENCODERS = {
    "hevc": ("hevc_nvenc", ["-preset", "p5", "-tune", "hq", "-profile:v", "main10", "-pix_fmt", "p010le",
                            "-rc", "vbr", "-b:v", "0", "-spatial-aq", "1", "-temporal-aq", "1", "-rc-lookahead", "32",
                            "-bf", "4", "-b_ref_mode", "middle", "-g", "250", "-cq", "28"],
             "libx265", ["-preset", "medium", "-pix_fmt", "yuv420p10le", "-crf", "22"]),
    "h264": ("h264_nvenc", ["-preset", "p5", "-tune", "hq", "-profile:v", "high", "-pix_fmt", "yuv420p",
                            "-rc", "vbr", "-b:v", "0", "-spatial-aq", "1", "-rc-lookahead", "32", "-g", "250",
                            "-cq", "23"],
             "libx264", ["-preset", "medium", "-pix_fmt", "yuv420p", "-crf", "20"]),
    "av1": ("av1_nvenc", ["-preset", "p5", "-tune", "hq", "-pix_fmt", "p010le", "-rc", "vbr", "-b:v", "0",
                          "-spatial-aq", "1", "-rc-lookahead", "32", "-g", "250", "-cq", "32"],
            "libsvtav1", ["-preset", "6", "-pix_fmt", "yuv420p10le", "-crf", "30"]),
}
_ffmpeg_encoders: set[str] | None = None


def video_encode_args(codec: str, cq: int | None) -> list[str]:
    global _ffmpeg_encoders
    if _ffmpeg_encoders is None:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
        _ffmpeg_encoders = {ln.split()[1] for ln in r.stdout.splitlines() if len(ln.split()) > 1}
    nv, nv_args, sw, sw_args = ENCODERS[codec]
    enc, args = (nv, list(nv_args)) if nv in _ffmpeg_encoders else (sw, list(sw_args))
    if enc == sw:
        print(f"Note: {nv} isn't available in this ffmpeg; encoding with {sw} on the CPU (much slower).", flush=True)
    if cq is not None:
        args[-1] = str(cq)
    return ["-c:v", enc, *args]


def subtitle_streams(video: Path) -> list[dict]:
    """[{codec, language}] for the subtitle streams in `video`, in order."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries",
                        "stream=codec_name:stream_tags=language", "-of", "json", str(video)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        streams = json.loads(r.stdout or "{}").get("streams", [])
    except json.JSONDecodeError:
        streams = []
    return [{"codec": s.get("codec_name", ""), "language": (s.get("tags") or {}).get("language", "").lower()}
            for s in streams]


def mux(video: Path, subs, out: Path, keep_existing: bool = False, existing_from: Path | None = None,
        fmt: str = "mkv", reencode: str | None = None, cq: int | None = None):
    """Write `out` from `video` plus subs [(srt_path, language, title), ...]; the first sub is the default track.
    keep_existing: also carry over the subtitle tracks of `existing_from` (default `video`), after ours and never
    default, with their language/title tags. fmt: mkv or mp4 (mp4 holds text subtitles as mov_text and can't hold
    bitmap PGS/VobSub tracks, which are dropped with a note). reencode: hevc/h264/av1, or None to copy the video."""
    existing_from = existing_from or video
    existing = subtitle_streams(existing_from) if keep_existing else []
    sub_codec = "mov_text" if fmt == "mp4" else "srt"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-stats", "-y", "-i", video]
    for path, _, _ in subs:
        cmd += ["-i", path]
    ex_input = None
    if existing:
        if existing_from == video:
            ex_input = 0
        else:
            ex_input = len(subs) + 1
            cmd += ["-i", existing_from]
    # 0:V (capital) leaves out attached pictures such as cover art, which a video encoder can't take
    cmd += ["-map", "0:V" if reencode else "0:v", "-map", "0:a?"]
    for k in range(len(subs)):
        cmd += ["-map", str(k + 1)]
    kept = []
    for i, s in enumerate(existing):
        if fmt == "mp4" and s["codec"] not in TEXT_SUB_CODECS:
            print(f"Note: dropping existing {s['codec']} subtitle track {i} (bitmap subtitles can't go in MP4)",
                  flush=True)
            continue
        cmd += ["-map", f"{ex_input}:s:{i}"]
        kept.append(s)
    cmd += video_encode_args(reencode, cq) if reencode else ["-c:v", "copy"]
    cmd += ["-c:a", "copy"]
    for k, (_, lang, title) in enumerate(subs):
        cmd += [f"-c:s:{k}", sub_codec, f"-metadata:s:s:{k}", f"language={lang}", f"-metadata:s:s:{k}",
                f"title={title}", f"-disposition:s:{k}", "default" if k == 0 else "0"]
    for j, s in enumerate(kept, start=len(subs)):
        if fmt == "mp4":
            codec = "mov_text"
        else:
            codec = "srt" if s["codec"] == "mov_text" else "copy"  # mov_text can't go into MKV as is
        cmd += [f"-c:s:{j}", codec, f"-disposition:s:{j}", "0"]
    if keep_existing:
        print(f"Keeping {len(kept)} subtitle track(s) already in the video" if kept else
              "Note: --keep-subs, but the video has no subtitle tracks that can be kept", flush=True)
    if fmt == "mp4":
        cmd += ["-movflags", "+faststart"]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + PARTIAL + out.suffix)  # a killed run never leaves a half-written final file
    run(cmd + [tmp])
    os.replace(tmp, out)


def extract_text_subs(video: Path, language: str, out: Path) -> bool:
    """Write the first text subtitle track of `video` tagged `language` (ISO 639-2, e.g. jpn) to `out` as SRT."""
    for i, s in enumerate(subtitle_streams(video)):
        if s["language"] == language and s["codec"] in TEXT_SUB_CODECS:
            r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                                "-map", f"0:s:{i}", "-c:s", "srt", str(out)], capture_output=True, text=True)
            return r.returncode == 0 and out.exists() and out.stat().st_size > 0
    return False


def existing_subtitles(src: Path, base: str, work: Path, opts) -> list:
    """OpenSubtitles: [(srt, iso639-2, title), ...] with the SUBS_LANG track first, or [] when nothing synced
    well. Synced files are cached as <work>/<name>.<lang>.opensubtitles.srt and reused on later runs."""
    import opensubs

    primary, primary_tag, primary_name = opensubs.lang_codes(opts.subs_lang)
    langs = [primary]
    orig = None if opts.no_original_subs else opensubs.audio_language(src)
    names = {primary: (primary_tag, primary_name)}
    if orig:
        o_code, o_tag, o_name = opensubs.lang_codes(orig)
        if o_code != primary:
            langs.append(o_code)
            names[o_code] = (o_tag, o_name)
    cached = {lang: work / f"{base}.{lang}.opensubtitles.srt" for lang in langs}
    if cached[primary].exists():
        print(f"Reusing synced OpenSubtitles subtitles: {cached[primary]}", flush=True)
    else:
        found = opensubs.find_existing_subtitles(src, langs, opts.imdb)
        if primary not in found:
            if os.environ.get("OPENSUBTITLES_API_KEY"):
                print(f"No well-synced {primary_name} subtitles on OpenSubtitles.", flush=True)
            return []
        for lang, f in found.items():
            cached[lang].write_text(f.srt_text, encoding="utf-8")
            print(f"Wrote {cached[lang]} ({f.release}; {f.grade})", flush=True)
    return [(cached[lang], names[lang][0], f"{names[lang][1]} (OpenSubtitles)")
            for lang in langs if cached[lang].exists()]


# ---------------- planning ----------------
class FileFailed(Exception):
    """This file can't be processed; the batch carries on with the next one."""


@dataclass
class Steps:
    decensor: bool = False
    smart_subs: bool = False   # --subs: OpenSubtitles, else ASR + translation
    opensubs: bool = False
    asr: bool = False
    translate: bool = False
    convert: bool = False

    @property
    def any_subs(self) -> bool:
        return self.smart_subs or self.opensubs or self.asr or self.translate


def resolve_steps(a) -> Steps:
    s = Steps(decensor=a.decensor, smart_subs=a.subs, opensubs=a.opensubs, asr=a.asr, translate=a.translate,
              convert=a.convert or bool(a.format) or bool(a.reencode))
    if a.transcribe_only:  # old flag: Japanese transcript only, as an SRT
        s.asr, a.no_mux = True, True
    if a.en_srt:  # old A/B flag: mux these English SRTs instead of translating
        s.translate = True
    if a.restored:  # decensored elsewhere: counts as the decensor step
        s.decensor = True
    if not (s.decensor or s.any_subs or s.convert):
        s.smart_subs = True  # no step named: subtitle it the smart way
    if a.no_opensubtitles and s.smart_subs:  # old flag: smart subtitles without the OpenSubtitles lookup
        s.smart_subs, s.asr, s.translate = False, True, True
    return s


def is_own_output(p: Path) -> bool:
    stem = p.stem.lower()
    return stem.endswith(OUTPUT_STEM_SUFFIXES) or stem.endswith(PARTIAL)


def collect_inputs(paths, recursive: bool, exts: set[str], out_dir: Path | None) -> list[tuple[Path, Path]]:
    """[(video, root)] where root is the folder the user named (the base for --out-dir's mirrored layout)."""
    found, seen = [], set()
    for p in paths:
        p = p.resolve()
        if p.is_file():
            items = [(p, p.parent)]  # named explicitly: taken whatever its extension
        elif p.is_dir():
            pattern = p.rglob("*") if recursive else p.glob("*")
            items = [(f, p) for f in sorted(pattern, key=lambda f: str(f).lower())
                     if f.is_file() and f.suffix.lower() in exts and not is_own_output(f)
                     and not (out_dir and out_dir in f.parents)]
        else:
            sys.exit(f"Not found: {p}")
        for f, root in items:
            if f not in seen:
                seen.add(f)
                found.append((f, root))
    return found


def output_dir_for(src: Path, root: Path, a) -> Path:
    if not a.out_dir:
        return src.parent
    return a.out_dir if a.flatten else a.out_dir / src.parent.relative_to(root)


def output_name(src: Path, base: str, steps: Steps, a, decensored: bool, subbed: bool, out_dir: Path) -> Path:
    tags = (".decensored" if decensored else "") + (".subbed" if subbed else "")
    if a.format:
        ext = "." + a.format
    elif subbed or steps.convert:
        ext = ".mkv"
    else:  # decensor only: keep Jasna's container
        ext = ".mp4"
    out = out_dir / f"{base}{tags}{ext}"
    if out.resolve() == src.resolve():  # converting in place would overwrite the input
        out = out_dir / f"{base}{tags}.converted{ext}"
    return out


# ---------------- per-file processing ----------------
class LLMSession:
    """One LLM client for the whole batch; with LLM_BACKEND=tabby, a model this run loaded is unloaded again."""

    def __init__(self, model: str | None):
        self.model = model
        self.llm = None
        self.we_loaded = False

    def get(self) -> LLM:
        if self.llm is None:
            self.llm = make_llm(self.model)
        if self.llm.backend == "lmstudio":
            ensure_model_loaded(self.llm.model)
        elif self.llm.backend == "tabby" and not self.we_loaded:
            self.we_loaded = tabby_ensure_loaded(self.llm)
        return self.llm

    def release(self):
        if self.we_loaded:  # leave the shared Tabby server as we found it
            tabby_unload(self.llm)
            self.we_loaded = False


def translation_source(work: Path, src: Path, base: str, asr_cues, os_tracks, temps: list):
    """Japanese cues to translate and where they came from, best first: this run's transcript, an earlier
    transcript of this file, Japanese subtitles from OpenSubtitles, a Japanese text track in the video."""
    if asr_cues is not None:
        return asr_cues, "this run's speech recognition"
    for p in (work / f"{base}.ja.srt", src.with_name(f"{base}.ja.srt")):
        if p.exists():
            return clean_cues(read_srt(p), verbose=False), f"an earlier transcript ({p.name})"
    for path, lang, _ in os_tracks:
        if lang == "jpn":
            return read_srt(path), "Japanese subtitles from OpenSubtitles"
    embedded = work / f"{base}.ja.embedded.srt"
    if embedded.exists() or extract_text_subs(src, "jpn", embedded):
        temps.append(embedded)
        return read_srt(embedded), "the Japanese subtitle track in the video"
    return None, None


def process_file(src: Path, root: Path, steps: Steps, a, llm_session: LLMSession) -> list[Path]:
    """Run the requested steps on one video. Returns the final output files."""
    base = src.stem.removesuffix(".restored")  # "X.restored.mp4" shares its subtitle files with "X.mp4"
    out_dir = output_dir_for(src, root, a)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir  # intermediates live next to the output, so a kept or failed run can be resumed
    temps: list[Path] = []  # intermediates to delete once this file succeeds
    finals: list[Path] = []

    def temp(p: Path) -> Path:
        temps.append(p)
        return p

    # 1. Decensor
    video, decensored = src, src.stem.endswith(".restored")
    if a.restored:
        video, decensored = a.restored.resolve(), True
        if not video.exists():
            raise FileFailed(f"--restored video not found: {video}")
    elif steps.decensor:
        restored = work / f"{base}.restored.mp4"
        if restored.exists():
            print(f"Reusing decensored video: {restored}", flush=True)
        else:
            free_vram_for_jasna()
            decensor(src, restored)
        video, decensored = temp(restored), True

    # 2. Existing subtitles (OpenSubtitles)
    os_tracks = []
    if steps.opensubs or steps.smart_subs:
        os_tracks = existing_subtitles(src, base, work, a)
        temps.extend(p for p, _, _ in os_tracks)
    smart_fallback = steps.smart_subs and not os_tracks
    do_asr = steps.asr or smart_fallback
    do_translate = steps.translate or (smart_fallback and not a.no_translate_fallback)

    # 3. Speech recognition (Japanese)
    asr_cues = None
    ja_srt = work / f"{base}.ja.srt"  # raw Whisper output: the resumable cache
    if do_asr:
        legacy = src.with_name(f"{base}.ja.srt")
        if ja_srt.exists() or legacy.exists():
            used = ja_srt if ja_srt.exists() else legacy
            print(f"Reusing existing transcript: {used}", flush=True)
            raw = read_srt(used)
            if used == ja_srt:
                temps.append(ja_srt)
        else:
            vocals = None
            if a.no_vocals:
                pass
            elif SEPARATOR_EXE.exists():
                vocals = temp(work / f"{base}.vocals.flac")
                free_vram_for_jasna()  # get the LLM out of VRAM where that's ours to do (see the function)
                separate_vocals(video, vocals)
            else:
                print(f"Note: {SEPARATOR_EXE} not found; transcribing the original audio only.", flush=True)
            raw = transcribe(video, vocals)
            write_srt(raw, temp(ja_srt), "ja")
        asr_cues = clean_cues(raw, verbose=a.verbose_cleanup)
        coverage_report(asr_cues, media_duration(video))
        print(f"{len(asr_cues)} Japanese lines ({len(raw) - len(asr_cues)} removed by cleanup)", flush=True)

    # 4. Translation (Japanese -> English)
    mt_tracks = []  # [(srt, title)]
    if do_translate:
        en_tag = ("." + re.sub(r"[^\w.-]+", "_", a.model) if a.model else "") + (".nodossier" if a.no_dossier else "")
        if a.reasoning_tag:
            en_tag += f".reason-{a.reasoning_tag}"
        en_srt = work / f"{base}.en{en_tag}.srt"
        if a.en_srt:
            for spec in a.en_srt:
                path_s, _, title = spec.partition("=")
                mt_tracks.append((Path(path_s).resolve(), title or "English (machine translation)"))
                print(f"Using existing English subtitles: {mt_tracks[-1][0]}", flush=True)
        else:
            cues, origin = translation_source(work, src, base, asr_cues, os_tracks, temps)
            if cues is None:
                msg = ("--translate found no Japanese to translate: no transcript, no Japanese subtitles from "
                       "OpenSubtitles and no Japanese text track in the video. Add --asr (or --opensubs).")
                if steps.translate:
                    raise FileFailed(msg)
                print("Note: " + msg, flush=True)
            else:
                cues = [dict(c) for c in cues]  # translate() writes c["en"]; keep the JA cues untouched
                llm = llm_session.get()
                print(f"Translating {len(cues)} lines from {origin} with {llm.provider}: {llm.model}", flush=True)
                dossier = None
                if not a.no_dossier:
                    dossier_path = temp(work / f"{base}.dossier.txt")
                    title = a.title if a.title is not None else find_title(src, base)
                    dossier = load_or_build_dossier(dossier_path, cues, title, llm)
                translate(cues, llm, a.batch_size, dossier)
                if steps.decensor:  # Jasna needs the whole GPU for the next file
                    llm_session.release()
                write_srt(cues, temp(en_srt), "en")
                print("Wrote", en_srt, flush=True)
                mt_tracks.append((en_srt, "English (machine translation)"))

    # 5. Assemble
    subs = list(os_tracks) + [(p, "eng", t) for p, t in mt_tracks]
    with tempfile.TemporaryDirectory() as tmp:
        if asr_cues is not None:
            ja_clean = Path(tmp) / "ja.srt"  # mux the cleaned JA cues; ja.srt on disk stays the raw cache
            write_srt(asr_cues, ja_clean, "ja")
            subs.append((ja_clean, "jpn", "Japanese (ASR)"))
        elif os_tracks and not any(lang == "jpn" for _, lang, _ in subs):
            earlier = next((p for p in (ja_srt, src.with_name(f"{base}.ja.srt")) if p.exists()), None)
            if earlier:  # an earlier transcript of this file: add it, cleaned, as before
                ja_clean = Path(tmp) / "ja.srt"
                write_srt(clean_cues(read_srt(earlier)), ja_clean, "ja")
                subs.append((ja_clean, "jpn", "Japanese (ASR)"))

        if a.no_mux:
            # The subtitle files (raw Japanese transcript, English, OpenSubtitles) and a decensored video are the
            # outputs; keep them
            finals = [p for p, _, _ in os_tracks] + [p for p, _ in mt_tracks if p.parent == work]
            if asr_cues is not None and ja_srt.exists():
                finals.append(ja_srt)
            if steps.decensor and not a.restored and video != src:
                finals.append(video)
            temps = [t for t in temps if t not in finals]
        elif not subs and not steps.convert and not steps.decensor:
            print("No subtitles were produced, so there's nothing to write for this file.", flush=True)
        elif steps.decensor and not subs and not steps.convert and not a.restored:
            final = output_name(src, base, steps, a, True, False, out_dir)
            os.replace(video, final)  # decensor only: Jasna's output is the result
            temps.remove(video)
            finals = [final]
        else:
            final = a.output.resolve() if a.output else output_name(src, base, steps, a, decensored, bool(subs),
                                                                   out_dir)
            fmt = a.format or ("mp4" if final.suffix.lower() == ".mp4" else "mkv")
            mux(video, subs, final, a.keep_subs, existing_from=src, fmt=fmt, reencode=a.reencode, cq=a.cq)
            finals = [final]

    # 6. Clean up
    if a.keep_intermediates:
        if temps:
            print(f"Kept {len([t for t in temps if t.exists()])} intermediate file(s) (--keep-intermediates)",
                  flush=True)
    else:
        for t in temps:
            if t.exists() and t not in finals and t.resolve() != src.resolve():
                t.unlink()
    return finals


# ---------------- CLI ----------------
EXAMPLES = """\
steps (combine any of them; with none, --subs is assumed):
  video.sh movie.mkv                          smart subtitles: OpenSubtitles if it has a good match, else
                                              speech recognition + translation
  video.sh X.mp4 --decensor --subs            decensor, then smart subtitles
  video.sh X.mp4 --decensor                   decensor only (writes X.decensored.mp4)
  video.sh X.mp4 --asr                        Japanese subtitles only (no LLM)
  video.sh X.mp4 --asr --translate            always transcribe and translate, even if OpenSubtitles has subs
  video.sh X.mkv --translate                  translate the Japanese text track already in the file
  video.sh X.avi --convert                    remux to MKV, nothing else
  video.sh X.mp4 --reencode hevc --format mkv re-encode to HEVC (NVENC) into an MKV
batches and output:
  video.sh "E:/incoming"                      every video in the folder
  video.sh "E:/incoming" -r --ext mkv,mp4     recurse into subfolders, only .mkv and .mp4
  video.sh "E:/in" -r --out-dir "F:/done"     write to F:/done, mirroring the folder structure
  video.sh "E:/in" -r --out-dir F:/done --flatten   everything directly in F:/done
  video.sh X.mp4 --keep-intermediates         keep the transcript, vocals, decensored video, ...

Files whose output already exists are skipped (--overwrite redoes them). Intermediates are written next to the
output, deleted when a file succeeds and kept when it fails, so re-running resumes where it stopped.
"""


def parse_args():
    global REASONING_EFFORT
    ap = argparse.ArgumentParser(
        prog="video.sh",
        description="Decensor, subtitle (OpenSubtitles or speech recognition), translate and convert videos.\n"
                    "Settings (LLM server, Jasna, OpenSubtitles key) come from .env; see .env.example.",
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    ap.add_argument("-h", "-?", "--help", action="help", help="show this help, with examples, and exit")
    ap.add_argument("inputs", nargs="+", type=Path, help="video files and/or folders")

    g = ap.add_argument_group("steps")
    g.add_argument("--decensor", action="store_true", help="decensor with Jasna (JASNA_EXE in .env)")
    g.add_argument("--subs", action="store_true",
                   help="smart subtitles: OpenSubtitles if it has subtitles that sync well, else speech recognition "
                        "+ translation (the default when no step is named)")
    g.add_argument("--opensubs", action="store_true",
                   help="subtitles from OpenSubtitles (OPENSUBTITLES_API_KEY in .env), synced to the audio")
    g.add_argument("--asr", action="store_true", help="Japanese subtitles by speech recognition (Whisper)")
    g.add_argument("--translate", action="store_true",
                   help="English subtitles by LLM translation of the best Japanese source: this run's speech "
                        "recognition, an earlier transcript, Japanese OpenSubtitles subs, or a Japanese text track "
                        "in the video")
    g.add_argument("--convert", action="store_true",
                   help="write the output in --format (default mkv) even if no other step changes the file")

    g = ap.add_argument_group("input")
    g.add_argument("-r", "--recursive", action="store_true", help="include videos in subfolders of input folders")
    g.add_argument("--ext", help="only these extensions when scanning folders, e.g. mp4,mkv (default: common "
                                 "video types)")

    g = ap.add_argument_group("output")
    g.add_argument("--out-dir", type=Path, help="write outputs here instead of next to each input; the input "
                                                "folder structure is mirrored unless --flatten")
    g.add_argument("--flatten", action="store_true", help="with --out-dir, put every output directly in it")
    g.add_argument("--output", type=Path, help="exact output file (single input file only)")
    g.add_argument("--format", choices=("mkv", "mp4"), help="output container (implies --convert; default mkv)")
    g.add_argument("--reencode", choices=sorted(ENCODERS),
                   help="re-encode the video (NVENC if available, else CPU); implies --convert. Default: copy")
    g.add_argument("--cq", type=int, help="quality for --reencode: NVENC CQ or CPU CRF, lower is better (default: "
                                         "hevc 28 / h264 23 / av1 32 on NVENC)")
    g.add_argument("--keep-intermediates", action="store_true",
                   help="keep the transcript, vocals, decensored video, context brief and synced subtitle files "
                        "(default: delete them once a file succeeds)")
    g.add_argument("--no-mux", action="store_true",
                   help="write the subtitle files as the output instead of a video")
    g.add_argument("--overwrite", action="store_true", help="redo files whose output already exists")
    g.add_argument("--keep-subs", action="store_true",
                   help="keep the video's own subtitle tracks, after the new ones (default: drop them)")

    g = ap.add_argument_group("subtitles and translation")
    g.add_argument("--subs-lang", default=SUBS_LANG,
                   help=f"language to look for on OpenSubtitles, e.g. en, pt-br (default: SUBS_LANG, now {SUBS_LANG})")
    g.add_argument("--imdb", metavar="ID", help="IMDb id of the movie or episode, e.g. tt1234567 (single file)")
    g.add_argument("--no-original-subs", action="store_true",
                   help="with OpenSubtitles, don't also fetch subtitles in the audio's own language")
    g.add_argument("--no-translate-fallback", action="store_true",
                   help="with --subs, fall back to speech recognition only (Japanese subtitles, no LLM)")
    g.add_argument("--model", help="translation model id (default: LLM_MODEL). With --model, EN subs go to "
                                   "<name>.en.<model>.srt so A/B runs don't collide")
    g.add_argument("--reasoning", default=REASONING_EFFORT,
                   help="reasoning_effort sent to the model, e.g. none (default: LLM_REASONING_EFFORT, now "
                        f"{REASONING_EFFORT or 'unset'})")
    g.add_argument("--no-dossier", action="store_true", help="translate without the context brief")
    g.add_argument("--title", help="video title for the context brief (default: guessed from a sibling file like "
                                   "'[CODE] Title.mp4')")
    g.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"lines per translation request "
                                                                      f"(default: {BATCH_SIZE})")
    g.add_argument("--no-vocals", action="store_true", help="skip vocal separation; transcribe the original audio")
    g.add_argument("--verbose-cleanup", action="store_true", help="print every line the cleanup filters drop")
    g.add_argument("--restored", type=Path, help="an already-decensored version of the (single) input, e.g. from "
                                                 "the Jasna GUI; counts as the decensor step")
    g.add_argument("--en-srt", action="append", metavar="PATH[=TITLE]",
                   help="mux this English SRT instead of translating (single file; repeatable for A/B)")

    # Old flags, still accepted
    ap.add_argument("--skip-decensor", action="store_true", help=argparse.SUPPRESS)  # now the default
    ap.add_argument("--transcribe-only", action="store_true", help=argparse.SUPPRESS)  # = --asr --no-mux
    ap.add_argument("--no-opensubtitles", action="store_true", help=argparse.SUPPRESS)  # = --asr --translate
    if len(sys.argv) == 1:  # no arguments at all: show the full help rather than a one-line error
        ap.print_help()
        sys.exit(0)
    a = ap.parse_args()

    # Tag the EN file only when --reasoning overrides .env, so A/B runs don't overwrite the normal translation
    a.reasoning_tag = a.reasoning if (a.reasoning and a.reasoning != "none" and a.reasoning != REASONING_EFFORT) \
        else None
    REASONING_EFFORT = a.reasoning
    a.out_dir = a.out_dir.resolve() if a.out_dir else None
    if a.flatten and not a.out_dir:
        ap.error("--flatten needs --out-dir")
    if a.decensor and a.skip_decensor:
        ap.error("--decensor and --skip-decensor contradict each other")
    return a


def main():
    a = parse_args()
    steps = resolve_steps(a)
    exts = {("." + e.strip().lower().lstrip(".")) for e in a.ext.split(",")} if a.ext else VIDEO_EXTS
    inputs = collect_inputs(a.inputs, a.recursive, exts, a.out_dir)
    if not inputs:
        sys.exit("No videos found" + (" (folders are only scanned one level deep; add -r to recurse)"
                                      if not a.recursive else "") + ".")
    single_only = [f for f, v in (("--output", a.output), ("--restored", a.restored), ("--imdb", a.imdb),
                                  ("--en-srt", a.en_srt)) if v]
    if single_only and len(inputs) > 1:
        sys.exit(f"{', '.join(single_only)} only work with a single input file ({len(inputs)} found).")

    # Fail on missing settings now, not after hours of decensoring and transcription
    if steps.decensor and not a.restored and (not JASNA_EXE or not Path(JASNA_EXE).is_file()):
        sys.exit("--decensor needs Jasna: set JASNA_EXE in .env to jasna.exe (see README, 'Jasna').")
    if steps.translate and not a.en_srt:
        make_llm(a.model)
    elif steps.smart_subs and not a.no_translate_fallback and (not LLM_BASE_URL or not (a.model or LLM_MODEL)):
        print("Note: LLM_BASE_URL/LLM_MODEL aren't set, so files without OpenSubtitles matches get Japanese "
              "subtitles only.", flush=True)
        a.no_translate_fallback = True

    plan = []
    for src, root in inputs:
        base = src.stem.removesuffix(".restored")
        out_dir = output_dir_for(src, root, a)
        decensored = steps.decensor or src.stem.endswith(".restored")
        target = None if a.no_mux else (a.output.resolve() if a.output else
                                        output_name(src, base, steps, a, decensored, steps.any_subs, out_dir))
        plan.append((src, root, target))
    targets = [t for _, _, t in plan if t]
    dupes = {t for t in targets if targets.count(t) > 1}
    if dupes:
        sys.exit("These outputs would collide (same name from different folders; drop --flatten or rename):\n  "
                 + "\n  ".join(str(d) for d in sorted(dupes)))

    named = [n for n, on in (("decensor", steps.decensor), ("smart subtitles", steps.smart_subs),
                             ("OpenSubtitles", steps.opensubs), ("speech recognition", steps.asr),
                             ("translation", steps.translate), ("convert", steps.convert)) if on]
    print(f"{len(plan)} video(s); steps: {', '.join(named)}", flush=True)

    llm_session = LLMSession(a.model)
    results = []
    try:
        for n, (src, root, target) in enumerate(plan, 1):
            print(f"\n=== [{n}/{len(plan)}] {src}", flush=True)
            if target and target.exists() and not a.overwrite:
                print(f"Skipping: {target} already exists (--overwrite to redo)", flush=True)
                results.append((src, "skipped", ""))
                continue
            t0 = time.time()
            try:
                finals = process_file(src, root, steps, a, llm_session)
                results.append((src, "done", ", ".join(str(f) for f in finals) or "(nothing written)"))
                print(f"Done in {time.time() - t0:.0f}s: " + (", ".join(str(f) for f in finals) or
                                                               "nothing written"), flush=True)
            except (SystemExit, Exception) as e:  # noqa: BLE001 - one bad file mustn't stop the batch
                if isinstance(e, SystemExit) and e.code in (0, None):
                    raise
                msg = str(e.code if isinstance(e, SystemExit) else e)
                print(f"FAILED: {msg}\nIntermediates were kept, so re-running resumes this file.", flush=True)
                results.append((src, "failed", msg))
    finally:
        llm_session.release()

    if len(results) > 1:
        print("\n=== Summary", flush=True)
        for src, status, detail in results:
            print(f"  {status:8} {src.name}" + (f"  ->  {detail}" if detail else ""), flush=True)
    if any(s == "failed" for _, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
