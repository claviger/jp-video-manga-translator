#!/usr/bin/env python3
"""
Existing subtitles first (OpenSubtitles, optional) -> decensor (Jasna, optional) -> transcribe Japanese
(faster-whisper, with vocal separation) -> translate to English (any OpenAI-compatible LLM) -> mux an MKV with
English and Japanese subtitle tracks. When OpenSubtitles has subtitles that sync well to this file's audio,
transcription and translation are skipped (see opensubs.py).

Usage (from anywhere; `video.sh` / `video.cmd` in the repo root run this with the right venv):
  video.sh input.mp4                                  # full pipeline: decensor, subtitle, mux
  video.sh input.mp4 --skip-decensor                  # subtitle the video as is
  video.sh input.mp4 --restored "E:/x/input_restored.mkv"  # already decensored elsewhere (e.g. the Jasna GUI)
  video.sh X.restored.mp4 --skip-decensor            # resume from an earlier Jasna run
  video.sh X.mp4 --skip-decensor --transcribe-only   # Japanese subtitles only; no LLM needed
  video.sh X.mp4 --skip-decensor --model other-model --no-mux  # A/B a translation model
  video.sh "Some.Movie.2001.1080p.mkv"               # existing English subs if OpenSubtitles has good ones
  video.sh Show.S01E02.mkv --imdb tt1234567 --subs-lang fr   # name the episode, look for French subs
  video.sh X.mp4 --no-opensubtitles                  # always transcribe and translate

If <input>.ja.srt already exists it is reused instead of re-transcribing. Delete it to force a fresh transcript.
A trailing ".restored" in the input name is ignored when naming outputs (X.restored.mp4 -> X.ja.srt).
The cached ja.srt is raw Whisper output; cleanup filters (e.g. counting hallucinations) run on every load.

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
        sys.exit("Decensoring needs Jasna: set JASNA_EXE in .env to jasna.exe (see README, 'Jasna'), or pass "
                 "--skip-decensor to subtitle the video as it is.")
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
                 "To make Japanese subtitles only, pass --transcribe-only.")
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


# ---------------- mux ----------------
def subtitle_codecs(video: Path) -> list[str]:
    """Codec names of the subtitle streams already in `video`, in order."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries", "stream=codec_name",
                        "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    return [c.strip() for c in r.stdout.splitlines() if c.strip()]


def mux(video: Path, subs, out: Path, keep_existing: bool = False):
    """subs: [(srt_path, language, title), ...]; the first track is the default.
    keep_existing: also copy the video's own subtitle tracks, after ours and not default. Their language and title
    tags carry over. MP4 text subtitles (mov_text) can't go into MKV as is, so those become SRT; everything else,
    including bitmap PGS/VobSub tracks, is copied untouched."""
    existing = subtitle_codecs(video) if keep_existing else []
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-i", video]
    for path, _, _ in subs:
        cmd += ["-i", path]
    cmd += ["-map", "0:v", "-map", "0:a?"]
    for k in range(len(subs)):
        cmd += ["-map", str(k + 1)]
    if existing:
        cmd += ["-map", "0:s"]
    cmd += ["-c:v", "copy", "-c:a", "copy"]
    for k, (_, lang, title) in enumerate(subs):
        cmd += [f"-c:s:{k}", "srt", f"-metadata:s:s:{k}", f"language={lang}", f"-metadata:s:s:{k}", f"title={title}",
                f"-disposition:s:{k}", "default" if k == 0 else "0"]
    for j, codec in enumerate(existing, start=len(subs)):
        cmd += [f"-c:s:{j}", "srt" if codec == "mov_text" else "copy", f"-disposition:s:{j}", "0"]
    if keep_existing:
        print(f"Keeping {len(existing)} subtitle track(s) already in the video" if existing else
              "Note: --keep-subs, but the video has no subtitle tracks of its own", flush=True)
    run(cmd + [out])


def existing_subtitles(src: Path, base: str, args) -> list:
    """Existing subtitles first: [(srt, iso639-2, title), ...] with the SUBS_LANG track first, or [] to fall back
    to speech recognition. Synced files are cached as <name>.<lang>.opensubtitles.srt and reused on later runs."""
    import opensubs

    primary, primary_tag, primary_name = opensubs.lang_codes(args.subs_lang)
    langs = [primary]
    orig = None if args.no_original_subs else opensubs.audio_language(src)
    if orig:
        o_code, o_tag, o_name = opensubs.lang_codes(orig)
        if o_code != primary:
            langs.append(o_code)
    names = {primary: (primary_tag, primary_name)}
    if len(langs) > 1:
        names[langs[1]] = (o_tag, o_name)
    cached = {lang: src.with_name(f"{base}.{lang}.opensubtitles.srt") for lang in langs}
    if cached[primary].exists():
        print(f"Reusing synced OpenSubtitles subtitles: {cached[primary]}", flush=True)
    else:
        found = opensubs.find_existing_subtitles(src, langs, args.imdb)
        if primary not in found:
            if os.environ.get("OPENSUBTITLES_API_KEY"):
                print(f"No well-synced {primary_name} subtitles on OpenSubtitles; transcribing instead.", flush=True)
            return []
        for lang, f in found.items():
            cached[lang].write_text(f.srt_text, encoding="utf-8")
            print(f"Wrote {cached[lang]} ({f.release}; {f.grade})", flush=True)
    return [(cached[lang], names[lang][0], f"{names[lang][1]} (OpenSubtitles)")
            for lang in langs if cached[lang].exists()]


def main():
    global REASONING_EFFORT
    ap = argparse.ArgumentParser(description="Decensor (optional), transcribe, translate and subtitle a Japanese video.")
    ap.add_argument("input", type=Path)
    ap.add_argument("--skip-decensor", action="store_true",
                    help="don't run Jasna; use --restored if given, else the input file as is")
    ap.add_argument("--restored", type=Path,
                    help="an already-decensored video (e.g. from the Jasna GUI)")
    ap.add_argument("--model", help="translation model id (default: LLM_MODEL from .env). With --model, EN subs go "
                                    "to <name>.en.<model>.srt so A/B runs don't collide")
    ap.add_argument("--no-dossier", action="store_true",
                    help="translate without the context brief (<name>.dossier.txt); EN subs get a .nodossier tag")
    ap.add_argument("--title", help="video title for the context brief (default: guessed from a sibling file "
                                    "like '[CODE] Title.mp4')")
    ap.add_argument("--no-mux", action="store_true", help="stop after writing the English SRT")
    ap.add_argument("--keep-subs", action="store_true",
                    help="keep the video's own subtitle tracks in the output, after the new ones (default: drop "
                         "them, since you're subtitling it because they weren't good enough)")
    ap.add_argument("--en-srt", action="append", metavar="PATH[=TITLE]",
                    help="mux this existing English SRT (e.g. from an A/B run) instead of translating; repeat it "
                         "to mux several English tracks for comparison, e.g. --en-srt a.srt='English (Artemis)'")
    ap.add_argument("--output", type=Path, help="output MKV (default: <name>.decensored.subbed.mkv after "
                                                "decensoring or with --restored, else <name>.subbed.mkv)")
    ap.add_argument("--reasoning", default=REASONING_EFFORT,
                    help="reasoning_effort sent to the model, e.g. none/low/high; any value other than none adds "
                         f".reason-<value> to the EN filename (default: LLM_REASONING_EFFORT, now {REASONING_EFFORT})")
    ap.add_argument("--transcribe-only", action="store_true",
                    help="stop after transcription, cleanup and the coverage report")
    ap.add_argument("--verbose-cleanup", action="store_true", help="print every line the cleanup filters drop")
    ap.add_argument("--no-vocals", action="store_true",
                    help="skip vocal separation; transcribe the original audio only")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help=f"lines per translation request (default: {BATCH_SIZE})")
    ap.add_argument("--no-opensubtitles", action="store_true",
                    help="don't look for existing subtitles on OpenSubtitles; always transcribe and translate")
    ap.add_argument("--subs-lang", default=SUBS_LANG,
                    help="language of the default subtitle track to look for on OpenSubtitles, e.g. en, pt-br "
                         f"(default: SUBS_LANG from .env, now {SUBS_LANG})")
    ap.add_argument("--imdb", metavar="ID", help="IMDb id of the movie, or of the episode itself (not the show), "
                                                 "e.g. tt1234567, for when the filename doesn't identify it")
    ap.add_argument("--no-original-subs", action="store_true",
                    help="with OpenSubtitles, don't also look for subtitles in the audio's own language")
    args = ap.parse_args()

    src = args.input.resolve()
    # "ABC-123.restored.mp4" shares its subtitle files with "ABC-123.mp4"
    base = src.stem.removesuffix(".restored")
    ja_srt = src.with_name(base + ".ja.srt")  # raw Whisper output, cached between runs
    dossier_path = src.with_name(base + ".dossier.txt")  # context brief, cached and hand-editable
    en_tag = "." + re.sub(r"[^\w.-]+", "_", args.model) if args.model else ""
    en_tag += ".nodossier" if args.no_dossier else ""
    en_srt = src.with_name(f"{base}.en{en_tag}.srt")
    # Tag the EN file only when --reasoning overrides .env, so A/B runs don't overwrite the normal translation
    if args.reasoning and args.reasoning != "none" and args.reasoning != REASONING_EFFORT:
        en_srt = en_srt.with_name(en_srt.name.removesuffix(".srt") + f".reason-{args.reasoning}.srt")
    REASONING_EFFORT = args.reasoning

    # Existing subtitles first. Skipped when the run is clearly about our own transcript or translation
    # (--transcribe-only, --en-srt, --model A/B runs). The synced file is cached like ja.srt: delete it to search again.
    os_tracks = []  # [(srt_path, iso639-2, title)] from OpenSubtitles
    if not (args.no_opensubtitles or args.transcribe_only or args.en_srt or args.model):
        os_tracks = existing_subtitles(src, base, args)
    # Fail on a missing LLM setting now, not after hours of decensoring and transcription
    llm = None if (args.transcribe_only or args.en_srt or os_tracks) else make_llm(args.model)

    skip_decensor = args.skip_decensor
    if not skip_decensor and not args.restored and not JASNA_EXE:
        print("Note: JASNA_EXE isn't set in .env, so the video is subtitled as is, without decensoring.", flush=True)
        skip_decensor = True
    decensored = not skip_decensor or args.restored or src.stem.endswith(".restored")
    final = src.with_name(base + (".decensored.subbed.mkv" if decensored else ".subbed.mkv"))

    if args.restored:
        restored = args.restored.resolve()
    elif skip_decensor:
        restored = src
    else:
        restored = src.with_name(src.stem + ".restored.mp4")
        free_vram_for_jasna()
        decensor(src, restored)

    if not restored.exists():
        sys.exit(f"Video not found: {restored}")

    if os_tracks:
        print("Using existing subtitles; skipping speech recognition and translation.", flush=True)
        if args.no_mux:
            return
        final = args.output.resolve() if args.output else final
        with tempfile.TemporaryDirectory() as tmp:
            subs = list(os_tracks)
            if ja_srt.exists() and not any(lang == "jpn" for _, lang, _ in subs):
                ja_clean = Path(tmp) / "ja.srt"  # an earlier transcript of this file: add it, cleaned, as before
                write_srt(clean_cues(read_srt(ja_srt)), ja_clean, "ja")
                subs.append((ja_clean, "jpn", "Japanese (ASR)"))
            mux(restored, subs, final, args.keep_subs)
        print("Done:", final)
        return

    if ja_srt.exists():
        print(f"Reusing existing transcript: {ja_srt}", flush=True)
        cues = read_srt(ja_srt)
    else:
        vocals = None
        if args.no_vocals:
            pass
        elif SEPARATOR_EXE.exists():
            vocals = src.with_name(base + ".vocals.flac")
            free_vram_for_jasna()  # get the LLM out of VRAM where that's ours to do (see the function)
            separate_vocals(restored, vocals)
        else:
            print(f"Note: {SEPARATOR_EXE} not found; transcribing the original audio only.", flush=True)
        cues = transcribe(restored, vocals)
        write_srt(cues, ja_srt, "ja")
    raw_count = len(cues)
    cues = clean_cues(cues, verbose=args.verbose_cleanup)
    coverage_report(cues, media_duration(restored))
    print(f"{len(cues)} cues to translate ({raw_count - len(cues)} removed by cleanup)", flush=True)
    if args.transcribe_only:
        return

    en_tracks = []  # [(path, title)]
    if args.en_srt:
        for spec in args.en_srt:
            path_s, _, title = spec.partition("=")
            path = Path(path_s).resolve()
            en = read_srt(path)
            # The EN file must come from this same cleaned transcript, cue for cue
            if len(en) != len(cues) or any(abs(e["start"] - c["start"]) > 0.01 for e, c in zip(en, cues)):
                sys.exit(f"{path} has {len(en)} cues that don't line up with the {len(cues)} cleaned JA cues; "
                         "it was made from a different transcript or cleanup.")
            en_tracks.append((path, title or "English (machine translation)"))
            print(f"Using existing English subtitles: {path}", flush=True)
    else:
        print(f"Translating with {llm.provider}: {llm.model}", flush=True)
        we_loaded = False
        if llm.backend == "lmstudio":
            ensure_model_loaded(llm.model)
        elif llm.backend == "tabby":
            we_loaded = tabby_ensure_loaded(llm)
        try:
            dossier = None
            if not args.no_dossier:
                title = args.title if args.title is not None else find_title(src, base)
                dossier = load_or_build_dossier(dossier_path, cues, title, llm)
            translate(cues, llm, args.batch_size, dossier)
        finally:
            if we_loaded:  # leave the shared Tabby server as we found it
                tabby_unload(llm)
        write_srt(cues, en_srt, "en")
        print("Wrote", en_srt, flush=True)
        en_tracks.append((en_srt, "English (machine translation)"))
    if args.no_mux:
        return

    final = args.output.resolve() if args.output else final
    # Mux the cleaned JA cues so all tracks line up; ja.srt on disk stays the raw cache.
    with tempfile.TemporaryDirectory() as tmp:
        ja_clean = Path(tmp) / "ja.srt"
        write_srt(cues, ja_clean, "ja")
        subs = [(p_, "eng", t) for p_, t in en_tracks] + [(ja_clean, "jpn", "Japanese (ASR)")]
        mux(restored, subs, final, args.keep_subs)
    print("Done:", final)


if __name__ == "__main__":
    main()
