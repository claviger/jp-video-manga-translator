"""Existing subtitles first: find English (or other) subtitles for a video on OpenSubtitles.com, sync them to the
file's audio with ffsubsync, and grade the sync against Silero VAD speech, before the pipeline falls back to
speech recognition and LLM translation.

Matching, best first:
  1. moviehash: OpenSubtitles' hash of this exact file (size + first/last 64 KiB). A match means someone uploaded
     subtitles for this very release, so they usually line up with little or no shift.
  2. identity: title/year/season/episode guessed from the filename with guessit, or an IMDb id from --imdb.
     Candidates must belong to the guessed title (and episode), then rank by fps match, release-name overlap,
     downloads and trust.

Each candidate costs one download from the account's daily quota (5/day per IP without a login, 20/day for a free
account), so at most OPENSUBTITLES_MAX_TRIES are tried. Search requests don't count against the quota.

What gets sent to OpenSubtitles: the API key, the login (only if set, only to /login), the moviehash, and the
guessed title/year/season/episode or IMDb id. The filename itself is not sent, only what guessit extracts from it.
"""
from __future__ import annotations

import base64
import difflib
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

APP_UA = "jp-video-manga-translator v0.2"  # OpenSubtitles asks for "<app> v<version>"; no personal data
API_BASE = "https://api.opensubtitles.com/api/v1"
SCRIPT_DIR = Path(__file__).resolve().parent
TOKEN_CACHE = SCRIPT_DIR.parent / ".cache" / "opensubtitles-token.json"  # gitignored; JWT is valid 24h
FFS_EXE = Path(sys.executable).parent / ("ffs.exe" if os.name == "nt" else "ffs")

# Sync quality (see grade_sync): share of subtitle time that falls on VAD speech, and how much better that is than
# the same subtitles shifted by a few seconds. Calibration in README "Existing subtitles".
MIN_SPEECH_OVERLAP = float(os.environ.get("OPENSUBTITLES_MIN_OVERLAP", "0.40"))
MIN_SYNC_MARGIN = float(os.environ.get("OPENSUBTITLES_MIN_MARGIN", "0.08"))
MAX_TRIES = int(os.environ.get("OPENSUBTITLES_MAX_TRIES", "3"))
GRADE_SHIFTS = (-20, -10, -5, -3, 3, 5, 10, 20)  # seconds
VAD_THRESHOLD = 0.5

# A JAV product code ("ABC-123", "SSIS-001.restored") parses as a TV episode in guessit and would match random
# shows, so for those only the moviehash is searched.
_PRODUCT_CODE = re.compile(r"^\[?[A-Za-z]{2,6}-?\d{2,5}\]?(?:[\s._-]|$)")


def log(msg: str):
    print(msg, flush=True)


# ---------------- identification ----------------
def moviehash(path: Path) -> str | None:
    """OpenSubtitles hash: file size plus the little-endian uint64 sums of the first and last 64 KiB, mod 2^64."""
    size = path.stat().st_size
    if size < 65536 * 2:
        return None
    h = size
    with path.open("rb") as f:
        for offset in (0, size - 65536):
            f.seek(offset)
            for (v,) in struct.iter_unpack("<Q", f.read(65536)):
                h = (h + v) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


@dataclass
class Identity:
    title: str = ""
    year: int | None = None
    season: int | None = None
    episode: int | None = None
    kind: str = ""          # movie | episode | ""
    imdb_id: int | None = None
    release_tokens: set = field(default_factory=set)

    def describe(self) -> str:
        if self.imdb_id:
            return f"IMDb tt{self.imdb_id:07d}"
        if not self.title:
            return "the filename (not searched: no usable title)"
        s = self.title
        if self.year:
            s += f" ({self.year})"
        if self.kind == "episode":
            s += f" S{self.season or 0:02}E{self.episode or 0:02}" if self.season is not None else f" E{self.episode}"
        return s


def identify(path: Path, imdb: str | None = None) -> Identity:
    stem = path.stem.removesuffix(".restored")
    tokens = {t for t in re.split(r"[^a-z0-9]+", stem.lower()) if len(t) > 1}
    if imdb:
        num = int(re.sub(r"\D", "", imdb) or 0)
        return Identity(imdb_id=num or None, release_tokens=tokens)
    if _PRODUCT_CODE.match(stem):
        return Identity(release_tokens=tokens)
    try:
        from guessit import guessit
    except ImportError:
        log("Note: guessit isn't installed, so only the moviehash is searched (run install.py --video).")
        return Identity(release_tokens=tokens)
    g = guessit(path.name)
    title = str(g.get("title") or "")
    ep = g.get("episode")
    return Identity(title=title, year=g.get("year"), season=g.get("season"),
                    episode=ep if isinstance(ep, int) else None,
                    kind=str(g.get("type") or ""), release_tokens=tokens)


# ---------------- API client ----------------
class OpenSubtitles:
    def __init__(self, api_key: str, username: str = "", password: str = ""):
        self.api_key, self.username, self.password = api_key, username, password
        self.base = API_BASE
        self.token: str | None = None
        self._last = 0.0

    def _headers(self) -> dict:
        h = {"Api-Key": self.api_key, "User-Agent": APP_UA, "Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _request(self, method: str, path: str, **kw) -> requests.Response:
        for attempt in range(4):
            wait = 0.25 - (time.time() - self._last)  # stay under the 5 requests/s limit
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = requests.request(method, self.base + path, headers=self._headers(), timeout=30, **kw)
            except requests.RequestException as e:
                if attempt == 3:
                    raise
                log(f"  OpenSubtitles: {type(e).__name__}, retrying")
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 502:
                time.sleep(float(r.headers.get("retry-after") or 2 ** attempt))
                continue
            return r
        return r

    # Login only when a download needs it; the token is cached on disk because /login allows 30 calls an hour.
    def login(self):
        if self.token or not (self.username and self.password):
            return
        try:
            cached = json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
            if cached.get("user") == self.username and _jwt_exp(cached["token"]) > time.time() + 600:
                self.token, self.base = cached["token"], cached["base"]
                return
        except (OSError, ValueError, KeyError):
            pass
        r = self._request("POST", "/login", json={"username": self.username, "password": self.password})
        if r.status_code == 401:
            raise OSError("OpenSubtitles login failed (401): check OPENSUBTITLES_USERNAME / OPENSUBTITLES_PASSWORD")
        r.raise_for_status()
        data = r.json()
        self.token = data["token"]
        self.base = f"https://{data.get('base_url') or 'api.opensubtitles.com'}/api/v1"
        allowed = (data.get("user") or {}).get("allowed_downloads")
        log(f"Logged in to OpenSubtitles ({allowed} downloads/day)")
        TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_CACHE.write_text(json.dumps({"user": self.username, "token": self.token, "base": self.base}),
                               encoding="utf-8")
        try:
            os.chmod(TOKEN_CACHE, 0o600)
        except OSError:
            pass

    def search(self, **params) -> list:
        # The API redirects unless params are sorted, lowercase and without defaults, so build the query that way.
        clean = {k: str(v).lower() for k, v in params.items() if v not in (None, "")}
        r = self._request("GET", "/subtitles", params=sorted(clean.items()))
        if r.status_code == 401 or r.status_code == 403:
            raise OSError(f"OpenSubtitles rejected the API key (HTTP {r.status_code}): check OPENSUBTITLES_API_KEY")
        r.raise_for_status()
        return r.json().get("data") or []

    def download(self, file_id: int) -> str:
        self.login()
        r = self._request("POST", "/download", json={"file_id": file_id})
        if r.status_code == 406 or (r.status_code == 429 and "remaining" in r.text):
            raise QuotaExceeded(_message(r))
        if not r.ok:
            raise OSError(f"OpenSubtitles download failed (HTTP {r.status_code}): {_message(r)}")
        data = r.json()
        log(f"  downloaded file {file_id}; {data.get('remaining')} downloads left today "
            f"(resets in {data.get('reset_time')})")
        f = requests.get(data["link"], headers={"User-Agent": APP_UA}, timeout=60)
        f.raise_for_status()
        return f.content.decode("utf-8-sig", errors="replace")


class QuotaExceeded(OSError):
    pass


def _message(r: requests.Response) -> str:
    try:
        return str(r.json().get("message") or r.text[:200])
    except ValueError:
        return r.text[:200]


def _jwt_exp(token: str) -> float:
    payload = token.split(".")[1]
    return float(json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))).get("exp", 0))


# ---------------- candidates ----------------
@dataclass
class Candidate:
    file_id: int
    release: str
    language: str
    score: float
    hash_match: bool
    attrs: dict


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _title_matches(guess: str, fd: dict) -> bool:
    g = _norm(guess)
    if not g:
        return False
    for name in (fd.get("parent_title"), fd.get("title"), fd.get("movie_name")):
        n = _norm(name)
        if n and (n == g or (len(g) >= 4 and (g in n or n in g))
                  or difflib.SequenceMatcher(None, g, n).ratio() >= 0.85):
            return True
    return False


def _belongs(ident: Identity, a: dict) -> bool:
    """Reject results for a different title or episode: a wrong download wastes quota and can't sync."""
    fd = a.get("feature_details") or {}
    if ident.imdb_id:
        return fd.get("imdb_id") == ident.imdb_id  # the episode's own id, not the show's
    if not _title_matches(ident.title, fd):
        return False
    if ident.kind == "episode":
        if ident.episode is not None and fd.get("episode_number") not in (None, ident.episode):
            return False
        if ident.season is not None and fd.get("season_number") not in (None, ident.season):
            return False
    elif ident.year and fd.get("year") and abs(int(fd["year"]) - int(ident.year)) > 1:
        return False
    return True


def rank(results: list, ident: Identity, video_fps: float | None, lang: str) -> list[Candidate]:
    out, seen = [], set()
    for item in results:
        a = item.get("attributes") or {}
        files = a.get("files") or []
        if (len(files) != 1 or a.get("foreign_parts_only") or a.get("machine_translated")
                or (a.get("language") or "").lower() != lang.lower()):
            continue  # multi-CD sets, forced-only and machine translations aren't what we want
        fid = files[0].get("file_id")
        if not fid or fid in seen:
            continue
        hash_match = bool(a.get("moviehash_match"))
        if not hash_match and not _belongs(ident, a):
            continue
        seen.add(fid)
        rel_tokens = {t for t in re.split(r"[^a-z0-9]+", (a.get("release") or "").lower()) if len(t) > 1}
        overlap = len(rel_tokens & ident.release_tokens) / max(1, len(rel_tokens | ident.release_tokens))
        fps = a.get("fps") or 0
        fps_ok = bool(video_fps and fps and abs(fps - video_fps) < 0.02)
        score = (100 * hash_match + 3 * fps_ok + 4 * overlap + math.log10(1 + (a.get("download_count") or 0))
                 + 0.5 * bool(a.get("from_trusted")) - 0.3 * bool(a.get("hearing_impaired"))
                 - 0.5 * bool(a.get("ai_translated")))
        out.append(Candidate(fid, a.get("release") or "", a.get("language") or lang, score, hash_match, a))
    return sorted(out, key=lambda c: -c.score)


def find_candidates(api: OpenSubtitles, video: Path, ident: Identity, lang: str,
                    video_fps: float | None) -> list[Candidate]:
    results = []
    mh = moviehash(video)
    if mh:
        results += api.search(languages=lang, moviehash=mh, moviehash_match="only")
    n_hash = len(results)
    if ident.imdb_id:
        results += api.search(languages=lang, imdb_id=ident.imdb_id)
    elif ident.title:
        if ident.kind == "episode":
            results += api.search(languages=lang, query=ident.title, type="episode",
                                  season_number=ident.season, episode_number=ident.episode)
        else:
            results += api.search(languages=lang, query=ident.title, type="movie", year=ident.year)
    cands = rank(results, ident, video_fps, lang)
    log(f"OpenSubtitles [{lang}]: {n_hash} moviehash result(s), {len(results) - n_hash} for {ident.describe()}; "
        f"{len(cands)} usable candidate(s)")
    for c in cands[:MAX_TRIES]:
        a = c.attrs
        log(f"  {'HASH ' if c.hash_match else ''}{c.release or '(no release name)'}  "
            f"[fps {a.get('fps') or '?'}, {a.get('download_count')} downloads"
            f"{', HI' if a.get('hearing_impaired') else ''}{', trusted' if a.get('from_trusted') else ''}]")
    return cands


# ---------------- sync and grading ----------------
_TAG = re.compile(r"<[^>]+>|\{[^}]*\}")
_NONSPEECH = re.compile(r"\[[^\]]*\]|\([^)]*\)|♪[^♪]*♪?|#[^#]*#?")


def srt_intervals(text: str) -> list[tuple[float, float]]:
    """(start, end) of cues that carry dialogue. Sound descriptions ([door slams]) and lyrics (♪...) are skipped,
    since there is no speech under them to sync or grade against."""
    out = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = block.strip().splitlines()
        idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if idx is None:
            continue
        try:
            a, b = [_parse_ts(x) for x in lines[idx].split("-->")]
        except ValueError:
            continue
        body = _NONSPEECH.sub("", _TAG.sub("", " ".join(lines[idx + 1:])))
        if re.search(r"\w", body) and b > a:
            out.append((a, b))
    return out


def _parse_ts(s: str) -> float:
    s = s.strip().split(" ")[0].replace(".", ",")
    h, m, rest = s.split(":")
    sec, _, ms = rest.partition(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int((ms or "0")[:3].ljust(3, "0")) / 1000


def speech_mask(audio, sr: int = 16000, rate: int = 100):
    """Silero VAD speech as a 0/1 array at `rate` Hz. Independent of ffsubsync's WebRTC VAD, so it can grade it."""
    import numpy as np
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    mask = np.zeros(int(len(audio) / sr * rate) + 1, dtype=bool)
    for t in get_speech_timestamps(audio, VadOptions(threshold=VAD_THRESHOLD, min_silence_duration_ms=300)):
        mask[int(t["start"] / sr * rate):int(t["end"] / sr * rate) + 1] = True
    return mask


@dataclass
class Grade:
    overlap: float   # share of subtitle time on detected speech
    baseline: float  # same, averaged over the subtitles shifted by GRADE_SHIFTS
    recall: float    # share of detected speech under some subtitle
    cues: int

    @property
    def margin(self) -> float:
        return self.overlap - self.baseline

    @property
    def ok(self) -> bool:
        return self.cues >= 20 and self.overlap >= MIN_SPEECH_OVERLAP and self.margin >= MIN_SYNC_MARGIN

    def __str__(self) -> str:
        return (f"speech overlap {self.overlap:.0%} vs {self.baseline:.0%} when shifted (margin {self.margin:+.0%}), "
                f"covers {self.recall:.0%} of detected speech, {self.cues} cues")


def grade_sync(intervals, speech, rate: int = 100) -> Grade:
    import numpy as np

    subs = np.zeros(len(speech), dtype=bool)
    for a, b in intervals:
        subs[int(a * rate):int(b * rate)] = True
    total = subs.sum()
    if not total or not speech.any():
        return Grade(0.0, 0.0, 0.0, len(intervals))
    overlap = float((subs & speech).sum() / total)
    shifted = []
    for s in GRADE_SHIFTS:
        k = s * rate
        sh = np.zeros_like(subs)
        if k > 0:
            sh[k:] = subs[:-k]
        else:
            sh[:k] = subs[-k:]
        if sh.any():
            shifted.append(float((sh & speech).sum() / sh.sum()))
    return Grade(overlap, float(np.mean(shifted)) if shifted else 0.0, float((subs & speech).sum() / speech.sum()),
                 len(intervals))


def ffsubsync(reference: Path, srt_in: Path, srt_out: Path, serialize: bool = False) -> dict:
    """Run ffsubsync; returns the score/offset/framerate it logs. `--vad webrtc` is explicit because the default
    (subs_then_webrtc) would sync to a subtitle track embedded in the reference instead of its audio."""
    cmd = [str(FFS_EXE), str(reference), "-i", str(srt_in), "-o", str(srt_out), "--max-offset-seconds", "120"]
    if reference.suffix.lower() not in (".npz", ".npy"):
        cmd += ["--vad", "webrtc"]
    if serialize:
        cmd.append("--serialize-speech")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    logtext = r.stdout + r.stderr
    info = {"ok": r.returncode == 0 and srt_out.exists()}
    for key, pat in (("score", r"score: (-?[\d.]+)"), ("offset", r"offset seconds: (-?[\d.]+)"),
                     ("scale", r"framerate scale factor: (-?[\d.]+)")):
        m = re.findall(pat, logtext)
        if m:
            info[key] = float(m[-1])
    if not info["ok"]:
        info["error"] = logtext.strip()[-600:]
    return info


@dataclass
class Found:
    srt_text: str
    release: str
    language: str
    grade: Grade


def fetch_and_sync(api: OpenSubtitles, cands: list[Candidate], workdir: Path, reference: Path, speech,
                   label: str) -> Found | None:
    """Download candidates best-first, sync each, return the first that grades well."""
    npz = reference.with_suffix(".npz")
    for n, c in enumerate(cands[:MAX_TRIES], 1):
        log(f"Trying {label} candidate {n}/{min(len(cands), MAX_TRIES)}: {c.release or c.file_id}")
        try:
            text = api.download(c.file_id)
        except QuotaExceeded as e:
            log(f"  OpenSubtitles download quota used up: {e}")
            return None
        raw = workdir / f"{c.file_id}.raw.srt"
        raw.write_text(text, encoding="utf-8")
        before = grade_sync(srt_intervals(text), speech)
        log(f"  as downloaded: {before}")
        out = workdir / f"{c.file_id}.synced.srt"
        t0 = time.time()
        info = ffsubsync(npz if npz.exists() else reference, raw, out, serialize=not npz.exists())
        if not info["ok"]:
            log(f"  ffsubsync failed: {info.get('error', '')[-300:]}")
            continue
        synced = out.read_text(encoding="utf-8-sig", errors="replace")
        after = grade_sync(srt_intervals(synced), speech)
        log(f"  synced in {time.time() - t0:.0f}s (offset {info.get('offset', 0):+.2f}s, framerate x"
            f"{info.get('scale', 1):.3f}, ffsubsync score {info.get('score', float('nan')):.0f}): {after}")
        best_text, best = (synced, after) if after.overlap >= before.overlap else (text, before)
        if best is before:
            log("  the downloaded timing grades better than the synced one; keeping it as downloaded")
        if best.ok:
            return Found(best_text, c.release, c.language, best)
        log(f"  sync below threshold (need overlap >= {MIN_SPEECH_OVERLAP:.0%} and margin >= "
            f"{MIN_SYNC_MARGIN:.0%}); trying the next candidate")
    return None


def audio_language(video: Path) -> str | None:
    """ISO 639-2 language tag of the first audio stream, e.g. 'jpn', or None if untagged."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream_tags=language",
                        "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    tag = r.stdout.strip().lower()
    return tag if re.fullmatch(r"[a-z]{3}", tag) and tag not in ("und", "mul", "zxx", "mis") else None


def video_fps(video: Path) -> float | None:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate",
                        "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    try:
        num, _, den = r.stdout.strip().partition("/")
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None


def lang_codes(code: str) -> tuple[str, str, str]:
    """'en' / 'eng' / 'pt-br' -> (OpenSubtitles code, ISO 639-2 for the MKV tag, English name)."""
    from babelfish import Language

    code = code.strip().lower()
    try:
        lang = Language.fromalpha3b(code) if len(code) == 3 else Language.fromietf(code)
    except Exception:
        return code, code, code
    os_code = lang.alpha2 if len(code) == 3 else code
    return os_code, lang.alpha3, lang.name


def find_existing_subtitles(video: Path, langs: list[str], imdb: str | None = None) -> dict[str, Found]:
    """Look up and sync subtitles for each language in `langs` (OpenSubtitles codes). Returns {lang: Found} for the
    ones that synced well. Raises nothing for network/API trouble: it logs and returns what it has."""
    key = os.environ.get("OPENSUBTITLES_API_KEY", "")
    if not key:
        log("OpenSubtitles: no OPENSUBTITLES_API_KEY in .env, skipping the existing-subtitles lookup.")
        return {}
    if not FFS_EXE.exists():
        log(f"OpenSubtitles: {FFS_EXE} not found (run install.py --video); skipping the lookup.")
        return {}
    api = OpenSubtitles(key, os.environ.get("OPENSUBTITLES_USERNAME", ""),
                        os.environ.get("OPENSUBTITLES_PASSWORD", ""))
    ident = identify(video, imdb)
    fps = video_fps(video)
    t0 = time.time()
    found: dict[str, Found] = {}
    try:
        per_lang = {lang: find_candidates(api, video, ident, lang, fps) for lang in langs}
    except (OSError, requests.RequestException) as e:
        log(f"OpenSubtitles lookup failed: {e}")
        return {}
    if not any(per_lang.values()):
        return {}
    from faster_whisper import decode_audio

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref = tmp / "reference.wav"  # extracted once; ffsubsync serializes its speech track next to it
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(ref)],
                       check=True)
        log("Detecting speech for the sync check (Silero VAD)")
        speech = speech_mask(decode_audio(str(ref), sampling_rate=16000))
        for lang, cands in per_lang.items():
            if not cands:
                continue
            try:
                f = fetch_and_sync(api, cands, tmp, ref, speech, lang)
            except (OSError, requests.RequestException) as e:
                log(f"OpenSubtitles [{lang}] failed: {e}")
                f = None
            if f:
                found[lang] = f
    log(f"OpenSubtitles step took {time.time() - t0:.0f}s")
    return found
