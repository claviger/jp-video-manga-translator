"""Japanese -> English translation of one manga page through an OpenAI-compatible endpoint.

Ported from the video subtitle pipeline (jav_sub_pipeline.py): adult-genre framing, the "you/I, not he/she" pronoun rule,
one JSON batch per request, tolerant JSON parsing that salvages well-formed objects, and a per-id retry for
anything the model skipped or mangled.

LLM_BACKEND picks how the model is managed (same rules as the video pipeline):
  * openai (default): just call /chat/completions; never load or unload anything.
  * tabby: TabbyAPI may be shared. If the configured model is loaded, use it; if a different one is loaded,
    refuse (never swap someone else's model) unless LLM_USE_LOADED_MODEL=1; if nothing is loaded, load ours with
    TABBY_LOAD_CTX / TABBY_CACHE_MODE and unload it on shutdown (only if we loaded it).
  * lmstudio: if the model isn't loaded, `lms load <model> --gpu max -c 16384`; on shutdown unload only what we
    loaded; never touch other loaded models.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

from . import config

log = logging.getLogger("mlt.llm")

SYSTEM_PROMPT = (
    "You translate Japanese manga dialogue into natural, idiomatic English for speech bubbles. "
    # Genre framing, as in Sugoi's own system prompt and the JAV pipeline: it makes the model pick the sexual
    # sense of ambiguous words instead of the everyday one.
    "This is an adult (hentai / ero) manga. When a word, idiom or phrase has both an everyday and a sexual "
    "meaning, prefer the sexual meaning if the context allows it (e.g. イく = to cum, 出す = to cum/ejaculate, "
    "先走り = precum, 濡れる = to get wet). Translate explicit language explicitly, without softening or censoring "
    "it, but don't add sexual content to lines that are clearly ordinary conversation. "
    "Keep each line short enough to fit its speech bubble and preserve tone and register. "
    "The Japanese comes from OCR of the page and may contain errors (wrong kana, missing small tsu, "
    "merged lines); infer the most likely meaning. Lines are listed in reading order (right to left, top to "
    "bottom). Translate each numbered line on its own: one bubble may continue the sentence of the previous "
    "one, so keep each part with its own id and never move words from one id to another. "
    "Characters mostly speak directly to each other: when Japanese omits the subject or object, use "
    "\"you\"/\"I\"/\"we\", not \"he\"/\"him\"/\"she\", unless the line clearly refers to someone else. "
    "Sound effects and moans (e.g. ドキッ, はぁはぁ, あんっ) become short English equivalents (\"Ba-dump\", "
    "\"Hah... hah...\", \"Ahn!\"). "
    'Return ONLY a JSON array like [{"id": 3, "en": "..."}], one object per input id, nothing else.'
)

_JSON_ITEM = re.compile(r'\{\s*"id"\s*:\s*"?(\d+)"?\s*,\s*"en"\s*:\s*("(?:[^"\\]|\\.)*")\s*\}')


def parse_json_array(text: str) -> list:
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    # One malformed object breaks the whole array: salvage well-formed objects so only broken ids get retried.
    items = []
    for m in _JSON_ITEM.finditer(text):
        try:
            items.append({"id": int(m.group(1)), "en": json.loads(m.group(2))})
        except json.JSONDecodeError:
            pass
    return items


def to_map(items) -> dict[int, str]:
    out = {}
    for item in items:
        try:
            out[int(item["id"])] = str(item["en"]).strip()
        except (KeyError, TypeError, ValueError):
            pass
    return out


class LLMError(RuntimeError):
    """A condition the user has to resolve (wrong model loaded, bad key, server down, bad settings)."""


def _prefer_ipv4(base: str) -> str:
    """Some LAN hostnames resolve to an IPv6 link-local address first while the LLM server listens on IPv4 only, so
    every new connection stalls for seconds before falling back. For plain-http URLs, resolve the host to IPv4 once
    and use the address (https keeps the name so certificate checks still work)."""
    u = urlsplit(base)
    host = u.hostname or ""
    if u.scheme != "http":
        return base
    try:
        ipaddress.ip_address(host)
        return base
    except ValueError:
        pass
    try:
        ip = socket.getaddrinfo(host, u.port or 80, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
    except OSError:
        return base
    netloc = ip + (f":{u.port}" if u.port else "")
    return urlunsplit((u.scheme, netloc, u.path, u.query, u.fragment))


@dataclass
class LLM:
    """OpenAI-compatible chat client with backend-specific model management (see the module docstring)."""
    base: str = ""
    model: str = ""
    backend: str = ""
    loaded_by_us: bool = False
    active_model: str | None = None
    _last_check: float = field(default=0.0, repr=False)
    http: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self):
        try:
            config.check_llm()
        except config.ConfigError as e:
            raise LLMError(str(e)) from e
        self.base = (self.base or config.LLM_BASE_URL).rstrip("/")
        self.model = self.model or config.LLM_MODEL
        self.backend = self.backend or config.LLM_BACKEND
        resolved = _prefer_ipv4(self.base)
        if resolved != self.base:
            log.info("LLM %s -> %s (IPv4)", self.base, resolved)
        self.base = resolved

    def _headers(self) -> dict:
        k = config.LLM_API_KEY
        return {"Authorization": f"Bearer {k}"} if k else {}

    def _get(self, path: str) -> requests.Response:
        try:
            r = self.http.get(self.base + path, headers=self._headers(), timeout=10)
        except requests.RequestException as e:
            raise LLMError(f"Can't reach the LLM server at {self.base}: {e}") from e
        if r.status_code in (401, 403):
            raise LLMError(f"The LLM server at {self.base} rejected the key. Check LLM_API_KEY.")
        return r

    # ---- model management ----

    def ensure_ready(self) -> str:
        """Return the model id to send requests to. Never swaps or unloads a model someone else loaded."""
        # Re-check at most every 30s; someone may load/unload between pages.
        if self.active_model and time.time() - self._last_check < 30:
            return self.active_model
        ready = {"openai": self._ready_openai, "tabby": self._ready_tabby, "lmstudio": self._ready_lmstudio}
        model = ready[self.backend]()
        self._last_check = time.time()
        self.active_model = model
        return model

    def _ready_openai(self) -> str:
        """Plain OpenAI-compatible server: nothing is loaded or unloaded. If /models lists models, check ours is
        among them so a typo gives one clear error instead of a failure per page."""
        r = self._get("/models")
        if r.ok:
            try:
                ids = [m.get("id") for m in r.json().get("data", []) if isinstance(m, dict)]
            except (ValueError, AttributeError):
                ids = []
            if ids and self.model not in ids:
                shown = ", ".join(map(str, ids[:10])) + (" ..." if len(ids) > 10 else "")
                raise LLMError(f"{self.base} doesn't serve LLM_MODEL={self.model!r}. It serves: {shown}")
        return self.model

    # TabbyAPI: one model at a time, possibly shared with other people.
    def _tabby_loaded(self) -> str | None:
        r = self._get("/model")
        return r.json().get("id") if r.ok else None

    def _ready_tabby(self) -> str:
        loaded = self._tabby_loaded()
        if loaded == self.model:
            return loaded
        if loaded:
            if config.LLM_USE_LOADED_MODEL:
                if loaded != self.active_model:
                    log.warning("TabbyAPI has %s loaded, not %s; using it (LLM_USE_LOADED_MODEL=1)", loaded,
                                self.model)
                return loaded
            self.active_model = None
            raise LLMError(
                f"TabbyAPI has {loaded} loaded, not {self.model}. Tabby may be shared, so this won't swap models. "
                f"Load {self.model} yourself, or set LLM_USE_LOADED_MODEL=1 to use whatever is loaded.")
        self.loaded_by_us = False  # if we loaded it before, someone unloaded it since
        body = {"model_name": self.model, "max_seq_len": config.TABBY_LOAD_CTX,
                "cache_size": config.TABBY_LOAD_CTX, "cache_mode": config.TABBY_CACHE_MODE}
        log.info("Nothing loaded on TabbyAPI; loading %s (ctx %d, cache %s)", self.model, config.TABBY_LOAD_CTX,
                 config.TABBY_CACHE_MODE)
        t0 = time.time()
        try:
            r = self.http.post(self.base + "/model/load", headers=self._headers(), json=body, stream=True,
                               timeout=900)
            text = "".join(c.decode("utf-8", "replace") for c in r.iter_content(chunk_size=None))
        except requests.RequestException as e:
            raise LLMError(f"TabbyAPI load request failed: {e}") from e
        if not r.ok or self._tabby_loaded() != self.model:
            raise LLMError(f"TabbyAPI couldn't load {self.model} (HTTP {r.status_code}): {text[-500:]}")
        self.loaded_by_us = True
        log.info("Loaded %s in %.0fs", self.model, time.time() - t0)
        return self.model

    # LM Studio: several models can be loaded at once; load ours if it isn't, never touch the others.
    @staticmethod
    def _lms(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
        exe = shutil.which("lms") or str(Path.home() / ".lmstudio" / "bin" / ("lms.exe" if os.name == "nt" else "lms"))
        if not Path(exe).exists():
            raise LLMError("LLM_BACKEND=lmstudio needs LM Studio's 'lms' CLI on PATH (or in ~/.lmstudio/bin).")
        try:
            return subprocess.run([exe, *args], capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise LLMError(f"'lms {' '.join(args)}' timed out after {timeout}s") from e

    def _lms_loaded(self) -> set[str]:
        p = self._lms("ps", "--json")
        try:
            items = json.loads(p.stdout or "[]")
        except json.JSONDecodeError as e:
            raise LLMError(f"Couldn't read 'lms ps --json' output: {p.stdout[:200]!r} {p.stderr[:200]!r}") from e
        names = set()
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict):
                names.update(str(it[k]) for k in ("identifier", "modelKey", "path") if it.get(k))
        return names

    def _ready_lmstudio(self) -> str:
        if self.model in self._lms_loaded():
            return self.model
        self.loaded_by_us = False
        log.info("%s isn't loaded in LM Studio; loading it (ctx %d)", self.model, config.LMSTUDIO_LOAD_CTX)
        t0 = time.time()
        p = self._lms("load", self.model, "--gpu", "max", "-c", str(config.LMSTUDIO_LOAD_CTX), "-y", timeout=900)
        if p.returncode != 0 or self.model not in self._lms_loaded():
            raise LLMError(f"LM Studio couldn't load {self.model}: {(p.stderr or p.stdout)[-500:]}")
        self.loaded_by_us = True
        log.info("Loaded %s in %.0fs", self.model, time.time() - t0)
        return self.model

    def unload_if_ours(self):
        """Unload the model on shutdown, but only if this process loaded it and it's still the same model."""
        if not self.loaded_by_us:
            return
        try:
            if self.backend == "tabby":
                if self._tabby_loaded() == self.model:
                    r = self.http.post(self.base + "/model/unload", headers=self._headers(), timeout=120)
                    if r.ok:
                        log.info("Unloaded %s from TabbyAPI", self.model)
                    else:
                        log.warning("TabbyAPI unload returned HTTP %s", r.status_code)
            elif self.backend == "lmstudio":
                if self.model in self._lms_loaded():
                    p = self._lms("unload", self.model)
                    if p.returncode == 0:
                        log.info("Unloaded %s from LM Studio", self.model)
                    else:
                        log.warning("lms unload failed: %s", (p.stderr or p.stdout)[-300:])
        except (requests.RequestException, LLMError) as e:
            log.warning("Couldn't unload %s: %s", self.model, e)
        self.loaded_by_us = False

    def status(self) -> tuple[str | None, str | None]:
        """(what's loaded/served, error) for health checks. Never loads anything."""
        try:
            if self.backend == "tabby":
                return self._tabby_loaded(), None
            if self.backend == "lmstudio":
                return (self.model if self.model in self._lms_loaded() else None), None
            return self._ready_openai(), None
        except LLMError as e:
            return None, str(e)

    # ---- chat ----

    def chat(self, user_msg: str, system: str = SYSTEM_PROMPT) -> str:
        model = self.ensure_ready()
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            "temperature": 0.3,
            "max_tokens": 2048,
        }
        if config.LLM_REASONING_EFFORT:
            body["reasoning_effort"] = config.LLM_REASONING_EFFORT
        url = self.base + "/chat/completions"
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                r = self.http.post(url, headers=self._headers(), json=body, timeout=300)
            except requests.RequestException as e:
                if attempt == attempts:
                    raise LLMError(f"LLM request failed: {e}") from e
                time.sleep(2 ** attempt)
                continue
            if (r.status_code == 429 or r.status_code >= 500) and attempt < attempts:
                time.sleep(2 ** attempt)
                continue
            break
        if r.status_code in (401, 403):
            raise LLMError(f"The LLM server at {self.base} rejected the key. Check LLM_API_KEY.")
        if r.status_code == 404:
            raise LLMError(f"{self.base} returned 404 for model {model!r}. Is LLM_MODEL right, and is it served?")
        if not r.ok:
            raise LLMError(f"LLM server returned {r.status_code} for {model}: {r.text[:300]}")
        j = r.json()
        log.info("llm usage: %s", j.get("usage"))
        out = j["choices"][0]["message"].get("content") or ""
        return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()


def translate_page(llm: LLM, texts: list[str], prev: list[tuple[str, str]] | None = None) -> list[str]:
    """Translate one page's bubbles (already in reading order) in one JSON batch.

    prev: (japanese, english) pairs from the previous page, sent as context only.
    """
    if not texts:
        return []
    ids = list(range(1, len(texts) + 1))
    msg = ""
    if prev:
        msg += "Previous page, for context only (do not translate):\n"
        msg += "\n".join(f"{ja} => {en}" for ja, en in prev[-config.CONTEXT_PAIRS:]) + "\n\n"
    msg += "Translate these speech bubbles:\n" + "\n".join(f"{i}: {t}" for i, t in zip(ids, texts))

    t0 = time.time()
    reply = llm.chat(msg)
    got = to_map(parse_json_array(reply))
    if sum(1 for i in ids if got.get(i)) < len(ids) / 2:  # badly mangled: retry the whole batch once
        log.warning("batch reply unusable (%d chars: %r...), retrying batch", len(reply), reply[:200])
        got.update(to_map(parse_json_array(llm.chat(msg))))
    out = []
    for i, ja in zip(ids, texts):
        if not got.get(i):  # model skipped or mangled this id: retry it alone
            log.info("retrying id %d alone", i)
            retry = to_map(parse_json_array(llm.chat(f"Translate these speech bubbles:\n{i}: {ja}")))
            got[i] = retry.get(i) or ""
            if not got[i]:
                log.warning("id %d left untranslated: %r", i, ja)
        out.append(got[i])
    log.info("translated %d bubbles in %.1fs", len(texts), time.time() - t0)
    return out
