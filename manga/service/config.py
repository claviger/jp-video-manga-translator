"""Settings from a .env file plus real environment variables (real environment wins).

The .env file is the first of these that exists:
  $LJT_ENV_FILE, <manga root>/.env, <parent of manga root>/.env (the monorepo root when this folder is manga/).
See .env.example for every variable. Paths in settings are relative to the manga root unless absolute.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # the manga root (this folder, or <monorepo>/manga)
MIT_DIR = ROOT / "third_party" / "mit"

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _env_file() -> Path | None:
    explicit = os.environ.get("LJT_ENV_FILE")
    if explicit:
        return Path(explicit)
    for p in (ROOT / ".env", ROOT.parent / ".env"):
        if p.is_file():
            return p
    return None


def parse_env(text: str) -> dict[str, str]:
    """KEY=VALUE lines; '#' comments, blank lines, optional single or double quotes, optional 'export '.
    ${NAME} in an unquoted or double-quoted value expands from the real environment (so a key can stay in your
    shell profile: LLM_API_KEY=${MY_TOKEN}). Single-quoted values are literal."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if len(val) >= 2 and val[0] == val[-1] == "'":
            out[key] = val[1:-1]
            continue
        if len(val) >= 2 and val[0] == val[-1] == '"':
            val = val[1:-1].replace('\\"', '"')
        else:
            val = re.sub(r"\s+#.*$", "", val)  # trailing comment on an unquoted value
        out[key] = _REF.sub(lambda r: os.environ.get(r.group(1), out.get(r.group(1), "")), val)
    return out


ENV_FILE = _env_file()
if ENV_FILE is not None and ENV_FILE.is_file():
    for _k, _v in parse_env(ENV_FILE.read_text(encoding="utf-8-sig")).items():
        os.environ.setdefault(_k, _v)  # real environment overrides the file


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _path(name: str, default: Path) -> Path:
    v = _get(name)
    if not v:
        return default
    p = Path(v).expanduser()
    return p if p.is_absolute() else ROOT / p


# ---- manga service ----
CACHE_DIR = _path("MANGA_CACHE_DIR", ROOT / "cache")
MODEL_DIR = _path("MANGA_MODEL_DIR", ROOT / "models")
TOKEN_FILE = _path("MANGA_TOKEN_FILE", ROOT / "service" / "token.txt")
# Bind address is deliberately not configurable to anything but loopback (see server.py).
HOST = "127.0.0.1"
PORT = int(_get("MANGA_PORT", "8765"))
EXTENSION_ID = _get("MANGA_EXTENSION_ID")
MAX_UPLOAD_BYTES = int(_get("MANGA_MAX_UPLOAD_MB", "40")) * 1024 * 1024
MAX_PIXELS = int(_get("MANGA_MAX_PIXELS", str(40_000_000)))

# ---- LLM (same names as the video pipeline) ----
LLM_BASE_URL = _get("LLM_BASE_URL")
LLM_API_KEY = _get("LLM_API_KEY")
LLM_MODEL = _get("LLM_MODEL")
LLM_BACKEND = _get("LLM_BACKEND", "openai").lower()
LLM_USE_LOADED_MODEL = _get("LLM_USE_LOADED_MODEL") == "1"
LLM_REASONING_EFFORT = _get("LLM_REASONING_EFFORT")  # sent as reasoning_effort only when set
# Only used when the backend loads the model itself. A manga page plus the previous page's context is ~2k tokens,
# so 16k is plenty and leaves VRAM for detection/OCR/inpainting on the same GPU.
TABBY_LOAD_CTX = int(_get("TABBY_LOAD_CTX", "16384"))
TABBY_CACHE_MODE = _get("TABBY_CACHE_MODE", "8,8")
LMSTUDIO_LOAD_CTX = int(_get("LMSTUDIO_LOAD_CTX", "16384"))
LLM_BACKENDS = ("openai", "tabby", "lmstudio")

CONTEXT_PAIRS = 30  # max previous-page (ja, en) pairs sent as context

# Bump when anything that changes output (prompt, pipeline config) changes, so stale cache entries are ignored.
PIPELINE_VERSION = "5"


class ConfigError(RuntimeError):
    pass


def check_llm() -> None:
    """Raise ConfigError if the LLM settings can't work. Called by anything that translates (not by prefetch)."""
    where = f" (read {ENV_FILE})" if ENV_FILE else " (no .env file found)"
    missing = [n for n, v in (("LLM_BASE_URL", LLM_BASE_URL), ("LLM_MODEL", LLM_MODEL)) if not v]
    if missing:
        raise ConfigError(f"{' and '.join(missing)} not set{where}. Copy .env.example to .env and fill it in.")
    if LLM_BACKEND not in LLM_BACKENDS:
        raise ConfigError(f"LLM_BACKEND={LLM_BACKEND!r}; use one of {', '.join(LLM_BACKENDS)} (see .env.example).")
