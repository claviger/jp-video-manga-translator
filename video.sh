#!/usr/bin/env bash
# Subtitle (and optionally decensor) a Japanese video. Works from any directory; see `video.sh --help`.
set -euo pipefail
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PY="$REPO/.venv-video/Scripts/python.exe"
[ -x "$PY" ] || PY="$REPO/.venv-video/bin/python"
[ -x "$PY" ] || { echo "No video environment found. Run: uv run install.py --video" >&2; exit 1; }
export PYTHONIOENCODING=utf-8
exec "$PY" "$REPO/video/pipeline.py" "$@"
