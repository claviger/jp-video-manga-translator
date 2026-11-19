#!/usr/bin/env bash
# Start the local translation service on 127.0.0.1 (MANGA_PORT, default 8765).
# Settings come from .env (see .env.example) and the environment; the environment wins.
set -euo pipefail
HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
. "$HERE/scripts/find_python.sh"
cd "$REPO"
exec "$PY" -m service.server "$@"
