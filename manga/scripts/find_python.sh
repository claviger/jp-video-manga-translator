# Sourced by run_service.sh and translate.sh. Sets REPO (the manga root) and PY (the venv's python).
# The venv is <manga root>/.venv (standalone checkout) or <monorepo root>/.venv-manga (made by the installer).
# Expects HERE to be set to the directory of the calling script before sourcing.
REPO="$(cd "$HERE" && pwd)"
PY=""
for cand in "$REPO/.venv" "$REPO/../.venv-manga"; do
  for exe in "$cand/Scripts/python.exe" "$cand/bin/python"; do
    if [ -x "$exe" ]; then PY="$exe"; break 2; fi
  done
done
if [ -z "$PY" ]; then
  echo "No Python venv found at $REPO/.venv or $(cd "$REPO/.." && pwd)/.venv-manga. Run the installer first." >&2
  exit 1
fi
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
