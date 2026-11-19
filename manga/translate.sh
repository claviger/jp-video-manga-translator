#!/usr/bin/env bash
# Translate archives (.zip/.cbz/.rar/.cbr) or single images from any directory:
#   ./translate.sh "Some Book.cbr" [-o out.cbz] [--format jpg] ...   -> scripts.translate_archive
#   ./translate.sh page1.png page2.png                               -> scripts.translate_file
# Picks the tool from the first file argument's extension. Paths are resolved against your current directory.
# Settings come from .env (see .env.example) and the environment; the environment wins.
set -euo pipefail
HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
. "$HERE/scripts/find_python.sh"

module=scripts.translate_archive
for a in "$@"; do
  case "$a" in
    -*) continue ;;
    *.png|*.PNG|*.jpg|*.JPG|*.jpeg|*.JPEG|*.webp|*.WEBP|*.bmp|*.BMP|*.gif|*.GIF) module=scripts.translate_file ;;
  esac
  break
done
exec "$PY" -m "$module" "$@"
