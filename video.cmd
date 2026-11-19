@echo off
rem Subtitle (and optionally decensor) a Japanese video. Works from any directory; see "video --help".
setlocal
set "PY=%~dp0.venv-video\Scripts\python.exe"
if not exist "%PY%" (
  echo No video environment found. Run: uv run install.py --video 1>&2
  exit /b 1
)
set PYTHONIOENCODING=utf-8
"%PY%" "%~dp0video\pipeline.py" %*
