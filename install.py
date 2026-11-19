#!/usr/bin/env python3
"""Set up jp-video-manga-translator after `git clone`.

    uv run install.py              # asks which parts to install
    uv run install.py --video      # video subtitles (and decensoring via Jasna, installed separately)
    uv run install.py --manga      # manga translator (Chrome extension service + archive tool)
    uv run install.py --all
    uv run install.py --video --cuda cu126 --no-models

What it does, per part:
  video: .venv-video (Python 3.12): PyTorch (CUDA), faster-whisper, audio-separator; downloads the Whisper
         large-v3 model (~3 GB) and the vocal-separation model (~900 MB) unless --no-models.
  manga: initializes the manga-image-translator submodule, .venv-manga (Python 3.12): PyTorch (CUDA) and its
         requirements; downloads its detection/OCR/inpainting models (~700 MB) unless --no-models.
  both:  copies .env.example to .env if there's no .env yet, then checks what's still missing.

It does NOT install: the translation LLM or its server, Jasna, ffmpeg, 7-Zip, NVIDIA drivers. See README.md.
Needs only the standard library, so any Python 3.10+ (or `uv run`) can run it.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = os.name == "nt"
PY = "3.12"  # 3.13 lacks Windows wheels for editdistance (manga); 3.14 lacks them for much of the torch stack
TORCH = "2.14.1"
TORCHVISION = "0.29.1"
# torch 2.14.1 is published for cu130, cu126 and cpu (no cu128 build).
# cu130 is required for RTX 50-series (Blackwell) and needs a recent driver (580 series or newer, as far as we
# know; check NVIDIA's CUDA 13 release notes if unsure). cu126 covers older GPUs on older drivers.
CUDA_CHOICES = ("auto", "cu130", "cu126", "cpu")
TORCH_INDEX = "https://download.pytorch.org/whl/{}"

VIDEO_VENV = ROOT / ".venv-video"
MANGA_VENV = ROOT / ".venv-manga"
MANGA_DIR = ROOT / "manga"
MIT_DIR = MANGA_DIR / "third_party" / "mit"
SEPARATOR_MODEL = "vocals_mel_band_roformer.ckpt"
WHISPER_REPO = "Systran/faster-whisper-large-v3"  # what faster-whisper's "large-v3" resolves to


def say(msg=""):
    print(msg, flush=True)


def step(msg):
    say(f"\n== {msg}")


def run(cmd, **kw):
    cmd = [str(c) for c in cmd]
    say("> " + " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if WIN else "bin/python")


def need(tool: str, hint: str):
    if not shutil.which(tool):
        sys.exit(f"{tool} not found on PATH. {hint}")


def nvidia_driver() -> tuple[int, str] | None:
    """(major version, GPU name) from nvidia-smi, or None if there's no NVIDIA driver."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=driver_version,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
        ver, name = [s.strip() for s in out.split(",", 1)]
        return int(ver.split(".")[0]), name
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def pick_cuda(choice: str) -> str:
    if choice != "auto":
        return choice
    drv = nvidia_driver()
    if not drv:
        say("No NVIDIA driver found (nvidia-smi missing): installing CPU-only PyTorch. Everything will be very slow;"
            " an NVIDIA GPU is strongly recommended. Pass --cuda to override.")
        return "cpu"
    major, name = drv
    if major >= 580:
        say(f"GPU: {name}, driver {major}: using CUDA 13 (cu130) PyTorch.")
        return "cu130"
    blackwell = re.search(r"RTX\s*50\d\d|Blackwell", name, re.I)
    if blackwell:
        sys.exit(f"GPU: {name}, driver {major}. RTX 50-series cards need the CUDA 13 build of PyTorch, which needs a "
                 "newer NVIDIA driver (580 series or later). Update the driver, then run this again.")
    say(f"GPU: {name}, driver {major}: using CUDA 12.6 (cu126) PyTorch. Update to a 580+ driver to use cu130.")
    return "cu126"


def make_venv(venv: Path):
    if venv_python(venv).exists():
        say(f"{venv.name} exists; updating it in place (delete the folder for a clean install).")
    else:
        run(["uv", "venv", "--python", PY, venv])


def uv_pip(venv: Path, *args):
    run(["uv", "pip", "install", "--python", venv_python(venv), *args])


def install_torch(venv: Path, cuda: str, vision: bool):
    pkgs = [f"torch=={TORCH}"] + ([f"torchvision=={TORCHVISION}"] if vision else [])
    uv_pip(venv, *pkgs, "--index-url", TORCH_INDEX.format(cuda))


# ---------------- video ----------------
def install_video(cuda: str, models: bool):
    step("Video: Python environment")
    make_venv(VIDEO_VENV)
    install_torch(VIDEO_VENV, cuda, vision=False)
    # The torch pin stays in the second install, so the resolver can't swap in a CPU build from PyPI.
    uv_pip(VIDEO_VENV, "-r", ROOT / "video" / "requirements.txt", f"torch=={TORCH}",
           "--extra-index-url", TORCH_INDEX.format(cuda), "--index-strategy", "unsafe-best-match")
    if not models:
        say("Skipping model downloads (--no-models); they download on first use instead.")
        return
    step("Video: models")
    sep_dir = ROOT / "models" / "separator"
    sep_dir.mkdir(parents=True, exist_ok=True)
    exe = VIDEO_VENV / ("Scripts/audio-separator.exe" if WIN else "bin/audio-separator")
    run([exe, "--download_model_only", "-m", SEPARATOR_MODEL, "--model_file_dir", sep_dir])
    run([venv_python(VIDEO_VENV), "-c",
         f"from huggingface_hub import snapshot_download; print(snapshot_download('{WHISPER_REPO}'))"],
        env={**os.environ, "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"})


# ---------------- manga ----------------
def mit_requirements() -> Path:
    """manga-image-translator's requirements minus what can't install from wheels on Windows (see README)."""
    src = MIT_DIR / "requirements.txt"
    keep = [ln for ln in src.read_text(encoding="utf-8").splitlines()
            if not re.match(r"\s*(pydensecrf|--extra-index-url)", ln, re.I)]
    out = ROOT / "models" / "mit-requirements.filtered.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(keep) + "\n", encoding="utf-8")
    return out


def install_manga(cuda: str, models: bool):
    step("Manga: manga-image-translator submodule")
    need("git", "Install Git (https://git-scm.com) and run this again.")
    run(["git", "-C", ROOT, "submodule", "update", "--init", "--recursive"])
    step("Manga: Python environment")
    make_venv(MANGA_VENV)
    install_torch(MANGA_VENV, cuda, vision=True)
    uv_pip(MANGA_VENV, "-r", mit_requirements(), "-r", MANGA_DIR / "requirements.txt",
           f"torch=={TORCH}", f"torchvision=={TORCHVISION}",
           "--extra-index-url", TORCH_INDEX.format(cuda), "--index-strategy", "unsafe-best-match")
    if not models:
        say("Skipping model downloads (--no-models); they download on the first translated page instead.")
        return
    step("Manga: models")
    run([venv_python(MANGA_VENV), "-m", "scripts.prefetch_models"], cwd=MANGA_DIR,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})


# ---------------- checks ----------------
def read_env() -> dict:
    env = {}
    p = ROOT / ".env"
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                v = v.split(" #", 1)[0].strip().strip("'\"")
                env[k.strip()] = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), v)
    # Real environment variables override the file, same as in the tools
    for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "LLM_BACKEND", "JASNA_EXE"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def final_checks(video: bool, manga: bool):
    step("Configuration")
    env_file = ROOT / ".env"
    if not env_file.exists():
        shutil.copyfile(ROOT / ".env.example", env_file)
        say(f"Created {env_file} from .env.example. Edit it before the first run.")
    env = read_env()
    todo = []
    if not env.get("LLM_BASE_URL") or not env.get("LLM_MODEL"):
        todo.append("Set LLM_BASE_URL and LLM_MODEL in .env (README: 'Translation model'). Without them you can "
                    "still make Japanese subtitles (video --transcribe-only).")
    if video:
        if not shutil.which("ffmpeg"):
            todo.append("Install ffmpeg and put it on PATH (README: 'Other tools').")
        jasna = env.get("JASNA_EXE", "")
        if not jasna:
            todo.append("Optional: install Jasna and set JASNA_EXE in .env to decensor videos (README: 'Jasna'). "
                        "Without it, videos are subtitled as they are.")
        elif not Path(jasna).is_file():
            todo.append(f"JASNA_EXE in .env points to {jasna}, which doesn't exist.")
    if manga:
        seven = shutil.which("7z") or (WIN and Path(r"C:\Program Files\7-Zip\7z.exe").exists())
        if not seven:
            todo.append("Optional: install 7-Zip to read .rar/.cbr archives (README: 'Other tools').")
        todo.append("Load the Chrome extension from manga/extension (README: 'Chrome extension').")
    say("\nInstalled. Still to do:" if todo else "\nInstalled; nothing else to do.")
    for t in todo:
        say(f"  - {t}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="store_true", help="install the video subtitle pipeline")
    ap.add_argument("--manga", action="store_true", help="install the manga translator")
    ap.add_argument("--all", action="store_true", help="install both")
    ap.add_argument("--cuda", choices=CUDA_CHOICES, default="auto",
                    help="PyTorch build: auto (from your NVIDIA driver), cu130, cu126 or cpu")
    ap.add_argument("--no-models", action="store_true", help="don't pre-download models")
    args = ap.parse_args()

    video, manga = args.video or args.all, args.manga or args.all
    if not (video or manga):
        if not sys.stdin.isatty():
            sys.exit("Pass --video, --manga or --all.")
        ans = input("Install which parts? [v]ideo, [m]anga, [b]oth (default b): ").strip().lower() or "b"
        video, manga = ans in ("v", "b", "video", "both"), ans in ("m", "b", "manga", "both")
        if not (video or manga):
            sys.exit("Nothing selected.")
    need("uv", "Install uv first: https://docs.astral.sh/uv/getting-started/installation/")
    if not WIN:
        say("Note: only Windows has been tested. Linux should work but is untested; please report problems.")
    cuda = pick_cuda(args.cuda)
    if video:
        install_video(cuda, not args.no_models)
    if manga:
        install_manga(cuda, not args.no_models)
    final_checks(video, manga)


if __name__ == "__main__":
    main()
