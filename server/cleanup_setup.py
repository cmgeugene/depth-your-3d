"""One-time server setup for image cleanup:  python -m server.cleanup_setup  (setup-flux.bat runs it)

1. Nunchaku ships wheels only for some torch versions, while run.bat used to install the newest torch.
   If this venv's torch has no wheel, switches torch to TORCH_PIN (the depth/upscale code runs on it too).
2. Installs the Nunchaku wheel built for this venv's Python, torch and CUDA (it is not on PyPI).
3. Checks Hugging Face access to FLUX.1 [dev] (gated: the license must be accepted on the website).
4. Downloads the models so the first cleanup job doesn't wait ~17GB of downloads.
"""
import subprocess
import sys

NUNCHAKU_VERSION = "1.2.1"
RELEASES = "https://github.com/nunchux-ai/nunchaku/releases/download"
# torch (major.minor) and Python versions Nunchaku 1.2.1 has wheels for. Keep TORCH_PIN in sync with run.bat/run.sh.
NUNCHAKU_TORCH = ("2.8", "2.9", "2.10", "2.11")
NUNCHAKU_PYTHON = ((3, 10), (3, 11), (3, 12), (3, 13))
TORCH_PIN = "2.11"
COMPANIONS = {"torchvision": "0.26", "torchaudio": "2.11"}  # builds that match TORCH_PIN


def _torch_minor() -> str:
    import torch

    return ".".join(torch.__version__.split("+")[0].split(".")[:2])


def ensure_torch() -> bool:
    if sys.version_info[:2] not in NUNCHAKU_PYTHON:
        print(f"[ERROR] Python {sys.version_info.major}.{sys.version_info.minor}용 Nunchaku가 없습니다 (3.10~3.13 지원).")
        print("        .venv 폴더를 지우고 Python 3.12로 run.bat을 다시 실행하세요.")
        return False
    import torch

    if _torch_minor() in NUNCHAKU_TORCH and torch.version.cuda:
        print(f"[READY] torch {torch.__version__}")
        return True
    cu = "cu130" if (torch.version.cuda or "").startswith("13") else "cu128"
    why = "CUDA가 없는 CPU 전용 빌드라" if not torch.version.cuda else "용 Nunchaku가 없어서"
    print(f"[TORCH] torch {torch.__version__}{' ' if not torch.version.cuda else ''}{why} torch {TORCH_PIN}({cu})로 바꿉니다 (약 3GB).")
    print("        Depth/업스케일도 이 버전에서 그대로 동작합니다. 스튜디오 서버가 켜져 있으면 먼저 끄세요.")
    from importlib.metadata import PackageNotFoundError, version

    extra = []  # torchvision/torchaudio only work with the torch they were built for
    for pkg, ver in COMPANIONS.items():
        try:
            version(pkg)
            extra.append(f"{pkg}=={ver}.*")
        except PackageNotFoundError:
            pass
    # --force-reinstall: a CPU build of the same version would otherwise count as already installed.
    cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", f"torch=={TORCH_PIN}.*", *extra,
           "--index-url", f"https://download.pytorch.org/whl/{cu}"]
    if subprocess.run(cmd).returncode != 0:
        print("[ERROR] torch 교체 실패. 서버(python.exe)가 실행 중이면 끄고 다시 시도하세요.")
        return False
    return True


def _wheel_url() -> str:
    import torch

    torch_ver = _torch_minor()
    cuda = torch.version.cuda or ""
    cu = "13.0" if cuda.startswith("13") else "12.8"
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    plat = "win_amd64" if sys.platform == "win32" else "linux_x86_64"
    name = f"nunchaku-{NUNCHAKU_VERSION}%2Bcu{cu}torch{torch_ver}-{py}-{py}-{plat}.whl"
    return f"{RELEASES}/v{NUNCHAKU_VERSION}/{name}"


def install_nunchaku() -> bool:
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("nunchaku")  # the module itself has no __version__
        if installed.startswith(NUNCHAKU_VERSION) and f"torch{_torch_minor()}" in installed:
            print(f"[READY] nunchaku {installed}")
            return True
    except PackageNotFoundError:
        pass
    url = _wheel_url()
    print(f"[INSTALL] {url}")
    if subprocess.run([sys.executable, "-m", "pip", "install", url]).returncode != 0:
        print("[ERROR] Nunchaku 설치 실패. torch/Python 버전에 맞는 wheel이 없을 수 있습니다.")
        print(f"        릴리스 목록: https://github.com/nunchux-ai/nunchaku/releases/tag/v{NUNCHAKU_VERSION}")
        return False
    return True


def download_models() -> bool:
    from huggingface_hub import get_token, hf_hub_download, snapshot_download

    from . import cleanup, config

    if not get_token():
        print("[LOGIN] Hugging Face 로그인이 필요합니다. 아래 순서로 한 번만 하세요:")
        print(f"   1) https://huggingface.co/{config.FLUX_REPO} 에서 라이선스 동의 (Agree and access repository)")
        print("   2) https://huggingface.co/settings/tokens 에서 Read 토큰 만들기")
        print("   3) .venv\\Scripts\\hf auth login  을 실행하고 토큰 붙여넣기")
        print("   그 다음 setup-flux.bat을 다시 실행하세요.")
        return False
    try:
        print(f"[DOWNLOAD] {config.FLUX_REPO} (text encoders, VAE, scheduler — 약 10GB)")
        snapshot_download(config.FLUX_REPO, allow_patterns=[
            "model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
            "tokenizer/*", "tokenizer_2/*", "vae/*",
        ])
    except Exception as exc:
        if "gated" in str(exc).lower() or type(exc).__name__ in ("GatedRepoError", "HfHubHTTPError"):
            print(f"[ERROR] {config.FLUX_REPO} 접근 권한이 없습니다. 웹에서 라이선스에 동의했는지 확인하세요:")
            print(f"        https://huggingface.co/{config.FLUX_REPO}")
            return False
        raise
    path = cleanup._transformer_path()
    repo, file = path.rsplit("/", 1)
    print(f"[DOWNLOAD] {path} (약 7GB)")
    hf_hub_download(repo, file)
    return True


STEPS = {"torch": ensure_torch, "nunchaku": install_nunchaku, "models": download_models}

if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    if len(sys.argv) > 1:  # one step, in its own interpreter
        sys.exit(0 if STEPS[sys.argv[1]]() else 1)
    # Each step runs in a fresh interpreter so it sees what the previous one installed (a new torch, Nunchaku).
    ok = all(subprocess.run([sys.executable, "-m", "server.cleanup_setup", step]).returncode == 0 for step in STEPS)
    print("[DONE] 얼룩 정리 준비 완료." if ok else "[STOP] 위 안내를 따라 해결한 뒤 다시 실행하세요.")
    sys.exit(0 if ok else 1)
