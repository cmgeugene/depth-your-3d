"""Upscale jobs: runs the upscale pipeline (upscale_core) inside the shared GPU queue.

The pipeline's frame caches live in the job directory and are removed when the job ends,
so a cancelled or failed job leaves nothing behind for long (the TTL cleaner catches the rest).
"""
import shutil
import subprocess

from . import config
from . import upscale_core as core
from .pipeline import Cancelled

core.TOOLS_DIR = config.UPSCALE_ENGINES_DIR

SCALES = (2, 4)
MODES = ("anime", "real")
BACKENDS = ("ai", "lanczos")
DENOISE = ("none", "low", "medium", "high")
QUALITIES = ("high", "balanced", "small")
# key -> (codec, extension, label)
FORMATS = {
    "mp4_h264": ("h264", ".mp4", "MP4 H.264"),
    "mp4_h265": ("h265", ".mp4", "MP4 H.265"),
    "mkv_h265": ("h265", ".mkv", "MKV H.265"),
}
MODE_LABEL = {"anime": "애니메이션", "real": "실사"}


def engines() -> dict:
    return {"realesrgan": bool(core.find_realesrgan()), "rife": bool(core.find_rife())}


def model_name(p: dict) -> str:
    if p["backend"] == "lanczos":
        return "Lanczos"
    return "realesr-animevideov3" if p["mode"] == "anime" else "realesrgan-x4plus"


def summary(media, p: dict) -> dict:
    fps = media.fps * (2 if p["interpolation"] else 1)
    return {
        "resolution": f"{media.width}×{media.height} → {media.width * p['scale']}×{media.height * p['scale']}",
        "frames": media.frames,
        "drawings": media.frames,
        "model": model_name(p),
        "outputs": [f"{p['scale']}배", MODE_LABEL[p["mode"]]]
                   + ([f"RIFE {fps:g}fps"] if p["interpolation"] else [])
                   + [FORMATS[p["format"]][2]],
    }


class _JobPipeline(core.VideoUpscalePipeline):
    """Keeps the frame cache in the job directory instead of the desktop app's user data folder."""

    def __init__(self, settings, work_dir, **kwargs):
        self._work_dir = work_dir
        super().__init__(settings, **kwargs)

    def _cache_dir(self):
        return self._work_dir


def run(job, media, progress) -> list[dict]:
    p = job.params["upscale"]
    codec, ext, label = FORMATS[p["format"]]
    name = f"{media.stem}_upscaled_{p['scale']}x{ext}"
    out = job.dir / name
    work = job.dir / "work"
    settings = core.UpscaleSettings(
        input_path=str(media.src),
        output_path=str(out),
        scale=p["scale"],
        content_mode=p["mode"],
        denoise=p["denoise"],
        backend=p["backend"],
        interpolation=p["interpolation"],
        codec=codec,
        quality=p["quality"],
        hardware_encode=True,
        keep_cache=False,
    )
    with open(job.dir / "upscale.log", "a", encoding="utf-8") as log:
        pipe = _JobPipeline(settings, work, log=lambda m: log.write(m + "\n"), progress=progress,
                            cancel_event=job.cancel)
        try:
            pipe.run()
        except core.PipelineCancelled:
            raise Cancelled()
        finally:
            shutil.rmtree(work, ignore_errors=True)
    if not p["audio"]:
        _strip_audio(out)
    return [{"name": name, "label": f"업스케일 영상 ({label})", "size": out.stat().st_size}]


def _strip_audio(path):
    """The core always keeps the source audio and subtitles; drop them when the user opted out."""
    tmp = path.with_name(f"noaudio_{path.name}")
    result = subprocess.run(
        [config.FFMPEG, "-v", "error", "-y", "-i", str(path), "-map", "0:v", "-map_metadata", "0",
         "-c", "copy", str(tmp)],
        capture_output=True,
    )
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("오디오 제거 실패: " + result.stderr.decode(errors="replace")[-600:])
    tmp.replace(path)
