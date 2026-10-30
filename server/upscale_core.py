# -*- coding: utf-8 -*-
"""Core pipeline for the local AI video upscaler.

The module deliberately has no third-party Python dependency.  Heavy lifting is
performed by ffmpeg, Real-ESRGAN-ncnn-vulkan and (optionally)
RIFE-ncnn-vulkan, so it is easy to run on a normal Windows Python install.

Originally video_upscaler_core.py from the anc_video-upscale-tool desktop app
(commit 4a64ea6). That repository is archived; this file is now the only copy,
so edit it here. Server-specific behaviour stays in server/upscale.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from functools import lru_cache
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable


APP_DIR = Path(__file__).resolve().parent


def _default_user_data_dir() -> Path:
    override = os.environ.get("VIDEO_UPSCALER_DATA_DIR")
    if override:
        return Path(override).expanduser()
    local_root = Path(os.environ.get("LOCALAPPDATA", str(APP_DIR)))
    local_data = local_root / "VideoUpscalerAI"
    try:
        str(local_data).encode("ascii")
        return local_data
    except UnicodeEncodeError:
        # Older ncnn Windows binaries cannot reliably open model/cache paths
        # containing Korean or other non-ASCII account names.
        public_root = Path(os.environ.get("PUBLIC", r"C:\Users\Public"))
        return public_root / "VideoUpscalerAI"


USER_DATA_DIR = _default_user_data_dir()
TOOLS_DIR = USER_DATA_DIR / "tools"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts", ".mts"}
TEXT_SUBTITLE_CODECS = {"ass", "ssa", "subrip", "srt", "mov_text", "text", "webvtt"}


class PipelineCancelled(RuntimeError):
    pass


class PipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class UpscaleSettings:
    input_path: str
    output_path: str
    scale: int = 2
    content_mode: str = "anime"
    denoise: str = "none"
    backend: str = "ai"
    interpolation: bool = False
    codec: str = "h264"
    quality: str = "balanced"
    hardware_encode: bool = True
    keep_cache: bool = True
    tile_size: int = 0


@dataclass
class VideoInfo:
    duration: float
    fps: float
    width: int
    height: int
    frame_count: int
    video_codec: str
    audio_streams: int
    subtitle_streams: list[dict]


def find_program(name: str, extra_roots: Iterable[Path] = ()) -> str:
    found = shutil.which(name)
    if found:
        return found
    candidates: list[Path] = []
    for root in extra_roots:
        if root.exists():
            candidates.extend(root.rglob(name))
    return str(candidates[0]) if candidates else ""


def find_ffmpeg() -> str:
    return find_program("ffmpeg.exe") or find_program("ffmpeg")


def find_ffprobe() -> str:
    return find_program("ffprobe.exe") or find_program("ffprobe")


def find_realesrgan() -> str:
    root = TOOLS_DIR / "realesrgan"
    if not root.exists():
        return ""
    for candidate in root.rglob("realesrgan-ncnn-vulkan.exe"):
        if (candidate.parent / "models" / "realesrgan-x4plus.param").is_file():
            return str(candidate)
    return ""


def find_rife() -> str:
    root = TOOLS_DIR / "rife"
    if not root.exists():
        return ""
    for candidate in root.rglob("rife-ncnn-vulkan.exe"):
        if (candidate.parent / "rife-v4.6").is_dir():
            return str(candidate)
    return ""


def detect_nvidia_gpu() -> str:
    command = shutil.which("nvidia-smi")
    if not command:
        return ""
    try:
        result = subprocess.run(
            [command, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode == 0:
            return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return ""


@lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """Return True only when the installed FFmpeg and driver can encode."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg or not detect_nvidia_gpu():
        return False
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                # Some recent NVENC generations reject 64x64 frames even when
                # the driver and encoder are otherwise healthy.
                "color=size=256x256:rate=1:duration=1",
                "-frames:v",
                "1",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _rate_to_float(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(path: str, ffprobe: str | None = None) -> VideoInfo:
    ffprobe = ffprobe or find_ffprobe()
    if not ffprobe:
        raise PipelineError("ffprobe를 찾지 못했습니다. FFmpeg를 먼저 설치해 주세요.")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode != 0:
        raise PipelineError(f"영상 정보를 읽지 못했습니다.\n{result.stderr.strip()}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if not videos:
        raise PipelineError("입력 파일에 영상 트랙이 없습니다.")
    video = videos[0]
    duration = float(data.get("format", {}).get("duration") or 0.0)
    fps = _rate_to_float(video.get("avg_frame_rate")) or _rate_to_float(video.get("r_frame_rate")) or 30.0
    try:
        frame_count = int(video.get("nb_frames") or 0)
    except ValueError:
        frame_count = 0
    if frame_count <= 0:
        frame_count = max(1, round(duration * fps))
    subtitles = [stream for stream in streams if stream.get("codec_type") == "subtitle"]
    return VideoInfo(
        duration=duration,
        fps=fps,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        frame_count=frame_count,
        video_codec=str(video.get("codec_name") or "unknown"),
        audio_streams=sum(stream.get("codec_type") == "audio" for stream in streams),
        subtitle_streams=subtitles,
    )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _parse_ffmpeg_time(value: str) -> float:
    match = re.match(r"(\d+):(\d+):(\d+(?:\.\d+)?)", value)
    if not match:
        return 0.0
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))


class VideoUpscalePipeline:
    def __init__(
        self,
        settings: UpscaleSettings,
        log: Callable[[str], None] | None = None,
        progress: Callable[[float, str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.settings = settings
        self.log = log or (lambda _message: None)
        self.progress = progress or (lambda _value, _message: None)
        self.cancel_event = cancel_event or threading.Event()
        self.current_process: subprocess.Popen[str] | None = None
        self.ffmpeg = find_ffmpeg()
        self.ffprobe = find_ffprobe()
        self.realesrgan = find_realesrgan()
        self.rife = find_rife()
        self.info: VideoInfo | None = None
        self.cache_dir = self._cache_dir()
        self.input_frames = self.cache_dir / "frames_input"
        self.upscaled_frames = self.cache_dir / "frames_upscaled"
        self.interpolated_frames = self.cache_dir / "frames_interpolated"
        self.manifest_path = self.cache_dir / "manifest.json"

    def _cache_dir(self) -> Path:
        source = Path(self.settings.input_path).resolve()
        stat = source.stat()
        signature = {
            "cache_schema": 3,
            "path": str(source).lower(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "scale": self.settings.scale,
            "mode": self.settings.content_mode,
            "denoise": self.settings.denoise,
            "backend": self.settings.backend,
        }
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        # Several ncnn Windows builds cannot open paths containing Korean or
        # other non-ASCII characters. Keep every AI-facing path in LocalAppData.
        return USER_DATA_DIR / "cache" / digest

    def cancel(self) -> None:
        self.cancel_event.set()
        process = self.current_process
        if process and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise PipelineCancelled("사용자가 작업을 중지했습니다. 캐시는 유지됩니다.")

    def _run_process(
        self,
        command: list[str],
        cwd: Path | None = None,
        line_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._check_cancelled()
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        self.current_process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            startupinfo=startupinfo,
        )
        recent: list[str] = []
        assert self.current_process.stdout is not None
        for raw_line in self.current_process.stdout:
            line = raw_line.strip()
            if line:
                recent.append(line)
                recent = recent[-20:]
                if line_callback:
                    line_callback(line)
            if self.cancel_event.is_set():
                self.cancel()
                break
        return_code = self.current_process.wait()
        self.current_process = None
        self._check_cancelled()
        if return_code != 0:
            details = "\n".join(recent[-20:])
            raise PipelineError(f"외부 프로그램 실행이 실패했습니다 (코드 {return_code}).\n{details}")

    def _write_manifest(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "settings": asdict(self.settings),
            "video": asdict(self.info) if self.info else None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def validate(self) -> None:
        source = Path(self.settings.input_path)
        output = Path(self.settings.output_path)
        if not source.is_file():
            raise PipelineError("입력 영상 파일을 찾을 수 없습니다.")
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            self.log("주의: 일반적이지 않은 확장자입니다. FFmpeg가 지원하면 계속 처리합니다.")
        if source.resolve() == output.resolve():
            raise PipelineError("입력 파일과 출력 파일은 같을 수 없습니다.")
        if self.settings.scale not in {2, 4}:
            raise PipelineError("확대 배율은 2배 또는 4배여야 합니다.")
        if not self.ffmpeg or not self.ffprobe:
            raise PipelineError("FFmpeg/ffprobe를 찾지 못했습니다.")
        if self.settings.backend == "ai" and not self.realesrgan:
            raise PipelineError("Real-ESRGAN AI 엔진이 없습니다. 먼저 'AI 엔진 설치'를 실행해 주세요.")
        if self.settings.interpolation and not self.rife:
            raise PipelineError("RIFE 프레임 보간 엔진이 없습니다. 먼저 'AI 엔진 설치'를 실행해 주세요.")
        output.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        self.validate()
        self.progress(0.01, "영상 정보 분석")
        self.info = probe_video(self.settings.input_path, self.ffprobe)
        self._write_manifest()
        self.log(
            f"입력: {self.info.width}x{self.info.height}, {self.info.fps:.3f}fps, "
            f"{format_duration(self.info.duration)}, 약 {self.info.frame_count:,}프레임"
        )
        self.log(f"예상 출력: {self.info.width * self.settings.scale}x{self.info.height * self.settings.scale}")

        self._extract_frames()
        self._upscale_frames()
        final_frames = self.upscaled_frames
        final_fps = self.info.fps
        if self.settings.interpolation:
            final_frames = self._interpolate_frames()
            final_fps *= 2
        output = self._encode(final_frames, final_fps)
        self.progress(1.0, "완료")
        if not self.settings.keep_cache:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.log("임시 캐시를 정리했습니다.")
        return output

    def _extract_frames(self) -> None:
        assert self.info is not None
        marker = self.input_frames / ".complete"
        existing = len(list(self.input_frames.glob("*.png"))) if self.input_frames.exists() else 0
        if marker.exists() and existing > 0:
            self.log(f"1/4 프레임 추출: 캐시 재사용 ({existing:,}장)")
            self.progress(0.15, "추출 프레임 캐시 재사용")
            return
        if self.input_frames.exists():
            shutil.rmtree(self.input_frames)
        self.input_frames.mkdir(parents=True, exist_ok=True)
        filters = []
        denoise_filters = {
            "none": "",
            "low": "hqdn3d=1.5:1.5:6:6",
            "medium": "hqdn3d=3:3:8:8",
            "high": "hqdn3d=4.5:4.5:10:10",
        }
        if denoise_filters.get(self.settings.denoise):
            filters.append(denoise_filters[self.settings.denoise])
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-y",
            "-i",
            self.settings.input_path,
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
        ]
        if filters:
            command.extend(["-vf", ",".join(filters)])
        command.extend(["-start_number", "1", str(self.input_frames / "%08d.png"), "-progress", "pipe:1", "-nostats"])
        self.log("1/4 원본 프레임 추출 중...")

        def on_line(line: str) -> None:
            if line.startswith("frame="):
                try:
                    value = int(line.split("=", 1)[1]) / max(1, self.info.frame_count)
                    self.progress(0.02 + min(1.0, value) * 0.13, "원본 프레임 추출")
                except ValueError:
                    pass

        self._run_process(command, line_callback=on_line)
        marker.touch()
        actual = len(list(self.input_frames.glob("*.png")))
        if actual == 0:
            raise PipelineError("영상에서 프레임을 추출하지 못했습니다.")
        self.info.frame_count = actual
        self._write_manifest()
        self.log(f"프레임 추출 완료: {actual:,}장")

    @staticmethod
    def _link_or_copy(source: Path, target: Path) -> None:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    def _upscale_frames(self) -> None:
        assert self.info is not None
        self.upscaled_frames.mkdir(parents=True, exist_ok=True)
        inputs = sorted(self.input_frames.glob("*.png"))
        missing = [path for path in inputs if not (self.upscaled_frames / path.name).exists()]
        completed = len(inputs) - len(missing)
        if not missing:
            self.log(f"2/4 업스케일: 캐시 재사용 ({len(inputs):,}장)")
            self.progress(0.72, "업스케일 프레임 캐시 재사용")
            return
        self.log(f"2/4 업스케일 시작: 완료 {completed:,} / 전체 {len(inputs):,}")
        if self.settings.backend == "lanczos":
            self._upscale_lanczos(inputs)
            return
        self._upscale_ai(missing, len(inputs), completed)

    def _upscale_lanczos(self, inputs: list[Path]) -> None:
        assert self.info is not None
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-y",
            "-framerate",
            f"{self.info.fps:.8f}",
            "-start_number",
            "1",
            "-i",
            str(self.input_frames / "%08d.png"),
            "-vf",
            f"scale=iw*{self.settings.scale}:ih*{self.settings.scale}:flags=lanczos",
            "-start_number",
            "1",
            str(self.upscaled_frames / "%08d.png"),
            "-progress",
            "pipe:1",
            "-nostats",
        ]

        def on_line(line: str) -> None:
            if line.startswith("frame="):
                try:
                    ratio = int(line.split("=", 1)[1]) / max(1, len(inputs))
                    self.progress(0.15 + min(1.0, ratio) * 0.57, "Lanczos 업스케일")
                except ValueError:
                    pass

        self._run_process(command, line_callback=on_line)

    def _model_name(self) -> str:
        if self.settings.content_mode == "anime":
            return "realesr-animevideov3"
        return "realesrgan-x4plus"

    def _upscale_ai(self, missing: list[Path], total: int, completed: int) -> None:
        assert self.realesrgan
        batch_size = 120
        stage_root = self.cache_dir / "batch_stage"
        safe_real_2x = self.settings.content_mode == "real" and self.settings.scale == 2
        if safe_real_2x:
            self.log("실사 2배 안전 모드: AI 4배 추론 후 고품질 2배 축소를 적용합니다.")
        for offset in range(0, len(missing), batch_size):
            self._check_cancelled()
            batch = missing[offset : offset + batch_size]
            stage_input = stage_root / "input"
            if stage_root.exists():
                shutil.rmtree(stage_root)
            stage_input.mkdir(parents=True)
            if safe_real_2x:
                ai_output = stage_root / "native_4x"
                resized_output = stage_root / "resized_2x"
                ai_output.mkdir(parents=True)
                resized_output.mkdir(parents=True)
                for index, source in enumerate(batch, start=1):
                    self._link_or_copy(source, stage_input / f"{index:08d}.png")
                ai_scale = 4
            else:
                ai_output = self.upscaled_frames
                resized_output = None
                for source in batch:
                    self._link_or_copy(source, stage_input / source.name)
                ai_scale = self.settings.scale
            command = [
                self.realesrgan,
                "-i",
                str(stage_input),
                "-o",
                str(ai_output),
                "-n",
                self._model_name(),
                "-s",
                str(ai_scale),
                "-g",
                "0",
                "-j",
                "1:2:2",
                "-f",
                "png",
            ]
            if self.settings.tile_size > 0:
                command.extend(["-t", str(self.settings.tile_size)])
            self._run_process(command, cwd=Path(self.realesrgan).parent)
            if safe_real_2x:
                assert resized_output is not None
                resize_command = [
                    self.ffmpeg,
                    "-hide_banner",
                    "-y",
                    "-framerate",
                    "1",
                    "-start_number",
                    "1",
                    "-i",
                    str(ai_output / "%08d.png"),
                    "-vf",
                    "scale=iw/2:ih/2:flags=lanczos",
                    "-fps_mode",
                    "passthrough",
                    "-start_number",
                    "1",
                    str(resized_output / "%08d.png"),
                ]
                self._run_process(resize_command)
                for index, source in enumerate(batch, start=1):
                    resized = resized_output / f"{index:08d}.png"
                    if not resized.is_file():
                        raise PipelineError(f"안전 모드 축소 프레임이 누락되었습니다: {resized.name}")
                    os.replace(resized, self.upscaled_frames / source.name)
            done = completed + min(offset + len(batch), len(missing))
            self.progress(0.15 + (done / max(1, total)) * 0.57, f"AI 업스케일 {done:,}/{total:,}")
            self.log(f"AI 업스케일: {done:,}/{total:,}프레임")
        shutil.rmtree(stage_root, ignore_errors=True)

    def _interpolate_frames(self) -> Path:
        assert self.rife and self.info is not None
        inputs = sorted(self.upscaled_frames.glob("*.png"))
        if len(inputs) < 2:
            raise PipelineError("프레임 보간에 필요한 프레임이 부족합니다.")
        # Directory mode already creates 2N frames by default. Avoid -n because
        # several packaged models accept custom counts only with exact rife-v4.
        # 2N frames at 2F fps preserve the original N/F duration.
        expected = len(inputs) * 2
        existing = len(list(self.interpolated_frames.glob("*.png"))) if self.interpolated_frames.exists() else 0
        if existing >= expected:
            self.log(f"3/4 프레임 보간: 캐시 재사용 ({existing:,}장)")
            self.progress(0.88, "보간 프레임 캐시 재사용")
            return self.interpolated_frames
        if self.interpolated_frames.exists():
            shutil.rmtree(self.interpolated_frames)
        self.interpolated_frames.mkdir(parents=True)
        self.log(f"3/4 RIFE 프레임 보간 시작: {self.info.fps:.3f} → {self.info.fps * 2:.3f}fps")
        model = "rife-anime" if self.settings.content_mode == "anime" else "rife-v4.6"
        command = [
            self.rife,
            "-i",
            str(self.upscaled_frames),
            "-o",
            str(self.interpolated_frames),
            "-m",
            model,
            "-f",
            "%08d.png",
        ]
        if self.info.width * self.settings.scale >= 3840 or self.info.height * self.settings.scale >= 2160:
            command.append("-u")
        self.progress(0.73, f"RIFE GPU 배치 보간 ({expected:,}프레임 생성)")
        self._run_process(command, cwd=Path(self.rife).parent)
        actual = len(list(self.interpolated_frames.glob("*.png")))
        if actual != expected:
            raise PipelineError(f"RIFE 출력 프레임 수가 예상과 다릅니다 ({actual:,}/{expected:,}).")
        self.progress(0.88, f"RIFE 보간 완료 ({actual:,}프레임)")
        return self.interpolated_frames

    def _encoder_args(self) -> list[str]:
        quality = {
            "high": {"cpu": "17", "nv": "16"},
            "balanced": {"cpu": "20", "nv": "19"},
            "small": {"cpu": "24", "nv": "23"},
        }.get(self.settings.quality, {"cpu": "20", "nv": "19"})
        if self.settings.hardware_encode and nvenc_available():
            encoder = "hevc_nvenc" if self.settings.codec == "h265" else "h264_nvenc"
            return ["-c:v", encoder, "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", quality["nv"], "-b:v", "0"]
        encoder = "libx265" if self.settings.codec == "h265" else "libx264"
        return ["-c:v", encoder, "-preset", "slow", "-crf", quality["cpu"]]

    def _encode(self, frames: Path, fps: float) -> Path:
        assert self.info is not None
        output = Path(self.settings.output_path).resolve()
        self.log("4/4 최종 영상 인코딩 및 원본 오디오 결합 중...")
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-y",
            "-framerate",
            f"{fps:.8f}",
            "-start_number",
            "1",
            "-i",
            str(frames / "%08d.png"),
            "-i",
            self.settings.input_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
        ]
        is_mkv = output.suffix.lower() == ".mkv"
        if is_mkv:
            command.extend(["-map", "1:s?", "-c:a", "copy", "-c:s", "copy"])
        else:
            supported_subtitles = [
                stream for stream in self.info.subtitle_streams if stream.get("codec_name") in TEXT_SUBTITLE_CODECS
            ]
            for stream in supported_subtitles:
                command.extend(["-map", f"1:{stream['index']}"])
            if self.info.subtitle_streams and len(supported_subtitles) < len(self.info.subtitle_streams):
                self.log("주의: MP4와 호환되지 않는 이미지 자막은 제외했습니다. 모두 보존하려면 MKV로 출력하세요.")
            command.extend(["-c:a", "aac", "-b:a", "320k"])
            if supported_subtitles:
                command.extend(["-c:s", "mov_text"])
        command.extend(self._encoder_args())
        command.extend(["-pix_fmt", "yuv420p", "-map_metadata", "1", "-map_chapters", "1"])
        if not is_mkv:
            command.extend(["-movflags", "+faststart"])
        command.extend([str(output), "-progress", "pipe:1", "-nostats"])

        def on_line(line: str) -> None:
            if line.startswith("out_time="):
                raw_time = line.split("=", 1)[1]
                if raw_time == "N/A":
                    return
                elapsed = _parse_ffmpeg_time(raw_time)
                ratio = elapsed / max(0.001, self.info.duration)
                self.progress(0.88 + min(1.0, ratio) * 0.12, "최종 영상 인코딩")

        self._run_process(command, line_callback=on_line)
        if not output.exists() or output.stat().st_size == 0:
            raise PipelineError("최종 영상 파일이 생성되지 않았습니다.")
        self.log(f"완료: {output}")
        return output


def engine_status() -> dict[str, str]:
    return {
        "ffmpeg": find_ffmpeg(),
        "ffprobe": find_ffprobe(),
        "realesrgan": find_realesrgan(),
        "rife": find_rife(),
        "gpu": detect_nvidia_gpu(),
        "nvenc": "ready" if nvenc_available() else "unavailable",
    }
