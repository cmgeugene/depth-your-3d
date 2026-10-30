"""ffmpeg helpers: probing, preview extraction, raw frame reading and writing.

Every decode uses `-fps_mode passthrough` so frame N means the same frame everywhere
(preview JPEGs, scene scores, depth inference and stepped output).
"""
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import config, motion


def probe(path: Path) -> dict:
    out = subprocess.run(
        [config.FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True,
        check=True,
    ).stdout
    info = json.loads(out)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("영상 스트림을 찾을 수 없습니다.")

    width, height = int(video["width"]), int(video["height"])
    if _rotation(video) % 180 == 90:
        width, height = height, width

    fps_str = video.get("avg_frame_rate", "0/0")
    if not _valid_rate(fps_str):
        fps_str = video.get("r_frame_rate", "24/1")
    if not _valid_rate(fps_str):
        fps_str = "24/1"

    duration = float(info.get("format", {}).get("duration") or video.get("duration") or 0)
    return {
        "width": width,
        "height": height,
        "fps_str": fps_str,
        "fps": float(Fraction(fps_str)),
        "duration": duration,
        "est_frames": int(video.get("nb_frames") or 0) or round(duration * float(Fraction(fps_str))),
        "has_audio": any(s.get("codec_type") == "audio" for s in info.get("streams", [])),
        "codec": video.get("codec_name", ""),
    }


def _valid_rate(rate: str) -> bool:
    try:
        return Fraction(rate) > 0
    except (ValueError, ZeroDivisionError):
        return False


def _rotation(stream: dict) -> int:
    rot = stream.get("tags", {}).get("rotate")
    if rot is None:
        for side in stream.get("side_data_list", []):
            if "rotation" in side:
                rot = side["rotation"]
                break
    try:
        return abs(int(float(rot or 0)))
    except ValueError:
        return 0


def prepare(src: Path, out_dir: Path, with_previews: bool) -> list[float]:
    """Decode once: scene-change score per frame, plus preview/thumb JPEGs and small gray
    frames for motion analysis (motion.gray) for the editor.

    Returns the per-frame scdet scores (0-100); its length is the true frame count.
    """
    with_motion = with_previews and motion.AVAILABLE
    graph = f"[0:v]split={4 if with_motion else 3}[s][p][t]{'[m]' if with_motion else ''};" if with_previews else "[0:v]null[s];"
    graph += "[s]scale=320:-2,scdet=threshold=100:sc_pass=0,metadata=print:key=lavfi.scd.score:file=scores.txt[so]"
    cmd = [config.FFMPEG, "-v", "error", "-y", "-i", str(src)]
    if with_previews:
        (out_dir / "preview").mkdir(exist_ok=True)
        (out_dir / "thumb").mkdir(exist_ok=True)
        graph += f";[p]scale={config.PREVIEW_WIDTH}:-2[po];[t]scale={config.THUMB_WIDTH}:-2[to]"
    if with_motion:
        graph += f";[m]scale={motion.WIDTH}:-2,format=gray[mo]"
    cmd += ["-filter_complex", graph, "-map", "[so]", "-fps_mode", "passthrough", "-f", "null", "-"]
    if with_previews:
        for name, q in (("preview", 4), ("thumb", 6)):
            cmd += [
                "-map", f"[{name[0]}o]", "-fps_mode", "passthrough", "-q:v", str(q),
                "-start_number", "0", f"{name}/%06d.jpg",
            ]
    if with_motion:
        cmd += ["-map", "[mo]", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "motion.gray"]
    # cwd keeps the metadata file path free of drive-letter colons, which the filter parser would split on.
    result = subprocess.run(cmd, cwd=out_dir, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg 디코딩 실패: " + result.stderr.decode(errors="replace")[-600:])

    scores: list[float] = []
    for line in (out_dir / "scores.txt").read_text().splitlines():
        if line.startswith("frame:"):
            scores.append(0.0)
        elif line.startswith("lavfi.scd.score=") and scores:
            scores[-1] = float(line.split("=", 1)[1])
    if not scores:
        raise RuntimeError("프레임을 읽지 못했습니다.")
    return scores


def detect_cuts(scores: list[float], threshold: float, min_shot: int) -> list[int]:
    """Frames whose scene score crosses the threshold start a new shot. Mirrored in static/common.js."""
    cuts: list[int] = []
    last = 0
    for i in range(1, len(scores)):
        if scores[i] >= threshold and i - last >= min_shot and len(scores) - i >= 1:
            cuts.append(i)
            last = i
    return cuts


class FrameReader:
    """Streams RGB frames (H, W, 3) uint8 from a video file."""

    def __init__(self, path: Path, width: int, height: int):
        self.shape = (height, width, 3)
        self.size = width * height * 3
        self.proc = subprocess.Popen(
            [config.FFMPEG, "-v", "error", "-i", str(path), "-fps_mode", "passthrough",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self.size * 2,
        )

    def read(self) -> np.ndarray | None:
        buf = self.proc.stdout.read(self.size)
        if len(buf) < self.size:
            return None
        return np.frombuffer(buf, np.uint8).reshape(self.shape)

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FrameWriter:
    """Encodes raw frames to H.264 mp4. pix_fmt is 'gray' or 'rgb24'."""

    def __init__(self, path: Path, width: int, height: int, fps_str: str, pix_fmt: str,
                 crf: int, log_path: Path, audio_src: Path | None = None, duration: float | None = None):
        cmd = [config.FFMPEG, "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{width}x{height}", "-r", fps_str, "-i", "-"]
        if audio_src is not None:
            cmd += ["-i", str(audio_src), "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-b:a", "192k"]
        cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "medium",
                "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if duration is not None:
            cmd += ["-t", f"{duration:.6f}"]
        cmd.append(str(path))
        self.log_path = log_path
        self._log = open(log_path, "ab")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log)

    def write(self, frame: np.ndarray):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        code = self.proc.wait()
        self._log.close()
        if code != 0:
            tail = self.log_path.read_text(errors="replace")[-600:]
            raise RuntimeError(f"ffmpeg 인코딩 실패 ({code}): {tail}")

    def abort(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self._log.close()
