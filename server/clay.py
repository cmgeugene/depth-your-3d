"""Clay reference videos: live-action people -> grey clay figures (SAM 3D Body), optionally over depth.

For Seedance v2v: the clay carries body motion, head angle and camera, while the actor's face shape
stays out (the head is rebuilt neutral). Laid over the clip's depth video it also carries the room.

SAM 3D Body needs its own Python environment (DS_CLAY_PYTHON) and repository checkout (DS_CLAY_REPO),
so it runs as a worker process (clay_worker.py). The worker stays loaded between jobs and is killed
after DS_CLAY_IDLE_SEC idle seconds, which gives its VRAM back. Jobs run in the shared GPU queue.
"""
import atexit
import json
import os
import queue
import subprocess
import threading
import time
from types import SimpleNamespace

from . import config, pipeline
from .pipeline import Cancelled

WORKER = config.ROOT / "server" / "clay_worker.py"
FIRST_TIMEOUT = 900  # the first job also loads SAM 3D Body (2.7GB), MoGe-2 and the helpers
STEP_TIMEOUT = 300  # longest silence between progress lines once loaded

MODES = {"head": "체형 유지 · 얼굴 중립", "neutral": "체형 · 얼굴 모두 중립"}
BACKGROUNDS = {"none": "클레이", "auto": "Depth + 클레이", "upload": "업로드 depth + 클레이"}
BG_MODES = {"overlay": "depth 위에 얹기", "remove": "사람 지우고 얹기"}
HANDS = {"full": "손 정확하게", "fast": "손 단순 (빠르게)"}

_lock = threading.Lock()  # one job at a time talks to the worker
_proc: subprocess.Popen | None = None
_lines: "queue.Queue[str | None]" = queue.Queue()
_last_used = 0.0
_busy = False
_reaper_started = False


def status() -> dict:
    if not config.CLAY_PYTHON or not config.CLAY_REPO:
        reason = "클레이 도구가 설정되지 않았습니다 (DS_CLAY_PYTHON, DS_CLAY_REPO)."
    elif not os.path.isfile(config.CLAY_PYTHON):
        reason = f"SAM 3D Body Python을 찾을 수 없습니다: {config.CLAY_PYTHON}"
    elif not os.path.isdir(os.path.join(config.CLAY_REPO, "sam_3d_body")):
        reason = f"SAM 3D Body 저장소를 찾을 수 없습니다: {config.CLAY_REPO}"
    else:
        reason = None
    return {"available": reason is None, "reason": reason, "loaded": _proc is not None and _proc.poll() is None,
            "idle_sec": config.CLAY_IDLE_SEC}


def summary(media, p: dict) -> dict:
    model = "SAM 3D Body"
    if p["bg"] == "auto":
        model = f"{config.MODELS[p['depth']['model']]['label']} + {model}"
    outs = ([BACKGROUNDS[p["bg"]]] + ([BG_MODES[p["bg_mode"]]] if p["bg"] != "none" else []) + [MODES[p["mode"]]]
            + [HANDS[p.get("hands", "full")]])
    return {
        "resolution": f"{media.width}×{media.height}",
        "frames": media.frames,
        "drawings": media.frames,
        "model": model,
        "outputs": outs,
    }


def _start():
    global _proc, _lines, _reaper_started
    log = open(config.DATA_DIR / "clay-worker.log", "a", encoding="utf-8", errors="replace")
    env = {**os.environ, "DS_CLAY_REPO": config.CLAY_REPO, "PYTHONIOENCODING": "utf-8", "PYOPENGL_PLATFORM": ""}
    if config.CLAY_CHECKPOINT:
        env["DS_CLAY_CHECKPOINT"] = config.CLAY_CHECKPOINT
    # -I: don't pick up the studio's working directory or PYTHON* variables in the other environment.
    _proc = subprocess.Popen([config.CLAY_PYTHON, "-I", str(WORKER), "--serve"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=log, env=env, text=True, encoding="utf-8",
                             cwd=config.CLAY_REPO)
    _lines = queue.Queue()
    proc, lines = _proc, _lines

    def pump():  # one reader per process; None marks its end
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True, name="clay-reader").start()
    if not _reaper_started:
        _reaper_started = True
        threading.Thread(target=_reaper, daemon=True, name="clay-reaper").start()


def stop():
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.kill()
    _proc = None


atexit.register(stop)


def _reaper():
    while True:
        time.sleep(15)
        if _busy or _proc is None or time.time() - _last_used <= config.CLAY_IDLE_SEC:
            continue
        with _lock:
            if not _busy and _proc is not None and time.time() - _last_used > config.CLAY_IDLE_SEC:
                stop()
                print(f"클레이: {config.CLAY_IDLE_SEC:g}초 동안 작업이 없어 SAM 3D Body를 종료했습니다 (VRAM 반환).",
                      flush=True)


def _auto_depth(job, media, progress) -> str:
    """Depth video of the clip with the studio's own pipeline and the depth settings chosen on the page."""
    sub = SimpleNamespace(
        params={"outputs": {"depth_mp4": True, "depth_png": False, "stepped_mp4": False, "audio": False},
                "depth": job.params["clay"]["depth"],
                "cuts": job.params["clay"]["cuts"], "map": None},
        dir=job.dir, cancel=job.cancel)
    out = pipeline.run(sub, media, progress)
    return str(job.dir / out[0]["name"])


def run(job, media, progress) -> list[dict]:
    global _last_used, _busy
    p = job.params["clay"]
    st = status()
    if not st["available"]:
        raise RuntimeError(st["reason"])
    results = []
    depth_path, span = None, (0.0, 1.0)
    if p["bg"] == "auto":
        depth_path = _auto_depth(job, media, lambda v, t: progress(0.25 * v, f"depth · {t}"))
        results.append({"name": os.path.basename(depth_path), "label": "Depth 영상 (mp4)"})
        span = (0.25, 1.0)
    elif p["bg"] == "upload":
        depth_path = p["depth_src"]

    suffix = "_depth+clay" if depth_path else "_clay"
    name = f"{media.stem}{suffix}.mp4"
    req = {"video": str(media.src), "out": str(job.dir / name), "frames": media.frames, "fps": media.fps_str,
           "cuts": p["cuts"], "mode": p["mode"], "smooth": p["smooth"], "max_people": p["max_people"],
           "bg": depth_path, "bg_mode": p["bg_mode"], "ffmpeg": config.FFMPEG,
           # over a depth video, its steady person depth replaces SAM's pose-driven depth wobble; not with
           # per-frame normalization, whose brightness scale moves from frame to frame
           "depth_guide": config.CLAY_DEPTH_GUIDE if depth_path and not (
               p["bg"] == "auto" and p["depth"].get("norm") in ("frame", "adaptive")) else 0,
           "bg_invert": p["bg"] == "auto" and bool(p["depth"].get("invert")),
           "fov_samples": config.CLAY_FOV_SAMPLES, "sam_hands": p.get("hands", "full") == "full"}
    a, b = span
    with _lock:
        _busy = True
        try:
            first = _proc is None or _proc.poll() is not None
            if first:
                progress(a, "SAM 3D Body 시작 중")
                _start()
            _proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            _proc.stdin.flush()
            reply = _wait(job, lambda v, t: progress(a + (b - a) * v, t), FIRST_TIMEOUT if first else STEP_TIMEOUT)
        except (OSError, TimeoutError) as exc:
            stop()
            raise RuntimeError(f"클레이 작업 프로세스가 응답하지 않습니다 ({exc}). 서버 폴더의 data/clay-worker.log를 확인하세요.")
        except Cancelled:
            stop()  # the worker can't be interrupted mid-job; a fresh one starts next time
            raise
        finally:
            _busy = False
            _last_used = time.time()
        if not reply.get("ok"):
            stop()  # often CUDA out of memory; start clean next time
            raise RuntimeError(f"클레이 생성 실패: {reply.get('error')}")
    results.append({"name": name, "label": "클레이 + depth 영상 (mp4)" if depth_path else "클레이 영상 (mp4)"})
    for r in results:
        r["size"] = (job.dir / r["name"]).stat().st_size
    return results


def _wait(job, progress, first_timeout: float) -> dict:
    """Relays progress lines until the final reply. Cancel kills the worker."""
    deadline = time.time() + first_timeout
    while True:
        if job.cancel.is_set():
            raise Cancelled()
        try:
            line = _lines.get(timeout=1.0)
        except queue.Empty:
            if time.time() > deadline:
                raise TimeoutError(f"{first_timeout:.0f}초 동안 진행 소식 없음")
            continue
        if line is None:
            raise OSError("작업 프로세스가 종료됨")
        msg = json.loads(line)
        if "progress" in msg:
            progress(msg["progress"], msg.get("stage", ""))
            deadline = time.time() + max(STEP_TIMEOUT, first_timeout if msg["progress"] == 0 else 0)
            continue
        return msg
