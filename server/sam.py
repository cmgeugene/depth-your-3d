"""Auto-select for the cleanup page: SAM 3.1 text-prompt masks ("hair" -> every character's hair).

SAM 3.1 needs numpy<2, so it can't share the studio's .venv. It runs in its own environment
(DS_SAM3_PYTHON) as a worker process (sam_worker.py) that stays loaded between requests and exits
after DS_SAM3_IDLE_SEC idle seconds, which also gives its VRAM back. Results are cached per image
and prompt, so asking again is instant.
"""
import atexit
import hashlib
import json
import os
import subprocess
import threading
import time

from . import config

WORKER = config.ROOT / "server" / "sam_worker.py"
FIRST_TIMEOUT = 600  # the first request also loads the 3.5GB checkpoint
TIMEOUT = 180

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_last_used = 0.0
_reaper_started = False


def status() -> dict:
    if not config.SAM3_PYTHON or not config.SAM3_CHECKPOINT:
        reason = "SAM 3.1이 설정되지 않았습니다 (DS_SAM3_PYTHON, DS_SAM3_CHECKPOINT)."
    elif not os.path.isfile(config.SAM3_PYTHON):
        reason = f"SAM 3.1 Python을 찾을 수 없습니다: {config.SAM3_PYTHON}"
    elif not os.path.isfile(config.SAM3_CHECKPOINT):
        reason = f"SAM 3.1 체크포인트를 찾을 수 없습니다: {config.SAM3_CHECKPOINT}"
    else:
        reason = None
    return {"available": reason is None, "reason": reason, "loaded": _proc is not None and _proc.poll() is None}


def _start():
    global _proc, _reaper_started
    log = open(config.DATA_DIR / "sam3-worker.log", "a", encoding="utf-8", errors="replace")
    env = {**os.environ, "DS_SAM3_CHECKPOINT": config.SAM3_CHECKPOINT, "PYTHONIOENCODING": "utf-8"}
    # -I: don't pick up the studio's working directory or PYTHON* variables in the other environment.
    _proc = subprocess.Popen([config.SAM3_PYTHON, "-I", str(WORKER), "--serve"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=log, env=env, text=True, encoding="utf-8")
    if not _reaper_started:
        _reaper_started = True
        threading.Thread(target=_reaper, daemon=True, name="sam3-reaper").start()


def stop():
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.kill()
    _proc = None


atexit.register(stop)


def _reaper():
    while True:
        time.sleep(15)
        with _lock:
            if _proc is not None and time.time() - _last_used > config.SAM3_IDLE_SEC:
                stop()
                print(f"SAM 3.1: {config.SAM3_IDLE_SEC:g}초 동안 요청이 없어 종료했습니다 (VRAM 반환).", flush=True)


def segment(media, prompt: str) -> bytes:
    """PNG mask (white = found) of every `prompt` instance in the media's image. Raises RuntimeError."""
    global _last_used
    key = hashlib.sha1(prompt.encode()).hexdigest()[:16]
    cached = media.dir / f"sam_{key}.png"
    if cached.exists():
        return cached.read_bytes()
    with _lock:
        first = _proc is None or _proc.poll() is not None
        if first:
            _start()
        out = media.dir / f"sam_{key}.tmp.png"
        try:
            _proc.stdin.write(json.dumps({"image": str(media.src), "prompt": prompt, "out": str(out)}) + "\n")
            _proc.stdin.flush()
            reply = _readline(FIRST_TIMEOUT if first else TIMEOUT)
        except (OSError, TimeoutError) as exc:
            stop()
            raise RuntimeError(f"SAM 3.1 작업 프로세스가 응답하지 않습니다 ({exc}). 서버 폴더의 data/sam3-worker.log를 확인하세요.")
        finally:
            _last_used = time.time()
    if not reply.get("ok"):
        # A failure (often CUDA out of memory) can leave the loaded model in a bad state; start clean next time.
        with _lock:
            stop()
        raise RuntimeError(f"SAM 3.1 실패: {reply.get('error')}")
    if reply["count"] == 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"\"{prompt}\"에 해당하는 영역을 찾지 못했습니다. 다른 단어(영어)로 시도해 보세요.")
    out.replace(cached)
    return cached.read_bytes()


def _readline(timeout: float) -> dict:
    result = {}

    def read():
        result["line"] = _proc.stdout.readline()

    t = threading.Thread(target=read, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"{timeout:.0f}초 초과")
    if not result.get("line"):
        raise OSError("작업 프로세스가 종료됨")
    return json.loads(result["line"])
