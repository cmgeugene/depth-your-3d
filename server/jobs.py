"""In-memory registry of uploaded media and render jobs, plus the single GPU worker.

Jobs run one at a time in submission order, so everyone shares the one GPU fairly.
State is lost on restart; files on disk are swept by the TTL cleaner.
"""
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import config, pipeline, video

ID_RE = re.compile(r"^[0-9a-f]{32}$")


def new_id() -> str:
    return uuid.uuid4().hex


def safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", stem).strip(" .")
    return stem[:80] or "video"


def _summary(media: "Media", params: dict) -> dict:
    outs = params["outputs"]
    fmap = params.get("map")
    want_depth = outs.get("depth_mp4") or outs.get("depth_png")
    with_audio = media.has_audio and outs.get("audio", True) and (outs.get("depth_mp4") or outs.get("stepped_mp4"))
    return {
        "resolution": f"{media.width}×{media.height}",
        "frames": media.frames,
        "drawings": len(set(fmap)) if fmap else media.frames,
        "model": config.MODELS[params["depth"]["model"]]["label"] if want_depth else None,
        "outputs": [label for key, label in (("stepped_mp4", "스텝 영상"), ("depth_mp4", "Depth mp4"),
                                             ("depth_png", "PNG16")) if outs.get(key)]
                   + (["오디오"] if with_audio else []),
    }


@dataclass
class Media:
    id: str
    name: str
    dir: Path
    src: Path
    status: str = "preparing"  # preparing | ready | error
    error: str | None = None
    created: float = field(default_factory=time.time)
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fps_str: str = ""
    duration: float = 0.0
    has_audio: bool = False
    frames: int = 0
    scores: list[float] = field(default_factory=list)
    cuts: list[int] = field(default_factory=list)
    editor: bool = False

    @property
    def stem(self) -> str:
        return safe_stem(self.name)

    def public(self) -> dict:
        d = {"id": self.id, "name": self.name, "status": self.status, "error": self.error}
        if self.status == "ready":
            d.update(
                width=self.width, height=self.height, fps=self.fps, fps_str=self.fps_str,
                duration=self.duration, has_audio=self.has_audio, frames=self.frames,
                scores=self.scores, cuts=self.cuts, editor=self.editor,
                preview_width=config.PREVIEW_WIDTH, max_editor_frames=config.MAX_EDITOR_FRAMES,
            )
        return d


@dataclass
class Job:
    id: str
    media_id: str
    params: dict
    dir: Path
    status: str = "queued"  # queued | running | done | error | cancelled
    progress: float = 0.0
    stage: str = "대기 중"
    error: str | None = None
    outputs: list[dict] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    summary: dict = field(default_factory=dict)  # workload shown in the shared queue view


class Registry:
    def __init__(self):
        self.media: dict[str, Media] = {}
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.queue: "queue.Queue[Job]" = queue.Queue()
        self.order: list[str] = []  # queued job ids, for queue positions
        self.prep_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="prep")
        config.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._worker, daemon=True, name="gpu-worker").start()
        threading.Thread(target=self._cleaner, daemon=True, name="cleaner").start()

    # ---- media ----
    def add_media(self, media: Media):
        with self.lock:
            self.media[media.id] = media
        self.prep_pool.submit(self._prepare, media)

    def _prepare(self, m: Media):
        try:
            info = video.probe(m.src)
            m.width, m.height = info["width"], info["height"]
            m.fps, m.fps_str = info["fps"], info["fps_str"]
            m.duration, m.has_audio = info["duration"], info["has_audio"]
            with_previews = info["est_frames"] <= config.MAX_EDITOR_FRAMES
            m.scores = video.prepare(m.src, m.dir, with_previews)
            m.frames = len(m.scores)
            m.editor = with_previews and m.frames <= config.MAX_EDITOR_FRAMES
            m.cuts = video.detect_cuts(m.scores, config.CUT_THRESHOLD, config.CUT_MIN_SHOT)
            m.status = "ready"
        except Exception as exc:  # surfaced to the uploader
            traceback.print_exc()
            m.status, m.error = "error", str(exc)

    def get_media(self, media_id: str) -> Media | None:
        return self.media.get(media_id) if ID_RE.match(media_id or "") else None

    # ---- jobs ----
    def submit(self, media: Media, params: dict) -> Job:
        job_id = new_id()
        job_dir = config.JOBS_DIR / job_id
        job_dir.mkdir(parents=True)
        job = Job(id=job_id, media_id=media.id, params=params, dir=job_dir, summary=_summary(media, params))
        with self.lock:
            self.jobs[job_id] = job
            self.order.append(job_id)
        self.queue.put(job)
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id) if ID_RE.match(job_id or "") else None

    def cancel(self, job: Job):
        job.cancel.set()
        if job.status == "queued":
            job.status, job.stage, job.finished = "cancelled", "취소됨", time.time()
            with self.lock:
                if job.id in self.order:
                    self.order.remove(job.id)

    def job_public(self, job: Job) -> dict:
        with self.lock:
            position = self.order.index(job.id) + 1 if job.id in self.order else 0
        media = self.media.get(job.media_id)
        return {
            "id": job.id, "media_id": job.media_id, "media_name": media.name if media else "",
            "status": job.status, "progress": round(job.progress, 4), "stage": job.stage,
            "position": position, "error": job.error, "outputs": job.outputs,
            "created": job.created, "started": job.started, "finished": job.finished,
        }

    def queue_length(self) -> int:
        with self.lock:
            return len(self.order) + sum(1 for j in self.jobs.values() if j.status == "running")

    def queue_snapshot(self, recent: int = 5) -> dict:
        """Everyone's jobs for the shared queue view. Only an 8-char tag is exposed, never the
        full job id, so other users can't open or download someone else's results."""
        with self.lock:
            queued = [self.jobs[i] for i in self.order if i in self.jobs]
            running = [j for j in self.jobs.values() if j.status == "running"]
            finished = sorted(
                (j for j in self.jobs.values() if j.status in ("done", "error", "cancelled") and j.finished),
                key=lambda j: j.finished, reverse=True,
            )[:recent]

        def row(job: Job, position: int) -> dict:
            media = self.media.get(job.media_id)
            return {
                "tag": job.id[:8], "name": media.name if media else "", "status": job.status,
                "progress": round(job.progress, 4), "stage": job.stage, "position": position,
                "created": job.created, "started": job.started, "finished": job.finished, **job.summary,
            }

        return {
            "now": time.time(),
            "running": [row(j, 0) for j in running],
            "queued": [row(j, i + 1) for i, j in enumerate(queued)],
            "recent": [row(j, 0) for j in finished],
        }

    def _worker(self):
        while True:
            job = self.queue.get()
            with self.lock:
                if job.id in self.order:
                    self.order.remove(job.id)
                if job.cancel.is_set():
                    continue
                # Flip to running under the lock so the queue view never loses the job in between.
                job.status, job.started, job.stage = "running", time.time(), "시작"
            media = self.media.get(job.media_id)

            def progress(value: float, text: str, job=job):
                job.progress, job.stage = value, text

            try:
                if media is None or media.status != "ready":
                    raise RuntimeError("원본 영상이 만료되었거나 준비되지 않았습니다.")
                job.outputs = pipeline.run(job, media, progress)
                job.status, job.progress, job.stage = "done", 1.0, "완료"
            except pipeline.Cancelled:
                job.status, job.stage = "cancelled", "취소됨"
            except Exception as exc:
                traceback.print_exc()
                job.status, job.error, job.stage = "error", str(exc), "실패"
            finally:
                job.finished = time.time()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def _cleaner(self):
        while True:
            time.sleep(600)
            cutoff = time.time() - config.TTL_HOURS * 3600
            busy = {j.media_id for j in self.jobs.values() if j.status in ("queued", "running")}
            for base, table in ((config.MEDIA_DIR, self.media), (config.JOBS_DIR, self.jobs)):
                for d in base.iterdir():
                    if not d.is_dir() or d.stat().st_mtime > cutoff or d.name in busy:
                        continue
                    item = table.get(d.name)
                    if isinstance(item, Job) and item.status in ("queued", "running"):
                        continue
                    shutil.rmtree(d, ignore_errors=True)
                    with self.lock:
                        table.pop(d.name, None)
