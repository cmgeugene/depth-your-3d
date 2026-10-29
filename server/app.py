"""HTTP API and static pages. Run with:  python -m server"""
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, depth
from .jobs import Media, Registry, new_id

app = FastAPI(title="Depth Studio")
registry = Registry()
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

IMMUTABLE = {"Cache-Control": "public, max-age=86400, immutable"}


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/editor")
def editor():
    return FileResponse(config.STATIC_DIR / "editor.html")


@app.get("/api/status")
def status():
    return {
        "gpu": depth.gpu_name(),
        "queue": registry.queue_length(),
        "models": [{"key": k, **config.MODELS[k]} for k in config.ENABLED_MODELS],
        "default_model": config.DEFAULT_MODEL if config.DEFAULT_MODEL in config.ENABLED_MODELS else config.ENABLED_MODELS[0],
        "input_sizes": config.INPUT_SIZES,
        "cut_threshold": config.CUT_THRESHOLD,
        "cut_min_shot": config.CUT_MIN_SHOT,
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "ttl_hours": config.TTL_HOURS,
    }


# ---------- media ----------

@app.post("/api/media")
async def upload(file: UploadFile, request: Request):
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    if int(request.headers.get("content-length") or 0) > limit + 1024 * 1024:
        raise HTTPException(413, f"파일이 너무 큽니다 (최대 {config.MAX_UPLOAD_MB}MB).")
    media_id = new_id()
    media_dir = config.MEDIA_DIR / media_id
    media_dir.mkdir(parents=True)
    suffix = "".join(c for c in (file.filename or "").rsplit(".", 1)[-1][:8] if c.isalnum()) or "mp4"
    src = media_dir / f"source.{suffix}"
    written = 0
    with open(src, "wb") as f:
        while chunk := await file.read(4 * 1024 * 1024):
            written += len(chunk)
            if written > limit:
                f.close()
                src.unlink(missing_ok=True)
                raise HTTPException(413, f"파일이 너무 큽니다 (최대 {config.MAX_UPLOAD_MB}MB).")
            f.write(chunk)
    media = Media(id=media_id, name=file.filename or "video.mp4", dir=media_dir, src=src)
    registry.add_media(media)
    return media.public()


def _media(media_id: str) -> Media:
    media = registry.get_media(media_id)
    if media is None:
        raise HTTPException(404, "영상을 찾을 수 없습니다 (만료되었을 수 있습니다).")
    return media


@app.get("/api/media/{media_id}")
def media_info(media_id: str):
    return _media(media_id).public()


@app.get("/api/media/{media_id}/{kind}/{n}.jpg")
def media_frame(media_id: str, kind: str, n: int):
    media = _media(media_id)
    if kind not in ("preview", "thumb") or not media.editor or not 0 <= n < media.frames:
        raise HTTPException(404)
    path = media.dir / kind / f"{n:06d}.jpg"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg", headers=IMMUTABLE)


# ---------- jobs ----------

class DepthParams(BaseModel):
    model: str = config.DEFAULT_MODEL
    input_size: int = 518
    norm: str = "shot"  # shot | adaptive | clip | frame
    clip_lo: float = Field(1.0, ge=0, le=20)
    clip_hi: float = Field(1.0, ge=0, le=20)
    adaptive_window: float = Field(1.0, ge=0.1, le=10)
    temporal: float = Field(0.0, ge=0, le=4)
    invert: bool = False
    gamma: float = Field(1.0, ge=0.2, le=5)


class Outputs(BaseModel):
    depth_mp4: bool = True
    depth_png: bool = False
    stepped_mp4: bool = False
    audio: bool = True  # copy the source audio track into the mp4 outputs


class JobRequest(BaseModel):
    media_id: str
    outputs: Outputs = Outputs()
    depth: DepthParams = DepthParams()
    map: list[int] | None = None  # source frame per output frame (timing editor)
    cuts: list[int] | None = None
    timing: dict | None = None  # editor spec, stored alongside the output for reference


@app.post("/api/jobs")
def create_job(req: JobRequest):
    media = _media(req.media_id)
    if media.status != "ready":
        raise HTTPException(409, "영상이 아직 준비되지 않았습니다.")
    o, d = req.outputs, req.depth
    if not (o.depth_mp4 or o.depth_png or o.stepped_mp4):
        raise HTTPException(400, "출력을 하나 이상 선택하세요.")
    if d.model not in config.ENABLED_MODELS:
        raise HTTPException(400, "사용할 수 없는 모델입니다.")
    if d.input_size not in config.INPUT_SIZES:
        raise HTTPException(400, "지원하지 않는 입력 해상도입니다.")
    if d.norm not in ("shot", "adaptive", "clip", "frame"):
        raise HTTPException(400, "정규화 방식이 올바르지 않습니다.")
    n = media.frames
    if req.map is not None:
        m = req.map
        if len(m) != n or any(not 0 <= v <= i for i, v in enumerate(m)) or any(m[i] < m[i - 1] for i in range(1, n)):
            raise HTTPException(400, "타이밍 맵이 올바르지 않습니다.")
    if o.stepped_mp4 and req.map is None:
        raise HTTPException(400, "스텝 영상은 편집기에서 타이밍을 지정해야 합니다.")
    if req.cuts is not None and any(not 0 < c < n for c in req.cuts):
        raise HTTPException(400, "컷 위치가 올바르지 않습니다.")
    job = registry.submit(media, req.model_dump())
    return registry.job_public(job)


@app.get("/api/queue")
def queue_view():
    return registry.queue_snapshot()


def _job(job_id: str):
    job = registry.get_job(job_id)
    if job is None:
        raise HTTPException(404, "작업을 찾을 수 없습니다 (만료되었을 수 있습니다).")
    return job


@app.get("/api/jobs/{job_id}")
def job_info(job_id: str):
    return registry.job_public(_job(job_id))


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    job = _job(job_id)
    registry.cancel(job)
    return registry.job_public(job)


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str):
    job = _job(job_id)
    if job.status != "done" or name not in {o["name"] for o in job.outputs}:
        raise HTTPException(404)
    return FileResponse(job.dir / name, filename=name)
