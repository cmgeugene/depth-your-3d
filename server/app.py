"""HTTP API and static pages. Run with:  python -m server"""
import io
import secrets

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import cleanup, config, depth, sam, upscale
from .jobs import Media, Registry, new_id

app = FastAPI(title="휴먼에이 영상 처리 스튜디오")
registry = Registry()
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

IMMUTABLE = {"Cache-Control": "public, max-age=86400, immutable"}
PAGES = {"/": "landing.html", "/depth": "index.html", "/upscale": "upscale.html", "/editor": "editor.html",
         "/cleanup": "cleanup.html"}


@app.middleware("http")
async def revalidate_pages(request: Request, call_next):
    # Pages and JS/CSS must revalidate, or browsers mix old and new files after a server update.
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") or path in PAGES:
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


def _page(name: str):
    return lambda: FileResponse(config.STATIC_DIR / name)


for _path, _name in PAGES.items():
    app.add_api_route(_path, _page(_name), methods=["GET"], include_in_schema=False)


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
        "upscale": {**upscale.engines(), "max_side": config.UPSCALE_MAX_SIDE},
        "cleanup": {**cleanup.status(), "sam": sam.status()},
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
    suffix = ("".join(c for c in (file.filename or "").rsplit(".", 1)[-1][:8] if c.isalnum()) or "mp4").lower()
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
    kind = "image" if src.suffix in cleanup.IMAGE_SUFFIXES else "video"
    media = Media(id=media_id, name=file.filename or "video.mp4", dir=media_dir, src=src, kind=kind)
    registry.add_media(media)
    return media.public()


def _media(media_id: str) -> Media:
    media = registry.get_media(media_id)
    if media is None:
        raise HTTPException(404, "파일을 찾을 수 없습니다 (만료되었을 수 있습니다).")
    return media


def _ready(media: Media, kind: str) -> Media:
    if media.status != "ready":
        raise HTTPException(409, "파일이 아직 준비되지 않았습니다.")
    if media.kind != kind:
        raise HTTPException(400, "이미지가 아닙니다." if kind == "image" else "영상이 아닙니다.")
    return media


@app.get("/api/media/{media_id}")
def media_info(media_id: str):
    return _media(media_id).public()


@app.get("/api/media/{media_id}/image")
def media_image(media_id: str):
    media = _ready(_media(media_id), "image")
    return FileResponse(media.src, headers=IMMUTABLE)


class SegmentRequest(BaseModel):
    prompt: str = Field("hair", min_length=1, max_length=100)


@app.post("/api/media/{media_id}/segment")
def media_segment(media_id: str, req: SegmentRequest):
    """Auto-select: SAM 3.1 mask (PNG, white = found) of every instance of `prompt` in the image."""
    media = _ready(_media(media_id), "image")
    st = sam.status()
    if not st["available"]:
        raise HTTPException(409, st["reason"])
    try:
        png = sam.segment(media, req.prompt.strip().lower())
    except RuntimeError as exc:
        raise HTTPException(422, str(exc))
    return Response(png, media_type="image/png")


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
    media = _ready(_media(req.media_id), "video")
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


class UpscaleRequest(BaseModel):
    media_id: str
    media_ids: list[str] | None = None  # images: every image of the batch (media_id is the first)
    image_format: str = "png"
    image_quality: int = Field(92, ge=1, le=100)
    scale: int = 2
    mode: str = "anime"
    backend: str = "ai"
    denoise: str = "none"
    format: str = "mp4_h265"
    quality: str = "balanced"
    interpolation: bool = False
    audio: bool = True  # keep the source audio and subtitle tracks


@app.post("/api/upscale-jobs")
def create_upscale_job(req: UpscaleRequest):
    first = _media(req.media_id)
    if first.kind == "image":
        return _create_image_upscale_job(req)
    media = _ready(first, "video")
    choices = (("scale", upscale.SCALES), ("mode", upscale.MODES), ("backend", upscale.BACKENDS),
               ("denoise", upscale.DENOISE), ("format", upscale.FORMATS), ("quality", upscale.QUALITIES))
    for key, allowed in choices:
        if getattr(req, key) not in allowed:
            raise HTTPException(400, f"{key} 값이 올바르지 않습니다.")
    engines = upscale.engines()
    if req.backend == "ai" and not engines["realesrgan"]:
        raise HTTPException(409, "서버에 Real-ESRGAN 엔진이 설치되지 않았습니다. 관리자에게 setup-upscale-engines.bat 실행을 요청하세요.")
    if req.interpolation and not engines["rife"]:
        raise HTTPException(409, "서버에 RIFE 엔진이 설치되지 않았습니다. 관리자에게 setup-upscale-engines.bat 실행을 요청하세요.")
    side = max(media.width, media.height) * req.scale
    if side > config.UPSCALE_MAX_SIDE:
        raise HTTPException(400, f"출력이 너무 큽니다 (긴 변 {side}px, 최대 {config.UPSCALE_MAX_SIDE}px). 배율을 낮추세요.")
    params = {"kind": "upscale", "upscale": req.model_dump(exclude={"media_id"})}
    job = registry.submit(media, params)
    return registry.job_public(job)


_CL = cleanup.LIMITS


class CleanupRequest(BaseModel):
    media_id: str
    denoise: float = Field(cleanup.DEFAULTS["denoise"], ge=_CL["denoise"][0], le=_CL["denoise"][1])
    steps: int = Field(cleanup.DEFAULTS["steps"], ge=_CL["steps"][0], le=_CL["steps"][1])
    guidance: float = Field(cleanup.DEFAULTS["guidance"], ge=_CL["guidance"][0], le=_CL["guidance"][1])
    feather: int = Field(cleanup.DEFAULTS["feather"], ge=_CL["feather"][0], le=_CL["feather"][1])
    count: int = Field(cleanup.DEFAULTS["count"], ge=_CL["count"][0], le=_CL["count"][1])
    seed: int | None = Field(None, ge=0, le=2**31 - 1)  # None: random
    prompt: str = Field(cleanup.DEFAULT_PROMPT, max_length=2000)
    mask: str | None = None  # PNG data URL, alpha = area to clean; None cleans the whole image


@app.post("/api/cleanup-jobs")
def create_cleanup_job(req: CleanupRequest):
    media = _ready(_media(req.media_id), "image")
    st = cleanup.status()
    if not st["available"]:
        raise HTTPException(409, st["reason"])
    if media.width * media.height > config.CLEANUP_MAX_MP * 1e6:
        raise HTTPException(400, f"이미지가 너무 큽니다 ({media.width}×{media.height}, 최대 {config.CLEANUP_MAX_MP:g}MP).")
    files = {}
    if req.mask:
        try:
            mask = cleanup.decode_mask(req.mask, (media.width, media.height))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        buf = io.BytesIO()
        mask.save(buf, "PNG")
        files["mask.png"] = buf.getvalue()
    p = req.model_dump(exclude={"media_id", "mask"})
    p.update(masked=bool(files), seed=req.seed if req.seed is not None else secrets.randbelow(2**31 - 4),
             prompt=req.prompt.strip() or cleanup.DEFAULT_PROMPT)
    job = registry.submit(media, {"kind": "cleanup", "cleanup": p}, files)
    return registry.job_public(job)


def _create_image_upscale_job(req: UpscaleRequest):
    ids = req.media_ids or [req.media_id]
    if len(ids) > upscale.MAX_IMAGES:
        raise HTTPException(400, f"한 번에 {upscale.MAX_IMAGES}장까지 올릴 수 있습니다.")
    images = [_ready(_media(i), "image") for i in dict.fromkeys(ids)]
    choices = (("scale", upscale.SCALES), ("mode", upscale.MODES), ("backend", upscale.BACKENDS),
               ("denoise", upscale.DENOISE), ("image_format", upscale.IMAGE_FORMATS))
    for key, allowed in choices:
        if getattr(req, key) not in allowed:
            raise HTTPException(400, f"{key} 값이 올바르지 않습니다.")
    if req.backend == "ai" and not upscale.engines()["realesrgan"]:
        raise HTTPException(409, "서버에 Real-ESRGAN 엔진이 설치되지 않았습니다. 관리자에게 setup-upscale-engines.bat 실행을 요청하세요.")
    for m in images:
        side = max(m.width, m.height) * req.scale
        if side > config.UPSCALE_MAX_SIDE:
            raise HTTPException(400, f"{m.name}: 출력이 너무 큽니다 (긴 변 {side}px, 최대 {config.UPSCALE_MAX_SIDE}px). 배율을 낮추세요.")
    p = req.model_dump(include={"scale", "mode", "backend", "denoise", "image_format", "image_quality"})
    p.update(image=True, media_ids=[m.id for m in images],
             images=[{"src": str(m.src), "name": m.name} for m in images])
    job = registry.submit(images[0], {"kind": "upscale", "upscale": p})
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


@app.post("/api/jobs/{job_id}/to-editor")
def job_to_editor(job_id: str):
    """Opens an upscaled result in the timing editor by registering it as new media."""
    job = _job(job_id)
    if job.kind != "upscale" or job.status != "done" or not job.outputs or job.params["upscale"].get("image"):
        raise HTTPException(409, "완료된 영상 업스케일 작업만 편집기로 보낼 수 있습니다.")
    return registry.media_from_output(job, job.outputs[0]["name"]).public()
