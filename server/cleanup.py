"""Image cleanup jobs: FLUX.1 [dev] img2img at low denoise to wipe the mottled grain and speckle that
GPT Image leaves on 2D art, optionally only inside a painted mask.

Two layers keep the untouched area exactly as it was:
  1. While sampling, latents outside a slightly dilated mask are reset to the noised source on every
     step (FluxInpaintPipeline with strength = denoise), so the model only redraws the masked area.
  2. After decoding, the result is blended onto the source pixels with a feathered mask. The VAE round
     trip alone would shift every pixel a little, so without this the "kept" area would still change.

The transformer is Nunchaku's SVDQuant INT4 build (~7GB VRAM) so it can share a 24GB card with SAM.
T5/CLIP never stay loaded: each prompt is encoded once and its embeddings are cached on disk.
The model is unloaded after DS_FLUX_IDLE_MIN minutes without jobs to give the VRAM back.
"""
import base64
import gc
import hashlib
import io
import threading
import time

from PIL import Image, ImageFilter, ImageOps

from . import config
from .pipeline import Cancelled

# The prompt of the ComfyUI workflow this tool replaces (flux1_dev_i2i_pixel_noise_cleanup).
DEFAULT_PROMPT = (
    "the exact same image, preserve the original composition, characters, faces, line art, shapes, colors, "
    "lighting and framing; clean coherent surfaces, smooth intentional gradients, crisp illustration detail, "
    "remove only fine global pixel noise, speckling, checkerboard texture and digital grain; "
    "no redesign, no new objects, no pose change"
)
DEFAULTS = {"denoise": 0.18, "steps": 20, "guidance": 3.5, "feather": 12, "count": 1}
LIMITS = {"denoise": (0.05, 0.5), "steps": (4, 50), "guidance": (1.0, 8.0), "feather": (0, 64), "count": (1, 4)}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}  # uploads treated as images
COMFY_SHIFT = 1.15  # ComfyUI's fixed Flux schedule shift (mu)
GPU_HEADROOM_FOR_T5 = 11 * 1024**3  # free VRAM needed to encode a prompt on the GPU instead of the CPU

_lock = threading.Lock()  # one user of the pipeline at a time (GPU worker vs. idle unloader)
_pipe = None
_last_used = 0.0
_unloader_started = False


# ---------- availability ----------

def _missing() -> str | None:
    try:
        import diffusers  # noqa: F401
    except ImportError:
        return "diffusers가 설치되지 않았습니다 (pip install -r requirements.txt)."
    try:
        import nunchaku  # noqa: F401
    except ImportError:
        return "Nunchaku가 설치되지 않았습니다. 서버에서 setup-flux.bat을 실행하세요."
    except Exception as exc:  # wheel built for another torch/CUDA
        return f"Nunchaku를 불러오지 못했습니다 ({exc}). setup-flux.bat을 다시 실행하세요."
    import torch

    if not torch.cuda.is_available():
        return "CUDA GPU가 없어 얼룩 정리를 쓸 수 없습니다."
    return None


_missing_cache: list = []


def status() -> dict:
    if not _missing_cache:
        _missing_cache.append(_missing())
    return {
        "available": _missing_cache[0] is None,
        "reason": _missing_cache[0],
        "model": "FLUX.1 [dev] · Nunchaku INT4",
        "offload": _offload(),
        "max_mp": config.CLEANUP_MAX_MP,
        "defaults": {**DEFAULTS, "prompt": DEFAULT_PROMPT},
        "limits": LIMITS,
    }


def _offload() -> bool:
    if config.FLUX_OFFLOAD in ("on", "off"):
        return config.FLUX_OFFLOAD == "on"
    try:
        import torch

        return torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3
    except Exception:
        return False


# ---------- request helpers (used by app.py) ----------

def decode_mask(data_url: str, size: tuple[int, int]) -> Image.Image:
    """PNG data URL from the page -> L mask (255 = clean here). Raises ValueError with a user message."""
    try:
        raw = base64.b64decode(data_url.split(",", 1)[-1], validate=True)
        mask = Image.open(io.BytesIO(raw))
        mask.load()
    except Exception:
        raise ValueError("마스크 이미지를 읽지 못했습니다.")
    if mask.size != size:
        raise ValueError("마스크 크기가 이미지와 다릅니다.")
    if mask.mode in ("RGBA", "LA"):
        mask = mask.getchannel("A")  # the page paints the mask into the alpha channel
    mask = mask.convert("L")
    if mask.getbbox() is None:
        raise ValueError("정리할 영역이 없습니다. 마스크를 칠하거나 반전하세요.")
    return mask


def summary(media, p: dict) -> dict:
    outputs = [f"denoise {p['denoise']:.2f}", f"{p['steps']}스텝", "마스크" if p["masked"] else "전체"]
    if p["count"] > 1:
        outputs.append(f"{p['count']}장")
    return {
        "resolution": f"{media.width}×{media.height}",
        "frames": 1,
        "drawings": 1,
        "model": "FLUX.1 dev INT4",
        "outputs": outputs,
    }


# ---------- models ----------

def _transformer_path() -> str:
    if config.FLUX_TRANSFORMER:
        return config.FLUX_TRANSFORMER
    from nunchaku.utils import get_precision

    # int4 on RTX 20/30/40, fp4 on 50-series (Blackwell).
    return f"nunchaku-ai/nunchaku-flux.1-dev/svdq-{get_precision()}_r32-flux.1-dev.safetensors"


def _load_pipeline(log):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, FluxInpaintPipeline
    from nunchaku import NunchakuFluxTransformer2dModel

    offload = _offload()
    path = _transformer_path()
    log(f"Flux 불러오는 중: {path} (offload={'on' if offload else 'off'})")
    t0 = time.time()
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(path, offload=offload)
    pipe = FluxInpaintPipeline.from_pretrained(
        config.FLUX_REPO, transformer=transformer, text_encoder=None, text_encoder_2=None,
        tokenizer=None, tokenizer_2=None, torch_dtype=torch.bfloat16,
    )
    # diffusers raises the noise-schedule shift with image size (1.15 at 1MP, ~1.9 at 2MP), so the same
    # denoise would start from much more noise on big images. ComfyUI keeps it at 1.15; do the same so
    # denoise means the same thing at every size and matches the ComfyUI workflow.
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config, base_shift=COMFY_SHIFT, max_shift=COMFY_SHIFT)
    if offload:
        # Nunchaku streams its own blocks; diffusers offloads the rest (only the VAE is left here).
        pipe._exclude_from_cpu_offload.append("transformer")
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_tiling()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    log(f"Flux 준비 완료 ({time.time() - t0:.0f}초)")
    return pipe


def _get_pipeline(log):
    global _pipe, _unloader_started
    if _pipe is None:
        _pipe = _load_pipeline(log)
    if not _unloader_started and config.FLUX_IDLE_MIN > 0:
        _unloader_started = True
        threading.Thread(target=_unloader, daemon=True, name="flux-unloader").start()
    return _pipe


def _unloader():
    global _pipe
    while True:
        time.sleep(30)
        with _lock:
            if _pipe is not None and time.time() - _last_used > config.FLUX_IDLE_MIN * 60:
                _pipe = None
                _free()
                print(f"Flux: {config.FLUX_IDLE_MIN:g}분 동안 작업이 없어 VRAM에서 내렸습니다.", flush=True)


def _free():
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _prompt_embeds(prompt: str, log):
    """(prompt_embeds, pooled_prompt_embeds) for `prompt`, from the disk cache when possible."""
    import torch

    key = hashlib.sha1(f"{config.FLUX_REPO}\n512\n{prompt}".encode()).hexdigest()
    cache = config.FLUX_CACHE_DIR / f"{key}.pt"
    if cache.exists():
        d = torch.load(cache, map_location="cpu")
        return d["prompt_embeds"], d["pooled_prompt_embeds"]

    from diffusers import FluxPipeline

    free, _ = torch.cuda.mem_get_info()
    device = "cuda" if free > GPU_HEADROOM_FOR_T5 else "cpu"
    log(f"프롬프트 인코딩 (T5/CLIP, {device}) — 같은 프롬프트는 다음부터 캐시를 씁니다.")
    t0 = time.time()
    enc = FluxPipeline.from_pretrained(config.FLUX_REPO, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    try:
        enc.text_encoder.to(device)
        enc.text_encoder_2.to(device)
        with torch.inference_mode():
            pe, pooled, _ = enc.encode_prompt(prompt=prompt, prompt_2=None, device=device, max_sequence_length=512)
        pe, pooled = pe.cpu(), pooled.cpu()
    finally:
        del enc
        _free()
    config.FLUX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    torch.save({"prompt_embeds": pe, "pooled_prompt_embeds": pooled, "prompt": prompt}, tmp)
    tmp.replace(cache)
    log(f"프롬프트 인코딩 완료 ({time.time() - t0:.0f}초)")
    return pe, pooled


# ---------- masks and images ----------

def _pad16(img: Image.Image) -> Image.Image:
    """Flux needs sides that are multiples of 16. Pad with edge pixels instead of resizing, so the source
    is never resampled; the padding is cropped off again after decoding."""
    w, h = img.size
    pw, ph = -w % 16, -h % 16
    if not (pw or ph):
        return img
    import numpy as np

    mode = "edge" if img.mode != "L" else "constant"
    arr = np.asarray(img)
    pad = ((0, ph), (0, pw)) + (((0, 0),) if arr.ndim == 3 else ())
    return Image.fromarray(np.pad(arr, pad, mode=mode))


def _dilate(mask: Image.Image, px: int) -> Image.Image:
    """Grows the mask by about `px` pixels (box blur + threshold is much faster than a big MaxFilter)."""
    if px <= 0:
        return mask
    return mask.filter(ImageFilter.BoxBlur(px)).point(lambda v: 255 if v > 0 else 0)


def _masks(mask: Image.Image | None, size: tuple[int, int], feather: int):
    """-> (latent mask for sampling, blend mask for the final composite, or None for the whole image)."""
    if mask is None:
        return Image.new("L", size, 255), None
    blend = mask.filter(ImageFilter.GaussianBlur(feather / 2)) if feather else mask
    # The sampler works in 16px patches; regenerate a margin wide enough to cover the feather and the
    # patch rounding, so the blend never mixes in latents that were held at the source.
    return _dilate(mask, feather + 16), blend


def _match_tone(out: Image.Image, src: Image.Image, region: Image.Image) -> tuple[Image.Image, list]:
    """Undoes the haze of Flux's VAE round trip: blacks rise, whites drop and contrast/saturation fall a few
    percent (it happens with the VAE alone, in fp32 too). Fits a gain and offset per channel from the result
    to the source over the regenerated region and applies it. The fit uses blurred copies, so it follows the
    tone shift and leaves the fine grain the cleanup removed where it is."""
    import numpy as np

    blur = ImageFilter.GaussianBlur(3)
    o = np.asarray(out.filter(blur), np.float32)
    s = np.asarray(src.filter(blur), np.float32)
    sel = np.asarray(region) > 127
    if sel.sum() < 1000:
        return out, []
    step = max(1, int(sel.sum() // 200_000))  # a sample is plenty for 2 numbers per channel
    res = np.asarray(out, np.float32).copy()
    fits = []
    for c in range(3):
        x, y = o[..., c][sel][::step], s[..., c][sel][::step]
        gain, offset = np.polyfit(x, y, 1)
        gain = float(np.clip(gain, 0.9, 1.15))  # a tone fix, never a recolour
        offset = float(np.clip(y.mean() - gain * x.mean(), -20, 20))
        res[..., c] = res[..., c] * gain + offset
        fits.append((round(gain, 3), round(offset, 1)))
    return Image.fromarray(np.clip(res + 0.5, 0, 255).astype(np.uint8)), fits


class _VramWatch:
    """Polls device-wide used memory so the log also shows what other processes (SAM services) held."""

    def __init__(self):
        import torch

        self.torch = torch
        torch.cuda.reset_peak_memory_stats()
        free, self.total = torch.cuda.mem_get_info()
        self.base = self.peak = self.total - free
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop.wait(0.2):
            free, total = self.torch.cuda.mem_get_info()
            self.peak = max(self.peak, total - free)

    def report(self) -> str:
        self._stop.set()
        self._t.join()
        gb = 1024**3
        own = self.torch.cuda.max_memory_reserved() / gb
        return (f"VRAM: 이 서버 피크 {own:.1f}GB · GPU 전체 피크 {self.peak / gb:.1f}GB / {self.total / gb:.0f}GB "
                f"(시작 시 {self.base / gb:.1f}GB 사용 중)")


# ---------- job ----------

def run(job, media, progress) -> list[dict]:
    global _last_used
    import torch

    p = job.params["cleanup"]
    log_file = open(job.dir / "cleanup.log", "a", encoding="utf-8")

    def log(msg: str):
        line = f"[cleanup {job.id[:8]}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    def check_cancel():
        if job.cancel.is_set():
            raise Cancelled()

    try:
        src = Image.open(media.src)
        src.load()
        icc = src.info.get("icc_profile")
        src = ImageOps.exif_transpose(src)
        alpha = src.getchannel("A") if src.mode in ("RGBA", "LA", "PA") else None
        if src.mode == "P" and "transparency" in src.info:
            alpha = src.convert("RGBA").getchannel("A")
        rgb = src.convert("RGB")
        w, h = rgb.size
        mask = Image.open(job.dir / "mask.png").convert("L") if p["masked"] else None
        latent_mask, blend = _masks(mask, (w, h), p["feather"])

        progress(0.02, "프롬프트 준비")
        pe, pooled = _prompt_embeds(p["prompt"], log)
        check_cancel()

        with _lock:
            progress(0.05, "모델 불러오는 중")
            watch = _VramWatch()
            pipe = _get_pipeline(log)
            check_cancel()
            # Like ComfyUI's KSampler: `steps` is the number of steps actually run, so the full schedule
            # is steps / denoise long and sampling starts at the matching noise level.
            total_steps = max(p["steps"], round(p["steps"] / p["denoise"]))
            image16, mask16 = _pad16(rgb), _pad16(latent_mask)
            W, H = image16.size
            embeds = {  # given embeddings are used as passed, so they must already be on the GPU
                "prompt_embeds": pe.to("cuda", torch.bfloat16), "pooled_prompt_embeds": pooled.to("cuda", torch.bfloat16)}
            results = []
            t0 = time.time()
            for i in range(p["count"]):
                seed = p["seed"] + i

                def on_step(_pipe, step, _t, kwargs, i=i):
                    if job.cancel.is_set():
                        raise Cancelled()
                    frac = (i + (step + 1) / p["steps"]) / p["count"]
                    progress(0.08 + 0.88 * min(frac, 1.0), f"정리 중 {i + 1}/{p['count']} · {min(step + 1, p['steps'])}/{p['steps']}스텝")
                    return kwargs

                out = pipe(
                    **embeds, image=image16, mask_image=mask16, height=H, width=W, strength=p["denoise"],
                    num_inference_steps=total_steps, guidance_scale=p["guidance"],
                    generator=torch.Generator("cpu").manual_seed(seed), callback_on_step_end=on_step,
                ).images[0].crop((0, 0, w, h))
                out, fits = _match_tone(out, rgb, latent_mask)
                log(f"톤 보정 (채널별 gain, offset): {fits}")
                if blend is not None:
                    out = Image.composite(out, rgb, blend)
                if alpha is not None:
                    out.putalpha(alpha)
                results.append((seed, out))
            _last_used = time.time()
            log(f"{p['count']}장 완료 ({time.time() - t0:.1f}초, {W}×{H}, 실제 {p['steps']}스텝 / 전체 {total_steps}) · {watch.report()}")

        progress(0.98, "저장 중")
        outputs = []
        for seed, img in results:
            name = f"{media.stem}_clean_d{round(p['denoise'] * 100):02d}_seed{seed}.png"
            img.save(job.dir / name, icc_profile=icc) if icc else img.save(job.dir / name)
            outputs.append({"name": name, "label": f"정리 결과 · seed {seed}", "size": (job.dir / name).stat().st_size})
        return outputs
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("GPU 메모리가 부족합니다. 다른 GPU 작업(SAM 등)이 끝난 뒤 다시 시도하거나 "
                           "DS_FLUX_OFFLOAD=on으로 서버를 실행하세요.")
    except Exception as exc:
        name = type(exc).__name__
        if name in ("GatedRepoError", "RepositoryNotFoundError") or "gated" in str(exc).lower():
            raise RuntimeError(f"{config.FLUX_REPO} 모델을 받을 권한이 없습니다. 서버 관리자가 Hugging Face에서 "
                               "라이선스에 동의하고 setup-flux.bat을 실행해야 합니다.")
        raise
    finally:
        _last_used = time.time()
        log_file.close()
