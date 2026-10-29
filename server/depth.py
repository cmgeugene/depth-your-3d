"""Depth Anything V2 inference on GPU batches."""
import threading

import numpy as np
import torch
import torch.nn.functional as F

from . import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

_models: dict[str, torch.nn.Module] = {}
_lock = threading.Lock()


def gpu_name() -> str:
    if DEVICE.type != "cuda":
        return "CPU (CUDA 없음)"
    props = torch.cuda.get_device_properties(0)
    return f"{props.name} {props.total_memory / 1024**3:.0f}GB"


def get_model(key: str) -> torch.nn.Module:
    from transformers import AutoModelForDepthEstimation

    with _lock:
        if key not in _models:
            repo = config.MODELS[key]["repo"]
            model = AutoModelForDepthEstimation.from_pretrained(repo, dtype=DTYPE)
            _models[key] = model.to(DEVICE).eval()
        return _models[key]


def infer_size(height: int, width: int, short_side: int) -> tuple[int, int]:
    """Scale so the short side matches the model input size; both sides multiples of 14 (ViT patch)."""
    scale = short_side / min(height, width)
    return max(14, round(height * scale / 14) * 14), max(14, round(width * scale / 14) * 14)


@torch.inference_mode()
def predict(model: torch.nn.Module, frames: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """RGB uint8 frames -> relative inverse depth (bigger = nearer), float16 (B, h, w) at `size`."""
    x = torch.from_numpy(np.stack(frames)).to(DEVICE).permute(0, 3, 1, 2).float().div_(255)
    x = F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True)
    x = ((x - _MEAN) / _STD).to(DTYPE)
    depth = model(pixel_values=x).predicted_depth
    if depth.shape[-2:] != size:
        depth = F.interpolate(depth[:, None].float(), size=size, mode="bilinear", align_corners=False)[:, 0]
    return depth.float().clamp_(min=0).cpu().numpy().astype(np.float16)


@torch.inference_mode()
def finish(maps: list[np.ndarray], weights: list[float], los: list[float], his: list[float],
           out_hw: tuple[int, int], invert: bool, gamma: float) -> np.ndarray:
    """Normalize (and temporally blend) raw depth maps, upsample to output size. Returns float32 0..1 (H, W)."""
    acc = None
    for raw, w, lo, hi in zip(maps, weights, los, his):
        d = torch.from_numpy(np.asarray(raw, dtype=np.float32)).to(DEVICE)
        d = ((d - lo) / max(hi - lo, 1e-6)).clamp_(0, 1) * w
        acc = d if acc is None else acc + d
    v = acc / sum(weights)
    if gamma != 1.0:
        v = v.pow(gamma)
    if invert:
        v = 1 - v
    v = F.interpolate(v[None, None], size=out_hw, mode="bilinear", align_corners=False)[0, 0]
    return v.clamp_(0, 1).cpu().numpy()
