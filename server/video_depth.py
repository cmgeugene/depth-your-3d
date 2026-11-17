"""Video Depth Anything: temporally consistent depth for whole shots.

Depth Anything V2 looks at one frame at a time, so surfaces (hair, cloth, skin) wobble from
frame to frame. VDA adds temporal attention over a window of frames. Two ways to run it:

- offline: 32-frame windows with overlap, aligned to each other. Most consistent; needs every
  frame of the shot in memory (at working size) and more VRAM.
- streaming: one frame at a time, reusing cached temporal states. Little VRAM (Large fits in
  ~5GB), slightly less accurate (an experimental, training-free mode upstream).

Each shot starts fresh so nothing leaks across a hard cut. Output has the same convention as
depth.predict: relative inverse depth, bigger = nearer.
"""
import threading
import warnings

import numpy as np
import torch

from . import config
from .depth import DEVICE

_CFG = {
    "vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
    "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
}
_models: dict[str, torch.nn.Module] = {}
_lock = threading.Lock()


def is_video_model(key: str) -> bool:
    return "vda" in config.MODELS[key]


def get_model(key: str, stream: bool) -> torch.nn.Module:
    from huggingface_hub import hf_hub_download

    spec = config.MODELS[key]["vda"]
    cache_key = f"{key}:{'stream' if stream else 'offline'}"
    with _lock:
        if cache_key not in _models:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # "xFormers not available": plain attention is used
                if stream:
                    from .vda.video_depth_anything.video_depth_stream import VideoDepthAnything
                else:
                    from .vda.video_depth_anything.video_depth import VideoDepthAnything
            model = VideoDepthAnything(**_CFG[spec["encoder"]])
            path = hf_hub_download(spec["repo"], spec["file"])
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            _models[cache_key] = model.to(DEVICE).eval()
        return _models[cache_key]


def _reset_stream(model: torch.nn.Module):
    model.transform = None
    model.frame_id_list = []
    model.frame_cache_list = []
    model.id = -1


def predict_shot(model: torch.nn.Module, frames: list[np.ndarray], input_size: int) -> np.ndarray:
    """Offline: RGB uint8 frames of one shot (already at working size) -> float16 (N, h, w)."""
    depths, _ = model.infer_video_depth(np.stack(frames), 24, input_size=input_size,
                                        device=DEVICE.type, fp32=DEVICE.type != "cuda")
    return np.clip(np.asarray(depths), 0, None).astype(np.float16)


class Stream:
    """Streaming: feed frames one by one; call start() at the first frame of every shot."""

    def __init__(self, model: torch.nn.Module, input_size: int):
        self.model, self.input_size = model, input_size

    def start(self):
        _reset_stream(self.model)

    def predict(self, frame: np.ndarray) -> np.ndarray:
        d = self.model.infer_video_depth_one(frame, input_size=self.input_size, device=DEVICE.type,
                                             fp32=DEVICE.type != "cuda")
        return np.clip(d, 0, None).astype(np.float16)
