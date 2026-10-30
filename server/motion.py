"""Per-frame motion, used by the editor to find where to animate on 1s.

Optical flow between consecutive small grayscale frames is split into a global part,
fitted as an affine field (camera: pan, tilt, zoom, roll), and what is left over
(action: things moving inside the frame). Both are in percent of frame width per frame.
Frames are contrast-normalized first so flashing stage lights don't read as motion.

OpenCV is optional: without it uploads skip this step and the editor uses fixed steps.
"""
import importlib.util
from pathlib import Path

import numpy as np

AVAILABLE = importlib.util.find_spec("cv2") is not None
WIDTH = 192  # analysis width; prepare() writes gray frames at this size
GRID = 2  # sample every 2nd flow vector for the fit


def _normalize(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    std = f.std()
    return np.clip((f - f.mean()) / (std if std > 1 else 1) * 48 + 128, 0, 255).astype(np.uint8)


def _median3(v: np.ndarray) -> np.ndarray:
    # Drops one-frame spikes (cuts, glitches) without smearing real motion.
    if len(v) < 3:
        return v
    p = np.pad(v, 1, mode="edge")
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def analyze(raw_path: Path, frames: int) -> dict:
    import cv2

    data = np.fromfile(raw_path, np.uint8)
    h = data.size // (frames * WIDTH)
    imgs = data[: frames * WIDTH * h].reshape(frames, h, WIDTH)
    ys, xs = np.mgrid[0:h:GRID, 0:WIDTH:GRID]
    basis = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 1).astype(np.float32)

    camera = np.zeros(frames, np.float32)
    action = np.zeros(frames, np.float32)
    prev = _normalize(imgs[0])
    for i in range(1, frames):
        cur = _normalize(imgs[i])
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 13, 3, 5, 1.1, 0)
        vec = flow[::GRID, ::GRID].reshape(-1, 2)
        keep = np.ones(len(vec), bool)
        for _ in range(2):  # refit without the moving subjects so they don't pull the camera estimate
            coef = np.linalg.lstsq(basis[keep], vec[keep], rcond=None)[0]
            fit = basis @ coef
            res = np.linalg.norm(vec - fit, axis=1)
            keep = res <= max(float(np.median(res)) * 2.5, 0.05)
        camera[i] = np.linalg.norm(fit, axis=1).mean()
        action[i] = np.percentile(res, 97)
        prev = cur

    scale = 100.0 / WIDTH
    return {
        "camera": [round(float(v) * scale, 3) for v in _median3(camera)],
        "action": [round(float(v) * scale, 3) for v in _median3(action)],
    }
