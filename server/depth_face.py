"""Simplify face and neck silhouettes in depth maps (medium shots and close-ups).

In a close-up the subject/background edge of a depth map *is* the actor's profile: nose tip,
lips, chin and the step from jaw to neck. A video model driven by that depth copies the
live-action jawline onto a 2D face. Here a face parser finds face+neck per frame, and only
that region is rewritten: concave notches (under the nose, lip-chin, chin-neck corner) are
filled, thin protrusions (nose tip, lips) are shaved, and depth inside the new shape is
low-passed so the jaw-to-neck step fades. Hair, body and background keep their depth, so
the motion and head direction they carry stay intact.

Strength ramps with face size, so wide shots are left alone and a push-in eases into it.
"""
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .depth import DEVICE, DTYPE

REPO = "jonathandinu/face-parsing"  # SegFormer trained on CelebAMask-HQ (19 classes)
FACE_IDS = list(range(1, 13))  # skin, nose, glasses, eyes, brows, ears, mouth, lips
NECK_IDS = [16, 17]  # necklace, neck
HAIR_IDS = [13, 14]  # hair, hat
CHANNELS = 3  # stored probabilities: face, neck, hair
_PARSE_SIZE = 512
_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

# Face area (fraction of the frame) where the effect starts and where it reaches full strength.
AREA_START, AREA_FULL = 0.012, 0.035

_model = None
_lock = threading.Lock()


def get_model() -> torch.nn.Module:
    from transformers import SegformerForSemanticSegmentation

    global _model
    with _lock:
        if _model is None:
            m = SegformerForSemanticSegmentation.from_pretrained(REPO, dtype=DTYPE)
            _model = m.to(DEVICE).eval()
        return _model


def _parse(model: torch.nn.Module, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """x: float RGB 0..1 (B, 3, H, W) -> (B, 3, *size) probabilities of [face, neck, hair]."""
    x = F.interpolate(x, size=(_PARSE_SIZE, _PARSE_SIZE), mode="bilinear", align_corners=False, antialias=True)
    x = ((x - _MEAN) / _STD).to(DTYPE)
    p = F.interpolate(model(pixel_values=x).logits.float(), size=size, mode="bilinear", align_corners=False).softmax(1)
    return torch.stack([p[:, FACE_IDS].sum(1), p[:, NECK_IDS].sum(1), p[:, HAIR_IDS].sum(1)], 1)


@torch.inference_mode()
def predict(model: torch.nn.Module, frames: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """RGB uint8 frames -> uint8 (B, 3, h, w) probabilities (0..255) of [face, neck, hair] at `size`.

    The parser takes a square input, so a wide frame is squeezed and its edges land several
    pixels off. A second pass on a square crop around the largest face fixes that.
    """
    x = torch.from_numpy(np.stack(frames)).to(DEVICE).permute(0, 3, 1, 2).float().div_(255)
    out = _parse(model, x, size)
    H, W = x.shape[-2:]
    h, w = size
    crops, boxes = [], []
    for b in range(len(frames)):
        m = ((out[b, 0] + out[b, 1]) > 0.5).cpu().numpy().astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n < 2:
            continue
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bx, by, bw, bh, area = stats[i]
        if area < 0.002 * h * w:
            continue
        side = 1.8 * max(bw * W / w, bh * H / h)
        side = min(side, H, W)
        cx, cy = (bx + bw / 2) * W / w, (by + bh / 2) * H / h
        x0 = int(round(min(max(cx - side / 2, 0), W - side)))
        y0 = int(round(min(max(cy - side / 2, 0), H - side)))
        s_ = int(round(side))
        # Crop box in working (h, w) coordinates; skip crops too small to help.
        wx0, wy0 = round(x0 * w / W), round(y0 * h / H)
        wx1, wy1 = round((x0 + s_) * w / W), round((y0 + s_) * h / H)
        if wx1 - wx0 < 8 or wy1 - wy0 < 8:
            continue
        crops.append(x[b:b + 1, :, y0:y0 + s_, x0:x0 + s_])
        boxes.append((b, wx0, wy0, wx1, wy1))
    for (b, wx0, wy0, wx1, wy1), crop in zip(boxes, crops):
        out[b, :, wy0:wy1, wx0:wx1] = _parse(model, crop, (wy1 - wy0, wx1 - wx0))[0]
    return (out * 255 + 0.5).clamp_(0, 255).to(torch.uint8).cpu().numpy()


def face_stats(probs: np.ndarray) -> tuple[float, float]:
    """(face area as a fraction of the frame, face scale in pixels = sqrt(face area))."""
    area = float((probs[0] > 127).sum())
    return area / probs[0].size, area ** 0.5


def strength_for(area_frac: float, strength: float) -> float:
    t = min(max((area_frac - AREA_START) / (AREA_FULL - AREA_START), 0.0), 1.0)
    return strength * t * t * (3 - 2 * t)


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0.3 else img


def _push_pull(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid pixels with a smooth extension of the valid ones (pyramid hole filling)."""
    w = valid.astype(np.float32)
    levels = [(img * w, w)]
    while min(levels[-1][1].shape) > 4 and levels[-1][1].min() <= 0:
        vw, ww = levels[-1]
        levels.append((cv2.pyrDown(vw), cv2.pyrDown(ww)))
    vw, ww = levels[-1]
    cur = np.where(ww > 1e-6, vw / np.maximum(ww, 1e-6), (vw.sum() / max(ww.sum(), 1e-6)))
    for vw, ww in reversed(levels[:-1]):
        up = cv2.pyrUp(cur, dstsize=(ww.shape[1], ww.shape[0]))
        a = np.clip(ww, 0, 1)
        cur = np.where(ww > 1e-6, vw / np.maximum(ww, 1e-6), 0) * a + up * (1 - a)
    return cur.astype(np.float32)


def _norm_blur(img: np.ndarray, valid: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    w = valid.astype(np.float32)
    num, den = _blur(img * w, sigma), _blur(w, sigma)
    return num / np.maximum(den, 1e-6), den


def simplify(v: np.ndarray, probs: np.ndarray, s: float, scale_px: float, invert: bool) -> np.ndarray:
    """Rewrite the face+neck silhouette of depth `v` (float32 0..1, output size).

    probs: (3, h, w) [face, neck, hair] probabilities 0..255 at working size.
    s: strength 0..1 for this frame. scale_px: face scale (sqrt of face area) in working pixels.
    """
    if s <= 0.01 or scale_px < 8:
        return v
    H, W = v.shape
    h, w = probs.shape[1:]
    near_full = 1 - v if invert else v
    near = cv2.resize(near_full, (w, h), interpolation=cv2.INTER_AREA)
    L = scale_px
    hair = probs[2] > 127

    fm = (probs[0] + probs[1]) > 127
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fm.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 0.05 * L * L
    fm = keep[lab]
    if not fm.any():
        return v

    # Background = clearly farther than the face. Only the face/background edge is reshaped;
    # where the face meets hair or clothing nothing is cut or added.
    ring = cv2.dilate(fm.astype(np.uint8), _disk(max(2, round(0.25 * L)))).astype(bool) & ~fm & ~hair
    face_d = float(np.median(near[fm]))
    bg_d = float(np.median(near[ring])) if ring.any() else 0.0
    if face_d - bg_d < 0.04:
        return v  # face is not separated from what is behind it; nothing to reshape
    # Hair hanging behind the neck counts as background too: it is what the jaw is seen against.
    bg = (near < face_d - 0.35 * (face_d - bg_d)) & ~fm
    # Snap the parser mask to the depth edge: in a thin band around it, face-depth pixels that
    # are not hair belong to the face.
    band = cv2.dilate(fm.astype(np.uint8), _disk(max(1, round(0.05 * L)))).astype(bool)
    fm = fm | (band & ~bg & ~hair)

    # New silhouette: close concave notches, open thin protrusions. Morphology runs at a
    # reduced size where the face is ~64 px across, so the radii stay small and fast.
    k = max(1.0, L / 64)
    sw, sh = max(8, round(w / k)), max(8, round(h / k))
    m = cv2.resize(fm.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_AREA)
    m = (m > 127).astype(np.uint8)
    Ls = L / k
    r_close, r_open = max(1, round(0.4 * Ls * s)), max(1, round(0.22 * Ls * s))
    pad = r_close + 2
    m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(r_close))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _disk(r_open))
    m = m[pad:-pad, pad:-pad].astype(np.float32)
    m = _blur(m, max(0.6, 0.05 * Ls * s))  # smooth the contour itself
    g = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5

    near_bg = cv2.dilate(bg.astype(np.uint8), _disk(max(1, round(0.3 * L)))).astype(bool)
    added = g & ~fm & bg  # background that the new face shape covers
    # Face that sticks out into the background (nose tip, lips). Skin seen between strands of
    # hair is left alone, or the gaps in the fringe would turn into holes.
    # Only the face is shaved; the neck keeps its width.
    removed = fm & ~g & near_bg & (probs[0] >= probs[1])
    removed &= ~cv2.dilate(hair.astype(np.uint8), _disk(max(1, round(0.08 * L)))).astype(bool)
    inside = fm & ~removed

    # Depth inside the new shape: low-pass of the face's own depth, extended outward.
    core = cv2.erode(fm.astype(np.uint8), _disk(max(1, round(0.03 * L)))).astype(bool)
    if not core.any():
        core = fm
    inner, wsum = _norm_blur(near, core, max(1.0, 0.12 * L * s))
    face_val = _push_pull(inner, wsum > 0.2)
    # Depth where the old face is removed: extend the background around it.
    bg_val = _push_pull(near, bg & ~cv2.dilate(fm.astype(np.uint8), _disk(2)).astype(bool))
    bg_val = _blur(bg_val, max(1.0, 0.03 * L))

    new = near.copy()
    new[inside] = face_val[inside]
    new[added] = face_val[added]
    new[removed] = bg_val[removed]

    changed = cv2.dilate((inside | added | removed).astype(np.uint8), _disk(2)).astype(np.float32)
    alpha = _blur(changed, 1.0) * min(1.0, s / 0.15)
    new_full = cv2.resize(new, (W, H), interpolation=cv2.INTER_LINEAR)
    a_full = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_LINEAR)
    out = np.clip(near_full * (1 - a_full) + new_full * a_full, 0, 1)
    return (1 - out if invert else out).astype(np.float32)
