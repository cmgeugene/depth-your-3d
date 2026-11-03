"""SAM 3.1 text-prompt segmentation worker for the cleanup page's auto-select ("hair" -> every character's hair).

Runs in SAM 3.1's own Python environment (DS_SAM3_PYTHON), not in the studio's .venv: sam3 needs numpy<2.
Imports nothing from the server package for that reason.

  python sam_worker.py --serve                    JSON lines on stdin: {"image", "prompt", "out"}
                                                  answers {"ok": true, "count": n} or {"ok": false, "error"}
  python sam_worker.py IMAGE PROMPT OUT.png       one-shot, for testing

OUT is the union of all instances found, white on black. The checkpoint path comes from DS_SAM3_CHECKPOINT.
The multiplex checkpoint is a video predictor, so each image is a one-frame video.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


class Detector:
    def __init__(self):
        import torch
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        checkpoint = os.environ.get("DS_SAM3_CHECKPOINT", "")
        if not Path(checkpoint).is_file():
            raise RuntimeError(f"SAM 3.1 체크포인트가 없습니다: {checkpoint or '(DS_SAM3_CHECKPOINT 미설정)'}")
        if not torch.cuda.is_available():
            raise RuntimeError("SAM 3.1은 CUDA GPU가 필요합니다.")
        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint, use_fa3=False, use_rope_real=True, async_loading_frames=False)
        # start_session passes offload_state_to_cpu, which the multiplex model's init_state does not accept.
        init_state = self.predictor.model.init_state
        self.predictor.model.init_state = lambda _init=init_state, **kw: _init(
            **{k: v for k, v in kw.items() if k != "offload_state_to_cpu"})

    def detect(self, image_path: str, prompt: str, out_path: str) -> int:
        img = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)  # non-ASCII paths are fine
        if img is None:
            raise RuntimeError("이미지를 읽지 못했습니다.")
        frames = Path(tempfile.mkdtemp(prefix="ds_sam3_"))
        try:
            cv2.imwrite(str(frames / "00000.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
            session = self.predictor.handle_request({"type": "start_session", "resource_path": str(frames)})["session_id"]
            try:
                out = self.predictor.handle_request(
                    {"type": "add_prompt", "session_id": session, "frame_index": 0, "text": prompt})["outputs"]
            finally:
                self.predictor.handle_request({"type": "close_session", "session_id": session})
        finally:
            shutil.rmtree(frames, ignore_errors=True)
        masks = np.asarray(out["out_binary_masks"]).astype(bool)
        union = masks.any(axis=0) if len(masks) else np.zeros(img.shape[:2], bool)
        if union.shape != img.shape[:2]:
            union = cv2.resize(union.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        ok, png = cv2.imencode(".png", union.astype(np.uint8) * 255)
        Path(out_path).write_bytes(png.tobytes())
        return int(len(masks))


def serve():
    replies, sys.stdout = sys.stdout, sys.stderr  # sam3 prints progress; keep stdout for the replies only
    detector = None
    for line in sys.stdin:
        try:
            req = json.loads(line)
            detector = detector or Detector()
            reply = {"ok": True, "count": detector.detect(req["image"], req["prompt"], req["out"])}
        except Exception as exc:
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
        replies.write(json.dumps(reply, ensure_ascii=False) + "\n")
        replies.flush()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:] == ["--serve"]:
        serve()
    else:
        print(Detector().detect(*sys.argv[1:4]), "instances")
