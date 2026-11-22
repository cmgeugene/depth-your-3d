"""Per-frame references for one FlowHMR run: SAM 3D Body (full, camera space) and ViTPose 2D keypoints.

Run with FlowHMR's Python (it imports sam_3d_body from FlowHMR's third_party by default):

  D:\\Projects\\FlowHMR\\.venv\\Scripts\\python.exe per_frame_refs.py <flowhmr_root> <video_stem>
      [--sam-repo DIR] [--out NAME.npz] [--no-vitpose] [--limit N]

Reads output/<stem>/<stem>_30fps.mp4, <stem>_bbox.npz and <stem>_static_camera.npz (or
<stem>_vggt_camera.npz) and writes output/<stem>/<out> (default refs.npz) with, per track frame:
  sam_verts (T,V,3) camera space, sam_kp2d (T,70,2), sam_kp3d (T,70,3), sam_cam_t (T,3),
  sam_faces (F,3), sam_sec (T,) SAM time per frame, vit_kp2d (T,17,2), vit_score (T,17), frames, K.
SAM 3D Body gets the same intrinsics FlowHMR used, so both methods share one camera.
--sam-repo points at another sam_3d_body checkout with the same API (e.g. Fast SAM 3D Body; its speed
switches are environment variables set by the caller).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("stem")
ap.add_argument("--sam-repo", default="")
ap.add_argument("--out", default="refs.npz")
ap.add_argument("--no-vitpose", action="store_true")
ap.add_argument("--limit", type=int, default=0, help="only the first N track frames (timing runs)")
args = ap.parse_args()
root, stem = os.path.abspath(args.root), args.stem
sys.path.insert(0, root)
sys.path.insert(0, os.path.abspath(args.sam_repo) if args.sam_repo else os.path.join(root, "third_party", "sam-3d-body"))
out_dir = os.path.join(root, "output", stem)

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: E402
import sam_3d_body  # noqa: E402
print("sam_3d_body from", os.path.dirname(sam_3d_body.__file__), flush=True)

bb = np.load(os.path.join(out_dir, f"{stem}_bbox.npz"))
boxes, (start, end) = bb["bbox"], bb["start_end"].astype(int)
if args.limit:
    end = min(end, start + args.limit - 1)
cam_path = os.path.join(out_dir, f"{stem}_static_camera.npz")
if not os.path.exists(cam_path):
    cam_path = os.path.join(out_dir, f"{stem}_vggt_camera.npz")
K_all = np.load(cam_path)["K"]

dev = "cuda"
ck = os.path.join(root, "ckpts", "sam-3d-body-dinov3")
model, cfg = load_sam_3d_body(checkpoint_path=os.path.join(ck, "model.ckpt"),
                              mhr_path=os.path.join(ck, "assets", "mhr_model.pt"), device=dev)
est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)

if not args.no_vitpose:
    from transformers import AutoProcessor, VitPoseForPoseEstimation  # noqa: E402
    vp_proc = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
    vp = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(dev).eval()

cap = cv2.VideoCapture(os.path.join(out_dir, f"{stem}_30fps.mp4"))
keys = ["frames", "sam_verts", "sam_kp2d", "sam_kp3d", "sam_cam_t", "sam_sec"]
res = {k: [] for k in keys + ([] if args.no_vitpose else ["vit_kp2d", "vit_score"])}
fi, t0 = -1, time.time()
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    fi += 1
    if fi < start:
        continue
    if fi > end:
        break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    box = boxes[fi - start].astype(np.float32)
    K = K_all[min(fi - start, len(K_all) - 1)] if len(K_all) > 1 else K_all[0]
    torch.cuda.synchronize()
    ts = time.perf_counter()
    with torch.no_grad():
        outs = est.process_one_image(rgb, bboxes=box[None], cam_int=torch.as_tensor(K[None], dtype=torch.float32))
    torch.cuda.synchronize()
    sec = time.perf_counter() - ts
    if not outs:
        continue
    o = outs[0]
    res["frames"].append(fi)
    res["sam_sec"].append(sec)
    res["sam_verts"].append(np.asarray(o["pred_vertices"], np.float32) + np.asarray(o["pred_cam_t"], np.float32))
    res["sam_kp2d"].append(np.asarray(o["pred_keypoints_2d"], np.float32))
    res["sam_kp3d"].append(np.asarray(o["pred_keypoints_3d"], np.float32) + np.asarray(o["pred_cam_t"], np.float32))
    res["sam_cam_t"].append(np.asarray(o["pred_cam_t"], np.float32))
    if not args.no_vitpose:
        x0, y0, x1, y1 = box
        inp = vp_proc(rgb, boxes=[[[x0, y0, x1 - x0, y1 - y0]]], return_tensors="pt").to(dev)
        with torch.no_grad():
            po = vp_proc.post_process_pose_estimation(vp(**inp), boxes=[[[x0, y0, x1 - x0, y1 - y0]]])[0][0]
        res["vit_kp2d"].append(po["keypoints"].cpu().numpy().astype(np.float32))
        res["vit_score"].append(po["scores"].cpu().numpy().astype(np.float32))
    n = len(res["frames"])
    if n % 30 == 0:
        print(f"  {n}/{end - start + 1} frames, {n / (time.time() - t0):.2f} fps overall, "
              f"SAM median {np.median(res['sam_sec']):.2f} s/frame", flush=True)

out_path = os.path.join(out_dir, args.out)
np.savez_compressed(out_path, **{k: np.stack(v) for k, v in res.items()},
                    sam_faces=np.asarray(est.faces, np.int32), K=K_all[0])
sec = np.array(res["sam_sec"])
print(f"saved {len(res['frames'])} frames -> {out_path}")
print(f"SAM time: median {np.median(sec):.3f} s, mean (after first 5) {sec[5:].mean():.3f} s, first {sec[0]:.2f} s")
