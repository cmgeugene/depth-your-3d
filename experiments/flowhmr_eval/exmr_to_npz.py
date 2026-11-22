"""ExMR results.npz -> camera-frame SMPL-X vertices and COCO-17 2D joints for the comparison scripts.

Run with ExMR's Python (needs the smplx package and SMPLX_NEUTRAL.npz):

  D:\\Projects\\ExMR\\.venv\\Scripts\\python.exe exmr_to_npz.py <exmr_output_dir> <out.npz> [exmr_root]

Writes verts (L,10475,3) camera frame (x right, y down, z forward), faces, coco2d (L,17,2) from
ExMR's own joints_2d, K (L,3,3) and frames (L,).
"""
import os
import sys

import numpy as np
import smplx
import torch

src, out = sys.argv[1], sys.argv[2]
root = sys.argv[3] if len(sys.argv) > 3 else r"D:\Projects\ExMR"
r = np.load(os.path.join(src, "results.npz"), allow_pickle=True)
names = [str(n) for n in r["joint_names"]]
p = {k[len("smplx_incam_"):]: torch.tensor(r[k], dtype=torch.float32) for k in r.files if k.startswith("smplx_incam_")}
L = len(p["body_pose"])
model = smplx.SMPLX(os.path.join(root, "checkpoints", "body_models", "smplx", "SMPLX_NEUTRAL.npz"),
                    use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=L)
with torch.no_grad():
    o = model(**p)
verts = o.vertices.numpy().astype(np.float32)

COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]
missing = [n for n in COCO if n not in names]
if missing:
    raise SystemExit(f"joint names missing in ExMR output: {missing}\navailable: {names}")
coco2d = r["joints_2d"][:, [names.index(n) for n in COCO]].astype(np.float32)
coco3d = r["joints_incam"][:, [names.index(n) for n in COCO]].astype(np.float32)
np.savez_compressed(out, verts=verts, faces=model.faces.astype(np.int32), coco2d=coco2d, coco3d=coco3d,
                    K=r["K_fullimg"].astype(np.float32), frames=np.arange(L))
print(f"saved {L} frames, verts {verts.shape} -> {out}")
