"""Original SAM 3D Body vs Fast SAM 3D Body on the same frames, boxes and camera.

  D:\\Projects\\FlowHMR\\.venv\\Scripts\\python.exe compare_fast.py <flowhmr_root> <video_stem> [out_dir]

Needs output/<stem>/refs.npz (original, with ViTPose) and refs_fast.npz (Fast, from per_frame_refs.py
--sam-repo ... --no-vitpose). Reports speed, how far the meshes differ, and jitter / alignment against
ViTPose for both, raw and with the clay's Gaussian sigma 2. Writes <stem>_fast_compare.mp4 (side by
side, both with sigma 2 like the clay) and <stem>_fast_metrics.json.
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

root, stem = os.path.abspath(sys.argv[1]), sys.argv[2]
out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out_dir, exist_ok=True)
run_dir = os.path.join(root, "output", stem)
A = np.load(os.path.join(run_dir, "refs.npz"))
B = np.load(os.path.join(run_dir, "refs_fast.npz"))
K = A["K"].astype(np.float64)
common = np.intersect1d(A["frames"], B["frames"])
ia = np.searchsorted(A["frames"], common)
ib = np.searchsorted(B["frames"], common)
MHR2COCO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
vit, vs = A["vit_kp2d"][ia], A["vit_score"][ia]
W, H = int(round(K[0, 2] * 2)), int(round(K[1, 2] * 2))
m = (vs > 0.5) & (vit[..., 0] > 0) & (vit[..., 0] < W) & (vit[..., 1] > 0) & (vit[..., 1] < H)


def project(X):
    return X[..., :2] / np.clip(X[..., 2:3], 1e-3, None) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def gauss(x, s=2.0):
    r = int(np.ceil(3 * s))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
    p = np.concatenate([x[:1].repeat(r, 0), x, x[-1:].repeat(r, 0)])
    return sum(wk * p[i:i + len(x)] for i, wk in enumerate(w)) / w.sum()


def metrics(kp3d):
    p = project(kp3d[:, MHR2COCO])
    acc = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=-1)
    err = np.linalg.norm(p - vit, axis=-1)
    return {"align_err_px_median": float(np.median(err[m])), "jitter_px_per_frame2_upper": float(acc[:, :11].mean())}


ka, kb = A["sam_kp3d"][ia].astype(np.float64), B["sam_kp3d"][ib].astype(np.float64)
va, vb = A["sam_verts"][ia], B["sam_verts"][ib]
v_diff = np.linalg.norm(va - vb, axis=-1)                      # metres, camera space
loc_diff = np.linalg.norm((va - A["sam_cam_t"][ia][:, None]) - (vb - B["sam_cam_t"][ib][:, None]), axis=-1)
kp2_diff = np.linalg.norm(A["sam_kp2d"][ia][:, MHR2COCO] - B["sam_kp2d"][ib][:, MHR2COCO], axis=-1)
z_rel = np.abs(A["sam_cam_t"][ia][:, 2] - B["sam_cam_t"][ib][:, 2]) / A["sam_cam_t"][ia][:, 2]
sa, sb = A["sam_sec"] if "sam_sec" in A.files else None, B["sam_sec"]
timing_orig = os.path.join(run_dir, "timing_orig.npz")
if os.path.exists(timing_orig):
    sa = np.load(timing_orig)["sam_sec"]
report = {
    "frames": int(len(common)),
    "speed_s_per_frame": {"original_median": float(np.median(sa)) if sa is not None else None,
                          "fast_median": float(np.median(sb))},
    "mesh_difference": {"vertex_mm_median": float(np.median(v_diff) * 1000),
                        "vertex_mm_p95": float(np.percentile(v_diff, 95) * 1000),
                        "pose_only_vertex_mm_median": float(np.median(loc_diff) * 1000),
                        "kp2d_px_median": float(np.median(kp2_diff)), "kp2d_px_p95": float(np.percentile(kp2_diff, 95)),
                        "depth_rel_pct_median": float(np.median(z_rel) * 100)},
    "original_raw": metrics(ka), "fast_raw": metrics(kb),
    "original_sigma2_like_clay": metrics(gauss(ka)), "fast_sigma2_like_clay": metrics(gauss(kb)),
}
if report["speed_s_per_frame"]["original_median"]:
    report["speedup_x"] = report["speed_s_per_frame"]["original_median"] / report["speed_s_per_frame"]["fast_median"]
with open(os.path.join(out_dir, f"{stem}_fast_metrics.json"), "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report, indent=1))

# ---- side-by-side render, both with the clay's sigma 2 ----
SCALE = 0.5
Ks = K.copy()
Ks[:2] *= SCALE
LIGHT = np.array([0.3, -0.4, -1.0]) / np.linalg.norm([0.3, -0.4, -1.0])


def draw_mesh(img, V, F, color, alpha=0.75):
    P = V[..., :2] / np.clip(V[..., 2:3], 1e-3, None) * [Ks[0, 0], Ks[1, 1]] + [Ks[0, 2], Ks[1, 2]]
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    c = tri.mean(1)
    n *= np.sign(((c - V.mean(0)) * n).sum(1).mean())
    front = (n * c).sum(1) < 0
    order = np.argsort(-c[front][:, 2])
    Ff, nf = F[front][order], n[front][order]
    shade = np.clip(-(nf @ LIGHT), 0.15, 1.0)
    layer, mask = img.copy(), np.zeros(img.shape[:2], np.uint8)
    for p, s in zip(np.round(P[Ff] * 4).astype(np.int32), shade):
        cv2.fillConvexPoly(layer, p, tuple(int(ch * s) for ch in color), lineType=cv2.LINE_AA, shift=2)
        cv2.fillConvexPoly(mask, p, 255, shift=2)
    mk = (mask > 0)[..., None]
    return np.where(mk, (alpha * layer + (1 - alpha) * img).astype(np.uint8), img)


def label(img, text):
    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, th, cv2.LINE_AA)
    return img


va_s, vb_s = gauss(va.astype(np.float64)), gauss(vb.astype(np.float64))
faces = A["sam_faces"]
cap = cv2.VideoCapture(os.path.join(run_dir, f"{stem}_30fps.mp4"))
pw, ph = int(W * SCALE), int(H * SCALE)
tmp = os.path.join(out_dir, f"{stem}_fast_tmp.mp4")
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 30, (pw * 2, ph))
idx = {int(f): k for k, f in enumerate(common)}
fi = -1
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    fi += 1
    if fi not in idx:
        continue
    k = idx[fi]
    base = cv2.resize(bgr, (pw, ph), interpolation=cv2.INTER_AREA)
    a = label(draw_mesh(base, va_s[k], faces, (200, 200, 200)), f"SAM 3D Body + s2  f{fi}")
    b = label(draw_mesh(base, vb_s[k], faces, (230, 200, 160)), "Fast SAM 3D Body + s2")
    vw.write(np.hstack([a, b]))
vw.release()
try:
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    ff = "ffmpeg"
final = os.path.join(out_dir, f"{stem}_fast_compare.mp4")
subprocess.run([ff, "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", final], check=True)
os.remove(tmp)
print("saved", final)
