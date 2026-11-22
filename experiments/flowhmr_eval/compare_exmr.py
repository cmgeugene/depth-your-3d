"""Current clay approach (SAM 3D Body + Gaussian sigma 2) vs ExMR's camera-frame SMPL-X, on one clip.

  D:\\Projects\\FlowHMR\\.venv\\Scripts\\python.exe compare_exmr.py <flowhmr_root> <video_stem> <exmr.npz> [out_dir]

<exmr.npz> comes from exmr_to_npz.py, run on ExMR's output for the same 30 fps video, so frame k is
the same image in both. Alignment and jitter are measured against ViTPose (refs.npz), as before.
Writes <stem>_exmr_compare.mp4 (2x2: source | clay-like SAM + s2 | ExMR raw | ExMR + s2) and
<stem>_exmr_metrics.json.
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

root, stem, exmr_path = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3]
out_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out_dir, exist_ok=True)
run_dir = os.path.join(root, "output", stem)
A = np.load(os.path.join(run_dir, "refs.npz"))
E = np.load(exmr_path)
frames = A["frames"]
ok = frames < len(E["frames"])
frames, ia = frames[ok], np.nonzero(ok)[0]
KA = A["K"].astype(np.float64)
KE = E["K"][frames].astype(np.float64)                         # per frame
MHR2COCO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
vit, vs = A["vit_kp2d"][ia], A["vit_score"][ia]
W, H = int(round(KA[0, 2] * 2)), int(round(KA[1, 2] * 2))
m = (vs > 0.5) & (vit[..., 0] > 0) & (vit[..., 0] < W) & (vit[..., 1] > 0) & (vit[..., 1] < H)


def project(X, K):
    K = K if K.ndim == 3 else K[None]
    f = np.stack([K[:, 0, 0], K[:, 1, 1]], -1)[:, None]
    c = K[:, :2, 2][:, None]
    if X.ndim == 2:
        return X[:, :2] / np.clip(X[:, 2:3], 1e-3, None) * f[0] + c[0]
    return X[..., :2] / np.clip(X[..., 2:3], 1e-3, None) * f + c


def gauss(x, s=2.0):
    r = int(np.ceil(3 * s))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
    p = np.concatenate([x[:1].repeat(r, 0), x, x[-1:].repeat(r, 0)])
    return sum(wk * p[i:i + len(x)] for i, wk in enumerate(w)) / w.sum()


def metrics(p2d):
    acc = np.linalg.norm(p2d[2:] - 2 * p2d[1:-1] + p2d[:-2], axis=-1)
    err = np.linalg.norm(p2d - vit, axis=-1)
    return {"align_err_px_median": float(np.median(err[m])), "align_err_px_mean": float(err[m].mean()),
            "jitter_px_per_frame2_upper": float(acc[:, :11].mean()),
            "jitter_px_per_frame2_all": float(acc.mean())}


sam3 = A["sam_kp3d"][ia][:, MHR2COCO].astype(np.float64)
ex3 = E["coco3d"][frames].astype(np.float64)
report = {
    "frames": int(len(frames)),
    "SAM_per_frame": metrics(project(sam3, KA)),
    "SAM_sigma2_like_clay": metrics(project(gauss(sam3), KA)),
    "ExMR_incam_raw": metrics(E["coco2d"][frames].astype(np.float64)),
    "ExMR_incam_sigma2": metrics(project(gauss(ex3), KE)),
    "ExMR_vs_own_2d_check_px": float(np.median(np.linalg.norm(project(ex3, KE) - E["coco2d"][frames], axis=-1))),
    "depth_z_relative_jitter_pct": {
        "SAM_per_frame": float(100 * np.abs(np.diff(sam3[:, 5:7, 2].mean(1), 2)).mean() / sam3[:, 5:7, 2].mean()),
        "SAM_sigma2": float(100 * np.abs(np.diff(gauss(sam3)[:, 5:7, 2].mean(1), 2)).mean() / sam3[:, 5:7, 2].mean()),
        "ExMR_raw": float(100 * np.abs(np.diff(ex3[:, 5:7, 2].mean(1), 2)).mean() / ex3[:, 5:7, 2].mean()),
        "ExMR_sigma2": float(100 * np.abs(np.diff(gauss(ex3)[:, 5:7, 2].mean(1), 2)).mean() / ex3[:, 5:7, 2].mean()),
    },
}
with open(os.path.join(out_dir, f"{stem}_exmr_metrics.json"), "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report, indent=1))

# ---- render ----
SCALE = 0.5
LIGHT = np.array([0.3, -0.4, -1.0]) / np.linalg.norm([0.3, -0.4, -1.0])


def draw_mesh(img, V, F, K, color, alpha=0.75):
    Ks = K.copy()
    Ks[:2] *= SCALE
    P = project(V, Ks)
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


sam_v = gauss(A["sam_verts"][ia].astype(np.float64))
ex_v = E["verts"][frames].astype(np.float64)
ex_vs = gauss(ex_v)
sf, ef = A["sam_faces"], E["faces"]
cap = cv2.VideoCapture(os.path.join(run_dir, f"{stem}_30fps.mp4"))
pw, ph = int(W * SCALE), int(H * SCALE)
tmp = os.path.join(out_dir, f"{stem}_exmr_tmp.mp4")
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 30, (pw * 2, ph * 2))
idx = {int(f): k for k, f in enumerate(frames)}
fi = -1
while True:
    okf, bgr = cap.read()
    if not okf:
        break
    fi += 1
    if fi not in idx:
        continue
    k = idx[fi]
    base = cv2.resize(bgr, (pw, ph), interpolation=cv2.INTER_AREA)
    a = label(base.copy(), f"source  f{fi}")
    b = label(draw_mesh(base, sam_v[k], sf, KA, (200, 200, 200)), "SAM 3D Body + s2 (~clay)")
    c = label(draw_mesh(base, ex_v[k], ef, KE[k], (150, 200, 240)), "ExMR camera-frame, raw")
    d = label(draw_mesh(base, ex_vs[k], ef, KE[k], (150, 220, 170)), "ExMR camera-frame + s2")
    vw.write(np.vstack([np.hstack([a, b]), np.hstack([c, d])]))
vw.release()
try:
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    ff = "ffmpeg"
final = os.path.join(out_dir, f"{stem}_exmr_compare.mp4")
subprocess.run([ff, "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", final], check=True)
os.remove(tmp)
print("saved", final)
