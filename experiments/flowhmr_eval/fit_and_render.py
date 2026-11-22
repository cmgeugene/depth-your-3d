"""Place FlowHMR's world-space SMPL-H motion in the camera and compare it with per-frame SAM 3D Body.

FlowHMR's output lives in its own gravity-aligned world (root starts at the origin); it carries no
camera placement. With a static camera one rigid transform (R, t) maps that world into the camera
for the whole clip, so it is fitted once against ViTPose 2D keypoints (independent of both methods).

  D:\\Projects\\FlowHMR\\.venv\\Scripts\\python.exe fit_and_render.py <flowhmr_root> <video_stem> [out_dir]

Writes <out_dir>/<stem>_compare.mp4 (2x2: SAM per-frame | SAM + Gaussian sigma 2 (~ clay) |
FlowHMR static camera + follow fit | FlowHMR VGGT camera + smooth t, or the static rigid fit without VGGT)
and <stem>_metrics.json. Uses refs.npz from per_frame_refs.py. flowhmr_latest_static.npz is the
static-camera run; flowhmr_latest.npz with <stem>_vggt_camera.npz is the VGGT run.
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import torch

root, stem = os.path.abspath(sys.argv[1]), sys.argv[2]
out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out_dir, exist_ok=True)
run_dir = os.path.join(root, "output", stem)

ref = np.load(os.path.join(run_dir, "refs.npz"))
fl = np.load(os.path.join(run_dir, "flowhmr_latest.npz"))
body = np.load(os.path.join(root, "assets", "body_models", "smplh", "neutral", "model.npz"))
frames, K = ref["frames"], ref["K"].astype(np.float64)
T = len(frames)

# MHR70 -> COCO-17 order (MHR70 has no wrists in its first 17; they sit at 62 / 41)
MHR2COCO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
# SMPL-H joints / vertices -> COCO-17 (face points are surface vertices, the rest are joints)
FACE_V = [332, 2800, 6260, 583, 4071]           # nose, l eye, r eye, l ear, r ear
BODY_J = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]  # shoulders, elbows, wrists, hips, knees, ankles
JOINT_W = np.array([1, 1, 1, .7, .7, 1, 1, 1, 1, 1, 1, .5, .5, 1, 1, 1, 1], np.float32)  # hips sit differently


def rodrigues(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True) + 1e-8
    k = aa / th
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    Kx = np.stack([np.zeros_like(kx), -kz, ky, kz, np.zeros_like(kx), -kx, -ky, kx, np.zeros_like(kx)], -1)
    Kx = Kx.reshape(aa.shape[:-1] + (3, 3))
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3) + s * Kx + (1 - c) * Kx @ Kx


def smplh(poses, betas, trans):
    """Plain SMPL-H linear blend skinning. Returns vertices (T,V,3) and joints (T,52,3)."""
    vt, sd, pd = body["v_template"], body["shapedirs"][..., :betas.shape[-1]], body["posedirs"]
    Jr, W, par = body["J_regressor"], body["weights"], body["kintree_table"][0].astype(int)
    v_shaped = vt + sd @ betas
    J = Jr @ v_shaped
    R = rodrigues(poses.reshape(len(poses), -1, 3))                     # (T,52,3,3)
    pf = (R[:, 1:] - np.eye(3)).reshape(len(poses), -1)                  # (T,459)
    v_posed = v_shaped[None] + np.einsum("vcp,tp->tvc", pd, pf)
    G = np.zeros((len(poses), 52, 4, 4))
    G[:, :, 3, 3] = 1
    G[:, 0, :3, :3], G[:, 0, :3, 3] = R[:, 0], J[0]
    for i in range(1, 52):
        L = np.eye(4)[None].repeat(len(poses), 0)
        L[:, :3, :3], L[:, :3, 3] = R[:, i], J[i] - J[par[i]]
        G[:, i] = G[:, par[i]] @ L
    Jt = G[:, :, :3, 3].copy()
    G[:, :, :3, 3] -= np.einsum("tjab,jb->tja", G[:, :, :3, :3], J)
    A = np.einsum("vj,tjab->tvab", W, G)
    v = np.einsum("tvab,tvb->tva", A[..., :3, :3], v_posed) + A[..., :3, 3]
    return (v + trans[:, None]).astype(np.float32), (Jt + trans[:, None]).astype(np.float32)


def project(X, K):
    z = np.clip(X[..., 2:3], 1e-3, None)
    return (X[..., :2] / z) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


def gaussian_smooth(x, sigma=2.0):
    r = int(np.ceil(3 * sigma))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    pad = np.concatenate([x[:1].repeat(r, 0), x, x[-1:].repeat(r, 0)])
    out = np.zeros_like(x)
    for k, wk in enumerate(w):
        out += wk * pad[k:k + len(x)]
    return out / w.sum()


# ---- FlowHMR in its world ----
fr = frames - frames[0]
fv, fj = smplh(fl["poses"][fr].astype(np.float64), fl["betas"][0].astype(np.float64), fl["trans"][fr].astype(np.float64))
f_coco = np.concatenate([fv[:, FACE_V], fj[:, BODY_J]], 1)          # (T,17,3) world

# ---- fit one rigid transform world -> camera against ViTPose ----
vit, vs = ref["vit_kp2d"], ref["vit_score"]
H, W = int(round(K[1, 2] * 2)), int(round(K[0, 2] * 2))
inside = (vit[..., 0] > 0) & (vit[..., 0] < W) & (vit[..., 1] > 0) & (vit[..., 1] < H)
wgt = torch.tensor(np.where((vs > 0.5) & inside, vs, 0) * JOINT_W, dtype=torch.float32)
tgt = torch.tensor(vit, dtype=torch.float32)
X = torch.tensor(f_coco, dtype=torch.float32)
Kt = torch.tensor(K, dtype=torch.float32)


def aa_to_mat(r):
    th = r.norm() + 1e-8
    k = r / th
    Kx = torch.stack([torch.zeros(()), -k[2], k[1], k[2], torch.zeros(()), -k[0], -k[1], k[0], torch.zeros(())]).reshape(3, 3)
    return torch.eye(3) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


flip = torch.diag(torch.tensor([1.0, -1.0, -1.0]))  # y-up world -> OpenCV camera
best = None
sh_px = np.nanmedian(np.where(wgt[:, 5:7].numpy().min(1) > 0, np.linalg.norm(vit[:, 5] - vit[:, 6], axis=1), np.nan))
z0 = K[0, 0] * 0.36 / max(sh_px, 1.0)
for yaw in np.deg2rad([0, 90, 180, 270]):
    r = torch.tensor([0.0, float(yaw), 0.0], requires_grad=True)
    t = torch.tensor([0.0, 0.0, float(z0)], requires_grad=True)
    opt = torch.optim.Adam([r, t], lr=0.02)
    for it in range(800):
        Rm = flip @ aa_to_mat(r)
        C = X @ Rm.T + t
        uv = C[..., :2] / C[..., 2:3].clamp(min=1e-3) * torch.stack([Kt[0, 0], Kt[1, 1]]) + torch.stack([Kt[0, 2], Kt[1, 2]])
        loss = (torch.nn.functional.huber_loss(uv, tgt, reduction="none", delta=20).sum(-1) * wgt).sum() / wgt.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    if best is None or loss.item() < best[0]:
        best = (loss.item(), (flip @ aa_to_mat(r)).detach().numpy().astype(np.float64), t.detach().numpy().astype(np.float64))
_, R_fit, t_fit = best
fv_cam = fv @ R_fit.T + t_fit
f2d = project(f_coco @ R_fit.T + t_fit, K)

# ---- camera-follow fit: one rotation, per-frame translation with a strong smoothness prior ----
# A tracking shot moves the camera with the person; under the static-camera assumption that shows up
# as world travel. A smooth per-frame camera translation absorbs it without re-adding frame jitter.
# With a VGGT camera, its per-frame rotation (relative to frame 0) is applied too:
#   C_k = Rrel_k @ Ra @ X_k + t_k
SMOOTH_W = 50.0
fx = float(K[0, 0])
vis = wgt.numpy() > 0
cen2d = np.array([vit[k][vis[k]].mean(0) if vis[k].any() else vit[k].mean(0) for k in range(T)])


def uv_of(C):
    return C[..., :2] / C[..., 2:3].clamp(min=1e-3) * torch.stack([Kt[0, 0], Kt[1, 1]]) + torch.stack([Kt[0, 2], Kt[1, 2]])


def reproj(C):
    return (torch.nn.functional.huber_loss(uv_of(C), tgt, reduction="none", delta=20).sum(-1) * wgt).sum() / wgt.sum()


def fit_follow(Xw, Rrel=None):
    """Ra (one rotation) + smooth per-frame t. Returns (loss, Ra, t (T,3), Rrel (T,3,3))."""
    Rrel = np.repeat(np.eye(3)[None], T, 0) if Rrel is None else Rrel
    Rr = torch.tensor(Rrel, dtype=torch.float32)
    Xt = torch.tensor(Xw, dtype=torch.float32)
    out = None
    for yaw in np.deg2rad([0, 90, 180, 270]):
        R0 = (flip @ aa_to_mat(torch.tensor([0.0, float(yaw), 0.0]))).numpy()
        Xc = np.einsum("tab,tjb->tja", Rrel @ R0, Xw)
        ray = np.concatenate([(cen2d - K[:2, 2]) / fx, np.ones((T, 1))], 1) * z0
        r = torch.tensor([0.0, float(yaw), 0.0], requires_grad=True)
        tk = torch.tensor(ray - Xc.mean(1), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([r, tk], lr=0.01)
        for it in range(1500):
            Ra = flip @ aa_to_mat(r)
            C = torch.einsum("tab,tjb->tja", Rr @ Ra, Xt) + tk[:, None]
            acc = (tk[2:] - 2 * tk[1:-1] + tk[:-2]) * (fx / z0)
            loss = reproj(C) + SMOOTH_W * (acc ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if out is None or loss.item() < out[0]:
            out = (loss.item(), (flip @ aa_to_mat(r)).detach().numpy().astype(np.float64),
                   tk.detach().numpy().astype(np.float64), Rrel)
    return out


def fit_vggt_full(Xw, Rrel, trel):
    """Pure VGGT camera: C_k = Rrel_k @ (Ra @ X_k + t0) + s * trel_k. Only Ra, t0, s are free."""
    Rr, tr = torch.tensor(Rrel, dtype=torch.float32), torch.tensor(trel, dtype=torch.float32)
    Xt = torch.tensor(Xw, dtype=torch.float32)
    out = None
    for yaw in np.deg2rad([0, 90, 180, 270]):
        r = torch.tensor([0.0, float(yaw), 0.0], requires_grad=True)
        t0 = torch.tensor([0.0, 0.0, float(z0)], requires_grad=True)
        ls = torch.tensor(0.0, requires_grad=True)                     # log scale of VGGT translation
        opt = torch.optim.Adam([r, t0, ls], lr=0.02)
        for it in range(1200):
            Ra = flip @ aa_to_mat(r)
            C = torch.einsum("tab,tjb->tja", Rr, Xt @ Ra.T + t0) + torch.exp(ls) * tr[:, None]
            loss = reproj(C)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if out is None or loss.item() < out[0]:
            Ra = (flip @ aa_to_mat(r)).detach().numpy().astype(np.float64)
            out = (loss.item(), Ra, t0.detach().numpy().astype(np.float64), float(torch.exp(ls)))
    return out


def place_follow(V, fit):
    _, Ra, tk, Rrel = fit
    return np.einsum("tab,tvb->tva", Rrel @ Ra, V) + tk[:, None]


def flow_coco(npz):
    v, j = smplh(npz["poses"][fr].astype(np.float64), npz["betas"][0].astype(np.float64), npz["trans"][fr].astype(np.float64))
    return v, np.concatenate([v[:, FACE_V], j[:, BODY_J]], 1)


static_path = os.path.join(run_dir, "flowhmr_latest_static.npz")
fl_static = np.load(static_path) if os.path.exists(static_path) else fl
fv_s, fc_s = flow_coco(fl_static)
fit_s = fit_follow(fc_s)
fv_fol, f2d_fol = place_follow(fv_s, fit_s), project(place_follow(fc_s, fit_s), K)

vggt_path = os.path.join(run_dir, f"{stem}_vggt_camera.npz")
fv_vg = f2d_vg = f2d_vgfull = None
if os.path.exists(vggt_path) and os.path.exists(static_path):
    RT = np.load(vggt_path)["camera_RT"].astype(np.float64)[fr]        # cam <- world, OpenCV
    M = RT @ np.linalg.inv(RT[0])                                       # cam_k <- cam_0
    fv_v, fc_v = flow_coco(fl)                                          # flowhmr_latest.npz = VGGT run
    fit_v = fit_follow(fc_v, M[:, :3, :3])
    fv_vg, f2d_vg = place_follow(fv_v, fit_v), project(place_follow(fc_v, fit_v), K)
    full = fit_vggt_full(fc_v, M[:, :3, :3], M[:, :3, 3])
    Cf = np.einsum("tab,tjb->tja", M[:, :3, :3], fc_v @ full[1].T + full[2]) + full[3] * M[:, None, :3, 3]
    f2d_vgfull = project(Cf, K)

# ---- SAM 3D Body per frame, and with the clay's temporal Gaussian (sigma 2) ----
s_verts = ref["sam_verts"]
s_verts_sm = gaussian_smooth(s_verts)
s2d = ref["sam_kp2d"][:, MHR2COCO]
s2d_sm = project(gaussian_smooth(ref["sam_kp3d"])[:, MHR2COCO], K)


def metrics(kp):
    m = wgt.numpy() > 0
    err = np.linalg.norm(kp - vit, axis=-1)
    acc = np.linalg.norm(kp[2:] - 2 * kp[1:-1] + kp[:-2], axis=-1)       # px / frame^2
    upper = slice(0, 11)
    return {"align_err_px_median": float(np.median(err[m])), "align_err_px_mean": float(err[m].mean()),
            "jitter_px_per_frame2_upper": float(acc[:, upper].mean()),
            "jitter_px_per_frame2_all": float(acc.mean())}


report = {"frames": int(T), "fit_loss": best[0], "R_fit": R_fit.tolist(), "t_fit": t_fit.tolist(),
          "SAM_per_frame": metrics(s2d), "SAM_gauss_sigma2_like_clay": metrics(s2d_sm),
          "FlowHMR_static_rigid_fit": metrics(f2d),
          "FlowHMR_static_camera_follow_fit": metrics(f2d_fol),
          **({"FlowHMR_vggt_rotation_plus_smooth_t": metrics(f2d_vg),
              "FlowHMR_vggt_full_camera_global_fit": metrics(f2d_vgfull)} if f2d_vg is not None else {}),
          "ViTPose_own_jitter_upper": float(np.linalg.norm(vit[2:] - 2 * vit[1:-1] + vit[:-2], axis=-1)[:, :11].mean())}
with open(os.path.join(out_dir, f"{stem}_metrics.json"), "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps({k: v for k, v in report.items() if k not in ("R_fit", "t_fit")}, indent=1))

# ---- render ----
SCALE = 0.5
Ks = K.copy()
Ks[:2] *= SCALE
LIGHT = np.array([0.3, -0.4, -1.0])
LIGHT /= np.linalg.norm(LIGHT)


def draw_mesh(img, V, F, color, alpha=0.75):
    P = project(V, Ks)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    c = tri.mean(1)
    n *= np.sign(((c - V.mean(0)) * n).sum(1).mean())             # outward normals whatever the winding
    front = (n * c).sum(1) < 0                                   # faces looking at the camera
    order = np.argsort(-c[front][:, 2])
    Ff, nf = F[front][order], n[front][order]
    shade = np.clip(-(nf @ LIGHT), 0.15, 1.0)
    layer, mask = img.copy(), np.zeros(img.shape[:2], np.uint8)
    pts = np.round(P[Ff] * 4).astype(np.int32)                   # 2 bits sub-pixel
    for p, s in zip(pts, shade):
        col = tuple(int(ch * s) for ch in color)
        cv2.fillConvexPoly(layer, p, col, lineType=cv2.LINE_AA, shift=2)
        cv2.fillConvexPoly(mask, p, 255, shift=2)
    m = (mask > 0)[..., None]
    return np.where(m, (alpha * layer + (1 - alpha) * img).astype(np.uint8), img)


def label(img, text):
    cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


s_faces, f_faces = ref["sam_faces"], body["f"].astype(np.int32)
cap = cv2.VideoCapture(os.path.join(run_dir, f"{stem}_30fps.mp4"))
tmp = os.path.join(out_dir, f"{stem}_compare_tmp.mp4")
pw, ph = int(W * SCALE), int(H * SCALE)
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 30, (pw * 2, ph * 2))
idx = {int(f): k for k, f in enumerate(frames)}
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
    a = label(draw_mesh(base, s_verts[k], s_faces, (200, 200, 200)), f"SAM 3D Body per-frame  f{fi}")
    b = label(draw_mesh(base, s_verts_sm[k], s_faces, (170, 200, 230)), "SAM 3D Body + Gauss s2 (~clay)")
    c = label(draw_mesh(base, fv_fol[k], f_faces, (150, 220, 160)), "FlowHMR static cam + follow fit")
    if fv_vg is not None:
        d = label(draw_mesh(base, fv_vg[k], f_faces, (120, 200, 240)), "FlowHMR VGGT cam + smooth t fit")
    else:
        d = label(draw_mesh(base, fv_cam[k], f_faces, (150, 220, 160)), "FlowHMR static cam, one rigid fit")
    vw.write(np.vstack([np.hstack([a, b]), np.hstack([c, d])]))
    if k % 60 == 0:
        print(f"  render {k}/{T}", flush=True)
vw.release()
final = os.path.join(out_dir, f"{stem}_compare.mp4")
ff = "ffmpeg"
try:
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
subprocess.run([ff, "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", final], check=True)
os.remove(tmp)
print("saved", final)
