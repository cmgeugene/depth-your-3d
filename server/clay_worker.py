"""Clay reference video worker: people in a live-action clip -> grey clay figures (SAM 3D Body).

Runs in SAM 3D Body's own Python environment (DS_CLAY_PYTHON) with its repository on the path
(DS_CLAY_REPO), not in the studio's .venv. Imports nothing from the server package for that reason.
The studio keeps one worker alive between jobs and kills it after DS_CLAY_IDLE_SEC idle seconds,
which gives all of its VRAM back.

  python clay_worker.py --serve      JSON lines on stdin (one job each), answers on stdout:
                                     {"progress": 0.42, "stage": "..."} while working, then
                                     {"ok": true, "people": n} or {"ok": false, "error": "..."}

Per job:
  1. every frame: people boxes (RT-DETR), IoU tracking that restarts at each cut, SAM 3D Body per
     person (MoGe-2 field of view), ViTPose wrist scores, SegFormer person mask (1/4 size).
  2. meshes rebuilt with a neutral head (shape and expression zeroed; "head" mode keeps the body
     shape, "neutral" zeroes it everywhere), one focal length per shot, temporal Gaussian smoothing,
     and forearms/hands dropped where the source shows no hand (SAM 3D Body always invents both arms).
  3. z-buffered grey clay render (moderngl), optionally over a depth video of the same clip.
"""
import json
import os
import subprocess
import sys
import traceback

os.environ.setdefault("PYOPENGL_PLATFORM", "")
REPO = os.environ.get("DS_CLAY_REPO", "")
if REPO:
    sys.path.insert(0, REPO)

import cv2
import numpy as np

DEV = "cuda"
CHECKPOINT = os.environ.get("DS_CLAY_CHECKPOINT", "facebook/sam-3d-body-dinov3")
PARAM_KEYS = ["global_rot", "body_pose_params", "hand_pose_params", "scale_params", "shape_params", "expr_params"]
FINGERS = ["pinky", "ring", "middle", "index", "thumb"]
WRIST = {"l": 9, "r": 10}  # COCO-17 left/right wrist, person-centric like MHR l_/r_
MHR2COCO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]  # MHR70 keypoints in COCO-17 order
MASK_DOWN = 4
GUIDE_TREND = 12  # frames: depth_guide keeps SAM's depth trend slower than this Gaussian sigma


def gaussian_smooth(x, idx, sigma):
    """Gaussian moving average over consecutive frame indices only (a gap or cut breaks the window)."""
    if sigma <= 0 or len(x) < 2:
        return x.copy()
    r = int(np.ceil(3 * sigma))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    pos = {int(f): k for k, f in enumerate(idx)}
    y = np.empty_like(x)
    for k, f in enumerate(idx):
        acc, ws = 0, 0.0
        for d, wd in zip(range(-r, r + 1), w):
            j = pos.get(int(f) + d)
            if j is not None:
                acc = acc + wd * x[j]
                ws += wd
        y[k] = acc / ws
    return y


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


class Engine:
    def __init__(self):
        import torch
        from transformers import (AutoProcessor, RTDetrForObjectDetection, RTDetrImageProcessor,
                                  SegformerForSemanticSegmentation, SegformerImageProcessor,
                                  VitPoseForPoseEstimation)
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body_hf
        from tools.build_fov_estimator import FOVEstimator

        if not torch.cuda.is_available():
            raise RuntimeError("SAM 3D Body는 CUDA GPU가 필요합니다.")
        self.torch = torch
        self.det_proc = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        self.det = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(DEV).eval()
        self.vp_proc = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        self.vp = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple").to(DEV).eval()
        self.seg_proc = SegformerImageProcessor()
        self.seg = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(DEV).eval()
        model, cfg = load_sam_3d_body_hf(CHECKPOINT)
        self.model = model.to(DEV).eval()
        self.head = self.model.head_pose
        self.est = SAM3DBodyEstimator(sam_3d_body_model=self.model, model_cfg=cfg, human_detector=None,
                                      human_segmentor=None, fov_estimator=FOVEstimator(name="moge2", device=DEV))
        self.faces = self.head.faces.cpu().numpy().astype(np.int64)
        self._skinning_groups()
        self.ctx = None

    # ---- mesh helpers ----
    def _skinning_groups(self):
        ct = self.head.mhr.character_torch
        names = list(ct.skeleton.joint_names)
        lbs = ct.linear_blend_skinning
        ji = lbs.skin_indices_flattened.cpu().numpy().astype(int)
        jw = lbs.skin_weights_flattened.cpu().numpy()
        vi = lbs.vert_indices_flattened.cpu().numpy().astype(int)
        nv = int(self.faces.max()) + 1

        def weight(joint_factor):
            jf = np.array([joint_factor(nm) for nm in names], np.float64)
            w = np.zeros(nv)
            np.add.at(w, vi, jw * jf[ji])
            return w

        def is_head(nm):
            if nm.startswith(("c_head", "c_jaw", "c_teeth", "c_tongue")) or nm.endswith(("_eye", "_eye_null")):
                return 1.0
            return 0.5 if nm.startswith("c_neck") else 0.0  # neck half: soft seam

        self.head_w = np.clip(weight(is_head), 0, 1)[None, :, None].astype(np.float32)
        self.hand_v, self.arm_faces = {}, {}
        for s in "lr":
            hand = weight(lambda nm, s=s: 1.0 if nm.startswith(tuple([f"{s}_wrist"] + [f"{s}_{f}" for f in FINGERS])) else 0.0)
            arm = weight(lambda nm, s=s: 1.0 if nm.startswith(tuple([f"{s}_lowarm", f"{s}_wrist"] + [f"{s}_{f}" for f in FINGERS])) else 0.0)
            self.hand_v[s] = hand > 0.5
            self.arm_faces[s] = (arm > 0.5)[self.faces].all(1)

    def _mhr(self, params, shape="pred", scale="pred", keypoints=False):
        """Vertices for stacked per-frame params. shape/scale: pred | median | zero. Expression is always zero.
        keypoints: also return the 70 MHR keypoints (same space as the vertices)."""
        torch = self.torch
        t = {k: torch.tensor(params[k], device=DEV) for k in PARAM_KEYS}

        def pick(v, how):
            if how == "zero":
                return torch.zeros_like(v)
            if how == "median":
                return torch.median(v, dim=0).values.expand_as(v).contiguous()
            return v

        out, kps = [], []
        with torch.no_grad():
            for a in range(0, len(t["global_rot"]), 64):
                sl = slice(a, a + 64)
                g = t["global_rot"][sl]
                r = self.head.mhr_forward(torch.zeros_like(g), g, t["body_pose_params"][sl], t["hand_pose_params"][sl],
                                          pick(t["scale_params"], scale)[sl], pick(t["shape_params"], shape)[sl],
                                          torch.zeros_like(t["expr_params"][sl]), return_keypoints=keypoints)
                if keypoints:
                    r, kp = r
                    kps.append(kp[:, :70].cpu().numpy())
                out.append(r.cpu().numpy())
        v = np.concatenate(out).astype(np.float32)
        return (v, np.concatenate(kps).astype(np.float32)) if keypoints else v

    # ---- job ----
    def run(self, req, emit):
        est = self.estimate(req, emit)
        emit(0.76, "클레이 메시 생성")
        per_frame, focal = self.build(est, req)
        self._render(req, per_frame, focal, est["masks"], est["W"], est["H"], est["n"], emit)
        return {"ok": True, "people": len(est["tracks"]), "frames": est["n"]}

    def estimate(self, req, emit):
        """Pass 1: track people and run SAM 3D Body + ViTPose + person mask on every frame.

        req options: fov_samples (MoGe-2 field of view from this many frames per shot, their median used for the
        whole shot; 0 = every frame, as SAM 3D Body does by itself), sam_hands (False: SAM 3D Body's body decoder
        only, without its two extra passes over hand crops; about a third of the time, coarser fingers),
        sam_bgr (True: hand SAM 3D Body the BGR frame as the studio did before; it expects RGB).
        """
        torch = self.torch
        from PIL import Image

        cap = cv2.VideoCapture(req["video"])
        if not cap.isOpened():
            raise RuntimeError("원본 영상을 열지 못했습니다.")
        n = int(req.get("frames") or cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cuts = sorted(int(c) for c in req.get("cuts", []) if 0 < int(c) < n)
        shot_of = np.searchsorted(np.array(cuts, int), np.arange(max(n, 1)), side="right")
        max_people, min_area = int(req.get("max_people", 2)), float(req.get("min_area", 0.03))
        fov_samples, bgr = int(req.get("fov_samples", 0)), bool(req.get("sam_bgr", False))
        sam_kw = {} if req.get("sam_hands", True) else {"inference_type": "body"}
        shot_K = self._shot_intrinsics(req["video"], shot_of, fov_samples, bgr, emit) if fov_samples > 0 else None

        # ---- pass 1: track + estimate ----
        tracks, active, masks = [], [], []
        W = H = 0
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i >= len(shot_of):
                shot_of = np.append(shot_of, shot_of[-1])
            H, W = fr.shape[:2]
            if i == 0 or shot_of[i] != shot_of[i - 1]:
                active = []
            img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                inp = self.det_proc(images=img, return_tensors="pt").to(DEV)
                r = self.det_proc.post_process_object_detection(self.det(**inp), target_sizes=torch.tensor([(H, W)]),
                                                                threshold=0.5)[0]
                lg = self.seg(**self.seg_proc(images=img, return_tensors="pt").to(DEV)).logits
                lg = torch.nn.functional.interpolate(lg, size=(H // MASK_DOWN, W // MASK_DOWN), mode="bilinear")
                masks.append((lg.argmax(1)[0] != 0).cpu().numpy())
            boxes = sorted(r["boxes"][r["labels"] == 0].cpu().numpy(), key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
            used, assign = set(), []
            for ti, last in active:
                cand = [(iou(last, b), j) for j, b in enumerate(boxes) if j not in used]
                best = max(cand, default=(0, -1))
                if best[0] > 0.2:
                    used.add(best[1])
                    assign.append((ti, boxes[best[1]]))
            for j, b in enumerate(boxes):
                if j in used or len(assign) >= max_people:
                    continue
                if (b[2] - b[0]) * (b[3] - b[1]) / (W * H) < min_area or any(iou(b, a[1]) > 0.5 for a in assign):
                    continue
                tracks.append({"shot": int(shot_of[i]), "frames": [], "box": [], "M": [], "cam_t": [], "focal": [],
                               "wrist": [], "kp": [], "kp_score": [], **{k: [] for k in PARAM_KEYS}})
                assign.append((len(tracks) - 1, b))
                used.add(j)
            active = assign
            if assign:
                bx = np.stack([b for _, b in assign])
                if shot_K is not None:
                    sam_kw["cam_int"] = shot_K[int(shot_of[i])]
                outs = self.est.process_one_image(fr if bgr else cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), bboxes=bx,
                                                  **sam_kw)
                coco = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in bx]
                with torch.no_grad():
                    pin = self.vp_proc(img, boxes=[coco], return_tensors="pt").to(DEV)
                    poses = self.vp_proc.post_process_pose_estimation(self.vp(**pin), boxes=[coco])[0]
                for (ti, b), o, pose in zip(assign, outs, poses):
                    tr = tracks[ti]
                    params = {k: np.asarray(o[k], np.float32)[None] for k in PARAM_KEYS}
                    v_fit = self._mhr(params)[0]
                    # pred_vertices = affine(v_fit) (axis flip + offset); keep the map, not the mesh
                    M, *_ = np.linalg.lstsq(np.c_[v_fit, np.ones(len(v_fit))], np.asarray(o["pred_vertices"]), rcond=None)
                    tr["frames"].append(i)
                    tr["box"].append(b)
                    tr["M"].append(M.astype(np.float32))
                    tr["cam_t"].append(np.asarray(o["pred_cam_t"], np.float32))
                    tr["focal"].append(float(o["focal_length"]))
                    sc = pose["scores"].cpu().numpy()
                    tr["wrist"].append({s: float(sc[WRIST[s]]) for s in "lr"})
                    tr["kp"].append(pose["keypoints"].cpu().numpy().astype(np.float32))
                    tr["kp_score"].append(sc.astype(np.float32))
                    for k in PARAM_KEYS:
                        tr[k].append(params[k][0])
            i += 1
            if i % 4 == 0:
                emit(0.75 * min(i / max(n, 1), 1.0), f"인물 추정 {i}/{n}")
        cap.release()
        n = i
        if n == 0:
            raise RuntimeError("원본 영상에서 프레임을 읽지 못했습니다.")
        shot_of = shot_of[:n]
        tracks = [t for t in tracks if len(t["frames"]) >= 3]
        for t in tracks:
            for k in PARAM_KEYS + ["M", "cam_t", "focal", "box", "kp", "kp_score"]:
                t[k] = np.stack(t[k]) if k != "focal" else np.array(t[k], np.float32)
            t["frames"] = np.array(t["frames"])
        return {"tracks": tracks, "masks": masks, "shot_of": shot_of, "W": W, "H": H, "n": n}

    def _shot_intrinsics(self, video, shot_of, k, bgr, emit):
        """One camera matrix per shot: MoGe-2 on k frames spread over the shot, element-wise median."""
        torch = self.torch
        n = len(shot_of)
        want = {}
        for s in np.unique(shot_of):
            idx = np.where(shot_of == s)[0]
            for f in np.unique(np.linspace(idx[0], idx[-1], min(k, len(idx))).round().astype(int)):
                want[int(f)] = int(s)
        cap = cv2.VideoCapture(video)
        Ks = {}
        for f in range(n):
            ok, fr = cap.read()
            if not ok:
                break
            if f in want:
                img = fr if bgr else cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    K = self.est.fov_estimator.get_cam_intrinsics(img)
                Ks.setdefault(want[f], []).append(np.asarray(torch.as_tensor(K).cpu(), np.float32).reshape(3, 3))
        cap.release()
        emit(0.02, "화각 추정")
        return {s: torch.as_tensor(np.median(np.stack(v), 0)[None]) for s, v in Ks.items()}

    def build(self, est, req, keypoints=False):
        """Pass 2: clay meshes in camera space, per frame [(vertices, face mask)], and one focal per frame.

        req options: mode (head | neutral), smooth (Gaussian sigma in frames over the vertices),
        keep_size (place the median-shape body along its ray so its on-screen size matches SAM's per-frame
        body: swapping in the median body at SAM's per-frame depth makes the figure grow and shrink),
        z_smooth (extra Gaussian sigma on the body's depth only, moved along its ray),
        depth_guide (Gaussian sigma; 0 = off; needs req["bg"], a depth video of the clip, bigger = nearer
        unless bg_invert):
        per track, the person's median depth value is fitted to SAM's 1/z (1/z = a*d + b) and the body is
        moved along its ray to the fitted depth. Video depth is steady over time where SAM's depth wobbles
        with the pose (an arm reaching out pushes the body back and forth).
        keypoints=True adds the 70 MHR keypoints after the same moves: [(vertices, face mask, keypoints)].
        """
        tracks, masks, shot_of = est["tracks"], est["masks"], est["shot_of"]
        W, H, n = est["W"], est["H"], est["n"]
        focal = np.zeros(n, np.float32)
        acc = [[] for _ in range(n)]
        for t in tracks:
            for k, f in enumerate(t["frames"]):
                acc[f].append(t["focal"][k])
        for s in np.unique(shot_of):  # one focal length per shot: MoGe's per-frame FOV jitters 1-3 %
            vals = [v for f in np.where(shot_of == s)[0] for v in acc[f]]
            focal[shot_of == s] = np.median(vals) if vals else 1.2 * W
        mode, sigma = req.get("mode", "head"), float(req.get("smooth", 2.0))
        keep_size, z_sigma = bool(req.get("keep_size", False)), float(req.get("z_smooth", 0.0))
        guide_sigma = float(req.get("depth_guide", 0.0))
        depth = self._depth_small(req["bg"], masks) if guide_sigma > 0 and req.get("bg") else None
        nv = self.head_w.shape[1]
        per_frame = [[] for _ in range(n)]
        for t in tracks:
            def place(v, t=t):  # model space -> camera space: SAM's per-frame affine map, then its translation
                h = np.concatenate([v, np.ones(v.shape[:2] + (1,), np.float32)], 2)
                return np.einsum("fvk,fkj->fvj", h, t["M"]) + t["cam_t"][:, None, :]

            v_neu = self._mhr(t, shape="zero", scale="median")
            need_kp = keypoints or depth is not None
            if mode == "head" or need_kp:
                v_body, kp = self._mhr(t, shape="median", scale="median", keypoints=True)
                v_out = v_body * (1 - self.head_w) + v_neu * self.head_w if mode == "head" else v_neu
                del v_body
            else:
                v_out = v_neu
            if need_kp:
                v_out = np.concatenate([v_out, kp], 1)
            P = place(v_out)
            del v_out, v_neu
            if keep_size:
                Pp = place(self._mhr(t))
                c_out, c_pred = P[:, :nv].mean(1), Pp.mean(1)
                l_out = np.sqrt(((P[:, :nv] - c_out[:, None]) ** 2).sum(-1).mean(1))
                l_pred = np.sqrt(((Pp - c_pred[:, None]) ** 2).sum(-1).mean(1))
                del Pp
                P += (c_out * (l_out / l_pred - 1)[:, None])[:, None, :]
            for k, f in enumerate(t["frames"]):  # re-seat along z so on-screen size is kept under the shot focal
                cz = P[k, :nv, 2].mean()
                P[k, :, 2] += cz * (focal[f] / t["focal"][k] - 1)
            if z_sigma > 0:
                c = P[:, :nv].mean(1)
                cz = gaussian_smooth(c[:, 2], t["frames"], z_sigma)
                P += (c * (cz / c[:, 2] - 1)[:, None])[:, None, :]
            if depth is not None:
                self._guide_depth(t, P, nv, depth, masks, guide_sigma, -1.0 if req.get("bg_invert") else 1.0)
            P = gaussian_smooth(P, t["frames"], sigma)
            fmask = self._hidden_hands(t, P[:, :nv], masks, focal, W, H, shot_of)
            for k, f in enumerate(t["frames"]):
                per_frame[f].append((P[k, :nv], fmask[k]) + ((P[k, nv:],) if keypoints else ()))
        return per_frame, focal

    @staticmethod
    def _depth_small(path, masks):
        """Depth video frames as grey arrays at the person-mask size (1/MASK_DOWN)."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("depth 영상을 열지 못했습니다.")
        out = []
        for m in masks:
            ok, d = cap.read()
            if not ok:
                break
            out.append(cv2.resize(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), (m.shape[1], m.shape[0]),
                                  interpolation=cv2.INTER_AREA).astype(np.float32))
        cap.release()
        return out

    @staticmethod
    def _guide_depth(t, P, nv, depth, masks, sigma, sign):
        """In place: move each frame's body along its ray to the depth the video depth implies."""
        kernel = np.ones((5, 5), np.uint8)
        g = np.full(len(P), np.nan)
        for k, f in enumerate(t["frames"]):
            if f >= len(depth):
                continue
            m = masks[f]
            x0, y0, x1, y1 = np.clip((t["box"][k] / MASK_DOWN).astype(int), 0, None)
            inside = np.zeros_like(m, np.uint8)
            inside[y0:y1, x0:x1] = m[y0:y1, x0:x1]
            inside = cv2.erode(inside, kernel) > 0
            vals = depth[f][inside]
            # skip frames where the person sits at the ends of the grey range: no depth change shows there
            if len(vals) >= 20 and ((vals >= 250) | (vals <= 5)).mean() < 0.05:
                g[k] = np.median(vals)
        ok = np.isfinite(g)
        if ok.sum() < 8:
            return
        g = gaussian_smooth(np.interp(np.arange(len(P)), np.where(ok)[0], g[ok]), t["frames"], sigma)
        # anchor: the joints ViTPose sees. Moving along the ray through them keeps the visible part in place;
        # through the body centre (off screen in a close-up) the visible head would slide up and down
        kp = P[:, nv:][:, MHR2COCO]
        vis = t["kp_score"] > 0.5
        c = np.stack([kp[k, vis[k]].mean(0) if vis[k].any() else P[k, :nv].mean(0) for k in range(len(P))])
        iz = 1 / c[:, 2]
        if np.ptp(g[ok]) < 2:  # the video sees no depth change
            lz_v = np.zeros(len(P))
        else:
            a, b = np.polyfit(g[ok], iz[ok], 1)
            if sign * a <= 0 or sign * np.corrcoef(g[ok], iz[ok])[0, 1] < 0.5:
                return  # video depth and SAM disagree on the direction of motion: leave SAM's depth
            lz_v = -np.log(np.maximum(a * g + b, 1e-3))
        # the video depth replaces only SAM's faster depth changes; slower trends stay SAM's, since the fit is
        # a straight line and the true relation bends over a long push-in
        lz_s = np.log(c[:, 2])
        trend = lambda x: gaussian_smooth(x, t["frames"], GUIDE_TREND)  # noqa: E731
        w = gaussian_smooth(ok.astype(np.float64), t["frames"], 3)  # fade back to SAM where the video can't see
        w = np.where(ok, w, 0.0)
        r = np.exp(w * ((lz_v - trend(lz_v)) - (lz_s - trend(lz_s))))
        P += (c * (r - 1)[:, None])[:, None, :]

    def _hidden_hands(self, t, P, masks, focal, W, H, shot_of):
        """Per frame face mask: drop a forearm + hand the source doesn't show (majority over +-3 frames)."""
        dil = max(3, int(0.02 * W / MASK_DOWN)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))
        raw = {}
        for k, f in enumerate(t["frames"]):
            p = P[k]
            m = cv2.dilate(masks[f].astype(np.uint8), kernel) > 0
            mh, mw = m.shape
            x = (p[:, 0] * focal[f] / p[:, 2] + W / 2) / MASK_DOWN
            y = (p[:, 1] * focal[f] / p[:, 2] + H / 2) / MASK_DOWN
            for s in "lr":
                sc = t["wrist"][k][s]
                hv = self.hand_v[s]
                behind = p[hv, 2].mean() > p[:, 2].mean() + 0.05  # far-side hand, behind the torso
                if sc < 0.3 or (behind and sc < 0.6):
                    raw[(k, s)] = True
                    continue
                xi, yi = x[hv].astype(int), y[hv].astype(int)
                on = (xi >= 0) & (xi < mw) & (yi >= 0) & (yi < mh)
                raw[(k, s)] = bool(on.mean() >= 0.5 and m[yi[on], xi[on]].mean() < 0.3)
        out = []
        for k in range(len(t["frames"])):
            fm = np.ones(len(self.faces), bool)
            for s in "lr":
                votes = [raw[(j, s)] for j in range(max(0, k - 3), min(len(t["frames"]), k + 4))]
                if np.mean(votes) > 0.5:
                    fm &= ~self.arm_faces[s]
            out.append(fm)
        return out

    def clay_frames(self, per_frame, focal, W, H, clear=(0.0, 0.0, 0.0, 0.0)):
        """Yields one (H, W, 4) uint8 clay render per frame. With the default clear the colour is
        premultiplied by alpha, ready to lay over a background."""
        import moderngl

        if self.ctx is None:
            self.ctx = moderngl.create_standalone_context()
            self.prog = self.ctx.program(
                vertex_shader="""
                    #version 330
                    uniform vec2 size; uniform float focal;
                    in vec3 pos; in vec3 nrm; out vec3 v_n;
                    void main() {
                        float x = pos.x * focal / pos.z + size.x * 0.5;
                        float y = pos.y * focal / pos.z + size.y * 0.5;
                        float zn = (pos.z - 0.05) / (100.0 - 0.05) * 2.0 - 1.0;
                        gl_Position = vec4(x / size.x * 2.0 - 1.0, 1.0 - y / size.y * 2.0, zn, 1.0);
                        v_n = nrm;
                    }""",
                fragment_shader="""
                    #version 330
                    uniform vec3 light; in vec3 v_n; out vec4 color;
                    void main() {
                        float c = (70.0 + 170.0 * abs(dot(normalize(v_n), light))) / 255.0;
                        color = vec4(c, c, c, 1.0);
                    }""")
            light = np.array([0.3, -0.4, -1.0])
            self.prog["light"].value = tuple(-light / np.linalg.norm(light))
        ctx, prog, faces = self.ctx, self.prog, self.faces
        prog["size"].value = (W, H)
        fbo_ms = ctx.framebuffer(ctx.renderbuffer((W, H), 4, samples=4), ctx.depth_renderbuffer((W, H), samples=4))
        fbo = ctx.framebuffer(ctx.renderbuffer((W, H), 4))
        ctx.enable(moderngl.DEPTH_TEST)
        try:
            for i in range(len(per_frame)):
                fbo_ms.use()
                ctx.clear(*clear, depth=1.0)
                prog["focal"].value = float(focal[i])
                for P, fmask, *_ in per_frame[i]:
                    if not np.isfinite(P).all() or (P[:, 2] < 0.05).any():
                        continue
                    tri = P[faces]
                    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                    vn = np.zeros_like(P)
                    for c in range(3):
                        np.add.at(vn, faces[:, c], fn)
                    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9
                    vbo = ctx.buffer(np.hstack([P, vn]).astype("f4").tobytes())
                    ibo = ctx.buffer(faces[fmask].astype("i4").tobytes())
                    vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "pos", "nrm")], ibo)
                    vao.render(moderngl.TRIANGLES)
                    vao.release(); vbo.release(); ibo.release()
                ctx.copy_framebuffer(fbo, fbo_ms)
                yield np.frombuffer(fbo.read(components=4), np.uint8).reshape(H, W, 4)[::-1]
        finally:
            fbo_ms.release()
            fbo.release()

    def _render(self, req, per_frame, focal, masks, W, H, n, emit):
        bg_path, bg_mode = req.get("bg"), req.get("bg_mode", "overlay")
        dcap = cv2.VideoCapture(bg_path) if bg_path else None
        if dcap is not None and not dcap.isOpened():
            raise RuntimeError("depth 영상을 열지 못했습니다.")
        k_rm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.03 * W) | 1,) * 2)
        writer = subprocess.Popen(
            [req.get("ffmpeg", "ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
             "-r", str(req.get("fps", "24")), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", req["out"]],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        last_d = np.full((H, W), 40, np.uint8)
        clear = (40 / 255, 40 / 255, 40 / 255, 1.0) if dcap is None else (0.0, 0.0, 0.0, 0.0)
        try:
            for i, rgba in enumerate(self.clay_frames(per_frame[:n], focal, W, H, clear)):
                clay = rgba[..., :3].astype(np.float32)
                if dcap is None:
                    out = clay
                else:
                    ok, d = dcap.read()
                    if ok:
                        last_d = cv2.resize(cv2.cvtColor(d, cv2.COLOR_BGR2GRAY), (W, H))
                    D = last_d
                    a = rgba[..., 3:4].astype(np.float32) / 255
                    if bg_mode == "remove":
                        back, props = self._remove_people(D, masks[i], k_rm, W, H)
                        out = np.repeat(back[..., None], 3, 2).astype(np.float32) * (1 - a) + clay
                        out[props] = D[props][:, None]
                    else:
                        out = np.repeat(D[..., None], 3, 2).astype(np.float32) * (1 - a) + clay
                writer.stdin.write(np.ascontiguousarray(np.clip(out, 0, 255)[..., ::-1]).astype(np.uint8).tobytes())
                if i % 8 == 0:
                    emit(0.8 + 0.2 * i / n, f"클레이 렌더 {i}/{n}")
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError("ffmpeg 인코딩 실패: " + writer.stderr.read().decode(errors="replace")[-400:])
        except BaseException:
            writer.kill()
            raise
        finally:
            if dcap is not None:
                dcap.release()

    @staticmethod
    def _remove_people(D, mask_small, k_rm, W, H):
        """Depth with the people filled in from their surroundings; nearer non-person blobs (props) kept."""
        pm = cv2.resize(mask_small.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        rm = cv2.dilate(pm.astype(np.uint8), k_rm) > 0
        q = 4
        small = cv2.resize(D, (W // q, H // q), interpolation=cv2.INTER_AREA)
        msmall = cv2.resize(rm.astype(np.uint8), (W // q, H // q), interpolation=cv2.INTER_NEAREST)
        fill = cv2.resize(cv2.inpaint(small, msmall, 7, cv2.INPAINT_TELEA), (W, H), interpolation=cv2.INTER_LINEAR)
        fill = cv2.GaussianBlur(fill, (0, 0), 3)
        back = np.where(rm, fill, D)
        dist = cv2.distanceTransform((~pm).astype(np.uint8), cv2.DIST_L2, 3)
        props = rm & ~pm & (dist > 0.006 * W) & (D.astype(np.int16) > fill.astype(np.int16) + 14)
        props = cv2.morphologyEx(props.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        return back, props


def serve():
    replies, sys.stdout = sys.stdout, sys.stderr  # libraries print; keep stdout for replies only

    def send(obj):
        replies.write(json.dumps(obj, ensure_ascii=False) + "\n")
        replies.flush()

    engine = None
    for line in sys.stdin:
        try:
            req = json.loads(line)
            if engine is None:
                send({"progress": 0.0, "stage": "모델 불러오는 중 (처음 한 번)"})
                engine = Engine()
            reply = engine.run(req, lambda p, s: send({"progress": round(float(p), 4), "stage": s}))
        except Exception as exc:
            traceback.print_exc()
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
        finally:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        send(reply)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    serve()
