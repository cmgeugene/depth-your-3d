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
MASK_DOWN = 4


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

    def _mhr(self, params, shape="pred", scale="pred"):
        """Vertices for stacked per-frame params. shape/scale: pred | median | zero. Expression is always zero."""
        torch = self.torch
        t = {k: torch.tensor(params[k], device=DEV) for k in PARAM_KEYS}

        def pick(v, how):
            if how == "zero":
                return torch.zeros_like(v)
            if how == "median":
                return torch.median(v, dim=0).values.expand_as(v).contiguous()
            return v

        out = []
        with torch.no_grad():
            for a in range(0, len(t["global_rot"]), 64):
                sl = slice(a, a + 64)
                g = t["global_rot"][sl]
                v = self.head.mhr_forward(torch.zeros_like(g), g, t["body_pose_params"][sl], t["hand_pose_params"][sl],
                                          pick(t["scale_params"], scale)[sl], pick(t["shape_params"], shape)[sl],
                                          torch.zeros_like(t["expr_params"][sl]))
                out.append(v.cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    # ---- job ----
    def run(self, req, emit):
        torch = self.torch
        from PIL import Image

        cap = cv2.VideoCapture(req["video"])
        if not cap.isOpened():
            raise RuntimeError("원본 영상을 열지 못했습니다.")
        n = int(req.get("frames") or cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cuts = sorted(int(c) for c in req.get("cuts", []) if 0 < int(c) < n)
        shot_of = np.searchsorted(np.array(cuts, int), np.arange(max(n, 1)), side="right")
        max_people, min_area = int(req.get("max_people", 2)), float(req.get("min_area", 0.03))

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
                               "wrist": [], **{k: [] for k in PARAM_KEYS}})
                assign.append((len(tracks) - 1, b))
                used.add(j)
            active = assign
            if assign:
                bx = np.stack([b for _, b in assign])
                outs = self.est.process_one_image(fr, bboxes=bx)
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
            for k in PARAM_KEYS + ["M", "cam_t", "focal", "box"]:
                t[k] = np.stack(t[k]) if k != "focal" else np.array(t[k], np.float32)
            t["frames"] = np.array(t["frames"])

        # ---- pass 2: meshes ----
        emit(0.76, "클레이 메시 생성")
        focal = np.zeros(n, np.float32)
        acc = [[] for _ in range(n)]
        for t in tracks:
            for k, f in enumerate(t["frames"]):
                acc[f].append(t["focal"][k])
        for s in np.unique(shot_of):  # one focal length per shot: MoGe's per-frame FOV jitters 1-3 %
            vals = [v for f in np.where(shot_of == s)[0] for v in acc[f]]
            focal[shot_of == s] = np.median(vals) if vals else 1.2 * W
        mode, sigma = req.get("mode", "head"), float(req.get("smooth", 2.0))
        per_frame = [[] for _ in range(n)]
        for t in tracks:
            v_neu = self._mhr(t, shape="zero", scale="median")
            if mode == "head":
                v_body = self._mhr(t, shape="median", scale="median")
                v_out = v_body * (1 - self.head_w) + v_neu * self.head_w
                del v_body
            else:
                v_out = v_neu
            P = np.einsum("fvk,fkj->fvj", np.concatenate([v_out, np.ones(v_out.shape[:2] + (1,), np.float32)], 2), t["M"])
            P += t["cam_t"][:, None, :]
            del v_out, v_neu
            for k, f in enumerate(t["frames"]):  # re-seat along z so on-screen size is kept under the shot focal
                cz = P[k, :, 2].mean()
                P[k, :, 2] += cz * (focal[f] / t["focal"][k] - 1)
            P = gaussian_smooth(P, t["frames"], sigma)
            fmask = self._hidden_hands(t, P, masks, focal, W, H, shot_of)
            for k, f in enumerate(t["frames"]):
                per_frame[f].append((P[k], fmask[k]))

        # ---- pass 3: render ----
        self._render(req, per_frame, focal, masks, W, H, n, emit)
        return {"ok": True, "people": len(tracks), "frames": n}

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

    def _render(self, req, per_frame, focal, masks, W, H, n, emit):
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
        try:
            for i in range(n):
                fbo_ms.use()
                if dcap is None:
                    ctx.clear(40 / 255, 40 / 255, 40 / 255, 1.0, depth=1.0)
                else:
                    ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)  # premultiplied over the depth
                prog["focal"].value = float(focal[i])
                for P, fmask in per_frame[i]:
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
                rgba = np.frombuffer(fbo.read(components=4), np.uint8).reshape(H, W, 4)[::-1]
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
            fbo_ms.release()
            fbo.release()
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
