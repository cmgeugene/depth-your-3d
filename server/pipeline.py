"""Render a job: stepped (on 2s/3s...) source video, depth mp4, 16-bit depth PNG sequence.

A timing map lists, for every output frame, which source frame to show. Depth is only
inferred once per unique source frame, so stepping on 2s halves the GPU work.
"""
import io
import json
import math
import zipfile

import numpy as np
from PIL import Image

from . import config, depth, video


class Cancelled(Exception):
    pass


def shot_ranges(n: int, cuts: list[int]) -> list[tuple[int, int]]:
    bounds = [0] + sorted(c for c in set(cuts) if 0 < c < n) + [n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(values) < 2:
        return values
    radius = int(math.ceil(sigma * 3))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel / kernel.sum(), mode="valid")


def run(job, media, progress):
    p = job.params
    n = media.frames
    fmap = p.get("map") or list(range(n))
    cuts = p["cuts"] if p.get("cuts") is not None else media.cuts
    outs = p["outputs"]
    want_depth = outs.get("depth_mp4") or outs.get("depth_png")
    audio_src = media.src if media.has_audio and outs.get("audio", True) else None

    # Progress budget per stage; depth inference dominates.
    stages = []
    if outs.get("stepped_mp4"):
        stages.append(("stepped", 1.0))
    if want_depth:
        stages += [("infer", 3.0), ("encode", 1.0)]
    total = sum(w for _, w in stages)
    spans, acc = {}, 0.0
    for name, w in stages:
        spans[name] = (acc / total, (acc + w) / total)
        acc += w

    def report(stage: str, frac: float, text: str):
        if job.cancel.is_set():
            raise Cancelled()
        a, b = spans[stage]
        progress(a + (b - a) * min(max(frac, 0.0), 1.0), text)

    results = []
    if outs.get("stepped_mp4"):
        name = f"{media.stem}_stepped.mp4"
        _render_stepped(media, fmap, job.dir / name, job.dir / "ffmpeg.log", audio_src, report)
        results.append({"name": name, "label": "스텝 적용 영상 (mp4)"})
    if want_depth:
        results += _render_depth(media, fmap, cuts, p["depth"], outs, job.dir, audio_src, report)
    if p.get("timing"):
        name = f"{media.stem}_timing.json"
        (job.dir / name).write_text(
            json.dumps({"source": media.name, "frames": n, "fps": media.fps_str, "spec": p["timing"],
                        "cuts": cuts, "map": fmap}, ensure_ascii=False),
            encoding="utf-8",
        )
        results.append({"name": name, "label": "타이밍 데이터 (json)"})
    for r in results:
        r["size"] = (job.dir / r["name"]).stat().st_size
    return results


def _render_stepped(media, fmap, path, log_path, audio_src, report):
    writer = video.FrameWriter(
        path, media.width, media.height, media.fps_str, "rgb24", crf=16, log_path=log_path,
        audio_src=audio_src, duration=len(fmap) / media.fps,
    )
    try:
        with video.FrameReader(media.src, media.width, media.height) as reader:
            cur_idx, cur = -1, None
            for i, target in enumerate(fmap):
                while cur_idx < target:
                    frame = reader.read()
                    if frame is None:
                        raise RuntimeError(f"원본 프레임이 부족합니다 ({cur_idx + 1}/{target + 1}).")
                    cur, cur_idx = frame, cur_idx + 1
                writer.write(cur)
                if i % 12 == 0:
                    report("stepped", i / len(fmap), f"스텝 영상 인코딩 {i}/{len(fmap)}")
    except BaseException:
        writer.abort()
        raise
    writer.close()


def _render_depth(media, fmap, cuts, dp, outs, job_dir, audio_src, report):
    n = len(fmap)
    h, w = media.height, media.width
    report("infer", 0, f"모델 로딩 ({config.MODELS[dp['model']]['label']})")
    model = depth.get_model(dp["model"])
    size = depth.infer_size(h, w, dp["input_size"])

    uniq = sorted(set(fmap))
    rank = {src: r for r, src in enumerate(uniq)}
    raw_path = job_dir / "raw.npy"
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float16, shape=(len(uniq), *size))

    # Pass 1: inference on each unique source frame, raw disparity cached at model resolution.
    batch, first = [], 0
    done = 0
    with video.FrameReader(media.src, w, h) as reader:
        idx = 0
        while done < len(uniq):
            frame = reader.read()
            if frame is None:
                break
            if idx == uniq[done]:
                if not batch:
                    first = done
                batch.append(frame)
                done += 1
                if len(batch) == config.BATCH_SIZE or done == len(uniq):
                    raw[first:done] = depth.predict(model, batch, size)
                    batch = []
                    report("infer", done / len(uniq), f"Depth 추론 {done}/{len(uniq)} (그림 수 기준)")
            idx += 1
    if done < len(uniq):
        raise RuntimeError(f"원본 프레임이 부족합니다 ({done}/{len(uniq)}).")
    raw.flush()

    # Pass 2: normalize per shot so brightness stays stable inside a shot, then encode.
    lo_pct, hi_pct = float(dp["clip_lo"]), 100.0 - float(dp["clip_hi"])

    def percentiles(ranks: list[int], max_samples: int = 48) -> tuple[float, float]:
        if len(ranks) > max_samples:
            ranks = [ranks[int(i)] for i in np.linspace(0, len(ranks) - 1, max_samples)]
        vals = np.concatenate([np.asarray(raw[r, ::4, ::4], dtype=np.float32).ravel() for r in ranks])
        lo, hi = np.percentile(vals, [lo_pct, hi_pct])
        return float(lo), float(hi)

    mode = dp["norm"]
    global_lohi = percentiles(list(range(len(uniq)))) if mode == "clip" else None
    sigma_t = float(dp["temporal"])
    stem = media.stem

    writer = None
    if outs.get("depth_mp4"):
        writer = video.FrameWriter(job_dir / f"{stem}_depth.mp4", w, h, media.fps_str, "gray", crf=12,
                                   log_path=job_dir / "ffmpeg.log", audio_src=audio_src, duration=n / media.fps)
    zf = zipfile.ZipFile(job_dir / f"{stem}_depth_png16.zip", "w", zipfile.ZIP_STORED) if outs.get("depth_png") else None
    try:
        written = 0
        for s, e in shot_ranges(n, cuts):
            shot_ranks = sorted({rank[fmap[i]] for i in range(s, e)})
            pos = {r: k for k, r in enumerate(shot_ranks)}
            if mode == "clip":
                los = np.full(len(shot_ranks), global_lohi[0])
                his = np.full(len(shot_ranks), global_lohi[1])
            elif mode == "shot":
                lo, hi = percentiles(shot_ranks)
                los, his = np.full(len(shot_ranks), lo), np.full(len(shot_ranks), hi)
            else:
                stats = np.array([percentiles([r]) for r in shot_ranks])
                los, his = stats[:, 0], stats[:, 1]
                if mode == "adaptive":
                    frames_per_drawing = (e - s) / len(shot_ranks)
                    sigma = float(dp["adaptive_window"]) * media.fps / frames_per_drawing
                    los, his = gaussian_smooth(los, sigma), gaussian_smooth(his, sigma)

            radius = int(math.ceil(sigma_t * 2)) if sigma_t > 0 else 0
            last_k, u8, png = None, None, None
            for i in range(s, e):
                k = pos[rank[fmap[i]]]
                if k != last_k:
                    ks = range(max(0, k - radius), min(len(shot_ranks), k + radius + 1))
                    weights = [math.exp(-0.5 * ((j - k) / sigma_t) ** 2) if sigma_t > 0 else 1.0 for j in ks]
                    v = depth.finish([raw[shot_ranks[j]] for j in ks], weights,
                                     [los[j] for j in ks], [his[j] for j in ks],
                                     (h, w), bool(dp["invert"]), float(dp["gamma"]))
                    u8 = (v * 255 + 0.5).astype(np.uint8)
                    if zf is not None:
                        buf = io.BytesIO()
                        Image.fromarray((v * 65535 + 0.5).astype(np.uint16)).save(buf, format="PNG", compress_level=3)
                        png = buf.getvalue()
                    last_k = k
                if writer is not None:
                    writer.write(u8)
                if zf is not None:
                    zf.writestr(f"{stem}_depth_{i:06d}.png", png)
                written += 1
                if written % 12 == 0:
                    report("encode", written / n, f"Depth 인코딩 {written}/{n}")
    except BaseException:
        if writer is not None:
            writer.abort()
        if zf is not None:
            zf.close()
        raise
    finally:
        del raw
        try:
            raw_path.unlink(missing_ok=True)
        except OSError:
            pass  # still mapped on Windows; the job directory cleanup removes it later
    if writer is not None:
        writer.close()
    if zf is not None:
        zf.close()

    results = []
    if outs.get("depth_mp4"):
        results.append({"name": f"{stem}_depth.mp4", "label": "Depth 영상 (mp4)"})
    if outs.get("depth_png"):
        results.append({"name": f"{stem}_depth_png16.zip", "label": "Depth 16bit PNG 시퀀스 (zip)"})
    return results
