"""SAM 3.1 timing test for character-only depth.

Runs SAM 3.1 video segmentation with a text prompt ("person") over a whole video and reports how long
each step takes and how much VRAM it needs. Run it on the GPU server, in the Python environment where
SAM 3.1 is installed (https://github.com/facebookresearch/sam3), not in Depth Studio's .venv:

    python tools/bench_sam3.py "K-pop Concert.mp4"
    python tools/bench_sam3.py "K-pop Concert.mp4" --frames 48 --save-mask mask.mp4

Needs ffmpeg/ffprobe on PATH. --save-mask writes the union of all masks (white = person) so the
result can be checked by eye.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den), int(s["nb_read_packets"])


class VramWatch:
    """Polls device-wide used memory, so it also sees memory held by other processes (e.g. a running SAM service)."""

    def __init__(self, torch):
        self.torch, self.peak, self._stop = torch, 0, threading.Event()
        free, total = torch.cuda.mem_get_info()
        self.total, self.base = total, total - free
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop.wait(0.2):
            free, total = self.torch.cuda.mem_get_info()
            self.peak = max(self.peak, total - free)

    def stop(self):
        self._stop.set()
        self._t.join()


def to_numpy(x):
    import numpy as np
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def objects_of(outputs):
    """(object ids, masks N×H×W or None). The output schema isn't documented, so try the known shapes."""
    if isinstance(outputs, dict):
        if "out_obj_ids" in outputs:
            masks = outputs.get("out_binary_masks", outputs.get("out_masks"))
            return [int(i) for i in to_numpy(outputs["out_obj_ids"]).reshape(-1)], masks
        if outputs and all(isinstance(k, int) for k in outputs):  # {obj_id: mask}
            return list(outputs), list(outputs.values())
    return None, None


def union_mask(masks):
    """All objects merged into one H×W bool mask, or None when there is no object."""
    import numpy as np
    if masks is None or len(masks) == 0:
        return None
    m = [to_numpy(x) for x in masks] if isinstance(masks, list) else [to_numpy(masks)]
    m = np.concatenate([x.reshape(-1, *x.shape[-2:]) for x in m])
    return (m > 0.5 if m.dtype != bool else m).any(0)


def write_mask_video(frames, count, fps, path):
    import numpy as np
    h, w = next(m.shape for m in frames.values() if m is not None)
    proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
                             "-r", f"{fps}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", "-crf", "18", path], stdin=subprocess.PIPE)
    blank = np.zeros((h, w), np.uint8)
    for i in range(count):
        m = frames.get(i)
        img = blank if m is None or m.shape != (h, w) else m.astype(np.uint8) * np.uint8(255)
        proc.stdin.write(np.ascontiguousarray(img).tobytes())
    proc.stdin.close()
    proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--prompt", default="person", help='text prompt (default "person")')
    ap.add_argument("--frames", type=int, default=0, help="test only the first N frames (default: whole video)")
    ap.add_argument("--save-mask", metavar="MP4", help="write the union mask video here")
    ap.add_argument("--sam3", action="store_true", help="use the SAM 3 builder instead of SAM 3.1 (multiplex)")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    w, h, fps, total_frames = probe(args.video)
    tmp = None
    video = os.path.abspath(args.video)
    if args.frames and args.frames < total_frames:
        tmp = tempfile.mkdtemp(prefix="bench-sam3-")
        video = os.path.join(tmp, "clip.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", args.video, "-frames:v", str(args.frames), "-an",
                        "-c:v", "libx264", "-crf", "12", video], check=True)
        total_frames = args.frames
    print(f"영상: {w}x{h}, {fps:g}fps, {total_frames}프레임 ({total_frames / fps:.1f}초), 프롬프트 \"{args.prompt}\"")

    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = VramWatch(torch)
    print(f"시작 전 GPU 사용 중: {vram.base / 2**30:.1f} / {vram.total / 2**30:.1f} GB (다른 프로그램 포함)")

    times = {}
    t = time.perf_counter()
    from sam3 import model_builder
    predictor = (model_builder.build_sam3_video_predictor() if args.sam3
                 else model_builder.build_sam3_multiplex_video_predictor())
    times["모델 불러오기"] = time.perf_counter() - t

    t = time.perf_counter()
    session_id = predictor.handle_request(request=dict(type="start_session", resource_path=video))["session_id"]
    times["영상 불러오기"] = time.perf_counter() - t

    t = time.perf_counter()
    out = predictor.handle_request(request=dict(type="add_prompt", session_id=session_id, frame_index=0,
                                                text=args.prompt))["outputs"]
    times["첫 프레임 찾기"] = time.perf_counter() - t
    ids, _ = objects_of(out)
    if ids is None:
        print("  (출력 형식을 알 수 없어 객체 수는 세지 않습니다. 키:", list(out)[:10] if isinstance(out, dict) else type(out), ")")

    masks, counts, all_ids, first_frame_at = {}, [], set(), None
    t = time.perf_counter()
    for resp in predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=session_id)):
        if first_frame_at is None:
            first_frame_at = time.perf_counter() - t
        ids, m = objects_of(resp["outputs"])
        if ids is not None:
            counts.append(len(ids))
            all_ids.update(ids)
        if args.save_mask and ids is not None:
            masks[resp["frame_index"]] = union_mask(m)
    times["전체 프레임 추적"] = time.perf_counter() - t
    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    vram.stop()

    run = times["영상 불러오기"] + times["첫 프레임 찾기"] + times["전체 프레임 추적"]
    print()
    for k, v in times.items():
        print(f"  {k:<10} {v:7.1f}초")
    track = times["전체 프레임 추적"] - (first_frame_at or 0)
    if total_frames > 1 and track > 0:
        print(f"  추적 속도      {(total_frames - 1) / track:5.1f} 프레임/초 ({1000 * track / (total_frames - 1):.0f}ms/프레임)")
    print(f"  모델 불러오기 뺀 처리 시간: {run:.1f}초 = 영상 1초당 {run / (total_frames / fps):.1f}초")
    if counts:
        print(f"  찾은 사람: 프레임당 평균 {sum(counts) / len(counts):.1f}명, 최대 {max(counts)}명, 서로 다른 ID {len(all_ids)}개")
    print(f"  GPU 메모리 최고: {vram.peak / 2**30:.1f} GB (이 테스트로 늘어난 양 약 {(vram.peak - vram.base) / 2**30:.1f} GB)")

    if args.save_mask:
        if any(m is not None for m in masks.values()):
            write_mask_video(masks, total_frames, fps, args.save_mask)
            print(f"  마스크 영상: {args.save_mask}")
        else:
            print("  출력 형식을 알 수 없어 마스크 영상은 만들지 않았습니다.")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
