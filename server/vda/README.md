# Video Depth Anything (vendored)

`video_depth_anything/` is copied from https://github.com/DepthAnything/Video-Depth-Anything
(commit 4f5ae23, Apache-2.0, see LICENSE). Changes:

- `utils/util.py` moved to `video_depth_anything/util/align.py` and imported relatively.
- `EasyDict` in `dpt_temporal.py` replaced with a plain `dict` (only used for `**kwargs`).
- `torchvision.transforms.Compose` replaced with a small `Compose` in `util/transform.py`.

Model weights are not stored here; `server/video_depth.py` downloads them from Hugging Face.
Video-Depth-Anything-Small weights are Apache-2.0, Base/Large are CC-BY-NC-4.0.
