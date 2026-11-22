# Temporally stable depth video and pose/skeleton video extraction for a single RTX 4090 (state as of 2026-11-15)

Conventions used in these notes
- All web sources were accessed on 2026-11-15. A date in brackets after a claim is the date printed on the source (release note, paper version, licence revision).
- "Not verified" means I could not open a primary source for it in this session; such items are repeated under Gaps.
- No number below was measured on an RTX 4090 by me. Almost every published VRAM/speed figure is for an A100 (40/80 GB), an RTX 3090, an RTX 5090 or an RTX A6000; the GPU is stated next to each figure.
- Licence notes are summaries of the linked licence text or README, not legal advice.

---

## 1. Which video-consistent depth models exist, and how do they compare (consistency, detail, speed, VRAM, clip length)?

### Takeaway
Video Depth Anything (VDA) is still the reference point in November 2026: it has the best published temporal-consistency score (TAE 0.570 for ViT-L, 0.703 for ViT-S versus 1.140 for per-frame Depth Anything V2-L), runs at roughly per-frame-model speed, and handles arbitrary length with a 32-frame sliding window. Diffusion-based models (DepthCrafter, GeometryCrafter, RollingDepth, ChronoDepth, Depth Any Video) are 5 to 100 times slower and mostly do not fit 24 GB at their default resolution; the only clearly new 2026 open release with video-depth claims that I could confirm is ViGeo (code Apache-2.0, weights on Hugging Face, weight licence not stated).

### Cited Findings

#### Video Depth Anything (ByteDance, CVPR 2025 Highlight)
- Release history: paper/code/models 2025-01-21; autocast (FP16) inference, grayscale/NPZ/EXR output and speed/memory improvements 2025-02-08; metric model on ViT-L 2025-04-25; experimental training-free streaming mode 2025-07-03; ViT-Base relative model and ViT-Small/Base metric models 2025-08-28; streaming mode for metric models 2025-09-12 — [VDA GitHub README](https://github.com/DepthAnything/Video-Depth-Anything)
- Model sizes: Small 28.4M params, Base 113.1M, Large 381.8M; the same three sizes exist as "Metric-Video-Depth-Anything-*" — [VDA GitHub README](https://github.com/DepthAnything/Video-Depth-Anything)
- Latency and VRAM on a single A100 at input shape 1x32x518x518: VDA-Small 9.1 ms (FP32) / 7.5 ms (FP16) per frame, 7.3 GB (FP32) / 6.8 GB (FP16); VDA-Large 67 ms (FP32) / 14 ms (FP16), 26.7 GB (FP32) / 23.6 GB (FP16). No Base row is given — [VDA GitHub README](https://github.com/DepthAnything/Video-Depth-Anything)
- Zero-shot video depth, paper Table 1 (AbsRel lower-better / δ1 higher-better; TAE = temporal alignment error on ScanNet, 170 frames, lower-better):

| Method | KITTI | ScanNet | Bonn | NYUv2 | Sintel (~50f) | TAE |
|---|---|---|---|---|---|---|
| Depth Anything V2-L (per frame) | 0.137 / 0.815 | 0.150 / 0.768 | 0.127 / 0.864 | 0.094 / 0.928 | 0.390 / 0.541 | 1.140 |
| NVDS | 0.233 / 0.614 | 0.207 / 0.628 | 0.199 / 0.674 | 0.217 / 0.598 | 0.408 / 0.464 | 2.176 |
| NVDS + DAv2-L | 0.227 / 0.617 | 0.194 / 0.658 | 0.191 / 0.700 | 0.184 / 0.679 | 0.449 / 0.503 | 2.536 |
| ChronoDepth | 0.243 / 0.576 | 0.190 / 0.682 | 0.159 / 0.793 | 0.199 / 0.665 | 0.551 / 0.478 | 1.022 |
| DepthCrafter | 0.164 / 0.753 | 0.169 / 0.730 | 0.153 / 0.803 | 0.141 / 0.822 | 0.299 / 0.695 | 0.639 |
| Depth Any Video | – | – | – | – | 0.405 / 0.659 | 0.967 |
| VDA-S | 0.086 / 0.942 | 0.110 / 0.876 | 0.083 / 0.950 | 0.077 / 0.959 | 0.339 / 0.584 | 0.703 |
| VDA-L | 0.083 / 0.944 | 0.089 / 0.926 | 0.071 / 0.959 | 0.062 / 0.971 | 0.295 / 0.644 | 0.570 |

  — [VDA paper, arXiv 2501.12375v4 (v4 dated 2026-10-14)](https://arxiv.org/html/2501.12375v4)
- TAE is described in the paper as the reprojection error of depth maps between consecutive frames — [VDA paper v4](https://arxiv.org/html/2501.12375v4)
- Per-frame latency, paper Table 3 (A100, 518x518): ChronoDepth 506 ms (FP16), DepthCrafter 910 ms (FP16), Depth Any Video 159 ms (FP16), NVDS 204 ms (FP32), DAv2-L 60 ms (FP32), VDA-L 67 ms (FP32), VDA-S 9.1 ms (FP32). VDA-L is about 10% slower than DAv2-L — [VDA paper v4](https://arxiv.org/html/2501.12375v4)
- Parameters and video training data: DepthCrafter 2156.7M params; Depth Any Video 1422.8M; VDA-L 381.8M — [VDA paper v4](https://arxiv.org/html/2501.12375v4)
- Single-image detail is essentially unchanged from DAv2-L (e.g. NYUv2 0.046/0.978 for VDA-L versus 0.045/0.979 for DAv2-L; DIODE 0.067/0.950 versus 0.066/0.952) — [VDA paper v4](https://arxiv.org/html/2501.12375v4)
- Long-video strategy: windows of 32 frames, composed of 22 new frames + 8 overlap frames from the previous window + 2 key frames (key frames sampled every 12 frames backwards); overlap frames are blended with a weight that linearly decays from 1 to 0. A 48-frame window gave no benefit over 32 (AbsRel/δ1/TAE 0.143/0.852/0.732 at 48 versus 0.144/0.851/0.718 at 32) — [VDA paper v4](https://arxiv.org/html/2501.12375v4)
- Depth Any Video could only be evaluated on Sintel and TAE because of its 192-frame limit — [VDA paper v4](https://arxiv.org/html/2501.12375v4)
- CLI options: `--input_size` (default 518), `--max_res` (default 1280), `--encoder vits|vitb|vitl`, `--max_len` (-1 = no limit), `--target_fps`, `--metric`, `--fp32` (default is FP16), `--grayscale`, `--save_npz`, `--save_exr`. The README states the method applies to arbitrarily long videos; no Windows notes are given — [VDA GitHub README](https://github.com/DepthAnything/Video-Depth-Anything)
- Streaming mode (`run_streaming.py`) is experimental and training-free, and the authors report a quality drop against the offline mode (ScanNet δ1 0.926 → 0.836) — [VDA GitHub README](https://github.com/DepthAnything/Video-Depth-Anything)
- Metric variants (trained on Virtual KITTI and IRS), README table: KITTI δ1 0.877 / 0.887 / 0.910 and ScanNet TAE 1.48 / 1.26 / 1.09 for metric S / B / L, versus DepthPro KITTI δ1 0.822 and TAE 2.73, MoGe-2-L TAE 2.56, UniDepthV2-L TAE 1.41 — [VDA GitHub README](https://github.com/DepthAnything/Video-Depth-Anything)
- Independent 2026 measurement on an RTX 5090 (192 frames at 518x924, FP16): VDA-L 43.0 ms/frame (8.3 s per sequence), FlashDepth 56.3 ms/frame (10.8 s), StableDPT 22.8 ms/frame (4.4 s); in FP32 VDA-L 124.3 ms/frame — [StableDPT, arXiv 2601.02793 (2026-01-06)](https://arxiv.org/html/2601.02793)

#### Online Video Depth Anything (oVDA, Heidelberg University)
- arXiv 2510.09182, submitted 2025-10-10; adapts VDA to online use by caching latent features at inference and masking future frames in training; reported 42 FPS on an A100 and 20 FPS on an NVIDIA Jetson edge device; the project page says code and compilation scripts will be released — [oVDA paper](https://ar5iv.arxiv.org/html/2510.09182); [oVDA project page](https://friedfeid.github.io/oVDA-website)
- A FiftyOne plugin wraps oVDA weights with two variants (8-frame cache for lower VRAM, 16-frame cache for better temporal consistency) and describes the project as having "mixed licensing" — [FiftyOne plugin page](https://docs.voxel51.com/plugins/plugins_ecosystem/online_video_depth_anything.html)

#### DepthCrafter (Tencent, CVPR 2025 Highlight)
- Needs about 26 GB GPU memory at 1024x576 (default), about 2.1 fps on an A100; about 9 GB at 512x256 with `--max-res 512`, about 8.6 fps on an A100 — [DepthCrafter GitHub README](https://github.com/Tencent/DepthCrafter)
- v1.0.1 (2024-11-26) benchmark at 1024x576, ms/frame and AbsRel/δ1: DepthCrafter v1.0.1 465.84 ms, Sintel 0.270/0.697, ScanNet 0.123/0.856, KITTI 0.104/0.896, Bonn 0.071/0.972; Depth-Anything-V2 180.46 ms, 0.367/0.554, 0.135/0.822, 0.140/0.804, 0.106/0.921; Marigold 1070.29 ms, 0.532/0.515, 0.166/0.769, 0.149/0.796, 0.091/0.931 — [DepthCrafter GitHub README](https://github.com/Tencent/DepthCrafter)
- ComfyUI integration 2024-10-19, Nuke integration 2024-10-08, EXR output 2024-12-10, codebase refactor 2025-12-01 — [DepthCrafter GitHub README](https://github.com/Tencent/DepthCrafter)

#### GeometryCrafter (TencentARC, ICCV 2025)
- Default 1024x576 with 110 frames needs about 40 GB and runs at 1.27 FPS; deterministic variant (`--model_type determ`) 1.50 FPS; 384x640 needs about 22 GB at 2.49 FPS; with `--low_memory_usage True --decode_chunk_size 6` under 20 GB at 1.76 FPS. Released 2025-04-01, accepted to ICCV 2025 on 2025-06-26 — [GeometryCrafter GitHub README](https://github.com/TencentARC/GeometryCrafter)

#### RollingDepth (ETH, CVPR 2025; the Marigold-lineage video model, "Video Depth without Video Models")
- Presets: `fast` (dilations [1, 25], fp16, no refinement, max res 768), `fast1024`, `full` (dilations [1, 10, 25], fp16, 10 refinement steps, max res 1024), `paper` (fp32, 10 refinement steps, max res 768). Tested on Debian 12, Python 3.12.7, CUDA 12.4, RTX 3090. Low-memory hints: `--max-vae-bs 1 --unload-snippet true`, `--res 512` — [RollingDepth GitHub README](https://github.com/prs-eth/rollingdepth)

#### Depth Any Video
- arXiv release 2024-10-15, Hugging Face demo 2024-10-20; CLI examples use `--max_resolution 960` for video. The README gives no VRAM, speed or frame-limit numbers — [DepthAnyVideo GitHub README](https://github.com/Nightmare-n/DepthAnyVideo)

#### FlashDepth (Eyeline / Netflix, ICCV 2025 Highlight)
- Streaming depth at 2044x1148 at 24 FPS on an A100 (100 frames in 4.15 s, 24.12 fps, "Full" model); three variants: FlashDepth (Full, fastest), FlashDepth-L (most accurate; recommended when the short side is under 518), FlashDepth-S — [FlashDepth GitHub README](https://github.com/Eyeline-Research/FlashDepth); [FlashDepth paper](https://arxiv.org/abs/2504.07093)
- Environment constraints: Python 3.11, "the torch version should be 2.4", the Mamba package must be installed from the repo's local folder, and "Mamba2 does not work if compiling torch 2.5 and above" — [FlashDepth GitHub README](https://github.com/Eyeline-Research/FlashDepth)
- The authors themselves name residual flickering as the main limitation — [FlashDepth paper](https://arxiv.org/abs/2504.07093)
- Third-party 2026 evaluation (averages over Infinigen, Sintel, KITTI, TUM RGB-D): FlashDepth AbsRel 0.23, δ1 71.40, TGM 0.43 — worse temporal metric than per-frame DAv2 (TGM 0.35) and released VDA-L (AbsRel 0.16, δ1 83.31, TGM 0.25). On Infinigen: OPW 0.054 (FlashDepth) versus 0.039 (DAv2), 0.026 (VDA), 0.023 (StableDPT) — [StableDPT, arXiv 2601.02793](https://arxiv.org/html/2601.02793)

#### Depth Anything 3 (ByteDance, released 2025-11-14)
- Model zoo: DA3NESTED-GIANT-LARGE(-1.1) 1.40B, DA3-GIANT(-1.1) 1.15B, DA3-LARGE(-1.1) 0.35B, DA3-BASE 0.12B, DA3-SMALL 0.08B (any-view models with relative depth + pose estimation), DA3METRIC-LARGE 0.35B (monocular metric), DA3MONO-LARGE 0.35B (monocular relative). The `-1.1` models were retrained after a training bug fix and are recommended over the originals — [DA3 GitHub README](https://github.com/ByteDance-Seed/Depth-Anything-3)
- DA3-Streaming (2025-12-11) does sliding-window inference for ultra-long video with a claimed < 12 GB GPU memory; a `da3 video` CLI command exists. The README does not discuss temporal (flicker) consistency for video, only multi-view consistency — [DA3 GitHub README](https://github.com/ByteDance-Seed/Depth-Anything-3)
- DA3-Streaming details: built on the ideas of VGGT-Long; explicitly "not an SLAM system"; chunk sizes 120/90/60/30 with overlap of half the chunk; peak VRAM at 504x378 (TUM): 28.3 / 25.1 / 21.2 / 18.7 GB for chunk 120 / 90 / 60 / 30; at 504x154 (KITTI): 15.9 / 14.3 / 12.7 / 11.5 GB. Speed on an A100: 8.51 FPS (VGGT-Long 2.91, Pi-Long 3.15). Outputs camera poses, intrinsics, point cloud and optional per-frame depth/confidence — [DA3-Streaming README](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md)
- Install: `xformers`, `torch>=2`, `torchvision`, `pip install -e .`; no Windows-specific instructions; loaded through the project's own `DepthAnything3.from_pretrained` API — [DA3 GitHub README](https://github.com/ByteDance-Seed/Depth-Anything-3)

#### DepthPro (Apple)
- Single-image metric depth with focal length; 2.25-megapixel depth map in 0.3 s on a standard GPU; no video or temporal-consistency mechanism — [ml-depth-pro GitHub](https://github.com/apple/ml-depth-pro)
- In the VDA README's metric comparison DepthPro has the worst ScanNet TAE of the listed models (2.73) — [VDA GitHub README](https://github.com/DepthAnything/Video-Depth-Anything)

#### Feed-forward reconstruction models (VGGT, π3, MapAnything)
- VGGT: reconstructs a scene "in less than 1 second"; a 2026-05-15 memory fix allows "roughly 2-3x more input frames" in the same GPU budget; VGGT-Omega was released 2026-05-18; the README has no video-depth or temporal-consistency section — [VGGT GitHub README](https://github.com/facebookresearch/vggt)
- π3 (Pi3, ICLR 2026): outputs global points, per-view local point maps, confidence and 4x4 camera-to-world poses; no separate depth output listed; Pi3X (2025-12-28) adds a convolutional head, optional pose/intrinsics/depth conditioning and approximate metric scale — [Pi3 GitHub README](https://github.com/yyfz/Pi3)
- MapAnything: the paper describes a universal feed-forward metric 3D reconstruction model released with an Apache-2.0 pre-trained model — [MapAnything paper](https://arxiv.org/html/2509.13414v1)

#### Optimisation / SLAM-style pipelines that also give consistent depth (MegaSaM, ViPE)
- MegaSaM (Google): outputs camera poses and consistent video depth (depth optimisation given the estimated cameras); tested with Python 3.10, CUDA 11.8, PyTorch 2.0.1; needs xformers for UniDepth and a compiled extension (`cd base; python setup.py install`); bundles Depth-Anything and UniDepth directories and a RAFT checkpoint; scripts are `.sh` — [MegaSaM GitHub README](https://github.com/mega-sam/mega-sam)
- ViPE (NVIDIA): estimates camera intrinsics, camera motion and dense near-metric depth from raw video (pinhole, wide-angle, 360° panorama); `pip install nvidia-vipe`; default keyframe depth prior MoGe v2 (`moge2-l`), DA3 pipeline (`dav3`) added 2026/01; v1.0.0 2026/05; v1.2.0 2026/06 with a 2.7x speed-up; long-sequence SLAM `pose_only_long` 2026/09; ViPE-Long `metrictrail` pipeline 2026/11 — [ViPE GitHub README](https://github.com/nv-tlabs/vipe)

#### New in 2026
- ViGeo ("Towards Consistent Video Geometry Estimation", arXiv 2605.30060, v1 2026-05-28, v3 2026-07-16): feed-forward plain-transformer model predicting depth, point maps and surface normals; one model supports streaming, full-sequence and long-video inference through "dynamic chunking attention"; authors claim state of the art on online, offline and long-video depth — [ViGeo arXiv abstract](https://arxiv.org/abs/2605.30060)
- ViGeo code is public and Apache-2.0; ViGeo 1.0, ViGeo 1.1 and VideoLDCM weights are on Hugging Face (news entries dated 2026-10-14 and 2026-10-15); default environment Python 3.10, PyTorch 2.7.1, CUDA 12.6; streaming via `model.infer(image_chunk, mode="online", kv_caches=...)`, long video via `chunk` mode with default chunk size 16. The README gives no VRAM/speed and no numeric comparison against VDA/DepthCrafter/DA3 — [ViGeo GitHub README](https://github.com/aigc3d/ViGeo); [ViGeo project page](https://pkqbajng.github.io/ViGeo/)
- StableDPT (ETH / DisneyResearch|Studios, arXiv 2601.02793, 2026-01-06): frozen DAv2 encoder + DPT head with temporal cross-attention over keyframes; strided inference without overlapping windows; trained on about 360k frames on a single RTX 5090 in about three days; ranks second behind the released VDA-L on most metrics (avg AbsRel 0.22, δ1 74.32, TGM 0.27 at 22.8 ms/frame); no code or weights mentioned — [StableDPT paper](https://arxiv.org/html/2601.02793)
- DyFN, "Stabilizing Streaming Video Geometry via Dynamic Feature Normalization" (CVPR 2026): a causal recurrent module that modulates feature statistics of a frozen monocular geometry model (about 2% extra parameters); the authors claim up to 14% improvement over prior streaming methods (from a search-result summary; the CVF page returned HTTP 403 to me) — [CVPR 2026 open access page](https://openaccess.thecvf.com/content/CVPR2026/html/Lyu_Stabilizing_Streaming_Video_Geometry_via_Dynamic_Feature_Normalization_CVPR_2026_paper.html)
- PVDepth (arXiv 2611.07873, 2026-11-05) targets panoramic (equirectangular) video depth; code "will be released" — not relevant to perspective footage — [PVDepth arXiv](https://arxiv.org/abs/2611.07873)

### Inferences
- For 5–15 s, 24 fps clips (120–360 frames) every candidate handles the length: VDA via its 32-frame sliding window, DepthCrafter/RollingDepth via their own windowing; Depth Any Video's 192-frame limit (8 s at 24 fps) is the one hard length constraint in the list.
- VDA-Small is the only VDA size whose published VRAM (6.8–7.3 GB at 32x518x518) leaves large headroom on 24 GB. VDA-Large at 23.6 GB FP16 for a square 518x518 window is at the edge of a 4090; a 16:9 frame at short side 518 (518x924) has about 1.78x the patch tokens of 518x518 (37x66 versus 37x37 patches), so VDA-Large at native 16:9 input will probably not fit 24 GB without lowering `--input_size`. This is arithmetic on published numbers, not a measurement.
- At 14–67 ms/frame (A100) a 360-frame clip is seconds to tens of seconds for VDA, versus about 168 s for DepthCrafter v1.0.1 at 465.84 ms/frame — and DepthCrafter's 1024x576 mode (about 26 GB) does not fit a 4090 at all, so it would have to run at reduced resolution.
- FlashDepth is a poor fit for this studio's stack: it is pinned to torch 2.4 with a custom Mamba build (the tool runs PyTorch 2.11 on Windows), the authors admit residual flicker, and a third-party 2026 evaluation measured it as less temporally stable than even per-frame DAv2.
- DA3's any-view models give geometric (multi-view) consistency, which is useful with camera motion and a static scene, but the README makes no claim about flicker on moving people; DA3MONO/DA3METRIC are per-frame models and should be expected to shimmer like DAv2 unless post-processed.
- The fact that VDA's detail metrics equal DAv2-L's suggests switching from per-frame DAv2 to VDA of the same encoder size costs no spatial detail; switching from DAv2-L to VDA-S (for licence reasons) trades detail (e.g. ScanNet δ1 0.876 versus 0.926 for VDA-L) for much better stability (TAE 0.703 versus 1.140).

### Gaps
- No RTX 4090 measurements for any model; VDA-Base latency/VRAM are not published in the README or paper.
- ChronoDepth: only the numbers reported in the VDA paper were found; I did not open its repository (VRAM, clip length unknown).
- Marigold's own video mode (`MarigoldDepthEstimation_v2_video` exists as a ComfyUI node name) was not researched beyond RollingDepth.
- ViGeo: no VRAM, speed, model size or independent benchmark found; the benchmark numbers are only on a separate branch/figure.
- oVDA: could not confirm that code/weights are actually published or under what terms.
- DyFN and VDPP (CVPR 2026 / CVPR 2026 workshop): CVF pages returned HTTP 403; code availability unknown.
- VGGT-Omega (2026-05-18): no details read. π3 VRAM/speed not stated in its README.
- I found no 2026 successor to Video Depth Anything from the original authors ("VDA v2"); the arXiv v4 of 2026-10-14 does not state what changed.

---

## 2. Exact licences of code and weights; what is safe for commercial use

### Takeaway
Commercially usable today with clearly stated terms: Video-Depth-Anything-Small (Apache-2.0), Metric-Video-Depth-Anything-Small (Apache-2.0 tag on Hugging Face), DA3-SMALL, DA3-BASE, DA3MONO-LARGE and DA3METRIC-LARGE (Apache-2.0), VGGT-1B-Commercial (gated, excludes military use), MapAnything's Apache variant, and NVDS (MIT). VDA Base/Large (relative and metric), DA3 Large/Giant/Nested, π3 weights, DepthCrafter and GVHMR-class research code are non-commercial.

### Cited Findings

| Model | Code licence | Weights licence | Source |
|---|---|---|---|
| Video-Depth-Anything-Small | Apache-2.0 (repo) | Apache-2.0 ("Video-Depth-Anything-Small model is under the Apache-2.0 license") | [VDA README](https://github.com/DepthAnything/Video-Depth-Anything) |
| Video-Depth-Anything-Base / Large | Apache-2.0 (repo) | CC-BY-NC-4.0 ("Video-Depth-Anything-Base/Large model is under the CC-BY-NC-4.0 license") | [VDA README](https://github.com/DepthAnything/Video-Depth-Anything); [HF card, Base](https://huggingface.co/depth-anything/Video-Depth-Anything-Base) |
| Metric-Video-Depth-Anything-Small | – | Hugging Face licence tag `apache-2.0` (model card body is empty) | [HF card](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small) |
| Metric-Video-Depth-Anything-Base | – | Hugging Face licence tag `cc-by-nc-4.0` | [HF card](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Base) |
| Metric-Video-Depth-Anything-Large | – | Hugging Face licence tag `cc-by-nc-4.0` | [HF card](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large) |
| DA3-SMALL, DA3-BASE | Apache-2.0 | Apache 2.0 | [DA3 README](https://github.com/ByteDance-Seed/Depth-Anything-3) |
| DA3MONO-LARGE, DA3METRIC-LARGE | Apache-2.0 | Apache 2.0 | [DA3 README](https://github.com/ByteDance-Seed/Depth-Anything-3) |
| DA3-LARGE(-1.1), DA3-GIANT(-1.1), DA3NESTED-GIANT-LARGE(-1.1) | Apache-2.0 | CC BY-NC 4.0 | [DA3 README](https://github.com/ByteDance-Seed/Depth-Anything-3) |
| DepthCrafter | Custom Tencent licence | Same licence covers "model inference code and weights"; "use the DepthCrafter only for academic, research and education purposes"; must "refrain from using it for any commercial or production purposes under any circumstances"; business licensing via wbhu@tencent.com | [DepthCrafter LICENSE](https://github.com/Tencent/DepthCrafter/blob/main/LICENSE); [README](https://github.com/Tencent/DepthCrafter) |
| RollingDepth | Apache-2.0 | RAIL++-M (HF tag `openrail`) | [RollingDepth README](https://github.com/prs-eth/rollingdepth); [HF model card](https://huggingface.co/prs-eth/rollingdepth-v1-0) |
| FlashDepth | Apache-2.0 (repo sidebar and README) | Not stated separately in the README | [FlashDepth README](https://github.com/Eyeline-Research/FlashDepth) |
| ViGeo | Apache-2.0 | Not stated | [ViGeo README](https://github.com/aigc3d/ViGeo) |
| VGGT | Licence updated 2025-07-29 "to permit commercial use (excluding military applications)" | Original VGGT-1B "remains non-commercial"; "only the newly released checkpoint VGGT-1B-Commercial is licensed for commercial usage" (access via an automatically processed application form on Hugging Face) | [VGGT README](https://github.com/facebookresearch/vggt) |
| π3 / Pi3X | BSD 3-Clause (commercial use permitted) | CC BY-NC 4.0, "Strictly Non-Commercial" — applies to both Pi3 and Pi3X | [Pi3 README](https://github.com/yyfz/Pi3) |
| MapAnything | Apache-2.0 model described in the paper | – | [MapAnything paper](https://arxiv.org/html/2509.13414v1) |
| MegaSaM | Apache-2.0 for software, CC-BY 4.0 for other materials (Copyright 2025 Google LLC, "not an officially supported Google product") | Uses third-party Depth Anything, UniDepth and RAFT checkpoints whose licences the README does not state | [MegaSaM README](https://github.com/mega-sam/mega-sam) |
| ViPE | Apache-2.0 for NVIDIA-authored source; UniK3D part BY-NC-SA 4.0; DPVO MIT; LieTorch BSD-3-Clause; MetricTrail weights under a separate NVIDIA License; third-party models "are not distributed by NVIDIA" | per third-party model | [ViPE README](https://github.com/nv-tlabs/vipe) |
| NVDS / NVDS+ | MIT ("adopts the widely-used MIT License") | Same statement covers the NVDS model; the VDW dataset is CC BY-NC-SA 4.0 | [NVDS README](https://github.com/RaymondWang987/NVDS) |
| All-In-One-Deflicker | Apache-2.0 (repo label) | – | [All-In-One-Deflicker README](https://github.com/ChenyangLEI/All-In-One-Deflicker) |

- Background confirmation of the parent models: Depth-Anything-V2-Small is Apache-2.0 and V2 Base/Large/Giant are CC-BY-NC-4.0 — [Depth Anything V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)
- GeometryCrafter's README links LICENSE and NOTICE files without stating terms and directs business licensing to wbhu@tencent.com — [GeometryCrafter README](https://github.com/TencentARC/GeometryCrafter)
- DepthPro: "The model weights are released under the LICENSE terms" (Apple sample-code licence file; the README does not name it or address commercial use) — [ml-depth-pro GitHub](https://github.com/apple/ml-depth-pro)
- Community re-packagings (ONNX, CoreML, ComfyUI conversions) of DA3 carry their own licence tags that sometimes differ from upstream; the artifact actually downloaded should be checked against the upstream table — [search summary over HF mirrors, e.g. SpatialHub/depth-anything-3-onnx](https://huggingface.co/SpatialHub/depth-anything-3-onnx)

### Inferences
- The straightforward commercially clean replacement for per-frame DAv2 is VDA-Small (relative) — the same licence position the studio already has with DAv2-Small. There is no commercially licensed VDA Base/Large; getting more detail under a permissive licence means either a DA3 Apache model (per-frame or any-view) plus stabilisation, or negotiating with the rights holders.
- RAIL++-M (RollingDepth weights) is a use-restricted "responsible AI" licence, not a non-commercial one in name, but its terms were not read here; treat as "needs legal review".
- DepthCrafter and (very likely) GeometryCrafter are unusable in a commercial production pipeline without a Tencent business licence.
- MegaSaM's Apache-2.0 code does not clean its dependencies: it ships with UniDepth and Depth Anything checkpoints and a DROID-SLAM-derived tracker, each with its own terms.

### Gaps
- GeometryCrafter, Depth Any Video and ChronoDepth: licence text not read — unknown.
- FlashDepth and ViGeo weight licences: not stated on the pages read — unknown.
- DepthPro's licence file was not opened; whether it permits commercial use is unverified here.
- oVDA licence: unknown ("mixed licensing" per a third-party plugin page).
- The Hugging Face tag for Video-Depth-Anything-Small was not opened directly (the README statement was); the metric Small card has an empty README, so the `apache-2.0` tag is the only evidence for it.
- MapAnything: the split between `facebook/map-anything` (CC-BY-NC 4.0) and `facebook/map-anything-apache` (Apache 2.0) came from a search-result summary of its docs, not a page I opened.
- Licences of the UniDepth / Depth Anything (v1) / RAFT checkpoints used by MegaSaM were not checked.

---

## 3. Hugging Face transformers integration, simple PyTorch inference, ComfyUI nodes and practical settings on 24 GB

### Takeaway
None of the video-consistent depth models was confirmed as a native `transformers` model; VDA and DA3 ship their own small PyTorch packages (`run.py` / `DepthAnything3.from_pretrained`) that are easy to call from Python. Community ComfyUI documentation points to the same practical settings as the VDA CLI: internal `input_size` 518, `max_res` 1280, FP16, Large only with a full 24 GB, and "lower `max_res` first, then split the clip, then drop to Small" on out-of-memory.

### Cited Findings
- The transformers documentation (docs paths show v5.23.0) has native pages for DepthPro (`DepthProForDepthEstimation`, checkpoint `apple/DepthPro-hf`) and ViTPose (`VitPoseForPoseEstimation`) — [transformers DepthPro doc](https://huggingface.co/docs/transformers/en/model_doc/depth_pro); [transformers ViTPose doc](https://huggingface.co/docs/transformers/en/model_doc/vitpose)
- The transformers Depth Anything page covers the image model through `AutoModelForDepthEstimation` — [transformers Depth Anything doc](https://huggingface.co/docs/transformers/en/model_doc/depth_anything)
- The only Hugging Face "transformers" hit for Video Depth Anything was an unofficial port of the Base variant that needs remote code enabled — [bigshanedogg/Video-Depth-Anything-Base](https://huggingface.co/bigshanedogg/Video-Depth-Anything-Base)
- DA3 checkpoints are hosted under the `depth-anything` organisation on the Hub but are loaded with the project's own `depth-anything-3` package, not a transformers pipeline; a `Comfy-Org/Depth-Anything-3` repackaging exists on the Hub — [DA3 README](https://github.com/ByteDance-Seed/Depth-Anything-3); [Comfy-Org/Depth-Anything-3](https://huggingface.co/Comfy-Org/Depth-Anything-3)
- VDA install is `pip install -r requirements.txt` plus `bash get_weights.sh`, then `python3 run.py --input_video ... --encoder vits|vitb|vitl [--metric] [--grayscale] [--save_npz|--save_exr]`; FP16 is the default — [VDA README](https://github.com/DepthAnything/Video-Depth-Anything)
- ComfyUI VDA node (`VideoDepthAnythingProcess`): `input_size` default 518 is the internal resolution the transformer sees; `max_res` (default 1280) caps the longer side of the input frames; third-party node documentation says the Large model needs at least 24 GB and otherwise suggests the Small model or a smaller input size — [RunComfy node page](https://www.runcomfy.com/comfyui-nodes/ComfyUI-Video-Depth-Anything/video-depth-anything-process); [comfy.icu node page](https://comfy.icu/node/VideoDepthAnythingProcess)
- A community node guide (`TS_VideoDepthNode`) states that `input_size` 644 gives about 54% more depth detail for about 54% more VRAM/time and is "safe on ≥24 GB", that values of 770 and above may hurt quality, that lowering `max_res` is the first lever against OOM, and that the node auto-retries 518 → 392 → 280 → 168 on OOM (reported via search-result summary; page not opened by me) — [comfy.icu TS_VideoDepthNode](https://comfy.icu/node/TS_VideoDepthNode)
- DepthCrafter ComfyUI nodes: `max_res` in increments of 64; the node author's Reddit post (October 2024) said VRAM need is "pretty high (>16GB)" for long videos at 768p and up — [RunComfy DepthCrafter node page](https://www.runcomfy.com/comfyui-nodes/ComfyUI-DepthCrafter-Nodes/depth-crafter); [Reddit r/StableDiffusion, 2024-10-19](https://redlib.hackliberty.org/r/StableDiffusion/comments/1g71vsc/depthcrafter_comfyui_nodes)
- RollingDepth documents low-memory flags (`--max-vae-bs 1 --unload-snippet true`, `--res 512`) and was tested on a 24 GB RTX 3090 — [RollingDepth README](https://github.com/prs-eth/rollingdepth)
- DA3 FAQ covers running without xformers (points to Issue #11) — [DA3 README](https://github.com/ByteDance-Seed/Depth-Anything-3)

### Inferences
- VDA is the lowest-friction drop-in: pure PyTorch + a DINOv2-style encoder and a temporal head, no diffusion stack, no custom CUDA ops, FP16 by default; nothing in its README is Linux-specific apart from `bash get_weights.sh` (weights can be downloaded manually). Not tested on Windows by me.
- Because the depth is computed at `input_size` (518 short side) regardless of 1080p/4K source, 4K sources bring no extra depth detail by default; raising `input_size` (e.g. 644) is the detail lever and costs VRAM roughly linearly in pixels.
- FlashDepth (torch 2.4 + custom Mamba), MegaSaM (torch 2.0.1/CUDA 11.8 + compiled extension + shell scripts) and NVDS (PyTorch 1.9, mmcv-full 1.x) all conflict with a PyTorch 2.11 Windows environment and would need a separate environment or WSL2.

### Gaps
- I could not confirm whether transformers 5.x has since added native Video Depth Anything, Depth Anything 3 or VGGT classes; I found no doc page for them but did not enumerate the full model index.
- No community 4090 benchmark (Reddit or otherwise) with concrete seconds-per-clip for VDA-S/B/L at 518 or 644 was found.
- Windows-specific install reports for VDA, DA3, ViGeo were not found.

---

## 4. Post-processing options if a per-frame model is kept

### Takeaway
Published evidence is not encouraging for bolt-on stabilisers: in the VDA paper NVDS applied to Depth Anything V2-L made temporal error worse (TAE 2.536 versus 1.140 for raw DAv2-L) and reduced accuracy, and the practical neural stabilisers (NVDS: PyTorch 1.9 + mmcv 1.x; All-In-One-Deflicker: per-video optimisation, ≤200 frames, one foreground object) are awkward to deploy. A new 2026 lightweight option (VDPP) claims VDA-level results on top of per-frame DAv2, but I could not confirm released code.

### Cited Findings
- NVDS (ICCV 2023) / NVDS+ (TPAMI 2024) is a plug-and-play stabiliser for any single-image depth model; MIT licence; on the VDW test set it reduced OPW from 0.676 to 0.180 for MiDaS and from 0.470 to 0.147 for DPT while improving δ1 (0.651 → 0.700; 0.730 → 0.742) — [NVDS README](https://github.com/RaymondWang987/NVDS)
- NVDS resource use: at 896x384, DPT-Large plus the stabiliser need about 20 GB and 5 GB respectively on an RTX A6000; default inference height 384; requires Python 3.8.13, PyTorch 1.9.0, mmcv-full 1.x and mmseg 0.11.0 (mmcv 2.x removes APIs the code uses); GMFlow is needed for the temporal loss/OPW metric — [NVDS README](https://github.com/RaymondWang987/NVDS)
- In the VDA paper's benchmark NVDS + DAv2-L scored worse than DAv2-L alone on both accuracy (e.g. ScanNet AbsRel 0.194 versus 0.150) and temporal error (TAE 2.536 versus 1.140) — [VDA paper v4](https://arxiv.org/html/2501.12375v4)
- All-In-One-Deflicker ("Blind Video Deflickering by Neural Filtering with a Flawed Atlas"): builds a neural atlas of the video and a neural filter to correct it; optimisation is per video (e.g. 10,000 iterations for an 80-frame clip at 4x downsampling); about 3 GB GPU memory at inference; videos over 200 frames are not evaluated and should be split; segmentation masks support only one foreground object; environment Python 3.10 / PyTorch 1.12 / CUDA 11; evaluated on generated, archival, time-lapse, slow-motion and image-processed videos — depth videos are not addressed — [All-In-One-Deflicker README](https://github.com/ChenyangLEI/All-In-One-Deflicker)
- VDPP (Seoul National University / Samsung, arXiv 2604.06665, 2026-04-08; CVPR 2026 ECV workshop): an RGB-free post-processing module that performs "targeted geometric refinement" in low-resolution space on top of an image depth model; more than 43.5 FPS on an NVIDIA Jetson Orin Nano; the authors say it outperforms existing post-processing frameworks by an order of magnitude in speed, performs on par with end-to-end VDA, and that NVDS shows severe flicker and degraded spatial fidelity in their comparison (from the abstract and search-result summaries; full PDF not opened) — [VDPP arXiv](https://arxiv.org/abs/2604.06665)
- StableDPT argues that overlapping-window inference introduces scale misalignment between windows and redundant computation, and instead uses keyframes sampled across the whole video with strided inference — [StableDPT paper](https://arxiv.org/html/2601.02793)
- VDA itself handles window-to-window alignment by blending 8 overlap frames with linearly decaying weights and re-feeding 2 key frames; this is the closest published "scale-and-shift to neighbours" recipe among the sources read — [VDA paper v4](https://arxiv.org/html/2501.12375v4)
- ffmpeg ships `deflicker`, `tmix`, `atadenoise`, `hqdn3d`, `gradfun` and `deband` filters (confirmed in the filter index of the official documentation; option text was not retrieved in this session) — [ffmpeg filters documentation](https://ffmpeg.org/ffmpeg-filters.html)
- The FlashDepth authors report residual flickering even for a model with a recurrent (Mamba) temporal module, i.e. online/causal smoothing alone does not fully remove flicker — [FlashDepth paper](https://arxiv.org/abs/2504.07093)

### Inferences
- The studio's current pipeline (per-shot min/max normalisation + Gaussian temporal blend) addresses global brightness flicker and high-frequency noise but not the two main sources of "boiling" in per-frame affine-invariant depth: (a) per-frame scale/shift drift and (b) spatially local inconsistency. Engineering options, in order of cost — none of these was benchmarked in a source I read, so treat them as hypotheses to A/B test:
  1. Per-frame scale-and-shift alignment: least-squares fit (a, b) of each frame's depth to the previous (or a running reference) frame over static/robust pixels before normalisation; replace per-shot min/max with robust percentiles (e.g. 1st–99th) computed after alignment. Cheap, removes global pumping, no ghosting.
  2. Flow-warped temporal EMA: warp the previous filtered depth to the current frame with optical flow (e.g. RAFT / GMFlow / torchvision RAFT), blend with the current depth, and gate the blend by forward–backward flow consistency and by depth difference so occlusions and fast motion fall back to the current frame. Removes local shimmer; failure modes are ghost trails behind fast limbs and smearing at occlusion boundaries when flow is wrong.
  3. Edge-aware (joint bilateral / guided) temporal filter using the RGB frame as guide: preserves silhouettes better than a plain Gaussian blend, same ghosting risk if not motion-compensated.
  4. A plain non-motion-compensated temporal Gaussian/`tmix` blend — what the tool has now — necessarily ghosts on fast motion and on camera moves, because it averages depth values of different surfaces.
- ffmpeg `deflicker` corrects frame-level luminance, so on a depth video it can only correct global (whole-frame) scale/shift flicker, not local boiling; it would also partly duplicate what an explicit scale/shift alignment does more exactly.
- Given that VDA-S is Apache-2.0, as fast as DAv2-S, and measurably more stable than DAv2-L with NVDS, replacing the model is likely the better return on effort than building a stabiliser — with post-processing options 1–2 kept as an optional extra pass.

### Gaps
- No source was found that benchmarks flow-warped EMA, per-frame scale/shift alignment or bilateral temporal filtering on Depth Anything V2 output; the trade-off statements above are engineering reasoning, not cited results.
- ffmpeg `deflicker` option names and defaults (window size, averaging mode) were not retrieved; verify against the documentation before use.
- VDPP code/weights availability and licence: unknown. Exact VDPP versus VDA numbers were not read.
- NVDS+ (TPAMI) improvements over NVDS, and whether NVDS has been retrained for DAv2, were not verified.
- No information found on NVDS/All-In-One-Deflicker running on Windows.

---

## 5. Does compression matter for a depth control video, and how should it be encoded for upload?

### Takeaway
I found no authoritative guidance specific to depth control videos for hosted generators; general encoding sources confirm that smooth gradients are exactly where 8-bit video bands, and that dithering when reducing to 8-bit helps. Third-party Seedance 2.5 API documentation lists only container, resolution, duration, frame-rate and size limits for reference videos (MP4/MOV, 24–60 fps, 2–30 s, validated codec) and gives no codec, bit-depth or quality recommendation.

### Cited Findings
- Banding appears when a format has too few brightness levels for a smooth gradient; encoding at 10-bit makes the codec's internal rounding happen on a finer grid and typically costs only 5–15% more bitrate for the same 4K scene; for H.264 the 10-bit profile "exists on paper" but almost no consumer device decodes it in hardware, and H.265 is suggested for 10-bit; converting a 10-bit master to 8-bit with dithering gives smoother gradients than truncation (article dated 2026-05-15) — [Fora Soft, bit depth and banding](https://www.forasoft.com/learn/video-encoding/articles/bit-depth-8-10-12-banding)
- Seedance 2.5 reference-to-video constraints as documented by a third-party API gateway: `.mp4` or `.mov`; 480p, 720p, 1080p or 4K; 2–30 s per clip, up to 10 clips totalling ≤ 30 s; aspect ratio 0.4–2.5; width and height each 300–6000 px, frame area 409,600–8,295,044 px; 24–60 FPS; ≤ 200 MB per video; the service checks codec and media validity but lists no specific codecs; reference videos are described as camera-movement and motion references; no encoding recommendations — [EvoLink Seedance 2.5 reference-to-video docs](https://evolink.ai/docs/en/api-manual/video-series/seedance2.5/seedance-2.5-reference-to-video.md)
- Other third-party pages disagree on Seedance limits (e.g. Seedance 2.0: 480p–720p input, frame area 409,600–927,408 px, ≤ 50 MB, total ≤ 15 s; some 2.5 pages list 1080p as the maximum) — [search summary across gateways, e.g. fal.ai Seedance 2.5 reference-to-video](https://fal.ai/models/bytedance/seedance-2.5/reference-to-video/api); [Dreamina video reference FAQ](https://dreamina.capcut.com/ai-video/ai-video-reference-inputs-faq)
- VDA can write lossless float outputs (`--save_npz`, `--save_exr`) alongside the visualisation video, so quantisation can be deferred to the final encode step — [VDA README](https://github.com/DepthAnything/Video-Depth-Anything)

### Inferences
- With a 720p–1080p (or 4K) 24 fps clip of 5–15 s, the 200 MB (or even 50 MB) cap is far above what near-lossless 8-bit H.264 needs, so there is no reason to compress hard: CRF 12 or lower is fine, and bitrate is not the binding constraint — codec compatibility is.
- Compatibility risk to check in the current pipeline: H.264 encoded with a true monochrome `gray` pixel format (4:0:0) and/or 10-bit (High 10) is valid H.264 but is rejected or mis-decoded by some hardware/web decoders; since the hosted service "checks codec validity" and its decoder is unknown, the conservative upload format is 8-bit H.264 High profile, `yuv420p`, full grey replicated in Y with neutral chroma, MP4 with `+faststart`. This is a precaution, not a documented requirement.
- Because the hosted model will decode to 8-bit and almost certainly resize and re-encode internally (its VAE works at a lower internal resolution), 10-bit upload is unlikely to survive end to end; the useful defence against banding is to do normalisation in float (from NPZ/EXR or float tensors), then quantise once to 8-bit with dithering (error-diffusion or small blue/Gaussian noise of about ±0.5 LSB) and encode with enough bitrate that the dither is not flattened (low CRF; x264 `-tune grain` or a raised `aq-strength` are the usual knobs).
- Temporal dithering noise is itself a (tiny) flicker source; if the generator proves sensitive to it, a static (per-shot fixed) dither pattern or simply a slightly stronger gamma/contrast stretch of the depth range are alternatives to test.
- Limited-range (16–235) versus full-range signalling matters for a control signal: a range mismatch on the server side compresses or clips the depth range. Tagging the file explicitly (`-color_range`, BT.709) and keeping depth values away from 0 and 255 is the safe choice.

### Gaps
- No official ByteDance / Volcengine documentation was read; all Seedance input constraints above come from third-party gateways that disagree with each other. Whether Seedance 2.5 treats a depth or skeleton video differently from an ordinary reference video is undocumented in the sources found.
- No source was found that measures the effect of CRF, bit depth or dithering of a depth control video on generated-video quality (for Seedance, Wan/VACE or ControlNet-style models).
- Whether Seedance accepts HEVC, 10-bit or monochrome H.264 uploads is unknown.

---

## 6. Best current options for 2D whole-body pose (body + hands + face) on video

### Takeaway
For a commercially usable, Windows-friendly whole-body stack, `rtmlib` (Apache-2.0, ONNX Runtime, no mmcv) running RTMW or DWPose with a YOLOX/RTMDet detector is the practical default — RTMW-l/x at 384x288 reach about 70 whole-body AP versus 66.5 for DWPose-l-384 — with ViTPose++ available natively in transformers. Ultralytics YOLO pose is body-only (17 keypoints) and AGPL-3.0/Enterprise; OpenPose is non-commercial; Sapiens v1 is CC BY-NC; Sapiens2 (April 2026, 308 keypoints) has a custom Meta licence with use restrictions.

### Cited Findings
- rtmlib: Apache-2.0; runs RTMPose, DWPose, RTMO, RTMW, RTMW3D and ViTPose/ViTPose++ models "without mmcv or mmpose"; backends onnxruntime, opencv, openvino (TensorRT optional); devices cpu, cuda, mps; install `onnxruntime-gpu` for CUDA. Its README carries a separate note (issue #65) that model weights trained on some non-commercial datasets such as COCO-WholeBody are a distinct question from the code licence — [rtmlib GitHub README](https://github.com/Tau-J/rtmlib)
- rtmlib whole-body (133 keypoints, COCO-WholeBody AP as listed): ViTPose++-s/b/l 54.4 / 57.4 / 60.6 (256x192); DWPose-t/s/m/l 48.5 / 53.8 / 60.6 / 63.1 (256x192), DWPose-l 66.5 (384x288); RTMW-m/l 58.2 / 66.0 (256x192), RTMW-l/x 70.1 / 70.2 (384x288); RTMW3D-x 68.0 (384x288). Body-only: RTMPose-l/x 78.3 / 78.8 AP (384x288), RTMO-s/m/l (one-stage) 68.6 / 72.6 / 74.8, ViTPose++-s/b/l 75.8 / 77.0 / 78.6. Detectors: YOLOX (38.9 AP nano to 61.3 AP x), RF-DETR, RTMDet — [rtmlib GitHub README](https://github.com/Tau-J/rtmlib)
- rtmlib API: `Wholebody(mode='performance'|'balanced'|'lightweight', backend=..., device=..., to_openpose=...)`; `PoseTracker(Body, det_frequency=10, ...)` for video; `draw_skeleton(img, keypoints, scores, kpt_thr=..., to_openpose=...)` — [rtmlib GitHub README](https://github.com/Tau-J/rtmlib)
- DWPose: repository Apache-2.0; COCO-WholeBody AP 0.485 / 0.538 / 0.606 / 0.631 for t/s/m/l at 256x192 (0.5 / 0.9 / 2.2 / 4.5 GFLOPs) and 0.665 for DWPose-l at 384x288 (10.1 GFLOPs); the ControlNet setup uses `dw-ll_ucoco_384.onnx` + `yolox_l.onnx`; ONNX release 2023-08-07, sd-webui-controlnet support 2023-08-09, Animate Anyone integration 2023-12-03 — [DWPose GitHub README](https://github.com/IDEA-Research/DWPose)
- RTMW (arXiv 2407.08634) is the MMPose team's real-time multi-person 2D/3D whole-body model; MMPose is Apache-2.0. A Hugging Face mirror of the DWPose/RTMW ONNX graphs warns that the mmpose licence does not assert terms for the training data (HumanArt, COCO-WholeBody) behind the checkpoints — [RTMW paper](https://arxiv.org/abs/2407.08634); [SceneWorks/dwpose-onnx model card](https://huggingface.co/SceneWorks/dwpose-onnx)
- ViTPose / ViTPose++ in transformers: `VitPoseForPoseEstimation` with checkpoints under the `usyd-community` collection (e.g. `vitpose-base-simple`, `vitpose-plus-base`, `vitpose-plus-huge`); top-down, so a person detector is needed first (the doc example uses `RTDetrForObjectDetection` with `PekingU/rtdetr_r50vd_coco_o365`); ViTPose++ has 6 MoE expert heads selected by `dataset_index` — COCO 0, AiC 1, MPII 2, AP-10K 3, APT-36K 4, COCO-WholeBody 5; default input 256x192 — [transformers ViTPose doc](https://huggingface.co/docs/transformers/en/model_doc/vitpose)
- Sapiens (v1): repository LICENSE is CC BY-NC 4.0 (non-commercial); models natively trained at 1024x1024, patch 16; sizes 0.3B/0.6B/1B/2B; a "lite" inference install is described as about 4x faster — [Sapiens LICENSE](https://raw.githubusercontent.com/facebookresearch/sapiens/main/LICENSE); [Sapiens GitHub README](https://github.com/facebookresearch/sapiens)
- Sapiens2 (initial release 2026-04-24): sizes 0.4B / 0.8B / 1B / 5B; pose model outputs 308 whole-body keypoints; trained at 1024x768 (a 1B "4K" variant at 4096x3072); pose inference uses a DETR ResNet-101 DC5 person detector; requires Python ≥ 3.12 and PyTorch ≥ 2.7; no transformers integration mentioned — [Sapiens2 GitHub README](https://github.com/facebookresearch/sapiens2)
- Sapiens2 License: custom Meta agreement granting a "non-exclusive, worldwide, non-transferable and royalty-free limited license"; covers trained model weights; does not address commercial use explicitly; prohibited uses include surveillance, biometric processing, re-identifying individuals, deepfakes and military applications; Meta may audit use — [Sapiens2 LICENSE.md](https://raw.githubusercontent.com/facebookresearch/sapiens2/main/LICENSE.md)
- Ultralytics YOLO26 pose: 17 COCO keypoints only (no whole-body set documented); mAP pose 50-95 at 640 px: n 57.2, s 63.0, m 68.8, l 70.4, x 71.6; T4 TensorRT latency 1.8 / 2.7 / 5.0 / 6.5 / 12.2 ms; params 2.9M to 57.6M; a YOLO27 preview is linked — [Ultralytics pose docs](https://docs.ultralytics.com/tasks/pose/)
- Ultralytics licensing: AGPL-3.0 requires you to "Open-source your entire project under AGPL-3.0"; "Internal business tools or private company applications" and SaaS/API use are listed as requiring an Enterprise License; "All Ultralytics YOLO trained models fall under the AGPL-3.0 License by default" — [Ultralytics licence page](https://www.ultralytics.com/license)
- MediaPipe: repository licence Apache-2.0 — [MediaPipe GitHub](https://github.com/google-ai-edge/mediapipe)
- OpenPose: licence is for "academic or non-profit organization noncommercial research use only"; community sources cite a commercial licence fee of $25,000 per year — [OpenPose LICENSE mirror](https://huggingface.co/OzzyGT/openpose/blob/main/LICENSE); [Intel community thread](https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/Is-royalty-for-human-pose-estimation-0001-commercial-use-free/td-p/1144783)
- Wan 2.2 Animate's preprocessing (as wrapped by Kijai's Apache-2.0 ComfyUI nodes) uses a YOLOv10m ONNX person detector plus ViTPose whole-body ONNX (Large from `JunkyByte/easy_ViTPose`, or Huge from `Kijai/vitpose_comfy`) — i.e. the newest open character-animation model moved from DWPose to ViTPose-H whole-body — [kijai/ComfyUI-WanAnimatePreprocess](https://github.com/kijai/ComfyUI-WanAnimatePreprocess)

### Inferences
- All listed whole-body models are top-down (detector → per-person crop), so multi-person support comes from the detector and cost scales with the number of people; RTMO and YOLO-pose are one-stage but body-only.
- These models are tiny next to depth models (DWPose-l-384 is 10.1 GFLOPs per person), so GPU speed and VRAM are not constraints on a 4090; ONNX Runtime CUDA on Windows is the main dependency.
- Licence ranking for a commercial studio tool: rtmlib + RTMW/DWPose (Apache-2.0 code and mmpose-released weights, with the caveat that the training datasets' terms are not asserted) and transformers ViTPose (checkpoint licence to be read per model card) are the low-risk choices; MediaPipe is also Apache-2.0 but was not evaluated for accuracy here; Ultralytics needs an Enterprise licence for an in-house tool; OpenPose and Sapiens v1 are out; Sapiens2 needs legal review, in particular whether pose extraction of actors counts as "biometric processing".
- Whole-body AP numbers come from each project's own tables at different input sizes and detectors, so RTMW-x (70.2) versus DWPose-l (66.5) versus ViTPose++-l (60.6) is indicative rather than a controlled comparison.

### Gaps
- Speed (FPS) for RTMW/DWPose/ViTPose-H whole-body on an RTX 4090 through ONNX Runtime: not found.
- ViTPose/ViTPose++ checkpoint licence on the Hub (per model card): not checked. Whether a ViTPose-H whole-body checkpoint (as used by Wan Animate) is available as a native transformers checkpoint: not verified.
- MediaPipe Holistic/Pose Landmarker accuracy, multi-person limits and Windows Python support: not verified (the repo page read does not list them).
- Sapiens2 pose accuracy, speed and VRAM: not stated on the page read. Whether Sapiens (v1) ever had a different (custom) licence than the CC BY-NC text now in the repo: not checked.
- Hand/face keypoint quality comparisons between DWPose, RTMW, ViTPose-H whole-body and Sapiens on video: no source found.

---

## 7. Temporal jitter in pose estimation, smoothing and identity tracking

### Takeaway
Per-frame 2D keypoints jitter and drop out, and the standard remedies are well documented: One-Euro filtering (speed-adaptive low-pass), Savitzky–Golay polynomial smoothing, or the learned SmoothNet refiner, combined with confidence gating and a tracker for stable identities. Published comparisons favour SmoothNet on accuracy, and warn that classical filters can slightly degrade accuracy while reducing jitter.

### Cited Findings
- SmoothNet (ECCV 2022) is a plug-and-play temporal refinement network for 2D/3D pose sequences; the paper reports an 86.88% decrease in acceleration error and an 8.82% MPJPE improvement relative to raw estimator outputs — [SmoothNet, arXiv 2112.13715](https://arxiv.org/abs/2112.13715)
- In the SmoothNet comparison, the Savitzky–Golay filter fits a local polynomial by least squares within a window (window 257, polyorder 2 for the acceleration-matched setting; window 31 for the MPJPE-matched setting), and the One-Euro filter used a minimum cutoff of 1e-4 with speed coefficient 0.7 — [SmoothNet paper](https://arxiv.org/pdf/2112.13715)
- The One-Euro filter uses an adaptive cut-off frequency so that it keeps lag small in rapid movements while reducing noise and jitter in slow movements — [Beyond Weak Perspective for Monocular 3D Human Pose Estimation, arXiv 2009.06549](https://arxiv.org/pdf/2009.06549)
- MoManifold reports that on 3DPW "the estimation performance often degrades when applying traditional filters (such as one euro)", with One-Euro raising MPJPE for several backbones while SmoothNet lowered it in most cases — [MoManifold, arXiv 2409.00736](https://arxiv.org/pdf/2409.00736)
- A 2026 SCA poster proposes combining the One-Euro filter with confidence-weighted filtering, noting that confidence-weighted filtering alone is less accurate than One-Euro during fast motion — [Bournemouth University ePrints 42215](https://eprints.bournemouth.ac.uk/42215/)
- rtmlib provides a `PoseTracker` wrapper for video with a `det_frequency` parameter (detector run every N frames, default example 10) — [rtmlib GitHub README](https://github.com/Tau-J/rtmlib)
- Ultralytics pose models work with ByteTrack through Track mode according to a team member's reply (not the official pose docs page) — [Ultralytics pose docs (comments)](https://docs.ultralytics.com/tasks/pose/)
- Community Wan-Animate preprocessing packs include a 3D refinement node ("Wan Pose 3D Refine — NLF") alongside the 2D detector node, i.e. lifting to 3D is used in practice to regularise 2D skeletons — [comfy.icu WanPose3DRefineNLFV2](https://comfy.icu/node/WanPose3DRefineNLFV2)
- The transformers ViTPose post-processing exposes a per-keypoint score and a `threshold`, and the doc's drawing helper only draws a link when both endpoint scores exceed `keypoint_score_threshold` (0.3 in the example) — the basis for confidence gating — [transformers ViTPose doc](https://huggingface.co/docs/transformers/en/model_doc/vitpose)

### Inferences
- A workable offline recipe for 5–15 s clips (offline means zero-lag, bidirectional filters are allowed, unlike the live-capture setting One-Euro was designed for): (1) detect people and track boxes (IoU/ByteTrack-style) to get a stable ID per person, and assign the skeleton colour by ID, never by per-frame detection order; (2) per track and per keypoint, mark samples below a confidence threshold as missing; (3) fill short gaps (a few frames) by linear or spline interpolation and leave long gaps undrawn rather than hallucinated; (4) smooth x/y with a short Savitzky–Golay window (order 2–3, about 5–9 frames at 24 fps) or a forward–backward One-Euro pass, with weaker smoothing for hands than for the torso, since fingers move fast and over-smoothing kills gesture; (5) run the detector every frame (not every 10th) for clips with fast action, because box jitter propagates into top-down keypoints. These parameter ranges are engineering suggestions, not cited values.
- The same ghosting-versus-jitter trade-off as in depth applies: a long smoothing window makes limbs lag and shortens fast swings; adaptive (speed-dependent) filters or short polynomial windows are the standard compromise.
- Hand and face keypoints flicker most because they are small in frame for full-body shots; running the pose model on a higher-resolution person crop from the 1080p/4K source (rather than a downscaled frame) is likely to help more than any filter.

### Gaps
- No source was found that evaluates smoothing specifically on DWPose/RTMW/ViTPose whole-body output for control-video use, or that gives recommended filter parameters at 24 fps.
- SmoothNet licence and whether pretrained 2D whole-body (133-keypoint) checkpoints exist: not checked.
- Which tracker rtmlib's `PoseTracker` implements (IoU-based versus ByteTrack) and whether it exposes persistent IDs: not verified.
- Identity switches during occlusion/crossing people in multi-person clips: no quantitative source read.

---

## 8. Rendering conventions for skeleton control videos, and 3D-aware alternatives (SMPL / mesh renders)

### Takeaway
The de-facto 2D convention is the ControlNet/DWPose drawing: OpenPose-18 body with a fixed 18-colour rainbow palette, limbs as filled ellipses of half-width 4 px dimmed by 0.6, joints as radius-4 dots, hands as 20 HSV-rainbow edges with red dots, face as white radius-3 dots, on black. For 3D-aware control, most video mesh-recovery code (GVHMR and the SMPL family) is non-commercial twice over — research-only code licences plus the SMPL body-model licence — while Meta's SAM 3D Body + MHR (Apache-2.0 body model, SAM License for model/code) is the one stack found without an explicit non-commercial clause, but it is single-image, not temporal.

### Cited Findings
- DWPose/ControlNet `draw_bodypose`: `stickwidth = 4`; 19 limb pairs on OpenPose-18 keypoints ([2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], [1,16], [16,18], [3,17], [6,18]) of which only the first 17 are drawn; 18 RGB colours running from [255,0,0] around the hue wheel to [255,0,85]; each limb is an ellipse polygon (`cv2.ellipse2Poly` with semi-axes length/2 and 4) filled with `cv2.fillConvexPoly`; the canvas is then multiplied by 0.6; joints are filled circles of radius 4 in the same palette — [DWPose util.py](https://raw.githubusercontent.com/IDEA-Research/DWPose/onnx/ControlNet-v1-1-nightly/annotator/dwpose/util.py)
- `draw_handpose`: 20 edges forming five finger chains from the wrist; edge i coloured `hsv_to_rgb([i/20, 1, 1]) * 255`; line thickness 2; keypoints as filled radius-4 circles in (0,0,255). `draw_facepose`: filled white circles of radius 3. Hand/face points are drawn only when coordinates exceed `eps = 0.01`; the functions take normalised [0,1] coordinates and do not read confidence — gating happens upstream by setting coordinates to -1 — [DWPose util.py](https://raw.githubusercontent.com/IDEA-Research/DWPose/onnx/ControlNet-v1-1-nightly/annotator/dwpose/util.py)
- rtmlib can emit and draw this format directly (`to_openpose=True` on the model and on `draw_skeleton`) — [rtmlib GitHub README](https://github.com/Tau-J/rtmlib)
- DWPose was adopted by Animate Anyone-style pipelines (integration noted 2023-12-03) and by sd-webui-controlnet (v1.1237+, 2023-08-09) — [DWPose GitHub README](https://github.com/IDEA-Research/DWPose)
- Wan-Animate drives the model with a skeleton image sequence plus a face crop; in animation mode the skeletons extracted from the reference video are retargeted because of differing bone proportions, and the authors note 2D skeleton bone-length estimates are posture-sensitive — [Wan-Animate paper, arXiv 2509.14055](https://arxiv.org/pdf/2509.14055)
- Community documentation for the Wan Animate preprocess nodes describes the detector output as OpenPose-18-style keypoints produced by YOLO + ViTPose (third-party description, not the official spec) — [comfy.icu PoseAndFaceDetectionV2](https://comfy.icu/node/PoseAndFaceDetectionV2)
- GVHMR (SIGGRAPH Asia 2024): custom ZJU 3D Vision licence — "for educational, research and non-profit purposes only"; "Any modification based on this work must be open-source and prohibited for commercial use"; commercial contact xwzhou@zju.edu.cn — [GVHMR LICENSE](https://raw.githubusercontent.com/zju3dv/GVHMR/main/LICENSE)
- GVHMR practicalities: `-s` flag skips visual odometry for a static camera; since 2025-03-08 the default camera estimation is a built-in "SimpleVO" instead of DPVO, and an `f_mm` option sets the full-frame focal length; builds on WHAM, 4D-Humans and ViTPose-Pytorch — [GVHMR GitHub README](https://github.com/zju3dv/GVHMR)
- SMPL model licence: free "for the sole purpose of performing non-commercial scientific research", non-commercial education or non-commercial artistic projects; "Any other use, in particular any use for commercial purposes, is prohibited", including "production of other artefacts for commercial purposes" and use "to train methods/algorithms/neural networks/etc. for commercial use"; commercial licensing through Meshcapade — [SMPL model licence](https://smpl.is.tue.mpg.de/modellicense.html)
- SMPL-family recovery methods output SMPL/SMPL-X parameters, and turning those into a mesh requires the SMPL model files (a startup's blog makes this point about WHAM and others; WHAM's README requires registering for SMPL downloads). PromptHMR estimates SMPL-X parameters per person — [dev.to blog post](https://dev.to/arkadiuss/a-3d-body-scan-for-nine-cents-without-smpl-4me1); [PromptHMR paper, arXiv 2504.06397](https://arxiv.org/html/2504.06397v2)
- SAM 3D Body (Meta; checkpoints released 2025-11-19; CVPR 2026): promptable single-image full-body mesh recovery with body, feet and hand pose on the Momentum Human Rig (MHR); backbones DINOv3-H+ (840M) and ViT-H (631M); 3DPW MPJPE 54.8; demo uses a ViTDet (or SAM3) detector and a MoGe2 FOV estimator; checkpoints are gated on Hugging Face; "The SAM 3D Body model checkpoints and code are licensed under SAM License"; no video/temporal mode described — [SAM 3D Body GitHub README](https://github.com/facebookresearch/sam-3d-body)
- SAM License (last updated 2025-11-19): grants a "non-exclusive, worldwide, non-transferable and royalty-free limited license"; covers trained model weights; does not mention commercial use either way; prohibits ITAR-covered, military/warfare, nuclear and espionage uses and reverse engineering; derivative works may be distributed under the same agreement — [SAM 3D Body LICENSE](https://raw.githubusercontent.com/facebookresearch/sam-3d-body/main/LICENSE)
- MHR (Momentum Human Rig): "licensed under the Apache Software License 2.0"; parametric body with 45 shape, 204 pose and 72 expression parameters, seven levels of detail; `pip install mhr` (labelled experimental), Python ≥ 3.11; includes an MHR ↔ SMPL/SMPL-X conversion tool — [MHR GitHub README](https://github.com/facebookresearch/MHR)

### Inferences
- For a hosted model such as Seedance, which was not trained on a documented skeleton format, matching the most common training-data convention (black background, DWPose/OpenPose colours, body + hands, optional face dots) is the best prior; whether it reads such a video as "motion reference" at all is an empirical question outside these notes.
- The fixed 4 px stick width was designed for about 512 px ControlNet inputs; at 1080p the lines become hair-thin and can vanish after the server's downscale. Scaling stick width, joint radius and hand line thickness with resolution (roughly height/128 for the stick half-width) preserves the look after downscaling. This is a derivation from the code's constants, not a documented rule.
- A neutral-shape mesh rendered as grey mannequin, depth or normals gives occlusion order and body volume that 2D sticks lack, and it is naturally stable if the 3D motion is smoothed; but the commercially clean route (SAM 3D Body + MHR) is per-frame, so it would need its own temporal smoothing of pose parameters and a tracker, and the video-native routes (GVHMR, WHAM, TRAM) are blocked for commercial use by code licence and/or by SMPL.
- SMPL's licence blocks commercial use of the body model itself (and "production of other artefacts for commercial purposes"), so rendering an SMPL mesh into a control video for commercial work requires a Meshcapade licence regardless of the recovery method's code licence.

### Gaps
- Licences of WHAM, TRAM, 4D-Humans/HMR2 and PromptHMR code/weights were not opened — unknown here (a secondary blog says WHAM's code "may be MIT or Apache 2.0"; unverified).
- VACE's and Wan Animate's exact skeleton drawing code (colours, widths, whether face/hand points are drawn) was not read; Animate Anyone/UniAnimate/MimicMotion drawing variants (e.g. confidence-modulated colour) were not verified.
- Whether the SAM License and Sapiens2 License are intended to allow commercial production use: the texts read do not say "commercial"; needs legal review. A secondary blog characterises SAM 3D Body as commercially permissive but not Apache-2.0.
- SAM 3D Body VRAM/speed and multi-person handling: not stated in the README. No 2026 video/temporal extension of SAM 3D Body was found.
- No evidence found on how Seedance 2.5 responds to skeleton versus mannequin versus depth reference videos.

---

## 9. Tools that estimate camera motion from the same clip

### Takeaway
ViPE (NVIDIA, pip-installable, actively updated through November 2026) and DA3 / DA3-Streaming are the most practical current tools for camera intrinsics and poses from casual, dynamic footage; VGGT has a commercial checkpoint; MegaSaM is accurate but tied to an old Linux-style research environment. All of them mix permissive code with third-party model weights that must be checked individually.

### Cited Findings
- ViPE: outputs camera intrinsics, camera motion and dense near-metric depth from raw video; `pip install nvidia-vipe` or build from source with conda + uv (CUDA e.g. `cu128`); CLI `vipe infer VIDEO.mp4`, long videos with `-p pose_only_long`; long-sequence mode keeps GPU/CPU memory roughly constant with a sliding keyframe window; MetricTrail pipeline requires building CUDA kernels with nvcc 12.8; supported operating systems are not stated — [ViPE GitHub README](https://github.com/nv-tlabs/vipe)
- ViPE licence caveats: Apache-2.0 for NVIDIA's source, but bundled/pulled third-party parts keep their own terms (UniK3D BY-NC-SA 4.0, DPVO MIT, LieTorch BSD-3-Clause) and MetricTrail weights use a separate NVIDIA License — [ViPE GitHub README](https://github.com/nv-tlabs/vipe)
- DA3 any-view models estimate camera pose jointly with depth (pose estimation and pose conditioning columns in the model zoo); DA3-BASE and DA3-SMALL are Apache 2.0 — [DA3 README](https://github.com/ByteDance-Seed/Depth-Anything-3)
- DA3-Streaming writes `camera_poses.txt` (extrinsics per frame) and `intrinsic.txt`; TUM RGB-D ATE RMSE (chunk 120, overlap 60, loop closure): DA3-Streaming 0.087 m, Pi-Long 0.094 m, VGGT-Long 0.110 m, uncalibrated DROID-SLAM 0.163 m, uncalibrated MASt3R-SLAM 0.060 m; accuracy degrades with small chunks (0.227 m at chunk 30) — [DA3-Streaming README](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md)
- VGGT: commercial use allowed only with the gated VGGT-1B-Commercial checkpoint (Co3D AUC@30 90.37 versus 89.98 for the original); a COLMAP export script was added 2025-06-02; 2026-05-15 memory fix allows about 2–3x more frames in the same GPU budget — [VGGT GitHub README](https://github.com/facebookresearch/vggt)
- π3: outputs 4x4 camera-to-world matrices (OpenCV convention); code BSD-3 but weights CC BY-NC 4.0 — [Pi3 GitHub README](https://github.com/yyfz/Pi3)
- MegaSaM: camera tracking + consistent video depth for dynamic scenes; environment Python 3.10 / CUDA 11.8 / PyTorch 2.0.1 with a compiled tracking extension, UniDepth (xformers) and Depth Anything priors, RAFT weights, shell-script entry points; no Windows notes — [MegaSaM GitHub README](https://github.com/mega-sam/mega-sam)
- GVHMR's own camera handling: SimpleVO by default since 2025-03-08, DPVO optional, `-s` for static camera — a human-centric alternative when only gravity-aligned human motion is needed — [GVHMR GitHub README](https://github.com/zju3dv/GVHMR)

### Inferences
- For 5–15 s clips (120–360 frames) all feed-forward options can take the clip in one or a few chunks; DA3-Streaming's VRAM table suggests chunk 60 at about 504x378 (21.2 GB) is the largest setting that fits 24 GB, and chunk 30 costs accuracy — so on a 4090 subsampling frames (e.g. every 2nd–4th frame, then interpolating poses) is likely preferable to small chunks.
- A camera-only pass has two uses in this pipeline: (a) rendering a static-background or camera-only control video, and (b) enforcing depth consistency (depth aligned across frames through the recovered geometry). Both require masking moving people; ViPE and MegaSaM are designed for dynamic scenes, while VGGT/DA3/π3 are feed-forward and less explicit about dynamic-object handling.
- ViPE is the best "practicality" candidate (pip package, releases every few months in 2026), but its default configurations pull non-commercial priors (UniK3D), so the pipeline config must be chosen with licences in mind (e.g. the DA3 pipeline with an Apache DA3 checkpoint, MoGe v2 terms to be checked).

### Gaps
- DROID-SLAM and COLMAP/GLOMAP were not researched directly in this session (licence, Windows builds, behaviour on dynamic scenes); only DROID-SLAM's uncalibrated TUM number quoted by the DA3 README is included.
- ViPE: Windows support, absolute FPS and VRAM, and MoGe v2's licence were not found. Which DA3 checkpoint ViPE's `dav3` pipeline uses (Apache or NC) was not verified.
- MegaSaM runtime/VRAM on a 24 GB card and any Windows/WSL2 reports: not found.
- No comparison of camera-pose accuracy on short handheld live-action clips with multiple people was found for these tools.
