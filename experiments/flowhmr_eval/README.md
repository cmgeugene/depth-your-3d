# 클레이 마네킹 안정화 실험 (FlowHMR / Fast SAM 3D Body / ExMR)

목표: depth 영상의 인물 윤곽 위에 클레이 모델을 화면 공간에서 정확히, 떨림 없이 겹치기.
지금 클레이(`server/clay_worker.py`)는 SAM 3D Body를 프레임마다 돌리고 정점에 Gaussian σ2를 거는 방식이다.

## 결과 (clip_004_10s, 300프레임 30fps, 움직이는 트래킹 샷, 허리 위)

정렬 오차 = COCO-17 투영과 ViTPose(conf>0.5)의 거리 중앙값, 떨림 = 상체 관절의 2D 가속도 평균(px/frame²).

| 방식 | 정렬 오차 | 떨림 |
|---|---:|---:|
| SAM 보정 없음 | 17.3 px | 14.0 |
| **SAM + σ2 (지금 클레이)** | **20.2 px** | **2.90** |
| SAM + σ4 | 25.6 px | 1.51 |
| SAM 자세 σ4 + 위치 재계산 | 20.6 px | 2.62 |
| Fast SAM + σ2 | 19.6 px | 2.93 |
| ExMR 카메라 기준, 보정 없음 | 26.8 px | 6.15 |
| ExMR 카메라 기준 + σ2 | 27.8 px | 2.52 |
| FlowHMR 고정 카메라 + 추종 피팅 | 45 px | 2.5 |
| FlowHMR + VGGT 카메라 + 위치 보정 | 42 px | 2.4 |

- 이 클립에서는 셋 다 "SAM + 시간 축 보정 조정"을 넘지 못했다. 수치 원본은 `out/*_metrics.json`, 비교 영상은 `out/*.mp4`.
- Fast SAM은 같은 가중치라 결과가 거의 같다(정점 차 5.8mm). eager 모드로 1.22배 빠르다(0.87 대 1.06 s/frame).
- SAM의 자세 오차와 위치 오차는 서로 상쇄한다. 한쪽만 보정하면 더 나빠지므로 같이 보정하거나, 자세를 보정한 뒤 위치를 다시 풀어야 한다.
- 실제 문제(고정 카메라에서 깊이 방향으로 튀며 크기가 바뀜, 미끄러짐)는 이 클립으로 재현되지 않았다.

## 다음 할 일

문제가 보이는 고정 카메라 클립(원본 + 지금 클레이 결과)으로 아래를 비교한다.
- 클레이 개선안: 중앙값 체형의 크기를 보존하며 재배치(t를 L_med/L_k로 스케일), 바닥면에 발 고정, 깊이(z)만 더 강하게 보정, 자세·위치 함께 보정
- ExMR 고정 카메라 모드: `tools/demo.py --video ... -s --snap_to_ground`

## 외부 설치 (이 저장소에 없음, `D:\Projects\` 아래에 따로 클론)

| 저장소 | 사용한 커밋 | 환경 |
|---|---|---|
| https://github.com/flowhmr/flowhmr | 03af0bf | `.venv` py3.10, torch 2.5.1+cu118, transformers 4.57.6 |
| https://github.com/yangtiming/Fast-SAM-3D-Body | d72aa36 | FlowHMR의 venv로 실행 |
| https://github.com/twehrbein/ExMR | eac72e4 | `.venv` py3.12, torch 2.8.0+cu128 |

- 가중치: SAM 3D Body는 Hugging Face `facebook/sam-3d-body-dinov3` 승인이 필요하다. FlowHMR은 yolox_l, vggt_omega_1b_512.pt, flowhmr_latest, SMPL-H `model.npz`. ExMR은 `download_models.py`(약 12GB).
- 라이선스: FlowHMR 가중치와 ExMR은 비상업용이다. 실험용으로만 쓴다.

### Windows에서 막히는 곳
- ExMR `download_models.py`의 YOLO 다운로드가 SSL 인증서 오류로 실패하면 `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt`, `yolo11x.pt`를 직접 받는다(SHA256은 스크립트에 고정된 값과 같았다).
- ExMR의 손 모델(HaMeR)이 `PYOPENGL_PLATFORM=egl`을 강제한다. 실행할 때 `PYOPENGL_PLATFORM=nt`를 준다.
- ExMR `exmr/preprocessing/visual_odometry.py:127`에서 `ctypes.util.find_library("c")`가 Windows에서 None을 돌려준다. 한 줄 고친다:
  ```python
  ctypes.CDLL(ctypes.util.find_library("c") or "msvcrt").srand(1)
  ```
- 8GB GPU에서는 VGGT(4.6GB)와 SAM을 한 프로세스에 같이 올리지 말고, 카메라 추정을 먼저 별도 프로세스로 돌린다(`runtime_vggt_camera.extract_vggt_camera`, interval 30).

## 실행 순서

```powershell
# 1) FlowHMR (D:\Projects\FlowHMR, 클립은 inputs\에)
.venv\Scripts\python.exe demo_generation.py inputs\clip_004_10s.mp4 --static_camera --sam_chunk_size 8
#    고정 카메라 결과를 VGGT 실행 전에 output\<stem>\flowhmr_latest_static.npz로 복사해 둔다
.venv\Scripts\python.exe demo_generation.py inputs\clip_004_10s.mp4 --vggt_interval 30 --sam_chunk_size 8

# 2) 프레임별 SAM 3D Body + ViTPose 기준값 (이 폴더에서)
D:\Projects\FlowHMR\.venv\Scripts\python.exe per_frame_refs.py D:\Projects\FlowHMR clip_004_10s
D:\Projects\FlowHMR\.venv\Scripts\python.exe per_frame_refs.py D:\Projects\FlowHMR clip_004_10s --sam-repo D:\Projects\Fast-SAM-3D-Body --out refs_fast.npz --no-vitpose

# 3) 비교
D:\Projects\FlowHMR\.venv\Scripts\python.exe fit_and_render.py D:\Projects\FlowHMR clip_004_10s
D:\Projects\FlowHMR\.venv\Scripts\python.exe compare_fast.py D:\Projects\FlowHMR clip_004_10s

# 4) ExMR (D:\Projects\ExMR, 입력은 FlowHMR이 만든 output\<stem>\<stem>_30fps.mp4를 inputs\clip_004_10s.mp4로 복사)
$env:PYOPENGL_PLATFORM = 'nt'; .venv\Scripts\python.exe tools\demo.py --video inputs\clip_004_10s.mp4 --no_render
.venv\Scripts\python.exe D:\Projects\depth-your-3d\experiments\flowhmr_eval\exmr_to_npz.py outputs\clip_004_10s D:\Projects\FlowHMR\output\clip_004_10s\exmr_global.npz
D:\Projects\FlowHMR\.venv\Scripts\python.exe D:\Projects\depth-your-3d\experiments\flowhmr_eval\compare_exmr.py D:\Projects\FlowHMR clip_004_10s D:\Projects\FlowHMR\output\clip_004_10s\exmr_global.npz
```

처리 시간(8GB GPU): SAM 3D Body 약 1 s/frame, ExMR 300프레임 약 10분.
