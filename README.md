# 휴먼에이 영상 처리 스튜디오

사내 공용 GPU 서버(RTX 4090 24GB 기준)에서 돌리는 영상 처리 웹 서비스. 접속하면 첫 화면(`/`)에서 도구를 고릅니다.

- **업스케일** (`/upscale`): 2D 애니메이션 영상을 Real-ESRGAN(animevideov3) 또는 Lanczos로 2배·4배 확대, 선택적으로 RIFE 프레임 보간(FPS 2배). 오디오·자막·메타데이터 보존. 결과를 바로 프레임 편집기로 보낼 수 있습니다.
  - **이미지도 됩니다**: png · jpg · webp · bmp · tif를 한 장 또는 여러 장(최대 200장) 한 번에 올리면 같은 설정으로 확대합니다. 출력은 PNG/JPG/WebP 중 선택하고, 여러 장이면 zip 하나로 받습니다. 투명도는 유지되며(JPG는 흰 배경), 투명도가 있는 이미지는 노이즈 제거를 건너뜁니다.
- **모션 트랜스퍼 영상 변환** (`/depth`, 예전 이름 Depth 변환): 실사 영상 업로드 → Depth Anything V2로 흑백 depth 영상(mp4) / 16bit PNG 시퀀스(zip) 생성 → 다운로드
  - **결과 종류**: Depth / 클레이 / Depth + 클레이 중에서 고릅니다. 컷은 세 가지 모두 같은 컷 감지 민감도 슬라이더로 정합니다.
  - **클레이**: Seedance v2v 레퍼런스용. 실사 인물을 SAM 3D Body로 3D 복원해 회색 클레이 마네킹으로 렌더합니다. 몸 동작·고개 각도·카메라는 넘기고, 머리는 평균 얼굴·무표정으로 바꿔 배우 얼굴형은 넘기지 않습니다. 체형 유지(머리만 중립)/모두 중립, 흔들림 보정 강도, 최대 인원을 고릅니다. 컷마다 인물 추적과 초점거리를 새로 잡으니, 빠른 푸시인처럼 컷이 아닌데 잡히면 민감도 값을 올리세요.
  - **Depth + 클레이**: 같은 작업에서 위 depth 설정으로 depth를 만들고 그 위에 클레이를 얹습니다(사람을 지우고 얹는 방식도 선택 가능). 방 배치·소품까지 넘어가서 지금까지 가장 결과가 좋은 조합입니다. depth mp4도 함께 받습니다. 설치는 아래 "클레이 설치".
  - **영상용 모델 (Video Depth Anything, 기본)**: 모델 목록의 "Video …" 항목은 앞뒤 프레임을 함께 보는 영상용 depth 모델입니다. 이미지용 V2는 프레임마다 따로 추정해서 머리카락·옷·피부 면이 꿈틀거리는데, 영상용은 이 떨림이 절반 정도로 줄어듭니다. Small(기본)은 Apache-2.0이라 상업 작업에 쓸 수 있고, Large는 비상업 전용입니다. 영상용 모델을 고르면 처리 방식을 고를 수 있습니다.
    - 오프라인 (기본): 32프레임 창을 겹쳐 가며 처리합니다. 가장 안정적입니다. 컷을 넘어도 금방 새 장면에 맞춰지므로 영상 전체를 한 번에 추론하고, 컷은 밝기 정규화에만 씁니다. 그래서 컷을 잘못 잡아도 depth가 끊기지 않습니다. 모든 그림을 메모리에 올리므로 `DS_VDA_ONEPASS_FRAMES`(기본 1500)장을 넘는 영상은 컷마다 따로 추론합니다. Small만 쓸 수 있습니다 (Large 오프라인은 VRAM이 24GB를 넘을 수 있어서 막아 둠).
    - 스트리밍: 한 프레임씩 처리해 VRAM을 적게 씁니다 (Small 약 1GB, Large 약 5GB). Large는 이 방식만 쓸 수 있습니다. 원 저장소에서는 실험 기능이라 정확도가 조금 낮습니다. 지난 프레임만 기억하므로 컷마다 새로 시작합니다 (이어 가면 새 컷의 처음 1~2초가 앞 컷의 깊이 범위를 따라감). 잘못 잡힌 컷이 있으면 그 자리에서 depth가 한 번 튀니, 컷 위치를 확인하고 변환하세요.
    - 코드는 `server/vda`에 옮겨 두었고 (출처·변경점은 `server/vda/README.md`), 가중치는 처음 쓸 때 Hugging Face에서 내려받습니다 (Small 약 120MB, Large 약 1.5GB).
    - 16:9보다 넓은 영상은 모델이 입력 해상도를 스스로 줄입니다 (21:9에서 짧은 변 약 406px). 그래서 V2보다 경계가 조금 부드럽습니다.
  - **얼굴 윤곽 단순화**: 미디엄·클로즈업에서 depth의 얼굴·목 실루엣(코끝, 입술, 턱 밑 꺾임)을 단순한 곡선으로 바꿉니다. 실사 턱선이 2D 얼굴에 그대로 찍히는 문제를 줄이기 위한 옵션입니다. 얼굴 부위 분할(SegFormer face-parsing, 처음 쓸 때 Hugging Face에서 내려받음)로 얼굴·목을 찾고, 배경과 맞닿은 경계만 고칩니다. 머리카락·몸·배경 depth는 그대로입니다. 얼굴이 화면에서 차지하는 크기에 따라 강도가 자동으로 올라가서 와이드샷은 바뀌지 않습니다.
- **얼룩 정리** (`/cleanup`): GPT Image로 만든 2D 이미지의 자글자글한 얼룩·픽셀 노이즈를 FLUX.1 [dev] img2img(denoise 0.15~0.2)로 약하게 다시 그려 정리합니다. 브러시로 정리할 곳만 칠하거나, 지킬 곳(얼굴 등)을 칠한 뒤 반전할 수 있습니다. 마스크 밖 픽셀은 원본 그대로 유지됩니다.
- **프레임 편집기 (Beta)** (`/editor`): 실사처럼 너무 부드럽게 움직이는 부분을 1s/1.5s/2s… 애니메이션 타이밍으로 다시 잡아 스텝 영상(mp4)으로 렌더. Depth는 쓰지 않습니다.
- 업스케일, 모션 트랜스퍼(클레이 포함), 얼룩 정리 작업은 GPU 하나를 공유하는 대기열로 순서대로 처리됩니다. 편집기 렌더는 GPU가 필요 없어 이 대기열을 거치지 않고 CPU에서 바로 처리됩니다. 각 페이지의 "서버 대기열"에서 모든 사용자의 작업을 볼 수 있습니다 (다운로드는 자기 작업만). 업로드/결과는 24시간 뒤 자동 삭제.
- mp4 결과물에 원본 오디오를 넣을지 선택할 수 있습니다.

## 실행

준비물: NVIDIA 드라이버, Python 3.10+, **ffmpeg 5 이상이 PATH에 있어야 함** (`ffmpeg -version`으로 확인).

```
run-lan.bat    # Windows, 내부망 공유 (다른 PC에서 http://192.168.x.x:8000)
run.bat        # Windows, 이 PC에서만 (http://localhost:8000)
./run.sh       # Linux, 내부망 공유
```

첫 실행 때 `.venv`를 만들고 torch 2.11(CUDA 12.8, 얼룩 정리의 Nunchaku가 지원하는 버전으로 고정)과 의존성을 설치합니다. 이후에는 `requirements.txt`가 바뀌었을 때만 다시 설치합니다. 모델은 처음 쓸 때 Hugging Face에서 내려받습니다 (기본 Video Small 약 120MB, 이미지 Large 약 1.3GB, Video Large 약 1.5GB).
서버가 뜨면 창에 접속 주소가 나옵니다 (`같은 네트워크의 다른 PC: http://192.168.x.x:8000`). 이 주소를 공유하면 됩니다.
다른 PC에서 안 열리면 방화벽에서 포트를 열어야 합니다. 필요한 명령도 같은 창에 나옵니다 (관리자 권한으로 한 번만 실행).

> Windows에서는 경로 길이 제한(260자) 때문에 `C:\depth-studio`처럼 짧은 경로에 두는 것을 권장합니다.

### 업스케일 엔진 설치 (서버에서 한 번)

```
setup-upscale-engines.bat    # Real-ESRGAN, RIFE를 공식 GitHub 릴리스에서 .\engines에 설치 (약 90MB)
```

엔진이 없어도 서버는 뜨고 Lanczos 업스케일은 쓸 수 있습니다. 빠진 엔진은 업스케일 페이지에 안내가 나옵니다. 엔진은 Windows용(ncnn-vulkan, Vulkan 지원 GPU 필요)이라 업스케일 AI 모드는 현재 Windows 서버에서만 동작합니다.
서버 폴더 경로에 한글 등 비 ASCII 문자가 있으면 엔진이 모델 파일을 못 여는 경우가 있으니, 그럴 땐 `DS_UPSCALE_ENGINES`로 영문 경로를 지정하세요.

업스케일 처리 코드(`server/upscale_core.py`)는 데스크톱 앱 anc_video-upscale-tool에서 옮겨 온 것입니다. 이미지 업스케일(`ImageUpscalePipeline`)은 그 저장소의 커밋 `36dcc04`(Add image upscaling and batch processing)에서 가져왔습니다. 그 저장소는 보관용으로만 남기므로, 업스케일 처리는 이 파일에서 고칩니다. 서버에 맞춘 부분(작업 폴더 캐시, 오디오 제거 등)은 `server/upscale.py`에 둡니다.

### 얼룩 정리 설치 (서버에서 한 번)

FLUX.1 [dev]는 Hugging Face에서 라이선스 동의가 필요한 모델입니다. 먼저:

1. https://huggingface.co/black-forest-labs/FLUX.1-dev 에서 로그인 후 라이선스 동의
2. https://huggingface.co/settings/tokens 에서 Read 토큰 생성
3. `.venv\Scripts\hf auth login` 실행 후 토큰 붙여넣기

```
setup-flux.bat    # Nunchaku(INT4 Flux 실행 엔진)를 torch 버전에 맞춰 설치하고 모델을 받습니다 (약 17GB)
```

- Flux 본체는 Nunchaku SVDQuant INT4(약 7GB VRAM)라 24GB 카드에서 SAM 같은 다른 프로세스와 함께 돌릴 수 있습니다. 16GB 미만 GPU에서는 자동으로 offload 모드(시스템 RAM에서 스트리밍, 느림)로 돌아갑니다.
- T5/CLIP 텍스트 인코더는 GPU에 상주하지 않습니다. 프롬프트마다 한 번만 인코딩해 `data/flux-prompts`에 캐시합니다 (GPU 여유가 11GB 미만이면 CPU에서 인코딩).
- 작업이 끝날 때마다 서버 창과 작업 폴더의 `cleanup.log`에 **VRAM 피크**(이 서버 / GPU 전체)가 찍힙니다. SAM과 함께 돌릴 때 여유를 확인하는 용도입니다.
- `DS_FLUX_IDLE_MIN`분 동안 정리 작업이 없으면 Flux를 VRAM에서 내립니다. 다음 작업은 다시 불러오느라 조금 늦게 시작합니다.
- **도움말**: 정리 페이지에 처음 들어오면 사용법·사례·한계를 담은 도움말이 뜹니다. 이미지·마스크 카드 우상단 아이콘이나 `?` 키로 다시 열 수 있습니다. 사례 이미지는 `static/help/`에 있습니다.
- **자동 선택** (SAM 3.1): 정리 페이지에 `hair`처럼 영어 단어를 넣고 "마스크에 추가/빼기"를 누르면, 이미지에서 그 단어에 해당하는 곳을 모두(인물마다) 찾아 마스크에 더하거나 뺍니다. 결과는 브러시·지우개로 다듬습니다.
  - SAM 3.1은 numpy 2 미만이 필요해서 스튜디오 `.venv`와 같이 설치할 수 없습니다. SAM 3.1이 설치된 다른 Python 환경을 `DS_SAM3_PYTHON`, 체크포인트(`sam3.1_multiplex.pt`)를 `DS_SAM3_CHECKPOINT`로 지정하면 그 환경에서 별도 프로세스로 돌립니다. 지정하지 않으면 자동 선택 버튼만 꺼집니다.
  - 첫 요청은 모델을 올리느라 1분 정도 걸리고, 그다음부터는 1초 안팎입니다. `DS_SAM3_IDLE_SEC`초 동안 요청이 없으면 프로세스를 끝내 VRAM을 돌려줍니다. 같은 이미지·같은 단어는 결과를 캐시합니다. 로그는 `data/sam3-worker.log`.
- **톤 보정**: Flux의 VAE를 한 번 거치기만 해도 검정이 들뜨고 대비·채도가 몇 % 떨어져 뿌옇게 보입니다. 정리한 결과를 원본과 비교해 채널별 밝기·대비(gain/offset)를 자동으로 맞춥니다. 값은 작업 로그에 남습니다.
- 기본값(20스텝, denoise 0.18, guidance 3.5, euler)은 ComfyUI 워크플로 `flux1_dev_i2i_pixel_noise_cleanup`과 같은 방식입니다. "스텝"은 실제로 도는 스텝 수입니다 (ComfyUI KSampler와 같음).
- **라이선스**: FLUX.1 [dev] 가중치는 비상업 라이선스입니다. 결과물은 상업적으로 써도 되지만, 서버를 상업 서비스로 운영하려면 BFL 라이선스가 필요합니다.

### 클레이 설치 (서버에서 한 번)

SAM 3D Body는 Python 3.11 전용 환경이 필요해서 스튜디오 `.venv`와 따로 둡니다. 다른 폴더에 설치하고 그 Python을 `DS_CLAY_PYTHON`, 저장소를 `DS_CLAY_REPO`로 지정하면 별도 프로세스로 돌립니다. 지정하지 않으면 모션 트랜스퍼 페이지의 클레이 선택지가 꺼집니다.

1. https://huggingface.co/facebook/sam-3d-body-dinov3 에서 접근 승인 (스튜디오와 같은 Hugging Face 토큰 사용)
2. 저장소 받기: `git clone https://github.com/facebookresearch/sam-3d-body.git`
3. 그 폴더에서 환경 만들기 (detectron2는 필요 없음: 사람 검출은 RT-DETR로 직접 함)

```
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-colorlog pyrootutils webdataset networkx==3.2.1 roma joblib jsonlines loguru optree fvcore huggingface_hub transformers moderngl "git+https://github.com/microsoft/MoGe.git"
```

- 첫 작업은 SAM 3D Body(약 2.7GB)·MoGe-2·RT-DETR·ViTPose·SegFormer를 올리느라 1~3분 더 걸립니다. 처음 쓸 때 Hugging Face와 torch hub에서 내려받습니다.
- VRAM: 작업 프로세스 최대 약 6GB. 작업이 끝나고 `DS_CLAY_IDLE_SEC`초 동안 다음 작업이 없으면 프로세스를 끝내 VRAM을 돌려줍니다. 취소하면 프로세스를 바로 끝냅니다.
- 처리 시간: 사람 1명당 프레임마다 SAM 3D Body를 한 번씩 돕니다. RTX 3070 Ti 8GB 기준 103프레임 1명 + depth 자동 생성에 약 11분(모델 로딩 2.5분 포함, 메모리가 빠듯해 느림). 4090은 훨씬 빠릅니다.
- 모든 프레임의 메시를 메모리에 올리므로 `DS_CLAY_MAX_FRAMES`프레임을 넘는 영상은 받지 않습니다.
- 얼굴 표정은 넘기지 않습니다. SAM 3D Body 공개 버전은 표정 출력이 꺼져 있고, 켜도 실제 표정을 따라가지 않습니다. 표정은 프롬프트로 지정합니다.
- 로그는 `data/clay-worker.log`.

## 설정 (환경 변수)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `DS_PORT` / `DS_HOST` | `8000` / `0.0.0.0` | 접속 포트 / 바인딩 주소 |
| `DS_DATA_DIR` | `./data` | 업로드·결과 저장 위치 |
| `DS_TTL_HOURS` | `24` | 파일 보관 시간 |
| `DS_MAX_UPLOAD_MB` | `4096` | 업로드 최대 크기 |
| `DS_BATCH` | `8` | 한 번에 추론할 프레임 수 (24GB 기준 8, VRAM 부족하면 낮춤) |
| `DS_MODELS` | `vda_small,vda_large,small,base,large` | 사용자에게 보여줄 모델 (`vda_*`는 영상용, 나머지는 이미지용 V2) |
| `DS_DEFAULT_MODEL` | `vda_small` | 기본 선택 모델 |
| `DS_VDA_ONEPASS_FRAMES` | `1500` | 영상용 모델 오프라인에서 영상 전체를 한 번에 추론할 최대 그림 수. 넘으면 컷마다 추론 |
| `DS_CPU_WORKERS` | `2` | 편집기 스텝 영상 렌더를 동시에 몇 개까지 돌릴지 (GPU 대기열과 별도) |
| `DS_MAX_EDITOR_FRAMES` | `3000` | 편집기를 쓸 수 있는 최대 프레임 수 (24fps 기준 약 2분) |
| `DS_CUT_THRESHOLD` / `DS_CUT_MIN_SHOT` | `12` / `6` | 컷 자동 감지 민감도 / 최소 샷 길이 |
| `DS_UPSCALE_ENGINES` | `./engines` | 업스케일 엔진(Real-ESRGAN, RIFE) 설치 위치 (영문 경로) |
| `DS_UPSCALE_MAX_SIDE` | `8192` | 업스케일 출력의 최대 긴 변(px). 넘으면 작업을 받지 않습니다 |
| `DS_FLUX_OFFLOAD` | `auto` | 얼룩 정리 Flux를 시스템 RAM에서 스트리밍할지. `auto`는 16GB 미만 GPU에서만 켬, `on`/`off`로 강제 |
| `DS_FLUX_IDLE_MIN` | `10` | 정리 작업이 없을 때 Flux를 VRAM에서 내리기까지의 시간(분). `0`이면 계속 올려 둠 |
| `DS_CLEANUP_MAX_MP` | `12` | 얼룩 정리 최대 이미지 크기(메가픽셀, 가로×세로 픽셀 수) |
| `DS_CLEANUP_TILE_MP` | `4.2` | 이보다 큰 이미지는 VAE를 1024px 타일로 나눠 처리함. 6.3MP 기준 VAE 피크 약 2.6GB, 통째로 처리한 결과와 차이 평균 0.6/255. VRAM 절약 모드에서는 항상 타일링 |
| `DS_SAM3_PYTHON` / `DS_SAM3_CHECKPOINT` | (없음) | 자동 선택용 SAM 3.1 Python 실행 파일 / `sam3.1_multiplex.pt` 경로 |
| `DS_SAM3_IDLE_SEC` | `120` | 자동 선택 요청이 없을 때 SAM 3.1 프로세스를 끝내기까지의 시간(초) |
| `DS_CLAY_PYTHON` / `DS_CLAY_REPO` | (없음) | 클레이용 SAM 3D Body 환경의 Python 실행 파일 / 저장소 폴더 |
| `DS_CLAY_CHECKPOINT` | `facebook/sam-3d-body-dinov3` | SAM 3D Body 체크포인트 (Hugging Face 저장소 이름) |
| `DS_CLAY_IDLE_SEC` | `300` | 클레이 작업이 없을 때 SAM 3D Body 프로세스를 끝내기까지의 시간(초) |
| `DS_CLAY_MAX_FRAMES` | `3000` | 클레이 작업으로 받을 최대 프레임 수 |
| `DS_FLUX_REPO` / `DS_FLUX_TRANSFORMER` | FLUX.1-dev / 자동 | 텍스트 인코더·VAE를 받을 저장소 / Nunchaku 가중치 경로 (`repo/file.safetensors`) |

## 라이선스 주의

- Depth-Anything-V2 **Small**: Apache-2.0 (상업 사용 가능)
- Depth-Anything-V2 **Base / Large**: **CC-BY-NC-4.0 (비상업)**
- Video-Depth-Anything **Small**: Apache-2.0 (상업 사용 가능)
- Video-Depth-Anything **Large**: **CC-BY-NC-4.0 (비상업)**

상업 프로젝트에 쓸 서버라면 `DS_MODELS=vda_small,small`로 제한하는 것을 고려하세요 (둘 다 Apache-2.0). UI에도 모델별 라이선스가 표시됩니다.

## Depth 처리 방식

1. 짧은 변을 518px(선택 가능)로 맞춰 FP16 추론 → 결과를 원본 해상도로 업샘플
2. **컷 단위 밝기 정규화**: 컷(샷)마다 depth 범위를 한 번만 정해서, 같은 샷 안에서 밝기가 출렁이지(깜빡이지) 않게 합니다. 가까운 곳 = 흰색.
3. 선택: 컷 안에서 천천히 적응 / 영상 전체 고정 / 프레임별, 클리핑·감마·반전·시간 스무딩

## 프레임 편집기 규칙

- **자동 타이밍**: 영상을 열면 **1.5s**(그림 하나를 2,1,2,1프레임씩)를 기본으로, 카메라가 크게 움직이거나 동작이 빠른 곳만 **1s**로 자동으로 나눕니다. 기본 스텝, 빠를 때 스텝, "빠른 곳 찾기" 감도는 바꿀 수 있습니다.
  - 움직임은 업로드 때 서버가 계산합니다 (OpenCV 광학 흐름, 192px 흑백). OpenCV가 없으면 이 단계만 건너뛰고 모두 기본 스텝으로 둡니다. 화면 전체가 같이 움직이는 양 = 카메라, 그걸 뺀 나머지 = 동작으로 나누고, 밝기를 프레임마다 맞춰서 조명 깜빡임은 움직임으로 치지 않습니다. 타임라인 '움직임' 줄에서 볼 수 있습니다.
- **샷 카드**: 샷마다 `자동` 또는 스텝 하나(1s/1.5s/2s/3s/멈춤)를 고릅니다. 직접 고른 샷은 자동 설정이나 컷을 바꿔도 그대로 둡니다.
- **세부 조정**: 타임라인을 드래그해 구간을 고르면 그 부분만 다른 스텝으로 바꿀 수 있고, 프레임마다 "여기서 새 그림 / 앞 그림 유지"를 지정할 수 있습니다. 0.5 단위 스텝(2.5s = 3,2,3,2)도 됩니다.
- **컷**마다 새 그림으로 시작합니다 (끌 수 있음). 빠진 컷은 `C` 키로 넣습니다.
- 오른쪽 패널 폭과 타임라인 높이는 경계선을 드래그해서 바꿉니다 (더블클릭: 기본 크기).
- 편집 내용은 브라우저에 자동 저장되고, JSON으로 저장/불러오기, X-시트 CSV로 내보낼 수 있습니다.
- 단축키는 편집기에서 `?` 키.

## 알려진 한계

- 얼룩 정리는 latent를 다시 그리는 방식이라 마스크 안쪽은 denoise가 낮아도 작은 글자, 손가락, 얇은 선이 바뀔 수 있습니다. 시드를 바꿔 여러 장(최대 4장) 뽑아 고르세요.
- 로그인/권한 기능이 없습니다. 사내망에서만 열어 두세요.
- 서버를 재시작하면 진행 중이던 작업 목록이 사라집니다 (파일은 TTL까지 남음).
- 가변 프레임레이트(VFR) 영상은 원본 평균 fps로 다시 인코딩됩니다.
- 프레임별 추론이라 Video Depth Anything 같은 영상 전용 모델보다 미세한 떨림이 남을 수 있습니다. 필요하면 고급 설정의 "시간 스무딩"을 올리세요.
- 업스케일은 작업 중 프레임을 무손실 PNG로 작업 폴더에 풀어 둡니다. 긴 영상·4배는 수십~수백 GB가 필요할 수 있으니 `DS_DATA_DIR` 드라이브의 여유 공간을 확인하세요 (작업이 끝나면 바로 지웁니다).
- 업스케일된 결과를 편집기로 보내면 편집기 프레임 한도(`DS_MAX_EDITOR_FRAMES`)가 그대로 적용됩니다. RIFE 보간을 켜면 프레임 수가 2배가 됩니다.
