"""Server settings. Every value can be overridden with a DS_* environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
DATA_DIR = Path(os.environ.get("DS_DATA_DIR", ROOT / "data")).resolve()
MEDIA_DIR = DATA_DIR / "media"
JOBS_DIR = DATA_DIR / "jobs"

HOST = os.environ.get("DS_HOST", "0.0.0.0")
PORT = int(os.environ.get("DS_PORT", "8000"))

# Uploaded media and finished jobs are deleted after this many hours.
TTL_HOURS = float(os.environ.get("DS_TTL_HOURS", "24"))
MAX_UPLOAD_MB = int(os.environ.get("DS_MAX_UPLOAD_MB", "4096"))

# Frames per forward pass. 8 is comfortable for Large at 518px on a 24GB card.
BATCH_SIZE = int(os.environ.get("DS_BATCH", "8"))
DEFAULT_MODEL = os.environ.get("DS_DEFAULT_MODEL", "large")

# Jobs without depth output (stepped video from the editor) skip the GPU queue and run
# on this many CPU workers alongside it.
CPU_WORKERS = int(os.environ.get("DS_CPU_WORKERS", "2"))

# The timing editor preloads one JPEG per frame, so very long clips are refused.
MAX_EDITOR_FRAMES = int(os.environ.get("DS_MAX_EDITOR_FRAMES", "3000"))
PREVIEW_WIDTH = int(os.environ.get("DS_PREVIEW_WIDTH", "480"))
THUMB_WIDTH = 120

# Shot-cut detection defaults (scdet score, 0-100).
CUT_THRESHOLD = float(os.environ.get("DS_CUT_THRESHOLD", "12"))
CUT_MIN_SHOT = int(os.environ.get("DS_CUT_MIN_SHOT", "6"))

# Upscale engines (Real-ESRGAN, RIFE) installed by setup-upscale-engines.bat. Keep the path ASCII:
# older ncnn Windows builds cannot open model files under non-ASCII paths.
UPSCALE_ENGINES_DIR = Path(os.environ.get("DS_UPSCALE_ENGINES", ROOT / "engines")).resolve()
# Longest side of an upscaled output. The PNG frame cache grows with the square of this.
UPSCALE_MAX_SIDE = int(os.environ.get("DS_UPSCALE_MAX_SIDE", "8192"))

FFMPEG = os.environ.get("DS_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("DS_FFPROBE", "ffprobe")

MODELS = {
    "small": {
        "repo": "depth-anything/Depth-Anything-V2-Small-hf",
        "label": "Small (25M)",
        "license": "Apache-2.0",
    },
    "base": {
        "repo": "depth-anything/Depth-Anything-V2-Base-hf",
        "label": "Base (98M)",
        "license": "CC-BY-NC-4.0",
    },
    "large": {
        "repo": "depth-anything/Depth-Anything-V2-Large-hf",
        "label": "Large (335M)",
        "license": "CC-BY-NC-4.0",
    },
}
ENABLED_MODELS = [
    m.strip() for m in os.environ.get("DS_MODELS", "small,base,large").split(",") if m.strip() in MODELS
]
INPUT_SIZES = [518, 644, 756, 1036]
