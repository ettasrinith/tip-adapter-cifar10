# =============================================================================
# config.py -- Central configuration for the Tip-Adapter / CIFAR-10 study
#
# All hyperparameter search spaces live here so that the experiment is fully
# reproducible. Every search is performed on the VALIDATION split only; the
# official test set is touched exactly once per method, after selection.
# =============================================================================
import os

import torch

# --- Reproducibility ---------------------------------------------------------
SEEDS = [42, 123, 2026]            # seeds for support/validation sampling + init
SHOTS = [1, 2, 4, 8, 16]           # support images per class
NUM_CLASSES = 10
VAL_PER_CLASS = 100                # validation images per class (drawn from TRAIN only)

# --- CLIP --------------------------------------------------------------------
CLIP_ARCH = "ViT-B-32"             # open_clip name for OpenAI CLIP ViT-B/32
CLIP_PRETRAINED = "openai"         # original OpenAI pretrained weights
CLIP_HF_FALLBACK = "openai/clip-vit-base-patch32"   # used only if open_clip fails
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)     # official CLIP normalization
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
TEMPLATES = [
    "a photo of a {}.",
    "a picture of a {}.",
    "an image of a {}.",
]

# --- Performance (tuned for a Kaggle Tesla T4, 16 GB VRAM) --------------------
BATCH_SIZE = 256
NUM_WORKERS = 2
USE_AMP = True                     # fp16 autocast for frozen feature extraction
torch.backends.cudnn.benchmark = True

# --- Hyperparameter search spaces (tuned on VALIDATION only) ------------------
# Tip-Adapter training-free: alpha (cache weight) x beta (affinity sharpness)
ALPHA_GRID = [round(0.25 * i, 2) for i in range(25)]        # 0.00 .. 6.00
BETA_GRID = [2.0, 3.0, 4.0, 5.0, 5.86, 7.0, 8.0]            # 5.86 = paper default
# Linear probe: lr x weight decay, selected on validation
LP_LR_GRID = [1e-3, 3e-3, 1e-2, 3e-2]
LP_WD_GRID = [1e-4, 1e-2]
LP_EPOCHS = 300
LP_EVAL_EVERY = 25
# Tip-Adapter-F: lr x weight decay, early stopping on validation
TIPF_LR_GRID = [1e-3, 5e-3, 1e-2]
TIPF_WD_GRID = [1e-4, 1e-3]
TIPF_MAX_EPOCHS = 20
TIPF_PATIENCE = 5

# --- Paths --------------------------------------------------------------------
MOCK = os.environ.get("TIP_MOCK", "0") == "1"   # local CPU logic test with synthetic features
WORK_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
DATA_ROOT = "/tmp/cifar10" if os.path.isdir("/kaggle/working") else os.path.join(WORK_DIR, "data")
CACHE_DIR = os.path.join(WORK_DIR, "feature_cache")
RESULTS_DIR = os.path.join(WORK_DIR, "results")
FIG_DIR = os.path.join(WORK_DIR, "figures")
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")


def ensure_dirs():
    for d in (DATA_ROOT, CACHE_DIR, RESULTS_DIR, FIG_DIR, CKPT_DIR):
        os.makedirs(d, exist_ok=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
