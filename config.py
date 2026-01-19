import torch
from pathlib import Path

# =====================
# General
# =====================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# Paths
# =====================
CSV_PATH  = r"C:\Users\PRASANTH\where2vec\data\multimodal_dataset.csv"
CHIP_ROOT = r"C:\Users\PRASANTH\where2vec\data\s2_chips"

OUTPUT_DIR = Path("outputs")
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
PLOT_DIR   = OUTPUT_DIR / "plots"
METRIC_DIR = OUTPUT_DIR / "metrics"

for d in [CKPT_DIR, PLOT_DIR, METRIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =====================
# Dataset split
# =====================
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1


# =====================
# Model
# =====================
IMG_EMB_DIM  = 256
TEXT_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LORA_R       = 8
LORA_ALPHA  = 16
LORA_DROPOUT = 0.1

# =====================
# Training
# =====================
BATCH_SIZE = 256
EPOCHS     = 500
LR         = 2e-4
WEIGHT_DECAY = 0.01
LAMBDA_SEM = 0.3
TEMP       = 0.07
ALPHA_TYPE = 0.5

# =====================
# Checkpointing
# =====================
SAVE_EVERY = 50

# =====================
# Pretrained
# =====================
USE_PRETRAINED = False
PRETRAINED_PATH = "outputs/checkpoints/best_epoch_1.pt"
