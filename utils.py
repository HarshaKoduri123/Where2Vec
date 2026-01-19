import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

from config import *


def set_seed():
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def create_or_load_splits(dataset):
    split_file = OUTPUT_DIR / "splits.json"

    if split_file.exists():
        with open(split_file, "r") as f:
            splits = json.load(f)
        print(f"Loaded dataset splits from {split_file}")
        return splits

    print("Creating new dataset splits (group-aware)")

    idx = np.arange(len(dataset))
    groups = dataset.df["entity_id"].values

    gss1 = GroupShuffleSplit(
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=SEED
    )
    train_idx, temp_idx = next(gss1.split(idx, groups=groups))

    temp_groups = groups[temp_idx]
    gss2 = GroupShuffleSplit(
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=SEED
    )
    val_rel, test_rel = next(
        gss2.split(temp_idx, groups=temp_groups)
    )

    splits = {
        "train": train_idx.tolist(),
        "val": temp_idx[val_rel].tolist(),
        "test": temp_idx[test_rel].tolist()
    }

    assert not set(splits["train"]) & set(splits["val"])
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["val"]) & set(splits["test"])

    with open(split_file, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"Saved dataset splits → {split_file}")
    return splits


def save_checkpoint(model, epoch, val_loss, best_loss):
    if val_loss < best_loss:
        ckpt_path = CKPT_DIR / f"best_epoch_{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"✔ Saved checkpoint → {ckpt_path}")
        return val_loss
    return best_loss


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    out = PLOT_DIR / "loss_curve.png"
    plt.savefig(out)
    plt.close()

    print(f"Loss plot saved → {out}")
