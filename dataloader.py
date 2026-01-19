import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from config import *

def preprocess_chip(chip):
    chip = np.clip(chip, 0, 10000).astype(np.float32) / 10000.0
    return torch.from_numpy(chip)

class Where2VecDataset(Dataset):
    def __init__(self, csv_path, chip_root):
        df = pd.read_csv(csv_path)

        def resolve_path(p):
            if os.path.isabs(str(p)) and os.path.exists(p):
                return p
            return os.path.join(chip_root, os.path.basename(str(p)))

        df["chip_path"] = df["chip_path"].apply(resolve_path)
        df = df[df["chip_path"].apply(os.path.exists)].reset_index(drop=True)

        self.df = df
        print("Loaded samples:", len(df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        chip = np.load(r["chip_path"])["chip"]

        return {
            "image": preprocess_chip(chip),
            "text_instance": str(r["text"]),
            "text_semantic": str(r["entity_type"]),
            "entity_type": str(r["entity_type"]),
            "entity_id": str(r["entity_id"]),
        }

def make_splits(dataset):
    groups = dataset.df["entity_id"].values

    gss = GroupShuffleSplit(
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=SEED
    )
    train_idx, temp_idx = next(gss.split(dataset.df, groups=groups))

    temp_groups = groups[temp_idx]
    gss2 = GroupShuffleSplit(
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=SEED
    )
    val_idx, test_idx = next(gss2.split(
        dataset.df.iloc[temp_idx], groups=temp_groups
    ))

    return (
        Subset(dataset, train_idx),
        Subset(dataset, temp_idx[val_idx]),
        Subset(dataset, temp_idx[test_idx]),
    )

def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "text_instance": [b["text_instance"] for b in batch],
        "text_semantic": [b["text_semantic"] for b in batch],
        "entity_type": [b["entity_type"] for b in batch],
    }
