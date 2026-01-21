import torch
import numpy as np
import json
import umap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from config import *
from dataloader import *
from model import Where2Vec
from utils import *


def retrieval_metrics(sim):
    ranks = []
    for i in range(sim.shape[0]):
        order = np.argsort(-sim[i])
        rank = np.where(order == i)[0][0]
        ranks.append(rank)

    ranks = np.array(ranks)

    return {
        "R@1": float(np.mean(ranks < 1)),
        "R@5": float(np.mean(ranks < 5)),
        "R@10": float(np.mean(ranks < 10)),
        "MedianRank": int(np.median(ranks) + 1),
        "MeanRank": float(np.mean(ranks) + 1),
    }


@torch.no_grad()
def collect_embeddings(model, loader):
    model.eval()

    img_embs, txt_embs, labels = [], [], []

    for batch in loader:
        zi = model.encode_image(batch["image"])
        zt = model.encode_text(batch["text_instance"])

        img_embs.append(zi.cpu())
        txt_embs.append(zt.cpu())
        labels.extend(batch["entity_type"])

    return (
        torch.cat(img_embs).numpy(),
        torch.cat(txt_embs).numpy(),
        np.array(labels),
    )


def plot_umap(img_embs, txt_embs, labels):
    X = np.vstack([img_embs, txt_embs])
    modality = np.array(["image"] * len(img_embs) + ["text"] * len(txt_embs))
    labels = np.concatenate([labels, labels])

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=SEED
    )
    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(10, 8))
    for cls in np.unique(labels):
        for mod, marker in [("image", "o"), ("text", "*")]:
            idx = (labels == cls) & (modality == mod)
            plt.scatter(
                X_2d[idx, 0],
                X_2d[idx, 1],
                marker=marker,
                s=80 if mod == "text" else 40,
                alpha=0.9 if mod == "text" else 0.5,
                label=f"{cls}-{mod}"
            )

    plt.legend(ncol=2)
    plt.grid(True)
    plt.title("Test Set UMAP (Image/Text)")
    out = PLOT_DIR / "umap_test.png"
    plt.savefig(out)
    plt.close()

    print(f"UMAP saved → {out}")


def run():
    set_seed()

    dataset = Where2VecDataset(CSV_PATH, CHIP_ROOT)
    splits = create_or_load_splits(dataset)

    test_ds = Subset(dataset, splits["test"])

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    model = Where2Vec().to(DEVICE)

    ckpt = PRETRAINED_PATH
    if ckpt is None:
        ckpts = sorted(CKPT_DIR.glob("best_epoch_*.pt"))
        if not ckpts:
            raise RuntimeError("No checkpoints found for testing")
        ckpt = ckpts[-1]

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"Loaded checkpoint → {ckpt}")

    img_embs, txt_embs, labels = collect_embeddings(model, test_loader)

    sim = img_embs @ txt_embs.T

    metrics = {
        "Image->Text": retrieval_metrics(sim),
        "Text->Image": retrieval_metrics(sim.T),
        "NumSamples": len(img_embs),
        "Checkpoint": str(ckpt)
    }

    out = METRIC_DIR / "test_metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(k, ":", v)

    plot_umap(img_embs, txt_embs, labels)


if __name__ == "__main__":
    run()

