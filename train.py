import torch
from torch.utils.data import DataLoader, Subset

from config import *
from dataloader import *
from model import Where2Vec
from loss import *
from utils import *


def run():
    set_seed()


    dataset = Where2VecDataset(CSV_PATH, CHIP_ROOT)
    splits = create_or_load_splits(dataset)

    train_ds = Subset(dataset, splits["train"])
    val_ds   = Subset(dataset, splits["val"])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )


    model = Where2Vec().to(DEVICE)

    if USE_PRETRAINED and PRETRAINED_PATH is not None:
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
        print(f"Loaded pretrained weights from {PRETRAINED_PATH}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val = float("inf")
    train_losses, val_losses = [], []


    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            zi = model.encode_image(batch["image"])
            zt_i = model.encode_text(batch["text_instance"])
            zt_s = model.encode_text(batch["text_semantic"])

            type_mask = build_type_mask(batch["entity_type"], zi.device)

            loss = (
                type_aware_clip_loss(zi, zt_i, type_mask) +
                LAMBDA_SEM * type_aware_clip_loss(zi, zt_s, type_mask)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)


        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                zi = model.encode_image(batch["image"])
                zt = model.encode_text(batch["text_instance"])
                type_mask = build_type_mask(batch["entity_type"], zi.device)
                val_loss += type_aware_clip_loss(zi, zt, type_mask).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:04d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if epoch % SAVE_EVERY == 0:
            best_val = save_checkpoint(model, epoch, val_loss, best_val)
            plot_losses(train_losses, val_losses)

    print("Training completed.")


if __name__ == "__main__":
    run()
