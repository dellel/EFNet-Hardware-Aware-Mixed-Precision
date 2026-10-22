import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.TemporalEFNetV2Kria import TemporalEFNetV2Kria
from models.mixed_precision_model import MixedPrecisionTemporalEFNet
from data.echonet_dataset import EchoDataset, load_records, SEQ_LEN

CHECKPOINT = "/home/dellel/kria/echonet_int8/TemporalEFNetV2Kria_best.pth"


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    predictions, labels = [], []
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for images, ef in loader:
            images = images.to(device, non_blocking=True)
            target = ef.to(device, non_blocking=True).unsqueeze(1)

            output = model(images)
            total_loss += F.smooth_l1_loss(output, target).item()

            predictions.extend((output.cpu().numpy().ravel() * 100).tolist())
            labels.extend((target.cpu().numpy().ravel() * 100).tolist())
            batches += 1

            if max_batches and batches >= max_batches:
                break

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    return {
        "loss": total_loss / max(batches, 1),
        "mae": float(np.mean(np.abs(predictions - labels))),
        "rmse": float(np.sqrt(np.mean((predictions - labels) ** 2))),
        "n": len(labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Limite temporaire de batches par époque; 0 = complet")
    parser.add_argument("--lambda-precision", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    records = load_records()
    for split in ("TRAIN", "VAL"):
        print(f"{split}: {len(records[split])} videos")
        if not records[split]:
            raise RuntimeError(f"No videos found for split {split}")

    train_loader = DataLoader(
        EchoDataset(records["TRAIN"]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        EchoDataset(records["VAL"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    base = TemporalEFNetV2Kria(seq_len=SEQ_LEN)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    base.load_state_dict(checkpoint["model_state_dict"], strict=True)

    # Freeze all original parameters: only precision distributions are learned.
    for parameter in base.parameters():
        parameter.requires_grad_(False)

    model = MixedPrecisionTemporalEFNet(base).to(device)
    model.eval()

    optimizer = torch.optim.Adam(
        [p for p in model.quantizers.parameters() if p.requires_grad],
        lr=args.lr,
    )

    max_batches = args.max_batches or None
    best_val_mae = float("inf")
    output_dir = os.path.join("results", "mixed_precision_search")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.eval()
        total_task_loss = 0.0
        total_loss = 0.0
        batches = 0

        for images, ef in train_loader:
            images = images.to(device, non_blocking=True)
            target = ef.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(images)
            task_loss = F.smooth_l1_loss(prediction, target)
            precision_penalty = model.precision_penalty()
            loss = task_loss + args.lambda_precision * precision_penalty

            loss.backward()
            optimizer.step()

            total_task_loss += task_loss.item()
            total_loss += loss.item()
            batches += 1

            if batches % 10 == 0:
                print(
                    f"Epoch {epoch} | batch {batches} | "
                    f"task={task_loss.item():.5f} | "
                    f"bits={model.expected_precisions()}",
                    flush=True,
                )

            if max_batches and batches >= max_batches:
                break

        val = evaluate(model, val_loader, device, max_batches)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train task loss: {total_task_loss / max(batches, 1):.6f}")
        print(f"Train total loss: {total_loss / max(batches, 1):.6f}")
        print(f"Validation loss: {val['loss']:.6f}")
        print(f"Validation MAE: {val['mae']:.4f} points")
        print(f"Validation RMSE: {val['rmse']:.4f} points")
        print("Expected bits:", model.expected_precisions())
        print("Selected bits:", model.selected_precisions())

        result = {
            "epoch": epoch,
            "expected_precisions": model.expected_precisions(),
            "selected_precisions": model.selected_precisions(),
            "validation": val,
            "lambda_precision": args.lambda_precision,
            "learning_rate": args.lr,
        }

        torch.save(result, os.path.join(output_dir, f"epoch_{epoch}.pt"))

        if val["mae"] < best_val_mae:
            best_val_mae = val["mae"]
            torch.save(result, os.path.join(output_dir, "best.pt"))
            print("Best validation result saved.")

    print("\nExperiment finished.")
    print("Results directory:", os.path.abspath(output_dir))


if __name__ == "__main__":
    main()
