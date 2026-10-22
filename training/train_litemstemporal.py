
import os
import sys
import csv
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.echonet_dataset import load_records, EchoDataset
from models.LiteMSTemporalEFNet import LiteMSTemporalEFNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_metrics(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)

    errors = predictions - targets
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if np.std(targets) > 0 and np.std(predictions) > 0:
        pearson = np.corrcoef(targets, predictions)[0, 1]
    else:
        pearson = float("nan")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson": float(pearson),
    }


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_predictions = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).view(-1, 1)

            if training:
                optimizer.zero_grad(set_to_none=True)

            predictions = model(images)
            loss = criterion(predictions, targets)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Dataset targets are normalized to [0, 1].
            all_targets.append(targets.detach().float().cpu().numpy() * 100.0)
            all_predictions.append(
                predictions.detach().float().cpu().numpy() * 100.0
            )

    targets = np.concatenate(all_targets).reshape(-1)
    predictions = np.concatenate(all_predictions).reshape(-1)

    metrics = calculate_metrics(targets, predictions)
    metrics["loss"] = total_loss / max(total_samples, 1)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/litemstemporal_baseline")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update({
        "architecture": "LiteMSTemporalEFNet",
        "device": str(device),
        "target_scale": "normalized EF / 100; metrics in EF percentage points",
        "image_size": [224, 224],
        "sequence_length": 32,
        "seed": args.seed,
    })

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Architecture: LiteMSTemporalEFNet")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Output:", output_dir.resolve())

    print("\nLoading TRAIN and VAL records...")
    records = load_records(splits=("TRAIN", "VAL"))

    for split in ("TRAIN", "VAL"):
        if not records.get(split):
            raise RuntimeError(f"No videos found for split {split}")
        print(f"{split}: {len(records[split])} videos")

    train_dataset = EchoDataset(records["TRAIN"])
    val_dataset = EchoDataset(records["VAL"])

    loader_args = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    if args.workers > 0:
        loader_args["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_args,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_args,
    )

    model = LiteMSTemporalEFNet(
        seq_len=32,
        temporal_dim=128,
        pretrained=not args.no_pretrained,
        dropout=0.2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Parameters: {total_params:,} total; {trainable_params:,} trainable")

    criterion = nn.HuberLoss(delta=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    history_path = output_dir / "history.csv"
    best_path = output_dir / "best_model.pth"
    best_mae = float("inf")
    epochs_without_improvement = 0

    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "lr", "train_loss", "train_mae", "train_rmse",
                "train_r2", "train_pearson", "val_loss", "val_mae",
                "val_rmse", "val_r2", "val_pearson", "epoch_seconds",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()

            train_metrics = run_epoch(
                model, train_loader, criterion, optimizer, device, True
            )
            val_metrics = run_epoch(
                model, val_loader, criterion, optimizer, device, False
            )

            scheduler.step(val_metrics["mae"])
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time

            row = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "train_pearson": train_metrics["pearson"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "val_pearson": val_metrics["pearson"],
                "epoch_seconds": elapsed,
            }
            writer.writerow(row)
            f.flush()

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"LR {current_lr:.2e} | "
                f"Train MAE {train_metrics['mae']:.3f} | "
                f"Val MAE {val_metrics['mae']:.3f} | "
                f"Val RMSE {val_metrics['rmse']:.3f} | "
                f"Val R2 {val_metrics['r2']:.4f} | "
                f"Val Pearson {val_metrics['pearson']:.4f} | "
                f"{elapsed:.1f}s"
            )

            if val_metrics["mae"] < best_mae:
                best_mae = val_metrics["mae"]
                epochs_without_improvement = 0

                torch.save({
                    "epoch": epoch,
                    "model_name": "LiteMSTemporalEFNet",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_metrics": val_metrics,
                    "config": config,
                }, best_path)

                print(f"  -> Best checkpoint saved: {best_path}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    summary = {
        "architecture": "LiteMSTemporalEFNet",
        "best_checkpoint": str(best_path.resolve()),
        "best_val_mae": best_mae,
        "epochs_completed": epoch,
        "test_evaluated": False,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining finished.")
    print("Best validation MAE:", round(best_mae, 4), "EF percentage points")
    print("Checkpoint:", best_path.resolve())
    print("History:", history_path.resolve())


if __name__ == "__main__":
    main()
