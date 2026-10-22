import sys
import json
import time
import random
import argparse
import csv
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

    ss_tot = np.sum((targets - targets.mean()) ** 2)

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "r2": float(1 - np.sum(errors ** 2) / ss_tot),
        "pearson": float(np.corrcoef(targets, predictions)[0, 1]),
        "bias": float(np.mean(errors)),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    targets_all = []
    predictions_all = []
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad() if training else torch.inference_mode()

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

            targets_all.extend((targets.detach().cpu().numpy().ravel() * 100).tolist())
            predictions_all.extend(
                (predictions.detach().cpu().numpy().ravel() * 100).tolist()
            )

    metrics = calculate_metrics(targets_all, predictions_all)
    metrics["loss"] = total_loss / total_samples
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="experiments/litemstemporal_baseline/best_model.pth",
    )
    parser.add_argument(
        "--output",
        default="experiments/litemstemporal_finetune",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    print("Device:", device)
    print("Initial checkpoint:", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    records = load_records(splits=("TRAIN", "VAL"))
    train_records = records["TRAIN"]
    val_records = records["VAL"]

    if not train_records or not val_records:
        raise RuntimeError("TRAIN or VAL is empty")

    train_ids = {r[0] for r in train_records}
    val_ids = {r[0] for r in val_records}
    if train_ids & val_ids:
        raise RuntimeError("TRAIN/VAL video overlap detected")

    print("TRAIN videos:", len(train_records))
    print("VAL videos:", len(val_records))

    loader_args = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    if args.workers > 0:
        loader_args["persistent_workers"] = True

    train_loader = DataLoader(
        EchoDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_args,
    )
    val_loader = DataLoader(
        EchoDataset(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_args,
    )

    model = LiteMSTemporalEFNet(
        seq_len=32,
        temporal_dim=128,
        pretrained=False,
        dropout=0.2,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    # Separate learning rates for spatial encoder and task-specific layers.
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.encoder.parameters()),
                "lr": args.encoder_lr,
                "name": "encoder",
            },
            {
                "params": (
                    list(model.temporal.parameters())
                    + list(model.temporal_pool.parameters())
                    + list(model.regressor.parameters())
                ),
                "lr": args.head_lr,
                "name": "temporal_and_regression",
            },
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )
    criterion = nn.HuberLoss(delta=0.05)

    config = vars(args).copy()
    config.update({
        "architecture": "LiteMSTemporalEFNet",
        "device": str(device),
        "initial_checkpoint_epoch": checkpoint["epoch"],
        "initial_validation_metrics": checkpoint["best_val_metrics"],
        "selection_metric": "validation MAE",
        "test_evaluated": False,
    })
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    history_path = output_dir / "history.csv"
    best_path = output_dir / "best_model.pth"

    best_mae = float(checkpoint["best_val_metrics"]["mae"])
    epochs_without_improvement = 0

    fields = [
        "epoch", "encoder_lr", "head_lr",
        "train_mae", "train_rmse", "train_r2",
        "val_mae", "val_rmse", "val_r2", "val_pearson", "val_bias",
        "epoch_seconds",
    ]

    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            start = time.time()

            train_metrics = run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            val_metrics = run_epoch(
                model, val_loader, criterion, device
            )

            scheduler.step(val_metrics["mae"])

            row = {
                "epoch": epoch,
                "encoder_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "val_pearson": val_metrics["pearson"],
                "val_bias": val_metrics["bias"],
                "epoch_seconds": time.time() - start,
            }
            writer.writerow(row)
            f.flush()

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"LR enc={row['encoder_lr']:.1e}, head={row['head_lr']:.1e} | "
                f"Train MAE={row['train_mae']:.3f} | "
                f"VAL MAE={row['val_mae']:.3f} | "
                f"RMSE={row['val_rmse']:.3f} | "
                f"R2={row['val_r2']:.4f}"
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
                    "initial_checkpoint": str(checkpoint_path),
                }, best_path)
                print("  -> Best fine-tuned checkpoint saved")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print("Early stopping.")
                break

    summary = {
        "architecture": "LiteMSTemporalEFNet",
        "initial_checkpoint": str(checkpoint_path),
        "initial_checkpoint_epoch": checkpoint["epoch"],
        "best_finetuned_checkpoint": str(best_path),
        "best_val_mae": best_mae,
        "epochs_completed": epoch,
        "test_evaluated": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nFine-tuning complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
