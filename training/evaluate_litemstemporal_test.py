import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.echonet_video_dataset import (
    EchoVideoDataset,
    load_video_records,
)
from models.LiteMSTemporalEFNet import LiteMSTemporalEFNet


def calculate_metrics(targets, predictions):
    errors = predictions - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    if len(targets) > 1 and np.var(targets) > 0:
        r2 = float(1.0 - np.sum(errors ** 2) /
                   np.sum((targets - np.mean(targets)) ** 2))
        pearson = float(np.corrcoef(targets, predictions)[0, 1])
    else:
        r2 = float("nan")
        pearson = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson,
        "bias": float(np.mean(errors)),
    }


def main():
    output_dir = ROOT / "experiments/litemstemporal_video64"
    checkpoint_path = output_dir / "best_model.pth"
    stats_path = output_dir / "train_pixel_stats.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    with stats_path.open() as f:
        stats = json.load(f)

    mean = float(stats["mean"])
    std = float(stats["std"])

    records = load_video_records(("TEST",))["TEST"]
    if len(records) != 1277:
        raise RuntimeError(
            f"Expected 1277 TEST videos, found {len(records)}"
        )

    missing = [r["path"] for r in records if not Path(r["path"]).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} TEST videos are missing. First: {missing[0]}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"TEST videos : {len(records)}")
    print(f"Normalization : mean={mean:.6f}, std={std:.6f}")
    print("TEST is used only for final evaluation.")

    dataset = EchoVideoDataset(
        records,
        training=False,
        seq_len=64,
        image_size=224,
        mean=mean,
        std=std,
    )

    # Sequential evaluation preserves the correspondence between
    # dataset records, targets, and predictions. Batch size 1 also
    # supports videos with different temporal lengths.
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )

    model = LiteMSTemporalEFNet(
        seq_len=64,
        temporal_dim=128,
        pretrained=False,
        dropout=0.2,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_targets = []
    all_predictions = []
    all_names = []

    with torch.inference_mode():
        for i, (images, targets) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            predictions = model(images).reshape(-1).cpu().numpy() * 100.0
            targets = targets.reshape(-1).numpy() * 100.0

            all_targets.extend(targets.tolist())
            all_predictions.extend(predictions.tolist())

            # The dataset is sorted by video name, but the sampler shuffles
            # batches only when shuffle=True; here order is deterministic.
            start_idx = len(all_names)
            batch_names = [
                records[j]["video"]
                for j in range(start_idx, start_idx + len(targets))
            ]
            all_names.extend(batch_names)

            if i % 100 == 0:
                print(f"Batch {i}/{len(loader)} terminé")

    targets = np.asarray(all_targets, dtype=np.float64)
    predictions = np.asarray(all_predictions, dtype=np.float64)

    metrics = calculate_metrics(targets, predictions)

    print("\n=== RÉSULTATS TEST ===")
    print(f"Checkpoint époque : {checkpoint['epoch']}")
    for key, value in metrics.items():
        print(f"{key.upper():10s}: {value:.4f}")

    predictions_path = output_dir / "test_predictions.csv"
    pd.DataFrame({
        "FileName": all_names,
        "EF_true": targets,
        "EF_pred": predictions,
        "error_pred_minus_true": predictions - targets,
        "absolute_error": np.abs(predictions - targets),
    }).to_csv(predictions_path, index=False)

    metrics_path = output_dir / "test_metrics.json"
    with metrics_path.open("w") as f:
        json.dump({
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "n_test_videos": len(targets),
            "normalization_mean": mean,
            "normalization_std": std,
            **metrics,
        }, f, indent=2)

    print(f"\nPrédictions : {predictions_path}")
    print(f"Métriques   : {metrics_path}")


if __name__ == "__main__":
    main()
