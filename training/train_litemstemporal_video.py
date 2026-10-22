import csv
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.echonet_video_dataset import (
    CSV_PATH,
    EchoVideoDataset,
    load_video_records,
)
from models.LiteMSTemporalEFNet import LiteMSTemporalEFNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(targets, predictions):
    targets = np.asarray(targets).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)

    errors = predictions - targets
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    ss_tot = np.sum((targets - targets.mean()) ** 2)
    ss_res = np.sum(errors ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    pearson = (
        np.corrcoef(targets, predictions)[0, 1]
        if np.std(targets) > 0 and np.std(predictions) > 0
        else float("nan")
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson": float(pearson),
    }


def compute_train_stats(records, cache_path, seed=42, n_videos=512):
    """Estimate pixel normalization from TRAIN videos only."""
    if cache_path.exists():
        with cache_path.open() as f:
            stats = json.load(f)
        print(f"Normalization chargée : {stats}")
        return stats["mean"], stats["std"]

    rng = random.Random(seed)
    selected = rng.sample(records, min(n_videos, len(records)))

    resize = transforms.Resize((224, 224))
    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0

    print(
        f"Calcul des statistiques TRAIN sur "
        f"{len(selected)} vidéos, 3 images par vidéo..."
    )

    for i, record in enumerate(selected, 1):
        cap = cv2.VideoCapture(record["path"])
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir {record['path']}")

        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n < 1:
                raise RuntimeError(f"Vidéo vide : {record['path']}")

            indices = sorted(set([
                min(n - 1, n // 4),
                min(n - 1, n // 2),
                min(n - 1, (3 * n) // 4),
            ]))

            frame_idx = 0
            wanted = set(indices)

            while frame_idx <= indices[-1]:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"Erreur de décodage : {record['video']}, "
                        f"image {frame_idx}"
                    )

                if frame_idx in wanted:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    image = resize(Image.fromarray(gray))
                    array = np.asarray(image, dtype=np.float64) / 255.0

                    total_sum += array.sum()
                    total_sq_sum += np.square(array).sum()
                    total_pixels += array.size

                frame_idx += 1
        finally:
            cap.release()

        if i % 100 == 0:
            print(f"  {i}/{len(selected)} vidéos")

    mean = total_sum / total_pixels
    variance = max(total_sq_sum / total_pixels - mean ** 2, 0.0)
    std = float(np.sqrt(variance))

    if std < 1e-8:
        raise RuntimeError("Écart-type invalide.")

    stats = {
        "mean": float(mean),
        "std": std,
        "resolution": [224, 224],
        "method": "3 quarter-spaced frames from 512 seeded TRAIN videos",
        "seed": seed,
        "num_videos": len(selected),
    }

    with cache_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"Statistiques calculées : mean={mean:.6f}, std={std:.6f}")
    return float(mean), std


class LengthBucketBatchSampler(Sampler):
    """Group videos of identical effective temporal length."""

    def __init__(self, dataset, batch_size, shuffle, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        buckets = {}
        for index in range(len(dataset)):
            length = dataset.sequence_length(index)
            buckets.setdefault(length, []).append(index)

        self.buckets = buckets

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []

        for indices in self.buckets.values():
            indices = list(indices)

            if self.shuffle:
                rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]

                # Shuffle batches, but never mix different sequence lengths.
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)

        yield from batches

    def __len__(self):
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.buckets.values()
        )


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

            all_targets.append(targets.detach().cpu().numpy() * 100.0)
            all_predictions.append(
                predictions.detach().cpu().numpy() * 100.0
            )

    metrics = calculate_metrics(
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
    )
    metrics["loss"] = total_loss / total_samples
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="experiments/litemstemporal_video64",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--loss",
        choices=["huber", "l1"],
        default="huber",
        help="Fonction de perte pour la régression de la FEVG.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_video_records(("TRAIN", "VAL"))

    train_records = records["TRAIN"]
    val_records = records["VAL"]

    if not train_records or not val_records:
        raise RuntimeError("TRAIN ou VAL est vide.")

    # Confirm the CSV source and splits before training.
    with CSV_PATH.open(newline="") as f:
        csv_rows = list(csv.DictReader(f))

    test_count = sum(row["Split"] == "TEST" for row in csv_rows)

    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")

    print(f"TRAIN : {len(train_records)} vidéos")
    print(f"VAL   : {len(val_records)} vidéos")
    print(f"TEST  : {test_count} vidéos, non chargé")

    mean, std = compute_train_stats(
        train_records,
        output_dir / "train_pixel_stats.json",
        seed=args.seed,
    )

    train_dataset = EchoVideoDataset(
        train_records, training=True, seq_len=64,
        image_size=224, mean=mean, std=std,
    )
    val_dataset = EchoVideoDataset(
        val_records, training=False, seq_len=64,
        image_size=224, mean=mean, std=std,
    )

    train_sampler = LengthBucketBatchSampler(
        train_dataset, args.batch_size, shuffle=True, seed=args.seed
    )
    val_sampler = LengthBucketBatchSampler(
        val_dataset, args.batch_size, shuffle=False, seed=args.seed
    )

    loader_args = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    if args.workers > 0:
        loader_args["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler, **loader_args
    )
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler, **loader_args
    )

    model = LiteMSTemporalEFNet(
        seq_len=64,
        temporal_dim=128,
        pretrained=not args.no_pretrained,
        dropout=0.2,
    ).to(device)

    print(
        f"Paramètres : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    if args.loss == "huber":
        criterion = nn.HuberLoss(delta=0.05)
    else:
        criterion = nn.L1Loss()
    print(f"Fonction de perte : {args.loss}")
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

    config = vars(args).copy()
    config.update({
        "architecture": "LiteMSTemporalEFNet",
        "sequence_length": 64,
        "image_size": [224, 224],
        "normalization_mean": mean,
        "normalization_std": std,
        "train_videos": len(train_records),
        "val_videos": len(val_records),
        "test_used": False,
        "loss": args.loss,
        "seed": args.seed,
    })
    with (output_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    history_path = output_dir / "history.csv"
    best_path = output_dir / "best_model.pth"
    best_mae = float("inf")
    epochs_without_improvement = 0

    columns = [
        "epoch", "lr", "train_loss", "train_mae", "train_rmse",
        "train_r2", "train_pearson", "val_loss", "val_mae",
        "val_rmse", "val_r2", "val_pearson", "epoch_seconds",
    ]

    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_sampler.set_epoch(epoch)

            start = time.time()
            train_metrics = run_epoch(
                model, train_loader, criterion, optimizer, device, True
            )
            val_metrics = run_epoch(
                model, val_loader, criterion, optimizer, device, False
            )

            scheduler.step(val_metrics["mae"])
            elapsed = time.time() - start
            lr = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "lr": lr,
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
                f"Train MAE {train_metrics['mae']:.3f} | "
                f"Val MAE {val_metrics['mae']:.3f} | "
                f"Val RMSE {val_metrics['rmse']:.3f} | "
                f"R2 {val_metrics['r2']:.4f} | "
                f"Pearson {val_metrics['pearson']:.4f} | "
                f"LR {lr:.2e} | {elapsed / 60:.1f} min"
            )

            if val_metrics["mae"] < best_mae:
                best_mae = val_metrics["mae"]
                epochs_without_improvement = 0

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mae": best_mae,
                    "config": config,
                }, best_path)

                print(f"  Nouveau meilleur modèle : {best_path}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print("Arrêt anticipé : patience atteinte.")
                break

    print(f"\nEntraînement terminé. Meilleure MAE VAL : {best_mae:.3f}")
    print(f"Checkpoint : {best_path}")
    print(f"Historique : {history_path}")


if __name__ == "__main__":
    main()
