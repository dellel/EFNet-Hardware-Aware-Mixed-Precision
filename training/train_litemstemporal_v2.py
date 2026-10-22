"""
Entraînement v2 : accumulation de gradient, AMP, EMA, LR différenciés,
warmup + cosine, gradient clipping, perte pondérée pour les FEVG basses,
métriques cliniques stratifiées.

À placer dans le MÊME dossier que votre script d'entraînement actuel.
Les fonctions set_seed, calculate_metrics, compute_train_stats et
LengthBucketBatchSampler sont réutilisées telles quelles.
"""
import argparse
import copy
import csv
import json
import math
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.echonet_video_dataset import (
    CSV_PATH,
    EchoVideoDataset,
    load_video_records,
)
from models.LiteMSTemporalEFNet_v2 import LiteMSTemporalEFNet

# >>> ADAPTEZ "train" au nom de votre script actuel (sans .py) <<<
from training.train_litemstemporal_video import (
    set_seed,
    calculate_metrics,
    compute_train_stats,
    LengthBucketBatchSampler,
)


# ----------------------------------------------------------------------
# EMA des poids
# ----------------------------------------------------------------------
class ModelEMA:
    def __init__(self, model, decay=0.998):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])


# ----------------------------------------------------------------------
# Perte pondérée : plus de poids sur les FEVG < 40 % (rares, cliniquement
# critiques). Les cibles sont en [0, 1].
# ----------------------------------------------------------------------
def weighted_loss(predictions, targets, base_loss, low_ef_weight):
    per_sample = base_loss(predictions, targets).view(-1)
    is_low = (targets.view(-1) < 0.40).float()
    weights = 1.0 + (low_ef_weight - 1.0) * is_low
    return (per_sample * weights).sum() / weights.sum()


# ----------------------------------------------------------------------
# Métriques cliniques (cibles et prédictions en points de FEVG, 0-100)
# ----------------------------------------------------------------------
def clinical_metrics(targets, predictions):
    t = np.asarray(targets).reshape(-1)
    p = np.asarray(predictions).reshape(-1)
    out = {}

    strata = {
        "low": t <= 40,
        "mid": (t > 40) & (t < 50),
        "high": t >= 50,
    }
    for name, mask in strata.items():
        out[f"mae_{name}"] = (
            float(np.abs(p[mask] - t[mask]).mean())
            if mask.any() else float("nan")
        )

    # Pente erreur ~ FEVG vraie : < 0 = régression vers la moyenne.
    out["bias_slope"] = (
        float(np.polyfit(t, p - t, 1)[0]) if np.std(t) > 0 else float("nan")
    )

    # Détection FEVG <= 40 % (HFrEF).
    pos = t <= 40
    pred_pos = p <= 40
    out["sens_le40"] = float((pred_pos & pos).sum() / max(pos.sum(), 1))
    out["spec_le40"] = float((~pred_pos & ~pos).sum() / max((~pos).sum(), 1))
    return out


# ----------------------------------------------------------------------
# Une époque
# ----------------------------------------------------------------------
def run_epoch(
    model, loader, loss_fn, device, training,
    optimizer=None, scaler=None, scheduler=None, ema=None,
    accum_steps=1, clip=1.0, use_amp=False,
):
    model.train(training)

    total_loss, total_samples = 0.0, 0
    all_targets, all_predictions = [], []

    n_batches = len(loader)
    context = torch.enable_grad() if training else torch.no_grad()

    if training:
        optimizer.zero_grad(set_to_none=True)

    with context:
        for i, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).view(-1, 1)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                predictions = model(images)

            loss = loss_fn(predictions.float(), targets)

            if training:
                scaler.scale(loss / accum_steps).backward()

                if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    if ema is not None:
                        ema.update(model)

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            all_targets.append(targets.detach().cpu().numpy() * 100.0)
            all_predictions.append(
                predictions.detach().float().cpu().numpy() * 100.0
            )

    t = np.concatenate(all_targets)
    p = np.concatenate(all_predictions)
    metrics = calculate_metrics(t, p)
    metrics.update(clinical_metrics(t, p))
    metrics["loss"] = total_loss / total_samples
    return metrics


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=8,
                        help="Batch effectif = batch-size * accum-steps.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="LR des têtes (encodeur de projection, temporel, régression).")
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-epochs", type=int, default=2,
                        help="Époques initiales avec backbone gelé.")
    parser.add_argument("--ema-decay", type=float, default=0.998)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--low-ef-weight", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=15,
                        help="0 = pas d'arrêt anticipé.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--temporal-dim", type=int, default=256)
    parser.add_argument("--output", default="experiments/litemstemporal_v2")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-freeze-bn", action="store_true")
    parser.add_argument("--loss", choices=["huber", "l1"], default="huber")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_video_records(("TRAIN", "VAL"))
    train_records, val_records = records["TRAIN"], records["VAL"]
    if not train_records or not val_records:
        raise RuntimeError("TRAIN ou VAL est vide.")

    with CSV_PATH.open(newline="") as f:
        test_count = sum(r["Split"] == "TEST" for r in csv.DictReader(f))

    print(f"Device : {device} | AMP : {use_amp}")
    print(f"TRAIN : {len(train_records)} | VAL : {len(val_records)} "
          f"| TEST : {test_count} (non chargé)")

    mean, std = compute_train_stats(
        train_records, output_dir / "train_pixel_stats.json", seed=args.seed
    )

    train_dataset = EchoVideoDataset(
        train_records, training=True, seq_len=args.seq_len,
        image_size=args.image_size, mean=mean, std=std,
    )
    val_dataset = EchoVideoDataset(
        val_records, training=False, seq_len=args.seq_len,
        image_size=args.image_size, mean=mean, std=std,
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
        seq_len=args.seq_len,
        temporal_dim=args.temporal_dim,
        pretrained=not args.no_pretrained,
        dropout=0.2,
        freeze_backbone_bn=not args.no_freeze_bn,
    ).to(device)

    print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")

    ema = ModelEMA(model, decay=args.ema_decay)

    # --- Perte -------------------------------------------------------
    base_loss = (
        nn.HuberLoss(delta=0.05, reduction="none")
        if args.loss == "huber"
        else nn.L1Loss(reduction="none")
    )

    def loss_fn(pred, target):
        return weighted_loss(pred, target, base_loss, args.low_ef_weight)

    # --- Optimiseur : LR différenciés --------------------------------
    backbone_params = list(model.encoder.features.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    # --- Schedule : warmup (1 époque) + cosine, par pas d'optimiseur --
    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args.freeze_epochs > 0:
        model.set_backbone_frozen(True)

    config = vars(args).copy()
    config.update({
        "architecture": "LiteMSTemporalEFNet_v2",
        "normalization_mean": mean,
        "normalization_std": std,
        "effective_batch_size": args.batch_size * args.accum_steps,
        "train_videos": len(train_records),
        "val_videos": len(val_records),
        "test_used": False,
    })
    with (output_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    history_path = output_dir / "history.csv"
    best_path = output_dir / "best_model.pth"
    best_mae = float("inf")
    stale = 0
    writer = None

    with history_path.open("w", newline="") as f:
        for epoch in range(1, args.epochs + 1):
            if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
                model.set_backbone_frozen(False)
                print("  Backbone dégelé.")

            train_sampler.set_epoch(epoch)
            start = time.time()

            train_m = run_epoch(
                model, train_loader, loss_fn, device, True,
                optimizer=optimizer, scaler=scaler, scheduler=scheduler,
                ema=ema, accum_steps=args.accum_steps,
                clip=args.clip, use_amp=use_amp,
            )
            # Validation avec les poids EMA.
            val_m = run_epoch(
                ema.module, val_loader, loss_fn, device, False,
                use_amp=use_amp,
            )

            elapsed = time.time() - start
            lr_head = optimizer.param_groups[1]["lr"]

            row = {"epoch": epoch, "lr": lr_head, "epoch_seconds": elapsed}
            row.update({f"train_{k}": v for k, v in train_m.items()})
            row.update({f"val_{k}": v for k, v in val_m.items()})

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Train MAE {train_m['mae']:.3f} | "
                f"Val MAE {val_m['mae']:.3f} | RMSE {val_m['rmse']:.3f} | "
                f"R2 {val_m['r2']:.4f} | "
                f"MAE low/mid/high "
                f"{val_m['mae_low']:.2f}/{val_m['mae_mid']:.2f}/"
                f"{val_m['mae_high']:.2f} | "
                f"slope {val_m['bias_slope']:+.3f} | "
                f"LR {lr_head:.2e} | {elapsed / 60:.1f} min"
            )

            if val_m["mae"] < best_mae:
                best_mae = val_m["mae"]
                stale = 0
                torch.save({
                    "epoch": epoch,
                    # Poids EMA : ce sont ceux à utiliser en inférence.
                    "model_state_dict": ema.module.state_dict(),
                    "best_val_mae": best_mae,
                    "config": config,
                }, best_path)
                print(f"  Nouveau meilleur modèle (EMA) : {best_path}")
            else:
                stale += 1

            if args.patience > 0 and stale >= args.patience:
                print("Arrêt anticipé : patience atteinte.")
                break

    print(f"\nTerminé. Meilleure MAE VAL (EMA) : {best_mae:.3f}")
    print(f"Checkpoint : {best_path}\nHistorique : {history_path}")


if __name__ == "__main__":
    main()
