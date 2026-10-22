import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.echonet_dataset import load_records, EchoDataset
from models.LiteMSTemporalEFNet import LiteMSTemporalEFNet


def calculate_metrics(y_true, y_pred):
    error = y_pred - y_true
    bias = float(np.mean(error))
    sd = float(np.std(error, ddof=1))

    return {
        "n_videos": int(len(y_true)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "R2": float(
            1 - np.sum(error ** 2) /
            np.sum((y_true - np.mean(y_true)) ** 2)
        ),
        "Pearson": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "bias_prediction_minus_reference": bias,
        "Bland_Altman_lower_95": bias - 1.96 * sd,
        "Bland_Altman_upper_95": bias + 1.96 * sd,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="experiments/litemstemporal_baseline/best_model.pth",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        default="experiments/litemstemporal_baseline/independent_validation.json",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)

    records = load_records(splits=("VAL",))
    val_records = records["VAL"]
    print("Validation videos:", len(val_records))

    dataset = EchoDataset(val_records)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    model = LiteMSTemporalEFNet(
        seq_len=32,
        temporal_dim=128,
        pretrained=False,
        dropout=0.2,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    targets, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images.to(device, non_blocking=True))
            targets.extend((labels.reshape(-1).numpy() * 100).tolist())
            predictions.extend(
                (outputs.reshape(-1).cpu().numpy() * 100).tolist()
            )

    y_true = np.asarray(targets, dtype=np.float64)
    y_pred = np.asarray(predictions, dtype=np.float64)

    result = calculate_metrics(y_true, y_pred)
    result.update({
        "architecture": "LiteMSTemporalEFNet",
        "split": "VAL",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_val_metrics": checkpoint.get("best_val_metrics"),
        "checkpoint": str(checkpoint_path),
        "evaluation_mode": "eval + inference_mode",
        "note": "Validation only; this is not a final test result.",
    })

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print("\nRe-evaluated validation metrics:")
    print(json.dumps(result, indent=2))
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
