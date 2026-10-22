import os
import re
import csv
import sys
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.TemporalEFNetV2Kria import TemporalEFNetV2Kria

# Paths
IMG_DIR = "/home/dellel/Downloads/archive/imagesecho"
CSV_PATH = "/home/dellel/Downloads/archive/EchoNet-Dynamic/FileList.csv"
CHECKPOINT = "/home/dellel/kria/echonet_int8/TemporalEFNetV2Kria_best.pth"

SEQ_LEN = 32
BATCH_SIZE = 4
MEAN = 0.12881848
STD = 0.18973795

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Read official metadata
metadata = {}
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        metadata[row["FileName"].removesuffix(".avi")] = row

# Reconstruct image index from filenames
pattern = re.compile(r"(.+\.avi)_(\d+)\.png$")
videos = {}

for entry in os.scandir(IMG_DIR):
    if not entry.name.endswith(".png"):
        continue
    match = pattern.fullmatch(entry.name)
    if not match:
        continue

    video = match.group(1).removesuffix(".avi")
    if video not in metadata:
        continue

    videos.setdefault(video, []).append(
        (int(match.group(2)), entry.name)
    )

# Official validation split only
test_records = []
for video, frames in videos.items():
    row = metadata[video]
    if row["Split"] != "VAL":
        continue
    frames.sort(key=lambda item: item[0])
    test_records.append((video, frames, float(row["EF"])))

test_records.sort(key=lambda item: item[0])
print("Validation videos indexed:", len(test_records))
print("Validation images indexed:", sum(len(r[1]) for r in test_records))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN], std=[STD]),
])

class EchoTestDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        video, frames, ef = self.records[index]

        # Same centered 32-frame evaluation window
        n = len(frames)
        if n >= SEQ_LEN:
            start = (n - SEQ_LEN) // 2
            selected = frames[start:start + SEQ_LEN]
        else:
            selected = frames + [frames[-1]] * (SEQ_LEN - n)

        images = []
        for _, filename in selected:
            path = os.path.join(IMG_DIR, filename)
            with Image.open(path) as img:
                images.append(transform(img.convert("L")))

        return torch.stack(images), torch.tensor(ef, dtype=torch.float32)

if not test_records:
    raise RuntimeError("No test videos found. Check CSV/image filename matching.")

loader = DataLoader(
    EchoTestDataset(test_records),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Load original FP32 checkpoint
model = TemporalEFNetV2Kria(seq_len=SEQ_LEN)
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.to(device).eval()

predictions, labels = [], []
criterion = torch.nn.SmoothL1Loss()
total_loss = 0.0

with torch.no_grad():
    for batch_idx, (images, ef) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        labels_norm = (ef.to(device, non_blocking=True).unsqueeze(1) / 100.0)

        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                output = model(images)
                loss = criterion(output, labels_norm)
        else:
            output = model(images)
            loss = criterion(output, labels_norm)

        total_loss += loss.item()
        predictions.extend((output.float().cpu().numpy().flatten() * 100).tolist())
        labels.extend((labels_norm.cpu().numpy().flatten() * 100).tolist())

        if batch_idx % 50 == 0:
            print(f"Processed {batch_idx}/{len(loader)} batches")

predictions = np.asarray(predictions)
labels = np.asarray(labels)

mae = np.mean(np.abs(predictions - labels))
rmse = np.sqrt(np.mean((predictions - labels) ** 2))
ss_res = np.sum((labels - predictions) ** 2)
ss_tot = np.sum((labels - labels.mean()) ** 2)
r2 = 1.0 - ss_res / ss_tot
pearson = np.corrcoef(labels, predictions)[0, 1]

print("\n========== FP32 VALIDATION RESULTS ==========")
print("Checkpoint epoch:", checkpoint["epoch"])
print("Test videos:", len(labels))
print(f"SmoothL1 loss: {total_loss / len(loader):.6f}")
print(f"MAE:           {mae:.4f} percentage points")
print(f"RMSE:          {rmse:.4f} percentage points")
print(f"R²:            {r2:.4f}")
print(f"Pearson:       {pearson:.4f}")

np.savez(
    "fp32_val_predictions.npz",
    predictions=predictions,
    labels=labels
)
print("Predictions saved to fp32_val_predictions.npz")
