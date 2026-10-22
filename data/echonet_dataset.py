import os
import re
import csv

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMG_DIR = "/home/dellel/Downloads/archive/imagesecho"
CSV_PATH = "/home/dellel/Downloads/archive/EchoNet-Dynamic/FileList.csv"

SEQ_LEN = 32
MEAN = 0.12881848
STD = 0.18973795


def load_records(splits=("TRAIN", "VAL")):

    metadata = {}

    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            video = row["FileName"].removesuffix(".avi")
            metadata[video] = row

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

    records = {split: [] for split in splits}

    for video, frames in videos.items():
        split = metadata[video]["Split"]

        if split not in records:
            continue

        frames.sort(key=lambda item: item[0])

        records[split].append(
            (video, frames, float(metadata[video]["EF"]))
        )

    for split in records:
        records[split].sort(key=lambda item: item[0])

    return records


class EchoDataset(Dataset):
    def __init__(self, records):
        self.records = records

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[MEAN], std=[STD]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        _, frames, ef = self.records[index]
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
                images.append(self.transform(img.convert("L")))

        return (
            torch.stack(images),
            torch.tensor(ef / 100.0, dtype=torch.float32),
        )
