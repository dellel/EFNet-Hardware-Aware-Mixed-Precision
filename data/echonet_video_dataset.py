"""EchoNet-Dynamic dataset decoded directly from original AVI videos."""

import csv
import random
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CSV_PATH = Path("/home/dellel/Downloads/archive/EchoNet-Dynamic/FileList.csv")
VIDEO_DIR = Path("/home/dellel/kria/data/EchoNet-Dynamic/Videos")

SEQ_LEN = 64
IMAGE_SIZE = 224

# Temporary values. Recompute from TRAIN videos before training.
MEAN = 0.12881848
STD = 0.18973795


def load_video_records(splits=("TRAIN", "VAL")):
    """Load official video-level metadata."""
    records = {split: [] for split in splits}

    with CSV_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            split = row["Split"]
            if split not in records:
                continue

            video_name = row["FileName"].removesuffix(".avi")
            video_path = VIDEO_DIR / f"{video_name}.avi"

            records[split].append({
                "video": video_name,
                "path": str(video_path.resolve()),
                "ef": float(row["EF"]),
                "split": split,
                "num_frames": int(row["NumberOfFrames"]),
                "fps": float(row["FPS"]),
            })

    for split in records:
        records[split].sort(key=lambda r: r["video"])

    return records


class EchoVideoDataset(Dataset):
    """
    One sample per video.

    Long videos: select 64 consecutive frames.
    Short videos: use every available frame, without artificial padding.
    TRAIN uses random windows; VAL uses deterministic centered windows.
    """

    def __init__(
        self,
        records,
        training=False,
        seq_len=SEQ_LEN,
        image_size=IMAGE_SIZE,
        mean=MEAN,
        std=STD,
    ):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if std <= 0:
            raise ValueError("std must be positive")

        self.records = list(records)
        self.training = training
        self.seq_len = seq_len

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ])

    def __len__(self):
        return len(self.records)

    def sequence_length(self, index):
        """Effective sequence length for batch grouping."""
        return min(self.seq_len, self.records[index]["num_frames"])

    def __getitem__(self, index):
        record = self.records[index]
        cap = cv2.VideoCapture(record["path"])

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {record['path']}")

        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n_frames < 1:
                raise RuntimeError(
                    f"Invalid frame count for {record['video']}: {n_frames}"
                )

            length = min(self.seq_len, n_frames)
            max_start = n_frames - length

            if self.training and max_start > 0:
                start = random.randint(0, max_start)
            else:
                start = max_start // 2

            # Decode sequentially for reliable AVI frame indexing.
            frames = []
            for frame_index in range(start + length):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"Decode failure in {record['video']} at frame "
                        f"{frame_index}; requested {start}:{start + length}."
                    )

                if frame_index >= start:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(self.transform(Image.fromarray(gray)))

            if len(frames) != length:
                raise RuntimeError(
                    f"{record['video']}: got {len(frames)} frames, "
                    f"expected {length}."
                )

            images = torch.stack(frames)

        finally:
            cap.release()

        target = torch.tensor(record["ef"] / 100.0, dtype=torch.float32)
        return images, target
