import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class FruitDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms

        self.img_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # load image
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # corresponding label file (YOLO txt)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])

                    # YOLO (cx, cy, w, h) ∈ [0,1] → pixel coords
                    x_center = cx * width
                    y_center = cy * height
                    bw = w * width
                    bh = h * height

                    x1 = x_center - bw / 2.0
                    y1 = y_center - bh / 2.0
                    x2 = x_center + bw / 2.0
                    y2 = y_center + bh / 2.0

                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls + 1)

        # tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target