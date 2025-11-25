import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from rcnn_dataset import FruitDetectionDataset
from rcnn_model import get_faster_rcnn_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# number of fruit classes (without background)
NUM_FRUIT_CLASSES = 10
NUM_CLASSES = NUM_FRUIT_CLASSES + 1  # +1 for background


def get_transforms(train=True):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, epoch):
    model.train()
    total_loss = 0.0

    for images, targets in data_loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch}: train loss = {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate_simple(model, data_loader):
    model.eval()
    total_iou = 0.0
    n_boxes = 0

    for images, targets in data_loader:
        images = [img.to(DEVICE) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            if len(out["boxes"]) == 0 or len(tgt["boxes"]) == 0:
                continue

            ious = box_iou(out["boxes"].cpu(), tgt["boxes"])
            max_iou, _ = ious.max(dim=0)
            total_iou += max_iou.sum().item()
            n_boxes += len(max_iou)

    if n_boxes > 0:
        mean_iou = total_iou / n_boxes
        print(f"Mean IoU over GT boxes: {mean_iou:.3f}")
        return mean_iou
    else:
        print("No boxes to evaluate (check labels).")
        return 0.0


def main():
    train_imgs = "../dataset/split/images/train"
    train_labels = "../dataset/split/labels/train"

    val_imgs = "../dataset/split/images/val"
    val_labels = "../dataset/split/labels/val"

    # datasets
    train_dataset = FruitDetectionDataset(
        images_dir=train_imgs,
        labels_dir=train_labels,
        transforms=get_transforms(train=True),
    )
    val_dataset = FruitDetectionDataset(
        images_dir=val_imgs,
        labels_dir=val_labels,
        transforms=get_transforms(train=False),
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    # data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # model
    model = get_faster_rcnn_model(num_classes=NUM_CLASSES)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1,
    )

    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, optimizer, train_loader, epoch)
        lr_scheduler.step()
        evaluate_simple(model, val_loader)

    # save model
    save_path = "../models/faster_rcnn_fruits.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Saved model to:", save_path)


if __name__ == "__main__":
    main()
