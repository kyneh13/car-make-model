from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT = Path("/workspace")
MODEL_PATH = PROJECT / "models/car_classifier_best.pth"
CLASSES_PATH = PROJECT / "models/classes.json"

IMAGE_SIZE = 224

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Improved Stanford Cars classifier training"
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT / "dataset/all_cars",
    )

    parser.add_argument(
        "--head-epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=35,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--head-lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=0.00003,
    )

    parser.add_argument(
        "--classifier-lr",
        type=float,
        default=0.0001,
    )

    return parser.parse_args()


def create_datasets(data_path: Path):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.70, 1.0),
            ratio=(0.85, 1.15),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(4),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        ),
        transforms.ToTensor(),
        NORMALIZE,
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        NORMALIZE,
    ])

    train_dataset = datasets.ImageFolder(
        data_path / "train",
        transform=train_transform,
    )

    validation_dataset = datasets.ImageFolder(
        data_path / "val",
        transform=evaluation_transform,
    )

    test_dataset = datasets.ImageFolder(
        data_path / "test",
        transform=evaluation_transform,
    )

    if train_dataset.classes != validation_dataset.classes:
        raise RuntimeError(
            "Training and validation classes do not match."
        )

    if train_dataset.classes != test_dataset.classes:
        raise RuntimeError(
            "Training and testing classes do not match."
        )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


def create_loader(
    dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def create_model(number_of_classes: int):
    weights = (
        models.MobileNet_V3_Small_Weights.DEFAULT
    )

    model = models.mobilenet_v3_small(
        weights=weights
    )

    input_features = (
        model.classifier[3].in_features
    )

    model.classifier[3] = nn.Linear(
        input_features,
        number_of_classes,
    )

    return model


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0

    use_amp = (
        device.type == "cuda"
        and scaler is not None
    )

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                scaler.step(optimizer)
                scaler.update()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        top_count = min(
            5,
            outputs.size(1),
        )

        predictions = outputs.topk(
            top_count,
            dim=1,
        ).indices

        matches = predictions.eq(
            labels.view(-1, 1)
        )

        correct_top1 += (
            matches[:, :1]
            .any(dim=1)
            .sum()
            .item()
        )

        correct_top5 += (
            matches[:, :top_count]
            .any(dim=1)
            .sum()
            .item()
        )

    return {
        "loss": total_loss / total_samples,
        "top1": correct_top1 / total_samples,
        "top5": correct_top5 / total_samples,
    }


def save_checkpoint(
    model,
    classes,
    validation_accuracy,
    stage,
):
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "architecture": "mobilenet_v3_small",
            "image_size": IMAGE_SIZE,
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "validation_accuracy":
                validation_accuracy,
            "training_stage": stage,
        },
        MODEL_PATH,
    )

    CLASSES_PATH.write_text(
        json.dumps(
            classes,
            indent=2,
        ),
        encoding="utf-8",
    )


def train_stage(
    name,
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    classes,
    epochs,
    best_validation_accuracy,
):
    print(f"\n{name}")
    print("=" * len(name))

    for epoch in range(1, epochs + 1):
        started = time.time()

        train_result = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )

        validation_result = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - started

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train loss: "
            f"{train_result['loss']:.4f} | "
            f"Train top-1: "
            f"{train_result['top1'] * 100:.2f}% | "
            f"Val top-1: "
            f"{validation_result['top1'] * 100:.2f}% | "
            f"Val top-5: "
            f"{validation_result['top5'] * 100:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        if (
            validation_result["top1"]
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_result["top1"]
            )

            save_checkpoint(
                model=model,
                classes=classes,
                validation_accuracy=(
                    best_validation_accuracy
                ),
                stage=name,
            )

            print(
                "Saved new best model: "
                f"{best_validation_accuracy * 100:.2f}%"
            )

    return best_validation_accuracy


def main():
    args = parse_arguments()

    torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Dataset: {args.data}")

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = create_datasets(args.data)

    classes = train_dataset.classes

    print(f"Classes: {len(classes)}")
    print(f"Training images: {len(train_dataset)}")
    print(
        f"Validation images: "
        f"{len(validation_dataset)}"
    )
    print(f"Testing images: {len(test_dataset)}")

    train_loader = create_loader(
        train_dataset,
        args.batch_size,
        args.workers,
        shuffle=True,
    )

    validation_loader = create_loader(
        validation_dataset,
        args.batch_size,
        args.workers,
        shuffle=False,
    )

    test_loader = create_loader(
        test_dataset,
        args.batch_size,
        args.workers,
        shuffle=False,
    )

    model = create_model(
        len(classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda"
    )

    best_validation_accuracy = 0.0

    # Stage 1:
    # Freeze the feature extractor and train only
    # the newly created classifier.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    head_optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.head_lr,
        weight_decay=0.0001,
    )

    head_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            head_optimizer,
            T_max=max(1, args.head_epochs),
        )
    )

    best_validation_accuracy = train_stage(
        name="Stage 1: classifier training",
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=head_optimizer,
        scheduler=head_scheduler,
        scaler=scaler,
        device=device,
        classes=classes,
        epochs=args.head_epochs,
        best_validation_accuracy=(
            best_validation_accuracy
        ),
    )

    # Stage 2:
    # Unfreeze the feature extractor and fine-tune
    # it with a much smaller learning rate.
    for parameter in model.features.parameters():
        parameter.requires_grad = True

    fine_tune_optimizer = torch.optim.AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": args.backbone_lr,
            },
            {
                "params": model.classifier.parameters(),
                "lr": args.classifier_lr,
            },
        ],
        weight_decay=0.0001,
    )

    fine_tune_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            fine_tune_optimizer,
            T_max=max(
                1,
                args.fine_tune_epochs,
            ),
        )
    )

    best_validation_accuracy = train_stage(
        name="Stage 2: full fine-tuning",
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=fine_tune_optimizer,
        scheduler=fine_tune_scheduler,
        scaler=scaler,
        device=device,
        classes=classes,
        epochs=args.fine_tune_epochs,
        best_validation_accuracy=(
            best_validation_accuracy
        ),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_result = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nFinal results")
    print("=============")

    print(
        "Best validation top-1: "
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print(
        "Test top-1 accuracy: "
        f"{test_result['top1'] * 100:.2f}%"
    )

    print(
        "Test top-5 accuracy: "
        f"{test_result['top5'] * 100:.2f}%"
    )

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
