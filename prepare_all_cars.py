from pathlib import Path
import random
import shutil

PROJECT = Path("/workspace")

TRAIN_SOURCE = PROJECT / "car_data/car_data/train"
TEST_SOURCE = PROJECT / "car_data/car_data/test"
OUTPUT = PROJECT / "dataset/all_cars"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALIDATION_PERCENT = 0.20
RANDOM_SEED = 42


def get_images(folder: Path):
    return sorted(
        file
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_images(files, destination):
    destination.mkdir(parents=True, exist_ok=True)

    for source in files:
        shutil.copy2(source, destination / source.name)


def main():
    random.seed(RANDOM_SEED)

    if not TRAIN_SOURCE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_SOURCE}"
        )

    if not TEST_SOURCE.exists():
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_SOURCE}"
        )

    train_classes = sorted(
        folder.name
        for folder in TRAIN_SOURCE.iterdir()
        if folder.is_dir()
    )

    test_classes = sorted(
        folder.name
        for folder in TEST_SOURCE.iterdir()
        if folder.is_dir()
    )

    if train_classes != test_classes:
        missing_from_test = sorted(
            set(train_classes) - set(test_classes)
        )

        missing_from_train = sorted(
            set(test_classes) - set(train_classes)
        )

        raise RuntimeError(
            "Training and testing classes do not match.\n"
            f"Missing from test: {missing_from_test}\n"
            f"Missing from train: {missing_from_train}"
        )

    print(f"Found {len(train_classes)} car classes.")

    if OUTPUT.exists():
        print(f"Removing old dataset: {OUTPUT}")
        shutil.rmtree(OUTPUT)

    for split in ("train", "val", "test"):
        (OUTPUT / split).mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_validation = 0
    total_test = 0

    for class_number, class_name in enumerate(
        train_classes,
        start=1,
    ):
        original_train_folder = TRAIN_SOURCE / class_name
        original_test_folder = TEST_SOURCE / class_name

        training_images = get_images(original_train_folder)
        testing_images = get_images(original_test_folder)

        random.shuffle(training_images)

        validation_count = max(
            1,
            round(len(training_images) * VALIDATION_PERCENT),
        )

        validation_images = training_images[:validation_count]
        final_training_images = training_images[validation_count:]

        copy_images(
            final_training_images,
            OUTPUT / "train" / class_name,
        )

        copy_images(
            validation_images,
            OUTPUT / "val" / class_name,
        )

        copy_images(
            testing_images,
            OUTPUT / "test" / class_name,
        )

        total_train += len(final_training_images)
        total_validation += len(validation_images)
        total_test += len(testing_images)

        print(
            f"[{class_number:03d}/{len(train_classes)}] "
            f"{class_name}: "
            f"{len(final_training_images)} train, "
            f"{len(validation_images)} val, "
            f"{len(testing_images)} test"
        )

    print("\nDataset preparation complete.")
    print(f"Classes: {len(train_classes)}")
    print(f"Training images: {total_train}")
    print(f"Validation images: {total_validation}")
    print(f"Testing images: {total_test}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
