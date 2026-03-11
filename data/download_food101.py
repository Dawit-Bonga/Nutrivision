"""Download Food-101 and generate reproducible split manifests.

This script keeps the original dataset under ``data/raw/food-101`` and writes
JSON metadata files under ``data/processed``. The training code can later read
the manifests instead of depending on ad hoc directory scans.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from torchvision.datasets import Food101


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse CLI options for dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Download Food-101 and create train/val/test manifests."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Project data directory. Defaults to ./data.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Fraction of the official training split reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for deterministic validation splitting.",
    )
    return parser.parse_args()


def ensure_directories(data_dir: Path) -> tuple[Path, Path, Path]:
    """Create the raw and processed directory structure if it does not exist."""
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    splits_dir = processed_dir / "splits"

    raw_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir, splits_dir


def ensure_dataset(raw_dir: Path) -> Path:
    """Download Food-101 if needed and return the extracted dataset root."""
    Food101(root=str(raw_dir), split="train", download=True)
    dataset_root = raw_dir / "food-101"
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Expected Food-101 at {dataset_root}, but it was not found."
        )
    return dataset_root


def load_split_metadata(dataset_root: Path) -> tuple[list[str], list[str], list[str]]:
    """Load class names and official train/test image stems from Food-101."""
    meta_dir = dataset_root / "meta"
    classes_path = meta_dir / "classes.txt"
    train_path = meta_dir / "train.txt"
    test_path = meta_dir / "test.txt"

    classes = read_lines(classes_path)
    train_stems = read_lines(train_path)
    test_stems = read_lines(test_path)
    return classes, train_stems, test_stems


def read_lines(path: Path) -> list[str]:
    """Return non-empty stripped lines from a text file."""
    if not path.exists():
        raise FileNotFoundError(f"Expected metadata file at {path}.")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_label_index(classes: list[str]) -> dict[str, int]:
    """Create a deterministic class-to-index mapping."""
    return {label: index for index, label in enumerate(sorted(classes))}


def build_items(
    stems: list[str], dataset_root: Path, label_to_index: dict[str, int]
) -> list[dict[str, Any]]:
    """Convert Food-101 image stems into manifest records."""
    images_dir = dataset_root / "images"
    items: list[dict[str, Any]] = []

    for stem in stems:
        label = stem.split("/", maxsplit=1)[0]
        image_path = images_dir / f"{stem}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image referenced by metadata: {image_path}")

        items.append(
            {
                "image_path": str(image_path),
                "label": label,
                "label_index": label_to_index[label],
            }
        )

    return items


def split_train_validation(
    items: list[dict[str, Any]], val_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the official training set into deterministic train/validation sets."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    shuffled_items = list(items)
    rng.shuffle(shuffled_items)

    val_count = max(1, int(len(shuffled_items) * val_ratio))
    val_items = shuffled_items[:val_count]
    train_items = shuffled_items[val_count:]

    if not train_items:
        raise ValueError("Validation split consumed the entire training set.")

    return train_items, val_items


def write_json(data: Any, path: Path) -> None:
    """Persist JSON with stable formatting for easy diffs and inspection."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def build_summary(
    train_items: list[dict[str, Any]],
    val_items: list[dict[str, Any]],
    test_items: list[dict[str, Any]],
    label_to_index: dict[str, int],
    dataset_root: Path,
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """Create a high-level summary of the prepared dataset artifacts."""
    return {
        "dataset_root": str(dataset_root),
        "num_classes": len(label_to_index),
        "train_count": len(train_items),
        "val_count": len(val_items),
        "test_count": len(test_items),
        "val_ratio": val_ratio,
        "seed": seed,
    }


def main() -> None:
    """Download Food-101 and emit reproducible split manifests."""
    args = parse_args()
    raw_dir, processed_dir, splits_dir = ensure_directories(args.data_dir)
    dataset_root = ensure_dataset(raw_dir)

    classes, train_stems, test_stems = load_split_metadata(dataset_root)
    label_to_index = build_label_index(classes)

    train_items = build_items(train_stems, dataset_root, label_to_index)
    test_items = build_items(test_stems, dataset_root, label_to_index)
    train_items, val_items = split_train_validation(
        train_items, val_ratio=args.val_ratio, seed=args.seed
    )

    write_json(train_items, splits_dir / "train.json")
    write_json(val_items, splits_dir / "val.json")
    write_json(test_items, splits_dir / "test.json")
    write_json(label_to_index, processed_dir / "labels.json")
    write_json(
        build_summary(
            train_items=train_items,
            val_items=val_items,
            test_items=test_items,
            label_to_index=label_to_index,
            dataset_root=dataset_root,
            val_ratio=args.val_ratio,
            seed=args.seed,
        ),
        processed_dir / "dataset_summary.json",
    )

    print("Prepared Food-101 dataset manifests.")
    print(f"Dataset root: {dataset_root}")
    print(f"Train items: {len(train_items)}")
    print(f"Validation items: {len(val_items)}")
    print(f"Test items: {len(test_items)}")
    print(f"Labels written to: {processed_dir / 'labels.json'}")


if __name__ == "__main__":
    main()
