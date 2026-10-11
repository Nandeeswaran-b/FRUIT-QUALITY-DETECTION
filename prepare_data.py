"""Validate the raw images and create duplicate-aware stratified splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold

from project_utils import CLASS_FOLDERS, project_path

PROJECT_DIR = Path(__file__).resolve().parent


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        root_a, root_b = self.find(first), self.find(second)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def difference_hash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def hash_bands(value: int) -> tuple[int, ...]:
    # Five disjoint bands ensure hashes at Hamming distance <= 4 share a band.
    widths = (13, 13, 13, 13, 12)
    bands = []
    offset = 64
    for width in widths:
        offset -= width
        bands.append((value >> offset) & ((1 << width) - 1))
    return tuple(bands)


def _has_all_classes(directory: Path) -> bool:
    return all((directory / folder).is_dir() for folder in CLASS_FOLDERS)


def discover_layout(source_dir: Path) -> tuple[dict[str, Path], bool]:
    """Find one source tree, preferring a supplied train/test split over flat classes."""
    split_layouts = []
    for train_dir in source_dir.rglob("train"):
        if not train_dir.is_dir():
            continue
        test_dir = train_dir.parent / "test"
        if _has_all_classes(train_dir) and _has_all_classes(test_dir):
            depth = len(train_dir.relative_to(source_dir).parts)
            split_layouts.append((depth, train_dir, test_dir))
    if split_layouts:
        _, train_dir, test_dir = min(split_layouts, key=lambda item: item[0])
        return {"train": train_dir, "test": test_dir}, True

    flat_layouts = []
    if _has_all_classes(source_dir):
        flat_layouts.append((0, source_dir))
    else:
        for candidate in source_dir.rglob("*"):
            if candidate.is_dir() and _has_all_classes(candidate):
                flat_layouts.append(
                    (len(candidate.relative_to(source_dir).parts), candidate)
                )
    if flat_layouts:
        _, class_root = min(flat_layouts, key=lambda item: item[0])
        return {"all": class_root}, False

    expected_paths = [
        f"{split}\\{folder}"
        for split in ("train", "test")
        for folder in CLASS_FOLDERS
    ]
    raise FileNotFoundError(
        f"No complete six-class layout was found under {source_dir}. "
        "Expected either train/<class>/ and test/<class>/ folders or six "
        "class folders under a common directory. Expected class names: "
        + ", ".join(CLASS_FOLDERS)
        + f". Example split paths: {', '.join(expected_paths[:3])}, ..."
    )


def scan_images(
    source_dir: Path,
) -> tuple[list[dict], list[dict], bool]:
    try:
        source_dir.relative_to(PROJECT_DIR)
    except ValueError as error:
        raise ValueError(
            f"Dataset source must be inside the project folder so manifest paths "
            f"remain portable: {source_dir}"
        ) from error
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {source_dir}")

    layout, has_source_split = discover_layout(source_dir)

    records: list[dict] = []
    corrupted: list[dict] = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for split_name, class_root in layout.items():
        for label, folder in enumerate(CLASS_FOLDERS):
            paths = sorted(
                path
                for path in (class_root / folder).rglob("*")
                if path.is_file() and path.suffix.lower() in extensions
            )
            for path in paths:
                try:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        visual_hash = difference_hash(image)
                    records.append(
                        {
                            "path": path.relative_to(PROJECT_DIR).as_posix(),
                            "label": label,
                            "class_name": folder,
                            "sha256": digest,
                            "visual_hash": visual_hash,
                            "source_split": None if split_name == "all" else split_name,
                        }
                    )
                except (
                    OSError,
                    ValueError,
                    UnidentifiedImageError,
                    Image.DecompressionBombError,
                ) as error:
                    corrupted.append(
                        {
                            "path": path.relative_to(PROJECT_DIR).as_posix(),
                            "source_split": None if split_name == "all" else split_name,
                            "error": str(error),
                        }
                    )
    empty = [
        name
        for name in CLASS_FOLDERS
        if not any(r["class_name"] == name for r in records)
    ]
    if empty:
        raise ValueError(
            "No valid supported images found for: "
            + ", ".join(empty)
            + ". See README.md for the required folder layout."
        )
    empty_source_splits = [
        split
        for split in ("train", "test")
        if has_source_split
        and not any(record["source_split"] == split for record in records)
    ]
    if empty_source_splits:
        raise ValueError(
            "No valid supported images found in source split(s): "
            + ", ".join(empty_source_splits)
        )
    return records, corrupted, has_source_split


def group_related_images(records: list[dict]) -> tuple[list[int], int]:
    union_find = UnionFind(len(records))
    exact_seen: dict[str, int] = {}
    candidates: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    near_duplicate_pairs = 0

    for index, record in enumerate(records):
        digest = record["sha256"]
        if digest in exact_seen:
            union_find.union(index, exact_seen[digest])
            continue
        exact_seen[digest] = index

        value = record["visual_hash"]
        nearby: set[int] = set()
        for band_index, band_value in enumerate(hash_bands(value)):
            nearby.update(
                candidates[(record["label"], band_index, band_value)]
            )
        for previous in nearby:
            other = records[previous]["visual_hash"]
            if (
                record["sha256"] != records[previous]["sha256"]
                and (value ^ other).bit_count() <= 4
            ):
                union_find.union(index, previous)
                near_duplicate_pairs += 1
        for band_index, band_value in enumerate(hash_bands(value)):
            candidates[(record["label"], band_index, band_value)].append(index)

    roots = [union_find.find(index) for index in range(len(records))]
    groups = {root: group_index for group_index, root in enumerate(sorted(set(roots)))}
    return [groups[root] for root in roots], near_duplicate_pairs


def create_splits(
    records: list[dict], groups: list[int], seed: int, has_source_split: bool
) -> tuple[pd.DataFrame, list[int]]:
    if has_source_split:
        test_groups = {
            groups[index]
            for index, record in enumerate(records)
            if record["source_split"] == "test"
        }
        excluded = [
            index
            for index, record in enumerate(records)
            if record["source_split"] == "train" and groups[index] in test_groups
        ]
        excluded_set = set(excluded)
        pool = [
            index
            for index, record in enumerate(records)
            if record["source_split"] == "train" and index not in excluded_set
        ]
        test_indexes = [
            index
            for index, record in enumerate(records)
            if record["source_split"] == "test"
        ]
        if not pool or not test_indexes:
            raise ValueError("Source train and test splits must both contain valid images.")

        labels = np.asarray([records[index]["label"] for index in pool])
        group_array = np.asarray([groups[index] for index in pool])
        if len(set(group_array)) < 5:
            raise ValueError(
                "Fewer than five independent image groups remain in the supplied "
                "training split after excluding duplicates of test images."
            )
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        candidates = []
        for _, validation_positions in splitter.split(
            np.zeros(len(pool)), labels, group_array
        ):
            validation_labels = set(labels[validation_positions].tolist())
            train_positions = np.setdiff1d(
                np.arange(len(pool)), validation_positions, assume_unique=True
            )
            training_labels = set(labels[train_positions].tolist())
            if (
                validation_labels == set(range(len(CLASS_FOLDERS)))
                and training_labels == set(range(len(CLASS_FOLDERS)))
            ):
                candidates.append(
                    (
                        abs(len(validation_positions) / len(pool) - 0.20),
                        train_positions,
                        validation_positions,
                    )
                )
        if not candidates:
            raise ValueError(
                "Could not create a group-separated validation split containing all "
                "six classes from the supplied training images. Add independent "
                "training images or inspect duplicate grouping."
            )
        _, train_positions, validation_positions = min(
            candidates, key=lambda candidate: candidate[0]
        )
        split_by_index = {
            pool[position]: "train" for position in train_positions
        }
        split_by_index.update(
            {pool[position]: "validation" for position in validation_positions}
        )
        split_by_index.update({index: "test" for index in test_indexes})
        rows = [
            {
                "path": record["path"],
                "class_name": record["class_name"],
                "label": record["label"],
                "group": groups[index],
                "split": split_by_index[index],
                "source_split": record["source_split"],
            }
            for index, record in enumerate(records)
            if index in split_by_index
        ]
        return pd.DataFrame(rows), excluded

    group_class_counts: dict[int, set[int]] = defaultdict(set)
    for record, group in zip(records, groups):
        group_class_counts[group].add(record["label"])
    for label, class_name in enumerate(CLASS_FOLDERS):
        count = sum(label in labels for labels in group_class_counts.values())
        if count < 20:
            raise ValueError(
                f"{class_name} has only {count} independent image groups; at least 20 "
                "are needed to create 20-fold duplicate-aware stratified splits. "
                "Add more images or manually review related-image grouping."
            )

    labels = np.asarray([record["label"] for record in records])
    group_array = np.asarray(groups)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold_numbers = np.full(len(records), -1, dtype=np.int32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for fold, (_, held_out) in enumerate(
            splitter.split(np.zeros(len(records)), labels, group_array)
        ):
            fold_numbers[held_out] = fold
    for warning in caught:
        print(f"Split warning: {warning.message}")
    if np.any(fold_numbers < 0):
        raise RuntimeError("Some images were not assigned to a split.")

    fold_to_split = {
        **{fold: "train" for fold in range(14)},
        **{fold: "validation" for fold in range(14, 17)},
        **{fold: "test" for fold in range(17, 20)},
    }
    for split in ("train", "validation", "test"):
        split_labels = {
            int(labels[index])
            for index, fold in enumerate(fold_numbers)
            if fold_to_split[int(fold)] == split
        }
        if split_labels != set(range(len(CLASS_FOLDERS))):
            missing_classes = [
                CLASS_FOLDERS[label]
                for label in range(len(CLASS_FOLDERS))
                if label not in split_labels
            ]
            raise ValueError(
                f"The {split} split is missing classes {missing_classes}. "
                "Collect more independent examples or review related-image grouping."
            )

    rows = []
    for index, record in enumerate(records):
        fold = int(fold_numbers[index])
        split = "train" if fold < 14 else "validation" if fold < 17 else "test"
        rows.append(
            {
                "path": record["path"],
                "class_name": record["class_name"],
                "label": record["label"],
                "group": groups[index],
                "split": split,
            }
        )
    return pd.DataFrame(rows), []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = next(
        (
            path
            for path in (
                PROJECT_DIR / "data" / "downloaded",
                PROJECT_DIR / "dataset",
                PROJECT_DIR / "data" / "raw",
            )
            if path.is_dir()
        ),
        PROJECT_DIR / "data" / "downloaded",
    )
    parser.add_argument(
        "--source-dir",
        "--raw-dir",
        dest="source_dir",
        type=Path,
        default=default_source,
        help="Dataset root; supports class folders or existing train/test folders.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_DIR / "data" / "processed" / "splits.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_DIR / "metrics" / "data_quality.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.source_dir = project_path(args.source_dir)
    args.manifest = project_path(args.manifest)
    args.report = project_path(args.report)
    try:
        records, corrupted, has_source_split = scan_images(args.source_dir)
        groups, near_duplicate_pairs = group_related_images(records)
        manifest, excluded_train_duplicates = create_splits(
            records, groups, args.seed, has_source_split
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)

    before_counts = Counter(record["class_name"] for record in records)
    before = {name: before_counts.get(name, 0) for name in CLASS_FOLDERS}
    after = {
        split: {
            name: int((manifest.loc[manifest["split"] == split, "class_name"] == name).sum())
            for name in CLASS_FOLDERS
        }
        for split in ("train", "validation", "test")
    }
    group_sizes = Counter(groups)
    exact_duplicate_sizes = Counter(record["sha256"] for record in records)
    group_labels: dict[int, set[int]] = defaultdict(set)
    for record, group in zip(records, groups):
        group_labels[group].add(record["label"])
    duplicate_label_conflicts = sum(
        len(group_labels[group]) > 1
        for group, size in group_sizes.items()
        if size > 1
    )
    split_counts = Counter(manifest["split"])
    report = {
        "dataset_source": str(args.source_dir),
        "preserved_source_test_split": has_source_split,
        "valid_images": len(records),
        "images_excluded_from_training_due_to_test_duplicates": len(
            excluded_train_duplicates
        ),
        "corrupted_images": corrupted,
        "exact_duplicate_groups": sum(size > 1 for size in exact_duplicate_sizes.values()),
        "exact_duplicate_extra_images": sum(
            size - 1 for size in exact_duplicate_sizes.values() if size > 1
        ),
        "near_duplicate_pairs_detected": near_duplicate_pairs,
        "unique_related_image_groups": len(group_sizes),
        "related_groups_with_multiple_class_labels": duplicate_label_conflicts,
        "class_counts_before_split": dict(before),
        "class_imbalance_ratio_max_to_min": max(before.values()) / min(before.values()),
        "class_counts_by_split": after,
        "split_image_counts": {
            name: int(split_counts.get(name, 0)) for name in ("train", "validation", "test")
        },
        "split_image_percentages": {
            name: round(100 * split_counts.get(name, 0) / len(records), 2)
            for name in ("train", "validation", "test")
        },
        "split_group_counts": {
            split: int(manifest.loc[manifest["split"] == split, "group"].nunique())
            for split in ("train", "validation", "test")
        },
        "split_grouping_note": (
            "Exact SHA-256 duplicates are grouped across all classes. dHash "
            "near-duplicates (Hamming distance <= 4) are grouped within the same "
            "class. When a group overlaps the supplied test split, its training "
            "copies are excluded. Near-duplicate matching is a heuristic."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Valid images: {len(records)}; corrupted images: {len(corrupted)}")
    print("Image counts by class and resulting split:")
    count_table = manifest.groupby(["class_name", "split"]).size().unstack(fill_value=0)
    for split in ("train", "validation", "test"):
        if split not in count_table.columns:
            count_table[split] = 0
    count_table = count_table[["train", "validation", "test"]]
    for class_name in CLASS_FOLDERS:
        counts = count_table.loc[class_name]
        print(
            f"  {class_name}: train={int(counts['train'])}, "
            f"validation={int(counts['validation'])}, test={int(counts['test'])}"
        )
    print(f"Per-class source-image counts: {dict(before)}")
    print(f"Class imbalance ratio (max/min): {report['class_imbalance_ratio_max_to_min']:.2f}")
    print(f"Near-duplicate pairs detected: {near_duplicate_pairs}")
    print(
        "Split totals: "
        + ", ".join(
            f"{split}={report['split_image_counts'][split]} "
            f"({report['split_image_percentages'][split]}%)"
            for split in ("train", "validation", "test")
        )
    )
    if has_source_split:
        print("Preserved the source test set; validation was split from source training images.")
    if excluded_train_duplicates:
        print(
            "Excluded training images whose duplicate/near-duplicate group occurs "
            f"in source test: {len(excluded_train_duplicates)}"
        )
    if duplicate_label_conflicts:
        print(
            "Warning: related-image groups have conflicting class labels; "
            "review related_groups_with_multiple_class_labels in the data-quality report."
        )
    print(f"Saved split manifest: {args.manifest}")
    print(f"Saved data-quality report: {args.report}")
    if corrupted:
        print("Corrupted files are listed in the report and excluded from the splits.")


if __name__ == "__main__":
    main()
