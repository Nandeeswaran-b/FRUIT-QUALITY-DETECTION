"""Evaluate the trained model once on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from project_utils import (
    CLASS_FOLDERS,
    class_details,
    load_model,
    project_path,
    read_label_mapping,
)

PROJECT_DIR = Path(__file__).resolve().parent


def make_test_dataset(frame: pd.DataFrame, batch_size: int = 16):
    import tensorflow as tf

    paths = [str((PROJECT_DIR / path).resolve()) for path in frame["path"]]
    dataset = tf.data.Dataset.from_tensor_slices(paths)

    def load_image(path):
        encoded = tf.io.read_file(path)
        image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, (224, 224), antialias=True)
        return tf.cast(image, tf.float32)

    return (
        dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def save_training_curves(history_path: Path, plots_dir: Path) -> None:
    if not history_path.is_file():
        print(f"Training history not found; skipping learning curves: {history_path}")
        return
    history = pd.read_csv(history_path)
    required = {"epoch", "accuracy", "val_accuracy", "loss", "val_loss"}
    if not required.issubset(history.columns):
        print(f"Training history has no expected metric columns: {history_path}")
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["accuracy"], label="Train")
    axes[0].plot(history["epoch"], history["val_accuracy"], label="Validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].plot(history["epoch"], history["loss"], label="Train")
    axes[1].plot(history["epoch"], history["val_loss"], label="Validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(plots_dir / "training_curves.png", dpi=150)
    plt.close(figure)


def save_misclassified(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    mapping: dict,
    path: Path,
    limit: int = 12,
) -> None:
    incorrect = np.flatnonzero(y_true != y_pred)[:limit]
    columns = 4
    rows = max(1, int(np.ceil(len(incorrect) / columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 3.6 * rows))
    axes = np.asarray(axes).reshape(-1)
    if len(incorrect) == 0:
        axes[0].text(0.5, 0.5, "No incorrect predictions in this test set.", ha="center")
        for axis in axes[1:]:
            axis.axis("off")
    else:
        for axis, index in zip(axes, incorrect):
            image_path = PROJECT_DIR / frame.iloc[index]["path"]
            with Image.open(image_path) as opened:
                axis.imshow(opened.convert("RGB"))
            actual = mapping["classes"][int(y_true[index])]["display_name"]
            predicted = mapping["classes"][int(y_pred[index])]["display_name"]
            score = float(probabilities[index, y_pred[index]])
            axis.set_title(f"True: {actual}\nPred: {predicted} ({score:.1%})", fontsize=9)
            axis.axis("off")
        for axis in axes[len(incorrect):]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data/processed/splits.csv")
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / "models/fruit_quality.keras")
    parser.add_argument(
        "--mapping", type=Path, default=PROJECT_DIR / "models/label_mapping.json"
    )
    parser.add_argument("--metrics-dir", type=Path, default=PROJECT_DIR / "metrics")
    parser.add_argument("--plots-dir", type=Path, default=PROJECT_DIR / "plots")
    args = parser.parse_args()
    args.manifest = project_path(args.manifest)
    args.model = project_path(args.model)
    args.mapping = project_path(args.mapping)
    args.metrics_dir = project_path(args.metrics_dir)
    args.plots_dir = project_path(args.plots_dir)
    if not args.manifest.is_file():
        parser.error(f"Split manifest not found: {args.manifest}. Run prepare_data.py first.")
    frame = pd.read_csv(args.manifest)
    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)
    if test_frame.empty:
        parser.error("The split manifest contains no held-out test images.")
    if set(test_frame["class_name"]) != set(CLASS_FOLDERS):
        parser.error("Test split does not contain all six configured classes.")
    mapping = read_label_mapping(args.mapping)
    model = load_model(args.model)

    # A single forward pass supplies predictions and all test metrics.
    test_dataset = make_test_dataset(test_frame)
    probabilities = model.predict(test_dataset, verbose=1)
    y_true = test_frame["label"].to_numpy(dtype=np.int32)
    y_pred = probabilities.argmax(axis=1)
    labels = list(range(len(CLASS_FOLDERS)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[class_details(name)["display_name"] for name in CLASS_FOLDERS],
        output_dict=True,
        zero_division=0,
    )
    results = {
        "test_image_count": int(len(test_frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            name: report[class_details(name)["display_name"]]
            for name in CLASS_FOLDERS
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": [
            class_details(name)["display_name"] for name in CLASS_FOLDERS
        ],
        "evaluation_note": "Held-out test set evaluated in one model prediction pass.",
    }
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    (args.metrics_dir / "evaluation.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    predictions = test_frame[["path", "class_name"]].copy()
    predictions["true_class"] = [CLASS_FOLDERS[label] for label in y_true]
    predictions["predicted_class"] = [CLASS_FOLDERS[label] for label in y_pred]
    predictions["confidence"] = probabilities.max(axis=1)
    predictions.to_csv(args.metrics_dir / "test_predictions.csv", index=False)

    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=labels,
        yticks=labels,
        xticklabels=results["confusion_matrix_labels"],
        yticklabels=results["confusion_matrix_labels"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(args.plots_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    save_training_curves(args.metrics_dir / "training_history.csv", args.plots_dir)
    save_misclassified(
        test_frame, y_true, y_pred, probabilities, mapping,
        args.plots_dir / "misclassified_examples.png"
    )
    print(json.dumps(results, indent=2))
    print(f"Metrics: {args.metrics_dir / 'evaluation.json'}")
    print(f"Plots: {args.plots_dir}")


if __name__ == "__main__":
    main()
