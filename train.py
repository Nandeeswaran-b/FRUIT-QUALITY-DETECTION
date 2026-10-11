"""Train a six-class MobileNetV2 classifier."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from project_utils import (
    CLASS_FOLDERS,
    IMAGE_SIZE,
    load_model,
    project_path,
    read_label_mapping,
    resize_image_tensor,
    write_label_mapping,
)

PROJECT_DIR = Path(__file__).resolve().parent


def configure_tensorflow(seed: int):
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError) as error:
        print(f"Deterministic TensorFlow ops could not be enabled: {error}")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print(f"Could not enable GPU memory growth for {gpu.name}: {error}")
    if not tf.config.list_physical_devices("GPU"):
        print("No GPU detected; training will use CPU.")
    return tf


def make_dataset(tf, frame: pd.DataFrame, batch_size: int, training: bool, seed: int):
    paths = [str((PROJECT_DIR / path).resolve()) for path in frame["path"]]
    labels = frame["label"].to_numpy(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        encoded = tf.io.read_file(path)
        image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        return resize_image_tensor(image, tf), label

    if training:
        dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(tf, num_classes: int):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="training_augmentation",
    )
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        name="mobilenetv2",
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = augmentation(inputs)
    # Equivalent to tf.keras.applications.mobilenet_v2.preprocess_input.
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenetv2_preprocess")(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="fruit_quality_mobilenetv2")
    return model, backbone


def compile_model(model, learning_rate: float):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def train_phase(
    tf,
    model,
    train_dataset,
    validation_dataset,
    checkpoint: Path,
    epochs: int,
    learning_rate: float,
    patience: int,
):
    compile_model(model, learning_rate)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=max(1, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    return model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )


def append_history(path: Path, histories: list[tuple[str, object]]) -> None:
    rows = []
    global_epoch = 0
    for phase, history in histories:
        for local_epoch, values in enumerate(history.history["loss"], start=1):
            row = {"phase": phase, "epoch_in_phase": local_epoch, "epoch": global_epoch + 1}
            for name, series in history.history.items():
                row[name] = float(series[local_epoch - 1])
            rows.append(row)
            global_epoch += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data/processed/splits.csv")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_DIR / "models")
    parser.add_argument("--metrics-dir", type=Path, default=PROJECT_DIR / "metrics")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--unfreeze-layers", type=int, default=30)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.unfreeze_layers < 1:
        parser.error("--batch-size and --unfreeze-layers must be positive.")
    if args.head_epochs < 1 or args.fine_tune_epochs < 0:
        parser.error("--head-epochs must be positive and --fine-tune-epochs cannot be negative.")
    args.manifest = project_path(args.manifest)
    args.models_dir = project_path(args.models_dir)
    args.metrics_dir = project_path(args.metrics_dir)
    if not args.manifest.is_file():
        parser.error(f"Split manifest not found: {args.manifest}. Run prepare_data.py first.")

    try:
        tf = configure_tensorflow(args.seed)
    except ImportError as error:
        print("TensorFlow is missing. Install dependencies with: pip install -r requirements.txt")
        raise SystemExit(1) from error

    manifest = pd.read_csv(args.manifest)
    required = {"path", "class_name", "label", "split", "group"}
    if not required.issubset(manifest.columns):
        parser.error(f"Split manifest must include columns: {sorted(required)}")
    train_rows = manifest[manifest["split"] == "train"]
    validation_rows = manifest[manifest["split"] == "validation"]
    if train_rows.empty or validation_rows.empty:
        parser.error("Manifest must contain non-empty train and validation splits.")
    if set(manifest["class_name"]) != set(CLASS_FOLDERS):
        parser.error("Manifest class labels do not match the configured six classes.")
    for split_name in ("train", "validation"):
        split_labels = set(
            manifest.loc[manifest["split"] == split_name, "class_name"]
        )
        if split_labels != set(CLASS_FOLDERS):
            parser.error(
                f"{split_name} split must contain all six classes; found "
                f"{sorted(split_labels)}."
            )
    expected_labels = {name: index for index, name in enumerate(CLASS_FOLDERS)}
    actual_labels = manifest.groupby("class_name")["label"].unique().to_dict()
    if any(
        len(actual_labels.get(name, [])) != 1
        or int(actual_labels[name][0]) != expected_label
        for name, expected_label in expected_labels.items()
    ):
        parser.error("Manifest numeric labels do not match the configured class order.")

    train_dataset = make_dataset(tf, train_rows, args.batch_size, True, args.seed)
    validation_dataset = make_dataset(tf, validation_rows, args.batch_size, False, args.seed)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    model, backbone = build_model(tf, len(CLASS_FOLDERS))
    head_path = args.models_dir / "head_best.keras"
    fine_tune_path = args.models_dir / "fine_tune_best.keras"
    print("Training classification head with the backbone frozen.")
    head_history = train_phase(
        tf, model, train_dataset, validation_dataset, head_path,
        args.head_epochs, args.head_learning_rate, args.patience
    )

    fine_tune_history = None
    if args.fine_tune_epochs > 0:
        print(f"Fine-tuning the last {args.unfreeze_layers} backbone layers.")
        model = tf.keras.models.load_model(head_path)
        backbone = model.get_layer("mobilenetv2")
        backbone.trainable = True
        cutoff = max(0, len(backbone.layers) - args.unfreeze_layers)
        for index, layer in enumerate(backbone.layers):
            layer.trainable = index >= cutoff and not isinstance(
                layer, tf.keras.layers.BatchNormalization
            )
        fine_tune_history = train_phase(
            tf, model, train_dataset, validation_dataset, fine_tune_path,
            args.fine_tune_epochs, args.fine_tune_learning_rate, args.patience
        )

    candidates = [head_path]
    if fine_tune_path.is_file():
        candidates.append(fine_tune_path)
    best_path = min(
        candidates,
        key=lambda candidate: float(
            tf.keras.models.load_model(candidate).evaluate(
                validation_dataset, verbose=0, return_dict=True
            )["loss"]
        ),
    )
    best_model = tf.keras.models.load_model(best_path)
    final_model_path = args.models_dir / "fruit_quality.keras"
    best_model.save(final_model_path)
    mapping_path = args.models_dir / "label_mapping.json"
    write_label_mapping(mapping_path)
    saved_model = load_model(final_model_path)
    saved_mapping = read_label_mapping(mapping_path)
    sample_images, _ = next(iter(validation_dataset.take(1)))
    sample_probabilities = saved_model.predict(sample_images[:1], verbose=0)
    if (
        sample_probabilities.shape != (1, len(saved_mapping["classes"]))
        or not np.isfinite(sample_probabilities).all()
    ):
        raise RuntimeError(
            "Saved-model prediction verification failed: expected one finite "
            f"probability vector with {len(saved_mapping['classes'])} class scores."
        )
    args.models_dir.joinpath("training_config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "image_size": list(IMAGE_SIZE),
                "batch_size": args.batch_size,
                "head_epochs": args.head_epochs,
                "fine_tune_epochs": args.fine_tune_epochs,
                "unfreeze_layers": args.unfreeze_layers,
                "head_learning_rate": args.head_learning_rate,
                "fine_tune_learning_rate": args.fine_tune_learning_rate,
                "best_validation_checkpoint": best_path.name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    histories = [("head", head_history)]
    if fine_tune_history is not None:
        histories.append(("fine_tune", fine_tune_history))
    append_history(args.metrics_dir / "training_history.csv", histories)
    for temporary in (head_path, fine_tune_path):
        if temporary.exists():
            temporary.unlink()
    print(f"Saved best validation model to {final_model_path}")
    print(
        "Verified saved model loading and prediction output shape: "
        f"{sample_probabilities.shape}"
    )
    print("No test-set metrics were computed during training.")


if __name__ == "__main__":
    main()
