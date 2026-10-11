"""Shared labels, preprocessing, and model-loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CLASS_FOLDERS = (
    "freshapples",
    "rottenapples",
    "freshbanana",
    "rottenbanana",
    "freshoranges",
    "rottenoranges",
)
IMAGE_SIZE = (224, 224)


def project_path(path: Path) -> Path:
    """Resolve relative project paths from this file, not the caller's cwd."""
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent / path).resolve()


def class_details(folder: str) -> dict[str, Any]:
    quality = "Fresh" if folder.startswith("fresh") else "Rotten"
    fruit_folder = folder.removeprefix("fresh").removeprefix("rotten")
    fruit_names = {"apples": "Apple", "banana": "Banana", "oranges": "Orange"}
    if fruit_folder not in fruit_names:
        raise ValueError(f"Unknown fruit class folder: {folder}")
    fruit = fruit_names[fruit_folder]
    return {
        "folder": folder,
        "fruit": fruit,
        "quality": quality,
        "display_name": f"{fruit} — {quality}",
    }


def make_label_mapping() -> dict[str, Any]:
    classes = []
    for index, folder in enumerate(CLASS_FOLDERS):
        details = class_details(folder)
        classes.append({"index": index, **details})
    return {
        "class_names": list(CLASS_FOLDERS),
        "classes": classes,
        "image_size": list(IMAGE_SIZE),
        "preprocessing": "RGB float pixels rescaled with x / 127.5 - 1",
    }


def write_label_mapping(path: Path) -> dict[str, Any]:
    mapping = make_label_mapping()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    return mapping


def read_label_mapping(path: Path) -> dict[str, Any]:
    path = project_path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Label mapping not found at {path}. Train the model first."
        )
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping.get("class_names") != list(CLASS_FOLDERS):
        raise ValueError(f"Unsupported or inconsistent class mapping in {path}.")
    expected_classes = make_label_mapping()["classes"]
    if mapping.get("classes") != expected_classes:
        raise ValueError(f"Invalid class mapping in {path}.")
    return mapping


def load_model(path: Path):
    path = project_path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run train.py first."
        )
    import tensorflow as tf

    model = tf.keras.models.load_model(path)
    if model.input_shape[1:] != (*IMAGE_SIZE, 3):
        raise ValueError(
            f"Model at {path} expects input shape {model.input_shape}; "
            f"the app requires (None, {IMAGE_SIZE[0]}, {IMAGE_SIZE[1]}, 3)."
        )
    if model.output_shape[-1] != len(CLASS_FOLDERS):
        raise ValueError(
            f"Model at {path} has {model.output_shape[-1]} outputs; "
            f"the app requires {len(CLASS_FOLDERS)} classes."
        )
    return model


def image_to_array(image: Image.Image) -> np.ndarray:
    """Resize RGB pixels with the same TensorFlow operation used by training."""
    import tensorflow as tf

    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    resized = resize_image_tensor(tf.convert_to_tensor(pixels), tf)
    return resized.numpy()


def resize_image_tensor(image: Any, tf: Any) -> Any:
    """Resize decoded RGB pixels identically for training and inference."""
    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear",
        antialias=True,
    )
    return tf.cast(image, tf.float32)


def classify_array(
    model: Any, image_array: np.ndarray, mapping: dict[str, Any]
) -> dict[str, Any]:
    probabilities = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
    if len(probabilities) != len(mapping["classes"]):
        raise ValueError("Model output does not match the saved class mapping.")
    predicted_index = int(np.argmax(probabilities))
    predicted_class = mapping["classes"][predicted_index]
    return {
        "class": predicted_class,
        "confidence": float(probabilities[predicted_index]),
        "probabilities": [
            {
                **mapping["classes"][index],
                "probability": float(probability),
            }
            for index, probability in enumerate(probabilities)
        ],
    }
