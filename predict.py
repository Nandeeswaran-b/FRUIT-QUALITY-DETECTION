"""Prediction helpers and optional command-line image inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from project_utils import (
    classify_array,
    image_to_array,
    load_model,
    project_path,
    read_label_mapping,
)

PROJECT_DIR = Path(__file__).resolve().parent


def predict_image(image: Image.Image, model, mapping: dict, threshold: float = 0.6) -> dict:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    result = classify_array(model, image_to_array(image), mapping)
    result["uncertain"] = result["confidence"] < threshold
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input JPG, JPEG, or PNG image")
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / "models/fruit_quality.keras")
    parser.add_argument(
        "--mapping", type=Path, default=PROJECT_DIR / "models/label_mapping.json"
    )
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    if args.image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        parser.error("Image must be JPG, JPEG, or PNG.")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")
    model_path = project_path(args.model)
    mapping_path = project_path(args.mapping)
    try:
        with Image.open(args.image) as opened:
            image = opened.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        parser.error(f"Could not open image {args.image}: {error}")
    model = load_model(model_path)
    mapping = read_label_mapping(mapping_path)
    result = predict_image(image, model, mapping, args.threshold)
    label = result["class"]
    print(
        f"Prediction: {label['fruit']} — {label['quality']} "
        f"(score {result['confidence']:.1%})"
    )
    if result["uncertain"]:
        print("Uncertain—try a clearer photo.")
    print("Class probabilities:")
    for item in result["probabilities"]:
        print(f"  {item['display_name']}: {item['probability']:.1%}")
    print("Confidence is a model score, not a guarantee or a food-safety assessment.")


if __name__ == "__main__":
    main()
