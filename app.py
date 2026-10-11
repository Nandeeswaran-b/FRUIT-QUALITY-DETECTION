"""Streamlit interface for fruit quality predictions."""

from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from predict import predict_image
from project_utils import load_model, project_path, read_label_mapping

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = project_path(PROJECT_DIR / "models" / "fruit_quality.keras")
MAPPING_PATH = project_path(PROJECT_DIR / "models" / "label_mapping.json")

st.set_page_config(page_title="Fruit Quality Classifier", page_icon="🍎")
st.title("Fruit Quality Classifier")
st.write("Upload a clear photo of an apple, banana, or orange.")
if not MODEL_PATH.is_file():
    st.warning(
        f"No trained model found at `{MODEL_PATH}`. Add the dataset to `data/raw/`, "
        "then run `python prepare_data.py` and `python train.py` from the project folder."
    )


@st.cache_resource
def get_artifacts():
    model = load_model(MODEL_PATH)
    mapping = read_label_mapping(MAPPING_PATH)
    return model, mapping


threshold = st.slider(
    "Uncertainty threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="Predictions below this model score are marked uncertain.",
)
uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded is not None:
    try:
        with Image.open(uploaded) as opened:
            image = opened.convert("RGB")
            image.load()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        st.error(f"That file could not be read as a valid image: {error}")
    else:
        st.image(image, caption="Uploaded image", use_container_width=True)
        try:
            model, mapping = get_artifacts()
            result = predict_image(image, model, mapping, threshold)
        except (FileNotFoundError, ValueError, OSError) as error:
            st.error(str(error))
        else:
            predicted = result["class"]
            st.subheader(f"{predicted['fruit']} — {predicted['quality']}")
            st.metric("Confidence score", f"{result['confidence']:.1%}")
            if result["uncertain"]:
                st.warning("Uncertain—try a clearer photo.")
            st.caption(
                "Confidence is a model score, not a guarantee. This model assesses visible "
                "appearance only; it cannot determine food safety or detect internal spoilage."
            )
            st.subheader("Probabilities for all six classes")
            for item in result["probabilities"]:
                st.progress(
                    item["probability"],
                    text=f"{item['display_name']}: {item['probability']:.1%}",
                )
else:
    st.info("Choose a JPG, JPEG, or PNG image to get a prediction.")
