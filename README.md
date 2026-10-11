# Fruit Quality Classifier

A TensorFlow/Keras transfer-learning project that predicts one of six visible
fruit categories: fresh or rotten apples, bananas, and oranges. The Streamlit
interface displays the predicted fruit, quality, confidence score, and all six
class probabilities.

## Project structure

```text
fruit-quality-classifier/
├── app.py
├── evaluate.py
├── predict.py
├── prepare_data.py
├── train.py
├── project_utils.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── freshapples/
│   │   ├── rottenapples/
│   │   ├── freshbanana/
│   │   ├── rottenbanana/
│   │   ├── freshoranges/
│   │   └── rottenoranges/
│   └── processed/             # created by prepare_data.py
├── models/                    # created by train.py
├── metrics/                   # created by prepare_data.py/train.py/evaluate.py
└── plots/                     # created by evaluate.py
```

## Dataset

This project is designed for the publicly listed Kaggle dataset **Fruits Fresh
and Rotten for Classification**:

- Dataset page: <https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification>
- Kaggle API reference: <https://github.com/Kaggle/kaggle-api>

The dataset is not bundled with this project. Kaggle may require an account,
acceptance of dataset terms, and API credentials. No training results are
included or implied.

1. Install and configure the Kaggle CLI for your account according to Kaggle's
   API instructions.
2. From the project root, download and extract the dataset:

   ```powershell
   python -m pip install kaggle
   New-Item -ItemType Directory -Force data\download
   kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification -p data/download --unzip
   ```

3. Inspect the extracted archive. Dataset releases can differ in their nested
   layout. Combine the images for each class into the six exact directories
   below. If the download already has train/test directories, combine those
   directories before splitting; the preparation script will create new
   duplicate-aware splits across all images.

   ```text
   data/raw/freshapples/
   data/raw/rottenapples/
   data/raw/freshbanana/
   data/raw/rottenbanana/
   data/raw/freshoranges/
   data/raw/rottenoranges/
   ```

Keep original image files intact; do not resize, recompress, or augment them
before preparation. The script searches each class folder recursively for JPG,
JPEG, PNG, and BMP images.

## Environment setup

Python 3.10 or 3.11 is recommended. Install a TensorFlow build compatible with
your Python version and platform:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow uses a CPU automatically when no supported GPU is available. GPU
support depends on the TensorFlow version, operating system, CUDA/cuDNN, and
device; for a laptop GPU with about 4 GB of VRAM, begin with batch size 16 (or
lower it to 8 if memory is insufficient). Reduce epochs or image count if
training is too slow. All training settings are command-line configurable.

## Prepare and split data

The preparation script accepts either six class folders in one directory or a
presplit dataset with `train/<class>/` and `test/<class>/` folders (also found
when those split folders are nested inside an extracted archive). Its default
source prefers `data\downloaded\`, then `dataset\`, then `data\raw\`. The
current project data is under `dataset\` and contains `train\` and `test\`
class directories; the script uses the shallowest complete split tree and
ignores the redundant nested copy.

Run preparation from PowerShell:

```powershell
Set-Location 'F:\FRUIT QUALITY CLASSIFIER'
python prepare_data.py
```

The script checks image readability, excludes corrupted files, and prints
image counts for every class and resulting split. It reports class counts,
exact duplicates, and visually similar images. Exact duplicates are
identified by SHA-256 across all classes; near duplicates are grouped within
their class using a perceptual difference hash (Hamming distance at most 4).
The near-duplicate check is a heuristic and cannot guarantee that every related
image is detected.

When source train/test folders exist, the test set is retained and validation
is selected only from source training images using duplicate-aware stratified
group folds. Training images belonging to a duplicate/near-duplicate group
that also occurs in the preserved test set are excluded to prevent leakage.
For a flat, unsplit dataset, `StratifiedGroupKFold` creates approximately
70%/15%/15% train/validation/test splits. Ratios may vary because groups are
indivisible. Review `metrics/data_quality.json` and
`data/processed/splits.csv` before training.

An alternate seed and paths can be supplied:

```powershell
python prepare_data.py --seed 123
```

## Train

Once preparation has created `data/processed/splits.csv`, run:

```powershell
Set-Location 'F:\FRUIT QUALITY CLASSIFIER'
python train.py
```

Defaults use 224×224 images, a batch size of 16, ImageNet MobileNetV2 weights,
and a reproducible seed of 42. Training augmentation (horizontal flip, mild
rotation/zoom, and contrast variation) is active only during training.
Pixel scaling is embedded in the model and is exactly `x / 127.5 - 1`, the
MobileNetV2 preprocessing convention. The classification head is trained with
the backbone frozen, then the last 30 backbone layers (excluding batch
normalization layers) are fine-tuned at a lower learning rate. Each phase uses
early stopping, best-validation-loss checkpoints, and learning-rate reduction.
The better phase checkpoint is selected using validation loss; the test split
is not consulted during training.

Useful options:

```powershell
python train.py --batch-size 8 --head-epochs 8 --fine-tune-epochs 8 --unfreeze-layers 20
python train.py --head-learning-rate 0.001 --fine-tune-learning-rate 0.00001 --patience 3 --seed 7
```

Outputs include `models/fruit_quality.keras`, `models/label_mapping.json`,
`models/training_config.json`, and `metrics/training_history.csv`. Before
exiting successfully, training reloads the saved model through the same loader
used by Streamlit and makes a one-image prediction on a validation sample; this
checks artifact loading and output shape but does not report test performance.

## Evaluate once on the held-out test set

After training is complete, run:

```powershell
python evaluate.py
```

This makes one prediction pass over the test set and reports accuracy, macro
precision, macro recall, macro F1, per-class precision/recall/F1/support, and
the confusion matrix. It saves metrics under `metrics/` and training/validation
curves, a confusion-matrix plot, and misclassified examples under `plots/`.
Test metrics are generated only when this command is run on the actual data and
trained model; none are fabricated or promised here.

## Predict one image

```powershell
python predict.py path\to\fruit.jpg --threshold 0.6
```

## Run the Streamlit app

```powershell
streamlit run app.py
```

Upload a JPG, JPEG, or PNG. Invalid images are reported without crashing the
app. Adjust the uncertainty threshold in the UI; if the top class score is
below it, the app displays **“Uncertain—try a clearer photo.”** The score is a
model confidence score, not a guarantee. The model assesses visible
appearance only; it cannot determine food safety or detect internal spoilage.
