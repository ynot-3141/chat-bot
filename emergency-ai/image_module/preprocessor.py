"""
preprocess.py — Emergency Image Detection Module
=================================================
Handles all image preprocessing before training:
  - Loads images from dataset/ folder structure
  - Resizes to 224x224 (MobileNetV2 requirement)
  - Normalizes pixel values from 0-255 to 0.0-1.0
  - Applies data augmentation to multiply training data
  - Splits dataset into 80% train / 20% validation
  - Returns data generators ready for train.py

Expected folder structure:
    dataset/
        fire/       <- images of fire, smoke
        medical/    <- accidents, injuries, collapsed people
        crime/      <- fights, break-ins, vandalism
        normal/     <- everyday scenes, no emergency

Usage:
    from preprocess import get_data_generators, get_class_info
    train_data, val_data = get_data_generators()
"""

import os
import numpy as np
from pathlib import Path
from importlib import import_module

# Keras/TensorFlow moved image utilities across versions.
# Import dynamically to avoid static resolver errors and keep runtime compatibility.
_keras_utils = import_module("tensorflow.keras.utils")
load_img = _keras_utils.load_img
img_to_array = _keras_utils.img_to_array

try:
    _keras_image = import_module("tensorflow.keras.legacy.preprocessing.image")
except ModuleNotFoundError:
    _keras_image = import_module("tensorflow.keras.preprocessing.image")

ImageDataGenerator = _keras_image.ImageDataGenerator

# ─────────────────────────────────────────────
# CONFIGURATION — change these if needed
# ─────────────────────────────────────────────

IMG_SIZE    = (224, 224)   # MobileNetV2 expects exactly this
BATCH_SIZE  = 32           # how many images to process at once
DATASET_DIR = "dataset/"   # root folder with class subfolders
VALID_SPLIT = 0.2          # 20% goes to validation
RANDOM_SEED = 42           # for reproducibility

# Emergency classes — must match your folder names exactly
CLASSES = ["fire", "medical", "crime", "normal"]
CLASS_NAMES = CLASSES
NUM_CLASSES = len(CLASS_NAMES)


# ─────────────────────────────────────────────
# MAIN FUNCTION — call this from train.py
# ─────────────────────────────────────────────

def get_data_generators():
    """
    Builds and returns training and validation data generators.
    Augmentation is applied ONLY to training data, not validation.

    Returns:
        train_data  — augmented generator for training (80%)
        val_data    — clean generator for validation (20%)
    """

    _verify_dataset_structure()

    # ── Training generator (with augmentation) ──────────────────
    # Each epoch, images are randomly transformed so the model
    # sees a slightly different version every time.
    # This prevents overfitting and simulates real-world variation.
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,           # normalize: 0-255 → 0.0-1.0
        rotation_range=20,           # randomly rotate up to 20 degrees
        width_shift_range=0.1,       # shift image left/right by 10%
        height_shift_range=0.1,      # shift image up/down by 10%
        shear_range=0.1,             # skew perspective slightly
        zoom_range=0.1,              # random zoom in/out
        brightness_range=[0.7, 1.3], # vary brightness (critical for fire/night scenes)
        horizontal_flip=True,        # randomly mirror images
        fill_mode="nearest",         # fill gaps after rotation with nearest pixel
        validation_split=VALID_SPLIT
    )

    # ── Validation generator (NO augmentation) ──────────────────
    # Validation images must be clean and unmodified so the
    # accuracy score you see is an honest measure of real performance.
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALID_SPLIT
    )

    # ── Load training images ─────────────────────────────────────
    train_data = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,        # resize all images to 224x224
        batch_size=BATCH_SIZE,
        class_mode="categorical",    # one-hot labels e.g. [1,0,0,0] for fire
        subset="training",           # use the 80%
        seed=RANDOM_SEED,
        shuffle=True,
        classes=CLASSES              # enforce consistent class ordering
    )

    # ── Load validation images ───────────────────────────────────
    val_data = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",         # use the 20%
        seed=RANDOM_SEED,
        shuffle=False,               # never shuffle validation
        classes=CLASSES
    )

    _print_summary(train_data, val_data)

    return train_data, val_data


# ─────────────────────────────────────────────
# SINGLE IMAGE PREPROCESSOR
# Used by image_analysis.py at inference time
# ─────────────────────────────────────────────

def preprocess_single_image(image_path):
    """
    Preprocesses one image file for prediction (not training).
    Only resizes + normalizes — no augmentation applied.

    Args:
        image_path: path to the image file (str or Path)

    Returns:
        numpy array of shape (1, 224, 224, 3) ready for model.predict()
    """
    img       = load_img(image_path, target_size=IMG_SIZE, color_mode="rgb")
    img_array = img_to_array(img)               # shape: (224, 224, 3)
    img_array = img_array / 255.0               # normalize to 0.0-1.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim → (1, 224, 224, 3)
    return img_array


def preprocess_image_bytes(image_bytes):
    """
    Preprocesses an image received as raw bytes from an API upload.
    Used by api.py when the frontend sends an image file over HTTP.

    Args:
        image_bytes: raw bytes from FastAPI UploadFile.read()

    Returns:
        numpy array of shape (1, 224, 224, 3)
    """
    import io
    from PIL import Image

    img       = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img       = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def get_class_info():
    """
    Returns class name <-> index mappings.
    Used by image_analysis.py to convert prediction index back to a label.

    Returns:
        class_to_idx: {"fire": 0, "medical": 1, "crime": 2, "normal": 3}
        idx_to_class: {0: "fire", 1: "medical", 2: "crime", 3: "normal"}
    """
    class_to_idx = {name: idx for idx, name in enumerate(CLASSES)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def get_dataset_stats():
    """
    Counts how many images are in each class folder.
    Useful to spot class imbalance before training — if one class
    has 400 images and another has 50, your model will be biased.

    Returns:
        dict e.g. {"fire": 312, "medical": 280, "crime": 190, "normal": 400}
    """
    stats        = {}
    dataset_path = Path(DATASET_DIR)
    valid_exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_name in CLASSES:
        class_dir = dataset_path / class_name
        if class_dir.exists():
            count = len([f for f in class_dir.iterdir() if f.suffix.lower() in valid_exts])
            stats[class_name] = count
        else:
            stats[class_name] = 0

    return stats


def validate_dataset():
    """
    Backward-compatible validator used by older test code.
    Returns True when dataset is valid, otherwise False.
    """
    try:
        _verify_dataset_structure()
        return True
    except (FileNotFoundError, ValueError):
        return False


def _verify_dataset_structure():
    """
    Checks that all required folders exist and contain images.
    Raises a clear, readable error instead of a cryptic TensorFlow crash.
    """
    dataset_path = Path(DATASET_DIR)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"\n[preprocess.py] ERROR: Dataset folder '{DATASET_DIR}' not found.\n"
            f"Create it and add subfolders: {CLASSES}\n"
            f"Example structure:\n"
            f"  dataset/fire/      <- add fire images here\n"
            f"  dataset/medical/   <- add medical emergency images\n"
            f"  dataset/crime/     <- add crime scene images\n"
            f"  dataset/normal/    <- add normal everyday images\n"
        )

    missing = []
    empty   = []

    for class_name in CLASSES:
        class_dir  = dataset_path / class_name
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if not class_dir.exists():
            missing.append(class_name)
        else:
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in valid_exts]
            if len(images) < 10:
                empty.append(f"{class_name} ({len(images)} images — need at least 10)")

    if missing:
        raise FileNotFoundError(
            f"\n[preprocess.py] ERROR: Missing class folders: {missing}\n"
            f"All four folders must exist inside '{DATASET_DIR}'"
        )

    if empty:
        raise ValueError(
            f"\n[preprocess.py] ERROR: These folders have too few images:\n"
            + "\n".join(f"  - {e}" for e in empty)
            + f"\nAdd more images before training."
        )


def _print_summary(train_data, val_data):
    """Prints a clean summary so you can verify everything loaded correctly."""
    stats = get_dataset_stats()
    total = sum(stats.values())

    print("\n" + "=" * 52)
    print("  PREPROCESSING COMPLETE")
    print("=" * 52)
    print(f"  Dataset path   : {DATASET_DIR}")
    print(f"  Image size     : {IMG_SIZE[0]} x {IMG_SIZE[1]}")
    print(f"  Batch size     : {BATCH_SIZE}")
    print(f"  Total images   : {total}")
    print(f"  Train samples  : {train_data.samples}  (80%)")
    print(f"  Val samples    : {val_data.samples}  (20%)")
    print(f"\n  Images per class:")
    for cls, count in stats.items():
        bar = "█" * min(count // 10, 30)
        print(f"    {cls:<10} {count:>4}  {bar}")
    print(f"\n  Class → index  : {train_data.class_indices}")
    print("=" * 52 + "\n")


# ─────────────────────────────────────────────
# QUICK TEST — run directly to verify setup
# python preprocess.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("  preprocess.py — verification mode")
    print("=" * 52)

    stats = get_dataset_stats()
    print("\nImage counts per class:")
    for cls, count in stats.items():
        status = "OK" if count >= 10 else "NEED MORE IMAGES"
        print(f"  {cls:<10} {count:>4} images  [{status}]")

    total = sum(stats.values())

    if total == 0:
        print("\n  No images found yet.")
        print("  Add images to dataset/fire/, dataset/medical/,")
        print("  dataset/crime/, dataset/normal/ and run again.\n")
    else:
        print(f"\n  Total: {total} images across {len(CLASSES)} classes")
        print("\n  Building data generators...")
        train_data, val_data = get_data_generators()
        print("  All good! You can now run: python train.py\n")