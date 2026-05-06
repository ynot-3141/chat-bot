# train.py
# Emergency Image Detection System — Model Training Module
# Architecture : MobileNetV2 (transfer learning) + custom classifier head
# Output       : model/emergency_model.h5 + model/class_indices.json

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import tensorflow as tf

# Use tf.keras namespace via tensorflow import to avoid unresolved static submodule imports.
keras = tf.keras
MobileNetV2 = keras.applications.MobileNetV2
Model = keras.models.Model
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
BatchNormalization = keras.layers.BatchNormalization
Adam = keras.optimizers.Adam
ModelCheckpoint = keras.callbacks.ModelCheckpoint
EarlyStopping = keras.callbacks.EarlyStopping
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
TensorBoard = keras.callbacks.TensorBoard

from preprocessor import get_data_generators, CLASSES, IMG_SIZE

CLASS_NAMES = CLASSES
NUM_CLASSES = len(CLASS_NAMES)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_DIR        = "model/"
MODEL_PATH       = os.path.join(MODEL_DIR, "emergency_model.h5")
CLASS_INDEX_PATH = os.path.join(MODEL_DIR, "class_indices.json")
LOG_DIR          = os.path.join(MODEL_DIR, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))

# Training hyperparameters
EPOCHS_FROZEN   = 10    # phase 1 — train only the new head, base frozen
EPOCHS_FINETUNE = 20    # phase 2 — unfreeze top layers, fine-tune
LEARNING_RATE_1 = 1e-3  # phase 1 learning rate
LEARNING_RATE_2 = 1e-5  # phase 2 — much lower to avoid destroying pretrained weights
DROPOUT_RATE    = 0.4
UNFREEZE_LAYERS = 30    # how many top layers of MobileNetV2 to unfreeze in phase 2


# ─────────────────────────────────────────
# STEP 1 — BUILD MODEL
# ─────────────────────────────────────────

def build_model():
    """
    Builds a transfer learning model:
      Base  : MobileNetV2 pretrained on ImageNet (weights frozen initially)
      Head  : GlobalAveragePooling → BatchNorm → Dense → Dropout → Softmax output

    MobileNetV2 is chosen because:
      - Runs fast on CPU (important for local/offline use)
      - Small file size (~14MB)
      - Strong accuracy on visual classification tasks
    """

    # Load MobileNetV2 without its top classification layer
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,          # remove ImageNet classifier
        weights="imagenet"          # use pretrained weights
    )

    # Freeze all base layers — we only train our new head in phase 1
    base_model.trainable = False

    # Build custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)          # flatten feature maps → single vector
    x = BatchNormalization()(x)              # stabilize activations
    x = Dense(256, activation="relu")(x)     # learn emergency-specific features
    x = Dropout(DROPOUT_RATE)(x)             # prevent overfitting
    x = Dense(128, activation="relu")(x)
    x = Dropout(DROPOUT_RATE / 2)(x)
    output = Dense(NUM_CLASSES, activation="softmax")(x)  # 4-class probability output

    model = Model(inputs=base_model.input, outputs=output)

    print(f"\n[train] Model built — total layers     : {len(model.layers)}")
    print(f"[train] Trainable params (phase 1)     : {model.count_params():,}")

    return model, base_model


# ─────────────────────────────────────────
# STEP 2 — CALLBACKS
# ─────────────────────────────────────────

def get_callbacks(phase: int):
    """
    Returns training callbacks for a given phase.
    Phase 1 : frozen base — standard early stopping
    Phase 2 : fine-tuning — tighter patience, lower LR reduction
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    callbacks = [
        # Save the best model weights automatically
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),

        # Stop training if val_accuracy stops improving
        EarlyStopping(
            monitor="val_accuracy",
            patience=5 if phase == 1 else 7,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate when training plateaus
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),

        # TensorBoard logging (optional — run: tensorboard --logdir model/logs)
        TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
    ]

    return callbacks


# ─────────────────────────────────────────
# STEP 3 — TRAIN PHASE 1 (frozen base)
# ─────────────────────────────────────────

def train_phase1(model, train_data, val_data):
    """
    Phase 1: Base model frozen, only the custom head trains.
    Fast to run, teaches the head to extract emergency features.
    """
    print("\n" + "=" * 55)
    print("  PHASE 1 — Training classification head (base frozen)")
    print("=" * 55)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_1),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_FROZEN,
        callbacks=get_callbacks(phase=1),
        verbose=1
    )

    print(f"\n[train] Phase 1 complete.")
    print(f"[train] Best val accuracy: {max(history1.history['val_accuracy']):.4f}")

    return history1


# ─────────────────────────────────────────
# STEP 4 — TRAIN PHASE 2 (fine-tuning)
# ─────────────────────────────────────────

def train_phase2(model, base_model, train_data, val_data):
    """
    Phase 2: Unfreeze the top N layers of MobileNetV2 and fine-tune.
    Uses a very low learning rate to gently adjust pretrained weights
    without destroying what ImageNet training already learned.
    """
    print("\n" + "=" * 55)
    print(f"  PHASE 2 — Fine-tuning top {UNFREEZE_LAYERS} layers of MobileNetV2")
    print("=" * 55)

    # Unfreeze the top UNFREEZE_LAYERS layers
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False

    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"[train] Trainable layers in phase 2: {trainable_count}")

    # Recompile with much lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_2),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_FINETUNE,
        callbacks=get_callbacks(phase=2),
        verbose=1
    )

    print(f"\n[train] Phase 2 complete.")
    print(f"[train] Best val accuracy: {max(history2.history['val_accuracy']):.4f}")

    return history2


# ─────────────────────────────────────────
# STEP 5 — SAVE CLASS INDICES
# ─────────────────────────────────────────

def save_class_indices(train_data):
    """
    Saves the class name → index mapping to a JSON file.
    image_analysis.py reads this to convert prediction index → class name.
    e.g. {"fire": 0, "medical": 1, "crime": 2, "normal": 3}
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(CLASS_INDEX_PATH, "w") as f:
        json.dump(train_data.class_indices, f, indent=2)
    print(f"[train] Class indices saved → {CLASS_INDEX_PATH}")


# ─────────────────────────────────────────
# STEP 6 — PLOT TRAINING HISTORY
# ─────────────────────────────────────────

def plot_history(history1, history2):
    """
    Saves a training curve plot (accuracy + loss) to model/training_plot.png.
    Useful for your college report / presentation.
    """
    # Combine both phase histories
    acc  = history1.history["accuracy"]      + history2.history["accuracy"]
    val  = history1.history["val_accuracy"]  + history2.history["val_accuracy"]
    loss = history1.history["loss"]          + history2.history["loss"]
    v_loss = history1.history["val_loss"]    + history2.history["val_loss"]
    phase_boundary = len(history1.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(acc,   label="Train accuracy",      color="#1D9E75")
    ax1.plot(val,   label="Val accuracy",        color="#534AB7", linestyle="--")
    ax1.axvline(x=phase_boundary, color="#D85A30", linestyle=":", label="Fine-tune starts")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss plot
    ax2.plot(loss,   label="Train loss",         color="#1D9E75")
    ax2.plot(v_loss, label="Val loss",           color="#534AB7", linestyle="--")
    ax2.axvline(x=phase_boundary, color="#D85A30", linestyle=":", label="Fine-tune starts")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "training_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[train] Training plot saved → {plot_path}")


# ─────────────────────────────────────────
# STEP 7 — EVALUATE FINAL MODEL
# ─────────────────────────────────────────

def evaluate_model(model, val_data):
    """
    Prints final accuracy and loss on the validation set.
    Also shows a per-class breakdown using a confusion matrix.
    """
    print("\n[train] Evaluating on validation set...")
    loss, accuracy = model.evaluate(val_data, verbose=0)
    print(f"[train] Final validation accuracy : {accuracy * 100:.2f}%")
    print(f"[train] Final validation loss     : {loss:.4f}")

    # Per-class prediction breakdown
    print("\n[train] Per-class prediction sample (first 5 batches):")
    y_true, y_pred = [], []

    for i, (images, labels) in enumerate(val_data):
        if i >= 5:
            break
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds,  axis=1))

    from collections import Counter
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    idx_to_class = {v: k for k, v in val_data.class_indices.items()}

    print(f"\n  {'Class':<12} {'True count':>12} {'Predicted count':>16}")
    print("  " + "-" * 42)
    for idx in sorted(idx_to_class):
        cls = idx_to_class[idx]
        print(f"  {cls:<12} {true_counts.get(idx, 0):>12} {pred_counts.get(idx, 0):>16}")


# ─────────────────────────────────────────
# MAIN — run everything
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Emergency Image Detection — train.py")
    print("=" * 55)

    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[train] GPU detected: {gpus[0].name} — training will be fast.")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("[train] No GPU found — using CPU. Training will take longer.")
        print("[train] Tip: Google Colab gives free GPU if your laptop is slow.\n")

    # Load data
    print("[train] Loading data generators from preprocess.py...")
    train_data, val_data = get_data_generators()

    # Build model
    model, base_model = build_model()

    # Save class indices before training
    save_class_indices(train_data)

    # Phase 1 — frozen base
    history1 = train_phase1(model, train_data, val_data)

    # Phase 2 — fine-tune
    history2 = train_phase2(model, base_model, train_data, val_data)

    # Save final model (best weights already saved by ModelCheckpoint)
    model.save(MODEL_PATH)
    print(f"\n[train] Final model saved → {MODEL_PATH}")

    # Plot training curves
    plot_history(history1, history2)

    # Evaluate
    evaluate_model(model, val_data)

    print("\n" + "=" * 55)
    print("  Training complete!")
    print(f"  Model saved to : {MODEL_PATH}")
    print(f"  Next step      : run image_analysis.py")
    print("=" * 55)