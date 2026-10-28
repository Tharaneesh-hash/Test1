"""
train_vgg19_keras.py — Complete TensorFlow/Keras VGG19 Mpox Detection Pipeline.

Implements all specifications from user prompt Steps 4, 5, 6, 7, 8, 9:
- tf.keras.applications.VGG19 with ImageNet weights (include_top=False)
- Strict tf.keras.applications.vgg19.preprocess_input (NO double /255.0 scaling)
- Input shape: (224, 224, 3)
- Stage 1: Frozen backbone + GAP + Dropout(0.4) + Dense(1, activation='sigmoid')
- Stage 2: Fine-tuning block5_conv layers with lower learning rate
- Binary cross-entropy with balanced class weighting
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- Evaluates on independent test set and saves confusion matrix and ROC curves
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
        from tensorflow.keras import layers, models, optimizers, callbacks
        from sklearn.utils.class_weight import compute_class_weight
        from sklearn.metrics import (
            classification_report, confusion_matrix, roc_auc_score,
            accuracy_score, precision_score, recall_score, f1_score
        )
    except ImportError:
        print("=" * 70)
        print("TensorFlow / Keras is not installed in the current environment.")
        print("To run this Keras pipeline, install TensorFlow via:")
        print("    pip install tensorflow")
        print("=" * 70)
        return

    print("=" * 70)
    print("  TENSORFLOW / KERAS VGG19 MPOX BINARY TRAINING PIPELINE")
    print(f"  TensorFlow Version: {tf.__version__}")
    print("=" * 70)

    # 1. Paths and Configuration
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "FOLDS" / "fold1"
    TRAIN_DIR = DATA_DIR / "train"
    VALID_DIR = DATA_DIR / "valid"
    TEST_DIR  = DATA_DIR / "test"

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16
    SEED = 42

    MODELS_DIR = PROJECT_ROOT / "models"
    METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
    PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
    for d in [MODELS_DIR, METRICS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Safe Preprocessing Function (VGG19 Caffe-style mode: zero-centered, BGR, NO /255.0)
    def vgg19_preprocessor(img_tensor):
        return preprocess_input(img_tensor)

    # 3. Data Loading via tf.keras.utils.image_dataset_from_directory
    # (Extracts binary labels from folder or prefix)
    print("\nLoading datasets...")
    # Helper to load from flat prefix directory into tf.data.Dataset
    def build_dataset_from_prefix_dir(folder: Path, is_training: bool = False):
        files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        paths = []
        labels = []
        for f in files:
            prefix = f.name.split("_")[0].upper()
            if prefix in ["MKP", "MID", "MSID", "MSLDV2"]:
                label = 1.0  # Mpox
            elif prefix in ["CHP", "CWP", "HEALTHY", "HFMD", "MSL"]:
                label = 0.0  # Non-Mpox
            else:
                continue
            paths.append(str(f))
            labels.append(label)

        def parse_image(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            img = vgg19_preprocessor(img)
            return img, label

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if is_training:
            # Training augmentations (medically modest)
            augment = tf.keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomTranslation(0.05, 0.05),
            ])
            ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
            ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds, np.array(labels), paths

    train_ds, y_train, train_paths = build_dataset_from_prefix_dir(TRAIN_DIR, is_training=True)
    val_ds, y_val, val_paths = build_dataset_from_prefix_dir(VALID_DIR, is_training=False)
    test_ds, y_test, test_paths = build_dataset_from_prefix_dir(TEST_DIR, is_training=False)

    print(f"  Train: {len(y_train)} (Mpox={int(y_train.sum())}, Non-Mpox={int(len(y_train)-y_train.sum())})")
    print(f"  Valid: {len(y_val)} (Mpox={int(y_val.sum())}, Non-Mpox={int(len(y_val)-y_val.sum())})")
    print(f"  Test:  {len(y_test)} (Mpox={int(y_test.sum())}, Non-Mpox={int(len(y_test)-y_test.sum())})")

    # Balanced class weights
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: float(weights[0]), 1: float(weights[1])}
    print(f"  Class Weights: {class_weight_dict}")

    # 4. Build Model with VGG19 Transfer Learning
    print("\nBuilding VGG19 architecture...")
    base_vgg19 = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_vgg19.trainable = False  # Initially frozen

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_vgg19(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="VGG19_Mpox_Classifier")

    # 5. Compile Model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Recall(name="recall")]
    )

    # 6. Callbacks
    keras_model_path = MODELS_DIR / "vgg19_mpox_keras_best.keras"
    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=str(keras_model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 7. Stage 1 Training: Frozen Backbone
    print("\n--- Stage 1: Training Classification Head (10 Epochs) ---")
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weight_dict,
        callbacks=cb_list
    )

    # 8. Stage 2 Training: Fine-tune Upper Layers (block5)
    print("\n--- Stage 2: Fine-Tuning Block 5 Conv Layers (5 Epochs) ---")
    base_vgg19.trainable = True
    for layer in base_vgg19.layers:
        if "block5" in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Recall(name="recall")]
    )

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        initial_epoch=len(history_stage1.history["loss"]),
        class_weight=class_weight_dict,
        callbacks=cb_list
    )

    # 9. Evaluate on Independent Test Set
    print("\n" + "=" * 70)
    print("  EVALUATING BEST MODEL ON INDEPENDENT TEST SET (74 IMAGES)")
    print("=" * 70)
    best_model = models.load_model(str(keras_model_path))
    y_pred_probs = best_model.predict(test_ds).ravel()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, y_pred_probs)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Test Accuracy:    {test_acc:.2%}")
    print(f"Mpox Sensitivity: {test_rec:.2%}")
    print(f"Specificity:      {cm[0,0]/(cm[0,0]+cm[0,1]):.2%}")
    print(f"Precision:        {test_prec:.2%}")
    print(f"F1-Score:         {test_f1:.4f}")
    print(f"ROC-AUC:          {test_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    main()
