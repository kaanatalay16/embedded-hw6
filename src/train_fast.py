"""
FAST Training Script - Uses Keras Pre-trained Models
EE 4065 - Embedded Digital Image Processing
Kaan Atalay - 150720057
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

# Disable GPU if causing issues
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def create_simple_cnn(input_shape=(32, 32, 1), num_classes=10, name="SimpleCNN"):
    """Simple CNN for fast training"""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name=name)


def create_squeezenet_mini(input_shape=(32, 32, 1), num_classes=10):
    """Mini SqueezeNet for MNIST"""

    def fire_module(x, s, e):
        squeeze = layers.Conv2D(s, 1, activation="relu", padding="same")(x)
        e1 = layers.Conv2D(e, 1, activation="relu", padding="same")(squeeze)
        e3 = layers.Conv2D(e, 3, activation="relu", padding="same")(squeeze)
        return layers.Concatenate()([e1, e3])

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = fire_module(x, 8, 32)
    x = fire_module(x, 8, 32)
    x = layers.MaxPooling2D(2)(x)
    x = fire_module(x, 16, 64)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="SqueezeNet-Mini")


def create_mobilenet_mini(input_shape=(32, 32, 1), num_classes=10):
    """Mini MobileNet for MNIST"""

    def depthwise_block(x, filters, strides=1):
        x = layers.DepthwiseConv2D(3, strides=strides, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = depthwise_block(x, 64)
    x = depthwise_block(x, 128, strides=2)
    x = depthwise_block(x, 128)
    x = depthwise_block(x, 256, strides=2)
    x = depthwise_block(x, 256)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="MobileNet-Mini")


def create_efficientnet_mini(input_shape=(32, 32, 1), num_classes=10):
    """Mini EfficientNet for MNIST"""

    def mbconv(x, filters, expand=4):
        expanded = layers.Conv2D(filters * expand, 1, padding="same")(x)
        expanded = layers.BatchNormalization()(expanded)
        expanded = layers.Activation("swish")(expanded)

        dw = layers.DepthwiseConv2D(3, padding="same")(expanded)
        dw = layers.BatchNormalization()(dw)
        dw = layers.Activation("swish")(dw)

        # SE block
        se = layers.GlobalAveragePooling2D()(dw)
        se = layers.Dense(filters, activation="swish")(se)
        se = layers.Dense(filters * expand, activation="sigmoid")(se)
        se = layers.Reshape((1, 1, filters * expand))(se)
        dw = layers.Multiply()([dw, se])

        out = layers.Conv2D(filters, 1, padding="same")(dw)
        out = layers.BatchNormalization()(out)

        if x.shape[-1] == filters:
            out = layers.Add()([x, out])
        return out

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    x = mbconv(x, 32)
    x = layers.MaxPooling2D(2)(x)
    x = mbconv(x, 64)
    x = layers.MaxPooling2D(2)(x)
    x = mbconv(x, 128)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="EfficientNet-Mini")


def plot_results(history, model_name, save_dir):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title(f"{model_name} - Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title(f"{model_name} - Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_training_history.png", dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, model_name, save_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_confusion_matrix.png", dpi=150)
    plt.close()
    return cm


def main():
    print("=" * 60)
    print("FAST Handwritten Digit Recognition")
    print("EE 4065 - Kaan Atalay - 150720057")
    print("=" * 60)

    # Settings
    EPOCHS = 3
    BATCH_SIZE = 256

    # Create directories
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../models", exist_ok=True)

    # Load data
    print("\n[1] Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocess - keep grayscale, resize to 32x32
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

    # Resize
    x_train = tf.image.resize(x_train, (32, 32)).numpy()
    x_test = tf.image.resize(x_test, (32, 32)).numpy()

    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    # Save sample images
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_train[i, :, :, 0], cmap="gray")
        ax.set_title(f"Label: {y_train[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("../results/sample_images.png", dpi=150)
    plt.close()
    print("Sample images saved.")

    # Models
    models = {
        "SqueezeNet": create_squeezenet_mini(),
        "EfficientNet-B0": create_efficientnet_mini(),
        "MobileNet": create_mobilenet_mini(),
    }

    results = {}

    for name, model in models.items():
        print(f"\n[2] Training {name}...")

        model.compile(
            optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"]
        )

        history = model.fit(
            x_train,
            y_train_cat,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=0.1,
            verbose=1,
        )

        # Evaluate
        print(f"\n[3] Evaluating {name}...")
        loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
        y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

        print(f"{name} Test Accuracy: {acc:.4f}")

        # Save plots
        plot_results(history, name, "../results")
        cm = plot_confusion(y_test, y_pred, name, "../results")

        # Save model
        model.save(f"../models/{name}_best.keras")

        # Store results
        results[name] = {
            "accuracy": float(acc),
            "loss": float(loss),
            "params": int(model.count_params()),
        }

        # Classification report
        report = classification_report(y_test, y_pred)
        with open(f"../results/{name}_report.txt", "w") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Test Accuracy: {acc:.4f}\n")
            f.write(f"Parameters: {model.count_params():,}\n\n")
            f.write(report)

    # Comparison plot
    print("\n[4] Creating comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    params = [results[n]["params"] / 1000 for n in names]

    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    axes[0].bar(names, accs, color=colors)
    axes[0].set_ylim([0.9, 1.0])
    axes[0].set_title("Model Accuracy Comparison")
    axes[0].set_ylabel("Test Accuracy")
    for i, a in enumerate(accs):
        axes[0].text(i, a + 0.005, f"{a:.4f}", ha="center", fontweight="bold")

    axes[1].bar(names, params, color=colors)
    axes[1].set_title("Model Parameters (K)")
    axes[1].set_ylabel("Parameters (thousands)")
    for i, p in enumerate(params):
        axes[1].text(
            i, p + max(params) * 0.02, f"{p:.1f}K", ha="center", fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig("../results/model_comparison.png", dpi=150)
    plt.close()

    # Save JSON
    with open("../results/results_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    for name, res in results.items():
        print(f"{name}: {res['accuracy']:.4f} accuracy, {res['params']:,} params")

    print(f"\nResults saved to: results/")
    print(f"Models saved to: models/")

    return results


if __name__ == "__main__":
    results = main()
