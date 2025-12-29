"""
Training Script for Handwritten Digit Recognition
EE 4065 - Embedded Digital Image Processing
Kaan Atalay - 150720057
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json
from datetime import datetime

from data_loader import load_mnist_data, prepare_for_cnn, visualize_samples
from models import create_squeezenet, create_efficientnet_b0, create_mobilenet


def train_model(model, x_train, y_train, x_val, y_val, model_name, 
                epochs=50, batch_size=64, save_dir='../models'):
    """
    Train a CNN model
    
    Args:
        model: Keras model
        x_train, y_train: Training data
        x_val, y_val: Validation data
        model_name: Model name for saving
        epochs: Number of epochs
        batch_size: Batch size
        save_dir: Directory to save model
    
    Returns:
        Training history
    """
    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(save_dir, f'{model_name}_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Train
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, x_test, y_test, model_name, save_dir='../results'):
    """
    Evaluate a trained model
    
    Args:
        model: Trained Keras model
        x_test, y_test: Test data
        model_name: Model name
        save_dir: Directory to save results
    
    Returns:
        Test accuracy and predictions
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n{model_name} Test Accuracy: {test_acc:.4f}")
    print(f"{model_name} Test Loss: {test_loss:.4f}")
    
    # Predictions
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    report = classification_report(y_true_classes, y_pred_classes, 
                                  target_names=[str(i) for i in range(10)])
    print(f"\nClassification Report for {model_name}:")
    print(report)
    
    # Save report
    with open(os.path.join(save_dir, f'{model_name}_report.txt'), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    return test_acc, test_loss, cm, y_pred_classes


def plot_confusion_matrix(cm, model_name, save_dir='../results'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=150)
    plt.close()


def plot_training_history(history, model_name, save_dir='../results'):
    """Plot and save training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title(f'{model_name} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title(f'{model_name} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_history.png'), dpi=150)
    plt.close()


def compare_models(results, save_dir='../results'):
    """Compare all models and create summary plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    params = [results[m]['params'] for m in model_names]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    axes[0].bar(model_names, accuracies, color=colors)
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_ylim([0.95, 1.0])
    for i, acc in enumerate(accuracies):
        axes[0].text(i, acc + 0.002, f'{acc:.4f}', ha='center', fontweight='bold')
    
    # Parameter comparison
    params_k = [p / 1000 for p in params]
    axes[1].bar(model_names, params_k, color=colors)
    axes[1].set_title('Model Parameters Comparison')
    axes[1].set_ylabel('Parameters (K)')
    for i, p in enumerate(params_k):
        axes[1].text(i, p + max(params_k)*0.02, f'{p:.1f}K', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150)
    plt.close()
    
    # Save results to JSON
    results_json = {k: {'accuracy': float(v['accuracy']), 
                        'loss': float(v['loss']),
                        'params': int(v['params'])} 
                   for k, v in results.items()}
    
    with open(os.path.join(save_dir, 'results_summary.json'), 'w') as f:
        json.dump(results_json, f, indent=2)


def main():
    """Main training pipeline"""
    print("="*60)
    print("Handwritten Digit Recognition with CNN Models")
    print("EE 4065 - Embedded Digital Image Processing")
    print("Kaan Atalay - 150720057")
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load and prepare data
    print("\n[1] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Split training data for validation
    val_split = 0.1
    val_size = int(len(x_train) * val_split)
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    
    # Prepare data for CNN (resize to 32x32, convert to RGB)
    print("\n[2] Preparing data for CNN models...")
    x_train_cnn, x_test_cnn = prepare_for_cnn(x_train, x_test, target_size=(32, 32), channels=3)
    x_val_cnn, _ = prepare_for_cnn(x_val, x_test[:1], target_size=(32, 32), channels=3)
    
    # Visualize samples
    visualize_samples(x_train_cnn[:10], y_train[:10], save_path='../results/sample_images.png')
    
    # Define models
    models = {
        'SqueezeNet': create_squeezenet(input_shape=(32, 32, 3), num_classes=10),
        'EfficientNet-B0': create_efficientnet_b0(input_shape=(32, 32, 3), num_classes=10),
        'MobileNet': create_mobilenet(input_shape=(32, 32, 3), num_classes=10)
    }
    
    results = {}
    
    # Training parameters - FAST MODE
    EPOCHS = 5  # Reduced for quick training
    BATCH_SIZE = 128  # Larger batch for speed
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n[3] Training {model_name}...")
        
        # Train
        history = train_model(
            model, x_train_cnn, y_train, x_val_cnn, y_val,
            model_name=model_name.replace('-', '_'),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        
        # Plot training history
        plot_training_history(history, model_name)
        
        # Evaluate
        print(f"\n[4] Evaluating {model_name}...")
        test_acc, test_loss, cm, y_pred = evaluate_model(
            model, x_test_cnn, y_test, model_name
        )
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, model_name)
        
        # Store results
        results[model_name] = {
            'accuracy': test_acc,
            'loss': test_loss,
            'params': model.count_params(),
            'confusion_matrix': cm.tolist()
        }
    
    # Compare models
    print("\n[5] Comparing models...")
    compare_models(results)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    for model_name, res in results.items():
        print(f"{model_name}:")
        print(f"  - Test Accuracy: {res['accuracy']:.4f}")
        print(f"  - Parameters: {res['params']:,}")
    
    return results


if __name__ == "__main__":
    results = main()

