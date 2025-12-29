"""
MNIST Dataset Loader and Preprocessing
EE 4065 - Embedded Digital Image Processing
Kaan Atalay - 150720057
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def load_mnist_data(normalize=True, categorical=True):
    """
    Load and preprocess MNIST dataset
    
    Args:
        normalize: Whether to normalize pixel values to [0, 1]
        categorical: Whether to convert labels to categorical format
    
    Returns:
        (x_train, y_train), (x_test, y_test): Training and test data
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print(f"Original training data shape: {x_train.shape}")
    print(f"Original test data shape: {x_test.shape}")
    
    # Normalize pixel values
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical
    if categorical:
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)


def prepare_for_cnn(x_train, x_test, target_size=(32, 32), channels=3):
    """
    Prepare data for CNN models (resize and add channels)
    
    Args:
        x_train: Training images
        x_test: Test images
        target_size: Target image size (height, width)
        channels: Number of channels (1 for grayscale, 3 for RGB)
    
    Returns:
        Resized training and test data
    """
    # Add channel dimension if needed
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
    
    # Resize images using tf.image.resize
    x_train_resized = tf.image.resize(x_train, target_size).numpy()
    x_test_resized = tf.image.resize(x_test, target_size).numpy()
    
    # Convert to RGB if needed (replicate grayscale to 3 channels)
    if channels == 3 and x_train_resized.shape[-1] == 1:
        x_train_resized = np.repeat(x_train_resized, 3, axis=-1)
        x_test_resized = np.repeat(x_test_resized, 3, axis=-1)
    
    print(f"Resized training data shape: {x_train_resized.shape}")
    print(f"Resized test data shape: {x_test_resized.shape}")
    
    return x_train_resized, x_test_resized


def visualize_samples(x_data, y_data, num_samples=10, save_path=None):
    """
    Visualize sample images from the dataset
    
    Args:
        x_data: Image data
        y_data: Labels (can be categorical or integer)
        num_samples: Number of samples to display
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i in range(num_samples):
        # Get label
        if len(y_data.shape) > 1:
            label = np.argmax(y_data[i])
        else:
            label = y_data[i]
        
        # Display image
        if x_data.shape[-1] == 1:
            axes[i].imshow(x_data[i, :, :, 0], cmap='gray')
        elif x_data.shape[-1] == 3:
            axes[i].imshow(x_data[i])
        else:
            axes[i].imshow(x_data[i], cmap='gray')
        
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()  # Don't block - close the figure


if __name__ == "__main__":
    # Test data loading
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    x_train_cnn, x_test_cnn = prepare_for_cnn(x_train, x_test, target_size=(32, 32), channels=3)
    visualize_samples(x_train_cnn, y_train, save_path='results/sample_images.png')

