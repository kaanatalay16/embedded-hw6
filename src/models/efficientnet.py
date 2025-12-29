"""
EfficientNet Model for Handwritten Digit Recognition
EE 4065 - Embedded Digital Image Processing
Kaan Atalay - 150720057

Reference: EfficientNet: Rethinking Model Scaling for CNNs
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import math


def swish(x):
    """Swish activation function"""
    return x * tf.keras.activations.sigmoid(x)


def se_block(inputs, filters, se_ratio=0.25):
    """
    Squeeze-and-Excitation block
    
    Args:
        inputs: Input tensor
        filters: Number of filters
        se_ratio: Squeeze ratio
    
    Returns:
        Output tensor with channel attention
    """
    se_filters = max(1, int(filters * se_ratio))
    
    # Squeeze
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Reshape((1, 1, filters))(x)
    
    # Excitation
    x = layers.Conv2D(se_filters, 1, activation='swish', padding='same')(x)
    x = layers.Conv2D(filters, 1, activation='sigmoid', padding='same')(x)
    
    # Scale
    return layers.Multiply()([inputs, x])


def mbconv_block(inputs, filters_in, filters_out, kernel_size, stride,
                 expand_ratio, se_ratio=0.25, drop_rate=0.2, name=''):
    """
    Mobile Inverted Residual Bottleneck (MBConv) block
    
    Args:
        inputs: Input tensor
        filters_in: Input filters
        filters_out: Output filters
        kernel_size: Convolution kernel size
        stride: Convolution stride
        expand_ratio: Expansion ratio
        se_ratio: SE ratio
        drop_rate: Dropout rate
        name: Block name
    
    Returns:
        Output tensor
    """
    filters = filters_in * expand_ratio
    
    x = inputs
    
    # Expansion phase
    if expand_ratio != 1:
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                         name=f'{name}_expand_conv')(x)
        x = layers.BatchNormalization(name=f'{name}_expand_bn')(x)
        x = layers.Activation('swish', name=f'{name}_expand_activation')(x)
    
    # Depthwise convolution
    x = layers.DepthwiseConv2D(kernel_size, stride, padding='same', use_bias=False,
                               name=f'{name}_dwconv')(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.Activation('swish', name=f'{name}_activation')(x)
    
    # SE block
    if se_ratio:
        x = se_block(x, filters, se_ratio)
    
    # Output phase
    x = layers.Conv2D(filters_out, 1, padding='same', use_bias=False,
                     name=f'{name}_project_conv')(x)
    x = layers.BatchNormalization(name=f'{name}_project_bn')(x)
    
    # Skip connection
    if stride == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1),
                              name=f'{name}_drop')(x)
        x = layers.Add(name=f'{name}_add')([x, inputs])
    
    return x


def create_efficientnet_b0(input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.2):
    """
    Create EfficientNet-B0 model adapted for MNIST digit recognition
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        dropout_rate: Dropout rate
    
    Returns:
        EfficientNet-B0 model
    """
    # Default block configs for EfficientNet-B0 (simplified for MNIST)
    block_configs = [
        # (expand_ratio, filters, repeats, stride, kernel_size)
        (1, 16, 1, 1, 3),
        (6, 24, 2, 2, 3),
        (6, 40, 2, 2, 5),
        (6, 80, 3, 2, 3),
        (6, 112, 3, 1, 5),
        (6, 192, 4, 2, 5),
        (6, 320, 1, 1, 3),
    ]
    
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Stem
    x = layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('swish', name='stem_activation')(x)
    
    # Build blocks
    filters_in = 32
    block_id = 0
    
    for expand_ratio, filters_out, repeats, stride, kernel_size in block_configs:
        for i in range(repeats):
            s = stride if i == 0 else 1
            x = mbconv_block(x, filters_in, filters_out, kernel_size, s,
                            expand_ratio, name=f'block{block_id}')
            filters_in = filters_out
            block_id += 1
    
    # Head
    x = layers.Conv2D(1280, 1, padding='same', use_bias=False, name='head_conv')(x)
    x = layers.BatchNormalization(name='head_bn')(x)
    x = layers.Activation('swish', name='head_activation')(x)
    
    x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='dropout')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='EfficientNet-B0')
    
    return model


def get_efficientnet_summary(input_shape=(32, 32, 3), num_classes=10):
    """Get model summary"""
    model = create_efficientnet_b0(input_shape, num_classes)
    model.summary()
    return model


if __name__ == "__main__":
    model = get_efficientnet_summary()
    print(f"\nTotal parameters: {model.count_params():,}")

