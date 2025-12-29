"""
MobileNet Model for Handwritten Digit Recognition
EE 4065 - Embedded Digital Image Processing
Kaan Atalay - 150720057

Reference: MobileNets: Efficient CNNs for Mobile Vision Applications
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def depthwise_separable_conv(x, pointwise_filters, strides=(1, 1), name=''):
    """
    Depthwise separable convolution block
    
    Args:
        x: Input tensor
        pointwise_filters: Number of pointwise convolution filters
        strides: Convolution strides
        name: Block name
    
    Returns:
        Output tensor
    """
    # Depthwise convolution
    x = layers.DepthwiseConv2D((3, 3), strides=strides, padding='same',
                               use_bias=False, name=f'{name}_dw')(x)
    x = layers.BatchNormalization(name=f'{name}_dw_bn')(x)
    x = layers.ReLU(6., name=f'{name}_dw_relu')(x)
    
    # Pointwise convolution
    x = layers.Conv2D(pointwise_filters, (1, 1), padding='same',
                     use_bias=False, name=f'{name}_pw')(x)
    x = layers.BatchNormalization(name=f'{name}_pw_bn')(x)
    x = layers.ReLU(6., name=f'{name}_pw_relu')(x)
    
    return x


def create_mobilenet(input_shape=(32, 32, 3), num_classes=10, alpha=1.0):
    """
    Create MobileNet model adapted for MNIST digit recognition
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        alpha: Width multiplier
    
    Returns:
        MobileNet model
    """
    def _make_divisible(v, divisor=8):
        return int((v + divisor / 2) // divisor * divisor)
    
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Initial convolution
    first_filters = _make_divisible(32 * alpha)
    x = layers.Conv2D(first_filters, (3, 3), strides=(1, 1), padding='same',
                     use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.ReLU(6., name='conv1_relu')(x)
    
    # Depthwise separable convolution blocks
    x = depthwise_separable_conv(x, _make_divisible(64 * alpha), strides=(1, 1), name='block1')
    x = depthwise_separable_conv(x, _make_divisible(128 * alpha), strides=(2, 2), name='block2')
    x = depthwise_separable_conv(x, _make_divisible(128 * alpha), strides=(1, 1), name='block3')
    x = depthwise_separable_conv(x, _make_divisible(256 * alpha), strides=(2, 2), name='block4')
    x = depthwise_separable_conv(x, _make_divisible(256 * alpha), strides=(1, 1), name='block5')
    x = depthwise_separable_conv(x, _make_divisible(512 * alpha), strides=(2, 2), name='block6')
    
    # 5 blocks with 512 filters
    for i in range(5):
        x = depthwise_separable_conv(x, _make_divisible(512 * alpha), 
                                    strides=(1, 1), name=f'block{7+i}')
    
    x = depthwise_separable_conv(x, _make_divisible(1024 * alpha), strides=(2, 2), name='block12')
    x = depthwise_separable_conv(x, _make_divisible(1024 * alpha), strides=(1, 1), name='block13')
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    
    # Dropout and dense layer
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='MobileNet')
    
    return model


def get_mobilenet_summary(input_shape=(32, 32, 3), num_classes=10):
    """Get model summary"""
    model = create_mobilenet(input_shape, num_classes)
    model.summary()
    return model


if __name__ == "__main__":
    model = get_mobilenet_summary()
    print(f"\nTotal parameters: {model.count_params():,}")

