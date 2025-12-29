"""
SqueezeNet Model for Handwritten Digit Recognition
EE 4065 - Embedded Digital Image Processing
Kaan Atalay - 150720057

Reference: SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def fire_module(x, squeeze_filters, expand_filters, name):
    """
    Fire module: squeeze layer followed by expand layer
    
    Args:
        x: Input tensor
        squeeze_filters: Number of filters in squeeze layer
        expand_filters: Number of filters in expand layer
        name: Module name
    
    Returns:
        Output tensor
    """
    # Squeeze layer (1x1 convolutions)
    squeeze = layers.Conv2D(squeeze_filters, (1, 1), activation='relu', 
                           padding='same', name=f'{name}_squeeze')(x)
    
    # Expand layer (1x1 and 3x3 convolutions)
    expand_1x1 = layers.Conv2D(expand_filters, (1, 1), activation='relu',
                               padding='same', name=f'{name}_expand_1x1')(squeeze)
    expand_3x3 = layers.Conv2D(expand_filters, (3, 3), activation='relu',
                               padding='same', name=f'{name}_expand_3x3')(squeeze)
    
    # Concatenate expand layers
    output = layers.Concatenate(name=f'{name}_concat')([expand_1x1, expand_3x3])
    
    return output


def create_squeezenet(input_shape=(32, 32, 3), num_classes=10):
    """
    Create SqueezeNet model adapted for MNIST digit recognition
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
    
    Returns:
        SqueezeNet model
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Initial convolution
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu',
                     padding='same', name='conv1')(inputs)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    
    # Fire modules
    x = fire_module(x, 16, 64, name='fire2')
    x = fire_module(x, 16, 64, name='fire3')
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    
    x = fire_module(x, 32, 128, name='fire4')
    x = fire_module(x, 32, 128, name='fire5')
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
    
    x = fire_module(x, 48, 192, name='fire6')
    x = fire_module(x, 48, 192, name='fire7')
    x = fire_module(x, 64, 256, name='fire8')
    x = fire_module(x, 64, 256, name='fire9')
    
    # Dropout
    x = layers.Dropout(0.5, name='dropout')(x)
    
    # Final convolution
    x = layers.Conv2D(num_classes, (1, 1), activation='relu',
                     padding='same', name='conv10')(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    
    # Softmax output
    outputs = layers.Activation('softmax', name='predictions')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='SqueezeNet')
    
    return model


def get_squeezenet_summary(input_shape=(32, 32, 3), num_classes=10):
    """Get model summary"""
    model = create_squeezenet(input_shape, num_classes)
    model.summary()
    return model


if __name__ == "__main__":
    model = get_squeezenet_summary()
    print(f"\nTotal parameters: {model.count_params():,}")

