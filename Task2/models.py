# Libraries and files
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers, models, losses
import ssl

def pretrained_model(num_classes):
    print("Pre-training model.")
    ssl._create_default_https_context = ssl._create_unverified_context

    # Get our base model:ResNet50 with the same input shape, pretrained weights and with out the output layer
    base_model = ResNet50(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

    # Replace top layers to reflect our number of classes instead of original 1000
    avg_pool = layers.GlobalAveragePooling2D()(base_model.output)
    out_layer = layers.Dense(num_classes, activation='softmax')(avg_pool)
    model = tf.keras.Model(base_model.input, out_layer)

    # model.summary()
    for layer in base_model.layers:
        layer.trainable = False

    # # To check
    # for i, layer in enumerate(base_model.layers):
    #     print(i, layer.name, "-", layer.trainable)

    return model


def scratch_model(num_classes):
    # Get our base model: Resnet50 with the same input shape, random weights and without the output layer
    base_model = ResNet50(include_top=False, input_shape=(224, 224, 3), weights=None)

    # Replace top layers to reflect our number of classes instead of original 1000
    avg_pool = layers.GlobalAveragePooling2D()(base_model.output)
    out_layer = layers.Dense(num_classes, activation='softmax')(avg_pool)
    model = tf.keras.Model(base_model.input, out_layer)

    # model.summary()
    for layer in base_model.layers:
        layer.trainable = False

    return model
