"""Neural networks for feature extraction."""

import tensorflow as tf
from tensorflow.keras import layers


def single_frame_model(frame_shape):
    """Create a Keras model that encodes a single frame.

    This defines a Keras autoencoder which takes as input a single frame.
    The outputs of the models are the encoded input, and the reconstruction.

    :param input_shape: the shape of the input frame (sequence of ints).
        Assuming channel is last.
    :return: the Keras model
    """

    inputs = tf.keras.Input(shape=frame_shape, dtype=tf.uint8)

    # Normalization
    x = inputs / 255

    # Encoding
    x = layers.Conv2D(
        filters=5, kernel_size=3, strides=1, padding="valid",
        activation="relu")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(
        filters=10, kernel_size=3, strides=1, padding="valid",
        activation="relu")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(
        filters=20, kernel_size=3, strides=1, padding="valid",
        activation="relu")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoding
    x = layers.Conv2DTranspose(
        filters=10, kernel_size=3, strides=2, padding="valid",
        activation="relu")(x)
    x = layers.Conv2DTranspose(
        filters=5, kernel_size=3, strides=2, padding="valid",
        activation="relu")(x)
    x = layers.Conv2DTranspose(
        filters=3, kernel_size=(6, 4), strides=2, padding="valid",
        activation="sigmoid")(x)

    # Denormalization
    outputs = x * 255

    # Build
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='frame_autoencoder')
    model.summary()
    model.compile(
        optimizer="adam", loss="mae", metrics=["mae"])

    return model
