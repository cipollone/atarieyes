"""Definitions of networks used for feature extraction."""

from abc import abstractmethod
import tensorflow as tf

from atarieyes import layers
from atarieyes.layers import BaseLayer
from atarieyes.tools import ABC2, AbstractAttribute


class Model(ABC2):
    """Interface of a model.

    Assuming the model is built on initialization.
    The purpose of this interface is efficiency: usually many computations are
    not required if we just need to make predictions, not train the model.

    The `keras` attribute is a keras model.
    """

    # This is the keras model
    keras = AbstractAttribute()

    @abstractmethod
    def predict(self, inputs):
        """Make a prediction with the model."""

    @abstractmethod
    def compute_all(self, inputs):
        """Compute all outputs, loss, and metrics.

        :param inputs: one batch.
        :return: a dict of {"outputs": out, "loss": loss, "metrics": metrics}.
            Where out is a sequence of output tensors, loss is the training
            loss, metrics is a dictionary of metrics.
        """

    @staticmethod
    @abstractmethod
    def output_images(outputs):
        """Returns the images from all outputs.

        :param outputs: a sequence of outputs, as returned by
            compute_all["outputs"].
        :return: a dict of {name: images} where each 'images' is a batch
            extracted from outputs. The dict can be empty.
        """


class FrameAutoencoder(Model):
    """This model encodes a single frame.

    This is an autoencoder which encodes a single frame of the game.
    """

    def __init__(self, frame_shape, env_name):
        """Initialize.

        :param frame_shape: the shape of the input frame (sequence of ints).
            Assuming channel is last.
        :param env_name: a gym environment name.
        """

        # Define structure
        self.encoder = self.Encoder(verbose=True)
        self.decoder = self.Decoder(verbose=True)

        self.preprocessing = layers.ImagePreprocessing(
            env_name=env_name, out_size=(128, 128))
        self.scale_to = layers.ScaleTo(from_range=(-1, 1), to_range=(0, 255))
        self.loss = layers.LossMSE()

        # Keras model
        inputs = tf.keras.Input(shape=frame_shape, dtype=tf.uint8)
        ret = self.compute_all.python_function(inputs)
        outputs = (*ret["outputs"], ret["loss"])

        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="frame_autoencoder")
        model.summary()

        # Store
        self.keras = model

    @tf.function
    def predict(self, inputs):
        """Make predictions."""

        inputs = self.preprocessing(inputs)
        return self.encoder(inputs)

    @tf.function
    def compute_all(self, inputs):
        """Compute all tensors."""

        # Forward
        inputs = self.preprocessing(inputs)
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        loss = self.loss((inputs, decoded))

        # Image
        decoded = self.scale_to(decoded)
        decoded = tf.cast(decoded, tf.uint8)

        # Ret
        ret = {"outputs": (encoded, decoded), "loss": loss, "metrics": {}}
        return ret

    @staticmethod
    def output_images(outputs):
        """Get images from outputs."""

        return {"decoded": outputs[1]}

    class Encoder(BaseLayer):

        def build(self, input_shape):
            """Build the layer."""

            self.layers_stack = [

                layers.ConvBlock(
                    filters=32, kernel_size=8, strides=4, padding="reflect",
                    activation="selu"),
                layers.ConvBlock(
                    filters=32, kernel_size=4, strides=2, padding="same",
                    activation="selu"),
                layers.ConvBlock(
                    filters=32, kernel_size=4, strides=2, padding="same",
                    activation="selu"),
            ]

            # Super
            BaseLayer.build(self, input_shape)

    class Decoder(BaseLayer):

        def build(self, input_shape):
            """Build the layer."""

            self.layers_stack = [

                layers.ConvBlock(
                    filters=32, kernel_size=4, strides=2, padding="same",
                    activation="selu", transpose=True),
                layers.ConvBlock(
                    filters=32, kernel_size=4, strides=2, padding="same",
                    activation="selu", transpose=True),
                layers.ConvBlock(
                    filters=3, kernel_size=8, strides=4, padding="same",
                    activation="tanh", transpose=True),
            ]

            # Super
            BaseLayer.build(self, input_shape)
