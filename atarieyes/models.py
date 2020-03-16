"""Neural networks for feature extraction."""

from abc import abstractmethod
import tensorflow as tf
from tensorflow.keras import layers

from atarieyes.layers import BaseLayer, ScaleTo, LossMAE
from atarieyes.tools import ABC2, AbstractAttribute


class Model(ABC2):
    """Interface of a model.

    Assuming the model is built on initialization.
    The purpose of this interface is efficiency: usually many computations are
    not required if we just need to make predictions, not train the model.

    The `keras` attribute is a keras model.
    """

    @abstractmethod
    def predict(self, inputs):
        """Make a prediction with the model."""

    @abstractmethod
    def compute_all(self, inputs):
        """Compute output, loss, and metrics for training.
        
        :param inputs: one batch.
        :return: a dict of {"output": out, "loss": loss, "metrics": metrics}.
            Where out is the result of predict(), loss is the training loss,
            metrics is a dictionary of metrics.
        """

    # This is the compiled keras model
    keras = AbstractAttribute()


class SingleFrameModel(Model):
    """This model encodes a single frame.

    This is an autoencoder which encodes a single frame of the game.
    """

    def __init__(self, frame_shape):
        """Initialize.

        :param frame_shape: the shape of the input frame (sequence of ints).
            Assuming channel is last.
        """

        # Define structure
        self.encoder = self.Encoder()
        self.decoder = self.Decoder()

        self.loss = LossMAE()

        # Keras model
        inputs = tf.keras.Input(shape=frame_shape, dtype=tf.uint8)
        ret = self.compute_all(inputs)
        outputs = (ret["output"], ret["loss"])

        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name='frame_autoencoder')
        model.summary()

        # Store
        self.keras = model

    def predict(self, inputs):
        """Make predictions."""

        return self.encoder(inputs)

    def compute_all(self, inputs):
        """Compute for training."""

        # Forward
        outputs = self.encoder(inputs)
        reconstruction = self.decoder(outputs)
        loss = self.loss((inputs, reconstruction))

        # Ret
        ret = {"output": outputs, "loss": loss, "metrics": {}}
        return ret

    class Encoder(BaseLayer):

        def build(self, input_shape):
            """Build the layer."""

            self.layers_stack = [

                ScaleTo(from_range=(0, 255), to_range=(0, 1)),

                layers.Conv2D(
                    filters=5, kernel_size=3, strides=1, padding="valid",
                    activation="relu"),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Conv2D(
                    filters=10, kernel_size=3, strides=1, padding="valid",
                    activation="relu"),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Conv2D(
                    filters=20, kernel_size=3, strides=1, padding="valid",
                    activation="relu"),
                layers.MaxPooling2D((2, 2), padding="same"),
            ]

            # Super
            BaseLayer.build(self, input_shape)

    class Decoder(BaseLayer):

        def build(self, input_shape):
            """Build the layer."""

            self.layers_stack = [

                layers.Conv2DTranspose(
                    filters=10, kernel_size=3, strides=2, padding="valid",
                    activation="relu"),
                layers.Conv2DTranspose(
                    filters=5, kernel_size=3, strides=2, padding="valid",
                    activation="relu"),
                layers.Conv2DTranspose(
                    filters=3, kernel_size=(6, 4), strides=2, padding="valid",
                    activation="sigmoid"),

                ScaleTo(from_range=(0, 1), to_range=(0, 255)),
            ]

            # Super
            BaseLayer.build(self, input_shape)

