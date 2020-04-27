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
    Some models require a non-standard training step. These can manually
    compute the gradient to be applied, inside the compute_all function.
    The `computed_gradient` attribute indicates this behaviour.
    """

    # This is the keras model
    keras = AbstractAttribute()

    # Custom training? bool
    computed_gradient = AbstractAttribute()

    @abstractmethod
    def predict(self, inputs):
        """Make a prediction with the model."""

    @abstractmethod
    def compute_all(self, inputs):
        """Compute all outputs, loss, metrics (and gradient optionally).

        :param inputs: one batch.
        :return: a dict of {"outputs": out, "loss": loss,
            "metrics": metrics, "gradients": grad}
            Where out is a sequence of output tensors, loss is the training
            loss, metrics is a dictionary of metrics, grad is an optional
            list of computed gradients.
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
            env_name=env_name, out_size=(80, 80), grayscale=True,
            resize_method="nearest")
        self.scale_to = layers.ScaleTo(from_range=(-1, 1), to_range=(0, 255))
        self.loss = layers.LossMAE()

        # Keras model
        inputs = tf.keras.Input(shape=frame_shape, dtype=tf.uint8)
        ret = self.compute_all.python_function(inputs)
        outputs = (*ret["outputs"], ret["loss"])

        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="frame_autoencoder")
        model.summary()

        # Store
        self.keras = model
        self.computed_gradient = False

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
        ret = dict(
            outputs=(encoded, decoded), loss=loss, metrics={}, gradients=None)
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
                    filters=32, kernel_size=4, strides=2, padding="reflect",
                    activation="relu"),
                layers.ConvBlock(
                    filters=32, kernel_size=4, strides=2, padding="reflect",
                    activation="relu"),
                layers.ConvBlock(
                    filters=16, kernel_size=4, strides=2, padding="reflect",
                    activation="relu"),
            ]

            # Super
            BaseLayer.build(self, input_shape)

    class Decoder(BaseLayer):

        def build(self, input_shape):
            """Build the layer."""

            self.layers_stack = [

                layers.ConvBlock(
                    filters=32, kernel_size=4, strides=2, padding="same",
                    activation="relu", transpose=True),
                layers.ConvBlock(
                    filters=32, kernel_size=4, strides=2, padding="same",
                    activation="relu", transpose=True),
                layers.ConvBlock(
                    filters=1, kernel_size=4, strides=2, padding="same",
                    activation="tanh", transpose=True),
            ]

            # Super
            BaseLayer.build(self, input_shape)


class BinaryRBM(Model):
    """Model of a Restricted Boltzmann Machine.

    This model assumes binary hidden and observed units.
    """

    def __init__(self, n_visible, n_hidden):
        """Initialize.

        :param n_visible: number of visible units.
        :param n_hidden: number of hidden units.
        """

        # Store
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Two layers
        self.layers = self.BernoulliPair(
            n_visible=n_visible, n_hidden=n_hidden)

        # Keras model
        inputs = tf.keras.Input(shape=[n_visible], dtype=tf.float32)
        ret = self.compute_all.python_function(inputs)
        outputs = (
            *ret["outputs"], *ret["gradients"], *ret["metrics"].values())

        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="BinaryRBM")

        # Let keras register the variables
        model._variables = self.layers.trainable_variables
        assert model.trainable_variables == self.layers.trainable_variables

        # Save
        self.keras = model
        self.computed_gradient = True

    @tf.function
    def compute_all(self, inputs):
        """Compute all tensors."""

        # Compute gradients and all
        inputs = tf.cast(inputs, tf.float32)
        gradients, tensors = self.layers.compute_gradients(inputs)
        free_energy = self.layers.free_energy(inputs)

        # Ret
        ret = dict(
            outputs=(tensors["expected_h"], tensors["expected_v"]),
            loss=None,
            metrics=dict(free_energy=free_energy),
            gradients=gradients)
        return ret

    @tf.function
    def predict(self, inputs):
        """Make a prediction with the model.

        An RBM has no real output. I define a prediction as the expected value
        of the hidden layer.

        :param inputs: one batch
        :return output: the expected value of the hidden layer
        """

        # Forward
        inputs = tf.cast(inputs, tf.float32)
        output = self.layers.expected_h(inputs)

        return output

    @staticmethod
    def output_images(outputs):
        """Get images from outputs."""

        return {}

    class BernoulliPair(BaseLayer):
        """A pair of layers composed of binary units.

        To avoid type casts, all values are assumed floats, even for binary
        units.
        """

        def __init__(self, *, n_visible, n_hidden, batch_size=1):
            """Initialize.

            :param n_visible: number of visible units.
            :param n_hidden: number of hidden units.
            :param batch_size: fixed size of the batch.
            """

            # Super
            BaseLayer.__init__(self)

            # Save options
            self.layer_options = dict(
                n_visible=n_visible, n_hidden=n_hidden, batch_size=batch_size)

            # Constants
            self._batch_size = batch_size
            self._n_visible = n_visible
            self._n_hidden = n_hidden

            # Define parameters
            self._W = self.add_weight(
                name="W", shape=(n_visible, n_hidden),
                dtype=tf.float32, trainable=True)
            self._bv = self.add_weight(
                name="bv", shape=(n_visible,),
                dtype=tf.float32, trainable=True)
            self._bh = self.add_weight(
                name="bh", shape=(n_hidden,),
                dtype=tf.float32, trainable=True)

            # Already built
            self.built = True

        def expected_h(self, v):
            """Expected hidden vector (a probability).

            :param v: batch of observed visible vectors.
                Binary values of shape (batch, n_visible).
            :return: batch of expected values of the hidden layers given the
                observations.
            """

            mean_h = tf.math.sigmoid(
                tf.einsum("ji,bj->bi", self._W, v) + self._bh)
            return tf.identity(mean_h, name="expected_h")

        def expected_v(self, h):
            """Expected visible vector (a probability).

            :param h: batch of "observed" hidden vectors.
                Binary values of shape (batch, n_hidden).
            :return: batch of expected values of the visible layers given the
                observations.
            """

            mean_v = tf.math.sigmoid(
                tf.einsum("ij,bj->bi", self._W, h) + self._bv)
            return tf.identity(mean_v, name="expected_v")

        def sample_h(self, v, expected_h=None):
            """Sample hidden vector given visible.

            :param v: batch of observed visible vectors.
                Binary values of shape (batch, n_visible).
            :param expected_h: use these expected values, if already available.
            :return: batch of sampled hidden units.
            """

            output_shape = (self._batch_size, self._n_hidden)

            mean_h = self.expected_h(v) if expected_h is None else expected_h
            uniform_samples = tf.random.uniform(output_shape, 0, 1)
            binary_samples = tf.where(uniform_samples < mean_h, 1.0, 0.0)

            return tf.identity(binary_samples, name="sampled_h")

        def sample_v(self, h, expected_v=None):
            """Sample visible vector given hidden.

            :param h: batch of observed hidden vectors.
                Binary values of shape (batch, n_hidden).
            :param expected_v: use these expected values, if already available.
            :return: batch of sampled visible units.
            """

            output_shape = (self._batch_size, self._n_visible)

            mean_v = self.expected_v(h) if expected_v is None else expected_v
            uniform_samples = tf.random.uniform(output_shape, 0, 1)
            binary_samples = tf.where(uniform_samples < mean_v, 1.0, 0.0)

            return tf.identity(binary_samples, name="sampled_v")

        def free_energy(self, v):
            """Compute the free energy.

            RBM is an energy based model. Given (v,h), The Energy is a measure
            of the likelihood of this observation. Since h is hidden, we can
            marginalize it out to compute the free energy.

            :param v: batch of observed visible vectors.
                Binary float values of shape (batch, n_visible).
            :return: a scalar
            """

            term0 = -tf.einsum("j,bj->b", self._bv, v)
            term1 = tf.einsum("ji,bj->bi", self._W, v) + self._bh
            term1 = -tf.math.reduce_sum(term1, axis=1)
            free_energy = term0 + term1

            return tf.identity(free_energy, "free_energy")

        def call(self, inputs):
            """I define a forward pass as computing the expected h.

            See expected_h().
            """

            return self.expected_h(inputs)

        def compute_gradients(self, v):
            """Compute the gradient for the current batch.

            The method is one-step Contrastive Divergence, CD-1.

            :param v: batch of observed visible vectors.
                Binary float values of shape (batch, n_visible).
            :return: (gradients, tensors). Gradients is a list, one for
                each trainable variable in this layer. Tensors is a dict
                of computed values.
            """

            # Sampling
            expected_h = self.expected_h(v)
            sampled_h = self.sample_h(v, expected_h=expected_h)
            expected_v = self.expected_v(sampled_h)
            sampled_v = self.sample_v(sampled_h, expected_v=expected_v)
            expected_h2 = self.expected_h(sampled_v)

            # CD-1 approximation
            W_gradients_batch = (
                tf.einsum("bi,bj->bij", v, expected_h) -
                tf.einsum("bi,bj->bij", sampled_v, expected_h2))
            bv_gradients_batch = (v - sampled_v)
            bh_gradients_batch = (expected_h - expected_h2)

            # Average batch dimension
            W_gradient = tf.math.reduce_mean(W_gradients_batch, axis=0)
            bv_gradient = tf.math.reduce_mean(bv_gradients_batch, axis=0)
            bh_gradient = tf.math.reduce_mean(bh_gradients_batch, axis=0)
            gradients = dict(
                W=tf.identity(W_gradient, name="W_gradient"),
                bv=tf.identity(bv_gradient, name="bv_gradient"),
                bh=tf.identity(bh_gradient, name="bh_gradient"),
            )

            # Return gradients with the correct association
            variables = [var.name for var in self.trainable_variables]
            variables = [
                (name[:name.find(":")] if ":" in name else name)
                for name in variables]
            assert len(variables) == 3, "Expected: W, bv, bh"
            gradients_vector = [gradients[var] for var in variables]

            # Ret
            tensors = dict(expected_h=expected_h, expected_v=expected_v)

            return gradients_vector, tensors


class LocalFeature(Model):
    # TODO
    pass
