"""Definitions of networks used for feature extraction."""

from abc import abstractmethod
import numpy as np
import gym
import tensorflow as tf

from atarieyes import layers
from atarieyes.layers import BaseLayer, make_layer
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

    @abstractmethod
    def images(self, outputs):
        """Returns a set of images to visualize.

        :param outputs: a sequence of outputs, as returned by
            compute_all["outputs"].
        :return: a dict of {name: images}. The dict can be empty.
        """

    @abstractmethod
    def histograms(self, outputs):
        """Returns a set of tensors to visualize as histograms.

        :param outputs: a sequence of outputs, as returned by
            compute_all["outputs"].
        :return: a dict of {name: tensor}. The dict can be empty.
        """


class FrameAutoencoder(Model):
    """This model encodes a single frame.

    This is an autoencoder which encodes a single frame of the game.
    """

    def __init__(self, env_name):
        """Initialize.

        :param env_name: a gym environment name.
        """

        frame_shape = gym.make(env_name).observation_space.shape

        # Define structure
        self.encoder = self.Encoder()
        self.decoder = self.Decoder()

        self.preprocessing = layers.ImagePreprocessing(
            env_name=env_name, out_size=(80, 80), grayscale=True,
            resize_method="nearest")
        self.scale_to = layers.ScaleTo(from_range=(-1, 1), to_range=(0, 255))
        self.loss = layers.LossMSE()

        # Keras model
        inputs = tf.keras.Input(shape=frame_shape, dtype=tf.uint8)
        ret = self.compute_all(inputs)
        outputs = (*ret["outputs"], ret["loss"])

        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="frame_autoencoder")
        model.summary()

        # Store
        self.keras = model
        self.computed_gradient = False

    def predict(self, inputs):
        """Make predictions."""

        inputs = self.preprocessing(inputs)
        return self.encoder(inputs)

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
            outputs=[encoded, decoded], loss=loss, metrics={}, gradients=None)
        return ret

    def images(self, outputs):
        """Images to visualize."""

        return {"decoded": outputs[1]}

    def histograms(self, outputs):
        """Tensors to visualize."""

        return {}

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
    Assuming every tensor is a float.
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
        ret = self.compute_all(inputs)
        outputs = (
            *ret["outputs"], *ret["gradients"], *ret["metrics"].values())

        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="BinaryRBM")

        # Let keras register the variables
        model._saved_layers = self.layers
        assert model.trainable_variables == self.layers.trainable_variables

        # Save
        self.keras = model
        self.computed_gradient = True

    def compute_all(self, inputs):
        """Compute all tensors."""

        # Compute gradients and all
        gradients, tensors = self.layers.compute_gradients(inputs)
        free_energy = tf.math.reduce_mean(self.layers.free_energy(inputs))

        # Ret
        ret = dict(
            outputs=[
                tensors["expected_h"],
                tensors["expected_v"],
                inputs,
                tensors["h_activations_ema"],
            ],
            loss=None,
            metrics=dict(
                free_energy=free_energy,
                W_loss_gradient=tensors["W_loss_gradient_size"],
                W_l2_gradient=tensors["W_l2_gradient_size"],
                reconstruction_error=tensors["reconstruction_error"],
                sparsity_gradient_size=tensors["sparsity_gradient_size"],
            ),
            gradients=gradients)
        return ret

    def predict(self, inputs):
        """Make a prediction with the model.

        An RBM has no real output. I define a prediction as the expected value
        of the hidden layer.

        :param inputs: one batch
        :return output: the expected value of the hidden layer
        """

        # Forward
        output = self.layers.expected_h(inputs)

        return output

    def images(self, outputs):
        """Images to visualize."""

        return {}

    def histograms(self, outputs):
        """Tensors to visualize."""

        tensors = {
            "weigths/W": self.layers._W.value(),
            "weigths/bv": self.layers._bv.value(),
            "weigths/bh": self.layers._bh.value(),
            "outputs/expected_h": outputs[0],
            "outputs/expected_v": outputs[1],
            "outputs/h_activations_ema": outputs[3],
        }

        return tensors

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
            self._l2_const = 0.05
            self._sparsity_const = 0.2
            self._h_distribution_target = 0.0
            self._h_ema_decay = 0.99

            # Define parameters
            self._W = self.add_weight(
                name="W", shape=(n_visible, n_hidden),
                dtype=tf.float32, trainable=True,
                initializer=tf.keras.initializers.TruncatedNormal(0, 0.01))
            self._bv = self.add_weight(
                name="bv", shape=(n_visible,),
                dtype=tf.float32, trainable=True, initializer="zeros")
            self._bh = self.add_weight(
                name="bh", shape=(n_hidden,),
                dtype=tf.float32, trainable=True, initializer="zeros")

            # Activation of hidden units (used for sparsity promoting)
            self._h_distribution = tf.Variable(
                initial_value=tf.constant(
                    self._h_distribution_target,
                    dtype=tf.float32, shape=[n_hidden]
                ),
                trainable=False, name="h_activations_ema",
            )

            # Buffer TODO: use for persistent CD.
            #self._saved_v = tf.Variable

            # Transform functions to layers (optional, for a nice graph)
            self.expected_h = make_layer("Expected_h", self.expected_h)()
            self.expected_v = make_layer("Expected_v", self.expected_v)()
            self.sample_h = make_layer("Sample_h", self.sample_h)()
            self.sample_v = make_layer("Sample_v", self.sample_v)()
            self.free_energy = make_layer("FreeEnergy", self.free_energy)()
            self.compute_gradients = make_layer(
                "ComputeGradients", self.compute_gradients)()

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
            term1 = -tf.math.reduce_sum(tf.math.softplus(term1), axis=1)
            free_energy = term0 + term1

            return tf.identity(free_energy, "free_energy")

        def call(self, inputs):
            """I define a forward pass as computing the expected h.

            See expected_h().
            """

            return self.expected_h(inputs)

        def compute_gradients(self, v):
            """Compute the gradient for the current batch.

            The method is Contrastive Divergence, CD-k.

            :param v: batch of observed visible vectors.
                Binary float values of shape (batch, n_visible).
            :return: (gradients, tensors). Gradients is a list, one for
                each trainable variable in this layer. Tensors is a dict
                of computed values.
            """

            # Init Gibbs sampling
            gibbs_sweeps = 1  # k
            sampled_v = v
            expected_h0 = self.expected_h(sampled_v)
            expected_h = expected_h0
            sampled_h0 = self.sample_h(sampled_v, expected_h=expected_h)
            sampled_h = sampled_h0

            # Sampling
            for i in range(gibbs_sweeps):
                expected_v = self.expected_v(sampled_h)
                sampled_v = self.sample_v(sampled_h, expected_v=expected_v)
                expected_h = self.expected_h(sampled_v)
                sampled_h = self.sample_h(sampled_v, expected_h=expected_h)

            # CD approximation
            W_gradients_batch = (
                tf.einsum("bi,bj->bij", v, sampled_h0) -
                tf.einsum("bi,bj->bij", expected_v, expected_h))
            bv_gradients_batch = (v - expected_v)
            bh_gradients_batch = (sampled_h0 - expected_h)

            # Average batch dimension
            W_gradient = tf.math.reduce_mean(W_gradients_batch, axis=0)
            bv_gradient = tf.math.reduce_mean(bv_gradients_batch, axis=0)
            bh_gradient = tf.math.reduce_mean(bh_gradients_batch, axis=0)

            # Negatives for gradient Descent
            W_gradient = -W_gradient
            bv_gradient = -bv_gradient
            bh_gradient = -bh_gradient

            # Regularization
            W_prel2_gradient = W_gradient
            W_l2_gradient = self._l2_const * self._W
            W_gradient += W_l2_gradient

            # Regularization on activations (like a sparsity promoting loss)
            h_activations = tf.reduce_mean(sampled_h, axis=0)
            self._h_distribution.assign_sub(
                (1-self._h_ema_decay) * (self._h_distribution - h_activations))
            h_activations = self._h_distribution.value()
            sparsity_gradient = self._sparsity_const * (
                h_activations - self._h_distribution_target)
            W_gradient += sparsity_gradient
            bh_gradient += sparsity_gradient
            
            # Collect and rename
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

            # Gradient metrics
            W_loss_gradient_size = tf.math.reduce_max(
                tf.math.abs(W_prel2_gradient))
            W_l2_gradient_size = tf.math.reduce_max(
                tf.math.abs(W_l2_gradient))
            reconstruction_error = tf.reduce_mean(
                tf.math.abs(v - expected_v))
            sparsity_gradient_size = tf.math.reduce_max(
                tf.math.abs(sparsity_gradient))

            # Ret
            tensors = dict(
                expected_h=expected_h,
                expected_v=expected_v,
                W_loss_gradient_size=W_loss_gradient_size,
                W_l2_gradient_size=W_l2_gradient_size,
                reconstruction_error=reconstruction_error,
                h_activations_ema=h_activations,
                sparsity_gradient_size=sparsity_gradient_size,
            )

            return gradients_vector, tensors


class LocalFluent(Model):
    """Model for binary local features.

    A LocalFluent is a binary function of a small portion of the observation.
    For each frame of the game, a LocalFluent has a fixed truth value which
    can be computed from just a small portion of the image.
    This model is composed by a RBM (and some other parts that will be added).
    """
    # TODO: n_hidden and region should be parametric. How?

    def __init__(self, env_name, region="green_bar"):
        """Initialize.

        :param env_name: a gym environment name.
        :param region: name of the selected region.
        """

        # Store
        self._env_name = env_name
        self._region_name = region
        self._frame_shape = gym.make(env_name).observation_space.shape

        # Preprocessing
        self.preprocessing = layers.LocalFeaturePreprocessing(
            env_name=env_name, region=region,
            threshold=0.2, max_pixels=500,
        )
        self.flatten = tf.keras.layers.Flatten()

        # Compute shape
        fake_input = np.zeros(shape=(1, *self._frame_shape), dtype=np.uint8)
        self._region_shape = self.preprocessing(fake_input).shape[1:]
        n_pixels = self._region_shape.num_elements()

        # RBM block
        self.rbm = BinaryRBM(n_visible=n_pixels, n_hidden=50)

        # Keras model
        inputs = tf.keras.Input(shape=self._frame_shape, dtype=tf.uint8)
        ret = self.compute_all(inputs)
        outputs = (
            *ret["outputs"], *ret["gradients"], *ret["metrics"].values())

        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="LocalFluentModel")

        # Let keras register the variables
        model._saved_layers = self.rbm.keras
        assert model.trainable_variables == self.rbm.keras.trainable_variables

        # Save
        self.keras = model
        self.computed_gradient = True

    def compute_all(self, inputs):
        """Compute all tensors."""

        # Compute all
        out = self.preprocessing(inputs)
        out = self.flatten(out)
        out = self.rbm.compute_all(out)

        # Last two outputs are images
        expected_v = out["outputs"][1]
        out["outputs"][1] = tf.reshape(expected_v, [-1, *self._region_shape])
        input_v = out["outputs"][2]
        out["outputs"][2] = tf.reshape(input_v, [-1, *self._region_shape])

        return out

    def predict(self, inputs):
        """Predict the most probable value of the fluent.

        :param inputs: one batch.
        :return: batch of zeros and ones.
        """

        # Maximum likelihood
        out = self.preprocessing(inputs)
        out = self.flatten(out)
        expected_h = self.rbm.predict(out)
        ml_h = tf.where(expected_h > 0.5, 1.0, 0.0)

        return ml_h

    def images(self, outputs):
        """Images to visualize."""

        return {"region/expected": outputs[1], "region/input": outputs[2]}

    def histograms(self, outputs):
        """Tensors to visualize."""

        return self.rbm.histograms(outputs)
