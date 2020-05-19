"""Definitions of networks used for feature extraction."""

from abc import abstractmethod
from collections import OrderedDict
import numpy as np
import gym
import tensorflow as tf

from atarieyes import layers
from atarieyes.features import genetic, selector
from atarieyes.layers import BaseLayer, make_layer
from atarieyes.tools import ABC2, AbstractAttribute


class Model(ABC2):
    """Interface of a model.

    Assuming the model is built on initialization.
    `compute_all` is the forward pass used during training, while `predict`
    should be the minimal set of operations needed to compute the output.

    The `model` attribute is the outer model: it can be a tf.keras.Model,
    a tf.keras.layers.Layer or any other tf.Module. Weights are saved and
    restored for this object.
    Some models require a non-standard training step. These can manually
    compute the gradient to be applied, inside the compute_all function.
    The `computed_gradient` must be set to True, in this case.
    If a completely different training step is necessary (not gradient-based),
    one can define it in a method called `train_step`. Otherwise, in __init__,
    we should set self.train_step to False. Train_step() must be compatible
    with Trainer.train_step().
    """

    # This is the main model
    model = AbstractAttribute()

    # Custom training?
    computed_gradient = AbstractAttribute()
    train_step = AbstractAttribute()

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
            inputs=inputs, outputs=outputs, name="FrameAutoencoder")
        model.summary()

        # Store
        self.model = model
        self.computed_gradient = False
        self.train_step = False

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

    def __init__(
        self, *, n_visible, n_hidden, batch_size, l2_const, sparsity_const,
        sparsity_target, trainable=True,
    ):
        """Initialize.

        See BinaryRBM.BernoulliPair for Doc.
        :param trainable: trainable layer boolean flag.
        """

        # Store
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.batch_size = batch_size

        # Two layers
        self.layers = self.BernoulliPair(
            n_visible=n_visible, n_hidden=n_hidden, batch_size=batch_size,
            l2_const=l2_const, sparsity_const=sparsity_const,
            sparsity_target=sparsity_target, trainable=trainable,
        )

        # Register the variables
        model = tf.Module(name="BinaryRBM")
        model.vars = [                           # Fix an order
            getattr(self.layers, v) for v in 
            ("_W", "_bv", "_bh", "_h_distribution", "_saved_v")]
        assert len(model.variables) == len(self.layers.variables)
        assert len(model.trainable_variables) == len(
            self.layers.trainable_variables)

        # Save
        self.model = model
        self.computed_gradient = True
        self.train_step = False

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

        See BernoulliPair.call().
        """

        return self.layers(inputs)

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

        _n_instances = 0

        def __init__(
            self, *, n_visible, n_hidden, batch_size, l2_const, sparsity_const,
            sparsity_target=0.1, trainable=True,
        ):
            """Initialize.

            :param n_visible: number of visible units.
            :param n_hidden: number of hidden units.
            :param batch_size: fixed size of the batch.
            :param l2_const: scale factor of the L2 loss on W.
            :param sparsity_const: scale factor of the sparsity promoting loss.
            :param sparsity_target: 0.1 means hidden units active 10% of the
                time.
            :param trainable: trainable layer flag
            """

            # Super
            BaseLayer.__init__(self, trainable=trainable)

            # Save options
            self.layer_options = dict(
                n_visible=n_visible, n_hidden=n_hidden, batch_size=batch_size,
                l2_const=l2_const, sparsity_const=sparsity_const,
                sparsity_target=sparsity_target, trainable=trainable,
            )

            # Constants
            self._batch_size = batch_size
            self._n_visible = n_visible
            self._n_hidden = n_hidden
            self._l2_const = l2_const
            self._sparsity_const = sparsity_const
            self._h_distribution_target = sparsity_target
            self._h_ema_decay = 0.99

            # Define parameters
            self._W = self.add_weight(
                name="W", shape=(n_visible, n_hidden),
                dtype=tf.float32, trainable=self.layer_options["trainable"],
                initializer=tf.keras.initializers.TruncatedNormal(0, 0.01))
            self._bv = self.add_weight(
                name="bv", shape=(n_visible,), dtype=tf.float32,
                trainable=self.layer_options["trainable"], initializer="zeros")
            self._bh = self.add_weight(
                name="bh", shape=(n_hidden,), dtype=tf.float32,
                trainable=self.layer_options["trainable"], initializer="zeros")

            # Activation of hidden units (used for sparsity promoting)
            self._h_distribution = tf.Variable(
                initial_value=tf.constant(
                    self._h_distribution_target,
                    dtype=tf.float32, shape=[n_hidden]
                ),
                trainable=False, name="h_activations_ema",
            )

            # Buffer of sampled units (persistent CD)
            self._saved_v = tf.Variable(
                np.ones((batch_size, n_visible), dtype=np.float32),
                trainable=False, name="saved_v_sample")

            # Transform functions to layers (optional, for a nice graph)
            str_id = "_" + str(self._n_instances)
            self.expected_h = make_layer(
                "Expected_h" + str_id, self.expected_h)()
            self.expected_v = make_layer(
                "Expected_v" + str_id, self.expected_v)()
            self.sample_h = make_layer(
                "Sample_h" + str_id, self.sample_h)()
            self.sample_v = make_layer(
                "Sample_v" + str_id, self.sample_v)()
            self.free_energy = make_layer(
                "FreeEnergy" + str_id, self.free_energy)()
            self.compute_gradients = make_layer(
                "ComputeGradients" + str_id, self.compute_gradients)()

            # Counter
            type(self)._n_instances += 1

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
            """Definition of a forward pass.

            Compute the most probable hidden units given a batch of values for
            the visible units. This function is deterministic.
            """

            expected_h = self.expected_h(inputs)
            ml_h = tf.where(expected_h > 0.5, 1.0, 0.0)

            return ml_h

        def compute_gradients(self, v):
            """Compute the gradient for the current batch.

            :param v: batch of observed visible vectors.
                Binary float values of shape (batch, n_visible).
            :return: (gradients, tensors). Gradients is a list, one for
                each trainable variable in this layer. Tensors is a dict
                of computed values.
            """

            # Model Markov chain
            expected_model_h = self.expected_h(self._saved_v)
            sampled_model_h = self.sample_h(
                self._saved_v, expected_h=expected_model_h)

            expected_model_v = self.expected_v(sampled_model_h)
            sampled_model_v = self.sample_v(
                sampled_model_h, expected_v=expected_model_v)

            self._saved_v.assign(sampled_model_v)

            # Data samples
            expected_data_h = self.expected_h(v)
            sampled_data_h = self.sample_h(v, expected_h=expected_data_h)
            expected_data_v = self.expected_v(sampled_data_h)

            # CD approximation
            W_gradients_batch = (
                tf.einsum("bi,bj->bij", v, sampled_data_h) -
                tf.einsum("bi,bj->bij", expected_model_v, expected_model_h))
            bv_gradients_batch = (v - expected_model_v)
            bh_gradients_batch = (sampled_data_h - expected_model_h)

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
            h_activations = tf.reduce_mean(sampled_model_h, axis=0)
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
            gradients_vector = [gradients[var] for var in variables]
            if self.trainable:
                assert len(variables) == 3, "Expected: W, bv, bh"
            else:
                assert len(variables) == 0

            # Gradient metrics
            W_loss_gradient_size = tf.math.reduce_max(
                tf.math.abs(W_prel2_gradient))
            W_l2_gradient_size = tf.math.reduce_max(
                tf.math.abs(W_l2_gradient))
            reconstruction_error = tf.reduce_mean(
                tf.math.abs(v - expected_data_v))
            sparsity_gradient_size = tf.math.reduce_max(
                tf.math.abs(sparsity_gradient))

            # Ret
            tensors = dict(
                expected_h=expected_data_h,
                expected_v=expected_data_v,
                W_loss_gradient_size=W_loss_gradient_size,
                W_l2_gradient_size=W_l2_gradient_size,
                reconstruction_error=reconstruction_error,
                h_activations_ema=h_activations,
                sparsity_gradient_size=sparsity_gradient_size,
            )

            return gradients_vector, tensors


class DeepBeliefNetwork(Model):
    """A Deep belief network stacks a number of RBM.

    This iterates the RBM model in a number of layers and train those in a
    greedy manner. I use this to achieve stronger compressions.
    """

    def __init__(self, layers_spec, training_layer=None):
        """Initialize.

        :param layers_spec: (layers specification) A list of dicts, where
            layers_spec[i] contains the init parameters for layer i.
            The layers are BinaryRBM. Make sure that the chained layers
            have compatible shapes.
        :param training_layer: the index of the layer to train. Can be None.
        """

        # Store
        self._layers_spec = layers_spec
        self.training_layer = training_layer
        self.input_shape = (
            layers_spec[0]["batch_size"], layers_spec[0]["n_visible"])

        # Set which layer is trainable
        for i in range(len(self._layers_spec)):
            self._layers_spec[i]["trainable"] = (
                self.training_layer is not None and i == self.training_layer)

        # Layers
        self.layers = []
        for spec in self._layers_spec:
            self.layers.append(
                BinaryRBM(**spec)
            )

        # Register the variables
        model = tf.Module(name="DeepBeliefNetwork")
        model.layers = [inner.model for inner in self.layers]
        assert len(model.variables) == sum(
            [len(inner.model.variables) for inner in self.layers])
        if self.training_layer is not None:
            assert len(model.trainable_variables) == len(
                self.layers[self.training_layer].model.trainable_variables)

        # Save
        self.model = model
        self.computed_gradient = True
        self.train_step = False

    def _forward(self, inputs, from_layer, to_layer):
        """Propagates inputs for a range of layers.

        The forward pass is defined by sampling on each h distribution.

        :param inputs: the model input tensor.
        :param from_layer: start of a range of layers.
        :param to_layer: end (excluded) of a range of layers.
        :return: output of the layer end-1
        """

        for i in range(from_layer, to_layer):
            layer = self.layers[i]
            inputs = layer.layers.sample_h(inputs)

        return inputs

    def compute_all(self, inputs):
        """Compute all tensors.

        """

        # No training
        if self.training_layer is None:
            output = self._forward(inputs, 0, len(self.layers))
            ret = dict(outputs=[output], loss=None, metrics={}, gradients=[])
            return ret

        # Forward
        outputs = self._forward(inputs, 0, self.training_layer)

        # Compute all for training
        ret = self.layers[self.training_layer].compute_all(outputs)
        outputs = self._forward(
            outputs, self.training_layer, self.training_layer + 1)

        # Forward
        outputs = self._forward(
            outputs, self.training_layer + 1, len(self.layers))

        # Ret
        ret["outputs"].insert(0, outputs)
        return ret

    def predict(self, inputs):
        """Repeated layer call()."""

        for layer in self.layers:
            inputs = layer.predict(inputs)

        return inputs

    def images(self, outputs):
        """Images to visualize."""

        return {}

    def histograms(self, outputs):
        """Tensors to visualize."""

        # No training
        if self.training_layer is None:
            return {"outputs/dbn_output": outputs[0]}

        # Training histograms
        training_layer = self.layers[self.training_layer]
        tensors = training_layer.histograms(outputs[1:])

        # Add the model output
        tensors["outputs/dbn_output"] = outputs[0]

        return tensors


class LocalFeatures(Model):
    """Model for binary local features.

    Takes a small portion of the observation (a "region") and encodes it
    in a binary vector. Other models can take this output and use it to
    evaluate the fluents defined in this region.

    Regions and fluents are defined in environment json file.
    See selector module.
    """

    _n_instances = 0

    def __init__(
        self, env_name, region, dbn_spec, training_layer, resize_pixels=500,
    ):
        """Initialize.

        :param env_name: a gym environment name.
        :param region: name of the selected region.
        :param dbn_spec: specification of a `DeepBeliefNetwork`.
            This is the same argument as `layers_spec` in `DeepBeliefNetwork`.
            `n_visible` parameters are not required, because they can be
            inferred.
        :param training_layer: index of the layer to train. Can be None.
        :param resize_pixels: the input region is resized to have less than
            this number of pixels.
        """

        # Store
        self._env_name = env_name
        self._region_name = region
        self._frame_shape = gym.make(env_name).observation_space.shape
        self._dbn_spec = dbn_spec
        self._training_layer = training_layer

        # Preprocessing
        self.preprocessing = layers.LocalFeaturePreprocessing(
            env_name=env_name, region=self._region_name,
            threshold=0.2, max_pixels=resize_pixels,
        )
        self.flatten = tf.keras.layers.Flatten()

        # Compute shapes
        fake_input = np.zeros(shape=(1, *self._frame_shape), dtype=np.uint8)
        self._region_shape = self.preprocessing(fake_input).shape[1:]
        n_elements = self._region_shape.num_elements()

        for spec in self._dbn_spec:
            spec["n_visible"] = n_elements
            n_elements = spec["n_hidden"]

        # DeepBeliefNetwork
        if self._training_layer is not None:
            assert 0 <= self._training_layer < len(dbn_spec)
        self.dbn = DeepBeliefNetwork(dbn_spec, training_layer)

        # Transform functions to layers (optional, for a nice graph)
        str_id = "_" + str(self._n_instances)
        self.predict = make_layer("LF_Predict" + str_id, self.predict)()

        # Counter
        type(self)._n_instances += 1

        # Register the variables
        model = tf.Module(name="LocalFeatures")
        model.layer = self.dbn.model
        assert len(model.variables) == len(self.dbn.model.variables)
        assert len(model.trainable_variables) == len(
            self.dbn.model.trainable_variables)

        # Save
        self.model = model
        self.computed_gradient = True
        self.train_step = False

    def compute_all(self, inputs):
        """Compute all tensors."""

        # Compute all
        out = self.preprocessing(inputs)
        out = self.flatten(out)
        ret = self.dbn.compute_all(out)

        return ret

    def predict(self, inputs):
        """A prediction with the model.

        :param inputs: one batch.
        :return: batch of values for all fluents.
        """

        out = self.preprocessing(inputs)
        out = self.flatten(out)
        out = self.dbn.predict(out)

        return out

    def images(self, outputs):
        """Images to visualize."""

        # Only the first layer can be visualized
        if self._training_layer == 0:

            expected = tf.reshape(outputs[2], (-1, *self._region_shape))
            inputs = tf.reshape(outputs[3], (-1, *self._region_shape))
            return {"region/expected": expected, "region/input": inputs}

        else:
            return {}

    def histograms(self, outputs):
        """Tensors to visualize."""

        return self.dbn.histograms(outputs)


class Fluents(Model):
    """Model for binary local features.

    This class is the outer Model: it represents all binary features (aka
    fluents) defined in each Atari game.  Each environment has all fluents
    defined in its json file: they are grouped in regions.

    The first part is composed of a set of LocalFeatures, one for each region.
    Then, last layer evaluates all fluents.
    """

    def __init__(self, env_name, dbn_spec, training_region, training_layer):
        """Initialize.

        :param env_name: a gym environment name.
        :param dbn_spec: see LocalFeatures `dbn_spec`; this is used for all.
        :param training_region: name of the region to train.
        :param training_layer: index of the region layer to train.
        """

        # Store
        self._env_name = env_name
        self._dbn_spec = dbn_spec
        self._training_region = training_region
        self._training_layer = training_layer
        self._frame_shape = gym.make(env_name).observation_space.shape

        # Read all regions
        env_data = selector.read_back(self._env_name)
        env_data.pop("_frame")
        self._region_names = list(env_data.keys())
        try:
            self._training_i = self._region_names.index(self._training_region)
        except ValueError:
            raise ValueError(
                str(self._training_region) +
                " not in " + str(self._region_names))

        # Collect fluents and their specification
        self.fluents = OrderedDict()
        for region_name in env_data:
            region = env_data[region_name]
            for fluent_name in region["fluents"]:

                if not fluent_name.startswith(region["abbrev"] + "_"):
                    raise ValueError(
                        '"fluent0" for region abbrev "b" must be called '
                        '"b_fluent0"')
                self.fluents[fluent_name] = region["fluents"][fluent_name]

        # One encoding for each region
        self.local_features = [
            LocalFeatures(
                env_name=self._env_name, region=region,
                dbn_spec=self._dbn_spec, training_layer=(
                    self._training_layer if self._training_region == region
                    else None),
            ) for region in self._region_names
        ]

        # NOTE: last layer is still missing because it should be different.

        # Register the variables
        model = tf.Module(name="Fluents")
        model.layers = [f.model for f in self.local_features]
        assert len(model.variables) == sum(
            [len(inner.model.variables) for inner in self.local_features])
        assert len(model.trainable_variables) == sum(
            [len(inner.model.trainable_variables)
                for inner in self.local_features])

        # Save
        self.model = model
        self.computed_gradient = True
        self.train_step = False

    def compute_all(self, inputs):
        """Compute all tensors."""

        # The first is the output of a prediction
        prediction = self.predict(inputs)

        # Then training data follow
        ret = self.local_features[self._training_i].compute_all(inputs)

        # Merge
        ret["outputs"].insert(0, prediction)
        return ret

    def predict(self, inputs):
        """A prediction with the model.

        :param inputs: one batch.
        :return: batch of values for all fluents.
        """

        predictions = [
            region.predict(inputs) for region in self.local_features]
        predictions = tf.concat(predictions, axis=1)
        return predictions

    def images(self, outputs):
        """Images to visualize."""

        # The first come from this model and its not an image
        outputs = outputs[1:]

        # Collect all images and add a namescope
        all_imgs = {}
        for region in self.local_features:
            imgs = region.images(outputs)
            imgs = {
                region._region_name + "/" + name: img
                for name, img in imgs.items()}
            all_imgs.update(imgs)

        return all_imgs

    def histograms(self, outputs):
        """Tensors to visualize."""

        # The first come from this model and its a duplicate
        outputs = outputs[1:]

        # Collect all histograms and add a namescope
        all_hists = {}
        for region in self.local_features:
            hists = region.histograms(outputs)
            hists = {
                region._region_name + "/" + name: hist
                for name, hist in hists.items()}
            all_hists.update(hists)

        return all_hists


class GeneticModel(Model):
    """Bridge between genetic algorithms and standard NN models.

    A GeneticAlgorithm is not a I/O Model. This class is only used to fit
    the algorithm into the same training loop, with merics.
    This model represents the training procedure, not the individuals.
    """

    def __init__(self, ga):
        """Initialize.

        :param ga: a genetic.GeneticAlgorithm instance
        """

        # Check
        if not isinstance(ga, genetic.GeneticAlgorithm):
            raise TypeError("Not a GeneticAlgorithm instance")

        # Store
        self.ga = ga

        # Init
        model = self.GALayer(ga)

        # Save
        self.model = model
        self.computed_gradient = False
        self.train_step = self.ga.train_step

    class GALayer(BaseLayer):
        """Layer wrapper around genetic algorithms."""

        def __init__(self, ga, **layer_kwargs):

            BaseLayer.__init__(self, **layer_kwargs)
            self._vars = [ga.population, ga.fitness]
            self._forward = ga.compute_train_step
            self.built = True

        def call(self, inputs):

            return self._forward(*inputs)

    def predict(self, inputs):
        """Make a prediction with the model."""

        raise NotImplementedError(
            "A generic GeneticAlgorithm is not a I/O model")

    def compute_all(self, inputs):
        """Nothing to compute. Just show the training graph."""

        population, fitness = self.ga.compute_train_step(*inputs)
        mean_fitness = tf.math.reduce_mean(fitness)

        # Ret
        ret = dict(
            outputs=[population, fitness],
            loss=None,
            metrics=dict(mean_fitness=mean_fitness),
            gradients=None,
        )
        return ret

    def images(self, outputs):
        """Returns a set of images to visualize."""

        return {}

    def histograms(self, outputs):
        """Returns a set of tensors to visualize as histograms."""

        # Visualizing population sparsity with 1D PCA
        population = tf.cast(outputs[0], dtype=tf.float32)
        population = tf.reshape(population, shape=(population.shape[0], -1))
        s, u, v = tf.linalg.svd(population)
        population_1d = u[:, 0] * s[0]

        # Histograms
        tensors = {
            "fitness": outputs[1],
            "population_1d": population_1d,
        }

        return tensors


class TestingGM(GeneticModel):
    """Just for debugging."""

    def __init__(self):

        GeneticModel.__init__(
            self, genetic.QueensGA(
                size=10, n_individuals=1000, mutation_p=0.005))

    def compute_all(self, inputs):

        return GeneticModel.compute_all(
            self, (self.ga.population, self.ga.fitness))
