"""Various custom layers. Any conceptual block can be a layer."""

import inspect
import tensorflow as tf
from tensorflow.keras import layers


class BaseLayer(layers.Layer):
    """Base class for all layers and model parts.

    This is mainly used to create namespaces in TensorBoard graphs.
    This must be subclassed, not instantiated directly.
    Follow the keras subclassing API to properly subclass.
    There are two special members that can be used in subclasses:
        - self.layers_stack is the list of inner computations. Sequential
          layers can simply push to this list without the need of defining
          call().
        - self.layer_options can be filled with subclasses' constructor
          arguments, without the need of defining get_config().
    """

    # Number all layers. Map classes to count
    _layers_count = {}

    def __init__(self, *, verbose=False, **kwargs):
        """Initialize.

        This must be called from subclasses.

        :param verbose: print debug informations for this layer.
        :param kwargs: Arguments forwarded to keras Layer.
        :raises: TypeError: ensure a correcty use of this base class.
        """

        # Only from subclasses
        this_class = self.__class__
        if this_class is BaseLayer:
            raise TypeError("BaseLayer is an abstract class")

        # Add this layer to count
        if this_class not in BaseLayer._layers_count:
            BaseLayer._layers_count[this_class] = 1
        else:
            BaseLayer._layers_count[this_class] += 1

        # Choose a name
        if "name" not in kwargs:
            name = this_class.__name__
            number = BaseLayer._layers_count[this_class]
            if number > 1:
                name += "_" + str(number - 1)
            kwargs["name"] = name

        # Initializations
        defaults = {
            "layers_stack": [],              # Empty layer list
            "layer_options": {},             # No options
        }
        for key in defaults:
            if not hasattr(self, key):
                setattr(self, key, defaults[key])

        # Parameters (non persistent)
        self._verbose_layer = verbose

        # Super
        layers.Layer.__init__(self, **kwargs)

    def call(self, inputs):
        """Sequential layer by default."""

        # This must be overridden if layers_stack is not used
        if not self.layers_stack:
            raise NotImplementedError(
                "call() must be overridden if self.layers_stack is not used.")

        # deb
        if self._verbose_layer:
            print(self.name)
            print(" - input shape:", inputs.shape)

        # Sequential model by default
        for layer in self.layers_stack:
            inputs = layer(inputs)

            # deb
            if self._verbose_layer:
                print(" - output shape:", inputs.shape)

        return inputs

    def get_config(self):
        """Subclasses should use layer_options, or override."""

        config = layers.Layer.get_config(self)
        config.update(self.layer_options)
        return config


def make_layer(name, function):
    """Create a Keras layer that calls the function.

    This function creates a layer, that is a tensorflow namespace.
    The first argument of the function must be for the inputs, and other
    keyword-only options may follow. The created layer is a class that can be
    initialized with arguments. These arguments will be used for 'function', at
    each call. Other arguments are passed to keras.layers.Layer.

    :param name: the name of the new layer.
    :param function: the function to call;
        the inputs must be the first argument.
    :return: a new Keras layer that calls the function
    """

    def __init__(self, **kwargs):
        """Initialize the layer."""

        # Store the inner function
        self._function = function

        # Set the argument defaults
        signature = inspect.signature(function)
        arg_names = list(signature.parameters)
        function_kwargs = {
            arg: kwargs[arg] for arg in kwargs if arg in arg_names}
        layer_kwargs = {
            arg: kwargs[arg] for arg in kwargs if arg not in arg_names}
        self._function_bound_args = signature.bind_partial(**function_kwargs)

        # Super
        BaseLayer.__init__(self, **layer_kwargs)

    def call(self, inputs, **kwargs):
        """Layer call method."""

        # Collect args
        defaults = self._function_bound_args.arguments
        kwargs.pop("training", None)

        # Merge args
        kwargs_all = dict(defaults)
        kwargs_all.update(kwargs)

        # Run
        return self._function(inputs, **kwargs_all)

    # Define a new layer
    LayerClass = type(name, (BaseLayer,), {'__init__': __init__, 'call': call})
    return LayerClass


class layerize:
    """Function decorator to create Keras layers.

    This decorator simplifies the use of 'make_layer'.
    See make_layer() for further help. Use it as:

        @layerize("MyLoss", globals())
        def my_loss(inputs, other_args):
            pass
    """

    def __init__(self, name, scope=None, **kwargs):
        """Initialize.

        :param name: name of the layer.
        :param scope: namespace where to define the layer (such as globals()).
            If omitted, the class is defined in this module.
        """

        self.name = name
        self.scope = scope if scope is not None else globals()

    def __call__(self, function):
        """Decorator: create the layer.

        :param function: the decorated TF function.
        :return: the original function. The layer is defined as a side-effect.
        """

        # Create and export
        NewLayer = make_layer(name=self.name, function=function)
        self.scope[self.name] = NewLayer

        return function


@layerize("ScaleTo")
def scale_to(inputs, from_range, to_range):
    """Lineary scale input values from an input range to the output range.

    :param inputs: input values.
    :param from_range: a sequence of two values, min and max of the input data.
    :param to_range: a sequence of two values, the output range.
    :return: scaled inputs.
    """

    scale = (to_range[1] - to_range[0]) / (from_range[1] - from_range[0])
    out = (inputs - from_range[0]) * scale + to_range[0]
    return out


@layerize("CrobToEnvBox")
def crop_to_env_box(inputs, env_name, strict=False):
    """Crop frames of a atari gym environment.
    
    The purpose of this function is to focus on the important part of the frame
    for each game. Each game has its own box.

    :param inputs: a batch of frames 4D.
    :param env_name: a gym environment name.
    :param strict: if true, raises an error when the env has no box defined.
    :return: batch of cropped frames.
    :raises: ValueError: if unknown env, and strict.
    """

    boxes = {
        "Breakout-v4": (slice(32, 196), slice(8, 152)),
    }

    # Unknown
    if env_name not in boxes:
        if strict:
            raise ValueError("No box defined for " + str(env_name))
        else:
            return inputs

    # Crop
    box = boxes[env_name]
    return inputs[:, box[0], box[1], :]


@layerize("ImagePreprocessing")
def image_preprocessing(inputs, env_name):
    """Input preprocessing function.

    :param inputs: input values.
    :param env_name: name of an atari environment.
    :return: transformed images.
    """

    # Crop
    inputs = crop_to_env_box(inputs, env_name, strict=True)

    # Cast from uint image
    inputs = tf.cast(inputs, tf.float32)

    # Choose an input range
    inputs = scale_to(inputs, from_range=(0, 255), to_range=(-1, 1))

    # Square shape is easier to handle
    inputs = tf.image.resize(inputs, size=(160, 160))

    return inputs


@layerize("LossMAE")
def loss_mae(inputs):
    """Mean absolute error.

    :param inputs: a sequence of (y_true, y_pred) with the same shape.
    :return: scalar
    """

    y_true, y_pred = inputs
    loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    return loss


@layerize("LossMSE")
def loss_mse(inputs):
    """Mean squared error.

    :param inputs: a sequence of (y_true, y_pred) with the same shape.
    :return: scalar
    """

    y_true, y_pred = inputs
    loss = tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))
    return loss
