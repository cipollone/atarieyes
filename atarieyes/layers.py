"""Various custom layers. Any conceptual block can be a layer."""

import inspect
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

    def __init__(self, **kwargs):
        """Initialize.

        This must be called from subclasses.

        :param kwargs: Arguments forwarded to keras Layer.
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

        # Super
        layers.Layer.__init__(self, **kwargs)

    def call(self, inputs):
        """Sequential layer by default."""

        # This must be overridden if layers_stack is not used
        if not self.layers_stack:
            raise NotImplementedError(
                "call() must be overridden if self.layers_stack is not used.")

        # Sequential model by default
        for layer in self.layers_stack:
            inputs = layer(inputs)
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
    options may follow. The created layer is a class that can be initialize
    with arguments. These arguments will be used for 'function', at each call.
    Other arguments are passed to keras.layers.Layer.

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
        function_kwargs = {arg: kwargs[arg]
            for arg in kwargs if arg in arg_names}
        layer_kwargs = {arg: kwargs[arg]
            for arg in kwargs if not arg in arg_names}
        self._function_bound_args = signature.bind_partial(**function_kwargs)

        # Super
        BaseLayer.__init__(self, **layer_kwargs)

    def call(self, inputs, *args, **kwargs):
        """Layer call method."""

        # Collect args
        defaults = self._function_bound_args.arguments
        kwargs.pop("training", None)
        
        # Merge args
        kwargs_all = dict(defaults)
        kwargs_all.update(kwargs)

        # Run
        return self._function(inputs, *args, **kwargs_all)

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

    def __init__(self, name, scope, **kwargs):
        """Initialize.

        :param name: name of the layer.
        :param scope: namespace where to define the layer (such as globals()).
        """

        self.name = name
        self.scope = scope

    def __call__(self, function):
        """Decorator: create the layer.
        
        :param function: the decorated TF function.
        :return: the original function. The layer is defined as a side-effect.
        """

        # Create and export
        NewLayer = make_layer(name=self.name, function=function)
        self.scope[self.name] = NewLayer

        return function
