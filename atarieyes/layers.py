"""Various custom layers. Any conceptual block can be a layer."""

from abc import abstractmethod
from tensorflow.keras import layers

from atarieyes.tools import ABC2


class BaseLayer(layers.Layer, ABC2):
    """Base class for all layers and model parts.

    This is mainly used to create namespaces in TensorBoard graphs.
    This must be subclassed, not instantiated directly.
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

    @abstractmethod
    def build(self, input_shape):
        """Define the model and optionally fill the special members."""
