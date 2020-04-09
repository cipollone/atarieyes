"""Agent definition: network and other parts.

The agent defined here is called "Atari agent" because it's intended to be
used for all environments. No ad-hoc changes.

Parts of this model have been taken from the original Double Dqn paper and
the example under `keras-rl/examples/dqn_atari.py`.
Keras-rl mostly relies on numpy instead of tensorflow, so I won't change this.
"""
# NOTE: I can't use atarieyes.layers here, because those use the tf.keras api.

import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
import keras.backend as K
from rl.core import Processor

from atarieyes.tools import ABCMeta2, AbstractAttribute


class QAgentDef(metaclass=ABCMeta2):
    """Core definitions for an agent based on deep variants of Q-learning.

    The structure of a keras-rl agent is mostly defined by it's model (that is
    the neural net of its Q-function), and a processor.
    """

    # Keras model
    model = AbstractAttribute()

    # keras-rl Processor
    processor = AbstractAttribute()


class AtariAgent(QAgentDef):
    """Agent definitions for Atari environments."""

    # Common hyperparameters
    frame_shape = (84, 84)     # frames resized to this size
    window_length = 4          # an observation contains 4 consecutive frames

    def __init__(self, n_actions):
        """Initialize.

        :param n_actions: number of actions for this environment.
        """

        self.n_actions = n_actions

        self.model = self._build_model()
        self.processor = self.Processor(frame_shape=self.frame_shape)

    def _build_model(self):
        """Define the Q-network of the agent.

        :return: a keras model
        """

        # The input of the model is a batch of groups of frames
        input_shape = (self.window_length,) + self.frame_shape

        # Define
        assert K.image_data_format() == "channels_last"
        model = Sequential([
            Permute((2, 3, 1), input_shape=input_shape),  # window -> channels
            Conv2D(32, (8, 8), strides=(4, 4)),
            Activation("relu"),
            Conv2D(64, (4, 4), strides=(2, 2)),
            Activation("relu"),
            Conv2D(64, (3, 3), strides=(1, 1)),
            Activation("relu"),
            Flatten(),
            Dense(512),
            Activation("relu"),
            Dense(self.n_actions),
        ])
        model.summary()

        return model

    class Processor(Processor):
        """Pre/post processing for Atari environment.

        A processor can modify observations, actions, rewards, etc.
        """

        def __init__(self, frame_shape):
            """Initialize.

            :param frame_shape: 2d size of the resized frames.
            """

            super().__init__()

            self.frame_shape = frame_shape

        def process_observation(self, observation):
            """Process an observation returned from the environment.

            Operations: resizing to a smaller frame; grayscale frame;
            composition of subsequent frames done by Memory with window lenght;
            normalization done by process_state_batch().
            """

            assert observation.ndim == 3

            img = Image.fromarray(observation)
            img = img.resize(self.frame_shape).convert("L")
            processed_observation = np.array(img, dtype=np.uint8)

            assert processed_observation.shape == self.frame_shape
            return processed_observation

        def process_state_batch(self, batch):
            """Process a batch of states.

            Each batch is a composed of sequences of frames.
            """

            processed_batch = batch.astype("float32") / 255.
            return processed_batch

        def process_reward(self, reward):
            """Process reward returned from the environment.

            Reward clipping.
            """

            return np.clip(reward, -1., 1.)
