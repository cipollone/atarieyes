"""Agent definition: network and other parts.

The agent defined here is called "Atari agent" because it's intended to be
used for all environments. No ad-hoc changes.

Parts of this model have been taken from the original Double Dqn paper and
the example under `keras-rl/examples/dqn_atari.py`.
Keras-rl mostly relies on numpy instead of tensorflow; I won't change this.
"""

import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, Permute
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

    def __init__(self, n_actions, training):
        """Initialize.

        :param n_actions: number of actions for this environment.
        :param training: boolean training flag.
        """

        self.n_actions = n_actions

        self.model = self._build_model()
        self.processor = self.Processor(
            frame_shape=self.frame_shape,
            one_life=True if training else False)  # for clarity

    def _build_model(self):
        """Define the Q-network of the agent.

        :return: a keras model
        """

        # The input of the model is a batch of groups of frames
        input_shape = (self.window_length,) + self.frame_shape

        # Define
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
        ], name="Agent_net")
        model.summary()

        return model

    class Processor(Processor):
        """Pre/post processing for Atari environment.

        A processor can modify observations, actions, rewards, etc.
        """

        def __init__(self, frame_shape, one_life=True):
            """Initialize.

            :param frame_shape: 2d size of the resized frames.
            :param one_life: if True, the game is stopped when a single life
                is lost.
            """

            Processor.__init__(self)

            self.frame_shape = frame_shape
            self.one_life = one_life

            self.lives = None

        def process_step(self, observation, reward, done, info):
            """Processes an entire step.

            NOTE: this overrides process_step in rl.Processor. I need to
                call other methods manually.

            :param observation: An observation as obtained by the environment.
            :param reward: A reward as obtained by the environment.
            :param done: True if the environment is in a terminal state,
                False otherwise.
            :param info: The debug info dictionary as obtained by the
                environment.
            :return: processed (observation, reward, done, reward)
            """

            # Standard processing
            observation = self.process_observation(observation)
            reward = self.process_reward(reward)

            # Early termination
            if self.one_life:
                if not self.lives:
                    self.lives = info["ale.lives"]
                if info["ale.lives"] < self.lives:
                    self.lives = None
                    done = True

            return observation, reward, done, info

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

        def process_reward(self, reward):
            """Process reward returned from the environment.

            Reward clipping.
            """

            return np.clip(reward, -1., 1.)

        def process_state_batch(self, batch):
            """Process a batch of states.

            Each batch is a composed of sequences of frames.
            """

            processed_batch = batch.astype("float32") / 255.
            return processed_batch
