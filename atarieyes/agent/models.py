"""Agent definition: network and other parts.

The agent defined here is called "Atari agent" because it's intended to be
used for all environments. No ad-hoc changes.

Parts of this model have been taken from the original Double Dqn paper and
the example under `keras-rl/examples/dqn_atari.py`.
Keras-rl mostly relies on numpy instead of tensorflow; I won't change this.
"""

import numpy as np
from PIL import Image
import gym
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from rl.core import Processor

from atarieyes.tools import ABCMeta2, AbstractAttribute
from atarieyes.layers import CropToEnvBox, ConvBlock


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
    resize_shape = (84, 84)     # frames resized to this size
    window_length = 4           # an observation contains 4 consecutive frames

    # Additional layers needed for restoring
    custom_layers = dict(
        ConvBlock=ConvBlock,
        VarianceScaling=keras.initializers.VarianceScaling,  # probable tf bug
    )

    def __init__(self, env_name, training):
        """Initialize.

        :param env_name: name of an Atari gym environment.
        :param training: boolean training flag.
        """
        
        # Init
        env = gym.make(env_name)
        self.n_actions = env.action_space.n        # discrete in Atari

        # Build models
        self.model = self._build_model()
        self.processor = self.Processor(
            env_name=env_name,
            resize_shape=self.resize_shape,
            one_life=True if training else False)  # for clarity

    def _build_model(self):
        """Define the Q-network of the agent.

        :return: a keras model
        """

        # The input of the model is a batch of groups of frames
        input_shape = (self.window_length,) + self.resize_shape

        # Define
        model = Sequential([
            Permute((2, 3, 1), input_shape=input_shape),  # window -> channels
            ConvBlock(
                filters=32, kernel_size=8, strides=4, padding="valid",
                activation="relu"),
            ConvBlock(
                filters=64, kernel_size=4, strides=2, padding="valid",
                activation="relu"),
            ConvBlock(
                filters=64, kernel_size=3, strides=1, padding="valid",
                activation="relu"),
            Flatten(),
            Dense(512, activation="relu"),
            Dense(self.n_actions),
        ], name="Agent_net")
        model.summary()

        return model

    class Processor(Processor):
        """Pre/post processing for Atari environment.

        A processor can modify observations, actions, rewards, etc.
        """

        def __init__(self, env_name, resize_shape, one_life=True):
            """Initialize.

            :param env_name: name of an Atari gym environment.
            :param resize_shape: 2d size of the resized frames.
            :param one_life: if True, every time a life is lost, the state
                is marked as terminal.
            """

            Processor.__init__(self)

            self._one_life = one_life
            self._lives = None
            self._life_lost = False

            self._resize_shape = resize_shape
            self._cropper = CropToEnvBox(env_name)

        #def process_step(self, observation, reward, done, info):
        #    """Processes an entire step.

        #    NOTE: this overrides process_step in rl.Processor. I need to
        #        call other methods manually.

        #    :param observation: An observation as obtained by the environment.
        #    :param reward: A reward as obtained by the environment.
        #    :param done: True if the environment is in a terminal state,
        #        False otherwise.
        #    :param info: The debug info dictionary as obtained by the
        #        environment.
        #    :return: processed (observation, reward, done, reward)
        #    """

        #    # Standard processing
        #    observation = self.process_observation(observation)
        #    reward = self.process_reward(reward)
        #    info = self.process_info(info)

        #    # Early termination
        #    if self._one_life:
        #        if not self._lives:
        #            self._lives = info["ale.lives"]
        #        self._life_lost = (info["ale.lives"] < self._lives)
        #        self._lives = info["ale.lives"]

        #    return observation, reward, done, info

        def process_observation(self, observation):
            """Process an observation returned from the environment.

            Operations: resizing to a smaller frame; grayscale frame;
            composition of subsequent frames done by Memory with window lenght;
            normalization done by process_state_batch().
            """

            assert observation.ndim == 3

            #observation = self._cropper.crop_one(observation)
            img = Image.fromarray(observation)
            img = img.resize(self._resize_shape).convert("L")
            processed_observation = np.array(img, dtype=np.uint8)

            assert processed_observation.shape == self._resize_shape
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

        #def process_memory(self, observation, action, reward, terminal):
        #    """Process data before storing them in memory.

        #    NOTE: these arguments are already processed by the functions above.

        #    :param observation: last env observation (altready processed
        #        by process_step).
        #    :param action: action choosen after observation
        #    :param reward: received reward
        #    :param terminal: terminal state flag
        #    """

        #    # Remember a state as terminal when a life is lost.
        #    #   Unless 0 lives, because the env may send it immediately after.
        #    if self._one_life and self._life_lost and self._lives > 0:
        #        terminal = True

        #    return observation, action, reward, terminal
