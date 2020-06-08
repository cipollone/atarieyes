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
from tensorflow.keras.layers import Dense, Flatten, Permute
from rl.core import Processor
from rl.policy import Policy
from rl.callbacks import Callback

from atarieyes.tools import ABCMeta2, AbstractAttribute
from atarieyes.layers import CropToEnv, ConvBlock


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
            self._cropper = CropToEnv(env_name)

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
            done = self.process_done(done)
            info = self.process_info(info)

            # Early termination
            if self._one_life:
                if not self._lives:
                    self._lives = info["ale.lives"]
                self._life_lost = (info["ale.lives"] < self._lives)
                self._lives = info["ale.lives"]

            return observation, reward, done, info

        def process_observation(self, observation):
            """Process an observation returned from the environment.

            Operations: resizing to a smaller frame; grayscale frame;
            composition of subsequent frames done by Memory with window lenght;
            normalization done by process_state_batch().
            """

            assert observation.ndim == 3

            observation = self._cropper.crop_one(observation)
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

        def process_done(self, done):
            """Process "done" boolean flag; do nothing."""

            return done

        def process_state_batch(self, batch):
            """Process a batch of states.

            Each batch is a composed of sequences of frames.
            """

            processed_batch = batch.astype(np.float32) / 255.
            return processed_batch

        def process_memory(self, observation, action, reward, terminal):
            """Process data before storing them in memory.

            NOTE: these arguments are already processed by the functions above.

            :param observation: last env observation (altready processed
                by process_step).
            :param action: action choosen after observation
            :param reward: received reward
            :param terminal: terminal state flag
            """

            # Remember a state as terminal when a life is lost.
            #   Unless 0 lives, because the env may send it immediately after.
            if self._one_life and self._life_lost and self._lives > 0:
                terminal = True

            return observation, action, reward, terminal


class RestrainedAtariAgent(AtariAgent):
    """Atari agent + Restraining Bolt."""

    def __init__(self, env_name, training, frames_sender, rb_receiver):
        """Initialize.

        :param env_name: name of an Atari gym environment.
        :param training: boolean training flag.
        :param frames_sender: an instance of AtariFramesSender
        :param rb_receiver: an instance of StateRewardReceiver
        """

        # Init
        env = gym.make(env_name)
        self.n_actions = env.action_space.n        # discrete in Atari

        # Number of states of the RestrainingBolt
        print("> Waiting init message from a connected RB")
        n_states, nan = rb_receiver.receive()
        n_states = int(n_states)
        if not np.isnan(nan):
            raise ValueError(
                "Expected NaN for this first message. Bad synchronization "
                "in StateReward stream")

        # Build models
        self.model = self._build_model()
        self.processor = self.Processor(
            env_name=env_name,
            resize_shape=self.resize_shape,
            one_life=True if training else False,  # for clarity
            frames_sender=frames_sender,
            rb_receiver=rb_receiver,
        )
        # TODO: build custom model

    class Processor(AtariAgent.Processor):
        """This processor inserts the Restraining Bolt into the loop."""

        def __init__(self, frames_sender, rb_receiver, **kwargs):
            """Initialize.

            :param frames_sender: an instance of AtariFramesSender
            :param rb_receiver: an instance of StateRewardReceiver
            :param kwargs: arguments of the basic processor.
            """

            # Super
            AtariAgent.Processor.__init__(self, **kwargs)

            # Store
            self._frames_sender = frames_sender
            self._rb_receiver = rb_receiver
            self._rb_reward = None
            self._last_frame = None

        def process_observation(self, observation):
            """Process and observation.

            The state of the restraining bolt is also part of the observation.
            """

            # Send this observation to RB
            self._frames_sender.send(observation, "continue")
            self._last_frame = observation

            # Standard processor
            processed = AtariAgent.Processor.process_observation(
                self, observation)

            # Receive from RB
            state, self._rb_reward = self._rb_receiver.receive()

            return processed, state

        def process_reward(self, reward):
            """Add the RB reward."""

            # Standard processor
            processed = AtariAgent.Processor.process_reward(self, reward)

            # RB reward
            processed += self._rb_reward

            return processed

        def process_done(self, done):
            """Notify the end of an episode."""

            # Notify, if end of an episode
            if done:
                self._frames_sender.send(self._last_frame, "repeated_last")

            return done
        
        def process_state_batch(self, batch):
            """Process a batch of states."""

            batch_size, window_length = batch.shape[0:2]

            # Combine frames together
            frames = batch[:, :, 0]
            frames = np.reshape(frames, [-1])
            frames = np.stack(frames, axis=0)
            frames = np.reshape(
                frames, [batch_size, window_length, *frames.shape[1:]])

            # Combine states together (take the first of each window)
            states = batch[:, 0, 1]
            states = np.stack(states)

            # TODO: also return states when the model is ready
            return frames


class EpisodeRandomEpsPolicy(Policy):
    """This is an EpsGreedyQPolicy with a different eps for each episode.

    Add self.callback to your callbacks to properly update the epsilon.
    """

    def __init__(self, min_eps=0.0, max_eps=1.0):
        """Initialize.

        :param min_eps: minimum epsilon value (prob of a random action).
        :param max_eps: maximum epsilon value.
        """

        # Super
        Policy.__init__(self)

        # Store
        self._min_eps = min_eps
        self._max_eps = max_eps
        self.eps = None

        self.callback = self._Callback(self)

    def get_config(self):
        """Return object configuration (Policy interface)."""

        config = Policy.get_config(self)
        config["max_eps"] = self._max_eps
        config["min_eps"] = self._min_eps

        return config

    def select_action(self, q_values):
        """Return an action.

        :param q_values: list of estimated Q for each action.
        :return: an action (int)
        """

        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.random_integers(0, nb_actions-1)
        else:
            action = np.argmax(q_values)
        return action

    class _Callback(Callback):
        """This callback updates the eps for each episode."""

        def __init__(self, policy):
            """Initialize."""

            # Super
            Callback.__init__(self)

            # Store
            self._min_eps = policy._min_eps
            self._max_eps = policy._max_eps
            self._policy = policy

        def on_episode_begin(self, episode, logs={}):
            """Called at beginning of each episode."""

            self._policy.eps = np.random.uniform(self._min_eps, self._max_eps)
