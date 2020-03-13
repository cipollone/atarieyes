"""This module allows to train a feature extractor."""

import gym
import tensorflow as tf

from atarieyes import models


class Trainer:
    """Train a feature extractor."""

    def __init__(self, args):
        """Initialize.

        :param args: namespace of arguments; see --help.
        """

        # Environment
        self.env = gym.make(args.env)
        self.frame_shape = self.env.observation_space.shape

        # Dataset
        dataset = make_dataset(
            lambda: random_play(args.env, args.render),
            args.batch, self.frame_shape)
        self.dataset_it = iter(dataset)

        # Model
        self.model = models.single_frame_model(self.frame_shape)

    def train(self):
        """Train."""

        self.step()

    def step(self):
        """Applies a single training step."""

        data = next(self.dataset_it)
        print(data.shape)


def make_dataset(game_player, batch, frame_shape):
    """Create a TF Dataset from frames of a game.

    Creates a TF Dataset which contains batches of frames.

    :param game_player: A callable which creates an interator. The interator
        must return Gym env.step outputs.
    :param batch: Batch size.
    :param frame_shape: Frame input shape.
    :return: Tensorflow Dataset.
    """

    # Extract observations
    def frame_iterate():
        env_step = game_player()
        while True:
            yield next(env_step)[0]

    # Dataset
    dataset = tf.data.Dataset.from_generator(
        frame_iterate, output_types=tf.uint8, output_shapes=frame_shape)

    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(1)

    return dataset


def random_play(env_name, render=False):
    """Play randomly a game.

    :param env_name: Gym Environment name
    :param render: When true, the environment is rendered.
    :return: a interator of Gym env.step return arguments.
    """

    # Make
    env = gym.make(env_name)

    n_game = 0

    # For each game
    while True:

        # Reset
        env.reset()
        done = False
        if render:
            env.render()

        # Until the end
        while not done:

            # Random agent moves
            action = env.action_space.sample()

            # Environment moves
            observation, reward, done, info = env.step(action)

            # Result
            if render:
                env.render()
            yield observation, reward, done, None

        n_game += 1

    # Leave
    env.close()
