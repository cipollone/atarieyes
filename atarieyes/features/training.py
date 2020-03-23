"""This module allows to train a feature extractor."""

import os
import shutil
import gym
import numpy as np
import tensorflow as tf

from atarieyes import models
from atarieyes import tftools
from atarieyes.tftools import CheckpointSaver, TensorBoardLogger


class Trainer:
    """Train a feature extractor."""

    def __init__(self, args):
        """Initialize.

        :param args: namespace of arguments; see --help.
        """

        # Init
        self.log_frequency = args.logs
        self.cont = args.cont
        model_path, log_path = tftools.prepare_directories(
            "features", args.env, resuming=self.cont)

        # Environment
        self.env = gym.make(args.env)
        self.frame_shape = self.env.observation_space.shape

        # Dataset
        dataset = make_dataset(
            lambda: random_play(args.env, args.render),
            args.batch, self.frame_shape)
        self.dataset_it = iter(dataset)

        # Model
        self.model = models.FrameAutoencoder(
            frame_shape=self.frame_shape,
            env_name=self.env.spec.id
        )

        # Optimization
        self.params = self.model.keras.trainable_variables
        self.optimizer = tf.optimizers.Adam(args.rate)

        # Tools
        self.saver = CheckpointSaver(self.model.keras, model_path)
        self.logger = TensorBoardLogger(self.model.keras, log_path)

    def train(self):
        """Train."""

        # New run
        if not self.cont:
            self.logger.save_graph(self.frame_shape)
            step = 0
        # Restore
        else:
            step = self.saver.load(self.frame_shape)
            print("> Weights restored.")

            # Initial valuation
            self.valuate(step)
            step += 1

        # Training loop
        print("> Training")
        while True:

            # Do
            outputs = self.train_step()

            # Logs and savings
            if step % self.log_frequency == 0:

                metrics = self.valuate(step, outputs)
                self.saver.save(step, -metrics["loss"])
                print("Step ", step, ", ", metrics, sep="", end="          \r")

            step += 1

    @tf.function
    def train_step(self):
        """Applies a single training step.

        :return: outputs of the model.
        """

        # Forward
        with tf.GradientTape() as tape:
            outputs = self.model.compute_all(next(self.dataset_it))

        # Compute and apply grandient
        gradients = tape.gradient(outputs["loss"], self.params)
        self.optimizer.apply_gradients(zip(gradients, self.params))

        return outputs

    def valuate(self, step, outputs=None):
        """Compute the metrics on one batch and save a log.

        When 'outputs' is not given, it runs the model to compute the metrics.

        :param step: current step.
        :param outputs: (optional) outputs returned by Model.compute_all.
        :return: the saved quantities (metrics and loss)
        """

        # Compute if not given
        if not outputs:
            frames = next(self.dataset_it)
            outputs = self.model.compute_all(frames)

        # Log scalars
        metrics = dict(outputs["metrics"])
        metrics["loss"] = outputs["loss"]
        self.logger.save_scalars(step, metrics)

        # Log images
        images = self.model.output_images(outputs["outputs"])
        self.logger.save_images(step, images)

        return metrics


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
