"""This module allows to train a feature extractor."""

import os
import shutil
import gym
import numpy as np
import tensorflow as tf

from atarieyes import models


class Trainer:
    """Train a feature extractor."""

    def __init__(self, args):
        """Initialize.

        :param args: namespace of arguments; see --help.
        """

        # Setup
        self.model_dir, self.log_dir = "models", "logs"
        self._prepare_directories((self.model_dir, self.log_dir))
        self.log_frequency = args.logs

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

        # Tools
        self.saver = self.CheckpointSaver(self.model, self.model_dir)
        self.logger = self.TensorBoardLogger(self.model, self.log_dir)

    def train(self):
        """Train."""

        # Init
        self.logger.save_graph(self.frame_shape)
        step = 0

        # Training loop
        print("> Training")
        while True:

            # Do
            metrics = self.step()

            # Logs and savings
            if step % self.log_frequency == 0:

                print("Step ", step, ", ", metrics, sep="", end="          \r")
                self.saver.save(-metrics["loss"])
                self.logger.save_scalars(metrics, step)

            step += 1

    def step(self):
        """Applies a single training step.
        
        :return: a dict of losses
        """

        frames = next(self.dataset_it)
        losses = self.model.train_on_batch(frames, frames)
        losses = {name: loss for name, loss in
            zip(self.model.metrics_names, losses)}
        return losses

    @staticmethod
    def _prepare_directories(dirs, resuming=False):
        """Prepare the directories where weights and logs are saved.

        :param dirs: sequence of directories to create.
        :param resuming: If True, the directories are not deleted.
        """

        # Delete old ones
        if not resuming:
            if any(os.path.exists(d) for d in dirs):
                print(
                    "Old logs and models will be deleted. Continue (Y/n)? ",
                    end="")
                c = input()
                if c not in ("y", "Y", ""):
                    quit()

            # New
            for d in dirs:
                if os.path.exists(d):
                    shutil.rmtree(d)
                os.makedirs(d)

    class CheckpointSaver:
        """Save weights and restore."""

        def __init__(self, model, path):
            """Initialize.

            :param model: the Keras model that will be saves.
            :param path: directory where checkpoints should be saved.
            """

            self.path = os.path.join(path, model.name)
            self.model = model
            self.score = float("-inf")

        def save(self, score=None):
            """Save.

            :param score: if provided, only save if score is higher than last
                saved score. If None, always save.
            :return: True when the model is saved.
            """

            # Return if not best
            if score is not None:
                if score < self.score:
                    return False
                else:
                    self.score = score

            # Save
            self.model.save_weights(self.path, overwrite=True, save_format="tf")
            return True

    class TensorBoardLogger:
        """Visualize data on TensorBoard."""

        def __init__(self, model, path):
            """Initialize.

            :param model: model that should be saved.
            :param path: directory where logs should be saved.
            """

            self.path = path
            self.model = model
            self.summary_writer = tf.summary.create_file_writer(path)

        def save_graph(self, input_shape):
            """Save the graph of the model.
            
            :param input_shape: the shape of the input tensor of the model
                (without batch).
            """

            # Forward pass TODO: test without tf.function in future release
            @tf.function
            def tracing_model_ops(inputs):
                return self.model(inputs)

            inputs = np.zeros((1, *input_shape), dtype=np.uint8)

            # Now trace
            tf.summary.trace_on(graph=True)
            tracing_model_ops(inputs)
            with self.summary_writer.as_default():
                tf.summary.trace_export(self.model.name, step=0)

        def save_scalars(self, metrics, step):
            """Save scalars.

            :param metrics: a dict of (name: value)
            :param step: the step number
            """

            with self.summary_writer.as_default():
                for name, value in metrics.items():
                    tf.summary.scalar(name, value, step=step)


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
