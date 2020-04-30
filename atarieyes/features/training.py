"""This module allows to train a feature extractor."""

import os
import json
import gym
import numpy as np
import tensorflow as tf

from atarieyes.features import models
from atarieyes import tools
from atarieyes.streaming import AtariFramesReceiver


class Trainer:
    """Train a feature extractor."""

    def __init__(self, args):
        """Initialize.

        :param args: namespace of arguments; see --help.
        """

        # Init
        self.log_frequency = args.log_frequency
        self.save_frequency = args.save_frequency
        self.cont = bool(args.cont)
        self.init_step = args.cont if self.cont else 0
        self.learning_rate = args.learning_rate
        self.const_rate = args.const_rate

        # Dirs
        model_path, log_path = tools.prepare_directories(
            "features", args.env, resuming=self.cont, args=args)

        # Environment
        self.env = gym.make(args.env)
        self.frame_shape = self.env.observation_space.shape

        # Dataset
        dataset = make_dataset(
            lambda: agent_player(args.env, args.stream),
            args.batch_size, self.frame_shape)
        self.dataset_it = iter(dataset)

        # Model
        self.model = models.LocalFluent(args.env)

        # Optimization
        if not self.const_rate:
            self.learning_rate = tf.optimizers.schedules.ExponentialDecay(
                args.learning_rate, decay_steps=args.decay_steps,
                decay_rate=0.95)
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.params = self.model.keras.trainable_variables

        # Tools
        self.saver = CheckpointSaver(self.model.keras, model_path)
        self.logger = TensorBoardLogger(self.model, log_path)

    def train(self):
        """Train."""

        step = self.init_step

        # New run
        if not self.cont:
            self.logger.save_graph(self.frame_shape)
        # Restore
        else:
            self.saver.load(step)

            # Initial valuation
            self.valuate(step)
            step += 1

        # Training loop
        print("> Training")
        while True:

            # Do
            outputs = self.train_step()

            # Logs and savings
            relative_step = step - self.init_step
            if relative_step % self.log_frequency == 0:
                metrics = self.valuate(step, outputs)
                print("Step ", step, ", ", metrics, sep="", end="          \r")
            if relative_step % self.save_frequency == 0 and relative_step > 0:
                self.saver.save(step)

            step += 1

    @tf.function
    def train_step(self):
        """Applies a single training step.

        :return: outputs of the model.
        """

        # Custom gradient
        if self.model.computed_gradient:
            outputs = self.model.compute_all(next(self.dataset_it))
            gradients = outputs["gradients"]

        # Compute
        else:
            with tf.GradientTape() as tape:
                outputs = self.model.compute_all(next(self.dataset_it))
            gradients = tape.gradient(outputs["loss"], self.params)

        # Apply
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
            outputs = self._model_compute_all(frames)

        # Log scalars
        metrics = {
            "metrics/" + name: value
            for name, value in outputs["metrics"].items()}
        if outputs["loss"] is not None:
            metrics["loss"] = outputs["loss"]
        metrics["learning_rate"] = self.learning_rate if self.const_rate else \
            self.learning_rate(step)
        self.logger.save_scalars(step, metrics)

        # Log images
        images = self.model.images(outputs["outputs"])
        self.logger.save_images(step, images)

        # Log histograms
        histograms = self.model.histograms(outputs["outputs"])
        self.logger.save_histogram(step, histograms)

        return metrics

    @tf.function
    def _model_compute_all(self, inputs):
        """Efficient graph call for Model.compute_all."""

        return self.model.compute_all(inputs)

class CheckpointSaver:
    """Save weights and restore."""

    save_format = "h5"

    def __init__(self, model, path):
        """Initialize.

        :param model: the Keras model that will be saved.
        :param path: directory where checkpoints should be saved.
        """

        # Store
        self.model = model
        self.score = float("-inf")

        self.counters_file = os.path.join(path, "counters.json")
        self.step_checkpoints = os.path.join(
            path, model.name + "_weights_{step}." + self.save_format)

    def _update_counters(self, filepath, step):
        """Updates the file of counters with a new entry.

        :param filepath: checkpoint that is being saved
        :param step: current global step
        """

        counters = {}

        # Load
        if os.path.exists(self.counters_file):
            with open(self.counters_file) as f:
                counters = json.load(f)

        counters[filepath] = dict(step=step)

        # Save
        with open(self.counters_file, "w") as f:
            json.dump(counters, f, indent=4)

    def save(self, step, score=None):
        """Save.

        :param step: current training step.
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
        filepath = self.step_checkpoints.format(step=step)
        self.model.save_weights(
            filepath, overwrite=True, save_format=self.save_format)
        self._update_counters(filepath=filepath, step=step)

        return True

    def load(self, step):
        """Load the weights from a checkpoint.

        :param step: specify which checkpoint to load
        """

        filepath = self.step_checkpoints.format(step=step)

        # Restore
        self.model.load_weights(filepath)
        print("> Loaded:", filepath)


class TensorBoardLogger:
    """Visualize data on TensorBoard."""

    def __init__(self, model, path):
        """Initialize.

        :param model: a features.Model that should be saved.
        :param path: directory where logs should be saved.
        """

        self.path = path
        self.model = model
        self.summary_writer = tf.summary.create_file_writer(path)

    def save_graph(self, input_shape):
        """Visualize the graph of the model in TensorBoard.

        :param input_shape: the shape of the input tensor of the model
            (without batch).
        """

        # Forward pass
        @tf.function
        def tracing_model_ops(inputs):
            return self.model.compute_all(inputs)

        inputs = np.zeros((1, *input_shape), dtype=np.uint8)

        # Now trace
        tf.summary.trace_on(graph=True)
        tracing_model_ops(inputs)
        with self.summary_writer.as_default():
            tf.summary.trace_export(self.model.__class__.__name__, step=0)

    def save_scalars(self, step, metrics):
        """Visualize scalar metrics in TensorBoard.

        :param step: the step number
        :param metrics: a dict of (name: value)
        """

        # Save
        with self.summary_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=step)

    def save_images(self, step, images):
        """Visualize images in TensorBoard.

        :param step: the step number
        :param images: a dict of batches of images. Only the first
            of each group is displayed.
        """

        # Save
        with self.summary_writer.as_default():
            for name, batch in images.items():
                image = batch[0]
                image = tf.expand_dims(image, axis=0)
                tf.summary.image(name, image, step)

    def save_histogram(self, step, tensors):
        """Visualize tensors as histograms.

        :param step: the step number
        :param tensors: a dict of (name: tensor)
        """

        # Save
        with self.summary_writer.as_default():
            for name, tensor in tensors.items():
                tf.summary.histogram(name, tensor, step)


def make_dataset(game_player, batch, frame_shape):
    """Create a TF Dataset from frames of a game.

    Creates a TF Dataset which contains batches of frames.

    :param game_player: A callable which creates an interator. The interator
        must return frames of the game.
    :param batch: Batch size.
    :param frame_shape: Frame shape retuned by game_player.
    :return: Tensorflow Dataset.
    """

    # Extract observations
    def frame_iterate():
        env_step = game_player()
        while True:
            yield next(env_step)

    # Dataset
    dataset = tf.data.Dataset.from_generator(
        frame_iterate, output_types=tf.uint8, output_shapes=frame_shape)

    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(1)

    return dataset


def random_player(env_name, render=False):
    """Play randomly a game.

    :param env_name: Gym Environment name
    :param render: When true, the environment is rendered.
    :return: an interator of frames.
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
            yield observation

        n_game += 1

    # Leave
    env.close()


def agent_player(env_name, ip="localhost"):
    """Returns frame from a trained agent.

    This requires a running instance of `atarieyes agent play`.

    :param env_name: name of an Atari Gym environment
    :param ip: machine where the agent is playing
    :return: a generator of frames
    """

    print("> Waiting for a stream of frames from:", ip)

    # Set up a connection
    receiver = AtariFramesReceiver(env_name, ip)

    # Collect
    try:
        while True:
            yield receiver.receive(wait=True)

    except ConnectionAbortedError:
        raise StopIteration
