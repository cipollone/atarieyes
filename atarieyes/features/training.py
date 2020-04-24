"""This module allows to train a feature extractor."""

import os
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
        self.log_frequency = args.logs
        self.cont = args.cont
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
        self.model = models.FrameAutoencoder(
            frame_shape=self.frame_shape,
            env_name=self.env.spec.id
        )

        # Optimization
        self.params = self.model.keras.trainable_variables \
            if not self.model.computed_gradient else None
        self.optimizer = tf.optimizers.Adam(args.learning_rate)

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
            step = self.saver.load()
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

        # Custom gradient
        if self.model.computed_gradient:
            gradients = self.model.compute_all(
                next(self.dataset_it))["gradients"]

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
            outputs = self.model.compute_all(frames)

        # Log scalars
        metrics = dict(outputs["metrics"])
        metrics["loss"] = outputs["loss"]
        self.logger.save_scalars(step, metrics)

        # Log images
        images = self.model.output_images(outputs["outputs"])
        self.logger.save_images(step, images)

        return metrics


class CheckpointSaver:
    """Save weights and restore."""

    def __init__(self, model, path):
        """Initialize.

        :param model: the Keras model that will be saved.
        :param path: directory where checkpoints should be saved.
        """

        self.save_path = os.path.join(path, model.name)
        self.counters_path = os.path.join(path, "counters.txt")
        self.model = model
        self.score = float("-inf")

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
        self.model.save_weights(
            self.save_path, overwrite=True, save_format="tf")
        with open(self.counters_path, "w") as f:
            f.write("step: " + str(step))
        return True

    def load(self):
        """Restore weights from the previous checkpoint.

        NOTE: optimizer state is not restored.

        :return: the step of the saved weights
        """

        # Restore
        self.model.load_weights(self.save_path)

        # Step
        with open(self.counters_path) as f:
            counters = f.read()
        step = int(counters.split(":")[1])

        return step


class TensorBoardLogger:
    """Visualize data on TensorBoard."""

    def __init__(self, model, path):
        """Initialize.

        :param model: tensorflow model that should be saved.
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

        # Forward pass
        @tf.function
        def tracing_model_ops(inputs):
            return self.model(inputs)

        inputs = np.zeros((1, *input_shape), dtype=np.uint8)

        # Now trace
        tf.summary.trace_on(graph=True)
        tracing_model_ops(inputs)
        with self.summary_writer.as_default():
            tf.summary.trace_export(self.model.name, step=0)

    def save_scalars(self, step, metrics):
        """Save scalars.

        :param step: the step number
        :param metrics: a dict of (name: value)
        """

        # Save
        with self.summary_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=step)

    def save_images(self, step, images):
        """Save images in TensorBoard.

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
