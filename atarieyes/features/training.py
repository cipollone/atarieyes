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
        self.initialize_from = args.initialize or args.cont
        self.resuming = self.initialize_from is not None
        self.new_run = not self.resuming or args.initialize is not None
        self.learning_rate = args.learning_rate
        self.decay_rate = args.decay_rate
        self.batch_size = args.batch_size

        # Dirs
        model_path, log_path = tools.prepare_directories(
            "features", args.env, resuming=self.resuming, args=args)

        # Environment
        self.env = gym.make(args.env)
        self.frame_shape = self.env.observation_space.shape

        # Model
        self.model = self.build_model(
            tools.Namespace(args), log_path=log_path)

        # Define the dataset and initialize it (if not custom training loop)
        dataset = make_dataset(
            lambda: agent_player(args.env, args.stream),
            args.batch_size, self.frame_shape, args.shuffle)
        if not self.model.train_step:
            self.dataset_it = iter(dataset)

        # Optimization
        if self.decay_rate:
            self.learning_rate = tf.optimizers.schedules.ExponentialDecay(
                args.learning_rate, decay_steps=args.decay_steps,
                decay_rate=0.95)
        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.params = self.model.model.trainable_variables

        # Tools
        self.saver = CheckpointSaver(self.model.model, model_path)
        self.logger = TensorBoardLogger(self.model, log_path)

        # Save on exit
        tools.QuitWithResources.add(
            "last_save", lambda: self.saver.save(self._step))

    @staticmethod
    def build_model(spec, log_path):
        """Instantiates a model from a specification.

        :param spec: a Namespace specification arguments (see --help).
        :param log_path: directory of logs.
        :return: an instance of features.models.Model
        """

        # Part of the model to train
        if spec.train_region_layer is not None:
            train_layer = int(spec.train_region_layer[1])
            train_region = spec.train_region_layer[0]
        else:
            train_layer = train_region = None

        # Model hyper-parameters
        encoding_spec = [dict(
            n_hidden=units, batch_size=spec.batch_size,
            l2_const=spec.l2_const, sparsity_const=spec.sparsity_const,
            sparsity_target=spec.sparsity_target,
            ) for units in spec.network_size
        ]
        genetic_spec = dict(
            n_individuals=spec.population_size,
            mutation_p=spec.mutation_p,
            crossover_p=spec.crossover_p,
            fitness_range=spec.fitness_range,
            n_episodes=spec.fitness_episodes,
        )

        # Model
        model = models.Fluents(
            env_name=spec.env,
            dbn_spec=encoding_spec,
            ga_spec=genetic_spec,
            training_layer=train_layer,
            training_region=train_region,
            receiver_gen=lambda: atari_frames_generator(spec.env, spec.stream),
            logdir=log_path,
        )

        return model

    def train(self):
        """Train."""

        self._step = step0 = 0

        # Load weights
        if self.resuming:
            ckp_counters = self.saver.load(self.initialize_from)

        # Save graph once
        if self.new_run:
            self.logger.save_graph((self.batch_size, *self.frame_shape))

        # Continue previous training
        else:
            self._step = step0 = ckp_counters["step"]
            if not self.model.train_step:
                self.valuate()
            self._step += 1

        # Training loop
        print("> Training")
        while True:

            # Do
            if self.model.train_step:
                outputs = self.model.train_step()
            else:
                outputs = self.train_step()

            # Logs and savings
            relative_step = self._step - step0
            if relative_step % self.log_frequency == 0:
                metrics = self.valuate(outputs)
                print("Step ", self._step, ", ", metrics, sep="", end="    \r")
            if relative_step % self.save_frequency == 0 and relative_step > 0:
                self.saver.save(self._step)

            self._step += 1

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

    def valuate(self, outputs=None):
        """Compute the metrics on one batch and save a log.

        When 'outputs' is not given, it runs the model to compute the metrics.
        When metrics are not scalars, it prints their mean and max value.

        :param outputs: (optional) outputs returned by Model.compute_all.
        :return: the saved quantities (metrics and loss)
        """

        # Compute if not given
        if not outputs:
            frames = next(self.dataset_it)
            outputs = self._model_compute_all(frames)

        # Collect all metrics
        metrics = {
            "metrics/" + name: value
            for name, value in outputs["metrics"].items()}
        if outputs["loss"] is not None:
            metrics["loss"] = outputs["loss"]
        metrics["learning_rate"] = self.learning_rate if not self.decay_rate \
            else self.learning_rate(self._step)

        # Log scalars (convert to mean max if necessary)
        scalars = {}
        for name, value in metrics.items():
            is_scalar = (
                isinstance(value, int) or isinstance(value, float) or
                value.ndim == 0)
            if is_scalar:
                scalars[name] = value
            else:
                scalars[name + "_mean"] = tf.math.reduce_mean(value)
                scalars[name + "_max"] = tf.math.reduce_max(value)
        self.logger.save_scalars(self._step, scalars)

        # Log images
        images = self.model.images(outputs["outputs"])
        self.logger.save_images(self._step, images)

        # Log histograms
        histograms = self.model.histograms(outputs["outputs"])
        self.logger.save_histogram(self._step, histograms)

        # Transform tensors to arrays for nice logs
        scalars = {
            name: var.numpy() if isinstance(var, tf.Tensor) else var
            for name, var in scalars.items()
        }

        return scalars

    @tf.function
    def _model_compute_all(self, inputs):
        """Efficient graph call for Model.compute_all."""

        return self.model.compute_all(inputs)


class CheckpointSaver:
    """Save weights and restore."""

    def __init__(self, model, path):
        """Initialize.

        :param model: any tf.Module, tf.keras.Model or tf.keras.layers.Layer
            to be saved.
        :param path: directory where checkpoints should be saved.
        """

        # Store
        self.score = float("-inf")
        self.checkpoint = tf.train.Checkpoint(model=model)

        self.counters_file = os.path.join(
            path, os.path.pardir, "counters.json")
        self.step_checkpoints = os.path.join(
            path, model.name + "_weights_{step}.tf")

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
        self.checkpoint.write(filepath)
        self._update_counters(filepath=filepath, step=step)

        return True

    def load(self, path):
        """Load the weights from a checkpoint.

        :param path: load checkpoint at this path
        :return: the counters (such as "step") associated to this checkpoint
        """

        # Restore
        self.checkpoint.restore(path).expect_partial()
        print("> Loaded:", path)

        # Read counters
        with open(self.counters_file) as f:
            data = json.load(f)
        return data[path]


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
            (with batch).
        """

        # Forward pass
        @tf.function
        def tracing_model_ops(inputs):
            return self.model.compute_all(inputs)

        inputs = np.zeros(input_shape, dtype=np.uint8)

        # Now trace
        tf.summary.trace_on(graph=True)
        tracing_model_ops(inputs)
        with self.summary_writer.as_default():
            tf.summary.trace_export(self.model.__class__.__name__, step=0)

    def save_scalars(self, step, metrics):
        """Visualize scalar metrics in TensorBoard.

        :param step: the step number
        :param metrics: a dict of (name: value), where value is a scalar
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


def make_dataset(game_player, batch, frame_shape, shuffle_size):
    """Create a TF Dataset from frames of a game.

    Creates a TF Dataset which contains batches of frames.

    :param game_player: A callable which creates an interator. The interator
        must return frames of the game.
    :param batch: Batch size.
    :param frame_shape: Frame shape retuned by game_player.
    :param shuffle_size: size of the shuffle buffer.
    :return: Tensorflow Dataset.
    """

    # Dataset
    dataset = tf.data.Dataset.from_generator(
        game_player, output_types=tf.uint8, output_shapes=frame_shape)

    dataset = dataset.shuffle(shuffle_size)
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


def agent_player(env_name, ip):
    """Returns frames from a remote player.

    This requires a running instance of `atarieyes agent play`.

    :param env_name: name of an Atari Gym environment
    :param ip: machine where the agent is playing
    :return: a generator of frames
    """

    # Create the main generator
    receiver_gen = atari_frames_generator(env_name, ip)

    # Loop
    while True:

        # Receive
        frame, termination = next(receiver_gen)

        # Skip if repeated
        assert termination in ("continue", "last", "repeated_last")
        if termination == "repeated_last":
            continue

        # Return
        yield frame


def atari_frames_generator(env_name, ip):
    """Returns data from an AtariFramesReceiver.

    This requires a running instance of `atarieyes agent play`.

    :param env_name: name of an Atari Gym environment
    :param ip: machine where the agent is playing
    :return: See AtariFramesReceiver for the return type
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
