"""This module allows to train a RL agent."""

import os
import numpy as np
import json
import gym
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.callbacks import Callback, FileLogger

from atarieyes import streaming, tools
from atarieyes.agent import models


class Trainer:
    """Train a RL agent on the Atari games."""

    def __init__(self, args):
        """Initialize.

        :param args: namespace of arguments; see --help.
        """

        # Params
        self.resuming = args.cont is not None
        self.initialize_from = args.cont
        self._log_interval = args.log_frequency
        self._max_episode_steps = args.max_episode_steps

        # Dirs
        model_path, log_path = tools.prepare_directories(
            "agent", args.env, resuming=self.resuming, args=args)
        log_filename = "log.json"
        self.log_file = os.path.join(log_path, log_filename)

        # Environment
        self.env = gym.make(args.env)
        self.env_name = args.env

        # Repeatability
        if args.deterministic:
            if "Deterministic" not in self.env_name:
                raise ValueError(
                    "--deterministic only works with deterministic"
                    " environments")
            self.env.seed(30013)
            np.random.seed(30013)
            tf.random.set_seed(30013)

        # Agent
        self.kerasrl_agent, self.atari_agent = self.build_agent(
            tools.Namespace(args, training=True))

        # Tools
        self.saver = CheckpointSaver(
            agent=self.kerasrl_agent, path=model_path, interval=args.saves)
        self.logger = TensorboardLogger(
            logdir=log_path, agent=self.atari_agent)

        # Callbacks
        self.callbacks = [
            self.saver,
            self.logger,
            FileLogger(filepath=self.log_file, interval=100),
        ]
        if args.random_epsilon:
            self.callbacks.append(self.kerasrl_agent.test_policy.callback)

        # Save on exit
        tools.QuitWithResources.add("last_save", lambda: self.saver.save())

    @staticmethod
    def build_agent(spec):
        """Defines a Keras-rl agent, ready for training.

        :param spec: a Namespace of agent specification options.
        :return: the rl agent
        """

        env = gym.make(spec.env)
        n_actions = env.action_space.n

        # Define network for Atari games
        if spec.rb_address is None:
            atari_agent = models.AtariAgent(
                env_name=spec.env,
                training=spec.training,
                one_life=not spec.no_onelife,
            )

        # Define network for Atari games + Restraining bolt
        else:
            atari_agent = models.RestrainedAtariAgent(
                env_name=spec.env,
                training=spec.training,
                one_life=not spec.no_onelife,
                frames_sender=streaming.AtariFramesSender(spec.env),
                rb_receiver=streaming.StateRewardReceiver(spec.rb_address),
            )

        # Samples are extracted from memory, not observed directly
        memory = SequentialMemory(
            limit=spec.memory_limit, window_length=atari_agent.window_length)

        # Linear dicrease of greedy actions
        train_policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(), attr="eps", value_max=spec.random_max,
            value_min=spec.random_min, value_test=spec.random_test,
            nb_steps=spec.random_decay_steps,
        )

        # Test policy: constant eps or per-episode
        test_policy = (
            EpsGreedyQPolicy(eps=spec.random_test) if not spec.random_epsilon
            else models.EpisodeRandomEpsPolicy(
                min_eps=0.0, max_eps=spec.random_test)
        )

        # RL agent
        dqn = DQNAgent(
            model=atari_agent.model,
            enable_double_dqn=True,
            enable_dueling_network=False,
            nb_actions=n_actions,
            policy=train_policy,
            test_policy=test_policy,
            memory=memory,
            processor=atari_agent.processor,
            nb_steps_warmup=spec.steps_warmup,
            gamma=spec.gamma,
            batch_size=spec.batch_size,
            train_interval=spec.train_interval,
            target_model_update=spec.target_update,
            delta_clip=1.0,
            custom_model_objects=atari_agent.custom_layers,
        )
        dqn.compile(
            optimizer=Adam(lr=spec.learning_rate),
            metrics=["mae"]
        )

        return dqn, atari_agent

    def train(self):
        """Train."""

        # Resume?
        init_step, init_episode = 0, 0
        if self.resuming:
            init_step, init_episode = self.saver.load(self.initialize_from)

        # Go
        self.kerasrl_agent.fit(
            self.env,
            callbacks=self.callbacks,
            nb_steps=10000000,
            log_interval=self._log_interval,
            init_step=init_step,
            init_episode=init_episode,
            nb_max_episode_steps=self._max_episode_steps,
        )

        # Save final weights
        self.saver.save()


class CheckpointSaver(Callback):
    """Save weights and restore.

    This class can be used as a callback or directly.
    """

    save_format = "h5"

    def __init__(self, agent, path, interval):
        """Initialize.

        :param agent: a keras-rl agent
        :param path: directory of checkpoints
        :param interval: save frequency in number of steps
        """

        # Super
        Callback.__init__(self)

        # Store
        self.agent = agent
        self.interval = interval
        self.init_step = 0
        self.step = 0
        self.episode = 0

        self.counters_file = os.path.join(
            path, os.path.pardir, "counters.json")
        self.step_checkpoints = os.path.join(
            path, "weights_{step}." + self.save_format)

    def _update_counters(self, filepath):
        """Updates the file of counters with a new entry.

        Counters is a json file which associates each checkpoint to
        an episode and step. The file may not exist.

        :param filepath: checkpoint that is being saved
        """

        counters = {}

        # Load
        if os.path.exists(self.counters_file):
            with open(self.counters_file) as f:
                counters = json.load(f)

        counters[filepath] = dict(episode=self.episode, step=self.step)

        # Save
        with open(self.counters_file, "w") as f:
            json.dump(counters, f, indent=4)

    def save(self):
        """Save now."""

        filepath = self.step_checkpoints.format(step=self.step)

        self.agent.save_weights(
            filepath, overwrite=True, save_format=self.save_format)
        self._update_counters(filepath)

    def load(self, path):
        """Load the weights from a checkpoint.

        :param path: load checkpoint at this path
        :return: tuple of (step, episode) of the restored checkpoint instant
        """

        # Restore
        self.agent.load_weights(path)
        print("> Loaded:", path)

        # Read counters
        with open(self.counters_file) as f:
            data = json.load(f)

        self.step, self.episode = [
            data[path][i] for i in ("step", "episode")]
        self.init_step = self.step

        return self.step, self.episode

    def on_step_end(self, episode_step, logs={}):
        """Keras-rl callback api."""

        self.step += 1
        if (self.step - self.init_step) % self.interval != 0:
            return

        self.save()

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""

        self.episode += 1


class TensorboardLogger(Callback):
    """Log metrics in Tensorboard."""

    def __init__(self, logdir, agent):
        """Initialize.

        :param logdir: directory of tensorboard logs
        :param agent: a QAgentDef instance.
        """

        # Super
        Callback.__init__(self)

        # Store
        self._agent = agent

        # Both are dicts of {episode: data}
        #   where data is a dict of metrics accumulated during an episode
        #   {metric_name: episode_values}
        self._episode_scalars = {}
        self._episode_hists = {}

        # Metrics returned after step and episode to be visualized as
        #   scalars or histograms
        self._kerasrl_step_scalars = ["reward", "metrics"]
        self._kerasrl_step_hists = ["action"]
        self._kerasrl_episode_scalars = ["episode_reward", "nb_episode_steps"]
        self._kerasrl_episode_hists = []

        # Tf writer
        self.summary_writer = tf.summary.create_file_writer(logdir)

    def on_train_begin(self, logs={}):
        """Initialization."""

        self._model_metrics = self.model.metrics_names

    def on_episode_begin(self, episode, logs={}):
        """Initialize the episode averages."""

        # New accumulators
        assert episode not in self._episode_scalars and \
            episode not in self._episode_hists
        self._episode_scalars[episode] = {
            name: [] for name in (self._kerasrl_step_scalars + list(
                self._agent.scalar_step_metrics))}
        self._episode_hists[episode] = {
            name: [] for name in (self._kerasrl_step_hists + list(
                self._agent.hist_step_metrics))}

    def on_episode_end(self, episode, logs={}):
        """Compute and log all metrics."""

        # Get episode metrics
        ep_scalars = {
            name: logs[name] for name in self._kerasrl_episode_scalars}
        ep_hists = {
            name: logs[name] for name in self._kerasrl_episode_hists}

        # Model metrics
        model_scalars_values = self._episode_scalars[episode].pop("metrics")
        model_scalars = ({
            name: value for name, value in
            zip(self._model_metrics, model_scalars_values)}
                if model_scalars_values is not None else {})

        # Accumulate step scalars
        reduced_step_scalars = {
            name: np.mean(self._episode_scalars[episode][name], axis=0)
            for name in self._episode_scalars[episode]
        }
        reduced_step_scalars.update({
            name: np.mean(model_scalars[name], axis=0)
            for name in model_scalars
        })

        # Accumulate step histograms
        reduced_step_hists = {
            name: np.array(self._episode_hists[episode][name])
            for name in self._episode_hists[episode]
        }

        # Combine step and episode
        all_scalars = {**ep_scalars, **reduced_step_scalars}
        all_hists = {**ep_hists, **reduced_step_hists}

        # Log
        self.save_scalars(episode, all_scalars)
        self.save_histograms(episode, all_hists)

        # Free space
        self._episode_scalars.pop(episode)
        self._episode_hists.pop(episode)

    def on_step_end(self, step, logs={}):
        """Collect metrics."""

        episode = logs["episode"]

        # Do not collect training metrics when NaNs
        #   (this happens at steps without a backward pass)
        kerasrl_step_scalars = set(self._kerasrl_step_scalars)
        if np.isnan(logs["metrics"]).all():
            kerasrl_step_scalars.remove("metrics")

        # Collect keras-rl metrics
        for scalar in kerasrl_step_scalars:
            self._episode_scalars[episode][scalar].append(logs[scalar])
        for hist in self._kerasrl_step_hists:
            self._episode_hists[episode][hist].append(logs[hist])

        # Collect custom metrics
        for scalar, value in self._agent.scalar_step_metrics.items():
            self._episode_scalars[episode][scalar].append(value)
        for hist, value in self._agent.hist_step_metrics.items():
            self._episode_hists[episode][hist].append(value)

    def save_scalars(self, step, metrics):
        """Save scalars.

        :param step: the step number (used in plot)
        :param metrics: a dict of {name: scalar}
        """

        # Save
        with self.summary_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=step)

    def save_histograms(self, step, tensors):
        """Visualize tensors as histograms.

        :param step: the step number
        :param tensors: a dict of {name: tensor}
        """

        # Save
        with self.summary_writer.as_default():
            for name, tensor in tensors.items():
                tf.summary.histogram(name, tensor, step)

    def save_graph(self, model):
        """Saves the graph of the Q network of the agent in Tensorboard.

        :param model: a keras model
        """
        # NOTE: This function have a side effect that causes an exception
        #   in keras-rl. Don't use it for now.

        # Forward pass
        @tf.function
        def tracing_model_ops(inputs):
            return model(inputs)

        # Define dummy inputs (of the correct shape)
        inputs = [
            np.zeros((1, *input_.shape[1:]), dtype=input_.dtype.as_numpy_dtype)
            for input_ in model.inputs
        ]

        # Now trace
        tf.summary.trace_on(graph=True)
        tracing_model_ops(inputs)
        with self.summary_writer.as_default():
            tf.summary.trace_export(model.name, step=0)
