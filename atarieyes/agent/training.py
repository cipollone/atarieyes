"""This module allows to train a RL agent."""

import os
import numpy as np
import gym
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.callbacks import Callback, FileLogger

from atarieyes.tools import prepare_directories
from atarieyes.agent.models import AtariAgent


class Trainer:
    """Train a RL agent on the Atari games."""

    def __init__(self, args):
        """Initialize.

        :param args: namespace of arguments; see --help.
        """

        # Store
        self.memory_limit = args.memory
        self.learning_rate = args.rate
        self.steps_warmup = args.warmup
        self.gamma = args.gamma
        self.batch_size = args.batch
        self.train_interval = args.train_interval
        self.cont = args.cont

        # Dirs
        model_path, log_path = prepare_directories(
            "agent", args.env, resuming=self.cont, args=args)
        log_filename = "log.json"
        self.log_file = os.path.join(log_path, log_filename) # TODO: TB logger?

        # Environment
        self.env_name = args.env
        self.env = gym.make(args.env)

        # Repeatability
        if args.deterministic:
            if "Deterministic" not in self.env_name:
                raise ValueError(
                    "--deterministic only works with deterministic"
                    " environments")
            self.env.seed(30013)
            np.random.seed(30013)

        # Agent
        self.kerasrl_agent = self.build_agent()

        # Tools
        self.saver = CheckpointSaver(
            agent=self.kerasrl_agent, path=model_path, interval=args.saves)

        # Callbacks
        self.callbacks = [
            self.saver,
            FileLogger(filepath=self.log_file, interval=100),
        ]

    def build_agent(self):
        """Defines a Keras-rl agent, ready for training.

        :return: the rl agent
        """

        # Samples are extracted from memory, not observed directly
        memory = SequentialMemory(
            limit=self.memory_limit, window_length=AtariAgent.window_length)

        # Linear dicrease of greedy actions
        policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(), attr="eps", value_max=1., value_min=.1,
            value_test=.05, nb_steps=1000000
        )

        # Define network for Atari games
        atari_agent = AtariAgent(n_actions=self.env.action_space.n)

        # RL agent
        dqn = DQNAgent(
            model=atari_agent.model,
            enable_double_dqn=True,
            enable_dueling_network=False,
            nb_actions=self.env.action_space.n,
            policy=policy,
            memory=memory,
            processor=atari_agent.processor,
            nb_steps_warmup=self.steps_warmup,
            gamma=self.gamma,
            batch_size=self.batch_size,
            train_interval=self.train_interval,
            target_model_update=10000,
            delta_clip=1.0,
        )
        dqn.compile(
            optimizer=Adam(lr=self.learning_rate),
            metrics=["mae"]
        )

        return dqn

    def train(self):
        """Train."""

        # Resume?
        if self.cont:
            self.saver.load(self.cont)

        # Go
        self.kerasrl_agent.fit(
            self.env, callbacks=self.callbacks, nb_steps=2000000,
            log_interval=10000)

        # Save final weights
        self.saver.save()


class CheckpointSaver(Callback):
    """Save weights and restore.
    
    This class can be used as a callback or directly.
    """

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
        self.checkpoint = os.path.join(path, "weights.h5f")
        self.step_checkpoints = os.path.join(path, "weights_{step}.h5f")
        self.steps = 0      # Number of trained steps before saving

    def save(self, step=None):
        """Save.

        :param step: if given, the step is appended to the filename
        """

        filepath = self.step_checkpoints.format(step=step) \
            if step else self.checkpoint
        self.agent.save_weights(filepath, overwrite=True)

    def load(self, step=None):
        """Load the weights from a checkpoint.

        :param step: if given, loads from a particular step, otherwise from
            the default (without step) file. True and None mean no step.
        """

        if step is True or step is None:
            filepath = self.checkpoint
        else:
            filepath = self.step_checkpoints.format(step=step)

        self.agent.load_weights(filepath)
        print("> Loaded:", filepath)

    def on_step_end(self, step, logs={}):
        """Keras-rl callback api."""

        self.steps += 1     # can't use step argument
        if self.steps % self.interval != 0:
            return

        self.save(self.steps)
