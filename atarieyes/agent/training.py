"""This module allows to train a RL agent."""

import os
import numpy as np
import gym
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from atarieyes.tools import prepare_directories
from atarieyes.agent.models import AtariAgent

WINDOW_LENGTH = 4


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

        # Dirs
        model_path, log_path = prepare_directories(
            "agent", args.env, resuming=args.cont, args=args)
        model_filenames = "weights_{step}.h5f"
        log_filename = "log.json"
        self.model_files = os.path.join(model_path, model_filenames)
        self.log_file = os.path.join(log_path, log_filename)

        # TODO: cont doesn't really resume

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

        # Callbacks
        self.callbacks = [
            ModelIntervalCheckpoint(
                filepath=self.model_files, interval=args.saves),
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
            value_test=.05, nb_steps=1000000)

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
            target_model_update=10000,
            train_interval=4,
            delta_clip=1.0,
        )
        dqn.compile(Adam(lr=self.learning_rate), metrics=["mae"])
        # TODO: which metrics?
        # TODO: why train interval is 4 != batch?

        return dqn

    def train(self):
        """Train."""

        # Go
        self.kerasrl_agent.fit(
            self.env, callbacks=self.callbacks, nb_steps=2000000,
            log_interval=10000)

        # Save final weights
        self.kerasrl_agent.save_weights(
            self.model_files.format("last"), overwrite=True)
