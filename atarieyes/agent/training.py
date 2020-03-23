"""This module allows to train a RL agent."""

import gym
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner

from atarieyes import tftools


class Trainer:
    """Train a RL agent on the atari games with TensorForce."""

    def __init__(self, args):
        """Initialize.

        :param args: namespace of arguments; see --help.
        """

        # Store
        self.log_frequency = args.logs
        self.discount = args.discount

        env_name = args.env
        learning_rate = args.rate
        batch = args.batch
        memory = args.memory

        # Dirs
        self.model_path, self.log_path = tftools.prepare_directories(
            "agent", args.env, resuming=False)

        # TensorForce Env
        self.env = Environment.create(
            environment="gym", level=env_name, max_episode_steps=1000
        )

        # TensorForce Agent
        self.agent = Agent.create(
            agent="dqn", environment=self.env, batch_size=batch,
            learning_rate=learning_rate, memory=memory + batch
        )

    def train(self):
        """Train."""

        # Init
        episode = 0

        # Training loop
        print("> Training")
        while True:

            # Do TODO
            #self.train_episode()

            # Logs and savings
            if episode % self.log_frequency == 0:

                cumulative_reward = self.valuate_episode()
                print("Episode ", episode, ", reward ", cumulative_reward,
                    sep="", end="    \r")
                # TODO: log and save

            episode += 1

    def train_episode(self):
        """Train on a single episode."""

        # Init episode
        state = self.env.reset()
        terminal = False

        # Iterate steps
        while not terminal:

            # Agent's turn
            action = self.agent.act(states=state)

            # Environment's turn
            state, terminal, reward = self.env.execute(actions=action)

            # Learn
            self.agent.observe(terminal=terminal, reward=reward)

    def valuate_episode(self):
        """Valuation on a single episode.
        
        :return: cumulative discounted reward.
        """

        # Init episode
        state = self.env.reset()
        internals = self.agent.initial_internals()
        terminal = False

        cumulative = 0
        discount_i = 1

        # Iterate steps
        while not terminal:

            # Agent's turn
            action, internals = self.agent.act(
                states=state, internals=internals, evaluation=True)

            # Environment's turn
            state, terminal, reward = self.env.execute(actions=action)

            # Learn
            cumulative = reward * discount_i
            discount_i *= self.discount

        return cumulative
