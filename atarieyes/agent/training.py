"""This module allows to train a RL agent."""

import gym
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner


class Trainer:
    """Train a RL agent on the atari games with TensorForce."""

    def __init__(self, args):
        """Initialize.

        :param args: namespace of arguments; see --help.
        """

        # Store
        self.env_name = args.env
        self.learning_rate = args.rate
        self.log_frequency = args.logs
        self.batch = args.batch

        # TensorForce Env
        self.env = Environment.create(
            environment="gym", level=self.env_name, max_episode_steps=1000
        )

        # TensorForce Agent
        self.agent = Agent.create(
            agent="dqn", environment=self.env, batch_size=self.batch,
            learning_rate=self.learning_rate, memory=1050,
        )

    def train(self):
        """Train."""

        # Init
        episode = 0
        metrics = "metrics"   # TODO: remove and add valuation function

        # Training loop
        print("> Training")
        while True:

            # Do
            self.train_episode()

            # Logs and savings
            if episode % self.log_frequency == 0:

                print("Episode ", episode, ", ", metrics, sep="", end="    \r")

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
