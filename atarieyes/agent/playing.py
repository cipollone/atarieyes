"""Play with a trained agent."""

import os
from tensorforce.environments import Environment
from tensorforce.agents import Agent

from atarieyes.agent.training import Trainer
from atarieyes.tftools import CheckpointSaver


class Player:
    """Play with a trained agent.

    This is useful to visualize the behaviour of a trained agent,
    not for evaluation (which can be done by Trainer).
    The agent must be trained and defined from the associated json file.
    """

    def __init__(self, args):
        """Initialize.

        :param args: namespace of arguments; see --help.
        """

        # Define (hopefully the same) environment
        self.env = Environment.create(
            environment="gym", level=args.env,
            max_episode_steps=args.max_episode_steps
        )
        self.env.visualize = True

        # Re-define the agent
        self.agent = Agent.create(agent=args.agent, environment=self.env)

        # Load the weights
        model_path = os.path.dirname(args.agent)
        saver = CheckpointSaver(
            self.agent, model_path, model_type="tensorforce")
        saver.load(env=self.env)
        print("> Weights restored.")

    def play(self):
        """Play."""

        # Init
        episode = 0
        self.discount = 1.0   # Needed by Trainer

        while True:

            print("Episode", episode, end="     \r")
            episode += 1

            # Run
            Trainer.run_episode(self)
