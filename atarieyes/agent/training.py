"""This module allows to train a RL agent."""

from atarieyes.tools import prepare_directories


class Trainer:
    """Train a RL agent on the Atari games."""

    def __init__(self, args):
        """Initialize.

        :param args: namespace of arguments; see --help.
        """

        # Store
        # TODO

        # Dirs
        model_path, log_path = prepare_directories(
            "agent", args.env, resuming=args.cont, args=args)

        # TODO

    def train(self):
        """Train."""

        # TODO
