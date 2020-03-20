#!/usr/bin/env python3

"""Main script file."""

import argparse
import gym

from atarieyes import training
from atarieyes import selector


def main():
    """Main function."""

    # Defaults
    batch = 10
    log_frequency = 20
    learning_rate = 0.001

    # Parsing arguments
    parser = argparse.ArgumentParser(
        description="Feature extraction on Atari Games.")
    op_parsers = parser.add_subparsers(help="Operation", dest="op")

    # List environment op
    op_parsers.add_parser("list", help="List all environments.")

    # Train op
    train_parser = op_parsers.add_parser(
        "train", help="Train the feature extractor.")

    train_parser.add_argument(
        "-e", "--env", type=_gym_environment_arg, required=True,
        help="Identifier of a Gym environment")
    train_parser.add_argument(
        "--render", action="store_true", help="Render while training.")
    train_parser.add_argument(
        "-b", "--batch", type=int, default=batch, help="Training batch size.")
    train_parser.add_argument(
        "-l", "--logs", type=int, default=log_frequency,
        help="Save logs after this number of batches")
    train_parser.add_argument(
        "-c", "--continue", action="store_true", dest="cont",
        help="Continue from previous training.")
    train_parser.add_argument(
        "-r", "--rate", type=float, default=learning_rate,
        help="Learning rate of the Adam optimizer.")

    # Feature selection op
    feature_parser = op_parsers.add_parser(
        "select", help="Explicit selection of local features")
    feature_parser.add_argument(
        "-e", "--env", type=_gym_environment_arg,
        help="Identifier of a Gym environment")

    args = parser.parse_args()

    # Go
    if args.op == "list":
        print(_environment_names())
        return
    elif args.op == "select":
        selector.selection_tool(args)
    elif args.op == "train":
        training.Trainer(args).train()


def _environment_names():
    """Return the available list of environments."""

    env_specs = gym.envs.registry.all()
    env_names = [spec.id for spec in env_specs]
    return env_names


def _gym_environment_arg(name):
    """Create a Gym environment, if name is a valid ID. """

    # Check
    if name not in _environment_names():
        msg = name + ' is not a Gym environment.'
        raise argparse.ArgumentTypeError(msg)

    # Don't build yet
    return name


if __name__ == "__main__":
    main()
