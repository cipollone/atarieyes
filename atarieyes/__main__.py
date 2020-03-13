#!/usr/bin/env python3

"""Main script file."""

import argparse
import gym

from atarieyes import training


def main():
    """Main function."""

    # Defaults
    batch = 10

    # Parsing arguments
    parser = argparse.ArgumentParser(
        description="Feature extraction on Atari Games.")
    env_group = parser.add_mutually_exclusive_group(required=True)
    env_group.add_argument(
        "-e", "--env", type=_gym_environment_arg,
        help="Identifier of a Gym environment")
    env_group.add_argument(
        "-l", "--list", action="store_true", help="List all environments")
    parser.add_argument(
        "-r", "--render", action="store_true", help="Render while training.")
    parser.add_argument(
        "-b", "--batch", type=int, default=batch, help="Training batch size.")

    args = parser.parse_args()

    # Go
    if args.list:
        print(_environment_names())
        return
    else:
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
