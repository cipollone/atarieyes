#!/usr/bin/env python3

"""Main script file."""

import argparse
import gym

import atarieyes.features.selector as features_selector
import atarieyes.features.training as features_training


def main():
    """Main function."""

    # Defaults
    features_defaults = dict(
        batch=10,
        log_frequency=20,
        learning_rate=0.001,
    )

    parser = argparse.ArgumentParser(
        description="Feature extraction and RL on Atari Games")

    # Help: list environments?
    class ListAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            print(_environment_names())
            exit()
    parser.add_argument(
        "--list", action=ListAction, nargs=0,
        help="List all environments, then exit")

    what_parsers = parser.add_subparsers(dest="what", help="Choose group")

    # RL agent op
    what_parsers.add_parser(
        "agent", help="Reinforcement Learning agent.")

    # Features op
    features_parser = what_parsers.add_parser(
        "features", help="Features extraction.")
    features_op = features_parser.add_subparsers(dest="op", help="What to do")

    # Features training op
    features_train = features_op.add_parser(
        "train", help="Train the feature extractor.")

    features_train.add_argument(
        "-e", "--env", type=_gym_environment_arg, required=True,
        help="Identifier of a Gym environment")
    features_train.add_argument(
        "--render", action="store_true", help="Render while training.")
    features_train.add_argument(
        "-b", "--batch", type=int, default=features_defaults["batch"],
        help="Training batch size.")
    features_train.add_argument(
        "-l", "--logs", type=int, default=features_defaults["log_frequency"],
        help="Save logs after this number of batches")
    features_train.add_argument(
        "-c", "--continue", action="store_true", dest="cont",
        help="Continue from previous training.")
    features_train.add_argument(
        "-r", "--rate", type=float, default=features_defaults["learning_rate"],
        help="Learning rate of the Adam optimizer.")

    # Feature selection op
    feature_select = features_op.add_parser(
        "select", help="Explicit selection of local features")
    feature_select.add_argument(
        "-e", "--env", type=_gym_environment_arg, required=True,
        help="Identifier of a Gym environment")

    args = parser.parse_args()

    # Go
    if args.what == "agent":
        raise NotImplementedError
    elif args.what == "features":
        if args.op == "select":
            features_selector.selection_tool(args)
        elif args.op == "train":
            features_training.Trainer(args).train()


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
