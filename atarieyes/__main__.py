#!/usr/bin/env python3

"""Main script file."""

import argparse
import gym

from atarieyes.tftools import ArgumentSaver


def main():
    """Main function."""

    # Defaults
    features_defaults = dict(
        batch=10,
        log_frequency=20,
        learning_rate=1e-3,
    )
    agent_defaults = dict(
        batch=100,
        log_frequency=50,
        save_frequency=5,
        learning_rate=1e-4,
        discount=1.0,
        episode_steps=1000,
        exploration_episodes=50,
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

    # Load arguments from file
    class LoadArguments(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            ArgumentSaver.load(values, namespace)

    parser.add_argument(
        "--from", action=LoadArguments, help="Load arguments from file")

    what_parsers = parser.add_subparsers(dest="what", help="Choose group")

    # RL agent
    agent_train = what_parsers.add_parser(
        "agent", help="Reinforcement Learning agent")
    agent_op = agent_train.add_subparsers(dest="op", help="What to do")

    # Agent train op
    agent_train = agent_op.add_parser("train", help="Train the RL agent")

    agent_train.add_argument(
        "-e", "--env", type=_gym_environment_arg, required=True,
        help="Identifier of a Gym environment")
    agent_train.add_argument(
        "-r", "--rate", type=float, default=agent_defaults["learning_rate"],
        help="Learning rate")
    agent_train.add_argument(
        "-b", "--batch", type=int, default=agent_defaults["batch"],
        help="Training batch size")
    agent_train.add_argument(
        "-l", "--log-frequency", type=int,
        default=agent_defaults["log_frequency"],
        help="Save TensorBorad after this number of STEPS")
    agent_train.add_argument(
        "-s", "--save-frequency", type=int,
        default=agent_defaults["save_frequency"],
        help="Save weights after this number of EPISODES")
    agent_train.add_argument(
        "-M", "--max-episode_steps", type=int,
        default=agent_defaults["episode_steps"],
        help="Max length of each episode. Note: this also affects memory.")
    agent_train.add_argument(
        "-d", "--discount", type=float, default=agent_defaults["discount"],
        help="RL discount factor")
    agent_train.add_argument(
        "-c", "--continue", action="store_true", dest="cont",
        help="Continue from previous training")
    agent_train.add_argument(
        "--render", action="store_true", help="Render while training")
    agent_train.add_argument(
        "--no-validation", action="store_true",
        help="Skip the validation step (useful with --render)")
    agent_train.add_argument(
        "--expl-episodes", type=int,
        default=agent_defaults["exploration_episodes"],
        help="Number of episodes after which exproration rate halves")
    agent_train.add_argument(
        "--stream", type=str,
        help="Generate a stream of frames and send them to this address")
    # TODO: only ip. missing receiving side from features

    # Agent play op
    agent_play = agent_op.add_parser("play", help="Show how the agent plays")

    agent_play.add_argument(
        "-e", "--env", type=_gym_environment_arg, required=True,
        help="Identifier of a Gym environment")
    agent_play.add_argument(
        "-a", "--agent", type=str, required=True,
        help="Trained agent json specification. "
        "Usually under: models/agent/<env_name>/agent.json. "
        "Assuming the checkpoint is in the same directory.")
    agent_play.add_argument(
        "-M", "--max-episode_steps", type=int,
        default=agent_defaults["episode_steps"],
        help="Max length of each episode")

    # Features
    features_parser = what_parsers.add_parser(
        "features", help="Features extraction")
    features_op = features_parser.add_subparsers(dest="op", help="What to do")

    # Features train op
    features_train = features_op.add_parser(
        "train", help="Train the feature extractor")

    features_train.add_argument(
        "-e", "--env", type=_gym_environment_arg, required=True,
        help="Identifier of a Gym environment")
    features_train.add_argument(
        "--render", action="store_true", help="Render while training")
    features_train.add_argument(
        "-b", "--batch", type=int, default=features_defaults["batch"],
        help="Training batch size")
    features_train.add_argument(
        "-l", "--logs", type=int, default=features_defaults["log_frequency"],
        help="Save logs after this number of batches")
    features_train.add_argument(
        "-c", "--continue", action="store_true", dest="cont",
        help="Continue from previous training")
    features_train.add_argument(
        "-r", "--rate", type=float, default=features_defaults["learning_rate"],
        help="Learning rate")

    # Feature selection op
    feature_select = features_op.add_parser(
        "select", help="Explicit selection of local features")
    feature_select.add_argument(
        "-e", "--env", type=_gym_environment_arg, required=True,
        help="Identifier of a Gym environment")

    args = parser.parse_args()

    # Go
    if args.what == "agent":
        if args.op == "train":
            import atarieyes.agent.training as agent_training
            agent_training.Trainer(args).train()
        elif args.op == "play":
            import atarieyes.agent.playing as agent_playing
            agent_playing.Player(args).play()
    elif args.what == "features":
        if args.op == "train":
            import atarieyes.features.training as features_training
            features_training.Trainer(args).train()
        elif args.op == "select":
            import atarieyes.features.selector as features_selector
            features_selector.selection_tool(args)

    # ^ imports are here because Tensorforce startup settings for TensorFlow
    #   are not compatible with mine.


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
