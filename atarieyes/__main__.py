#!/usr/bin/env python3

"""Main script file."""

import argparse
import gym

from atarieyes.tools import ArgumentSaver


def main():
    """Main function."""

    # Defaults
    features_defaults = dict(
        log_frequency=10,
        save_frequency=50,
        batch_size=10,
        learning_rate=1e-3,
    )
    agent_defaults = dict(
        memory_limit=1000000,
        learning_rate=0.00025,
        gamma=0.99,
        batch_size=32,
        train_interval=4,
        random_max=1.0,
        random_min=0.1,
        random_test=0.03,
        steps_warmup=50000,
        save_frequency=200000,
        random_decay_steps=1000000,
        target_update=10000,
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
        dest="learning_rate", help="Learning rate")
    agent_train.add_argument(
        "-g", "--gamma", type=float, default=agent_defaults["gamma"],
        help="RL discount factor")
    agent_train.add_argument(
        "-b", "--batch", type=int, default=agent_defaults["batch_size"],
        dest="batch_size", help="Training batch size")
    agent_train.add_argument(
        "-c", "--continue", dest="cont", type=int,
        metavar="STEP", help="Continue from the checkpoint of step numer..")
    agent_train.add_argument(
        "-d", "--deterministic", action="store_true",
        help="Set a constant seed to ensure repeatability. Note: this is just "
        "for testing, as it could negatively affect initalization of weights.")
    agent_train.add_argument(
        "-m", "--memory", type=int, default=agent_defaults["memory_limit"],
        dest="memory_limit", help="Maximum size of the replay memory")
    agent_train.add_argument(
        "-s", "--saves", type=int, default=agent_defaults["save_frequency"],
        help="Save models after this number of steps")
    agent_train.add_argument(
        "-t", "--train-interval", type=int,
        default=agent_defaults["train_interval"],
        help="Train every <t> number of steps/observations")
    agent_train.add_argument(
        "--warmup", type=int, default=agent_defaults["steps_warmup"],
        dest="steps_warmup", help="Number of observations to collect "
        "before training")
    agent_train.add_argument(
        "--rand-decay", type=int, metavar="STEPS", dest="random_decay_steps",
        default=agent_defaults["random_decay_steps"],
        help="The linar decay policy chooses a random action from rand-max% "
        "to rand-min%, in this number of steps")
    agent_train.add_argument(
        "--rand-max", type=float, metavar="PROB", dest="random_max",
        default=agent_defaults["random_max"], help="The initial (maximum) "
        "value of the probability of a random action")
    agent_train.add_argument(
        "--rand-min", type=float, metavar="PROB", dest="random_min",
        default=agent_defaults["random_min"],
        help="The final (minimum) value of the probability of a random action")
    agent_train.add_argument(
        "--rand-test", type=float, metavar="PROB", dest="random_test",
        default=agent_defaults["random_test"],
        help="Probability of a random action while testing/playing")
    agent_train.add_argument(
        "--target-update", type=int, metavar="STEPS", dest="target_update",
        default=agent_defaults["target_update"], help="Frequency, in steps, "
        "at which the target model is updated (see DDQN)")

    # Agent play op
    agent_play = agent_op.add_parser("play", help="Show how the agent plays")

    agent_play.add_argument(
        "args_file", type=str,
        help="Json file of arguments of a previous training")
    agent_play.add_argument(
        "-c", "--continue", dest="cont", type=int, required=True,
        metavar="STEP", help="Continue from the checkpoint of step numer..")
    agent_play.add_argument(
        "-w", "--watch", choices=["render", "stream", "both"],
        default="render", help="Choose how to follow the game: "
        "render on screen, streaming frames, both.")
    agent_play.add_argument(
        "--skip", type=int, metavar="N_FRAMES", help="Stream frames skipping "
        "a random number of frames (N_FRAMES at most).")
    agent_play.add_argument(
        "-d", "--deterministic", action="store_true",
        help="Set a constant seed to ensure repeatability")
    agent_play.add_argument(
        "--rand-test", type=float, metavar="PROB", dest="random_test",
        default=agent_defaults["random_test"],
        help="Probability of a random action while testing/playing")

    # Agent watch op
    agent_watch = agent_op.add_parser(
        "watch", help="Display a frames while an agent is training")

    agent_watch.add_argument(
        "--stream", type=str, required=True,
        help="Ip address of a stream of frames. "
        "That machine could be stated with `--watch stream` option.")
    agent_watch.add_argument(
        "-e", "--env", type=_gym_environment_arg, required=True,
        help="Identifier of the Gym environmen the agent is being trained on.")

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
        "-b", "--batch", type=int, default=features_defaults["batch_size"],
        dest="batch_size", help="Training batch size")
    features_train.add_argument(
        "-l", "--logs", type=int, default=features_defaults["log_frequency"],
        dest="log_frequency", help="Save logs after this number of batches")
    features_train.add_argument(
        "-s", "--saves", type=int, default=features_defaults["save_frequency"],
        dest="save_frequency",
        help="Save checkpoints after this number of batches")
    features_train.add_argument(
        "-c", "--continue", dest="cont", type=int, metavar="STEP",
        help="Continue from the checkpoint of step numer..")
    features_train.add_argument(
        "-r", "--rate", type=float, default=features_defaults["learning_rate"],
        dest="learning_rate", help="Learning rate")
    features_train.add_argument(
        "--stream", type=str, default="localhost",
        help="Ip address of a stream of frames. "
        "That machine could be stated with `--watch stream` option.")

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
        elif args.op == "watch":
            import atarieyes.streaming as streaming
            streaming.display_atari_frames(
                env_name=args.env, ip=args.stream)
    elif args.what == "features":
        if args.op == "train":
            import atarieyes.features.training as features_training
            features_training.Trainer(args).train()
        elif args.op == "select":
            import atarieyes.features.selector as features_selector
            features_selector.selection_tool(args)


def _environment_names():
    """Return the available list of environments."""

    env_specs = gym.envs.registry.all()
    env_names = [spec.id for spec in env_specs]
    return env_names


def _gym_environment_arg(name):
    """Create a Gym environment, if name is a valid ID. """

    # Check
    if name not in _environment_names():
        msg = name + " is not a Gym environment."
        raise argparse.ArgumentTypeError(msg)

    # Don't build yet
    return name


if __name__ == "__main__":
    main()
