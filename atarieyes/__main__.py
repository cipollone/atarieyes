#!/usr/bin/env python3

"""Main script file."""

import argparse
import gym

from atarieyes.tools import ArgumentSaver


def main():
    """Main function."""

    # Defaults
    features_defaults = dict(
        log_frequency=50,
        save_frequency=2000,
        batch_size=50,
        learning_rate=1e-3,
        decay_steps=50,
        l2_const=0.1,
        sparsity_const=0.0,
        sparsity_target=0.2,
        shuffle=10000,
        network_size=[50, 20],
        population_size=5000,
        mutation_p=0.02,
        crossover_p=0.02,
        fitness_range=[30, 100],
        fitness_episodes=2,
        rb_reward=1,
    )
    agent_defaults = dict(
        log_frequency=5000,
        target_update=5000,
        memory_limit=1000000,
        learning_rate=0.00025,
        gamma=0.99,
        batch_size=32,
        train_interval=4,
        random_max=1.0,
        random_min=0.1,
        random_test=0.03,
        steps_warmup=20000,
        save_frequency=100000,
        random_decay_steps=500000,
        max_episode_steps=3000,
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
        "-l", "--logs", type=int, default=agent_defaults["log_frequency"],
        dest="log_frequency", help="Number of steps in each interval")
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
        "-c", "--continue", dest="cont", type=str, metavar="FILE",
        help="Continue training from checkpoint file")
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
        "--rand-eps", action="store_true", dest="random_epsilon",
        help="Randomness varies from 0 to --rand-test for each episode.")
    agent_train.add_argument(
        "--target-update", type=int, metavar="STEPS", dest="target_update",
        default=agent_defaults["target_update"], help="Frequency, in steps, "
        "at which the target model is updated (see DDQN)")
    agent_train.add_argument(
        "--rb", type=str, metavar="IP", dest="rb_address",
        help="Apply to this agent a Restraining Bolt that runs at this "
        "address. The net structure may also change")
    agent_train.add_argument(
        "--no-onelife", action="store_true", dest="no_onelife",
        help="The agent has multiple lives available. It may ecourage "
        "exploration but slow down training")
    agent_train.add_argument(
        "-M", "--max-episode-steps", type=int, metavar="MAX",
        dest="max_episode_steps", default=agent_defaults["max_episode_steps"],
        help="Maximum number of steps in each episode")

    # Agent play op
    agent_play = agent_op.add_parser("play", help="Show how the agent plays")

    agent_play.add_argument(
        "args_file", type=str,
        help="Json file of arguments of a previous training")
    agent_play.add_argument(
        "-c", "--continue", dest="cont", type=str, required=True,
        metavar="FILE", help="Load checkpoint file")
    agent_play.add_argument(
        "-w", "--watch", choices=["render", "stream", "both"],
        default="render", help="Choose how to follow the game: "
        "render on screen, streaming frames, both.")
    agent_play.add_argument(
        "-p", "--port", type=int,
        help="If watching through stream, overrides the default port")
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
    agent_play.add_argument(
        "--rand-eps", action="store_true", dest="random_epsilon",
        help="Randomness varies from 0 to --rand-test for each episode.")

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
    agent_watch.add_argument(
        "-p", "--port", type=int, help="Overrides the default port")


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
        "-r", "--rate", type=float, default=features_defaults["learning_rate"],
        dest="learning_rate", help="Learning rate")
    features_train.add_argument(
        "--stream", type=str, default="localhost",
        help="Ip address of a stream of frames. "
        "That machine could be stated with `--watch stream` option.")
    features_train.add_argument(
        "--decay-steps", type=int, default=features_defaults["decay_steps"],
        help="Learning rate decays of 5% after this number of steps.")
    features_train.add_argument(
        "--decay-rate", action="store_true", dest="decay_rate",
        help="Use a decaying learning rate.")
    features_train.add_argument(
        "--l2-const", type=float, dest="l2_const",
        default=features_defaults["l2_const"],
        help="Scale factor of the L2 loss")
    features_train.add_argument(
        "--sparsity-const", type=float, dest="sparsity_const",
        default=features_defaults["sparsity_const"],
        help="Scale factor of the sparsity promoting loss")
    features_train.add_argument(
        "--sparsity-target", type=float, dest="sparsity_target",
        default=features_defaults["sparsity_target"],
        help="e.g: 0.1 means hidden units active 10% of the time.")
    features_train.add_argument(
        "--shuffle", type=int, default=features_defaults["shuffle"],
        help="Size of the shuffle buffer.")
    features_train.add_argument(
        "--network", type=int, nargs="+", metavar="N_UNITS",
        dest="network_size", default=features_defaults["network_size"],
        help="Number of hidden units, one for each layer (last omitted).")
    features_train.add_argument(
        "--train", type=str, nargs=2, metavar=("REGION", "LAYER"),
        required=True, dest="train_region_layer",
        help="Choose which model to train. "
        "Models are organized regions (a name) and layers (an int). "
        "Last layer is common to all regions, and the region name is ignored.")
    features_train.add_argument(
        "--mutation-p", type=float, metavar="PROBABILITY", dest="mutation_p",
        default=features_defaults["mutation_p"],
        help="Probability of a random mutation inside the genetic algorithm.")
    features_train.add_argument(
        "--crossover-p", type=float, metavar="PROBABILITY", dest="crossover_p",
        default=features_defaults["crossover_p"],
        help="Probability of crossover between each pair.")
    features_train.add_argument(
        "--pop-size", type=int, metavar="SIZE", dest="population_size",
        default=features_defaults["population_size"],
        help="Number of individuals in the genetic algorithm.")
    features_train.add_argument(
        "--fitness", type=int, nargs=2, metavar=("MIN", "MAX"),
        dest="fitness_range", default=features_defaults["fitness_range"],
        help="Min max values of the fitness function.")
    features_train.add_argument(
        "--fitness-episodes", type=int, metavar="N", dest="fitness_episodes",
        default=features_defaults["fitness_episodes"],
        help="Number of episodes to run to evaluate fitness")
    features_train_resuming = features_train.add_mutually_exclusive_group()
    features_train_resuming.add_argument(
        "-c", "--continue", dest="cont", type=str, metavar="FILE.tf",
        help="Continue training from checkpoint")
    features_train_resuming.add_argument(
        "-i", "--init", dest="initialize", type=str, metavar="FILE.tf",
        help="Start from step 0 but initialize from checkpoint ")

    # Feature selection op
    feature_select = features_op.add_parser(
        "select", help="Explicit selection of local features")
    feature_select.add_argument(
        "-e", "--env", type=_gym_environment_arg, required=True,
        help="Identifier of a Gym environment")

    # RestrainingBolt op
    features_rb = features_op.add_parser(
        "rb", help="Start a Restraining Bolt from trained features")

    features_rb.add_argument(
        "args_file", type=str,
        help="Json file of arguments of a previous training")
    features_rb.add_argument(
        "-i", "--init", dest="initialize", type=str, metavar="FILE.tf",
        required=True, help="Load model weights from checkpoint")
    features_rb.add_argument(
        "--stream", type=str, required=True,
        help="Ip address of an agent. This address is used to receive frames "
        "from an agent.")
    features_rb.add_argument(
        "-r", "--reward", dest="rb_reward", type=float,
        default=features_defaults["rb_reward"],
        help="Reward returned by the Bolt at each event")
    features_rb.add_argument(
        "--new", action="store_true", help="By default, it will try to load "
        "a RB previously saved (in the log directory of the initialization). "
        "This option asks to create a new one instead.")

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
                env_name=args.env, ip=args.stream, port=args.port)
    elif args.what == "features":
        if args.op == "train":
            import atarieyes.features.training as features_training
            features_training.Trainer(args).train()
        elif args.op == "select":
            import atarieyes.features.selector as features_selector
            features_selector.selection_tool(args)
        elif args.op == "rb":
            import atarieyes.features.rb as features_rb
            features_rb.Runner(args).run()


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
