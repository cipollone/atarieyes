"""Play with a trained agent."""

import gym

from atarieyes.tools import ArgumentSaver, Namespace, prepare_directories
from atarieyes.agent.training import Trainer, CheckpointSaver


class Player:
    """Play with a trained agent.

    This is useful to visualize the behaviour of a trained agent.
    The agent must be already trained and saved.
    """

    def __init__(self, args):
        """Initialize.

        The agent is reconstructed from a json of saved arguments.
        If args_file is: "logs/agent/BreakoutDeterministic-v4/0/args.json",
        the weights are loaded from a checkpoint in
        "models/agent/BreakoutDeterministic-v4/".

        :param args: namespace of arguments; see --help.
        """

        # Load the arguments
        agent_args = ArgumentSaver.load(args.args_file)

        # Check that this was a trained agent
        if agent_args.what != "agent" or agent_args.op != "train":
            raise RuntimeError(
                "Arguments must represent a `atarieyes agent train ...` "
                "command")

        # Dirs
        model_path, log_path = prepare_directories(
            "agent", agent_args.env, no_create=True)

        # Environment
        self.env = gym.make(agent_args.env)
        self.env_name = agent_args.env

        # Agent
        self.kerasrl_agent = Trainer.build_agent(
            Namespace(agent_args, n_actions=self.env.action_space.n))

        # Load weights
        saver = CheckpointSaver(
            agent=self.kerasrl_agent, path=model_path,
            interval=agent_args.saves
        )
        saver.load(args.step)

    def play(self):
        """Play."""

        # Go
        self.kerasrl_agent.test(
            self.env, nb_episodes=1000, visualize=True)
