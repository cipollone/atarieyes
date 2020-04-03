"""This module allows to train a RL agent."""

from tensorforce.environments import Environment
from tensorforce.agents import Agent

from atarieyes.tools import prepare_directories
from atarieyes.streaming import AtariFramesSender


class Trainer:
    """Train a RL agent on the atari games with TensorForce."""

    def __init__(self, args):
        """Initialize.

        :param args: namespace of arguments; see --help.
        """

        # Store
        self.discount = args.discount
        self.streaming = args.stream

        # Dirs
        model_path, log_path = prepare_directories(
            "agent", args.env, resuming=args.cont, args=args)

        # TensorForce Env
        self.env = Environment.create(
            environment="gym", level=args.env,
            max_episode_steps=args.max_episode_steps
        )
        if args.render:
            self.env.visualize = True

        # TensorForce Agent (new)
        if not args.cont:
            self.agent = Agent.create(
                agent="dqn", environment=self.env,
                batch_size=args.batch, discount=self.discount,
                memory=args.max_episode_steps + args.batch,
                learning_rate={
                    "type": "decaying",
                    "unit": "episodes", "decay": "exponential",
                    "initial_value": args.rate, "decay_rate": 0.5,
                    "decay_steps": args.rate_episodes, "staircase": False,
                } if args.rate_episodes > 0 else args.rate,
                exploration={
                    "type": "decaying",
                    "unit": "episodes", "decay": "exponential",
                    "initial_value": 0.9, "decay_rate": 0.5,
                    "decay_steps": args.expl_episodes, "staircase": True,
                },
                summarizer={
                    "directory": log_path, "frequency": args.log_frequency,
                    "labels": ["losses", "rewards"],
                },
                saver={
                    "directory": model_path, "filename": "agent",
                    "frequency": args.save_frequency, "max-checkpoints": 3,
                },
            )

        # (resume)
        else:
            self.agent = Agent.load(
                directory=model_path, filename="agent", format="tensorflow",
                environment=self.env)
            print("> Weights restored.")

        # Setup for streaming
        if self.streaming:
            self.sender = AtariFramesSender(args.env)

    def train(self):
        """Train."""

        print("> Training")
        print("Watch it on Tensorboard, or from --stream.")

        # Training loop
        while True:

            # Do
            queries = self.train_episode()

            print(queries, end="          \r")

    def train_episode(self):
        """Train on a single episode.

        :return: a dict of queried tensors
        """

        # Init episode
        state = self.env.reset()
        terminal = False

        act_queries = ["exploration"]
        observe_queries = ["episode", "timestep"]

        # Iterate steps
        while not terminal:

            # Agent's turn
            action, act_tensors = self.agent.act(
                states=state, query=act_queries)

            # Environment's turn
            state, terminal, reward = self.env.execute(actions=action)

            # Learn
            _, observe_tensors = self.agent.observe(
                terminal=terminal, reward=reward, query=observe_queries)

            # Stream?
            if self.streaming:
                self.sender.send(state)

        # Queries at the end of episode
        tensors = dict(zip(act_queries, act_tensors))
        tensors.update(dict(zip(observe_queries, observe_tensors)))
        return tensors
