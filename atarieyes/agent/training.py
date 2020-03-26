"""This module allows to train a RL agent."""

from tensorforce.environments import Environment
from tensorforce.agents import Agent

from atarieyes import tftools
from atarieyes.tftools import CheckpointSaver


class Trainer:
    """Train a RL agent on the atari games with TensorForce."""

    def __init__(self, args):
        """Initialize.

        :param args: namespace of arguments; see --help.
        """

        # Store
        self.save_frequency = args.save_frequency
        self.discount = args.discount
        self.cont = args.cont

        # Dirs
        model_path, log_path = tftools.prepare_directories(
            "agent", args.env, resuming=self.cont, args=args)

        # TensorForce Env
        self.env = Environment.create(
            environment="gym", level=args.env,
            max_episode_steps=args.max_episode_steps
        )

        # TensorForce Agent
        self.agent = Agent.create(
            agent="dqn", environment=self.env, batch_size=args.batch,
            discount=self.discount, learning_rate=args.rate,
            memory=args.max_episode_steps + args.batch,
            summarizer={
                "directory": log_path, "frequency": args.log_frequency,
                "max-summaries": 1, "labels": ["rewards"]
            }
        )

        # Tools
        self.saver = CheckpointSaver(
            self.agent, model_path, model_type="tensorforce")

    def train(self):
        """Train."""

        # New run
        if not self.cont:
            episode = 0
        # Restore
        else:
            episode = self.saver.load(env=self.env)
            print("> Weights restored.")

            # Initial valuation
            self.valuate(episode)
            episode += 1

        # Training loop
        print("> Training")
        while True:

            # Do
            self.train_episode()

            # Periodic savings
            if episode % self.save_frequency == 0:

                metrics = self.valuate()
                self.saver.save(episode, score=metrics["return"])
                print("Episode ", episode, ", metrics: ", metrics,
                      sep="", end="          \r")

            episode += 1

    def train_episode(self):
        """Train on a single episode."""

        # Init episode
        state = self.env.reset()
        terminal = False

        # Iterate steps
        while not terminal:

            # Agent's turn
            action = self.agent.act(states=state)

            # Environment's turn
            state, terminal, reward = self.env.execute(actions=action)

            # Learn
            self.agent.observe(terminal=terminal, reward=reward)

    def valuate(self):
        """Valuate (on a single episode).

        :return: A dictionary of metrics that includes "return"
            (the cumulative discounted reward)
        """

        # Init episode
        state = self.env.reset()
        internals = self.agent.initial_internals()
        terminal = False

        cumulative = 0
        discount_i = 1

        # Iterate steps
        while not terminal:

            # Agent's turn
            action, internals = self.agent.act(
                states=state, internals=internals, evaluation=True)

            # Environment's turn
            state, terminal, reward = self.env.execute(actions=action)

            # Learn
            cumulative = reward * discount_i
            discount_i *= self.discount

        # Metrics
        metrics = {"return": cumulative}
        return metrics