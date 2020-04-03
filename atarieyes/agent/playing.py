"""Play with a trained agent."""

from tensorforce.environments import Environment
from tensorforce.agents import Agent


class Player:
    """Play with a trained agent.

    This is useful to visualize the behaviour of a trained agent.
    The agent must be already trained and saved.
    """

    def __init__(self, args):
        """Initialize.

        :param args: namespace of arguments; see --help.
        """

        # Define (hopefully the same) environment
        self.env = Environment.create(
            environment="gym", level=args.env,
            max_episode_steps=args.max_episode_steps
        )
        self.env.visualize = True

        # Load the agent
        self.agent = Agent.load(
            directory=args.agent, filename="agent", format="tensorflow",
            environment=self.env)
        print("> Weights restored.")

    def play(self):
        """Play."""

        episode = 0

        while True:

            print("Episode", episode, end="     \r")
            episode += 1

            # Run
            self.run_episode()

    def run_episode(self):
        """Execute a single episode."""

        # Init episode
        state = self.env.reset()
        internals = self.agent.initial_internals()
        terminal = False

        # Iterate steps
        while not terminal:

            # Agent's turn
            action, internals = self.agent.act(
                states=state, internals=internals, evaluation=True)

            # Environment's turn
            state, terminal, reward = self.env.execute(actions=action)
