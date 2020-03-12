"""This module allows to train a feature extractor."""

import gym


def random_play(env, render=False):
    """Play randomly a game.

    :param env: Gym Environment name.
    :param render: When true, the environment is rendered.
    :return: a interator of observations, and the boolean 'done'
    """

    env = gym.make(env)
    n_game = 0

    # For each game
    while True:

        # Reset
        env.reset()
        done = False
        if render:
            env.render()

        # Until the end
        while not done:

            # Random agent moves
            action = env.action_space.sample()

            # Environment moves
            observation, reward, done, info = env.step(action)

            # Result
            if render:
                env.render()
            yield observation, done

        n_game += 1

    # Leave
    env.close()


class Trainer:
    """Train a feature extractor."""

    def __init__(self, args):
        """Initialize.
        
        :param args: namespace of arguments; see --help.
        """

        self.game_step = random_play(args.env, args.render)


    def train(self):
        """Train."""

        while True:
            print(next(self.game_step))
            input()
