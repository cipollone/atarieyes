"""This module allows to train a RL agent."""

import gym

from atarieyes.tools import prepare_directories
from atarieyes.agent import models


class Trainer:
    """Train a RL agent on the Atari games."""

    def __init__(self, args):
        """Initialize.

        :param args: namespace of arguments; see --help.
        """

        # Store
        # TODO

        # Dirs
        # TODO

        # Environment
        self.env = gym.make(args.env)

        # Agent: 
        self.agent = models.build_agent()

    def train(self):
        """Train."""

        # NOTE: copied from keras-rl examples TODO

        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
        weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
        checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(args.env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        callbacks += [FileLogger(log_filename, interval=100)]
        dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

        # After training is done, we save the final weights one more time.
        dqn.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        dqn.test(env, nb_episodes=10, visualize=False)
