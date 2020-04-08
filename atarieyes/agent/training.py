"""This module allows to train a RL agent."""

import numpy as np
import gym
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from atarieyes.tools import prepare_directories
from atarieyes.agent.models import AtariAgent

WINDOW_LENGTH = 4


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
        self.env_name = args.env
        self.env = gym.make(args.env)

        # Repeatability
        if args.deterministic:
            if "Deterministic" not in self.env_name:
                raise ValueError(
                    "--deterministic only works with deterministic"
                    " environments")
            self.env.seed(30013)
            np.random.seed(30013)

        # Agent: 
        self.keras_agent = self.build_agent()

    def build_agent(self):


        memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

        # Select a policy. We use eps-greedy action selection, which means that a random action is selected
        # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
        # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
        # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
        # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                      nb_steps=1000000)

        # The trade-off between exploration and exploitation is difficult and an on-going research topic.
        # If you want, you can experiment with the parameters or use a different policy. Another popular one
        # is Boltzmann-style exploration:
        # policy = BoltzmannQPolicy(tau=1.)
        # Feel free to give it a try!

        atari_agent = AtariAgent(n_actions=self.env.action_space.n)

        dqn = DQNAgent(model=atari_agent.model,
            nb_actions=self.env.action_space.n, policy=policy, memory=memory,
            processor=atari_agent.processor, nb_steps_warmup=50000, gamma=.99,
            target_model_update=10000, train_interval=4, delta_clip=1.)
        dqn.compile(Adam(lr=.00025), metrics=['mae'])

        return dqn

    def train(self):
        """Train."""

        # NOTE: copied from keras-rl examples TODO

        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
        weights_filename = 'dqn_{}_weights.h5f'.format(self.env_name)
        checkpoint_weights_filename = 'dqn_' + self.env_name + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(self.env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        callbacks += [FileLogger(log_filename, interval=100)]
        self.keras_agent.fit(self.env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

        # After training is done, we save the final weights one more time.
        self.keras_agent.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        self.keras_agent.test(self.env, nb_episodes=10, visualize=False)
