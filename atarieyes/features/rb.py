"""Restraining Bolt module."""

import os
import itertools
import pickle
import numpy as np
from flloat.parser.ldlf import LDLfParser
from flloat.ldlf import LDLfFormula
from pythomata.impl.symbolic import SymbolicAutomaton
from pythomata.simulator import AutomatonSimulator

from atarieyes import tools, streaming
from atarieyes.features import selector
from atarieyes.features import training


class Runner:
    """Run the Restraining Bolt in a loop."""

    def __init__(self, args):
        """Initialize.

        The network is reconstructed from a json of saved arguments.

        :param args: namespace of arguments; see --help.
        """

        # Load the arguments
        loaded_args = tools.ArgumentSaver.load(args.args_file)

        # Check that this was a trained features extractor
        if loaded_args.what != "features" or loaded_args.op != "train":
            raise RuntimeError(
                "Arguments file must come from a previous "
                "`atarieyes features train ...` command")

        # Dirs
        model_path, log_path = tools.prepare_directories(
            "features", loaded_args.env, no_create=True)

        # Model of fluents (don't train this time)
        loaded_args.train_region_layer = None
        self.fluents = training.Trainer.build_model(
            loaded_args, log_path=log_path)

        # Restore weights
        self.saver = training.CheckpointSaver(self.fluents.model, model_path)
        self.saver.load(args.initialize)

        # Restraining Bolt
        self.rb = RestrainingBolt(
            env_name=loaded_args.env,
            fluents=self.fluents.fluents,
            reward=args.rb_reward,
            logdir=log_path,
            load=None if args.new else os.path.join(log_path, "rb.pickle"),
        )

        # States are mapped to contiguous indices (0 based)
        self.states_map = {
            state: i for state, i in zip(
                sorted(self.rb._automaton.states), itertools.count())
        }

        # I/O connection with the agent
        self.rb_sender = streaming.StateRewardSender()
        self.frames_receiver = streaming.AtariFramesReceiver(
            loaded_args.env, args.stream)

    def run(self):
        """Execute the Restraining Bolt.

        This assumes a running instance of an agent.
        """

        # Special message: transmit the number of states
        self.rb_sender.send(len(self.states_map), float("nan"))

        # Loop
        while True:

            # Receive an observation
            frame, termination = self.frames_receiver.receive(wait=True)

            # End of episode?
            if termination == "repeated_last":
                self.rb.step(None)
                continue

            # Make a prediction of all fluents
            inputs = np.expand_dims(frame, 0)
            predicted = self.fluents.predict(inputs)
            assert predicted.shape[0] == 1
            predicted = predicted[0]

            # Update
            state, reward = self.rb.step(predicted)

            # Send a feedback to the agent
            self.rb_sender.send(self.states_map[state], reward)

            # End of episode?
            if termination == "last":
                self.rb.step(None)


class RestrainingBolt:
    """RB class.

    "Restraining Bolt" refers to a module that augments a MDP with additional
    rewards and observations. It allows to transform some non-Markovian MDP to
    a Markovian one, thanks to the additional information.  The non-Markovian
    MDP is a classic MDP + a temporal goal.

    A reward is given when each prefix of the execution satisfies the
    restraining specification. This allows to pass multiple rewards to the
    agent during the execution.  The actual restraining specification is the
    conjunction of "restraining_bolt" and "constraints" fields. This usually
    eliminates many impossible trajectories and simplifies "restraining_bolt"
    formula.

    NOTE: a more correct way to specify to desired behaviours would be just
    to state the whole trace in "restraining_bolt", not select its prefixes
    (which is strange). However, I'm not applying any reward shaping,
    so I'll refer to prefixes to manually provide rewards.

    Load an automaton previously saved, when playing with the Restraining Bolt.
    """

    def __init__(
        self, env_name, fluents, reward, logdir, load=None, verbose=True,
    ):
        """Initialize.

        :param env_name: a gym atari environment name.
        :param fluents: the list of propositional atoms that are known at
            each step.
        :param reward: (float) this reward is returned when the execution
            reaches a final state (at the first instant an execution satisfies
            the restraining specification).
        :param logdir: the automaton just parsed is saved here.
        :param load: if provided, the automaton is not computed but loaded
            from this file. A path to a rb.pickle file.
        :param verbose: verbose flag (automaton conversion may take a while).
        """

        # Read
        data = selector.read_back(env_name)
        json_rb = data["restraining_bolt"] + data["constraints"]

        # Parsing
        restraining_spec = " & ".join(json_rb)
        formula = LDLfParser()(restraining_spec)  # type: LDLfFormula

        # Check: all atoms must be evaluated
        atoms = formula.find_labels()
        all_predicted = all((a in fluents for a in atoms))
        if not all_predicted:
            raise ValueError(
                "One of the atoms " + str(atoms) + " is not in fluents")

        # Parse
        if load is None:

            # Conversion (slow op)
            if verbose:
                print("> Parsing", env_name, "restraining specification")
            automaton = formula.to_automaton()  # type: SymbolicAutomaton
            automaton = automaton.determinize().complete()
            if verbose:
                print("> Parsed")

            # Save and visualize the automaton
            graphviz = automaton.to_graphviz()
            graphviz.render(
                "rb.gv", directory=logdir, view=False, cleanup=False)
            with open(os.path.join(logdir, "rb.pickle"), "wb") as f:
                pickle.dump(automaton, f)

        # Load
        else:
            with open(load, "rb") as f:
                automaton = pickle.load(f)
            if verbose:
                print(">", load, "loaded")

            # Visualize the automaton loaded (debugging)
            graphviz = automaton.to_graphviz()
            graphviz.render(
                "loaded_rb.gv", directory=logdir, view=False, cleanup=False)

        # Runner
        simulator = AutomatonSimulator(automaton)

        # Store
        self.env_name = env_name
        self.fluents = fluents
        self._str = restraining_spec
        self._formula = formula
        self._automaton = automaton
        self._simulator = simulator
        self._reward = reward
        self._last_state = None

    def step(self, observation):
        """One step of the Restraining Bolt.

        The observation is a boolean valuation for all fluents.
        At each step, given the current observation, it computes the next
        state and reward that should be passed to the agent.

        :param observation: a numpy array of boolean predictions for all
            self._fluents, or None at the end of each episode.
        :return: a tuple of state (int) and reward (float)
        """

        # Initialize for a new episode
        if observation is None:
            states = self._simulator.reset()
            assert len(states) == 1
            self._last_state = None
            return set(states).pop(), 0.0

        # Transform to Propositional interpretation
        interpretation = {
            fluent: bool(val == 1)
            for fluent, val in zip(self.fluents, observation)
        }

        # Move the automaton
        states = self._simulator.step(interpretation)
        assert len(states) == 1
        state = set(states).pop()

        # Reward
        if state != self._last_state and self._simulator.is_true():
            reward = self._reward
        else:
            reward = 0.0
        self._last_state = state

        return state, reward
