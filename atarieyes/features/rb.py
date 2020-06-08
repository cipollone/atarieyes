"""Restraining Bolt module."""

from flloat.parser.ldlf import LDLfParser
from flloat.ldlf import LDLfFormula
from pythomata.impl.symbolic import SymbolicAutomaton
from pythomata.simulator import AutomatonSimulator

from atarieyes import tools
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

        # Model (don't train this time)
        loaded_args.train_region_layer = None
        self.model = training.Trainer.build_model(
            loaded_args, log_path=log_path)

        # Restore weights
        self.saver = training.CheckpointSaver(self.model.model, model_path)
        self.saver.load(args.initialize)

        # Restraining Bolt
        self.rb = RestrainingBolt(
            env_name=loaded_args.env,
            fluents=self.model.fluents,
            logdir=log_path,
        )

    def run(self):
        # TODO
        import pdb; pdb.set_trace()
        pass


class RestrainingBolt:
    """RB class.

    "Restraining Bolt" refers to a module that augments a MDP with additional
    rewards and observations. It allows to transform some non-Markovian MDP to
    a Markovian one, thanks to the additional information.

    The non-Markovian MDP is a classic MDP + a temporal goal.
    """

    def __init__(self, env_name, fluents, logdir=None, verbose=True):
        """Initialize.

        :param env_name: a gym atari environment name.
        :param fluents: the list of propositional atoms that are known at
            each step.
        :param logdir: if provided, the automaton just parsed is saved here.
        :param verbose: verbose flag (automaton conversion may take a while).
        """

        # Loading
        if verbose:
            print("> Parsing", env_name, "restraining specification")
        data = selector.read_back(env_name)
        json_rb = data["restraining_bolt"]

        # Parsing
        restraining_spec = " & ".join(json_rb)
        formula = LDLfParser()(restraining_spec)  # type: LDLfFormula

        # Check: all atoms must be evaluated
        atoms = formula.find_labels()
        all_predicted = all((a in fluents for a in atoms))
        if not all_predicted:
            raise ValueError(
                "One of the atoms " + str(atoms) + " is not in fluents")

        # Conversion (slow op)
        automaton = formula.to_automaton()  # type: SymbolicAutomaton
        automaton = automaton.determinize()
        if verbose:
            print("> Parsed")

        # Visualize the automaton
        if logdir is not None:
            graphviz = automaton.to_graphviz()
            graphviz.render(
                "rb.gv", directory=logdir, view=False, cleanup=False)

        # Runner
        simulator = AutomatonSimulator(automaton)

        # Store
        self._env_name = env_name
        self._fluents = fluents
        self._str = restraining_spec
        self._formula = formula
        self._automaton = automaton
        self._simulator = simulator

    def step(self, observation):
        """One step of the Restraining Bolt.

        The observation is a boolean valuation for all fluents.
        At each step, given the current observation, it computes the next
        state and reward that should be passed to the agent.

        :param observation: a dict of {fluent_names: bools}
        :return: a tuple of state (int) and reward (float)
        """

        # TODO: send or return
        # TODO: reset?
