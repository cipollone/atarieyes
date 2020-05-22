"""Exploit the temporal specification of the fluents.

This module contains anything related to the temporal specifications,
or constraints, of the symbols.
"""

from flloat.parser.ldlf import LDLfParser
from flloat.ldlf import LDLfFormula
from pythomata.impl.symbolic import SymbolicAutomaton
from pythomata.simulator import AutomatonSimulator
import tensorflow as tf

from atarieyes.features import selector


class TemporalConstraints:
    """This represents the set of temporal constraints on all fluents.

    Fluents are a set of propositional atoms that can change over time.
    This class first reads the LDLf formulae from the constraints section in
    the json file (parsing is a slow operation). These are constraints
    that these fluents must respect.
    Then, this class can be used to assess if and how much a sequence of
    observations of those variables respects the constraints.
    This is used to assess the goodness of n boolean functions in evaluating
    the fluents.
    """

    def __init__(
        self, env_name, fluents, n_functions, logdir=None, verbose=True
    ):
        """Initialize.

        :param env_name: a gym environment name.
        :param fluents: the list of propositional atoms that will be predicted.
        :param n_functions: number of predictions to valuate in parallel.
        :param verbose: log the parsing step, because it may take a long time!
        """

        # Loading
        if verbose:
            print("> Parsing", env_name, "constraints")
        data = selector.read_back(env_name)
        json_constraints = data["constraints"]

        # Parsing
        constraint = " & ".join(json_constraints)
        formula = LDLfParser()(constraint)  # type: LDLfFormula

        # Check: all atoms must be evaluated
        atoms = formula.find_labels()
        all_predicted = all((l in fluents for l in atoms))
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
                "constraint.gv", directory=logdir, view=False, cleanup=False)

        # Store
        self.env_name = env_name
        self.fluents = fluents
        self._str = constraint
        self._formula = formula
        self._automaton = automaton
        self._n_functions = n_functions
        self._n_fluents = len(self.fluents)

        # Prepare all parallel simulators (automaton runners)
        # TODO: how? all in python?
        self._simulators = [
            AutomatonSimulator(self._automaton)
            for i in range(self._n_functions)
        ]

    def observe(self, values):
        """Observe a list of predicted values for all fluents.

        This function must be called at each step of the temporal sequence
        to process the input values correctly.

        :param values: a batch of values. A prediction of values for each
            function to evaluate. This should be a Tensor of shape
            (n_functions, n_fluents).
        """

        # Check values shape
        values = tf.convert_to_tensor(values)
        assert values.shape == (self._n_functions, self._n_fluents)

        # Move the automata

    def compute(self):
        """Signal the end of the trace and compute the metrics.

        Call this function once at the end of a trace (sequence of values).
        It also computes the score on the metrics for each of the n functions.

        The metrics are:
            1- consistency. Rate of observations spent on final states.
            2- sensitivity. Rate of reached final states.

        :return: a list of metrics; each is a tensor of shape [n_functions].
        """
        pass
