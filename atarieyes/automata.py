"""A bridbe between pythomata and Tensorflow.

Pythomata is the representation I adopt. This module define tools for a more
efficient execution in tensorflow.
"""

from pythomata.simulator import AbstractSimulator
from pythomata.impl.symbolic import SymbolicAutomaton
import numpy as np


class TfSymbolicAutomaton:
    """A static representation of a SymbolicAutomaton with arrays.

    This representation stores the description of a SymbolicAutomaton with
    static arrays. It should be easier to simulate this automaton with
    Tensorflow ops.
    """

    def __init__(self, automaton, atoms, verbose=True):
        """Initialize.

        :param automaton: a pythomata SymbolicAutomaton with transitions
            defined on atoms.
        :param atoms: list of propositional boolean variables.
            It must be a sequence, not a set, because order is important.
        :param verbose: whether to log the operations
        """

        # Store
        self.pythomaton = automaton  # type: SymbolicAutomaton
        self.atoms = atoms

        # Check
        if not isinstance(self.pythomaton, SymbolicAutomaton):
            raise TypeError("Automaton is not a SymbolicAutomaton")

        # Determinize
        if verbose:
            print("> Automaton conversion")
        self.pythomaton = self.pythomaton.determinize().minimize().complete()

        # Begin conversion: copy states
        self.initial_state = int(self.pythomaton.initial_state)
        self.states = list(self.pythomaton.states)
        self.final_states = list(self.pythomaton._final_states)

        # Transform transitions from symbolic to atomic
        #   Storing transitions as single vectors:
        #   [from_state, to_state, atom0_value, ..., atomN_value]
        self.transitions = []
        for state in self.states:
            for symbol in self._all_interpretations():
                successor = self.pythomaton.get_successor(state, symbol)
                array_symbol = [int(symbol[atom]) for atom in self.atoms]
                self.transitions.append(
                    np.concatenate(([state], [successor], array_symbol)))
        self.transitions = np.array(self.transitions)

        if verbose:
            print("> Converted")

    def _all_interpretations(self):
        """Return all propositional interpretations of the atoms."""

        n_atoms = len(self.atoms)
        n_atoms_str = str(n_atoms)
        for i in range(2 ** n_atoms):
            binary_repr = ("{:0" + n_atoms_str + "b}").format(i)
            interpretation = {
                self.atoms[pos]: bool(int(binary_repr[pos]))
                for pos in range(n_atoms)
            }
            yield interpretation


class TfSymbolicSimulator(AbstractSimulator):
    """Execution of symbolic automata with Tensorflow."""
    pass
