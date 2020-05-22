"""A bridbe between pythomata and Tensorflow.

Pythomata is the representation I adopt. This module define tools for a more
efficient execution in tensorflow.
"""

from pythomata.impl.symbolic import SymbolicAutomaton
import numpy as np
import tensorflow as tf


class TfSymbolicAutomaton:
    """A static representation of a SymbolicAutomaton with arrays.

    This representation stores the description of a SymbolicAutomaton with
    static arrays. It should be easier to simulate this automaton with
    Tensorflow ops.

    NOTE: this implementation is not efficient. Memory and time required are
    exponential in len(atoms).
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
        #   [from_state, atom0_value, ..., atomN_value, to_state]
        self.transitions = []
        for state in self.states:
            for symbol in self._all_interpretations():
                successor = self.pythomaton.get_successor(state, symbol)
                array_symbol = [int(symbol[atom]) for atom in self.atoms]
                self.transitions.append(
                    np.concatenate(([state], array_symbol, [successor])))
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


class TfSymbolicSimulator:
    """Execution of symbolic automata with Tensorflow.

    This class can simulate many instances of the same automaton in parallel.
    """

    def __init__(self, automaton, n_instances=1):
        """Initialize.

        :param automaton: a TfSymbolicAutomaton
        :param n_instances: number of parallel simulations
        """

        # Store
        self.automaton = automaton       # type: TfSymbolicAutomaton
        self.n_instances = n_instances

        # Check
        if not isinstance(self.automaton, TfSymbolicAutomaton):
            raise TypeError("Automaton must be a TfSymbolicAutomaton")

        # Current state for each instance
        self.curr_states = tf.Variable(
            np.full([self.n_instances], -1, dtype=np.int32), trainable=False)

        # Ready to go
        self.reset()

    def reset(self):
        """Resets all instances to the initial state.

        :return: the current states
        """

        self.curr_states.assign(
            tf.broadcast_to(self.automaton.initial_state, [self.n_instances]))

        return self.curr_states

    def step(self, symbols):
        """Move all instances.

        :param symbols: one batch of symbols of shape [n_instances, n_atoms].
            Values can be 0 or 1.
        :return: The current states
        """

        # Values are not checked
        assert symbols.shape == [self.n_instances, len(self.automaton.atoms)]

        # Find arcs
        state_and_symbol = tf.concat((self.curr_states, symbols), axis=0)

