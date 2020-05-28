"""Tools for a fast simulation of automata in Tensorflow."""

import math
from pythomata.impl.symbolic import SymbolicAutomaton
import numpy as np
import tensorflow as tf


class TfSymbolicAutomaton:
    """A static representation of a SymbolicAutomaton with tensors.

    This representation stores the description of a SymbolicAutomaton with
    static arrays. It can be used to simulate many runs in parallel.

    NOTE: This implementation is not memory efficient. Memory and time required
    during the initialization step are exponential in len(atoms).
    This is proportional to the number of symbols of the DFA, btw.
    Likely, the same cost would be spent when building the input automaton.
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

        # Transform transitions from symbolic to atomic. Transitions are
        #   ([from_state], [atom0_value, ..., atomN_value], [to_state])
        from_states = []
        symbols = []
        to_states = []
        for state in self.states:
            for symbol in self._all_interpretations():
                successor = self.pythomaton.get_successor(state, symbol)
                array_symbol = [int(symbol[atom]) for atom in self.atoms]

                from_states.append(state)
                symbols.append(array_symbol)
                to_states.append(successor)

        from_states = np.array(from_states, dtype=np.int32)
        symbols = np.array(symbols, dtype=np.int32)
        to_states = np.array(to_states, dtype=np.int32)

        assert np.all(from_states >= 0) and np.all(to_states >= 0), (
            "Expected states with positive indices")
        assert np.all(0 <= symbols) and np.all(symbols <= 1), "Logic error"

        # Use state_values as keys
        n_atoms = len(self.atoms)
        self._symbols_field = 10 ** math.ceil(math.log10(2 ** n_atoms))
        self._powers2 = np.array(
            [2 ** i for i in range(n_atoms)], dtype=np.int64)

        # Lookup format
        self.transitions = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                self._to_keys(from_states, symbols), to_states),
            default_value=-1,
        )

        assert self.transitions.key_dtype == tf.int64
        assert self.transitions.value_dtype == tf.int32

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

    def _to_keys(self, states, symbols):
        """Convert the input symbols to keys.

        :param states: a batch of current states; shape [N]
        :param symbols: a batch of assignments for all atoms;
            shape [N, n_atoms]
        """

        states = tf.cast(states, dtype=tf.int64)
        symbols = tf.cast(symbols, dtype=tf.int64)

        symbols_encoded = tf.einsum("ni,i->n", symbols, self._powers2)
        all_encoded = states * self._symbols_field + symbols_encoded

        return all_encoded

    @tf.function
    def successors(self, states, symbols):
        """Return the next state for each transition.

        :param states: a batch of current states; shape [N]
        :param symbols: a batch of assignments for all atoms;
            shape [N, n_atoms]
        :return: a batch of next states (always positive; -1 means logic error)
        """

        # Lookup
        keys = self._to_keys(states, symbols)
        next_states = self.transitions.lookup(keys)

        return next_states

    def initial_states(self, n_instances):
        """Return a batch of initial states

        :param n_instances: batch size
        :return: a one dimensional tesor of initial states
        """

        return tf.broadcast_to([self.initial_state], [n_instances])

    @tf.function
    def is_final(self, states):
        """Return which of these states are final.

        :param states: a [n_instances] batch of states
        """

        # Check against each final state
        checks = (
            tf.expand_dims(states, -1) == tf.expand_dims(self.final_states, 0))
        checks = tf.reduce_any(checks, axis=1)
        return checks
