"""Implementation of genetic algorithms.

Genetic algorithms are really different from other gradient-based methods.
I'll implement the custom training here, then cast this as a generic Model.
Values and lists are actually Tf tensors.
"""

from abc import abstractmethod
import numpy as np
import tensorflow as tf

from atarieyes.tools import ABC2, AbstractAttribute

# TODO: tf.function


class GeneticAlgorithm(ABC2):
    """Top down structure of a Genetic Algorithm."""

    # This variable holds the current population (a list of individuals)
    population = AbstractAttribute()

    def __init__(self, mutation_p):
        """Initialize.
        
        :param mutation_p: probability of random mutation for each symbol
            (should be rather small).
        """

        # Population
        self.population = self.initial_population()

        assert self.population.ndim == 3, (
            "Expecting 3D shape: (individuals, symbols, symbol_len)")
        assert self.population.shape[0] % 2 == 0, "Population be of even size"

        # Constants
        self.n_individuals = self.population.shape[0]
        self.n_symbols = self.population.shape[1]
        self.symbol_len = self.population.shape[2]
        self.mutation_p = mutation_p

    @abstractmethod
    def initial_population(self):
        """Generate the initial population.

        Called by GeneticAlgorithm.__init__.

        :return: a list of individuals
        """

    @abstractmethod
    def compute_fitness(self):
        """Compute the fitness value (euristic score) for each individual.

        :return: a list of positive fitness values
        """

    @abstractmethod
    def sample_symbols(self, n):
        """Sample n new symbols randomly.

        Each individual is made of a sequence of symbols. This function
        samples on that space.

        :param n: number of symbols to sample
        :return: a 2D Tensor; a sequence of symbols
        """

    def mutate(self):
        """Apply rare random mutations to each symbol."""

        # Select mutations
        samples = tf.random.uniform((self.n_individuals, self.n_symbols), 0, 1)
        mutations = samples < self.mutation_p
        mutations_idx = tf.where(mutations)
        n_mutations = tf.shape(mutations_idx)[0]

        # Sampling
        sampled = self.sample_symbols(n_mutations)
        assert sampled.shape == (n_mutations, self.symbol_len)

        # Collect indices of the mutated positions
        symbol_indices = tf.range(self.symbol_len, dtype=tf.int64)
        indices2 = tf.reshape(
            tf.tile(symbol_indices[:,tf.newaxis], (1, n_mutations)), (-1, 1))
        indices = tf.tile(mutations_idx, (self.symbol_len, 1))
        indices = tf.concat((indices, indices2), 1)
        sampled = tf.reshape(sampled, [-1])

        # Apply
        sampled_population = tf.sparse.SparseTensor(
            indices, sampled, self.population.shape)
        sampled_population = tf.sparse.to_dense(
            tf.sparse.reorder(sampled_population), validate_indices=False)
        self.population = tf.where(
            mutations[..., tf.newaxis], sampled_population, self.population)

    def reproduce(self, fitness):
        """Updates the individuals in the population based on their fitness.

        :param fitness: returned from compute_fitness; assumed positive.
        """

        # Sample a new generation
        logits = tf.math.log(fitness)
        selection = tf.random.categorical([logits], self.n_individuals)
        population = tf.gather(self.population, selection[0])

        assert population.shape == self.population.shape, (
            "Logic error: unexpected shape")
        self.population = population

    def crossover(self):
        """Apply the crossover to each consecutive pair of individuals."""

        # Choose crossover points
        n_pairs = tf.math.floordiv(self.n_individuals, 2)
        positions = tf.random.uniform(
            [n_pairs], 0, self.n_symbols, dtype=tf.int32)

        # Prepare individuals
        parents = tf.reshape(
            self.population, (n_pairs, 2, self.n_symbols, self.symbol_len))

        # Apply
        population = tf.map_fn(
            self._crossover_fn, [parents, positions], dtype=parents.dtype,
            parallel_iterations=20,
        )

        # Return to population
        population = tf.reshape(
            population, (self.n_individuals, self.n_symbols, self.symbol_len))
        assert population.shape == self.population.shape, (
            "Logic error: unexpected shape")
        self.population = population

    @staticmethod
    def _crossover_fn(tensor):
        """Crossover function (internal use).

        :param tensor: tuple of (pair, position). Pair are two individuals,
            position is a crossover point.
        :return: the new population.
        """

        # Inputs
        pair, position = tensor

        # Swap
        elem0 = tf.concat((pair[0, :position, :], pair[1, position:, :]), 0)
        elem1 = tf.concat((pair[1, :position, :], pair[0, position:, :]), 0)
        pair = tf.stack((elem0, elem1))

        return pair

    def train_step(self):
        """One training step."""

        fitness = self.compute_fitness()
        self.reproduce(fitness)
        self.crossover()
        self.mutate()
