"""Implementation of genetic algorithms.

Genetic algorithms are really different from other gradient-based methods.
I'll implement the custom training here, then cast this as a generic Model.
"""

from abc import abstractmethod
import math
import tensorflow as tf

from atarieyes.tools import ABC2, AbstractAttribute


class GeneticAlgorithm(ABC2):
    """Top down structure of a Genetic Algorithm.
    
    Values and lists are actually Tf tensors.
    'population' and 'fitness' are two variables that store the current states
    after each training step.
    All subclasses should override with methods that actually work with
    tf.function (use tf.py_function, if necessary).
    """

    def __init__(self, n_individuals, mutation_p):
        """Initialize.

        :param n_individuals: population size.
        :param mutation_p: probability of random mutation for each symbol
            (should be rather small).
        """

        # Store
        self.mutation_p = mutation_p
        self.n_individuals = n_individuals

        assert n_individuals % 2 == 0, "Population must be of even size"

        # Initialize
        self.population = tf.Variable(
            self.initial_population(), trainable=False, name="Population_var")
        self.fitness = tf.Variable(
            self.compute_fitness(self.population), trainable=False,
            name="Fitness_var")

        assert self.population.shape.ndims == 3 and \
            self.population.shape[0] == n_individuals, \
            "Expecting 3D shape: (individuals, symbols, symbol_len)"

        # Constants
        self.n_symbols = self.population.shape[1]
        self.symbol_len = self.population.shape[2]

    @abstractmethod
    def initial_population(self):
        """Generate the initial population.

        Called by GeneticAlgorithm.__init__.

        :return: a list of individuals
        """

    @abstractmethod
    def compute_fitness(self, population):
        """Compute the fitness value (euristic score) for each individual.

        Values (0, +inf) are mapped in probabilities (0, 1).

        :param population: the list of individuals
        :return: a list of float positive fitness values
        """

    @abstractmethod
    def sample_symbols(self, n):
        """Sample n new symbols randomly.

        Each individual is made of a sequence of symbols. This function
        samples on that space.

        :param n: number of symbols to sample
        :return: a 2D Tensor; a sequence of symbols
        """

    @abstractmethod
    def have_solution(self):
        """Return whether a solution has been reached.

        Not all problems can easily tell whether a solution has been found.
        Those can always return False.

        :return: None, if training can continue, or an individual from
            the population if training can stop and consider that as a
            solution.
        """

    def mutate(self, population):
        """Apply rare random mutations to each symbol.
        
        :param population: the list of individuals
        :return: updated list of individuals
        """

        # Select mutations
        samples = tf.random.uniform((self.n_individuals, self.n_symbols), 0, 1)
        mutations = samples < self.mutation_p
        mutations_idx = tf.where(mutations)
        n_mutations = tf.shape(mutations_idx)[0]

        # Sampling
        sampled = self.sample_symbols(n_mutations)

        # Apply
        updates = tf.scatter_nd(
            indices=mutations_idx, updates=sampled,
            shape=population.shape,
        )
        new_population = tf.where(
            mutations[..., tf.newaxis], updates, population)

        return new_population

    def reproduce(self, population, fitness):
        """Updates the individuals in the population based on their fitness.

        :param population: the list of individuals
        :param fitness: returned from compute_fitness; assumed positive.
        :return: updated list of individuals
        """

        assert fitness.shape == [self.n_individuals]

        # Sample a new generation
        logits = tf.math.log(fitness)
        selection = tf.random.categorical([logits], self.n_individuals)
        new_population = tf.gather(population, selection[0])

        assert new_population.shape == population.shape, (
            "Logic error: unexpected shape")

        return new_population

    def crossover(self, population):
        """Apply the crossover to each consecutive pair of individuals.

        :param population: the list of individuals
        :return: updated list of individuals
        """

        # Choose crossover points
        n_pairs = tf.math.floordiv(self.n_individuals, 2)
        positions = tf.random.uniform(
            [n_pairs], 0, self.n_symbols, dtype=tf.int32)

        # Prepare individuals
        parents = tf.reshape(
            population, (n_pairs, 2, self.n_symbols, self.symbol_len))

        # Apply
        new_population = tf.map_fn(
            self._crossover_fn, [parents, positions], dtype=parents.dtype,
            parallel_iterations=20,
        )

        # Return to population
        new_population = tf.reshape(
            new_population, (
                self.n_individuals, self.n_symbols, self.symbol_len))
        assert new_population.shape == population.shape, (
            "Logic error: unexpected shape")

        return new_population

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

    def compute_train_step(self, population, fitness):
        """Computations of the training step.

        :param population: list of individuals
        :param fitness: their computed fitness value
        :return: (new_population, new_fitness)
        """

        population = self.reproduce(population, fitness)
        population = self.crossover(population)
        population = self.mutate(population)
        fitness = self.compute_fitness(population)

        return population, fitness

    @tf.function
    def train_step(self):
        """One training step.

        Graph execution of the ops defined above.
        """

        # Compute new
        population, fitness = self.compute_train_step(
            self.population, self.fitness)

        # Store
        self.population.assign(population)
        self.fitness.assign(fitness)


class BooleanRulesGA(GeneticAlgorithm):
    """Genetic algorithm for Boolean functions.

    The biggest assumption made by this class is that the target concept
    is representable as a boolean expression of NOT, AND only.
    Each individual is a vector of constraints on the input. Each symbol
    has the following meaning: -1 don't care, 0 must be false, 1 must be true.
    """

    def __init__(self, n_inputs, **kwargs):
        """Initialize.

        :param n_inputs: lenght of the binary input vector.
        :param kwargs: GeneticAlgorithm params.
        """

        # Store
        self._n_inputs = n_inputs

        # Super
        GeneticAlgorithm.__init__(self, **kwargs)

    def initial_population(self):
        """Generate the initial population."""

        population = tf.random.uniform(
            (self.n_individuals, self._n_inputs, 1), -1, 2, dtype=tf.int32)
        return tf.cast(population, tf.int8)

    # TODO: compute_fitness

    def sample_symbols(self, n):
        """Sample random symbols."""

        sampled = tf.random.uniform((n, 1), -1, 2, dtype=tf.int32)
        return tf.cast(sampled, tf.int8)

    def have_solution(self):
        """Cannot tell because it's unsupervised."""

        return None


class QueensGA(GeneticAlgorithm):
    """N-queens problem.

    This class is only used to test the algorithm. Use it as a reference.
    Each individual is a vector of heights of each queen.
    """

    def __init__(self, size, **kwargs):

        # Store
        self._size = size
        self._n_pairs = int(
            math.factorial(size) / (2 * math.factorial(size - 2)))

        # Super
        GeneticAlgorithm.__init__(self, **kwargs)

    def initial_population(self):

        positions = tf.random.uniform(
            (self.n_individuals, self._size, 1), 0, self._size, dtype=tf.int32)
        return positions

    def sample_symbols(self, n):

        sampled = tf.random.uniform((n, 1), 0, self._size, dtype=tf.int32)
        return sampled

    def compute_fitness(self, population):
        """Number of non-attacking queens."""

        # Compute positions
        rows = population[:,:,0]
        cols = tf.tile(
            tf.expand_dims(tf.range(self._size), 0), (self.n_individuals, 1))
        diag1 = cols + rows
        diag2 = cols - rows

        # Compute conflicts  (columns are already satisfacted in this repr)
        row_conflicts = tf.map_fn(self._count_conflicts, rows)
        diag1_conflicts = tf.map_fn(self._count_conflicts, diag1)
        diag2_conflicts = tf.map_fn(self._count_conflicts, diag2)
        conflicts = row_conflicts + diag1_conflicts + diag2_conflicts 

        # Compute fitness: number of non-attacking queens
        tf.debugging.assert_less_equal(conflicts, self._n_pairs, "Got " +
            str(conflicts) + " conflicts for " + str(self._size) + " queens")
        non_attacking = self._n_pairs - conflicts

        return tf.cast(non_attacking, tf.float32)

    def _count_conflicts(self, individual):

        _, _, counts = tf.unique_with_counts(individual)
        conflicts = tf.reduce_sum(counts - 1)

        return conflicts

    def have_solution(self):
        """When all queens are non_attacking."""

        solutions = tf.where(self.fitness == self._n_pairs)[:,0]
        if tf.shape(solutions)[0] > 0:
            return self.population[solutions[0]]
        else:
            return None
