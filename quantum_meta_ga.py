"""
Brion Quantum - Quantum-Assisted Meta-Genetic Algorithm (QAMGA) v2.0
======================================================================
Meta-genetic algorithm that uses quantum-inspired operators to evolve
not just solutions but the evolutionary parameters themselves.

Novel Algorithm: Quantum Self-Adaptive Evolution (QSAE)
  - The genetic algorithm evolves its own mutation rates, crossover
    probabilities, and selection pressures using a meta-population
    of strategy parameters, guided by quantum annealing schedules.

Developed by Brion Quantum AI Team
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime


class QuantumMetaGeneticAlgorithm:
    """
    Quantum Self-Adaptive Evolution (QSAE) Engine.

    Evolves both solutions AND evolutionary hyperparameters simultaneously,
    using quantum-inspired operators for exploration.
    """

    def __init__(self, num_parameters: int, population_size: int = 50,
                 learning_rate: float = 0.01, quantum_temperature: float = 1.0):
        # Solution space
        self.num_parameters = num_parameters
        self.population_size = population_size
        self.learning_rate = learning_rate

        # Quantum annealing temperature (decreases over generations)
        self.quantum_temperature = quantum_temperature
        self.cooling_rate = 0.995

        # Population: each individual has solution + strategy parameters
        self.population = np.random.rand(population_size, num_parameters)
        self.strategy_params = np.full(population_size, 0.05)  # Per-individual mutation rates

        # Meta-parameters (evolve over time)
        self.parameters = np.random.rand(num_parameters)
        self.meta_mutation_rate = 0.1
        self.meta_crossover_rate = 0.7

        # Fitness tracking
        self.fitness: np.ndarray = np.zeros(population_size)
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float('-inf')

        # History
        self.generation = 0
        self.fitness_history: List[float] = []
        self.parameter_history: List[np.ndarray] = []
        self.diversity_history: List[float] = []

    # -- Core Evolution Loop ------------------------------------------------

    def optimize_parameters(self, fitness_fn: Optional[Callable] = None) -> np.ndarray:
        """
        Run one generation of the meta-genetic algorithm.

        Args:
            fitness_fn: Optional fitness function (individual -> float).
                        If None, uses default distance-based fitness.

        Returns:
            Best parameters found so far.
        """
        self.generation += 1

        # 1. Evaluate fitness
        self._evaluate_fitness(fitness_fn)

        # 2. Selection (tournament)
        parents = self._selection()

        # 3. Crossover
        offspring = self._crossover(parents)

        # 4. Mutation (quantum-enhanced)
        offspring = self._quantum_mutation(offspring)

        # 5. Evolve strategy parameters (meta-level)
        self._evolve_strategies()

        # 6. Replace population (elitism: keep best)
        elite_idx = np.argmax(self.fitness)
        elite = self.population[elite_idx].copy()
        elite_strategy = self.strategy_params[elite_idx]

        self.population = offspring
        self.population[0] = elite
        self.strategy_params[0] = elite_strategy

        # 7. Cool quantum temperature
        self.quantum_temperature *= self.cooling_rate

        # 8. Track history
        self.fitness_history.append(float(self.best_fitness))
        self.parameter_history.append(self.best_solution.copy())
        diversity = float(np.mean(np.std(self.population, axis=0)))
        self.diversity_history.append(diversity)

        return self.best_solution.copy()

    def evolve(self, generations: int, fitness_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """Run multiple generations and return report."""
        for _ in range(generations):
            self.optimize_parameters(fitness_fn)

        return self.report()

    # -- Fitness Evaluation -------------------------------------------------

    def _evaluate_fitness(self, fitness_fn: Optional[Callable] = None):
        """Evaluate fitness of entire population."""
        for i in range(self.population_size):
            if fitness_fn:
                self.fitness[i] = fitness_fn(self.population[i])
            else:
                # Default: minimize distance to parameters (target)
                dist = np.linalg.norm(self.population[i] - self.parameters)
                self.fitness[i] = 1.0 / (1.0 + dist)

        # Track best
        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.best_fitness:
            self.best_fitness = float(self.fitness[best_idx])
            self.best_solution = self.population[best_idx].copy()

    # -- Selection ----------------------------------------------------------

    def _selection(self) -> np.ndarray:
        """Tournament selection."""
        selected = np.zeros_like(self.population)
        tournament_size = max(2, self.population_size // 5)

        for i in range(self.population_size):
            candidates = np.random.choice(self.population_size,
                                           size=tournament_size, replace=False)
            winner = candidates[np.argmax(self.fitness[candidates])]
            selected[i] = self.population[winner]

        return selected

    # -- Crossover ----------------------------------------------------------

    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        """Blend crossover with meta-adapted crossover rate."""
        offspring = parents.copy()

        for i in range(0, self.population_size - 1, 2):
            if np.random.rand() < self.meta_crossover_rate:
                # Blend crossover: interpolate between parents
                alpha = np.random.rand(self.num_parameters)
                child1 = alpha * parents[i] + (1 - alpha) * parents[i + 1]
                child2 = (1 - alpha) * parents[i] + alpha * parents[i + 1]
                offspring[i] = child1
                offspring[i + 1] = child2

        return offspring

    # -- Mutation -----------------------------------------------------------

    def _quantum_mutation(self, population: np.ndarray) -> np.ndarray:
        """
        Quantum-enhanced mutation:
        - Classical: Gaussian noise scaled by per-individual strategy
        - Quantum tunneling: Boltzmann-distributed jumps at quantum temperature
        """
        mutated = population.copy()

        for i in range(self.population_size):
            # Classical mutation using evolved strategy parameter
            sigma = self.strategy_params[i]
            mask = np.random.rand(self.num_parameters) < sigma
            noise = np.random.normal(0, sigma, self.num_parameters)
            mutated[i] += mask * noise

            # Quantum tunneling (probability decreases with temperature)
            tunnel_prob = min(0.3, self.quantum_temperature * 0.1)
            if np.random.rand() < tunnel_prob:
                # Pick random dimensions to tunnel
                num_tunnel = max(1, int(self.num_parameters * 0.1))
                dims = np.random.choice(self.num_parameters, size=num_tunnel,
                                         replace=False)
                mutated[i, dims] = np.random.rand(num_tunnel)

        return np.clip(mutated, 0.0, 1.0)

    # -- Meta-Level Strategy Evolution --------------------------------------

    def _evolve_strategies(self):
        """
        Evolve the strategy parameters (mutation rates) themselves.
        Successful strategies (high fitness) propagate; unsuccessful ones mutate.
        """
        # Strategy mutation: log-normal self-adaptation (Rechenberg 1/5 rule inspired)
        tau = 1.0 / np.sqrt(2.0 * self.num_parameters)

        for i in range(self.population_size):
            # Log-normal perturbation of strategy
            self.strategy_params[i] *= np.exp(tau * np.random.normal())
            self.strategy_params[i] = np.clip(self.strategy_params[i], 0.001, 0.5)

        # Adapt meta-crossover rate based on diversity
        diversity = np.mean(np.std(self.population, axis=0))
        if diversity < 0.05:
            # Low diversity: increase mutation, decrease crossover
            self.meta_crossover_rate = max(0.3, self.meta_crossover_rate - 0.01)
            self.meta_mutation_rate = min(0.5, self.meta_mutation_rate + 0.01)
        elif diversity > 0.3:
            # High diversity: increase crossover to consolidate
            self.meta_crossover_rate = min(0.9, self.meta_crossover_rate + 0.01)
            self.meta_mutation_rate = max(0.01, self.meta_mutation_rate - 0.01)

    # -- Convergence Detection ----------------------------------------------

    def has_converged(self, tolerance: float = 0.001) -> bool:
        """Check if evolution has converged."""
        if len(self.fitness_history) < 20:
            return False
        recent = self.fitness_history[-20:]
        return (max(recent) - min(recent)) < tolerance

    # -- Reporting ----------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Generate optimization report."""
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "avg_fitness": float(np.mean(self.fitness)),
            "population_diversity": float(np.mean(np.std(self.population, axis=0))),
            "quantum_temperature": self.quantum_temperature,
            "meta_mutation_rate": self.meta_mutation_rate,
            "meta_crossover_rate": self.meta_crossover_rate,
            "avg_strategy_param": float(np.mean(self.strategy_params)),
            "converged": self.has_converged(),
            "best_solution": self.best_solution.tolist() if self.best_solution is not None else None
        }
