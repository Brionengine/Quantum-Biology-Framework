"""
Brion Quantum - Quantum Evolutionary Environment v2.0
=======================================================
Simulates evolutionary environments with quantum-enhanced interaction
dynamics, fitness landscapes, selection pressures, and speciation events.

Novel Algorithm: Quantum Fitness Landscape Navigation (QFLN)
  - Agents navigate a fitness landscape where quantum tunneling allows
    crossing local fitness barriers that classical evolution cannot.
  - Environmental changes create dynamic landscape deformations that
    test adaptive capacity.

Developed by Brion Quantum AI Team
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class QuantumEvolutionaryEnvironment:
    """
    Quantum Fitness Landscape Navigation (QFLN) Engine.

    Simulates multi-agent evolutionary dynamics with quantum-enhanced
    mutation, selection, and environmental interaction.
    """

    def __init__(self, num_agents: int, genome_size: int = 10,
                 mutation_rate: float = 0.05, selection_pressure: float = 0.3,
                 quantum_tunneling_rate: float = 0.01):
        # Population parameters
        self.num_agents = num_agents
        self.genome_size = genome_size
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        self.quantum_tunneling_rate = quantum_tunneling_rate

        # Population state (each row = agent genome)
        self.environment_state = np.random.rand(num_agents, genome_size)

        # Fitness tracking
        self.fitness: np.ndarray = np.zeros(num_agents)
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []

        # Environment dynamics
        self.generation = 0
        self.environment_params = np.random.rand(genome_size)  # Optimal target
        self._env_drift_rate = 0.001

        # Species tracking
        self.species: Dict[int, List[int]] = {0: list(range(num_agents))}

        # History
        self.event_log: List[Dict[str, Any]] = []

    # -- Simulation Core ----------------------------------------------------

    def simulate_interaction(self) -> np.ndarray:
        """
        Simulate one generation of evolutionary interaction.
        Returns updated population state.
        """
        self.generation += 1

        # 1. Evaluate fitness
        self._evaluate_fitness()

        # 2. Selection
        selected = self._selection()

        # 3. Crossover
        offspring = self._crossover(selected)

        # 4. Mutation (classical + quantum tunneling)
        mutated = self._mutate(offspring)

        # 5. Replace population
        self.environment_state = mutated

        # 6. Environmental drift
        self._environmental_drift()

        # 7. Track history
        self.best_fitness_history.append(float(np.max(self.fitness)))
        self.avg_fitness_history.append(float(np.mean(self.fitness)))

        # 8. Detect speciation
        if self.generation % 10 == 0:
            self._detect_speciation()

        return self.environment_state

    def _evaluate_fitness(self):
        """Evaluate fitness of each agent against the environment."""
        # Fitness = inverse of distance to environmental optimum
        distances = np.linalg.norm(
            self.environment_state - self.environment_params, axis=1
        )
        # Sigmoid fitness function
        self.fitness = 1.0 / (1.0 + distances)

    def _selection(self) -> np.ndarray:
        """Tournament selection with quantum-enhanced exploration."""
        selected_indices = []
        tournament_size = max(2, int(self.num_agents * self.selection_pressure))

        for _ in range(self.num_agents):
            # Tournament selection
            candidates = np.random.choice(self.num_agents, size=tournament_size,
                                           replace=False)
            winner = candidates[np.argmax(self.fitness[candidates])]
            selected_indices.append(winner)

        return self.environment_state[selected_indices].copy()

    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        """Uniform crossover between pairs of parents."""
        offspring = parents.copy()

        for i in range(0, len(offspring) - 1, 2):
            # Uniform crossover mask
            mask = np.random.rand(self.genome_size) > 0.5
            offspring[i][mask], offspring[i+1][mask] = (
                offspring[i+1][mask].copy(), offspring[i][mask].copy()
            )

        return offspring

    def _mutate(self, population: np.ndarray) -> np.ndarray:
        """
        Apply classical mutation and quantum tunneling.
        Classical: small Gaussian perturbations
        Quantum: occasional large jumps to escape local optima
        """
        # Classical mutation
        mutation_mask = np.random.rand(*population.shape) < self.mutation_rate
        classical_noise = np.random.normal(0, 0.1, population.shape)
        population += mutation_mask * classical_noise

        # Quantum tunneling: rare large mutations
        tunnel_mask = np.random.rand(*population.shape) < self.quantum_tunneling_rate
        tunnel_target = np.random.rand(*population.shape)
        population = np.where(tunnel_mask, tunnel_target, population)

        # Clamp to valid range
        return np.clip(population, 0.0, 1.0)

    def _environmental_drift(self):
        """Slowly drift the environmental optimum (changing conditions)."""
        drift = np.random.normal(0, self._env_drift_rate, self.genome_size)
        self.environment_params = np.clip(
            self.environment_params + drift, 0.0, 1.0
        )

    # -- Speciation ---------------------------------------------------------

    def _detect_speciation(self):
        """Detect species clusters based on genomic distance."""
        if self.num_agents < 4:
            return

        # Simple clustering: agents closer than threshold are same species
        threshold = 0.5
        visited = set()
        species = {}
        species_id = 0

        for i in range(self.num_agents):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)

            for j in range(i + 1, self.num_agents):
                if j in visited:
                    continue
                dist = np.linalg.norm(
                    self.environment_state[i] - self.environment_state[j]
                )
                if dist < threshold:
                    cluster.append(j)
                    visited.add(j)

            species[species_id] = cluster
            species_id += 1

        if len(species) != len(self.species):
            self.event_log.append({
                "event": "speciation_change",
                "generation": self.generation,
                "old_species": len(self.species),
                "new_species": len(species),
                "timestamp": datetime.now().isoformat()
            })

        self.species = species

    # -- Environmental Events -----------------------------------------------

    def catastrophic_event(self, severity: float = 0.5):
        """Simulate a catastrophic environmental change."""
        # Dramatically shift the optimum
        shift = np.random.normal(0, severity, self.genome_size)
        self.environment_params = np.clip(
            self.environment_params + shift, 0.0, 1.0
        )

        self.event_log.append({
            "event": "catastrophe",
            "severity": severity,
            "generation": self.generation,
            "timestamp": datetime.now().isoformat()
        })

    def introduce_predator(self, kill_rate: float = 0.2):
        """Simulate predator introduction (remove least fit agents)."""
        self._evaluate_fitness()
        num_killed = int(self.num_agents * kill_rate)
        weakest = np.argsort(self.fitness)[:num_killed]

        # Replace with random new agents
        self.environment_state[weakest] = np.random.rand(num_killed, self.genome_size)

        self.event_log.append({
            "event": "predator_introduction",
            "killed": num_killed,
            "generation": self.generation
        })

    # -- Reporting ----------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Generate environment simulation report."""
        self._evaluate_fitness()

        return {
            "generation": self.generation,
            "num_agents": self.num_agents,
            "num_species": len(self.species),
            "best_fitness": float(np.max(self.fitness)),
            "avg_fitness": float(np.mean(self.fitness)),
            "fitness_std": float(np.std(self.fitness)),
            "diversity": float(np.mean(np.std(self.environment_state, axis=0))),
            "events": len(self.event_log),
            "mutation_rate": self.mutation_rate,
            "tunneling_rate": self.quantum_tunneling_rate
        }
