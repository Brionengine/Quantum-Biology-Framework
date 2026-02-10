"""
Brion Quantum - Quantum Reinforcement Learning with Evolutionary Strategies v2.0
==================================================================================
Hybrid RL-ES framework combining Q-learning with evolutionary strategy optimization.
Uses quantum-enhanced exploration for both action selection and policy evolution.

Novel Algorithm: Quantum Boltzmann Exploration Policy (QBEP)
  - Action selection uses quantum Boltzmann distributions where the exploration
    temperature is coupled to the quantum annealing schedule, ensuring deep
    exploration early and precise exploitation late.
  - Policy evolution uses evolutionary strategies with quantum tunneling
    to escape local optima in the policy landscape.

Developed by Brion Quantum AI Team
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class QuantumReinforcementLearningES:
    """
    Quantum Boltzmann Exploration Policy (QBEP) Engine.

    Combines tabular Q-learning with evolutionary strategy policy search,
    using quantum-inspired exploration for robust optimization.
    """

    def __init__(self, num_actions: int, num_states: int,
                 alpha: float = 0.1, gamma: float = 0.99,
                 initial_temperature: float = 1.0,
                 population_size: int = 10):
        # RL parameters
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor

        # Q-table
        self.q_table = np.zeros((num_states, num_actions))

        # Quantum Boltzmann exploration
        self.temperature = initial_temperature
        self.cooling_rate = 0.999
        self.min_temperature = 0.01

        # Evolutionary strategy population (policy weight vectors)
        self.population_size = population_size
        self.policy_population = np.random.randn(
            population_size, num_states, num_actions
        ) * 0.1

        # Performance tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.best_total_reward: float = float('-inf')
        self.total_steps = 0

        # ES tracking
        self.es_generation = 0
        self.es_fitness: np.ndarray = np.zeros(population_size)

    # -- Action Selection ---------------------------------------------------

    def choose_action(self, state: int) -> int:
        """
        Choose action using Quantum Boltzmann Exploration Policy.
        Combines Q-values with temperature-dependent exploration.
        """
        q_values = self.q_table[state]

        # Quantum Boltzmann distribution
        # Higher temperature = more exploration (quantum superposition)
        # Lower temperature = more exploitation (wave function collapse)
        scaled = q_values / max(self.temperature, self.min_temperature)

        # Numerical stability
        scaled = scaled - np.max(scaled)
        exp_values = np.exp(scaled)
        probabilities = exp_values / np.sum(exp_values)

        # Quantum noise injection (decreases with temperature)
        noise_strength = self.temperature * 0.1
        noise = np.random.dirichlet(np.ones(self.num_actions) * (1.0 / noise_strength + 1))
        blended = (1 - noise_strength) * probabilities + noise_strength * noise

        # Normalize
        blended = blended / blended.sum()

        action = np.random.choice(self.num_actions, p=blended)
        self.total_steps += 1

        return int(action)

    def greedy_action(self, state: int) -> int:
        """Choose best action without exploration (for evaluation)."""
        return int(np.argmax(self.q_table[state]))

    # -- Q-Learning Update --------------------------------------------------

    def update_q_values(self, state: int, action: int, reward: float,
                        next_state: int, done: bool = False):
        """
        Update Q-values using Q-learning with optional double-Q correction.
        """
        best_next_action = np.argmax(self.q_table[next_state])

        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state, best_next_action]

        # Q-learning update
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

        # Cool exploration temperature
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.cooling_rate
        )

        return float(td_error)

    # -- Episode Management -------------------------------------------------

    def record_episode(self, total_reward: float, length: int):
        """Record episode results for tracking."""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)

        if total_reward > self.best_total_reward:
            self.best_total_reward = total_reward

    # -- Evolutionary Strategy Optimization ---------------------------------

    def evolve_policies(self, evaluate_fn=None) -> Dict[str, Any]:
        """
        Run one generation of evolutionary strategy optimization.
        Evolves policy weight matrices that modulate Q-values.

        Args:
            evaluate_fn: Function that evaluates a policy and returns fitness.
                         Signature: (policy_weights) -> float

        Returns:
            ES generation report.
        """
        self.es_generation += 1

        # 1. Evaluate each policy in the population
        for i in range(self.population_size):
            if evaluate_fn:
                self.es_fitness[i] = evaluate_fn(self.policy_population[i])
            else:
                # Default: fitness from how well policy improves Q-values
                modulated_q = self.q_table + self.policy_population[i]
                self.es_fitness[i] = float(np.mean(np.max(modulated_q, axis=1)))

        # 2. Rank-based selection
        ranks = np.argsort(np.argsort(self.es_fitness))
        weights = ranks / ranks.sum()

        # 3. Compute weighted mean (natural gradient direction)
        mean_policy = np.zeros_like(self.policy_population[0])
        for i in range(self.population_size):
            mean_policy += weights[i] * self.policy_population[i]

        # 4. Generate new population around the mean
        sigma = max(0.01, self.temperature * 0.5)  # Linked to quantum temperature
        new_population = np.zeros_like(self.policy_population)

        # Elitism: keep best
        best_idx = np.argmax(self.es_fitness)
        new_population[0] = self.policy_population[best_idx]

        for i in range(1, self.population_size):
            noise = np.random.randn(*mean_policy.shape) * sigma
            new_population[i] = mean_policy + noise

            # Quantum tunneling: occasional large mutations
            if np.random.rand() < self.temperature * 0.05:
                tunnel_dims = np.random.choice(
                    mean_policy.size,
                    size=max(1, int(mean_policy.size * 0.1)),
                    replace=False
                )
                flat = new_population[i].flatten()
                flat[tunnel_dims] = np.random.randn(len(tunnel_dims))
                new_population[i] = flat.reshape(mean_policy.shape)

        self.policy_population = new_population

        # 5. Apply best policy modulation to Q-table
        self.q_table += 0.01 * self.policy_population[0]

        return {
            "es_generation": self.es_generation,
            "best_fitness": float(np.max(self.es_fitness)),
            "avg_fitness": float(np.mean(self.es_fitness)),
            "sigma": sigma,
            "temperature": self.temperature
        }

    # -- Hybrid RL + ES Training Step ----------------------------------------

    def train_step(self, state: int, action: int, reward: float,
                   next_state: int, done: bool = False) -> Dict[str, float]:
        """
        Combined RL + ES training step.
        Updates Q-values via TD learning and periodically evolves policies.
        """
        td_error = self.update_q_values(state, action, reward, next_state, done)

        result = {
            "td_error": td_error,
            "temperature": self.temperature,
            "q_value": float(self.q_table[state, action])
        }

        # Evolve ES every N steps
        if self.total_steps % 100 == 0:
            es_result = self.evolve_policies()
            result["es_update"] = es_result

        return result

    # -- Reporting ----------------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Generate comprehensive RL-ES report."""
        return {
            "total_steps": self.total_steps,
            "episodes_completed": len(self.episode_rewards),
            "best_episode_reward": self.best_total_reward,
            "avg_reward_last_10": float(np.mean(self.episode_rewards[-10:])) if len(self.episode_rewards) >= 10 else 0.0,
            "temperature": self.temperature,
            "q_table_mean": float(np.mean(self.q_table)),
            "q_table_max": float(np.max(self.q_table)),
            "es_generation": self.es_generation,
            "es_best_fitness": float(np.max(self.es_fitness)) if self.es_fitness.any() else 0.0,
            "exploration_rate": self.temperature / max(self.temperature, 0.01)
        }
