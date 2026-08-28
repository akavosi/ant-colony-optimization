"""Orchestrates the Elitist Ant System loop and records full run history.

Mirrors the genetic algorithm project's `engine.py` design deliberately:
the engine is decoupled from rendering, producing an `ACOHistory` of
`IterationState` snapshots that the visualization layer consumes
independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.aco.construction import construct_colony
from src.aco.graph import CityLayout, tour_length
from src.aco.pheromone import (
    deposit,
    deposit_elite_bonus,
    evaporate,
    init_pheromone,
    pheromone_entropy,
)


@dataclass
class ACOConfig:
    """Hyperparameters for a single Elitist Ant System run.

    Defaults were chosen via a sweep across 20 seeds on the project's
    17-city instance, for the same reason as the genetic algorithm
    project's tuned defaults: the naive "reasonable-looking" settings
    (beta=3, num_elite=4) solved this instance in 2-13 iterations on
    every seed tested — reliable, but too easy to be an interesting
    animation. The dominant lever turned out to be `beta`: at beta=3,
    the nearness heuristic alone is strong enough to build near-optimal
    tours before pheromone learning contributes anything. Lowering it
    to 0.7 (forcing more reliance on learned pheromone rather than
    built-in nearness) and lowering `num_elite` from 4 to 1 (a gentler
    reinforcement of the best-so-far tour) pushed convergence into the
    16-121 iteration range while staying reliable on all 20 seeds swept.

    The default seed (18) was hand-picked for a particularly
    instructive trajectory: dramatic early progress (from more than
    100% above the true optimum to within 10% in under 20 iterations),
    followed by a genuine 25-iteration plateau stuck around 3% above
    optimal before finally closing the gap at iteration 54 — a clear
    illustration that finding the right neighborhood and precisely
    converging within it are two different achievements.
    """

    num_ants: int = 24
    num_iterations: int = 75
    alpha: float = 1.0
    beta: float = 0.7
    rho: float = 0.35
    initial_pheromone: float = 1.0
    deposit_strength: float = 100.0
    num_elite: float = 1.0
    seed: int = 18


@dataclass
class IterationState:
    """A single iteration's full state, enough to render one frame.

    Attributes:
        iteration: 0-indexed iteration number.
        tours: Shape (num_ants, n), every ant's constructed tour.
        tour_lengths: Shape (num_ants,), each tour's length.
        pheromone: Shape (n, n), pheromone level *after* this
            iteration's evaporation + deposit + elite bonus.
        iteration_best_index: Index into `tours` of this iteration's
            best tour.
        best_ever_tour: Best tour found across all iterations so far.
        best_ever_length: That tour's length.
        is_new_best: True if this iteration improved on the previous
            best-ever length.
        diversity: Normalized pheromone entropy (see
            `pheromone.pheromone_entropy`); 1.0 = uniform/undecided,
            0.0 = fully committed.
    """

    iteration: int
    tours: np.ndarray
    tour_lengths: np.ndarray
    pheromone: np.ndarray
    iteration_best_index: int
    best_ever_tour: np.ndarray
    best_ever_length: float
    is_new_best: bool
    diversity: float


@dataclass
class ACOHistory:
    """Full record of an ACO run, iteration by iteration."""

    layout: CityLayout
    config: ACOConfig
    generations: list[IterationState] = field(default_factory=list)

    @property
    def final(self) -> IterationState:
        return self.generations[-1]


def run_aco(layout: CityLayout, config: ACOConfig) -> ACOHistory:
    """Run Elitist Ant System and return the full iteration history.

    Args:
        layout: The city layout to search over.
        config: Hyperparameters (see `ACOConfig`).

    Returns:
        An `ACOHistory` with one `IterationState` per iteration.
    """
    rng = np.random.default_rng(config.seed)
    n = layout.n

    pheromone = init_pheromone(n, config.initial_pheromone)
    history = ACOHistory(layout=layout, config=config)

    best_ever_length = np.inf
    best_ever_tour = np.arange(n)

    for iteration in range(config.num_iterations):
        tours = construct_colony(
            pheromone, layout.distances, rng, config.num_ants, config.alpha, config.beta
        )
        tour_lengths = np.array([tour_length(t, layout.distances) for t in tours])

        iteration_best_index = int(np.argmin(tour_lengths))
        iteration_best_length = float(tour_lengths[iteration_best_index])

        is_new_best = iteration_best_length < best_ever_length
        if is_new_best:
            best_ever_length = iteration_best_length
            best_ever_tour = tours[iteration_best_index].copy()

        pheromone = evaporate(pheromone, config.rho)
        pheromone = deposit(pheromone, tours, tour_lengths, config.deposit_strength)
        pheromone = deposit_elite_bonus(
            pheromone, best_ever_tour, best_ever_length, config.num_elite, config.deposit_strength
        )

        diversity = pheromone_entropy(pheromone)

        history.generations.append(
            IterationState(
                iteration=iteration,
                tours=tours,
                tour_lengths=tour_lengths,
                pheromone=pheromone,
                iteration_best_index=iteration_best_index,
                best_ever_tour=best_ever_tour.copy(),
                best_ever_length=best_ever_length,
                is_new_best=is_new_best,
                diversity=diversity,
            )
        )

    return history
