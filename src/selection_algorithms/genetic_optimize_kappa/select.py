"""
Algorithm M: Genetic Algorithm for Condition Number Optimization

Population-based evolutionary search with multiple fitness modes:
- 'kappa': Minimize condition number
- 'logdet': Maximize log-determinant
- 'correlation': Minimize maximum pairwise correlation

Time: O(G·P·k·m·min(m,k)) where G=generations, P=population, k=dipoles, m=probes
Space: O(P·n) for population storage
"""

import numpy as np
from typing import List, Set, Tuple, Dict, Any
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from selection_algorithms.common import (
    load_forbidden_triads,
    check_triad_violation,
    compute_condition_number,
    save_selection_results
)
from model.utils import build_dipoles
from model.base_matrices.generate_base_matrices import build_s_matrix


def compute_fitness(
    indices: List[int],
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    fitness_mode: str
) -> float:
    """
    Compute fitness score for a given selection.
    
    Lower is better for all modes.
    """
    if fitness_mode == 'kappa':
        # Minimize condition number
        kappa = compute_condition_number(F, B, W, indices)
        return kappa
    
    elif fitness_mode == 'logdet':
        # Maximize log-determinant (minimize negative logdet)
        # log|det(G)| where G = (F@B@S)^T @ (F@B@S)
        S_test = build_s_matrix(indices, F.shape[1])
        G = F @ B @ S_test
        if W is not None:
            G = W @ G
        
        GTG = G.T @ G
        sign, logdet = np.linalg.slogdet(GTG)
        if sign <= 0:
            return 1e10  # Invalid (singular or negative determinant)
        return -logdet  # Negative so lower is better
    
    elif fitness_mode == 'correlation':
        # Minimize maximum pairwise correlation
        S_test = build_s_matrix(indices, F.shape[1])
        cols = F @ B @ S_test
        if W is not None:
            cols = W @ cols
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(cols.T)
        # Mask diagonal
        np.fill_diagonal(corr_matrix, 0)
        max_corr = np.max(np.abs(corr_matrix))
        return max_corr
    
    else:
        raise ValueError(f"Unknown fitness mode: {fitness_mode}")


def is_valid_selection(
    indices: List[int],
    all_dipoles: List[Tuple[int, int]],
    forbidden_triads: List[Set[Tuple[int, int]]]
) -> bool:
    """Check if selection violates any triads."""
    dipoles = [all_dipoles[i] for i in indices]
    return not check_triad_violation(dipoles, forbidden_triads)


def initialize_population(
    n_dipoles: int,
    n_dipoles_target: int,
    population_size: int,
    all_dipoles: List[Tuple[int, int]],
    forbidden_triads: List[Set[Tuple[int, int]]],
    rng: np.random.Generator
) -> List[List[int]]:
    """
    Initialize population with random valid selections.
    """
    population = []
    max_attempts = 1000
    
    for _ in range(population_size):
        attempts = 0
        while attempts < max_attempts:
            # Random selection without replacement
            indices = sorted(rng.choice(n_dipoles, size=n_dipoles_target, replace=False).tolist())
            
            if is_valid_selection(indices, all_dipoles, forbidden_triads):
                population.append(indices)
                break
            
            attempts += 1
        
        if attempts >= max_attempts:
            # Fallback: use first n_dipoles_target indices
            population.append(list(range(n_dipoles_target)))
    
    return population


def tournament_selection(
    population: List[List[int]],
    fitness_scores: List[float],
    tournament_size: int,
    rng: np.random.Generator
) -> List[int]:
    """Select individual via tournament selection."""
    tournament_indices = rng.choice(len(population), size=tournament_size, replace=False)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmin(tournament_fitness)]
    return population[winner_idx].copy()


def crossover(
    parent1: List[int],
    parent2: List[int],
    n_dipoles_target: int,
    rng: np.random.Generator
) -> List[int]:
    """
    Uniform crossover: each gene from parent1 or parent2.
    
    Ensures exactly n_dipoles_target unique indices.
    """
    # Start with random selection from both parents
    child = []
    all_genes = set(parent1 + parent2)
    
    # Take some from parent1, some from parent2
    n_from_p1 = rng.integers(0, n_dipoles_target + 1)
    
    child.extend(rng.choice(parent1, size=min(n_from_p1, len(parent1)), replace=False).tolist())
    
    # Fill rest from parent2
    remaining = n_dipoles_target - len(child)
    if remaining > 0:
        available = [g for g in parent2 if g not in child]
        if len(available) >= remaining:
            child.extend(rng.choice(available, size=remaining, replace=False).tolist())
        else:
            child.extend(available)
    
    # If still not enough, add random genes from all_genes
    while len(child) < n_dipoles_target:
        available = [g for g in all_genes if g not in child]
        if not available:
            break
        child.append(rng.choice(available))
    
    return sorted(child)


def mutate(
    individual: List[int],
    n_dipoles: int,
    mutation_rate: float,
    rng: np.random.Generator
) -> List[int]:
    """
    Mutate individual: replace genes with probability mutation_rate.
    """
    mutated = individual.copy()
    
    for i in range(len(mutated)):
        if rng.random() < mutation_rate:
            # Replace with random gene not in individual
            available = [g for g in range(n_dipoles) if g not in mutated]
            if available:
                mutated[i] = rng.choice(available)
    
    return sorted(mutated)


def select_dipoles_genetic_optimize_kappa(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray = None,
    forbidden_triads: List[Set[Tuple[int, int]]] = None,
    all_dipoles: List[Tuple[int, int]] = None,
    population_size: int = 50,
    n_generations: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    fitness_mode: str = 'kappa',
    n_dipoles_target: int = 18,
    tournament_size: int = 3,
    random_seed: int = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Genetic algorithm for dipole selection.
    
    Evolves a population of candidate selections using:
    - Tournament selection
    - Uniform crossover
    - Random mutation
    - Elitism (best individual always survives)
    
    Parameters
    ----------
    F : np.ndarray (21, 36)
        Dipole-to-scalp transfer matrix
    B : np.ndarray (36, 9)
        Antenna-to-dipole matrix
    W : np.ndarray (21, 21), optional
        Probe weighting matrix
    forbidden_triads : List[Set], optional
        Forbidden triads
    all_dipoles : List[Tuple], optional
        All 36 dipoles
    population_size : int, default=50
        Number of individuals in population
    n_generations : int, default=100
        Number of generations to evolve
    mutation_rate : float, default=0.1
        Probability of mutating each gene
    crossover_rate : float, default=0.8
        Probability of performing crossover
    fitness_mode : str, default='kappa'
        Fitness function: 'kappa', 'logdet', or 'correlation'
    n_dipoles_target : int, default=18
        Target number of dipoles
    tournament_size : int, default=3
        Tournament size for selection
    random_seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected,
                    fitness_history, best_fitness, condition_number,
                    algorithm, parameters
    """
    # Build default dipole list if not provided
    if all_dipoles is None:
        all_dipoles = build_dipoles()
    
    # Load forbidden triads if not provided
    if forbidden_triads is None:
        triads_path = Path(__file__).parent.parent.parent / 'model' / 'forbidden_triads.npy'
        forbidden_triads = load_forbidden_triads(triads_path)
    
    n_dipoles = len(all_dipoles)
    rng = np.random.default_rng(random_seed)
    
    if verbose:
        print("="*60)
        print("Algorithm M: Genetic Algorithm Optimization")
        print("="*60)
        print(f"Fitness mode: {fitness_mode}")
        print(f"Population size: {population_size}")
        print(f"Generations: {n_generations}")
        print(f"Mutation rate: {mutation_rate}")
        print(f"Crossover rate: {crossover_rate}")
        print(f"Target dipoles: {n_dipoles_target}")
        print(f"Tournament size: {tournament_size}")
        if random_seed is not None:
            print(f"Random seed: {random_seed}")
        print()
    
    # Initialize population
    if verbose:
        print("Initializing population...")
    population = initialize_population(
        n_dipoles, n_dipoles_target, population_size,
        all_dipoles, forbidden_triads, rng
    )
    
    # Evaluate initial population
    fitness_scores = [
        compute_fitness(ind, F, B, W, fitness_mode)
        for ind in population
    ]
    
    best_fitness_history = []
    mean_fitness_history = []
    
    # Evolution loop
    for generation in range(n_generations):
        # Track best and mean fitness
        best_fitness = min(fitness_scores)
        mean_fitness = np.mean(fitness_scores)
        best_fitness_history.append(best_fitness)
        mean_fitness_history.append(mean_fitness)
        
        if verbose and (generation % 10 == 0 or generation == n_generations - 1):
            print(f"Gen {generation:3d}: best={best_fitness:.4e}, mean={mean_fitness:.4e}")
        
        # Elitism: keep best individual
        best_idx = np.argmin(fitness_scores)
        elite = population[best_idx].copy()
        
        # Create next generation
        next_population = [elite]  # Elitism
        
        while len(next_population) < population_size:
            # Selection
            parent1 = tournament_selection(population, fitness_scores, tournament_size, rng)
            parent2 = tournament_selection(population, fitness_scores, tournament_size, rng)
            
            # Crossover
            if rng.random() < crossover_rate:
                child = crossover(parent1, parent2, n_dipoles_target, rng)
            else:
                child = parent1.copy()
            
            # Mutation
            child = mutate(child, n_dipoles, mutation_rate, rng)
            
            # Validate (if invalid, retry with parent1)
            if not is_valid_selection(child, all_dipoles, forbidden_triads):
                child = parent1.copy()
            
            next_population.append(child)
        
        # Update population
        population = next_population
        fitness_scores = [
            compute_fitness(ind, F, B, W, fitness_mode)
            for ind in population
        ]
    
    # Final best individual
    best_idx = np.argmin(fitness_scores)
    best_individual = population[best_idx]
    best_fitness = fitness_scores[best_idx]
    
    selected_dipoles = [all_dipoles[i] for i in best_individual]
    
    # Build S matrix
    S = build_s_matrix(best_individual, n_dipoles)
    
    # Compute condition number (regardless of fitness mode)
    kappa_final = compute_condition_number(F, B, W, best_individual)
    
    if verbose:
        print(f"\nFinal best fitness: {best_fitness:.4e}")
        print(f"Selected {len(selected_dipoles)} dipoles, κ={kappa_final:.2e}")
        print("Selected dipoles:", selected_dipoles)
    
    return {
        'S': S,
        'selected_dipoles': selected_dipoles,
        'selected_indices': best_individual,
        'n_selected': len(selected_dipoles),
        'best_fitness': best_fitness,
        'fitness_history': best_fitness_history,
        'mean_fitness_history': mean_fitness_history,
        'condition_number': kappa_final,
        'algorithm': 'genetic_optimize_kappa',
        'parameters': {
            'population_size': population_size,
            'n_generations': n_generations,
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'fitness_mode': fitness_mode,
            'n_dipoles_target': n_dipoles_target,
            'tournament_size': tournament_size,
            'random_seed': random_seed
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Algorithm M: Genetic Algorithm for dipole selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize condition number (default)
  python -m src.selection_algorithms.genetic_optimize_kappa.select \\
      --run-dir run --fitness-mode kappa --n-generations 100

  # Optimize log-determinant
  python -m src.selection_algorithms.genetic_optimize_kappa.select \\
      --run-dir run --fitness-mode logdet --population-size 100 --n-generations 200

  # Minimize correlation
  python -m src.selection_algorithms.genetic_optimize_kappa.select \\
      --run-dir run --fitness-mode correlation --mutation-rate 0.15 --random-seed 42
        """
    )
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Run directory (mesh-specific)')
    parser.add_argument('--f-matrix-path', type=str, required=True,
                        help='Path to F_matrix.npy')
    parser.add_argument('--b-matrix-path', type=str, required=True,
                        help='Path to B_matrix.npy')
    parser.add_argument('--w-matrix-path', type=str, default=None,
                        help='Path to W_matrix.npy (optional)')
    parser.add_argument('--forbidden-triads', type=str, default='src/model/forbidden_triads.npy',
                        help='Path to forbidden triads file')
    parser.add_argument('--population-size', type=int, default=50,
                        help='Population size (default: 50)')
    parser.add_argument('--n-generations', type=int, default=100,
                        help='Number of generations (default: 100)')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                        help='Mutation probability per gene (default: 0.1)')
    parser.add_argument('--crossover-rate', type=float, default=0.8,
                        help='Crossover probability (default: 0.8)')
    parser.add_argument('--fitness-mode', type=str, default='kappa',
                        choices=['kappa', 'logdet', 'correlation'],
                        help='Fitness function (default: kappa)')
    parser.add_argument('--n-dipoles-target', type=int, default=18,
                        help='Target number of dipoles (default: 18)')
    parser.add_argument('--tournament-size', type=int, default=3,
                        help='Tournament size for selection (default: 3)')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to disk')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Build paths from run-dir
    run_dir = Path(args.run_dir)
    output_dir = run_dir / 'results' / 'genetic_optimize_kappa'
    
    # Load matrices
    F = np.load(Path(args.f_matrix_path))
    B = np.load(Path(args.b_matrix_path))
    W = np.load(Path(args.w_matrix_path)) if args.w_matrix_path else None
    
    forbidden_triads = load_forbidden_triads(Path(args.forbidden_triads))
    all_dipoles = build_dipoles()
    
    # Run algorithm
    result = select_dipoles_genetic_optimize_kappa(
        F=F,
        B=B,
        W=W,
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        population_size=args.population_size,
        n_generations=args.n_generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        fitness_mode=args.fitness_mode,
        n_dipoles_target=args.n_dipoles_target,
        tournament_size=args.tournament_size,
        random_seed=args.random_seed,
        verbose=not args.quiet
    )
    
    # Save if requested
    if not args.no_save:
        # Build filename with parameters
        filename_parts = ['S_genetic_optimize_kappa']
        filename_parts.append(f'{args.fitness_mode}')
        if args.n_dipoles_target != 18:
            filename_parts.append(f'n{args.n_dipoles_target}')
        filename_parts.append(f'gen{args.n_generations}')
        if args.random_seed is not None:
            filename_parts.append(f'seed{args.random_seed}')
        
        filename = '_'.join(filename_parts)
        
        save_selection_results(
            S=result['S'],
            metadata={k: v for k, v in result.items() if k != 'S'},
            output_dir=output_dir,
            filename=filename
        )
    
    return result


if __name__ == '__main__':
    main()
