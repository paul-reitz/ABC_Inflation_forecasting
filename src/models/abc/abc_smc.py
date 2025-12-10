"""
ABC Sequential Monte Carlo (ABC-SMC) algorithm.
"""

import numpy as np
from typing import Callable, Dict, List
from .distance_metrics import compute_distance, summary_statistics


class ABCSMC:
    """
    ABC Sequential Monte Carlo algorithm.
    
    More efficient than ABC rejection by using a sequence of decreasing
    tolerance levels and importance resampling.
    
    Reference:
    Sisson, S. A., Fan, Y., & Beaumont, M. A. (2018). 
    Handbook of approximate Bayesian computation. CRC Press.
    """
    
    def __init__(
        self,
        model: Callable,
        prior: Dict[str, Callable],
        distance_metric: str = 'euclidean',
        summary_func: Callable = summary_statistics
    ):
        """
        Initialize ABC-SMC sampler.
        
        Parameters:
        -----------
        model : Callable
            Function that simulates data given parameters
        prior : Dict[str, Callable]
            Dictionary of prior distributions
        distance_metric : str
            Distance metric to use
        summary_func : Callable
            Function to compute summary statistics
        """
        self.model = model
        self.prior = prior
        self.distance_metric = distance_metric
        self.summary_func = summary_func
        
        self.particles = []
        self.weights = []
        self.epsilons = []
    
    def perturbation_kernel(
        self,
        params: Dict,
        kernel_std: Dict[str, float]
    ) -> Dict:
        """
        Perturb parameters using a Gaussian kernel.
        
        Parameters:
        -----------
        params : Dict
            Current parameters
        kernel_std : Dict[str, float]
            Standard deviation for perturbation kernel
            
        Returns:
        --------
        perturbed_params : Dict
            Perturbed parameters
        """
        perturbed = {}
        for param, value in params.items():
            std = kernel_std.get(param, 0.1 * abs(value))
            perturbed[param] = value + np.random.normal(0, std)
        return perturbed
    
    def run(
        self,
        observed_data: np.ndarray,
        epsilon_schedule: List[float],
        n_particles: int = 1000,
        verbose: bool = True
    ) -> Dict:
        """
        Run ABC-SMC algorithm.
        
        Parameters:
        -----------
        observed_data : np.ndarray
            Observed data
        epsilon_schedule : List[float]
            Decreasing sequence of tolerance levels
        n_particles : int
            Number of particles
        verbose : bool
            Print progress
            
        Returns:
        --------
        results : Dict
            Dictionary with particles, weights, and diagnostics
        """
        # Compute summary statistics for observed data
        summary_obs = self.summary_func(observed_data)
        
        # Initialize storage
        self.epsilons = epsilon_schedule
        T = len(epsilon_schedule)
        
        for t, epsilon in enumerate(epsilon_schedule):
            if verbose:
                print(f"\n=== Population {t+1}/{T}, epsilon = {epsilon:.4f} ===")
            
            particles_t = []
            weights_t = []
            distances_t = []
            
            if t == 0:
                # First population: sample from prior
                accepted = 0
                iteration = 0
                max_iterations = n_particles * 1000
                
                while accepted < n_particles and iteration < max_iterations:
                    iteration += 1
                    
                    # Sample from prior
                    params = {param: dist() for param, dist in self.prior.items()}
                    
                    # Simulate and compute distance
                    sim_data = self.model(params)
                    summary_sim = self.summary_func(sim_data)
                    distance = compute_distance(
                        summary_obs, summary_sim, 
                        metric=self.distance_metric
                    )
                    
                    if distance < epsilon:
                        particles_t.append(params)
                        weights_t.append(1.0 / n_particles)  # Uniform weights
                        distances_t.append(distance)
                        accepted += 1
                        
                        if verbose and accepted % 100 == 0:
                            print(f"Accepted {accepted}/{n_particles} particles")
                
                if verbose:
                    print(f"Acceptance rate: {accepted/iteration:.4f}")
            
            else:
                # Subsequent populations: importance resampling
                prev_particles = self.particles[-1]
                prev_weights = self.weights[-1]
                
                # Estimate kernel bandwidth from previous population
                kernel_std = {}
                for param in self.prior.keys():
                    values = np.array([p[param] for p in prev_particles])
                    kernel_std[param] = 2 * np.std(values) * (n_particles ** (-1/5))
                
                accepted = 0
                iteration = 0
                max_iterations = n_particles * 1000
                
                while accepted < n_particles and iteration < max_iterations:
                    iteration += 1
                    
                    # Sample particle from previous population
                    idx = np.random.choice(len(prev_particles), p=prev_weights)
                    params_prev = prev_particles[idx]
                    
                    # Perturb
                    params = self.perturbation_kernel(params_prev, kernel_std)
                    
                    # Simulate and compute distance
                    sim_data = self.model(params)
                    summary_sim = self.summary_func(sim_data)
                    distance = compute_distance(
                        summary_obs, summary_sim,
                        metric=self.distance_metric
                    )
                    
                    if distance < epsilon:
                        # Compute weight
                        # weight = prior / (sum of kernels)
                        # Simplified: uniform weight (can be improved)
                        weight = 1.0 / n_particles
                        
                        particles_t.append(params)
                        weights_t.append(weight)
                        distances_t.append(distance)
                        accepted += 1
                        
                        if verbose and accepted % 100 == 0:
                            print(f"Accepted {accepted}/{n_particles} particles")
                
                # Normalize weights
                weights_t = np.array(weights_t)
                weights_t = weights_t / np.sum(weights_t)
                weights_t = weights_t.tolist()
                
                if verbose:
                    print(f"Acceptance rate: {accepted/iteration:.4f}")
                    print(f"Mean distance: {np.mean(distances_t):.4f}")
            
            self.particles.append(particles_t)
            self.weights.append(weights_t)
        
        return {
            'particles': self.particles,
            'weights': self.weights,
            'epsilons': self.epsilons
        }
    
    def get_posterior_samples(
        self,
        param_name: str,
        population: int = -1
    ) -> np.ndarray:
        """
        Get posterior samples for a parameter.
        
        Parameters:
        -----------
        param_name : str
            Name of parameter
        population : int
            Which population to use (-1 for last)
            
        Returns:
        --------
        samples : np.ndarray
            Weighted posterior samples
        """
        particles = self.particles[population]
        weights = self.weights[population]
        
        values = np.array([p[param_name] for p in particles])
        
        # Resample according to weights
        indices = np.random.choice(
            len(particles),
            size=len(particles),
            replace=True,
            p=weights
        )
        
        return values[indices]
