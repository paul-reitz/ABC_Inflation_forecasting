"""
ABC Rejection algorithm.
"""

import numpy as np
from typing import Callable, Dict, List
from .distance_metrics import compute_distance, summary_statistics


class ABCRejection:
    """
    Basic ABC rejection algorithm.
    
    Algorithm:
    1. Sample parameters from prior
    2. Simulate data from model with sampled parameters
    3. Compute summary statistics
    4. Accept if distance < epsilon
    5. Repeat until N samples accepted
    """
    
    def __init__(
        self,
        model: Callable,
        prior: Dict[str, Callable],
        distance_metric: str = 'euclidean',
        summary_func: Callable = summary_statistics
    ):
        """
        Initialize ABC rejection sampler.
        
        Parameters:
        -----------
        model : Callable
            Function that simulates data given parameters
            Signature: model(params) -> np.ndarray
        prior : Dict[str, Callable]
            Dictionary of prior distributions for each parameter
            Each callable should return a single sample
        distance_metric : str
            Distance metric to use
        summary_func : Callable
            Function to compute summary statistics
        """
        self.model = model
        self.prior = prior
        self.distance_metric = distance_metric
        self.summary_func = summary_func
        
        self.accepted_params = []
        self.accepted_distances = []
    
    def sample_prior(self) -> Dict:
        """
        Sample parameters from prior distributions.
        
        Returns:
        --------
        params : Dict
            Sampled parameters
        """
        return {param: dist() for param, dist in self.prior.items()}
    
    def run(
        self,
        observed_data: np.ndarray,
        epsilon: float,
        n_samples: int = 1000,
        max_iterations: int = 100000,
        verbose: bool = True
    ) -> Dict:
        """
        Run ABC rejection algorithm.
        
        Parameters:
        -----------
        observed_data : np.ndarray
            Observed data
        epsilon : float
            Acceptance threshold
        n_samples : int
            Number of samples to accept
        max_iterations : int
            Maximum number of iterations
        verbose : bool
            Print progress
            
        Returns:
        --------
        results : Dict
            Dictionary with accepted parameters and diagnostics
        """
        # Compute summary statistics for observed data
        summary_obs = self.summary_func(observed_data)
        
        self.accepted_params = []
        self.accepted_distances = []
        
        iteration = 0
        accepted = 0
        
        while accepted < n_samples and iteration < max_iterations:
            iteration += 1
            
            # Sample from prior
            params = self.sample_prior()
            
            # Simulate data
            simulated_data = self.model(params)
            
            # Compute summary statistics
            summary_sim = self.summary_func(simulated_data)
            
            # Compute distance
            distance = compute_distance(
                summary_obs, 
                summary_sim, 
                metric=self.distance_metric
            )
            
            # Accept or reject
            if distance < epsilon:
                self.accepted_params.append(params)
                self.accepted_distances.append(distance)
                accepted += 1
                
                if verbose and accepted % 100 == 0:
                    print(f"Accepted {accepted}/{n_samples} samples "
                          f"(iteration {iteration}, acceptance rate: "
                          f"{accepted/iteration:.4f})")
        
        if verbose:
            print(f"\nFinal acceptance rate: {accepted/iteration:.4f}")
            print(f"Total iterations: {iteration}")
        
        return {
            'accepted_params': self.accepted_params,
            'distances': self.accepted_distances,
            'acceptance_rate': accepted / iteration,
            'iterations': iteration
        }
    
    def get_posterior_samples(self, param_name: str) -> np.ndarray:
        """
        Get posterior samples for a specific parameter.
        
        Parameters:
        -----------
        param_name : str
            Name of parameter
            
        Returns:
        --------
        samples : np.ndarray
            Posterior samples
        """
        return np.array([params[param_name] for params in self.accepted_params])
    
    def get_posterior_mean(self, param_name: str) -> float:
        """
        Get posterior mean for a parameter.
        
        Parameters:
        -----------
        param_name : str
            Name of parameter
            
        Returns:
        --------
        mean : float
            Posterior mean
        """
        samples = self.get_posterior_samples(param_name)
        return np.mean(samples)
    
    def get_posterior_quantiles(
        self,
        param_name: str,
        quantiles: List[float] = [0.025, 0.5, 0.975]
    ) -> np.ndarray:
        """
        Get posterior quantiles for a parameter.
        
        Parameters:
        -----------
        param_name : str
            Name of parameter
        quantiles : List[float]
            Quantiles to compute
            
        Returns:
        --------
        quantile_values : np.ndarray
            Posterior quantiles
        """
        samples = self.get_posterior_samples(param_name)
        return np.quantile(samples, quantiles)
