"""
Approximate Bayesian Computation (ABC) Model for Inflation Forecasting

This module implements an ABC-based approach to inflation forecasting using
Sequential Monte Carlo (SMC) sampling with adaptive tolerance schedules.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Callable, Optional
import warnings


class ABCInflationModel:
    """
    ABC-based inflation forecasting model.
    
    This model uses Approximate Bayesian Computation to estimate posterior
    distributions of model parameters for inflation forecasting. It employs
    Sequential Monte Carlo (SMC) sampling with adaptive tolerance schedules.
    
    Parameters
    ----------
    prior_distributions : dict
        Dictionary mapping parameter names to scipy.stats distributions
    summary_stats_fn : callable
        Function to compute summary statistics from simulated data
    distance_metric : str, optional
        Distance metric to use ('euclidean', 'manhattan', 'weighted')
    n_particles : int, optional
        Number of particles for ABC-SMC
    epsilon_schedule : list, optional
        Tolerance schedule for ABC-SMC
    """
    
    def __init__(
        self,
        prior_distributions: Dict[str, stats.rv_continuous],
        summary_stats_fn: Callable,
        distance_metric: str = "euclidean",
        n_particles: int = 1000,
        epsilon_schedule: Optional[List[float]] = None
    ):
        self.prior_distributions = prior_distributions
        self.summary_stats_fn = summary_stats_fn
        self.distance_metric = distance_metric
        self.n_particles = n_particles
        self.epsilon_schedule = epsilon_schedule or [1.0, 0.5, 0.25, 0.1]
        
        self.particles = None
        self.weights = None
        self.posterior_samples = None
        
    def sample_prior(self) -> Dict[str, float]:
        """
        Sample from prior distributions.
        
        Returns
        -------
        dict
            Dictionary of sampled parameter values
        """
        return {
            param: dist.rvs()
            for param, dist in self.prior_distributions.items()
        }
    
    def compute_distance(
        self,
        summary_stats_obs: np.ndarray,
        summary_stats_sim: np.ndarray
    ) -> float:
        """
        Compute distance between observed and simulated summary statistics.
        
        Parameters
        ----------
        summary_stats_obs : np.ndarray
            Observed summary statistics
        summary_stats_sim : np.ndarray
            Simulated summary statistics
            
        Returns
        -------
        float
            Distance value
        """
        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((summary_stats_obs - summary_stats_sim) ** 2))
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(summary_stats_obs - summary_stats_sim))
        elif self.distance_metric == "weighted":
            # Weighted Euclidean with element-wise inverse variance weighting
            # Compute variance for each summary statistic dimension
            variance = np.var(summary_stats_obs) if summary_stats_obs.ndim == 1 else 1.0
            weights = 1.0 / (variance + 1e-10)
            return np.sqrt(np.sum(weights * (summary_stats_obs - summary_stats_sim) ** 2))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def simulate_model(
        self,
        parameters: Dict[str, float],
        n_steps: int,
        initial_conditions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate the inflation model with given parameters.
        
        This is a placeholder that should be overridden with actual model dynamics.
        
        Parameters
        ----------
        parameters : dict
            Model parameters
        n_steps : int
            Number of time steps to simulate
        initial_conditions : np.ndarray, optional
            Initial values for the simulation
            
        Returns
        -------
        np.ndarray
            Simulated inflation time series
        """
        # Placeholder: AR(1) process for demonstration
        phi = parameters.get('phi', 0.8)
        sigma = parameters.get('sigma', 0.5)
        mu = parameters.get('mu', 2.0)
        
        if initial_conditions is None:
            y = np.zeros(n_steps)
            y[0] = mu
        else:
            y = np.zeros(n_steps)
            y[0] = initial_conditions[-1] if len(initial_conditions) > 0 else mu
        
        for t in range(1, n_steps):
            y[t] = mu + phi * (y[t-1] - mu) + np.random.normal(0, sigma)
        
        return y
    
    def abc_rejection(
        self,
        observed_data: np.ndarray,
        epsilon: float,
        max_iterations: int = 10000
    ) -> List[Dict[str, float]]:
        """
        ABC Rejection sampling algorithm.
        
        Parameters
        ----------
        observed_data : np.ndarray
            Observed inflation time series
        epsilon : float
            Tolerance threshold
        max_iterations : int, optional
            Maximum number of iterations
            
        Returns
        -------
        list
            List of accepted parameter samples
        """
        observed_stats = self.summary_stats_fn(observed_data)
        accepted_samples = []
        
        for _ in range(max_iterations):
            # Sample from prior
            params = self.sample_prior()
            
            # Simulate model
            simulated_data = self.simulate_model(params, len(observed_data))
            simulated_stats = self.summary_stats_fn(simulated_data)
            
            # Compute distance
            distance = self.compute_distance(observed_stats, simulated_stats)
            
            # Accept/reject
            if distance < epsilon:
                accepted_samples.append(params)
                
            if len(accepted_samples) >= self.n_particles:
                break
        
        return accepted_samples
    
    def abc_smc(
        self,
        observed_data: np.ndarray,
        verbose: bool = True
    ) -> Tuple[List[Dict[str, float]], np.ndarray]:
        """
        ABC Sequential Monte Carlo algorithm.
        
        Parameters
        ----------
        observed_data : np.ndarray
            Observed inflation time series
        verbose : bool, optional
            Whether to print progress
            
        Returns
        -------
        tuple
            (particles, weights) - Posterior samples and their weights
        """
        observed_stats = self.summary_stats_fn(observed_data)
        
        # Initialize particles from prior
        particles = [self.sample_prior() for _ in range(self.n_particles)]
        weights = np.ones(self.n_particles) / self.n_particles
        
        for t, epsilon in enumerate(self.epsilon_schedule):
            if verbose:
                print(f"SMC iteration {t+1}/{len(self.epsilon_schedule)}, epsilon={epsilon}")
            
            new_particles = []
            new_weights = []
            
            for i in range(self.n_particles):
                accepted = False
                attempts = 0
                max_attempts = 1000
                
                while not accepted and attempts < max_attempts:
                    # Perturb particle if not first iteration
                    if t == 0:
                        candidate = self.sample_prior()
                    else:
                        # Sample from previous particles with weights
                        idx = np.random.choice(self.n_particles, p=weights)
                        candidate = self._perturb_particle(particles[idx])
                    
                    # Simulate and compute distance
                    simulated_data = self.simulate_model(candidate, len(observed_data))
                    simulated_stats = self.summary_stats_fn(simulated_data)
                    distance = self.compute_distance(observed_stats, simulated_stats)
                    
                    if distance < epsilon:
                        accepted = True
                        new_particles.append(candidate)
                        
                        # Compute weight
                        if t == 0:
                            weight = 1.0
                        else:
                            weight = self._compute_weight(candidate, particles, weights)
                        new_weights.append(weight)
                    
                    attempts += 1
                
                if not accepted:
                    warnings.warn(f"Particle {i} not accepted after {max_attempts} attempts")
                    # Use previous particle
                    new_particles.append(particles[i] if t > 0 else self.sample_prior())
                    new_weights.append(1e-10)
            
            # Normalize weights
            particles = new_particles
            weights = np.array(new_weights)
            weights = weights / np.sum(weights)
        
        self.particles = particles
        self.weights = weights
        self.posterior_samples = particles
        
        return particles, weights
    
    def _perturb_particle(
        self,
        particle: Dict[str, float],
        perturbation_scale: float = 0.1
    ) -> Dict[str, float]:
        """
        Perturb a particle for ABC-SMC.
        
        Parameters
        ----------
        particle : dict
            Parameter dictionary
        perturbation_scale : float, optional
            Scale of perturbation
            
        Returns
        -------
        dict
            Perturbed parameter dictionary
        """
        perturbed = {}
        for param, value in particle.items():
            # Add Gaussian noise
            noise = np.random.normal(0, perturbation_scale * abs(value) + 0.01)
            perturbed[param] = value + noise
            
            # Enforce bounds from prior if available
            if param in self.prior_distributions:
                prior = self.prior_distributions[param]
                if hasattr(prior, 'a') and hasattr(prior, 'b'):
                    perturbed[param] = np.clip(perturbed[param], prior.a, prior.b)
        
        return perturbed
    
    def _compute_weight(
        self,
        particle: Dict[str, float],
        prev_particles: List[Dict[str, float]],
        prev_weights: np.ndarray
    ) -> float:
        """
        Compute importance weight for ABC-SMC.
        
        Parameters
        ----------
        particle : dict
            Current particle
        prev_particles : list
            Previous generation particles
        prev_weights : np.ndarray
            Previous generation weights
            
        Returns
        -------
        float
            Importance weight
        """
        # Compute prior probability
        prior_prob = 1.0
        for param, value in particle.items():
            if param in self.prior_distributions:
                prior_prob *= self.prior_distributions[param].pdf(value)
        
        # Compute proposal probability (sum of perturbed previous particles)
        proposal_prob = 0.0
        for prev_particle, prev_weight in zip(prev_particles, prev_weights):
            # Gaussian kernel density
            kernel_prob = 1.0
            for param, value in particle.items():
                prev_value = prev_particle[param]
                kernel_prob *= stats.norm.pdf(value, prev_value, 0.1 * abs(prev_value) + 0.01)
            proposal_prob += prev_weight * kernel_prob
        
        # Weight is prior / proposal
        if proposal_prob > 0:
            return prior_prob / proposal_prob
        else:
            return 1e-10
    
    def fit(self, observed_data: np.ndarray, method: str = "smc", **kwargs):
        """
        Fit the ABC model to observed data.
        
        Parameters
        ----------
        observed_data : np.ndarray
            Observed inflation time series
        method : str, optional
            Fitting method ('rejection' or 'smc')
        **kwargs
            Additional arguments for the fitting method
        """
        if method == "rejection":
            epsilon = kwargs.get('epsilon', 0.5)
            max_iterations = kwargs.get('max_iterations', 10000)
            self.posterior_samples = self.abc_rejection(observed_data, epsilon, max_iterations)
        elif method == "smc":
            verbose = kwargs.get('verbose', True)
            self.abc_smc(observed_data, verbose)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def forecast(
        self,
        n_steps: int,
        n_simulations: int = 100,
        initial_conditions: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecasts using posterior samples.
        
        Parameters
        ----------
        n_steps : int
            Number of steps to forecast
        n_simulations : int, optional
            Number of simulations to run
        initial_conditions : np.ndarray, optional
            Initial values for forecasting
            
        Returns
        -------
        tuple
            (mean_forecast, lower_bound, upper_bound) - Forecast mean and 95% CI
        """
        if self.posterior_samples is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecasts = np.zeros((n_simulations, n_steps))
        
        for i in range(n_simulations):
            # Sample from posterior
            if self.weights is not None:
                idx = np.random.choice(len(self.posterior_samples), p=self.weights)
            else:
                idx = np.random.choice(len(self.posterior_samples))
            
            params = self.posterior_samples[idx]
            
            # Simulate forecast
            forecasts[i, :] = self.simulate_model(params, n_steps, initial_conditions)
        
        # Compute statistics
        mean_forecast = np.mean(forecasts, axis=0)
        lower_bound = np.percentile(forecasts, 2.5, axis=0)
        upper_bound = np.percentile(forecasts, 97.5, axis=0)
        
        return mean_forecast, lower_bound, upper_bound
    
    def get_posterior_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics of posterior distributions.
        
        Returns
        -------
        dict
            Dictionary with mean, std, and quantiles for each parameter
        """
        if self.posterior_samples is None:
            raise ValueError("Model must be fitted before getting posterior summary")
        
        summary = {}
        param_names = self.posterior_samples[0].keys()
        
        for param in param_names:
            values = np.array([p[param] for p in self.posterior_samples])
            
            if self.weights is not None:
                # Weighted statistics
                mean = np.average(values, weights=self.weights)
                std = np.sqrt(np.average((values - mean) ** 2, weights=self.weights))
            else:
                mean = np.mean(values)
                std = np.std(values)
            
            summary[param] = {
                'mean': mean,
                'std': std,
                'q025': np.percentile(values, 2.5),
                'q975': np.percentile(values, 97.5),
            }
        
        return summary
