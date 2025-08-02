"""
Treatment effect estimation using causal inference methods.

This module provides comprehensive treatment effect estimation using various
causal inference techniques including backdoor adjustment, propensity score
matching, instrumental variables, and regression discontinuity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from dowhy import CausalModel
from dowhy.causal_estimators import LinearDOWHYEstimator, PropensityScoreEstimator
from .config import ESTIMATION_METHODS, MODEL_CONFIG, STATS_CONFIG

logger = logging.getLogger(__name__)

class TreatmentEffectEstimator:
    """
    Comprehensive treatment effect estimation using multiple methods.
    
    This class implements various causal inference techniques for estimating
    treatment effects, including backdoor adjustment, propensity score methods,
    instrumental variables, and regression discontinuity.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the treatment effect estimator.
        
        Args:
            config: Configuration dictionary for estimation parameters
        """
        self.config = config or {}
        self.results = {}
        self.true_effect = None
        
    def estimate_backdoor_adjustment(self, data: pd.DataFrame,
                                   treatment: str = 'treatment',
                                   outcome: str = 'outcome',
                                   covariates: Optional[List[str]] = None) -> Dict:
        """
        Estimate treatment effect using backdoor adjustment.
        
        Args:
            data: DataFrame containing the data
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            covariates: List of covariates to adjust for
            
        Returns:
            Dictionary with estimation results
        """
        logger.info("Estimating treatment effect using backdoor adjustment")
        
        if covariates is None:
            covariates = [col for col in data.columns 
                         if col not in [treatment, outcome]]
        
        # Prepare data
        X = data[covariates]
        T = data[treatment]
        Y = data[outcome]
        
        # Fit outcome model
        outcome_model = LinearRegression()
        outcome_model.fit(X, Y)
        
        # Calculate potential outcomes
        Y_treated = outcome_model.predict(X)
        Y_control = outcome_model.predict(X)
        
        # Add treatment effect
        Y_treated += outcome_model.coef_[0] if len(outcome_model.coef_) > 0 else 0
        
        # Estimate ATE
        ate = np.mean(Y_treated - Y_control)
        
        # Bootstrap confidence intervals
        bootstrap_ates = []
        n_bootstrap = self.config.get('bootstrap_samples', 1000)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(data), len(data), replace=True)
            sample_data = data.iloc[indices]
            
            X_sample = sample_data[covariates]
            T_sample = sample_data[treatment]
            Y_sample = sample_data[outcome]
            
            model_sample = LinearRegression()
            model_sample.fit(X_sample, Y_sample)
            
            Y_treated_sample = model_sample.predict(X_sample)
            Y_control_sample = model_sample.predict(X_sample)
            
            if len(model_sample.coef_) > 0:
                Y_treated_sample += model_sample.coef_[0]
            
            bootstrap_ate = np.mean(Y_treated_sample - Y_control_sample)
            bootstrap_ates.append(bootstrap_ate)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)
        
        # Calculate p-value
        p_value = self._calculate_p_value(bootstrap_ates, 0)
        
        results = {
            "method": "backdoor_adjustment",
            "ate": ate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "covariates": covariates,
            "bootstrap_ates": bootstrap_ates
        }
        
        logger.info(f"Backdoor adjustment ATE: {ate:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return results
    
    def estimate_propensity_score_matching(self, data: pd.DataFrame,
                                         treatment: str = 'treatment',
                                         outcome: str = 'outcome',
                                         covariates: Optional[List[str]] = None) -> Dict:
        """
        Estimate treatment effect using propensity score matching.
        
        Args:
            data: DataFrame containing the data
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            covariates: List of covariates for propensity score
            
        Returns:
            Dictionary with estimation results
        """
        logger.info("Estimating treatment effect using propensity score matching")
        
        if covariates is None:
            covariates = [col for col in data.columns 
                         if col not in [treatment, outcome]]
        
        # Fit propensity score model
        X = data[covariates]
        T = data[treatment]
        
        propensity_model = LogisticRegression(random_state=42)
        propensity_model.fit(X, T)
        
        # Calculate propensity scores
        propensity_scores = propensity_model.predict_proba(X)[:, 1]
        
        # Perform matching
        treated_indices = data[treatment] == 1
        control_indices = data[treatment] == 0
        
        treated_scores = propensity_scores[treated_indices]
        control_scores = propensity_scores[control_indices]
        treated_outcomes = data[outcome][treated_indices]
        control_outcomes = data[outcome][control_indices]
        
        # Simple nearest neighbor matching
        matched_pairs = []
        for i, treated_score in enumerate(treated_scores):
            distances = np.abs(control_scores - treated_score)
            best_match = np.argmin(distances)
            matched_pairs.append((treated_outcomes.iloc[i], control_outcomes.iloc[best_match]))
        
        # Calculate ATE
        treatment_effects = [t - c for t, c in matched_pairs]
        ate = np.mean(treatment_effects)
        
        # Bootstrap confidence intervals
        bootstrap_ates = []
        n_bootstrap = self.config.get('bootstrap_samples', 1000)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(data), len(data), replace=True)
            sample_data = data.iloc[indices]
            
            X_sample = sample_data[covariates]
            T_sample = sample_data[treatment]
            
            model_sample = LogisticRegression(random_state=42)
            model_sample.fit(X_sample, T_sample)
            
            propensity_scores_sample = model_sample.predict_proba(X_sample)[:, 1]
            
            # Perform matching on bootstrap sample
            treated_indices_sample = sample_data[treatment] == 1
            control_indices_sample = sample_data[treatment] == 0
            
            treated_scores_sample = propensity_scores_sample[treated_indices_sample]
            control_scores_sample = propensity_scores_sample[control_indices_sample]
            treated_outcomes_sample = sample_data[outcome][treated_indices_sample]
            control_outcomes_sample = sample_data[outcome][control_indices_sample]
            
            matched_pairs_sample = []
            for i, treated_score in enumerate(treated_scores_sample):
                distances = np.abs(control_scores_sample - treated_score)
                if len(distances) > 0:
                    best_match = np.argmin(distances)
                    matched_pairs_sample.append((treated_outcomes_sample.iloc[i], 
                                               control_outcomes_sample.iloc[best_match]))
            
            if matched_pairs_sample:
                treatment_effects_sample = [t - c for t, c in matched_pairs_sample]
                bootstrap_ate = np.mean(treatment_effects_sample)
                bootstrap_ates.append(bootstrap_ate)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)
        
        # Calculate p-value
        p_value = self._calculate_p_value(bootstrap_ates, 0)
        
        results = {
            "method": "propensity_score_matching",
            "ate": ate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "covariates": covariates,
            "propensity_scores": propensity_scores,
            "bootstrap_ates": bootstrap_ates
        }
        
        logger.info(f"Propensity score matching ATE: {ate:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return results
    
    def estimate_instrumental_variables(self, data: pd.DataFrame,
                                      treatment: str = 'treatment',
                                      outcome: str = 'outcome',
                                      instrument: str = 'instrument',
                                      covariates: Optional[List[str]] = None) -> Dict:
        """
        Estimate treatment effect using instrumental variables.
        
        Args:
            data: DataFrame containing the data
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            instrument: Name of instrumental variable
            covariates: List of covariates
            
        Returns:
            Dictionary with estimation results
        """
        logger.info("Estimating treatment effect using instrumental variables")
        
        if covariates is None:
            covariates = [col for col in data.columns 
                         if col not in [treatment, outcome, instrument]]
        
        # First stage: regress treatment on instrument and covariates
        X_first = data[[instrument] + covariates]
        T = data[treatment]
        
        first_stage = LinearRegression()
        first_stage.fit(X_first, T)
        T_hat = first_stage.predict(X_first)
        
        # Second stage: regress outcome on predicted treatment and covariates
        X_second = pd.concat([pd.Series(T_hat, name='treatment_hat'), 
                             data[covariates]], axis=1)
        Y = data[outcome]
        
        second_stage = LinearRegression()
        second_stage.fit(X_second, Y)
        
        # Extract treatment effect
        ate = second_stage.coef_[0]
        
        # Bootstrap confidence intervals
        bootstrap_ates = []
        n_bootstrap = self.config.get('bootstrap_samples', 1000)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(data), len(data), replace=True)
            sample_data = data.iloc[indices]
            
            X_first_sample = sample_data[[instrument] + covariates]
            T_sample = sample_data[treatment]
            Y_sample = sample_data[outcome]
            
            first_stage_sample = LinearRegression()
            first_stage_sample.fit(X_first_sample, T_sample)
            T_hat_sample = first_stage_sample.predict(X_first_sample)
            
            X_second_sample = pd.concat([pd.Series(T_hat_sample, name='treatment_hat'), 
                                       sample_data[covariates]], axis=1)
            
            second_stage_sample = LinearRegression()
            second_stage_sample.fit(X_second_sample, Y_sample)
            
            bootstrap_ate = second_stage_sample.coef_[0]
            bootstrap_ates.append(bootstrap_ate)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)
        
        # Calculate p-value
        p_value = self._calculate_p_value(bootstrap_ates, 0)
        
        results = {
            "method": "instrumental_variables",
            "ate": ate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "instrument": instrument,
            "covariates": covariates,
            "first_stage_coef": first_stage.coef_[0],
            "bootstrap_ates": bootstrap_ates
        }
        
        logger.info(f"IV estimation ATE: {ate:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return results
    
    def estimate_regression_discontinuity(self, data: pd.DataFrame,
                                        treatment: str = 'treatment',
                                        outcome: str = 'outcome',
                                        running_var: str = 'running_variable',
                                        threshold: float = 0.0,
                                        bandwidth: Optional[float] = None) -> Dict:
        """
        Estimate treatment effect using regression discontinuity.
        
        Args:
            data: DataFrame containing the data
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            running_var: Name of running variable
            threshold: Discontinuity threshold
            bandwidth: Bandwidth around threshold
            
        Returns:
            Dictionary with estimation results
        """
        logger.info("Estimating treatment effect using regression discontinuity")
        
        # Filter data within bandwidth if specified
        if bandwidth is not None:
            data_filtered = data[
                (data[running_var] >= threshold - bandwidth) &
                (data[running_var] <= threshold + bandwidth)
            ].copy()
        else:
            data_filtered = data.copy()
        
        # Create interaction terms
        data_filtered['above_threshold'] = (data_filtered[running_var] > threshold).astype(int)
        data_filtered['running_centered'] = data_filtered[running_var] - threshold
        data_filtered['interaction'] = data_filtered['above_threshold'] * data_filtered['running_centered']
        
        # Fit regression discontinuity model
        X = data_filtered[['above_threshold', 'running_centered', 'interaction']]
        Y = data_filtered[outcome]
        
        rd_model = LinearRegression()
        rd_model.fit(X, Y)
        
        # Extract treatment effect (coefficient on above_threshold)
        ate = rd_model.coef_[0]
        
        # Bootstrap confidence intervals
        bootstrap_ates = []
        n_bootstrap = self.config.get('bootstrap_samples', 1000)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(data_filtered), len(data_filtered), replace=True)
            sample_data = data_filtered.iloc[indices]
            
            X_sample = sample_data[['above_threshold', 'running_centered', 'interaction']]
            Y_sample = sample_data[outcome]
            
            model_sample = LinearRegression()
            model_sample.fit(X_sample, Y_sample)
            
            bootstrap_ate = model_sample.coef_[0]
            bootstrap_ates.append(bootstrap_ate)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)
        
        # Calculate p-value
        p_value = self._calculate_p_value(bootstrap_ates, 0)
        
        results = {
            "method": "regression_discontinuity",
            "ate": ate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "running_variable": running_var,
            "threshold": threshold,
            "bandwidth": bandwidth,
            "n_observations": len(data_filtered),
            "bootstrap_ates": bootstrap_ates
        }
        
        logger.info(f"RD estimation ATE: {ate:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return results
    
    def estimate_all_methods(self, data: pd.DataFrame,
                           treatment: str = 'treatment',
                           outcome: str = 'outcome',
                           covariates: Optional[List[str]] = None) -> Dict:
        """
        Estimate treatment effects using all available methods.
        
        Args:
            data: DataFrame containing the data
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            covariates: List of covariates
            
        Returns:
            Dictionary with results from all methods
        """
        logger.info("Estimating treatment effects using all methods")
        
        if covariates is None:
            covariates = [col for col in data.columns 
                         if col not in [treatment, outcome]]
        
        results = {}
        
        # Backdoor adjustment
        try:
            results['backdoor_adjustment'] = self.estimate_backdoor_adjustment(
                data, treatment, outcome, covariates
            )
        except Exception as e:
            logger.error(f"Backdoor adjustment failed: {e}")
            results['backdoor_adjustment'] = {"error": str(e)}
        
        # Propensity score matching
        try:
            results['propensity_score_matching'] = self.estimate_propensity_score_matching(
                data, treatment, outcome, covariates
            )
        except Exception as e:
            logger.error(f"Propensity score matching failed: {e}")
            results['propensity_score_matching'] = {"error": str(e)}
        
        # Instrumental variables (if instrument available)
        if 'instrument' in data.columns:
            try:
                results['instrumental_variables'] = self.estimate_instrumental_variables(
                    data, treatment, outcome, 'instrument', covariates
                )
            except Exception as e:
                logger.error(f"IV estimation failed: {e}")
                results['instrumental_variables'] = {"error": str(e)}
        
        # Regression discontinuity (if running variable available)
        if 'running_variable' in data.columns:
            try:
                results['regression_discontinuity'] = self.estimate_regression_discontinuity(
                    data, treatment, outcome, 'running_variable'
                )
            except Exception as e:
                logger.error(f"RD estimation failed: {e}")
                results['regression_discontinuity'] = {"error": str(e)}
        
        self.results = results
        
        return results
    
    def compare_methods(self) -> pd.DataFrame:
        """
        Compare results across different estimation methods.
        
        Returns:
            DataFrame with comparison of results
        """
        if not self.results:
            raise ValueError("No results available. Run estimate_all_methods() first.")
        
        comparison_data = []
        
        for method, result in self.results.items():
            if "error" not in result:
                comparison_data.append({
                    "Method": method.replace("_", " ").title(),
                    "ATE": result["ate"],
                    "CI Lower": result["ci_lower"],
                    "CI Upper": result["ci_upper"],
                    "P-value": result["p_value"]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        return comparison_df
    
    def set_true_effect(self, true_effect: float) -> None:
        """
        Set the true treatment effect for comparison.
        
        Args:
            true_effect: True average treatment effect
        """
        self.true_effect = true_effect
    
    def evaluate_accuracy(self) -> Dict:
        """
        Evaluate accuracy of estimates against true effect.
        
        Returns:
            Dictionary with accuracy metrics
        """
        if self.true_effect is None:
            raise ValueError("True effect not set. Use set_true_effect() first.")
        
        if not self.results:
            raise ValueError("No results available. Run estimate_all_methods() first.")
        
        accuracy_metrics = {}
        
        for method, result in self.results.items():
            if "error" not in result:
                estimated_ate = result["ate"]
                bias = estimated_ate - self.true_effect
                relative_bias = bias / self.true_effect if self.true_effect != 0 else float('inf')
                
                accuracy_metrics[method] = {
                    "estimated_ate": estimated_ate,
                    "true_ate": self.true_effect,
                    "bias": bias,
                    "relative_bias": relative_bias,
                    "ci_contains_true": (result["ci_lower"] <= self.true_effect <= result["ci_upper"])
                }
        
        return accuracy_metrics
    
    def _calculate_p_value(self, bootstrap_estimates: List[float], 
                          null_value: float = 0.0) -> float:
        """
        Calculate p-value from bootstrap estimates.
        
        Args:
            bootstrap_estimates: List of bootstrap estimates
            null_value: Null hypothesis value
            
        Returns:
            P-value
        """
        if not bootstrap_estimates:
            return 1.0
        
        # Two-sided p-value
        mean_estimate = np.mean(bootstrap_estimates)
        std_estimate = np.std(bootstrap_estimates)
        
        if std_estimate == 0:
            return 1.0
        
        z_score = abs(mean_estimate - null_value) / std_estimate
        p_value = 2 * (1 - stats.norm.cdf(z_score))
        
        return p_value
    
    def save_results(self, filepath: str) -> None:
        """
        Save estimation results to file.
        
        Args:
            filepath: Path to save results
        """
        logger.info(f"Saving results to {filepath}")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        
        for method, result in self.results.items():
            if "error" not in result:
                serializable_result = result.copy()
                if "bootstrap_ates" in serializable_result:
                    serializable_result["bootstrap_ates"] = serializable_result["bootstrap_ates"].tolist()
                if "propensity_scores" in serializable_result:
                    serializable_result["propensity_scores"] = serializable_result["propensity_scores"].tolist()
                serializable_results[method] = serializable_result
            else:
                serializable_results[method] = result
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info("Results saved successfully")
    
    def load_results(self, filepath: str) -> None:
        """
        Load estimation results from file.
        
        Args:
            filepath: Path to load results from
        """
        logger.info(f"Loading results from {filepath}")
        
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        
        logger.info("Results loaded successfully") 