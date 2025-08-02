"""
DoWhy-Powered Visual Toolkit for Causal Inference & Treatment Effect Estimation

A comprehensive toolkit for causal inference using the DoWhy framework,
providing interactive visualization of causal graphs, backdoor paths,
do-calculus logic, and average treatment effect (ATE) estimation.
"""

__version__ = "1.0.0"
__author__ = "Causal Inference Research Team"
__email__ = "contact@causal-toolkit.org"

from .causal_graphs import CausalGraphBuilder
from .treatment_effects import TreatmentEffectEstimator
from .visualization import CausalVisualizer
from .data_generator import HealthcareDataGenerator

__all__ = [
    "CausalGraphBuilder",
    "TreatmentEffectEstimator", 
    "CausalVisualizer",
    "HealthcareDataGenerator"
] 