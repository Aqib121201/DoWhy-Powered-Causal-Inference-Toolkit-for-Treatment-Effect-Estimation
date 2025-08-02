"""
Configuration file for the Causal Inference Toolkit.

This module contains all configuration parameters, paths, and hyperparameters
used throughout the project for reproducibility and easy modification.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Output directories
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
REPORTS_DIR = PROJECT_ROOT / "report"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, 
                  MODELS_DIR, VISUALIZATIONS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data generation parameters
DATA_CONFIG = {
    "n_samples": 10000,
    "random_state": 42,
    "treatment_effect": 0.15,
    "noise_level": 0.1,
    "confounding_strength": 0.3
}

# Causal graph configuration
CAUSAL_GRAPH_CONFIG = {
    "nodes": [
        "age", "gender", "bmi", "smoker", "diabetes", 
        "hypertension", "treatment", "outcome"
    ],
    "edges": [
        ("age", "treatment"),
        ("age", "outcome"),
        ("gender", "treatment"),
        ("gender", "outcome"),
        ("bmi", "treatment"),
        ("bmi", "outcome"),
        ("smoker", "treatment"),
        ("smoker", "outcome"),
        ("diabetes", "treatment"),
        ("diabetes", "outcome"),
        ("hypertension", "treatment"),
        ("hypertension", "outcome"),
        ("treatment", "outcome")
    ]
}

# Treatment effect estimation methods
ESTIMATION_METHODS = [
    "backdoor.linear_regression",
    "backdoor.propensity_score_matching",
    "iv.instrumental_variable",
    "regression_discontinuity"
]

# Visualization settings
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "Set2",
    "font_size": 12,
    "save_format": "png"
}

# Model parameters
MODEL_CONFIG = {
    "propensity_score": {
        "method": "logistic",
        "cv_folds": 5
    },
    "outcome_model": {
        "method": "linear",
        "cv_folds": 5
    },
    "instrumental_variable": {
        "method": "2sls",
        "cv_folds": 5
    }
}

# Statistical analysis parameters
STATS_CONFIG = {
    "confidence_level": 0.95,
    "bootstrap_samples": 1000,
    "random_state": 42,
    "test_size": 0.2
}

# File paths
FILE_PATHS = {
    "raw_data": RAW_DATA_DIR / "healthcare_data.csv",
    "processed_data": PROCESSED_DATA_DIR / "healthcare_data_processed.csv",
    "causal_model": MODELS_DIR / "causal_model.pkl",
    "results": MODELS_DIR / "treatment_effects.json",
    "causal_graph": VISUALIZATIONS_DIR / "causal_graph.png",
    "treatment_effects": VISUALIZATIONS_DIR / "treatment_effects.png",
    "sensitivity_analysis": VISUALIZATIONS_DIR / "sensitivity_analysis.png"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "causal_inference.log"
}

# Streamlit app configuration
STREAMLIT_CONFIG = {
    "page_title": "Causal Inference Toolkit",
    "page_icon": "🧠",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Test configuration
TEST_CONFIG = {
    "test_data_size": 1000,
    "random_state": 42,
    "tolerance": 1e-6
}

def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary.
    
    Returns:
        Dict containing all configuration parameters
    """
    return {
        "data_config": DATA_CONFIG,
        "causal_graph_config": CAUSAL_GRAPH_CONFIG,
        "estimation_methods": ESTIMATION_METHODS,
        "visualization_config": VISUALIZATION_CONFIG,
        "model_config": MODEL_CONFIG,
        "stats_config": STATS_CONFIG,
        "file_paths": FILE_PATHS,
        "logging_config": LOGGING_CONFIG,
        "streamlit_config": STREAMLIT_CONFIG,
        "test_config": TEST_CONFIG
    } 