"""
Utility functions for the Streamlit application.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def create_summary_metrics(data: pd.DataFrame) -> Dict:
    """
    Create summary metrics for the dataset.
    
    Args:
        data: Input dataset
        
    Returns:
        Dictionary with summary metrics
    """
    return {
        "n_samples": len(data),
        "n_features": len(data.columns),
        "treatment_rate": data['treatment'].mean(),
        "outcome_rate": data['outcome'].mean(),
        "age_mean": data['age'].mean(),
        "bmi_mean": data['bmi'].mean(),
        "missing_values": data.isnull().sum().sum()
    }

def create_correlation_plot(data: pd.DataFrame) -> go.Figure:
    """
    Create correlation heatmap.
    
    Args:
        data: Input dataset
        
    Returns:
        Plotly figure with correlation heatmap
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    
    return fig

def create_distribution_plots(data: pd.DataFrame) -> List[go.Figure]:
    """
    Create distribution plots for key variables.
    
    Args:
        data: Input dataset
        
    Returns:
        List of plotly figures
    """
    figures = []
    
    # Age distribution
    fig_age = px.histogram(data, x='age', title="Age Distribution")
    figures.append(fig_age)
    
    # BMI distribution
    fig_bmi = px.histogram(data, x='bmi', title="BMI Distribution")
    figures.append(fig_bmi)
    
    # Treatment distribution
    treatment_counts = data['treatment'].value_counts()
    fig_treatment = px.bar(
        x=treatment_counts.index,
        y=treatment_counts.values,
        title="Treatment Distribution",
        labels={'x': 'Treatment', 'y': 'Count'}
    )
    figures.append(fig_treatment)
    
    # Outcome distribution
    outcome_counts = data['outcome'].value_counts()
    fig_outcome = px.bar(
        x=outcome_counts.index,
        y=outcome_counts.values,
        title="Outcome Distribution",
        labels={'x': 'Outcome', 'y': 'Count'}
    )
    figures.append(fig_outcome)
    
    return figures

def format_results_table(results: Dict) -> pd.DataFrame:
    """
    Format estimation results into a table.
    
    Args:
        results: Dictionary with estimation results
        
    Returns:
        Formatted DataFrame
    """
    table_data = []
    
    for method, result in results.items():
        if "error" not in result:
            table_data.append({
                "Method": method.replace("_", " ").title(),
                "ATE": f"{result['ate']:.4f}",
                "CI Lower": f"{result['ci_lower']:.4f}",
                "CI Upper": f"{result['ci_upper']:.4f}",
                "P-value": f"{result['p_value']:.4f}"
            })
    
    return pd.DataFrame(table_data)

def create_forest_plot(results: Dict) -> go.Figure:
    """
    Create forest plot of treatment effects.
    
    Args:
        results: Dictionary with estimation results
        
    Returns:
        Plotly figure with forest plot
    """
    methods = []
    ates = []
    ci_lowers = []
    ci_uppers = []
    
    for method, result in results.items():
        if "error" not in result:
            methods.append(method.replace("_", " ").title())
            ates.append(result["ate"])
            ci_lowers.append(result["ci_lower"])
            ci_uppers.append(result["ci_upper"])
    
    fig = go.Figure()
    
    # Add point estimates
    fig.add_trace(go.Scatter(
        x=ates,
        y=methods,
        mode='markers',
        name='ATE Estimates',
        marker=dict(size=10, color='blue')
    ))
    
    # Add confidence intervals
    for i, (ate, ci_lower, ci_upper) in enumerate(zip(ates, ci_lowers, ci_uppers)):
        fig.add_trace(go.Scatter(
            x=[ci_lower, ci_upper],
            y=[methods[i], methods[i]],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False
        ))
    
    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color="red", 
                 annotation_text="No Effect")
    
    fig.update_layout(
        title="Treatment Effect Estimates with 95% CI",
        xaxis_title="Average Treatment Effect (ATE)",
        yaxis_title="Estimation Method",
        height=400
    )
    
    return fig

def validate_data(data: pd.DataFrame) -> Dict:
    """
    Validate dataset for causal inference analysis.
    
    Args:
        data: Input dataset
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "has_required_columns": True,
        "missing_values": False,
        "treatment_binary": True,
        "outcome_binary": True,
        "sufficient_sample_size": True,
        "issues": []
    }
    
    # Check required columns
    required_columns = ['treatment', 'outcome']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        validation_results["has_required_columns"] = False
        validation_results["issues"].append(f"Missing required columns: {missing_columns}")
    
    # Check for missing values
    if data.isnull().sum().sum() > 0:
        validation_results["missing_values"] = True
        validation_results["issues"].append("Dataset contains missing values")
    
    # Check treatment variable
    if 'treatment' in data.columns:
        unique_treatment = set(data['treatment'].unique())
        if not unique_treatment.issubset({0, 1}):
            validation_results["treatment_binary"] = False
            validation_results["issues"].append("Treatment variable is not binary")
    
    # Check outcome variable
    if 'outcome' in data.columns:
        unique_outcome = set(data['outcome'].unique())
        if not unique_outcome.issubset({0, 1}):
            validation_results["outcome_binary"] = False
            validation_results["issues"].append("Outcome variable is not binary")
    
    # Check sample size
    if len(data) < 100:
        validation_results["sufficient_sample_size"] = False
        validation_results["issues"].append("Sample size is too small for reliable analysis")
    
    return validation_results 