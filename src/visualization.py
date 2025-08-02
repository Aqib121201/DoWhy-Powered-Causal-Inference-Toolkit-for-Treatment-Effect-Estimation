"""
Visualization components for causal inference results.

This module provides comprehensive visualization tools for causal inference,
including causal graphs, treatment effect distributions, sensitivity analysis,
and interactive plots using matplotlib, seaborn, and plotly.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import logging
from .config import VISUALIZATION_CONFIG

logger = logging.getLogger(__name__)

class CausalVisualizer:
    """
    Comprehensive visualization toolkit for causal inference results.
    
    This class provides methods to create various visualizations including
    causal graphs, treatment effect plots, sensitivity analysis, and
    interactive dashboards.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the causal visualizer.
        
        Args:
            config: Configuration dictionary for visualization settings
        """
        self.config = config or VISUALIZATION_CONFIG
        self.setup_style()
        
    def setup_style(self) -> None:
        """Set up matplotlib and seaborn styling."""
        plt.style.use(self.config["style"])
        sns.set_palette(self.config["color_palette"])
        plt.rcParams['font.size'] = self.config["font_size"]
        
    def plot_causal_graph(self, graph: nx.DiGraph, 
                         filepath: Optional[str] = None,
                         interactive: bool = False) -> Any:
        """
        Plot causal graph with node highlighting and path analysis.
        
        Args:
            graph: NetworkX directed graph
            filepath: Path to save the plot
            interactive: Whether to create interactive plot
            
        Returns:
            Plot object (matplotlib figure or plotly figure)
        """
        logger.info("Creating causal graph visualization")
        
        if interactive:
            return self._plot_causal_graph_interactive(graph, filepath)
        else:
            return self._plot_causal_graph_static(graph, filepath)
    
    def _plot_causal_graph_static(self, graph: nx.DiGraph, 
                                 filepath: Optional[str] = None) -> plt.Figure:
        """Create static causal graph plot."""
        fig, ax = plt.subplots(figsize=self.config["figure_size"])
        
        # Layout
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Node colors based on type
        node_colors = []
        for node in graph.nodes():
            if node == 'treatment':
                node_colors.append('lightcoral')
            elif node == 'outcome':
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightblue')
        
        # Draw graph
        nx.draw_networkx_nodes(graph, pos, 
                             node_color=node_colors,
                             node_size=2000,
                             alpha=0.8)
        
        nx.draw_networkx_edges(graph, pos,
                             edge_color='gray',
                             arrows=True,
                             arrowsize=20,
                             arrowstyle='->',
                             width=2)
        
        nx.draw_networkx_labels(graph, pos,
                              font_size=10,
                              font_weight='bold')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='lightcoral', markersize=10, label='Treatment'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='lightgreen', markersize=10, label='Outcome'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='lightblue', markersize=10, label='Covariates')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.title("Causal Graph", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=self.config["dpi"], 
                       bbox_inches='tight', format=self.config["save_format"])
            logger.info(f"Causal graph saved to {filepath}")
        
        return fig
    
    def _plot_causal_graph_interactive(self, graph: nx.DiGraph,
                                     filepath: Optional[str] = None) -> go.Figure:
        """Create interactive causal graph plot."""
        # Convert graph to plotly format
        edge_x = []
        edge_y = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            if node == 'treatment':
                node_colors.append('lightcoral')
            elif node == 'outcome':
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightblue')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=30,
                color=node_colors,
                line=dict(width=2, color='black')
            ))
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Interactive Causal Graph',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        if filepath:
            fig.write_html(filepath)
            logger.info(f"Interactive causal graph saved to {filepath}")
        
        return fig
    
    def plot_treatment_effects(self, results: Dict,
                             filepath: Optional[str] = None,
                             interactive: bool = False) -> Any:
        """
        Plot treatment effect estimates with confidence intervals.
        
        Args:
            results: Dictionary with estimation results
            filepath: Path to save the plot
            interactive: Whether to create interactive plot
            
        Returns:
            Plot object
        """
        logger.info("Creating treatment effects visualization")
        
        if interactive:
            return self._plot_treatment_effects_interactive(results, filepath)
        else:
            return self._plot_treatment_effects_static(results, filepath)
    
    def _plot_treatment_effects_static(self, results: Dict,
                                     filepath: Optional[str] = None) -> plt.Figure:
        """Create static treatment effects plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data for plotting
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
        
        # Forest plot
        y_pos = np.arange(len(methods))
        ax1.errorbar(ates, y_pos, xerr=[np.array(ates) - np.array(ci_lowers),
                                       np.array(ci_uppers) - np.array(ates)],
                    fmt='o', capsize=5, capthick=2, markersize=8)
        
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Effect')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(methods)
        ax1.set_xlabel('Average Treatment Effect (ATE)')
        ax1.set_title('Treatment Effect Estimates with 95% CI')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bootstrap distributions
        for i, method in enumerate(methods):
            if method.lower().replace(" ", "_") in results:
                result = results[method.lower().replace(" ", "_")]
                if "bootstrap_ates" in result:
                    ax2.hist(result["bootstrap_ates"], alpha=0.6, 
                            label=method, bins=30)
        
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Effect')
        ax2.set_xlabel('Average Treatment Effect (ATE)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Bootstrap Distributions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=self.config["dpi"], 
                       bbox_inches='tight', format=self.config["save_format"])
            logger.info(f"Treatment effects plot saved to {filepath}")
        
        return fig
    
    def _plot_treatment_effects_interactive(self, results: Dict,
                                          filepath: Optional[str] = None) -> go.Figure:
        """Create interactive treatment effects plot."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Treatment Effect Estimates', 'Bootstrap Distributions'),
            specs=[[{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Forest plot
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
        
        fig.add_trace(
            go.Bar(
                x=ates,
                y=methods,
                orientation='h',
                error_x=dict(
                    type='data',
                    array=np.array(ci_uppers) - np.array(ates),
                    arrayminus=np.array(ates) - np.array(ci_lowers),
                    visible=True
                ),
                name='ATE Estimates'
            ),
            row=1, col=1
        )
        
        # Bootstrap distributions
        for method, result in results.items():
            if "error" not in result and "bootstrap_ates" in result:
                fig.add_trace(
                    go.Histogram(
                        x=result["bootstrap_ates"],
                        name=method.replace("_", " ").title(),
                        opacity=0.7
                    ),
                    row=1, col=2
                )
        
        fig.update_layout(
            title_text="Treatment Effect Analysis",
            showlegend=True,
            height=500
        )
        
        if filepath:
            fig.write_html(filepath)
            logger.info(f"Interactive treatment effects plot saved to {filepath}")
        
        return fig
    
    def plot_sensitivity_analysis(self, results: Dict,
                                filepath: Optional[str] = None) -> plt.Figure:
        """
        Plot sensitivity analysis results.
        
        Args:
            results: Dictionary with sensitivity analysis results
            filepath: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating sensitivity analysis visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # Plot 1: Propensity score distribution
        if 'propensity_score_matching' in results:
            result = results['propensity_score_matching']
            if "propensity_scores" in result:
                ax = axes[0]
                ax.hist(result["propensity_scores"], bins=30, alpha=0.7, color='skyblue')
                ax.set_xlabel('Propensity Score')
                ax.set_ylabel('Frequency')
                ax.set_title('Propensity Score Distribution')
                ax.grid(True, alpha=0.3)
        
        # Plot 2: Balance plot
        if 'propensity_score_matching' in results:
            result = results['propensity_score_matching']
            if "covariates" in result:
                ax = axes[1]
                # This would need actual balance statistics
                ax.text(0.5, 0.5, 'Balance Plot\n(Requires balance statistics)',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Covariate Balance')
        
        # Plot 3: Bootstrap distribution comparison
        ax = axes[2]
        for method, result in results.items():
            if "error" not in result and "bootstrap_ates" in result:
                ax.hist(result["bootstrap_ates"], alpha=0.6, 
                       label=method.replace("_", " ").title(), bins=30)
        ax.set_xlabel('Average Treatment Effect')
        ax.set_ylabel('Frequency')
        ax.set_title('Bootstrap Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Method comparison
        ax = axes[3]
        methods = []
        ates = []
        for method, result in results.items():
            if "error" not in result:
                methods.append(method.replace("_", " ").title())
                ates.append(result["ate"])
        
        bars = ax.bar(methods, ates, alpha=0.7)
        ax.set_xlabel('Estimation Method')
        ax.set_ylabel('Average Treatment Effect')
        ax.set_title('Method Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=self.config["dpi"], 
                       bbox_inches='tight', format=self.config["save_format"])
            logger.info(f"Sensitivity analysis plot saved to {filepath}")
        
        return fig
    
    def plot_data_summary(self, data: pd.DataFrame,
                         filepath: Optional[str] = None) -> plt.Figure:
        """
        Plot data summary statistics and distributions.
        
        Args:
            data: DataFrame with the data
            filepath: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating data summary visualization")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # Plot 1: Treatment distribution
        ax = axes[0]
        data['treatment'].value_counts().plot(kind='bar', ax=ax, color=['lightblue', 'lightcoral'])
        ax.set_title('Treatment Distribution')
        ax.set_xlabel('Treatment')
        ax.set_ylabel('Count')
        
        # Plot 2: Outcome distribution
        ax = axes[1]
        data['outcome'].value_counts().plot(kind='bar', ax=ax, color=['lightgreen', 'lightcoral'])
        ax.set_title('Outcome Distribution')
        ax.set_xlabel('Outcome')
        ax.set_ylabel('Count')
        
        # Plot 3: Age distribution
        ax = axes[2]
        ax.hist(data['age'], bins=30, alpha=0.7, color='skyblue')
        ax.set_title('Age Distribution')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        
        # Plot 4: BMI distribution
        ax = axes[3]
        ax.hist(data['bmi'], bins=30, alpha=0.7, color='lightgreen')
        ax.set_title('BMI Distribution')
        ax.set_xlabel('BMI')
        ax.set_ylabel('Frequency')
        
        # Plot 5: Treatment by age group
        ax = axes[4]
        if 'age_group' in data.columns:
            treatment_by_age = data.groupby('age_group')['treatment'].mean()
            treatment_by_age.plot(kind='bar', ax=ax, color='lightblue')
            ax.set_title('Treatment Rate by Age Group')
            ax.set_xlabel('Age Group')
            ax.set_ylabel('Treatment Rate')
            ax.tick_params(axis='x', rotation=45)
        
        # Plot 6: Outcome by treatment
        ax = axes[5]
        outcome_by_treatment = data.groupby('treatment')['outcome'].mean()
        outcome_by_treatment.plot(kind='bar', ax=ax, color=['lightblue', 'lightcoral'])
        ax.set_title('Outcome Rate by Treatment')
        ax.set_xlabel('Treatment')
        ax.set_ylabel('Outcome Rate')
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=self.config["dpi"], 
                       bbox_inches='tight', format=self.config["save_format"])
            logger.info(f"Data summary plot saved to {filepath}")
        
        return fig
    
    def create_interactive_dashboard(self, data: pd.DataFrame,
                                   results: Dict,
                                   filepath: Optional[str] = None) -> go.Figure:
        """
        Create an interactive dashboard with multiple visualizations.
        
        Args:
            data: DataFrame with the data
            results: Dictionary with estimation results
            filepath: Path to save the dashboard
            
        Returns:
            Plotly figure with dashboard
        """
        logger.info("Creating interactive dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Treatment Distribution', 'Outcome Distribution', 'Age Distribution',
                'Treatment Effects', 'Bootstrap Distributions', 'Data Summary'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "histogram"}, {"type": "scatter"}]
            ]
        )
        
        # Treatment distribution
        treatment_counts = data['treatment'].value_counts()
        fig.add_trace(
            go.Bar(x=treatment_counts.index, y=treatment_counts.values,
                  name='Treatment', marker_color=['lightblue', 'lightcoral']),
            row=1, col=1
        )
        
        # Outcome distribution
        outcome_counts = data['outcome'].value_counts()
        fig.add_trace(
            go.Bar(x=outcome_counts.index, y=outcome_counts.values,
                  name='Outcome', marker_color=['lightgreen', 'lightcoral']),
            row=1, col=2
        )
        
        # Age distribution
        fig.add_trace(
            go.Histogram(x=data['age'], name='Age', marker_color='skyblue'),
            row=1, col=3
        )
        
        # Treatment effects
        methods = []
        ates = []
        for method, result in results.items():
            if "error" not in result:
                methods.append(method.replace("_", " ").title())
                ates.append(result["ate"])
        
        fig.add_trace(
            go.Bar(x=methods, y=ates, name='ATE', marker_color='lightblue'),
            row=2, col=1
        )
        
        # Bootstrap distributions
        for method, result in results.items():
            if "error" not in result and "bootstrap_ates" in result:
                fig.add_trace(
                    go.Histogram(x=result["bootstrap_ates"],
                               name=method.replace("_", " ").title(),
                               opacity=0.7),
                    row=2, col=2
                )
        
        # Data summary scatter
        fig.add_trace(
            go.Scatter(x=data['age'], y=data['bmi'], mode='markers',
                      marker=dict(color=data['treatment'], colorscale='Viridis'),
                      name='Age vs BMI'),
            row=2, col=3
        )
        
        fig.update_layout(
            title_text="Causal Inference Dashboard",
            showlegend=True,
            height=800
        )
        
        if filepath:
            fig.write_html(filepath)
            logger.info(f"Interactive dashboard saved to {filepath}")
        
        return fig
    
    def save_all_plots(self, data: pd.DataFrame, results: Dict,
                      base_path: str) -> None:
        """
        Save all visualization plots to files.
        
        Args:
            data: DataFrame with the data
            results: Dictionary with estimation results
            base_path: Base path for saving plots
        """
        logger.info("Saving all visualization plots")
        
        # Causal graph (if available)
        try:
            from .causal_graphs import CausalGraphBuilder
            graph_builder = CausalGraphBuilder()
            graph = graph_builder.build_graph_from_config()
            self.plot_causal_graph(graph, f"{base_path}_causal_graph.png")
        except Exception as e:
            logger.warning(f"Could not create causal graph: {e}")
        
        # Treatment effects
        self.plot_treatment_effects(results, f"{base_path}_treatment_effects.png")
        
        # Sensitivity analysis
        self.plot_sensitivity_analysis(results, f"{base_path}_sensitivity_analysis.png")
        
        # Data summary
        self.plot_data_summary(data, f"{base_path}_data_summary.png")
        
        # Interactive dashboard
        self.create_interactive_dashboard(data, results, f"{base_path}_dashboard.html")
        
        logger.info("All plots saved successfully") 