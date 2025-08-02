#!/usr/bin/env python3
"""
Main pipeline orchestrator for the Causal Inference Toolkit.

This script runs the complete causal inference analysis pipeline including
data generation, causal graph construction, treatment effect estimation,
and visualization generation.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_generator import HealthcareDataGenerator
from src.causal_graphs import CausalGraphBuilder
from src.treatment_effects import TreatmentEffectEstimator
from src.visualization import CausalVisualizer
from src.config import get_config, FILE_PATHS

def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('causal_inference.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def generate_data(n_samples: int = 10000, save_data: bool = True) -> pd.DataFrame:
    """
    Generate synthetic healthcare data.
    
    Args:
        n_samples: Number of samples to generate
        save_data: Whether to save data to file
        
    Returns:
        Generated dataset
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating {n_samples} samples of healthcare data")
    
    # Initialize data generator
    data_generator = HealthcareDataGenerator()
    
    # Generate dataset
    data = data_generator.generate_dataset(n_samples)
    
    # Get summary statistics
    summary = data_generator.get_data_summary(data)
    logger.info(f"Data summary: {summary}")
    
    # Save data if requested
    if save_data:
        data.to_csv(FILE_PATHS["raw_data"], index=False)
        logger.info(f"Data saved to {FILE_PATHS['raw_data']}")
    
    return data

def build_causal_graph() -> CausalGraphBuilder:
    """
    Build and validate causal graph.
    
    Returns:
        CausalGraphBuilder instance
    """
    logger = logging.getLogger(__name__)
    logger.info("Building causal graph")
    
    # Initialize graph builder
    graph_builder = CausalGraphBuilder()
    
    # Build graph from configuration
    graph = graph_builder.build_graph_from_config()
    
    # Validate graph
    validation_results = graph_builder.validate_graph()
    logger.info(f"Graph validation results: {validation_results}")
    
    # Get graph statistics
    stats = graph_builder.get_graph_statistics()
    logger.info(f"Graph statistics: {stats}")
    
    # Export graph
    graph_builder.export_graph(FILE_PATHS["causal_graph"])
    
    return graph_builder

def estimate_treatment_effects(data: pd.DataFrame, 
                             true_effect: float = None) -> TreatmentEffectEstimator:
    """
    Estimate treatment effects using multiple methods.
    
    Args:
        data: Input dataset
        true_effect: True treatment effect for comparison
        
    Returns:
        TreatmentEffectEstimator instance with results
    """
    logger = logging.getLogger(__name__)
    logger.info("Estimating treatment effects")
    
    # Initialize estimator
    estimator = TreatmentEffectEstimator()
    
    # Set true effect if provided
    if true_effect is not None:
        estimator.set_true_effect(true_effect)
        logger.info(f"True treatment effect set to: {true_effect}")
    
    # Estimate effects using all methods
    results = estimator.estimate_all_methods(data)
    
    # Compare methods
    comparison = estimator.compare_methods()
    logger.info(f"Method comparison:\n{comparison}")
    
    # Evaluate accuracy if true effect is known
    if true_effect is not None:
        accuracy = estimator.evaluate_accuracy()
        logger.info(f"Accuracy evaluation: {accuracy}")
    
    # Save results
    estimator.save_results(FILE_PATHS["results"])
    
    return estimator

def create_visualizations(data: pd.DataFrame, 
                         results: dict,
                         interactive: bool = False) -> CausalVisualizer:
    """
    Create comprehensive visualizations.
    
    Args:
        data: Input dataset
        results: Treatment effect estimation results
        interactive: Whether to create interactive plots
        
    Returns:
        CausalVisualizer instance
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating visualizations")
    
    # Initialize visualizer
    visualizer = CausalVisualizer()
    
    # Create and save all plots
    base_path = str(FILE_PATHS["visualizations_dir"] / "causal_analysis")
    visualizer.save_all_plots(data, results, base_path)
    
    # Create interactive dashboard if requested
    if interactive:
        dashboard = visualizer.create_interactive_dashboard(data, results)
        logger.info("Interactive dashboard created")
    
    return visualizer

def run_complete_pipeline(n_samples: int = 10000,
                         true_effect: float = None,
                         interactive: bool = False,
                         save_data: bool = True) -> dict:
    """
    Run the complete causal inference pipeline.
    
    Args:
        n_samples: Number of samples to generate
        true_effect: True treatment effect for comparison
        interactive: Whether to create interactive plots
        save_data: Whether to save generated data
        
    Returns:
        Dictionary with pipeline results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting complete causal inference pipeline")
    
    try:
        # Step 1: Generate data
        logger.info("Step 1: Data Generation")
        data = generate_data(n_samples, save_data)
        
        # Step 2: Build causal graph
        logger.info("Step 2: Causal Graph Construction")
        graph_builder = build_causal_graph()
        
        # Step 3: Estimate treatment effects
        logger.info("Step 3: Treatment Effect Estimation")
        estimator = estimate_treatment_effects(data, true_effect)
        
        # Step 4: Create visualizations
        logger.info("Step 4: Visualization Generation")
        visualizer = create_visualizations(data, estimator.results, interactive)
        
        # Step 5: Generate summary report
        logger.info("Step 5: Summary Report Generation")
        summary = generate_summary_report(data, estimator, graph_builder)
        
        logger.info("Pipeline completed successfully!")
        
        return {
            "data": data,
            "graph_builder": graph_builder,
            "estimator": estimator,
            "visualizer": visualizer,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

def generate_summary_report(data: pd.DataFrame,
                          estimator: TreatmentEffectEstimator,
                          graph_builder: CausalGraphBuilder) -> dict:
    """
    Generate a summary report of the analysis.
    
    Args:
        data: Input dataset
        estimator: Treatment effect estimator with results
        graph_builder: Causal graph builder
        
    Returns:
        Dictionary with summary statistics
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating summary report")
    
    # Data summary
    data_summary = {
        "n_samples": len(data),
        "n_features": len(data.columns),
        "treatment_rate": data['treatment'].mean(),
        "outcome_rate": data['outcome'].mean(),
        "missing_values": data.isnull().sum().sum()
    }
    
    # Graph summary
    graph_stats = graph_builder.get_graph_statistics()
    graph_summary = {
        "n_nodes": graph_stats["n_nodes"],
        "n_edges": graph_stats["n_edges"],
        "density": graph_stats["density"],
        "is_connected": graph_stats["is_connected"]
    }
    
    # Estimation summary
    estimation_summary = {}
    if estimator.results:
        comparison = estimator.compare_methods()
        estimation_summary = {
            "n_methods": len(comparison),
            "methods_used": list(estimator.results.keys()),
            "ate_range": (comparison['ATE'].min(), comparison['ATE'].max()),
            "mean_ate": comparison['ATE'].mean()
        }
    
    # Accuracy summary
    accuracy_summary = {}
    if estimator.true_effect is not None:
        accuracy = estimator.evaluate_accuracy()
        accuracy_summary = {
            "methods_with_ci_containing_true": sum(
                1 for method_acc in accuracy.values() 
                if method_acc["ci_contains_true"]
            ),
            "mean_bias": np.mean([
                abs(method_acc["bias"]) for method_acc in accuracy.values()
            ])
        }
    
    summary = {
        "data_summary": data_summary,
        "graph_summary": graph_summary,
        "estimation_summary": estimation_summary,
        "accuracy_summary": accuracy_summary,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    logger.info(f"Summary report generated: {summary}")
    
    return summary

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Causal Inference Toolkit Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --samples 5000
  python run_pipeline.py --true-effect 0.15 --interactive
  python run_pipeline.py --samples 10000 --save-data --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=10000,
        help="Number of samples to generate (default: 10000)"
    )
    
    parser.add_argument(
        "--true-effect", "-t",
        type=float,
        default=None,
        help="True treatment effect for comparison"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Create interactive plots"
    )
    
    parser.add_argument(
        "--save-data", "-s",
        action="store_true",
        help="Save generated data to file"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Update output directory if specified
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Update file paths in config
        global FILE_PATHS
        for key, path in FILE_PATHS.items():
            if "visualizations" in str(path) or "models" in str(path):
                FILE_PATHS[key] = output_path / path.name
    
    logger.info("Causal Inference Toolkit Pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Run pipeline
        results = run_complete_pipeline(
            n_samples=args.samples,
            true_effect=args.true_effect,
            interactive=args.interactive,
            save_data=args.save_data
        )
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {FILE_PATHS['visualizations_dir']}")
        
        # Print summary
        summary = results["summary"]
        print("\n" + "="*50)
        print("PIPELINE SUMMARY")
        print("="*50)
        print(f"Data: {summary['data_summary']['n_samples']} samples")
        print(f"Graph: {summary['graph_summary']['n_nodes']} nodes, {summary['graph_summary']['n_edges']} edges")
        print(f"Methods: {summary['estimation_summary']['n_methods']} estimation methods")
        if args.true_effect:
            print(f"Accuracy: {summary['accuracy_summary']['methods_with_ci_containing_true']} methods with CI containing true effect")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 