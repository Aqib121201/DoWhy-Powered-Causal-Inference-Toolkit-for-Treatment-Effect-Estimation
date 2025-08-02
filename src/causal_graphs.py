"""
Causal graph construction and manipulation utilities.

This module provides tools for building, validating, and analyzing causal graphs
using the DoWhy framework, including backdoor path identification and
do-calculus operations.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
from dowhy import CausalModel
from .config import CAUSAL_GRAPH_CONFIG

logger = logging.getLogger(__name__)

class CausalGraphBuilder:
    """
    Builder class for constructing and analyzing causal graphs.
    
    This class provides methods to create causal graphs from specifications,
    validate their structure, and identify important paths and relationships.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the causal graph builder.
        
        Args:
            config: Configuration dictionary for graph construction
        """
        self.config = config or CAUSAL_GRAPH_CONFIG
        self.graph = None
        self.dowhy_model = None
        
    def build_graph_from_config(self) -> nx.DiGraph:
        """
        Build causal graph from configuration.
        
        Returns:
            NetworkX directed graph representing the causal structure
        """
        logger.info("Building causal graph from configuration")
        
        # Create directed graph
        self.graph = nx.DiGraph()
        
        # Add nodes
        for node in self.config["nodes"]:
            self.graph.add_node(node)
        
        # Add edges
        for source, target in self.config["edges"]:
            self.graph.add_edge(source, target)
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def build_graph_from_dot(self, dot_string: str) -> nx.DiGraph:
        """
        Build causal graph from DOT string specification.
        
        Args:
            dot_string: DOT format string describing the graph
            
        Returns:
            NetworkX directed graph
        """
        logger.info("Building causal graph from DOT specification")
        
        # Parse DOT string and create graph
        # This is a simplified implementation - in practice, you might use pydot
        lines = dot_string.strip().split('\n')
        
        self.graph = nx.DiGraph()
        
        for line in lines:
            line = line.strip()
            if '->' in line and not line.startswith('digraph'):
                # Parse edge: "source -> target;"
                parts = line.replace(';', '').split('->')
                source = parts[0].strip()
                target = parts[1].strip()
                self.graph.add_edge(source, target)
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def create_dowhy_model(self, data: pd.DataFrame, 
                          treatment: str = 'treatment',
                          outcome: str = 'outcome') -> CausalModel:
        """
        Create DoWhy causal model from graph and data.
        
        Args:
            data: DataFrame containing the variables
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            
        Returns:
            DoWhy CausalModel object
        """
        logger.info("Creating DoWhy causal model")
        
        if self.graph is None:
            self.build_graph_from_config()
        
        # Convert graph to DOT format for DoWhy
        dot_string = self.graph_to_dot()
        
        # Create DoWhy model
        self.dowhy_model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            graph=dot_string
        )
        
        logger.info("DoWhy causal model created successfully")
        
        return self.dowhy_model
    
    def graph_to_dot(self) -> str:
        """
        Convert NetworkX graph to DOT format string.
        
        Returns:
            DOT format string representation of the graph
        """
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_graph_from_config() first.")
        
        dot_lines = ["digraph {"]
        
        # Add edges
        for source, target in self.graph.edges():
            dot_lines.append(f"    {source} -> {target};")
        
        dot_lines.append("}")
        
        return "\n".join(dot_lines)
    
    def identify_backdoor_paths(self, treatment: str = 'treatment', 
                              outcome: str = 'outcome') -> List[List[str]]:
        """
        Identify all backdoor paths between treatment and outcome.
        
        Args:
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            
        Returns:
            List of backdoor paths (each path is a list of nodes)
        """
        logger.info(f"Identifying backdoor paths between {treatment} and {outcome}")
        
        if self.graph is None:
            self.build_graph_from_config()
        
        backdoor_paths = []
        
        # Find all paths from treatment to outcome
        all_paths = list(nx.all_simple_paths(self.graph, treatment, outcome))
        
        for path in all_paths:
            # Check if this is a backdoor path (doesn't go through treatment->outcome edge)
            if len(path) > 2:  # Path has intermediate nodes
                backdoor_paths.append(path)
        
        logger.info(f"Found {len(backdoor_paths)} backdoor paths")
        
        return backdoor_paths
    
    def find_backdoor_adjustment_sets(self, treatment: str = 'treatment',
                                    outcome: str = 'outcome') -> List[Set[str]]:
        """
        Find valid backdoor adjustment sets.
        
        Args:
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            
        Returns:
            List of valid adjustment sets
        """
        logger.info("Finding backdoor adjustment sets")
        
        if self.graph is None:
            self.build_graph_from_config()
        
        # Get all backdoor paths
        backdoor_paths = self.identify_backdoor_paths(treatment, outcome)
        
        # Find nodes that block all backdoor paths
        all_nodes = set(self.graph.nodes())
        treatment_outcome_nodes = {treatment, outcome}
        
        # Simple heuristic: use all non-treatment, non-outcome nodes
        adjustment_set = all_nodes - treatment_outcome_nodes
        
        return [adjustment_set]
    
    def validate_graph(self) -> Dict[str, bool]:
        """
        Validate the causal graph structure.
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating causal graph")
        
        if self.graph is None:
            self.build_graph_from_config()
        
        validation_results = {
            "is_directed": self.graph.is_directed(),
            "is_acyclic": nx.is_directed_acyclic_graph(self.graph),
            "has_treatment": 'treatment' in self.graph.nodes(),
            "has_outcome": 'outcome' in self.graph.nodes(),
            "treatment_has_outcome_edge": self.graph.has_edge('treatment', 'outcome')
        }
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))
            validation_results["has_cycles"] = len(cycles) > 0
            validation_results["cycle_count"] = len(cycles)
        except:
            validation_results["has_cycles"] = False
            validation_results["cycle_count"] = 0
        
        logger.info(f"Graph validation results: {validation_results}")
        
        return validation_results
    
    def get_graph_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the causal graph.
        
        Returns:
            Dictionary with graph statistics
        """
        if self.graph is None:
            self.build_graph_from_config()
        
        stats = {
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph),
            "node_degrees": dict(self.graph.degree()),
            "in_degrees": dict(self.graph.in_degree()),
            "out_degrees": dict(self.graph.out_degree())
        }
        
        # Centrality measures
        stats["betweenness_centrality"] = nx.betweenness_centrality(self.graph)
        stats["closeness_centrality"] = nx.closeness_centrality(self.graph)
        stats["eigenvector_centrality"] = nx.eigenvector_centrality_numpy(self.graph)
        
        return stats
    
    def identify_confounders(self, treatment: str = 'treatment',
                           outcome: str = 'outcome') -> Set[str]:
        """
        Identify confounding variables.
        
        Args:
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            
        Returns:
            Set of confounding variable names
        """
        logger.info("Identifying confounding variables")
        
        if self.graph is None:
            self.build_graph_from_config()
        
        confounders = set()
        
        # Find nodes that have edges to both treatment and outcome
        for node in self.graph.nodes():
            if (node != treatment and node != outcome and
                self.graph.has_edge(node, treatment) and
                self.graph.has_edge(node, outcome)):
                confounders.add(node)
        
        logger.info(f"Identified confounders: {confounders}")
        
        return confounders
    
    def identify_mediators(self, treatment: str = 'treatment',
                          outcome: str = 'outcome') -> Set[str]:
        """
        Identify mediating variables.
        
        Args:
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            
        Returns:
            Set of mediating variable names
        """
        logger.info("Identifying mediating variables")
        
        if self.graph is None:
            self.build_graph_from_config()
        
        mediators = set()
        
        # Find nodes that are on directed paths from treatment to outcome
        # but are not confounders
        confounders = self.identify_confounders(treatment, outcome)
        
        for node in self.graph.nodes():
            if (node != treatment and node != outcome and
                node not in confounders and
                nx.has_path(self.graph, treatment, node) and
                nx.has_path(self.graph, node, outcome)):
                mediators.add(node)
        
        logger.info(f"Identified mediators: {mediators}")
        
        return mediators
    
    def get_causal_relationships(self) -> Dict[str, List[str]]:
        """
        Get all causal relationships in the graph.
        
        Returns:
            Dictionary mapping each node to its direct effects
        """
        if self.graph is None:
            self.build_graph_from_config()
        
        relationships = {}
        
        for node in self.graph.nodes():
            effects = list(self.graph.successors(node))
            relationships[node] = effects
        
        return relationships
    
    def export_graph(self, filepath: str, format: str = 'png') -> None:
        """
        Export the causal graph to a file.
        
        Args:
            filepath: Path to save the graph
            format: Output format (png, svg, pdf, etc.)
        """
        logger.info(f"Exporting graph to {filepath}")
        
        if self.graph is None:
            self.build_graph_from_config()
        
        try:
            # Use networkx to draw and save the graph
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.graph, pos, 
                                 node_color='lightblue', 
                                 node_size=2000)
            
            # Draw edges
            nx.draw_networkx_edges(self.graph, pos, 
                                 edge_color='gray', 
                                 arrows=True, 
                                 arrowsize=20)
            
            # Draw labels
            nx.draw_networkx_labels(self.graph, pos, 
                                  font_size=10, 
                                  font_weight='bold')
            
            plt.title("Causal Graph", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filepath, format=format, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Graph exported successfully to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
            raise 