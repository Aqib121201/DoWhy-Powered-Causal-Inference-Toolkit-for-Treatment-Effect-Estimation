"""
Streamlit web application for the Causal Inference Toolkit.

This application provides an interactive interface for causal inference analysis,
including data generation, causal graph visualization, treatment effect estimation,
and result interpretation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_generator import HealthcareDataGenerator
from src.causal_graphs import CausalGraphBuilder
from src.treatment_effects import TreatmentEffectEstimator
from src.visualization import CausalVisualizer
from src.config import get_config

# Page configuration
st.set_page_config(
    page_title="Causal Inference Toolkit",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Causal Inference Toolkit</h1>', unsafe_allow_html=True)
    st.markdown("""
    A comprehensive toolkit for causal inference and treatment effect estimation using the DoWhy framework.
    This application demonstrates causal graphs, backdoor paths, do-calculus logic, and average treatment effect (ATE) estimation.
    """)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Data generation parameters
    st.sidebar.header("Data Generation")
    n_samples = st.sidebar.slider("Number of samples", 1000, 20000, 10000, 1000)
    treatment_effect = st.sidebar.slider("True treatment effect", 0.0, 0.5, 0.15, 0.01)
    confounding_strength = st.sidebar.slider("Confounding strength", 0.0, 1.0, 0.3, 0.1)
    
    # Estimation parameters
    st.sidebar.header("Estimation Settings")
    bootstrap_samples = st.sidebar.slider("Bootstrap samples", 100, 2000, 1000, 100)
    confidence_level = st.sidebar.slider("Confidence level", 0.90, 0.99, 0.95, 0.01)
    
    # Visualization settings
    st.sidebar.header("Visualization")
    interactive_plots = st.sidebar.checkbox("Interactive plots", value=True)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Generation", 
        "🕸️ Causal Graph", 
        "📈 Treatment Effects", 
        "🔍 Analysis", 
        "📋 Results"
    ])
    
    with tab1:
        show_data_generation(n_samples, treatment_effect, confounding_strength)
    
    with tab2:
        show_causal_graph()
    
    with tab3:
        show_treatment_effects(bootstrap_samples, confidence_level)
    
    with tab4:
        show_analysis()
    
    with tab5:
        show_results()

def show_data_generation(n_samples, treatment_effect, confounding_strength):
    """Show data generation section."""
    st.markdown('<h2 class="section-header">📊 Data Generation</h2>', unsafe_allow_html=True)
    
    if st.button("Generate New Data", type="primary"):
        with st.spinner("Generating healthcare data..."):
            # Update config
            config = get_config()
            config["data_config"]["treatment_effect"] = treatment_effect
            config["data_config"]["confounding_strength"] = confounding_strength
            config["data_config"]["n_samples"] = n_samples
            
            # Generate data
            data_generator = HealthcareDataGenerator(config["data_config"])
            data = data_generator.generate_dataset(n_samples)
            
            # Store in session state
            st.session_state.data = data
            st.session_state.data_generator = data_generator
            st.session_state.true_effect = treatment_effect
            
            st.success(f"Generated {n_samples} samples successfully!")
    
    if 'data' in st.session_state:
        data = st.session_state.data
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Samples", f"{len(data):,}")
        
        with col2:
            st.metric("Treatment Rate", f"{data['treatment'].mean():.3f}")
        
        with col3:
            st.metric("Outcome Rate", f"{data['outcome'].mean():.3f}")
        
        with col4:
            st.metric("True ATE", f"{st.session_state.true_effect:.3f}")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Data distributions
        col1, col2 = st.columns(2)
        
        with col1:
            # Treatment distribution
            fig = px.bar(
                x=data['treatment'].value_counts().index,
                y=data['treatment'].value_counts().values,
                title="Treatment Distribution",
                labels={'x': 'Treatment', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Outcome distribution
            fig = px.bar(
                x=data['outcome'].value_counts().index,
                y=data['outcome'].value_counts().values,
                title="Outcome Distribution",
                labels={'x': 'Outcome', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Variable distributions
        st.subheader("Variable Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(data, x='age', title="Age Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(data, x='bmi', title="BMI Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Matrix")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Click 'Generate New Data' to start the analysis.")

def show_causal_graph():
    """Show causal graph section."""
    st.markdown('<h2 class="section-header">🕸️ Causal Graph</h2>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state:
        st.warning("Please generate data first in the Data Generation tab.")
        return
    
    # Build causal graph
    with st.spinner("Building causal graph..."):
        graph_builder = CausalGraphBuilder()
        graph = graph_builder.build_graph_from_config()
        
        # Store in session state
        st.session_state.graph_builder = graph_builder
        st.session_state.graph = graph
    
    # Graph statistics
    stats = graph_builder.get_graph_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nodes", stats["n_nodes"])
    
    with col2:
        st.metric("Edges", stats["n_edges"])
    
    with col3:
        st.metric("Density", f"{stats['density']:.3f}")
    
    with col4:
        st.metric("Connected", "Yes" if stats["is_connected"] else "No")
    
    # Graph visualization
    st.subheader("Causal Graph Visualization")
    
    # Create interactive graph using plotly
    pos = nx.spring_layout(graph, k=1, iterations=50)
    
    # Edges
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
        mode='lines',
        name='Edges'
    )
    
    # Nodes
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
        ),
        name='Nodes'
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Causal Graph',
                       showlegend=True,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Graph analysis
    st.subheader("Graph Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confounders
        confounders = graph_builder.identify_confounders()
        st.write("**Confounding Variables:**")
        if confounders:
            for confounder in confounders:
                st.write(f"- {confounder}")
        else:
            st.write("No confounders identified")
    
    with col2:
        # Mediators
        mediators = graph_builder.identify_mediators()
        st.write("**Mediating Variables:**")
        if mediators:
            for mediator in mediators:
                st.write(f"- {mediator}")
        else:
            st.write("No mediators identified")
    
    # Backdoor paths
    st.subheader("Backdoor Paths")
    backdoor_paths = graph_builder.identify_backdoor_paths()
    
    if backdoor_paths:
        st.write(f"Found {len(backdoor_paths)} backdoor path(s):")
        for i, path in enumerate(backdoor_paths, 1):
            st.write(f"{i}. {' → '.join(path)}")
    else:
        st.write("No backdoor paths found")

def show_treatment_effects(bootstrap_samples, confidence_level):
    """Show treatment effects estimation section."""
    st.markdown('<h2 class="section-header">📈 Treatment Effects</h2>', unsafe_allow_html=True)
    
    if 'data' not in st.session_state:
        st.warning("Please generate data first in the Data Generation tab.")
        return
    
    if st.button("Estimate Treatment Effects", type="primary"):
        with st.spinner("Estimating treatment effects..."):
            data = st.session_state.data
            
            # Configure estimator
            config = {
                "bootstrap_samples": bootstrap_samples,
                "confidence_level": confidence_level
            }
            
            estimator = TreatmentEffectEstimator(config)
            
            # Set true effect
            if 'true_effect' in st.session_state:
                estimator.set_true_effect(st.session_state.true_effect)
            
            # Estimate effects
            results = estimator.estimate_all_methods(data)
            
            # Store in session state
            st.session_state.estimator = estimator
            st.session_state.results = results
            
            st.success("Treatment effects estimated successfully!")
    
    if 'results' in st.session_state:
        results = st.session_state.results
        
        # Results summary
        st.subheader("Estimation Results")
        
        # Create results table
        results_data = []
        for method, result in results.items():
            if "error" not in result:
                results_data.append({
                    "Method": method.replace("_", " ").title(),
                    "ATE": f"{result['ate']:.4f}",
                    "CI Lower": f"{result['ci_lower']:.4f}",
                    "CI Upper": f"{result['ci_upper']:.4f}",
                    "P-value": f"{result['p_value']:.4f}"
                })
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Forest plot
            st.subheader("Forest Plot")
            
            methods = [row["Method"] for row in results_data]
            ates = [float(row["ATE"]) for row in results_data]
            ci_lowers = [float(row["CI Lower"]) for row in results_data]
            ci_uppers = [float(row["CI Upper"]) for row in results_data]
            
            fig = go.Figure()
            
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
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Bootstrap distributions
            st.subheader("Bootstrap Distributions")
            
            fig = go.Figure()
            
            for method, result in results.items():
                if "error" not in result and "bootstrap_ates" in result:
                    fig.add_trace(go.Histogram(
                        x=result["bootstrap_ates"],
                        name=method.replace("_", " ").title(),
                        opacity=0.7
                    ))
            
            fig.add_vline(x=0, line_dash="dash", line_color="red", 
                         annotation_text="No Effect")
            
            fig.update_layout(
                title="Bootstrap Distributions",
                xaxis_title="Average Treatment Effect (ATE)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("No successful estimations found.")

def show_analysis():
    """Show analysis section."""
    st.markdown('<h2 class="section-header">🔍 Analysis</h2>', unsafe_allow_html=True)
    
    if 'results' not in st.session_state:
        st.warning("Please estimate treatment effects first in the Treatment Effects tab.")
        return
    
    results = st.session_state.results
    data = st.session_state.data
    
    # Method comparison
    st.subheader("Method Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ATE comparison
        methods = []
        ates = []
        for method, result in results.items():
            if "error" not in result:
                methods.append(method.replace("_", " ").title())
                ates.append(result["ate"])
        
        if methods:
            fig = px.bar(
                x=methods,
                y=ates,
                title="ATE Comparison Across Methods",
                labels={'x': 'Method', 'y': 'ATE'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # P-value comparison
        methods = []
        p_values = []
        for method, result in results.items():
            if "error" not in result:
                methods.append(method.replace("_", " ").title())
                p_values.append(result["p_value"])
        
        if methods:
            fig = px.bar(
                x=methods,
                y=p_values,
                title="P-value Comparison",
                labels={'x': 'Method', 'y': 'P-value'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Accuracy analysis
    if 'true_effect' in st.session_state:
        st.subheader("Accuracy Analysis")
        
        estimator = st.session_state.estimator
        accuracy = estimator.evaluate_accuracy()
        
        # Create accuracy table
        accuracy_data = []
        for method, acc in accuracy.items():
            accuracy_data.append({
                "Method": method.replace("_", " ").title(),
                "Estimated ATE": f"{acc['estimated_ate']:.4f}",
                "True ATE": f"{acc['true_ate']:.4f}",
                "Bias": f"{acc['bias']:.4f}",
                "Relative Bias": f"{acc['relative_bias']:.4f}",
                "CI Contains True": "Yes" if acc['ci_contains_true'] else "No"
            })
        
        if accuracy_data:
            accuracy_df = pd.DataFrame(accuracy_data)
            st.dataframe(accuracy_df, use_container_width=True)
    
    # Sensitivity analysis
    st.subheader("Sensitivity Analysis")
    
    # Propensity score distribution
    if 'propensity_score_matching' in results:
        result = results['propensity_score_matching']
        if "propensity_scores" in result:
            fig = px.histogram(
                x=result["propensity_scores"],
                title="Propensity Score Distribution",
                labels={'x': 'Propensity Score', 'y': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)

def show_results():
    """Show results summary section."""
    st.markdown('<h2 class="section-header">📋 Results Summary</h2>', unsafe_allow_html=True)
    
    if 'results' not in st.session_state:
        st.warning("Please complete the analysis first.")
        return
    
    # Summary metrics
    data = st.session_state.data
    results = st.session_state.results
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{len(data):,}")
    
    with col2:
        successful_methods = sum(1 for r in results.values() if "error" not in r)
        st.metric("Successful Methods", successful_methods)
    
    with col3:
        if 'true_effect' in st.session_state:
            st.metric("True ATE", f"{st.session_state.true_effect:.3f}")
    
    with col4:
        if 'estimator' in st.session_state:
            estimator = st.session_state.estimator
            if estimator.true_effect is not None:
                accuracy = estimator.evaluate_accuracy()
                correct_ci = sum(1 for acc in accuracy.values() if acc['ci_contains_true'])
                st.metric("Correct CIs", f"{correct_ci}/{len(accuracy)}")
    
    # Key findings
    st.subheader("Key Findings")
    
    if results:
        # Find best method
        best_method = None
        best_ate = None
        
        for method, result in results.items():
            if "error" not in result:
                if best_method is None or abs(result["ate"]) > abs(best_ate):
                    best_method = method
                    best_ate = result["ate"]
        
        if best_method:
            st.write(f"**Best performing method:** {best_method.replace('_', ' ').title()}")
            st.write(f"**Estimated ATE:** {best_ate:.4f}")
            
            if 'true_effect' in st.session_state:
                bias = best_ate - st.session_state.true_effect
                st.write(f"**Bias:** {bias:.4f}")
    
    # Recommendations
    st.subheader("Recommendations")
    
    st.write("""
    **For Researchers:**
    - Consider using multiple estimation methods for robustness
    - Validate causal assumptions through sensitivity analysis
    - Report confidence intervals and uncertainty measures
    
    **For Practitioners:**
    - Focus on methods that align with your causal assumptions
    - Consider the trade-off between bias and variance
    - Validate results with domain experts
    """)
    
    # Export options
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Results CSV"):
            # Create results CSV
            results_data = []
            for method, result in results.items():
                if "error" not in result:
                    results_data.append({
                        "Method": method,
                        "ATE": result["ate"],
                        "CI_Lower": result["ci_lower"],
                        "CI_Upper": result["ci_upper"],
                        "P_value": result["p_value"]
                    })
            
            if results_data:
                results_df = pd.DataFrame(results_data)
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="treatment_effects_results.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("Download Data"):
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download Data CSV",
                data=csv,
                file_name="healthcare_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main() 