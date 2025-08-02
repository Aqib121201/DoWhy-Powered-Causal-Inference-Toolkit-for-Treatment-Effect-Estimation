#!/bin/bash

# Entrypoint script for the Causal Inference Toolkit Docker container

set -e

echo "🧠 Starting Causal Inference Toolkit..."

# Function to run the pipeline
run_pipeline() {
    echo "Running causal inference pipeline..."
    python run_pipeline.py "$@"
}

# Function to run the Streamlit app
run_streamlit() {
    echo "Starting Streamlit web application..."
    cd app
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0
}

# Function to run Jupyter notebook
run_jupyter() {
    echo "Starting Jupyter notebook server..."
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
}

# Function to run tests
run_tests() {
    echo "Running tests..."
    pytest tests/ -v
}

# Function to show help
show_help() {
    echo "Causal Inference Toolkit Docker Container"
    echo "========================================"
    echo ""
    echo "Usage:"
    echo "  docker run causal-toolkit pipeline [options]  - Run the complete pipeline"
    echo "  docker run causal-toolkit streamlit           - Start Streamlit web app"
    echo "  docker run causal-toolkit jupyter             - Start Jupyter notebook"
    echo "  docker run causal-toolkit test                - Run tests"
    echo "  docker run causal-toolkit help                - Show this help"
    echo ""
    echo "Pipeline options:"
    echo "  --samples N              Number of samples to generate (default: 10000)"
    echo "  --true-effect FLOAT      True treatment effect for comparison"
    echo "  --interactive            Create interactive plots"
    echo "  --save-data              Save generated data"
    echo "  --log-level LEVEL        Logging level (DEBUG, INFO, WARNING, ERROR)"
    echo ""
    echo "Examples:"
    echo "  docker run -p 8501:8501 causal-toolkit streamlit"
    echo "  docker run -p 8888:8888 causal-toolkit jupyter"
    echo "  docker run causal-toolkit pipeline --samples 5000 --true-effect 0.15"
}

# Main logic
case "${1:-help}" in
    "pipeline")
        shift
        run_pipeline "$@"
        ;;
    "streamlit")
        run_streamlit
        ;;
    "jupyter")
        run_jupyter
        ;;
    "test")
        run_tests
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use 'help' to see available commands"
        exit 1
        ;;
esac 