# Makefile for Causal Inference Toolkit

.PHONY: help install test clean run-pipeline run-app run-notebook docker-build docker-run

# Default target
help:
	@echo "Causal Inference Toolkit - Available Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install          Install dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo ""
	@echo "Running:"
	@echo "  run-pipeline     Run the complete causal inference pipeline"
	@echo "  run-app          Start the Streamlit web application"
	@echo "  run-notebook     Start Jupyter notebook server"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-cov         Run tests with coverage"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            Clean generated files"
	@echo "  format           Format code with black"
	@echo "  lint             Run linting with flake8"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install black flake8 pytest-cov

# Running the application
run-pipeline:
	python run_pipeline.py

run-app:
	cd app && streamlit run app.py

run-notebook:
	jupyter notebook

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Docker
docker-build:
	docker build -t causal-toolkit .

docker-run:
	docker run -p 8501:8501 causal-toolkit streamlit

docker-run-pipeline:
	docker run causal-toolkit pipeline

# Utilities
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf data/processed/*
	rm -rf models/*
	rm -rf visualizations/*
	rm -rf logs/*

format:
	black src/ tests/ app/ run_pipeline.py

lint:
	flake8 src/ tests/ app/ run_pipeline.py

# Development
dev-setup: install-dev
	pre-commit install

# Quick start
quick-start: install
	python run_pipeline.py --samples 1000 --true-effect 0.15

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "Documentation would be generated here"

# Package
package:
	python setup.py sdist bdist_wheel

# Environment
env-create:
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

env-activate:
	@echo "Activate virtual environment with: source venv/bin/activate"

# Data
data-generate:
	python run_pipeline.py --samples 10000 --save-data

# Analysis
analysis-full:
	python run_pipeline.py --samples 10000 --true-effect 0.15 --interactive --save-data

analysis-quick:
	python run_pipeline.py --samples 1000 --true-effect 0.15

# Monitoring
logs:
	tail -f causal_inference.log

# Backup
backup:
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		--exclude=venv \
		--exclude=__pycache__ \
		--exclude=*.pyc \
		--exclude=.git \
		.

# Help for specific commands
help-pipeline:
	@echo "Pipeline Options:"
	@python run_pipeline.py --help

help-docker:
	@echo "Docker Commands:"
	@echo "  docker run -p 8501:8501 causal-toolkit streamlit"
	@echo "  docker run -p 8888:8888 causal-toolkit jupyter"
	@echo "  docker run causal-toolkit pipeline --samples 5000"
	@echo "  docker run causal-toolkit test" 