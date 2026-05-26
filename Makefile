# Makefile for AgentEcon Project

.PHONY: help install setup-data setup-news setup-fabric run-smoke run-pilot run-experiment run-mixed-mini run-calibration run-prompt-injection run-sybil run-ablation run-analysis figures clean test

# Default target
help:
	@echo "AgentEcon - Reproducible Prediction Market Oracle"
	@echo ""
	@echo "Available targets:"
	@echo "  install          - Install Python dependencies"
	@echo "  setup-data       - Download and process financial data"
	@echo "  setup-news       - Download/cache GDELT news contexts"
	@echo "  setup-fabric     - Setup Fabric testbed"
	@echo "  run-smoke        - Run smoke test"
	@echo "  run-pilot        - Run pilot experiment (200 queries)"
	@echo "  run-experiment   - Run full stylised experiment grid"
	@echo "  run-mixed-mini   - Run mini mixed experiment (requires vLLM/model weights)"
	@echo "  run-calibration  - Run LLM/QRE calibration (requires vLLM/model weights)"
	@echo "  run-prompt-injection - Run stylised prompt-injection robustness"
	@echo "  run-sybil        - Run Sybil/reputation dynamics"
	@echo "  run-ablation     - Run oracle mechanism ablations"
	@echo "  run-analysis     - Run statistical analysis"
	@echo "  clean            - Clean generated files"
	@echo "  test             - Run unit tests"

# Environment setup
PYTHON ?= python3
ifeq ($(OS),Windows_NT)
VENV ?= .venv
PYTHON_BIN := $(VENV)/Scripts/python.exe
else
VENV ?= .venv-wsl
PYTHON_BIN := $(VENV)/bin/python
endif

# Installation
install:
	@echo "Creating Python virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing dependencies..."
	$(PYTHON_BIN) -m pip install --upgrade pip
	$(PYTHON_BIN) -m pip install -r requirements.txt
	@echo "Installation complete!"

# Data setup
setup-data:
	@echo "Setting up financial data..."
	$(PYTHON_BIN) -c "from src.data.asset_loader import AssetLoader; l = AssetLoader('data'); print('Asset loader initialized')"
	$(PYTHON_BIN) scripts/download_data.py

setup-news:
	@echo "Downloading/caching structured GDELT news contexts..."
	$(PYTHON_BIN) scripts/download_news_contexts.py --n-queries 10

# Fabric setup
setup-fabric:
	@echo "Setting up Fabric testbed..."
	@if command -v docker &> /dev/null; then \
		cd fabric-deploy && docker-compose up -d; \
		echo "Fabric started"; \
	else \
		echo "Docker not found. Install Docker first."; \
	fi

# Experiment runs
run-smoke:
	@echo "Running smoke test..."
	$(PYTHON_BIN) -c "from src.experiments.runner import ExperimentConfig, ExperimentRunner; \
		c = ExperimentConfig(experiment_name='smoke_test', mode='stylised'); \
		r = ExperimentRunner(c); \
		r.run(scale='smoke')"

run-pilot:
	@echo "Running pilot experiment..."
	$(PYTHON_BIN) -c "from src.experiments.runner import ExperimentConfig, ExperimentRunner; \
		c = ExperimentConfig(experiment_name='pilot', mode='stylised', \
			seeds=[1234], tau_values=[1.0], lambda_values=[0.5]); \
		r = ExperimentRunner(c); \
		r.run(scale='pilot')"

run-experiment:
	@echo "Running full experiment grid..."
	$(PYTHON_BIN) -c "from src.experiments.runner import ExperimentConfig, ExperimentRunner; \
		c = ExperimentConfig(experiment_name='full_grid', mode='stylised'); \
		r = ExperimentRunner(c); \
		r.run(scale='full')"

run-mixed-mini:
	@echo "Running mini mixed experiment (4 LLM + 60 stylised validators)..."
	$(PYTHON_BIN) -c "from src.experiments.runner import ExperimentConfig, ExperimentRunner; \
		c = ExperimentConfig(experiment_name='mixed_mini', mode='mixed', \
			assets=['XAU_USD'], seeds=[1234], tau_values=[0.7], lambda_values=[0.5], \
			n_queries_pilot=200, n_validators=64, n_llm_validators=4, n_stylised_validators=60); \
		r = ExperimentRunner(c); \
		r.run(scale='mini')"

run-calibration:
	@echo "Running LLM/QRE calibration. Requires vLLM and downloaded Hugging Face models."
	$(PYTHON_BIN) scripts/run_llm_calibration.py --n-queries 200

run-prompt-injection:
	@echo "Running stylised prompt-injection robustness experiment..."
	$(PYTHON_BIN) scripts/run_prompt_injection.py --n-queries 100

run-sybil:
	@echo "Running Sybil/reputation dynamics experiment..."
	$(PYTHON_BIN) scripts/run_sybil_reputation.py --n-rounds 120

run-ablation:
	@echo "Running oracle mechanism ablation experiment..."
	$(PYTHON_BIN) scripts/run_ablation.py --n-queries 100

# Analysis
run-analysis:
	@echo "Running statistical analysis..."
	$(PYTHON_BIN) scripts/analyze_results.py

# Generate figures
figures:
	@echo "Generating figures..."
	$(PYTHON_BIN) scripts/generate_figures.py

# Testing
test:
	@echo "Running unit tests..."
	$(PYTHON_BIN) -m pytest tests/ -v

# Clean
clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf data/runs/*.csv data/runs/*.json
	rm -rf $(VENV)
	@echo "Clean complete"

# Docker commands for Fabric
fabric-start:
	cd fabric-deploy && docker-compose up -d

fabric-stop:
	cd fabric-deploy && docker-compose down

fabric-logs:
	cd fabric-deploy && docker-compose logs -f

# AWS deployment
aws-init:
	cd aws-deploy && terraform init

aws-plan:
	cd aws-deploy && terraform plan

aws-deploy:
	cd aws-deploy && terraform apply

aws-destroy:
	cd aws-deploy && terraform destroy

# Build chaincode
build-chaincode:
	cd chaincode && go mod tidy && go build -o agentecon

# Deploy chaincode
deploy-chaincode:
	cd fabric-deploy && ./deploy-chaincode.sh

# Full setup and first run
setup: install setup-data
	@echo "Setup complete! Run 'make run-smoke' to verify."

# Development helpers
lint:
	$(PYTHON_BIN) -m flake8 src/ --max-line-length=120

format:
	$(PYTHON_BIN) -m black src/ tests/

# Documentation
docs:
	@echo "Generating documentation..."
	cd docs && make html
