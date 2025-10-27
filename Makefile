# RC-GNN Makefile
# Comprehensive workflow for data processing, training, and evaluation

.PHONY: help install clean data train-synth train-air validate compare all test
.DEFAULT_GOAL := help

#=============================================================================
# Configuration
#=============================================================================

PYTHON := python3
PIP := pip
CONDA_ENV := rcgnn-env
DATA_DIR := data/interim
ARTIFACTS := artifacts
CONFIGS := configs

# Synthetic data parameters
SYNTH_SMALL := $(DATA_DIR)/synth_small
SYNTH_NONLINEAR := $(DATA_DIR)/synth_nonlinear
SYNTH_D := 10
SYNTH_EDGES := 15

# UCI Air data parameters
UCI_AIR := $(DATA_DIR)/uci_air

# Model outputs
ADJ_SYNTH := $(ARTIFACTS)/adjacency/A_mean.npy
ADJ_AIR := $(ARTIFACTS)/adjacency/A_mean_air.npy

# Node names for UCI Air (13 sensors)
UCI_AIR_NODES := "CO,PT08.S1,NMHC,C6H6,PT08.S2,NOx,PT08.S3,NO2,PT08.S4,PT08.S5,T,RH,AH"

#=============================================================================
# Help
#=============================================================================

help: ## Show this help message
	@echo "════════════════════════════════════════════════════════════════════"
	@echo "  RC-GNN Makefile - Robust Causal Graph Neural Networks"
	@echo "════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make install              # Install dependencies"
	@echo "  2. make data                 # Generate all datasets"
	@echo "  3. make train-synth          # Train on synthetic data"
	@echo "  4. make train-air            # Train on UCI Air data"
	@echo "  5. make validate-all         # Run all validation"
	@echo ""
	@echo "Or simply: make all           # Complete pipeline"
	@echo ""

#=============================================================================
# Installation & Setup
#=============================================================================

install: ## Install dependencies
	@echo "📦 Installing dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed successfully"

create-env: ## Create conda environment
	@echo "🐍 Creating conda environment: $(CONDA_ENV)..."
	conda create -y -n $(CONDA_ENV) python=3.12 pip 2>/dev/null || true
	@echo "✅ Conda environment created. Activate with: conda activate $(CONDA_ENV)"

setup: create-env install ## Complete setup (environment + dependencies)

check-env: ## Check Python environment
	@echo "🔍 Checking environment..."
	@$(PYTHON) -c "import torch; import numpy; import networkx; print('✅ Core packages available')"
	@$(PYTHON) -c "import sys; print(f'Python: {sys.version}')"
	@echo "✅ Environment check complete"

#=============================================================================
# Data Processing
#=============================================================================

data: data-synth-small data-synth-nonlinear data-air ## Generate all datasets

data-synth-small: ## Generate small synthetic dataset (linear)
	@echo "🔬 Generating small synthetic dataset (linear)..."
	@mkdir -p $(SYNTH_SMALL)
	$(PYTHON) scripts/synth_bench.py \
		--graph_type er \
		--d $(SYNTH_D) \
		--edges $(SYNTH_EDGES) \
		--mechanism linear \
		--missing_type mcar \
		--missing_rate 0.2 \
		--noise_scale 0.3 \
		--output $(SYNTH_SMALL) \
		--seed 42
	@echo "✅ Small synthetic dataset created at $(SYNTH_SMALL)"

data-synth-nonlinear: ## Generate synthetic dataset (nonlinear MLP)
	@echo "🔬 Generating nonlinear synthetic dataset (MLP)..."
	@mkdir -p $(SYNTH_NONLINEAR)
	$(PYTHON) scripts/synth_bench.py \
		--graph_type er \
		--d $(SYNTH_D) \
		--edges $(SYNTH_EDGES) \
		--mechanism mlp \
		--missing_type mcar \
		--missing_rate 0.2 \
		--noise_scale 0.3 \
		--output $(SYNTH_NONLINEAR) \
		--seed 300
	@echo "✅ Nonlinear synthetic dataset created at $(SYNTH_NONLINEAR)"

data-air: ## Download and process UCI Air Quality dataset
	@echo "🌍 Downloading UCI Air Quality dataset..."
	@mkdir -p $(UCI_AIR)
	$(PYTHON) scripts/download_and_convert_uci_safe.py \
		--out $(UCI_AIR)
	@echo "✅ UCI Air dataset prepared at $(UCI_AIR)"

data-inspect: ## Inspect generated datasets
	@echo "📊 Dataset Statistics:"
	@echo ""
	@if [ -f $(SYNTH_SMALL)/meta.json ]; then \
		echo "📁 Synthetic Small:"; \
		$(PYTHON) -c "import json,numpy as np; meta=json.load(open('$(SYNTH_SMALL)/meta.json')); X=np.load('$(SYNTH_SMALL)/X.npy'); print(f\"  Nodes: {meta.get('d','N/A')}\n  Edges: {meta.get('edges','N/A')}\n  Mechanism: {meta.get('mechanism','N/A')}\n  Shape: {X.shape}\")"; \
		echo ""; \
	fi
	@if [ -f $(UCI_AIR)/X.npy ]; then \
		echo "📁 UCI Air Quality:"; \
		$(PYTHON) -c "import numpy as np; X=np.load('$(UCI_AIR)/X.npy'); M=np.load('$(UCI_AIR)/M.npy'); print(f\"  Shape: {X.shape}\n  Missing rate: {(M==0).mean():.2%}\")"; \
	fi

#=============================================================================
# Training
#=============================================================================

train-synth: data-synth-small ## Train RC-GNN on synthetic data
	@echo "🚀 Training RC-GNN on synthetic dataset..."
	@mkdir -p $(ARTIFACTS)/checkpoints $(ARTIFACTS)/adjacency
	$(PYTHON) scripts/train_rcgnn.py \
		$(CONFIGS)/data.yaml \
		$(CONFIGS)/model.yaml \
		$(CONFIGS)/train.yaml \
		--adj-output $(ADJ_SYNTH)
	@echo "✅ Training complete! Adjacency at $(ADJ_SYNTH)"

train-air: data-air ## Train RC-GNN on UCI Air Quality dataset
	@echo "🚀 Training RC-GNN on UCI Air Quality..."
	@mkdir -p $(ARTIFACTS)/checkpoints $(ARTIFACTS)/adjacency
	$(PYTHON) scripts/train_rcgnn.py \
		$(CONFIGS)/data_uci.yaml \
		$(CONFIGS)/model.yaml \
		$(CONFIGS)/train.yaml \
		--adj-output $(ADJ_AIR)
	@echo "✅ Training complete! Adjacency at $(ADJ_AIR)"

train-air-live: data-air ## Train UCI Air with live log streaming (tee)
	@echo "📺 Live training (logs streaming to terminal and saved to artifacts/train_air_live.log)"
	@mkdir -p $(ARTIFACTS)/checkpoints $(ARTIFACTS)/adjacency
	$(PYTHON) -u scripts/train_rcgnn.py \
		$(CONFIGS)/data_uci.yaml \
		$(CONFIGS)/model.yaml \
		$(CONFIGS)/train.yaml \
		--adj-output $(ADJ_AIR) 2>&1 | tee $(ARTIFACTS)/train_air_live.log

train-all: train-synth train-air ## Train on both datasets

train-quick: ## Quick training (5 epochs, for testing)
	@echo "⚡ Quick training run (5 epochs)..."
	$(PYTHON) scripts/train_rcgnn.py \
		$(CONFIGS)/data.yaml \
		$(CONFIGS)/model.yaml \
		$(CONFIGS)/train.yaml \
		--epochs 5
	@echo "✅ Quick training complete"

#=============================================================================
# Validation & Evaluation
#=============================================================================

validate-synth: ## Validate synthetic results (basic)
	@echo "📊 Validating synthetic results..."
	@mkdir -p $(ARTIFACTS)/validation_synth
	$(PYTHON) scripts/validate_and_visualize.py \
		--adjacency $(ADJ_SYNTH) \
		--data-root $(SYNTH_SMALL) \
		--threshold 0.5 \
		--export $(ARTIFACTS)/validation_synth
	@echo "✅ Validation complete at $(ARTIFACTS)/validation_synth"

validate-synth-advanced: ## Validate synthetic (PUBLICATION READY)
	@echo "📊 Advanced validation - synthetic..."
	@mkdir -p $(ARTIFACTS)/validation_synth_advanced
	$(PYTHON) scripts/validate_and_visualize_advanced.py \
		--adjacency $(ADJ_SYNTH) \
		--data-root $(SYNTH_SMALL) \
		--threshold 0.5 \
		--export $(ARTIFACTS)/validation_synth_advanced
	@echo "✅ Advanced validation complete!"

validate-air: ## Validate UCI Air results (basic)
	@echo "📊 Validating UCI Air results..."
	@mkdir -p $(ARTIFACTS)/validation_air
	$(PYTHON) scripts/validate_and_visualize.py \
		--adjacency $(ADJ_AIR) \
		--data-root $(UCI_AIR) \
		--threshold 0.5 \
		--export $(ARTIFACTS)/validation_air
	@echo "✅ Validation complete at $(ARTIFACTS)/validation_air"

validate-air-advanced: ## Validate UCI Air (PUBLICATION READY)
	@echo "📊 Advanced validation - UCI Air..."
	@mkdir -p $(ARTIFACTS)/validation_air_advanced
	$(PYTHON) scripts/validate_and_visualize_advanced.py \
		--adjacency $(ADJ_AIR) \
		--data-root $(UCI_AIR) \
		--threshold 0.5 \
		--export $(ARTIFACTS)/validation_air_advanced \
		--node-names $(UCI_AIR_NODES)
	@echo "✅ Advanced validation complete!"
	@echo ""
	@echo "📈 Key Results:"
	@$(PYTHON) -c "import json; m=json.load(open('$(ARTIFACTS)/validation_air_advanced/metrics.json')); print(f\"  AUPRC: {m['auprc']:.4f} (+{m['auprc_vs_chance']*100:.1f}% vs chance)\n  F1: {m['f1']:.4f}\n  Orientation Acc: {m['orientation_acc']:.1%}\")" 2>/dev/null || echo "  (metrics.json not found)"

validate-all: validate-synth-advanced validate-air-advanced ## Run all advanced validation

#=============================================================================
# Baseline Comparison
#=============================================================================

compare-baselines: ## Compare RC-GNN vs baselines
	@echo "📊 Running baseline comparison..."
	@mkdir -p $(ARTIFACTS)/baseline_comparison
	$(PYTHON) scripts/compare_baselines.py \
		--config $(CONFIGS)/data.yaml \
		--export $(ARTIFACTS)/baseline_comparison
	@echo "✅ Baseline comparison complete!"

#=============================================================================
# Results & Reports
#=============================================================================

results: ## Show summary of all results
	@echo "════════════════════════════════════════════════════════════════════"
	@echo "  RC-GNN Results Summary"
	@echo "════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "📊 SYNTHETIC DATA:"
	@if [ -f $(ARTIFACTS)/validation_synth_advanced/metrics.json ]; then \
		$(PYTHON) -c "import json; m=json.load(open('$(ARTIFACTS)/validation_synth_advanced/metrics.json')); print(f\"  AUPRC: {m['auprc']:.4f}\n  F1: {m['f1']:.4f}\n  SHD: {m['shd']}\n  Orientation: {m['orientation_acc']:.1%}\")"; \
	else \
		echo "  ❌ No results. Run: make validate-synth-advanced"; \
	fi
	@echo ""
	@echo "🌍 UCI AIR QUALITY:"
	@if [ -f $(ARTIFACTS)/validation_air_advanced/metrics.json ]; then \
		$(PYTHON) -c "import json; m=json.load(open('$(ARTIFACTS)/validation_air_advanced/metrics.json')); print(f\"  AUPRC: {m['auprc']:.4f} (+{m['auprc_vs_chance']*100:.1f}% vs chance)\n  F1: {m['f1']:.4f}\n  Orientation: {m['orientation_acc']:.1%}\n  Bootstrap CI: [{m['auprc_ci_low']:.3f}, {m['auprc_ci_high']:.3f}]\")"; \
	else \
		echo "  ❌ No results. Run: make validate-air-advanced"; \
	fi
	@echo ""

results-paper: ## Generate LaTeX table for paper
	@echo "\\begin{table}[htbp]"
	@echo "\\centering"
	@echo "\\caption{RC-GNN Performance on UCI Air Quality}"
	@echo "\\begin{tabular}{lcc}"
	@echo "\\hline"
	@echo "Metric & Value & 95\\% CI \\\\"
	@echo "\\hline"
	@if [ -f $(ARTIFACTS)/validation_air_advanced/metrics.json ]; then \
		$(PYTHON) -c "import json; m=json.load(open('$(ARTIFACTS)/validation_air_advanced/metrics.json')); print(f\"AUPRC & {m['auprc']:.4f} & [{m['auprc_ci_low']:.3f}, {m['auprc_ci_high']:.3f}] \\\\\\\\\nF1 & {m['f1']:.4f} & [{m['best_f1_ci_low']:.3f}, {m['best_f1_ci_high']:.3f}] \\\\\\\\\nOrientation Acc & {m['orientation_acc']:.2f} & -- \\\\\\\\\")"; \
	fi
	@echo "\\hline"
	@echo "\\end{tabular}"
	@echo "\\end{table}"

#=============================================================================
# Testing
#=============================================================================

test: ## Run all tests
	@echo "🧪 Running tests..."
	pytest -v tests/
	@echo "✅ Tests passed!"

test-quick: ## Run smoke tests only
	@echo "🧪 Running smoke tests..."
	pytest -v tests/test_synth_smoke.py tests/test_training_step.py
	@echo "✅ Smoke tests passed!"

#=============================================================================
# Cleanup
#=============================================================================

clean: ## Clean generated files (keep data)
	@echo "🧹 Cleaning artifacts..."
	rm -rf $(ARTIFACTS)/*
	rm -rf __pycache__ src/__pycache__ scripts/__pycache__
	rm -rf .pytest_cache
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

clean-data: ## Clean datasets (WARNING!)
	@echo "⚠️  WARNING: This will delete ALL datasets!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(DATA_DIR)/*; \
		echo "✅ Datasets cleaned"; \
	fi

clean-all: clean clean-data ## Clean everything

#=============================================================================
# Complete Pipelines
#=============================================================================

all: install data train-all validate-all compare-baselines results ## Complete pipeline
	@echo ""
	@echo "════════════════════════════════════════════════════════════════════"
	@echo "  ✅ Complete pipeline finished!"
	@echo "════════════════════════════════════════════════════════════════════"

pipeline-synth: data-synth-small train-synth validate-synth-advanced ## Synthetic pipeline
	@echo "✅ Synthetic pipeline complete!"

pipeline-air: data-air train-air validate-air-advanced ## UCI Air pipeline
	@echo "✅ UCI Air pipeline complete!"

pipeline-paper: pipeline-air compare-baselines results-paper ## Paper results
	@echo "✅ Paper-ready results generated!"

#=============================================================================
# Utilities
#=============================================================================

status: ## Show project status
	@echo "📊 RC-GNN Project Status:"
	@echo ""
	@echo "Data:"
	@echo -n "  Synthetic: "; [ -f $(SYNTH_SMALL)/X.npy ] && echo "✅" || echo "❌"
	@echo -n "  UCI Air: "; [ -f $(UCI_AIR)/X.npy ] && echo "✅" || echo "❌"
	@echo ""
	@echo "Models:"
	@echo -n "  Synthetic trained: "; [ -f $(ADJ_SYNTH) ] && echo "✅" || echo "❌"
	@echo -n "  UCI Air trained: "; [ -f $(ADJ_AIR) ] && echo "✅" || echo "❌"
	@echo ""
	@echo "Validation:"
	@echo -n "  Synthetic: "; [ -f $(ARTIFACTS)/validation_synth_advanced/metrics.json ] && echo "✅" || echo "❌"
	@echo -n "  UCI Air: "; [ -f $(ARTIFACTS)/validation_air_advanced/metrics.json ] && echo "✅" || echo "❌"

version: ## Show version info
	@echo "RC-GNN Version Information:"
	@$(PYTHON) --version
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@$(PYTHON) -c "import numpy; print(f'NumPy: {numpy.__version__}')"

docs: ## Show documentation
	@echo "📚 RC-GNN Documentation:"
	@echo ""
	@echo "  📋 VALIDATION_INDEX.md            - Complete guide"
	@echo "  ⚡ VALIDATION_QUICK_REF.md        - Quick reference"
	@echo "  📊 VALIDATION_ADVANCED_GUIDE.md   - Advanced features"
	@echo "  📈 VALIDATION_SUMMARY.md          - Results summary"
	@echo ""
