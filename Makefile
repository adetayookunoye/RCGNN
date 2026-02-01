# RC-GNN Makefile
# Works on local machines and Sapelo HPC cluster

.PHONY: help install setup data train evaluate clean test
.DEFAULT_GOAL := help

#=============================================================================
# Configuration - Auto-detect environment
#=============================================================================

# Detect if running on Sapelo (SLURM environment)
ifdef SLURM_JOB_ID
    ENV := sapelo
    PYTHON := python
else ifdef SLURM_CLUSTER_NAME
    ENV := sapelo
    PYTHON := python
else
    ENV := local
    PYTHON := python3
endif

# Directories
DATA_DIR := data/interim
ARTIFACTS := artifacts
CONFIGS := configs
LOGS := logs

# Dataset paths
UCI_AIR := $(DATA_DIR)/uci_air
UCI_AIR_C := $(DATA_DIR)/uci_air_c
SYNTH := $(DATA_DIR)/synth_small

#=============================================================================
# Help
#=============================================================================

help: ## Show available commands
	@echo "================================================================"
	@echo "  RC-GNN Makefile"
	@echo "  Environment: $(ENV)"
	@echo "================================================================"
	@echo ""
	@echo "SETUP:"
	@echo "  make install          Install Python dependencies"
	@echo "  make setup            Full setup (env + dependencies)"
	@echo "  make check            Verify environment is ready"
	@echo ""
	@echo "DATA:"
	@echo "  make data             Generate all datasets"
	@echo "  make data-synth       Generate synthetic data"
	@echo "  make data-uci         Download UCI Air Quality"
	@echo ""
	@echo "TRAINING (Local):"
	@echo "  make train            Train on compound_full (default)"
	@echo "  make train-quick      Quick test (10 epochs)"
	@echo "  make train-all        Train on all 12 datasets"
	@echo ""
	@echo "TRAINING (Sapelo HPC):"
	@echo "  make submit           Submit GPU job to Sapelo"
	@echo "  make status           Check job status"
	@echo "  make logs             View latest job logs"
	@echo ""
	@echo "EVALUATION:"
	@echo "  make evaluate         Run comprehensive evaluation"
	@echo "  make results          Show results summary"
	@echo ""
	@echo "UTILITIES:"
	@echo "  make test             Run unit tests"
	@echo "  make clean            Clean generated files"
	@echo "  make sync-sapelo      Sync code to Sapelo"
	@echo ""

#=============================================================================
# Setup & Installation
#=============================================================================

install: ## Install Python dependencies
	@echo "[INFO] Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "[DONE] Dependencies installed"

setup-local: ## Setup for local development
	@echo "[INFO] Setting up local environment..."
	conda create -y -n rcgnn-env python=3.10 pip || true
	@echo "[INFO] Activate with: conda activate rcgnn-env"
	@echo "[INFO] Then run: make install"

setup-sapelo: ## Setup for Sapelo HPC
	@echo "[INFO] Setting up Sapelo environment..."
	@echo "Run these commands on Sapelo:"
	@echo "  module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1"
	@echo "  pip install --user -r requirements.txt"

setup: ## Full setup (auto-detects environment)
ifeq ($(ENV),sapelo)
	@$(MAKE) setup-sapelo
else
	@$(MAKE) setup-local
endif

check: ## Verify environment is ready
	@echo "[INFO] Checking environment ($(ENV))..."
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@$(PYTHON) -c "import numpy; print(f'NumPy: {numpy.__version__}')"
	@$(PYTHON) -c "import scipy; print(f'SciPy: {scipy.__version__}')"
	@$(PYTHON) -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
	@$(PYTHON) -c "from src.models.rcgnn import RCGNN; print('RC-GNN: OK')"
	@echo "[DONE] Environment ready"

#=============================================================================
# Data Generation
#=============================================================================

data: data-synth data-uci ## Generate all datasets

data-synth: ## Generate synthetic dataset
	@echo "[INFO] Generating synthetic dataset..."
	@mkdir -p $(SYNTH)
	$(PYTHON) scripts/synth_bench.py \
		--d 15 \
		--edges 30 \
		--n_envs 3 \
		--missing_type mcar \
		--missing_rate 0.2 \
		--output $(SYNTH) \
		--seed 42
	@echo "[DONE] Synthetic data at $(SYNTH)"

data-uci: ## Download and prepare UCI Air Quality
	@echo "[INFO] Preparing UCI Air Quality dataset..."
	@mkdir -p $(UCI_AIR)
	$(PYTHON) scripts/download_and_convert_uci_safe.py --out $(UCI_AIR)
	@echo "[DONE] UCI Air data at $(UCI_AIR)"

data-corrupted: ## Generate corrupted UCI variants
	@echo "[INFO] Generating corruption variants..."
	$(PYTHON) scripts/generate_uci_air_c.py
	@echo "[DONE] Corrupted datasets at $(UCI_AIR_C)"

#=============================================================================
# Training - Local
#=============================================================================

train: ## Train on compound_full (default)
	@echo "[INFO] Training RC-GNN on compound_full..."
	@mkdir -p $(ARTIFACTS) $(LOGS)
	$(PYTHON) scripts/train_rcgnn_unified.py \
		--data_dir $(UCI_AIR_C)/compound_full \
		--output_dir $(ARTIFACTS)/compound_full \
		--epochs 100
	@echo "[DONE] Training complete"

train-quick: ## Quick training test (10 epochs)
	@echo "[INFO] Quick training test..."
	@mkdir -p $(ARTIFACTS)
	$(PYTHON) scripts/train_rcgnn_unified.py \
		--data_dir $(UCI_AIR_C)/compound_full \
		--output_dir $(ARTIFACTS)/quick_test \
		--epochs 10
	@echo "[DONE] Quick test complete"

train-dataset: ## Train on specific dataset (usage: make train-dataset DS=mcar_20)
	@echo "[INFO] Training on $(DS)..."
	@mkdir -p $(ARTIFACTS)
	$(PYTHON) scripts/train_rcgnn_unified.py \
		--data_dir $(UCI_AIR_C)/$(DS) \
		--output_dir $(ARTIFACTS)/$(DS) \
		--epochs 100
	@echo "[DONE] Training on $(DS) complete"

train-all: ## Train on all 12 datasets (local, sequential)
	@echo "[INFO] Training on all datasets (this will take a while)..."
	@for ds in clean_full compound_full compound_mnar_bias extreme mcar_20 mcar_40 \
		mnar_structural moderate noise_0.5 regimes_5 sensor_failure; do \
		echo "[INFO] Training on $$ds..."; \
		$(PYTHON) scripts/train_rcgnn_unified.py \
			--data_dir $(UCI_AIR_C)/$$ds \
			--output_dir $(ARTIFACTS)/$$ds \
			--epochs 100; \
	done
	@echo "[DONE] All training complete"

#=============================================================================
# Training - Sapelo HPC
#=============================================================================

submit: ## Submit training job to Sapelo (from local)
	@echo "[INFO] Submitting job to Sapelo..."
	ssh sapelo2 "cd /scratch/aoo29179/rcgnn && sbatch slurm/train_unified_gpu.sh"

submit-local: ## Submit from Sapelo login node (on Sapelo)
	@mkdir -p $(LOGS)
	sbatch slurm/train_unified_gpu.sh

status: ## Check Sapelo job status
	@ssh sapelo2 "squeue -u aoo29179" 2>/dev/null || squeue -u $$USER

logs: ## View latest Sapelo job logs
	@echo "[INFO] Latest log files:"
	@ls -lt $(LOGS)/*.out 2>/dev/null | head -5 || echo "No logs found in $(LOGS)/"

cancel: ## Cancel all Sapelo jobs
	@ssh sapelo2 "scancel -u aoo29179" 2>/dev/null || scancel -u $$USER

sync-sapelo: ## Sync code to Sapelo (from local)
	@echo "[INFO] Syncing to Sapelo..."
	rsync -avz --exclude='.git' --exclude='artifacts*' --exclude='__pycache__' \
		--exclude='*.pyc' --exclude='.venv' --exclude='data/interim' \
		./ sapelo2:/scratch/aoo29179/rcgnn/
	@echo "[DONE] Code synced to Sapelo"

sync-from-sapelo: ## Download results from Sapelo (to local)
	@echo "[INFO] Downloading results from Sapelo..."
	rsync -avz sapelo2:/scratch/aoo29179/rcgnn/artifacts/ ./artifacts_sapelo/
	@echo "[DONE] Results downloaded to artifacts_sapelo/"

#=============================================================================
# Evaluation
#=============================================================================

evaluate: ## Run comprehensive evaluation
	@echo "[INFO] Running comprehensive evaluation..."
	@mkdir -p $(ARTIFACTS)
	$(PYTHON) scripts/comprehensive_evaluation.py \
		--artifacts-dir $(ARTIFACTS) \
		--data-dir $(DATA_DIR) \
		--output $(ARTIFACTS)/evaluation_report.json
	@echo "[DONE] Evaluation complete"

results: ## Show results summary
	@echo "================================================================"
	@echo "  RC-GNN Results Summary"
	@echo "================================================================"
	@if [ -f $(ARTIFACTS)/evaluation_report.json ]; then \
		$(PYTHON) -c "import json; r=json.load(open('$(ARTIFACTS)/evaluation_report.json')); \
			print('Ground Truth Metrics:'); \
			[print(f\"  {d['Corruption']}: SHD={d['SHD']}, F1={d['Directed_F1']:.3f}\") \
			for d in r.get('ground_truth', [])[:6]]" 2>/dev/null || \
		echo "  Run 'make evaluate' first"; \
	else \
		echo "  No results found. Run 'make evaluate' first."; \
	fi

#=============================================================================
# Testing
#=============================================================================

test: ## Run all tests
	@echo "[INFO] Running tests..."
	$(PYTHON) -m pytest tests/ -v
	@echo "[DONE] Tests complete"

test-quick: ## Run quick smoke tests
	@echo "[INFO] Running smoke tests..."
	$(PYTHON) -m pytest tests/test_synth_smoke.py tests/test_training_step.py -v

test-import: ## Test that all modules import correctly
	@echo "[INFO] Testing imports..."
	@$(PYTHON) -c "from src.models.rcgnn import RCGNN; print('RCGNN: OK')"
	@$(PYTHON) -c "from src.models.structure import StructureLearner; print('StructureLearner: OK')"
	@$(PYTHON) -c "from src.models.encoders import EncoderS; print('Encoders: OK')"
	@echo "[DONE] All imports successful"

#=============================================================================
# Cleanup
#=============================================================================

clean: ## Clean generated files (keep data)
	@echo "[INFO] Cleaning artifacts..."
	rm -rf $(ARTIFACTS)/*
	rm -rf __pycache__ src/__pycache__ scripts/__pycache__
	rm -rf src/*/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "[DONE] Cleanup complete"

clean-logs: ## Clean log files
	rm -rf $(LOGS)/*.out $(LOGS)/*.err

clean-all: clean clean-logs ## Clean everything except data
	@echo "[DONE] Full cleanup complete"

#=============================================================================
# Information
#=============================================================================

info: ## Show project information
	@echo "================================================================"
	@echo "  RC-GNN Project Information"
	@echo "================================================================"
	@echo ""
	@echo "Environment: $(ENV)"
	@echo "Python: $(PYTHON)"
	@echo ""
	@echo "Directories:"
	@echo "  Data:      $(DATA_DIR)"
	@echo "  Artifacts: $(ARTIFACTS)"
	@echo "  Configs:   $(CONFIGS)"
	@echo ""
	@echo "Datasets available:"
	@ls -1 $(UCI_AIR_C) 2>/dev/null | head -12 || echo "  None (run 'make data-corrupted')"

version: ## Show version information
	@$(PYTHON) --version
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: not installed"
	@$(PYTHON) -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "NumPy: not installed"
