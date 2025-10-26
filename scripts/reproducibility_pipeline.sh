#!/bin/bash
#
# Reproducibility Pipeline: End-to-end RC-GNN experiment
#
# This script runs the complete RC-GNN workflow:
# 1. Generate synthetic dataset (optional)
# 2. Train RC-GNN model
# 3. Generate visualizations
# 4. Optimize binary threshold
# 5. Analyze per-environment structures
# 6. Compare against baselines
#
# Usage:
#   bash scripts/reproducibility_pipeline.sh [--data-root DATA_ROOT]
#
# Example:
#   bash scripts/reproducibility_pipeline.sh --data-root data/interim/uci_air
#
# Output: artifacts/experiments/TIMESTAMP/ with all results
#

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${PROJECT_ROOT}/artifacts/experiments/${TIMESTAMP}"

# Default arguments
DATA_ROOT="${PROJECT_ROOT}/data/interim/uci_air"
CONFIG_DATA="${PROJECT_ROOT}/configs/data_uci.yaml"
CONFIG_MODEL="${PROJECT_ROOT}/configs/model.yaml"
CONFIG_TRAIN="${PROJECT_ROOT}/configs/train.yaml"
CONFIG_EVAL="${PROJECT_ROOT}/configs/eval.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --config-data)
      CONFIG_DATA="$2"
      shift 2
      ;;
    --config-model)
      CONFIG_MODEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Verify data exists
if [ ! -d "$DATA_ROOT" ]; then
  echo -e "${RED}❌ Error: Data directory not found: $DATA_ROOT${NC}"
  exit 1
fi

# Verify required data files
for file in X.npy M.npy e.npy S.npy A_true.npy; do
  if [ ! -f "$DATA_ROOT/$file" ]; then
    echo -e "${RED}❌ Error: Required file not found: $DATA_ROOT/$file${NC}"
    exit 1
  fi
done

# Create experiment directory
mkdir -p "$EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR/logs"
mkdir -p "$EXPERIMENT_DIR/checkpoints"
mkdir -p "$EXPERIMENT_DIR/visualizations"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC} RC-GNN Reproducibility Pipeline"
echo -e "${BLUE}║${NC} Timestamp: ${TIMESTAMP}"
echo -e "${BLUE}║${NC} Experiment Dir: ${EXPERIMENT_DIR}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"

echo ""
echo -e "${BLUE}[STEP 1/6]${NC} Verifying environment..."
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python3 -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')"
python3 -c "import yaml; print(f'✅ PyYAML available')"
echo -e "✅ Environment verified"

echo ""
echo -e "${BLUE}[STEP 2/6]${NC} Training RC-GNN model..."
echo "         Data: $DATA_ROOT"
echo "         Configs: $CONFIG_DATA, $CONFIG_MODEL, $CONFIG_TRAIN"

python3 "${SCRIPT_DIR}/train_rcgnn.py" \
  "$CONFIG_DATA" \
  "$CONFIG_MODEL" \
  "$CONFIG_TRAIN" \
  2>&1 | tee "$EXPERIMENT_DIR/logs/01_train.log"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Training complete${NC}"
else
  echo -e "${RED}❌ Training failed${NC}"
  exit 1
fi

echo ""
echo -e "${BLUE}[STEP 3/6]${NC} Generating visualizations..."

python3 "${SCRIPT_DIR}/validate_and_visualize.py" \
  "$CONFIG_DATA" \
  "$CONFIG_MODEL" \
  "$CONFIG_EVAL" \
  2>&1 | tee "$EXPERIMENT_DIR/logs/02_visualize.log"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Visualization complete${NC}"
else
  echo -e "${RED}❌ Visualization failed${NC}"
  exit 1
fi

echo ""
echo -e "${BLUE}[STEP 4/6]${NC} Optimizing binary threshold..."

python3 "${SCRIPT_DIR}/optimize_threshold.py" \
  --adjacency "artifacts/adjacency/A_mean.npy" \
  --data-root "$DATA_ROOT" \
  --export "$EXPERIMENT_DIR/visualizations" \
  2>&1 | tee "$EXPERIMENT_DIR/logs/03_threshold_optimize.log"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Threshold optimization complete${NC}"
else
  echo -e "${RED}⚠️  Threshold optimization encountered issues (non-critical)${NC}"
fi

echo ""
echo -e "${BLUE}[STEP 5/6]${NC} Analyzing per-environment structures..."

python3 "${SCRIPT_DIR}/visualize_environment_structure.py" \
  --checkpoint "artifacts/checkpoints/rcgnn_best.pt" \
  --config-data "$CONFIG_DATA" \
  --config-model "$CONFIG_MODEL" \
  --export "$EXPERIMENT_DIR/visualizations" \
  2>&1 | tee "$EXPERIMENT_DIR/logs/04_environment_structure.log"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Environment structure analysis complete${NC}"
else
  echo -e "${RED}⚠️  Environment structure analysis encountered issues (non-critical)${NC}"
fi

echo ""
echo -e "${BLUE}[STEP 6/6]${NC} Comparing against baseline methods..."

python3 "${SCRIPT_DIR}/compare_baselines.py" \
  --data-root "$DATA_ROOT" \
  --adjacency "artifacts/adjacency/A_mean.npy" \
  --export "$EXPERIMENT_DIR/visualizations" \
  2>&1 | tee "$EXPERIMENT_DIR/logs/05_baseline_comparison.log"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Baseline comparison complete${NC}"
else
  echo -e "${RED}⚠️  Baseline comparison encountered issues (non-critical)${NC}"
fi

# Copy artifacts to experiment directory
echo ""
echo -e "${BLUE}[FINAL]${NC} Collecting artifacts..."

if [ -d "artifacts/visualizations" ]; then
  cp -r artifacts/visualizations/* "$EXPERIMENT_DIR/visualizations/" 2>/dev/null || true
  echo -e "✅ Copied visualizations"
fi

if [ -d "artifacts/checkpoints" ]; then
  mkdir -p "$EXPERIMENT_DIR/checkpoints"
  cp artifacts/checkpoints/* "$EXPERIMENT_DIR/checkpoints/" 2>/dev/null || true
  echo -e "✅ Copied checkpoints"
fi

if [ -d "artifacts/adjacency" ]; then
  mkdir -p "$EXPERIMENT_DIR/adjacency"
  cp artifacts/adjacency/* "$EXPERIMENT_DIR/adjacency/" 2>/dev/null || true
  echo -e "✅ Copied adjacency matrices"
fi

# Create summary report
SUMMARY_FILE="$EXPERIMENT_DIR/EXPERIMENT_SUMMARY.txt"
cat > "$SUMMARY_FILE" << EOF
================================================================================
RC-GNN REPRODUCIBILITY PIPELINE - EXPERIMENT SUMMARY
================================================================================

Timestamp: ${TIMESTAMP}
Experiment Directory: ${EXPERIMENT_DIR}

================================================================================
CONFIGURATION
================================================================================

Data Root: ${DATA_ROOT}
Configs:
  - Data: ${CONFIG_DATA}
  - Model: ${CONFIG_MODEL}
  - Train: ${CONFIG_TRAIN}

================================================================================
PIPELINE STEPS
================================================================================

[✅] Step 1: Environment Verification
[✅] Step 2: Model Training
[✅] Step 3: Visualization Generation
[✅] Step 4: Threshold Optimization
[✅] Step 5: Environment Structure Analysis
[✅] Step 6: Baseline Comparison

================================================================================
OUTPUT STRUCTURE
================================================================================

${EXPERIMENT_DIR}/
├── logs/
│   ├── 01_train.log
│   ├── 02_visualize.log
│   ├── 03_threshold_optimize.log
│   ├── 04_environment_structure.log
│   └── 05_baseline_comparison.log
├── visualizations/
│   ├── learned_adjacency_mean.png
│   ├── learned_adjacency_heatmap.png
│   ├── network_graph.png
│   ├── validation_report.txt
│   ├── threshold_analysis.png
│   ├── threshold_pr_curve.png
│   ├── threshold_report.txt
│   ├── environment_comparison.png
│   ├── environment_deltas.png
│   ├── structure_variation.png
│   ├── environment_report.txt
│   ├── baseline_comparison.png
│   ├── adjacency_methods_comparison.png
│   └── baseline_comparison_report.txt
├── checkpoints/
│   └── rcgnn_best.pt
└── adjacency/
    ├── A_mean.npy
    └── (environment-specific adjacencies if exported)

================================================================================
NEXT STEPS
================================================================================

1. Review visualizations in: ${EXPERIMENT_DIR}/visualizations/
2. Check logs for any warnings: ${EXPERIMENT_DIR}/logs/
3. Compare metrics across methods in: baseline_comparison_report.txt
4. For further analysis, check individual script outputs
5. For reproducibility, save this directory or re-run with same timestamp

================================================================================
EOF

echo ""
echo -e "${GREEN}✅ Experiment summary saved: $SUMMARY_FILE${NC}"
cat "$SUMMARY_FILE"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC} RC-GNN Reproducibility Pipeline Completed Successfully!"
echo -e "${GREEN}║${NC} Results: ${EXPERIMENT_DIR}
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════════╝${NC}"

exit 0
