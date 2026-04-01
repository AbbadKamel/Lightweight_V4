#!/bin/bash
# ============================================================================
# EXPERIMENT 1: More Training Data (WINDOW_STEP=1)
# ============================================================================
# WHAT: Generate 10x more training windows with maximum overlap
# WHY: More data → better generalization → higher AUROC
# CHANGE: WINDOW_STEP_TRAIN=1 (already in config.py)
# EXPECTED: +3-5% AUROC improvement
# ============================================================================

echo "=============================================="
echo "EXPERIMENT 1: MORE TRAINING DATA"
echo "=============================================="
echo "Config: WINDOW_STEP_TRAIN = 1 (was 10)"
echo "Expected windows: ~5000 (was ~500)"
echo "Expected AUROC gain: +3-5%"
echo "=============================================="

# Go to project root
cd "$(dirname "$0")/.."

# Step 1: Regenerate datasets with more overlap
echo ""
echo "[1/4] Regenerating training datasets..."
cd Phase1/scripts
python prepare_training_datasets.py
echo "✓ Datasets regenerated"

# Step 2: Retrain all 9 models
echo ""
echo "[2/4] Retraining models (this takes ~2 hours)..."
cd ../../Phase2/scripts
python train_cascade.py
echo "✓ Models retrained"

# Step 3: Recalculate thresholds
echo ""
echo "[3/4] Calculating new thresholds..."
python calculate_thresholds_scenario.py main
echo "✓ Thresholds calculated"

# Step 4: Evaluate and get AUROC
echo ""
echo "[4/4] Evaluating performance..."
cd ../../Phase3/scripts
python evaluate_complete.py
echo "✓ Evaluation complete"

echo ""
echo "=============================================="
echo "EXPERIMENT 1 COMPLETE!"
echo "=============================================="
echo "Check results in: Phase3/results/"
