#!/bin/bash
# ============================================================================
# N2KShield AUROC Improvement Experiments - Run All in Parallel Screens
# ============================================================================
# Usage: ./run_all_experiments.sh
# Server: N315L-G17G01.ressource.unicaen.fr
# ============================================================================

echo "=============================================="
echo "N2KSHIELD EXPERIMENTS - LAUNCHING ALL IN SCREENS"
echo "=============================================="
echo ""

# Create results directory
mkdir -p ../Phase3/results/experiments

# ============================================================================
# EXPERIMENT 1: Baseline with more training data (WINDOW_STEP=1)
# ============================================================================
echo "Starting Experiment 1: More Training Data..."
screen -dmS exp1_data -L -Logfile exp1_data.log bash -c '
    echo "=== EXPERIMENT 1: More Training Data (WINDOW_STEP=1) ===" 
    echo "Changes: WINDOW_STEP_TRAIN=1 (10x more training samples)"
    cd ../Phase1/scripts
    python prepare_training_datasets.py
    cd ../../Phase2/scripts
    python train_cascade.py
    python calculate_thresholds_scenario.py main
    cd ../../Phase3/scripts
    python evaluate_complete.py
    echo "=== EXPERIMENT 1 COMPLETE ==="
    exec bash
'

# ============================================================================
# EXPERIMENT 2: Extended Training (500 epochs)
# ============================================================================
echo "Starting Experiment 2: Extended Training..."
screen -dmS exp2_epochs -L -Logfile exp2_epochs.log bash -c '
    echo "=== EXPERIMENT 2: Extended Training (500 epochs) ==="
    echo "Changes: MAX_EPOCHS=500, EARLY_STOPPING_PATIENCE=30"
    # This requires config change - will use exp2 config
    cd ../Phase2/scripts
    python train_cascade.py --config exp2
    python calculate_thresholds_scenario.py main
    cd ../../Phase3/scripts
    python evaluate_complete.py
    echo "=== EXPERIMENT 2 COMPLETE ==="
    exec bash
'

# ============================================================================
# EXPERIMENT 3: Three-Step Threshold Tuning
# ============================================================================
echo "Starting Experiment 3: Three-Step Tuning..."
screen -dmS exp3_threshold -L -Logfile exp3_threshold.log bash -c '
    echo "=== EXPERIMENT 3: Three-Step Threshold Grid Search ==="
    echo "Changes: Grid search P_LOSS, P_TIME, R_SIGNAL parameters"
    cd ../Phase3/scripts
    python grid_search_three_step.py
    echo "=== EXPERIMENT 3 COMPLETE ==="
    exec bash
'

# ============================================================================
# EXPERIMENT 4: Scenario C (Mean + Std = 30 features)
# ============================================================================
echo "Starting Experiment 4: Scenario C..."
screen -dmS exp4_scenarioC -L -Logfile exp4_scenarioC.log bash -c '
    echo "=== EXPERIMENT 4: Scenario C (Mean + Std) ==="
    echo "Changes: Using 30 features instead of 60"
    cd ../Phase1/scripts
    python prepare_training_datasets_scenario.py C_mean_std
    cd ../../Phase2/scripts
    python train_cascade_scenario.py C_mean_std
    python calculate_thresholds_scenario.py C_mean_std
    cd ../../Phase3/scripts
    python compare_all_scenarios.py
    echo "=== EXPERIMENT 4 COMPLETE ==="
    exec bash
'

# ============================================================================
# EXPERIMENT 5: Architecture Tuning (Larger model)
# ============================================================================
echo "Starting Experiment 5: Architecture..."
screen -dmS exp5_arch -L -Logfile exp5_arch.log bash -c '
    echo "=== EXPERIMENT 5: Architecture Tuning ==="
    echo "Changes: Filters [64,32,32,32,64], Dropout 0.2"
    # This requires models.py modification
    cd ../Phase2/scripts
    python train_cascade.py --config exp5_arch
    python calculate_thresholds_scenario.py main
    cd ../../Phase3/scripts
    python evaluate_complete.py
    echo "=== EXPERIMENT 5 COMPLETE ==="
    exec bash
'

echo ""
echo "=============================================="
echo "ALL 5 EXPERIMENTS LAUNCHED IN SCREEN SESSIONS"
echo "=============================================="
echo ""
echo "To check running screens:   screen -ls"
echo "To attach to experiment 1:  screen -r exp1_data"
echo "To attach to experiment 2:  screen -r exp2_epochs"
echo "To attach to experiment 3:  screen -r exp3_threshold"
echo "To attach to experiment 4:  screen -r exp4_scenarioC"
echo "To attach to experiment 5:  screen -r exp5_arch"
echo ""
echo "To detach from screen:      Ctrl+A, then D"
echo "To view log:                tail -f exp1_data.log"
echo ""
