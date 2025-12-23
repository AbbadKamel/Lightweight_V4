# N2KShield AUROC Improvement Experiments

## Current Baseline
- **AUROC**: 0.785
- **Target**: 0.90+

## 5 Experiments to Run

| # | Experiment | Change | Expected Gain | Time |
|---|------------|--------|---------------|------|
| 1 | More Data | WINDOW_STEP_TRAIN=1 | +3-5% | 2h |
| 2 | Extended Training | MAX_EPOCHS=500 | +2-3% | 3h |
| 3 | Threshold Tuning | Grid search P_LOSS, P_TIME, R_SIGNAL | +3-5% | 30min |
| 4 | Scenario C | Mean+Std (30 features) | +2-3% | 2h |
| 5 | Architecture | Larger filters, dropout | +2% | 3h |

## Quick Start (Cloud Server)

```bash
# Connect to server
ssh user@N315L-G17G01.ressource.unicaen.fr

# Clone repo
git clone https://github.com/AbbadKamel/Lightweight_V4.git
cd Lightweight_V4

# Run single experiment in screen
screen -S exp1
./experiments/exp1_more_data.sh
# Ctrl+A D to detach

# Or run all 5 in parallel
./experiments/run_all_experiments.sh
```

## Screen Commands

```bash
screen -ls                 # List all screens
screen -r exp1_data        # Attach to experiment 1
Ctrl+A D                   # Detach from screen
screen -X -S exp1 quit     # Kill screen
```

## Monitor Progress

```bash
tail -f experiments/exp1_data.log
```

## Results Location
- `Phase3/results/` - Evaluation metrics
- `Phase3/figures/` - Plots
- `Phase2/models/` - Trained models
