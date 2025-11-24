# N2KShield: Complete Step-by-Step Implementation Plan
**Project**: NMEA2000 Intrusion Detection System  
**Based on**: CANShield Methodology (IEEE IoT 2023)  
**Approach**: Build incrementally, validate each step, adapt to OUR data  
**Date Started**: November 24, 2025

---

## 🎯 **PROJECT GOAL**

Build a complete intrusion detection pipeline for NMEA2000 maritime CAN bus:
1. **Preprocess** NMEA2000 data (decode, clean, structure)
2. **Train** CNN-Autoencoder models on normal navigation data
3. **Detect** anomalies using ensemble method
4. **Evaluate** performance with metrics and visualizations

**Key Principle**: We follow CANShield methodology BUT adapt parameters to OUR NMEA2000 data characteristics!

---

## 📊 **CURRENT ASSETS**

### What We Have:
- ✅ **3 million NMEA2000 frames** from real boat navigation
- ✅ **N2K Decoder** (decode_N2K/n2k_decoder.py) - 42 PGNs supported
- ✅ **CANShield reference** code and paper
- ✅ **GPU Server** access for training

### What We DON'T Have Yet:
- ❌ Understanding of OUR data characteristics
- ❌ Optimal parameters for NMEA2000 (not automotive CAN)
- ❌ Any code/models/results

---

## 🗺️ **COMPLETE WORKFLOW (10 PHASES)**

---

# PHASE 0: DATA ANALYSIS & PARAMETER DETERMINATION
**Goal**: Understand OUR data before making ANY decisions  
**Duration**: 1 day  
**Output**: Data characteristics report

## Step 0.1: Analyze Sample Data (FIRST PRIORITY!)
**What**: Run decoder on 1 file (Frame(0-9999).txt = 10,000 frames)

**Actions**:
1. Decode 10K frames using existing n2k_decoder.py
2. Count unique PGNs present
3. Count unique signals extracted
4. Analyze message frequencies
5. Check for missing/corrupt data
6. Identify most frequent signals

**Questions to Answer**:
- How many unique PGNs in our data? (CANShield had ~10 CAN IDs)
- How many unique signals? (CANShield used 20)
- What are message frequencies? (affects sampling periods)
- Which signals are critical for maritime safety?
- Any fast-packet messages? (NMEA2000 specific)
- Signal update rates? (100ms, 1s, 10s?)

**Validation**: 
✅ Can decode successfully  
✅ Have enough signal variety  
✅ Data quality is good (no major corruption)

---

## Step 0.2: Determine Signal Selection Strategy
**What**: Decide which signals to use (DON'T blindly copy CANShield's 20!)

**Analysis**:
1. List all decoded signals by frequency
2. Identify critical maritime signals:
   - Position (lat/lon) - safety critical
   - Heading/Course - navigation critical
   - Speed - maneuvering critical
   - Depth - grounding prevention
   - Wind - weather/stability
3. Check signal correlations (preliminary)
4. Consider signal update rates

**Decision Points**:
- **Minimum signals needed**: Cover all safety-critical functions
- **Maximum signals feasible**: Model complexity vs performance
- **Our choice**: Based on data, not CANShield default

**Output**: 
- List of candidate signals (maybe 10? maybe 30? Let data decide!)
- Justification for each signal

---

## Step 0.3: Determine Temporal Parameters
**What**: Decide window size and sampling periods based on OUR data

**Analysis**:
1. Measure typical message intervals per PGN
2. Identify fast-changing signals (heading, speed)
3. Identify slow-changing signals (GPS, depth)
4. Calculate how many timesteps needed to capture patterns

**CANShield used**:
- Window size: 50 timesteps
- Sampling: [1, 5, 10]

**Our analysis should determine**:
- Is 50 timesteps enough for NMEA2000?
- Do our signals update faster/slower than automotive CAN?
- What sampling periods make sense for boat navigation?

**Output**:
- Window size (w) recommendation with justification
- Sampling periods (T1, T2, T3) with justification

---

## Step 0.4: Create Data Characteristics Report
**What**: Document findings in ONE clear file

**Report Contents**:
```
1. Dataset Overview
   - Total frames analyzed
   - Date/time range
   - Unique PGNs found
   
2. Signal Analysis
   - Total unique signals
   - Signal frequency distribution
   - Update rates per signal
   - Missing data percentage
   
3. Recommended Parameters
   - Number of signals: X (because...)
   - Signal list: [list with justification]
   - Window size: Y (because...)
   - Sampling periods: [T1, T2, T3] (because...)
   
4. Differences from CANShield
   - What we keep same
   - What we change
   - Why we change it
```

**Validation**:
✅ We understand our data  
✅ Parameters are data-driven, not copied  
✅ Justification for every decision  

---

# PHASE 1: PROJECT SETUP
**Goal**: Create organized project structure  
**Duration**: 0.5 day  
**Output**: Clean folder structure + Git repository

## Step 1.1: Create Project Structure
**What**: Build N2KShield folder hierarchy (ONE STEP, NOT 14 FILES!)

**Actions**:
```bash
Create:
N2KShield/
├── data/
│   ├── raw/           (link to NMEA2000 folder)
│   ├── decoded/       (empty)
│   ├── preprocessed/  (empty)
│   └── scalers/       (empty)
├── results/           (empty)
└── temp/              (for testing)
```

**Validation**:
✅ Folder structure makes sense  
✅ You approve before proceeding  

---

## Step 1.2: Initialize Git Repository
**What**: Version control (AFTER you approve structure)

**Actions**:
1. Show you the .gitignore content FIRST
2. You approve
3. Create .gitignore
4. git init
5. Show you what will be committed
6. You approve
7. Make initial commit

**Validation**:
✅ .gitignore protects large files  
✅ Only essential files committed  

---

# PHASE 2: DATA PREPROCESSING IMPLEMENTATION
**Goal**: Build decoder and preprocessing pipeline  
**Duration**: 2-3 days  
**Output**: Decoded and preprocessed data ready for training

## Step 2.1: Create Simple Data Loader
**What**: ONE Python script to decode NMEA2000 frames

**File**: `decode_nmea2000.py` (standalone script, not complex structure)

**Functionality**:
```python
# Simple script that:
1. Loads Frame(0-9999).txt
2. Uses n2k_decoder.py to decode
3. Saves to CSV
4. Prints statistics
```

**Testing**:
- Run on 1 file (10K frames)
- Verify output
- Check statistics
- You approve before proceeding

**Validation**:
✅ Decoding works correctly  
✅ Output format is correct  
✅ No errors or data loss  

---

## Step 2.2: Decode Sample Dataset
**What**: Decode more files to get enough data for initial testing

**Actions**:
1. Decide sample size (e.g., 10 files = 100K frames)
2. Run decoder on sample
3. Save decoded data
4. Verify quality

**Output**: 
- decoded_sample.csv (~100K decoded messages)
- Statistics report

**Validation**:
✅ Enough data for testing  
✅ Good quality (low corruption rate)  

---

## Step 2.3: Signal Selection & Clustering
**What**: ONE script to select and cluster signals

**File**: `select_signals.py`

**Functionality**:
```python
1. Load decoded data
2. Calculate correlation matrix
3. Perform hierarchical clustering
4. Visualize correlations
5. Save selected signals list
```

**Testing**:
- Run on sample data
- Generate correlation heatmap
- You review and approve signal selection

**Output**:
- selected_signals.txt (list of N signals)
- correlation_matrix.png (visualization)

**Validation**:
✅ Correlation matrix makes sense  
✅ Clustering is logical  
✅ Signal selection justified  

---

## Step 2.4: Data Queue Generation
**What**: ONE script to create time-series data queues

**File**: `create_data_queue.py`

**Functionality**:
```python
1. Load decoded data
2. Create time-series structure (rows=time, cols=signals)
3. Apply forward-filling for missing values
4. Save data queue
```

**Testing**:
- Run on sample
- Verify queue structure
- Check forward-fill logic
- You approve

**Output**:
- data_queue.csv (structured time-series)

**Validation**:
✅ Queue structure is correct  
✅ Missing data handled properly  

---

## Step 2.5: Multi-View Generation
**What**: ONE script to create different temporal views

**File**: `create_multiviews.py`

**Functionality**:
```python
1. Load data queue
2. Create view D1 (sampling period T1)
3. Create view D2 (sampling period T2)
4. Create view D3 (sampling period T3)
5. Save as numpy arrays
```

**Testing**:
- Generate views from sample
- Verify dimensions
- Check sampling logic
- You approve

**Output**:
- view_sp1.npy
- view_sp5.npy
- view_sp10.npy

**Validation**:
✅ Views have correct dimensions  
✅ Sampling periods are correct  

---

## Step 2.6: Min-Max Scaling
**What**: ONE script to scale data to [0,1]

**File**: `scale_data.py`

**Functionality**:
```python
1. Load views
2. Compute min/max per signal (from training data only!)
3. Scale all views
4. Save scaler for later use
```

**Testing**:
- Scale sample views
- Verify range [0,1]
- You approve

**Output**:
- scaled_views (npy files)
- scaler.pkl (for future use)

**Validation**:
✅ All values in [0,1]  
✅ No data leakage from test set  

---

# PHASE 3: MODEL IMPLEMENTATION
**Goal**: Build CNN-Autoencoder exactly as CANShield  
**Duration**: 1 day  
**Output**: Model architecture ready for training

## Step 3.1: Build CNN-Autoencoder
**What**: ONE Python file with model architecture

**File**: `cnn_autoencoder.py`

**Functionality**:
```python
def build_autoencoder(time_step, num_signals):
    """
    CNN-AE exactly as CANShield Section V-B1
    Input: (time_step, num_signals, 1)
    Output: Reconstructed (time_step, num_signals, 1)
    """
    # Encoder: Conv2D(32,16,16) with LeakyReLU
    # Decoder: Conv2D(16,16,32) with LeakyReLU
    # Output: Sigmoid
    return model
```

**Testing**:
- Create model
- Print summary
- Verify architecture matches CANShield
- Test on dummy data
- You approve

**Validation**:
✅ Architecture matches paper  
✅ Input/output shapes correct  
✅ Parameter count reasonable  

---

## Step 3.2: Test Model on Sample
**What**: Quick test to ensure model works

**File**: `test_model.py`

**Actions**:
1. Load ONE scaled view
2. Create model
3. Train for 5 epochs (quick test!)
4. Check loss decreases
5. Visualize reconstruction

**Output**:
- Training curve plot
- Sample reconstruction visualization

**Validation**:
✅ Model trains without errors  
✅ Loss decreases (learning something)  
✅ Reconstructions look reasonable  

---

# PHASE 4: TRAINING PIPELINE
**Goal**: Train all 3 AE models with transfer learning  
**Duration**: 2-3 days (including actual training time)  
**Output**: 3 trained models

## Step 4.1: Train First Model (AE₁)
**What**: Train on view with sampling period T1

**File**: `train_ae.py` (single training script)

**Actions**:
1. Load view_sp1 (training + validation split)
2. Create AE model
3. Train for 100 epochs with early stopping
4. Save best model
5. Plot training curves
6. You review results

**Output**:
- ae_sp1.h5 (trained model)
- training_history_sp1.png
- validation metrics

**Validation**:
✅ Training converges  
✅ Validation loss stabilizes  
✅ No overfitting  

---

## Step 4.2: Train Second Model (AE₂) with Transfer Learning
**What**: Initialize with AE₁ weights, fine-tune on view sp2

**Actions**:
1. Load ae_sp1.h5
2. Load view_sp2
3. Fine-tune for ~70 epochs
4. Save ae_sp2.h5
5. Compare training time (should be faster!)

**Output**:
- ae_sp2.h5
- training_history_sp2.png

**Validation**:
✅ Transfer learning works (faster convergence)  
✅ Model performs well on sp2 view  

---

## Step 4.3: Train Third Model (AE₃) with Transfer Learning
**What**: Initialize with AE₂ weights, fine-tune on view sp3

**Actions**:
1. Load ae_sp2.h5
2. Load view_sp3
3. Fine-tune for ~70 epochs
4. Save ae_sp3.h5

**Output**:
- ae_sp3.h5
- training_history_sp3.png

**Validation**:
✅ All 3 models trained  
✅ Transfer learning reduced training time  

---

# PHASE 5: THRESHOLD SELECTION
**Goal**: Determine optimal thresholds for detection  
**Duration**: 1 day  
**Output**: Threshold values for each model

## Step 5.1: Compute Reconstruction Losses
**What**: Get reconstruction losses on NORMAL validation data

**File**: `compute_losses.py`

**Actions**:
1. Load validation data (normal only)
2. For each AE model:
   - Reconstruct inputs
   - Compute absolute loss per signal per timestep
3. Save losses

**Output**:
- reconstruction_losses.npy (3D array)

**Validation**:
✅ Losses computed correctly  

---

## Step 5.2: Three-Step Threshold Selection
**What**: Implement CANShield Algorithm 1

**File**: `select_thresholds.py`

**Functionality**:
```python
# Step 1: Signal-wise reconstruction threshold (R_Loss)
# Step 2: Signal-wise timestep violations (R_Time)
# Step 3: Overall signal violations (R_Signal)
# Grid search over p, q, r percentiles
```

**Actions**:
1. Run grid search (p: 90-99%, q: 95-99%, r: 95-99%)
2. Optimize for F1 score with FPR < 1%
3. Save optimal thresholds
4. Visualize threshold impact

**Output**:
- optimal_thresholds.json
- threshold_analysis.png

**Validation**:
✅ FPR on normal data < 1%  
✅ Thresholds make sense  

---

# PHASE 6: ENSEMBLE DETECTION
**Goal**: Combine 3 models for final detection  
**Duration**: 0.5 day  
**Output**: Ensemble detector

## Step 6.1: Implement Ensemble Detector
**What**: ONE script to run all 3 models and combine scores

**File**: `ensemble_detector.py`

**Functionality**:
```python
def detect_anomaly(data_queue):
    # Create 3 views from queue
    # Run each AE, get anomaly score
    # Average scores (ensemble)
    # Compare to threshold
    # Return: is_attack (True/False)
```

**Testing**:
- Test on normal validation data
- Compute FPR
- You approve

**Validation**:
✅ Ensemble works  
✅ FPR acceptable  

---

# PHASE 7: EVALUATION & METRICS
**Goal**: Comprehensive performance evaluation  
**Duration**: 1 day  
**Output**: Metrics and comparison tables

## Step 7.1: Compute All Metrics
**What**: Calculate AUROC, AUPRC, F1, etc.

**File**: `evaluate_models.py`

**Metrics**:
- AUROC (Area Under ROC)
- AUPRC (Area Under PR Curve)
- Precision
- Recall
- F1 Score
- FPR

**Compare**:
- Individual AE models (sp1, sp2, sp3)
- Ensemble model
- Baseline (mean absolute loss)

**Output**:
- metrics_comparison.csv
- roc_curves.png
- pr_curves.png

**Validation**:
✅ Metrics computed correctly  
✅ Ensemble outperforms individuals  

---

# PHASE 8: VISUALIZATION
**Goal**: Create publication-ready plots  
**Duration**: 1 day  
**Output**: All figures for paper

## Step 8.1: Create All Visualizations
**What**: ONE script to generate all plots

**File**: `create_visualizations.py`

**Plots**:
1. Correlation matrix (before/after clustering)
2. Signal clustering dendrogram
3. Training curves (all models)
4. Reconstruction loss distributions
5. ROC curves (all models)
6. Precision-Recall curves
7. Anomaly score timeline
8. Model comparison bar chart

**Output**:
- results/figures/ (all PNG and PDF)

**Validation**:
✅ All plots clear and interpretable  
✅ Publication quality  

---

# PHASE 9: FULL DATASET PROCESSING
**Goal**: Scale to all 3M frames  
**Duration**: 3-5 days (processing + training time)  
**Output**: Final models trained on complete dataset

## Step 9.1: Decode Full Dataset
**What**: Process all 300+ Frame files

**Actions**:
1. Run decoder on ALL files (batch processing)
2. Monitor for errors
3. Save complete decoded dataset

**Output**:
- decoded_full.csv (~3M decoded messages)

**Note**: May take hours/days depending on processing speed

---

## Step 9.2: Full Preprocessing
**What**: Run entire preprocessing pipeline on full data

**Actions**:
1. Signal selection (verify on full data)
2. Data queue generation
3. Multi-view creation
4. Scaling

**Output**:
- Complete preprocessed dataset

---

## Step 9.3: Final Model Training
**What**: Retrain all 3 models on complete dataset

**Actions**:
1. Train AE₁ on full data
2. Transfer learn to AE₂
3. Transfer learn to AE₃
4. Save final models

**Output**:
- Final trained models (may take 1-2 days on GPU)

---

# PHASE 10: ATTACK SYNTHESIS (FUTURE)
**Goal**: Create NMEA2000-specific synthetic attacks  
**Duration**: TBD  
**Output**: Attack dataset for testing

## Step 10.1: Design NMEA2000 Attacks
**What**: Define maritime-specific attacks

**Attack Types**:
1. Position Spoofing (GPS manipulation)
2. Heading Manipulation
3. Speed Falsification
4. Depth Sensor Attack
5. Wind Data Attack

**Implementation**: Future work after main pipeline validated

---

# 📋 **VALIDATION CHECKPOINTS**

After each phase, we verify:

### Phase 0 Checklist:
- [ ] Data decoded successfully
- [ ] Signal characteristics understood
- [ ] Parameters determined from data (not copied)
- [ ] You approve all parameter choices

### Phase 2 Checklist:
- [ ] All preprocessing scripts work
- [ ] Data quality verified
- [ ] Output formats correct
- [ ] You approve each script before next

### Phase 4 Checklist:
- [ ] All 3 models trained
- [ ] Transfer learning works
- [ ] Training curves look good
- [ ] You approve results

### Phase 7 Checklist:
- [ ] Metrics comparable to CANShield
- [ ] Ensemble performs best
- [ ] FPR < 1% on normal data

---

# 🎯 **SUCCESS CRITERIA**

## Minimum Viable Product (MVP):
1. ✅ Decode all NMEA2000 data
2. ✅ Train 3 CNN-AE models
3. ✅ Achieve FPR < 1% on normal data
4. ✅ Ensemble detector works
5. ✅ Basic metrics computed

## Extended Goals:
1. ⭐ Performance comparable to CANShield (AUROC > 0.95)
2. ⭐ All visualizations publication-ready
3. ⭐ Synthetic attacks tested
4. ⭐ IEEE paper ready

---

# ⚠️ **KEY PRINCIPLES**

1. **ONE STEP AT A TIME**: Complete and validate before next
2. **YOU APPROVE**: Every file, every decision
3. **DATA-DRIVEN**: Parameters from OUR data, not blindly copied
4. **PRINT & CHECK**: Always print results and verify logic
5. **INCREMENTAL**: Start small (sample), then scale

---

# 📅 **ESTIMATED TIMELINE**

- Phase 0: 1 day (analyze data)
- Phase 1: 0.5 day (setup)
- Phase 2: 2-3 days (preprocessing)
- Phase 3: 1 day (model implementation)
- Phase 4: 2-3 days (training on sample)
- Phase 5: 1 day (thresholds)
- Phase 6: 0.5 day (ensemble)
- Phase 7: 1 day (evaluation)
- Phase 8: 1 day (visualization)
- Phase 9: 3-5 days (full dataset)

**Total**: ~2-3 weeks for complete pipeline

---

# 🚀 **NEXT ACTION**

**IMMEDIATE**: Start Phase 0, Step 0.1
- Decode ONE file (Frame(0-9999).txt)
- Analyze what we actually have
- Make data-driven decisions

**WAITING FOR YOUR APPROVAL TO BEGIN!**

---

**Document Version**: 1.0  
**Last Updated**: November 24, 2025  
**Status**: Ready to Start Phase 0
