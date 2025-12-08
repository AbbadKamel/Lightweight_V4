# N2KShield: Simplified Implementation Plan
**Project**: NMEA2000 Intrusion Detection System  
**Based on**: CANShield Methodology (IEEE IoT 2023)  
**Date**: November 25, 2025  
**Total Duration**: 10-14 days

---

## 🎯 **PROJECT OVERVIEW**

Build a complete intrusion detection system for NMEA2000 maritime CAN bus:
1. **Explore** data characteristics and determine optimal parameters
2. **Preprocess** raw NMEA2000 frames into ML-ready windowed datasets
3. **Train** CNN-Autoencoder ensemble with multi-scale temporal analysis
4. **Evaluate** detection performance and create publication-ready results

**Key Principle**: Data-driven decisions based on OUR maritime data, not blindly copying automotive CAN parameters!

---

## 📊 **CURRENT STATUS**

### ✅ Completed Work:
- Decoded 2,984,250 NMEA2000 frames → 654,013 messages (21.9% decode rate)
- Identified 9 working PGNs, extracted 19 unique signals
- Analyzed signal update rates (33ms - 501ms)
- Created time-aligned dataset (9,859 rows, 1-second intervals)
- Analyzed data quality & sparsity (33% rows too sparse)
- Generated correlation matrix and signal analysis
- Identified problematic signals (speed_ground: 0%, deviation: 0%)
- Recommended preprocessing strategy (forward-fill 5s + drop sparse rows)

### 📝 Remaining Decisions:
- Finalize signal list (10-13 signals from 19 candidates)
- Determine window size (50 timesteps or different?)
- Choose sampling periods ([1s, 5s, 10s] or maritime-specific?)

---

# 🗺️ **4-PHASE IMPLEMENTATION PLAN**

---

# PHASE 0: DATA EXPLORATION & ANALYSIS
**Goal**: Understand OUR data and make data-driven parameter decisions  
**Duration**: 1 day (almost complete!)  
**Status**: 90% done, finalizing decisions

---

## Step 0.1: Decode All Frames ✅ DONE
**What**: Decode complete NMEA2000 dataset using Python decoder

**Completed Actions**:
- Decoded all 299 files (2.98M frames)
- Fixed numeric file sorting bug (chronological order)
- Extracted 654K messages across 9 PGNs
- Created trajectory visualization (85-minute boat journey)
- Generated statistics report

**Outputs**:
- `Phase0/results/decoded_frames.csv` (47MB, 654K messages)
- `Phase0/results/decoding_statistics.json` (PGN counts, signal coverage)
- `Phase0/results/boat_trajectory.html` (interactive GPS map)

**Key Findings**:
- 9 working PGNs: PositionRapid, Rudder, WindSpeed, COGSOG, Attitude, Heading, RateOfTurn, WaterDepth, Speed
- 19 signals extracted
- Time range: 15:05:29 to 16:30:23 (5,098.9 seconds)
- Location: North France coast, river/canal navigation

---

## Step 0.2: Analyze Signal Characteristics ✅ DONE
**What**: Understand signal update rates and temporal behavior

**Completed Actions**:
- Measured message intervals per PGN
- Analyzed update frequency distributions
- Identified fast vs slow sensors
- Determined optimal resampling period

**Outputs**:
- Signal update rate analysis
- Recommendations for time-alignment strategy

**Key Findings**:
- **Fast signals** (30-50ms): Position (33ms), Rudder (45ms), Wind (16-50ms)
- **Medium signals** (~100ms): Attitude, Heading, RateOfTurn, Depth (105ms)
- **Slow signals** (500ms): Speed (501ms)
- **Recommendation**: 1-second resampling captures 90-100% of updates

---

## Step 0.3: Data Quality & Sparsity Analysis ✅ DONE
**What**: Analyze time-aligned data quality to determine preprocessing strategy

**Completed Actions**:
- Created time-aligned dataset (1-second resampling)
- Analyzed sparsity distribution (values per row)
- Calculated correlation matrix on aligned data
- Identified zero-data and constant signals
- Determined preprocessing requirements

**Outputs**:
- `Phase0/results/decoded_frames_aligned.csv` (2MB, 9,859 rows)
- `Phase0/results/signal_correlation_matrix.png` (correlation heatmap)
- `Phase0/results/signal_selection_analysis.json` (detailed statistics)

**Key Findings**:
- **Sparsity**: Mean 12.9 signals/row, median 17/20
- **Zero-data signals**: speed_ground (0%), deviation (0%)
- **High coverage**: rudder_position (86%), lat/lon (78%), depth/attitude (73%)
- **Correlations**: cog↔heading (0.88), sog↔speed_water (0.94), yaw↔heading (-0.73)
- **ML-ready rows**: 67% have ≥10 signals, 49% have ≥18 signals

**Preprocessing Strategy**:
1. Remove 2 zero-data signals (speed_ground, deviation)
2. Forward-fill gaps (max 5 seconds)
3. Drop rows with < 8-10 signals
4. Result: Clean, dense dataset for training

---

## Step 0.4: Finalize Signal Selection ⏳ IN PROGRESS
**What**: Decide which signals to use for N2KShield IDS

**Analysis**:
Based on correlation matrix, coverage, and maritime importance:

**Signals to REMOVE (6)**:
1. `speed_ground` - 0% data
2. `deviation` - 0% data
3. `latitude` - constant within correlation analysis (or keep for GPS)
4. `longitude` - constant within correlation analysis (or keep for GPS)
5. `offset` - constant value (73.8% coverage but no variance)
6. `variation` - constant value (73.3% coverage but no variance)

**Signals to KEEP (13)**:
1. `depth` - Grounding prevention (73.8% coverage)
2. `rudder_angle_order` - Steering command (62.3% coverage)
3. `rudder_position` - Actual rudder state (86.0% coverage)
4. `cog` - Course over ground (51.7% coverage) - consider removing if redundant with heading
5. `sog` - Speed over ground (51.7% coverage)
6. `speed_water` - Speed through water (49.4% coverage)
7. `yaw` - Vessel rotation (73.3% coverage)
8. `pitch` - Vessel pitch angle (73.3% coverage)
9. `roll` - Vessel roll angle (73.3% coverage)
10. `heading` - Compass heading (73.3% coverage)
11. `rate_of_turn` - Turning rate (73.4% coverage)
12. `wind_speed` - Wind speed (74.5% coverage)
13. `wind_angle` - Wind direction (74.5% coverage)

**Redundancy Considerations**:
- **cog** vs **heading**: 0.88 correlation → Keep both or drop `cog`?
- **sog** vs **speed_water**: 0.94 correlation → Keep both or drop one?
- **yaw** vs **heading**: -0.73 correlation → Different enough to keep both

**Decision Options**:
- **Option A**: Keep all 13 signals (robust but some redundancy)
- **Option B**: Remove redundant pairs → 10-11 signals (leaner model)
- **Recommended**: Keep all 13, let model learn which are important

**Output** (to complete):
- Final signal list with justification
- Expected model input dimensions

---

## Step 0.5: Determine Temporal Parameters ⏳ PENDING
**What**: Decide window size and sampling periods based on signal dynamics

**Analysis Required**:

### Window Size Decision:
- **CANShield used**: 50 timesteps
- **Our consideration**: 
  - At 1-second sampling: 50 timesteps = 50 seconds
  - Boat maneuvers: Turn time? Acceleration time?
  - Need to capture: Rudder change → heading change → position change
- **Question**: Is 50 seconds enough to see complete maritime maneuvers?

### Sampling Periods Decision:
- **CANShield used**: [1s, 5s, 10s] for automotive
- **Our consideration**:
  - T1 (fine-grained): 1 second (matches our resampling)
  - T2 (medium): 5 seconds (capture slow changes)
  - T3 (coarse): 10 seconds (long-term trends)
- **Question**: Do maritime dynamics need different granularity?

**Recommended Approach**:
1. Start with CANShield parameters [1s, 5s, 10s] and w=50
2. Validate on our data during preprocessing
3. Adjust if needed based on results

**Output** (to complete):
- Window size: w timesteps
- Sampling periods: [T1, T2, T3] seconds
- Justification based on maritime dynamics

---

## Step 0.6: Create Phase 0 Report ⏳ PENDING
**What**: Document all findings and decisions in comprehensive report

**Report Contents**:
```
1. Dataset Overview
   - Total frames: 2,984,250
   - Decoded messages: 654,013 (21.9%)
   - Time range: 85 minutes
   - Location: North France coast
   - PGNs: 9 working
   - Signals: 19 extracted → 13 final

2. Signal Analysis
   - Update rates: 33ms (position) to 501ms (speed)
   - Coverage: 49% (speed_water) to 86% (rudder_position)
   - Correlations: Key redundancies identified
   - Removed: speed_ground, deviation (0% data)

3. Data Quality Findings
   - Time-aligned: 9,859 seconds
   - Sparsity: 67% rows have ≥10 signals
   - Preprocessing: Forward-fill (5s max) + drop sparse rows

4. Final Parameters
   - Signals: 13 (list with coverage %)
   - Window size: 50 timesteps
   - Sampling periods: [1s, 5s, 10s]
   - Resampling: 1-second intervals

5. Differences from CANShield
   - Signals: 13 vs CANShield's 20 (maritime-specific)
   - Update rates: Slower than automotive (33-501ms vs <10ms)
   - Data sparsity: More gaps than automotive CAN
   - Preprocessing: Added forward-fill strategy
```

**Output**:
- `Phase0/PHASE0_REPORT.md` (complete analysis documentation)
- Ready to proceed to Phase 1!

---

**Phase 0 Validation Checklist**:
- [ ] All 2.98M frames decoded successfully
- [ ] Signal characteristics understood (update rates, coverage)
- [ ] Data quality analyzed (sparsity, correlations)
- [ ] Final signal list decided and justified (10-13 signals)
- [ ] Temporal parameters determined (window size, sampling periods)
- [ ] Phase 0 report written and reviewed
- [ ] **You approve all decisions before Phase 1**

---

# PHASE 1: DATA PREPROCESSING
**Goal**: Transform raw NMEA2000 data into ML-ready windowed datasets  
**Duration**: 3-5 days  
**Input**: decoded_frames.csv (654K messages, 19 signals)  
**Output**: Scaled 3D arrays ready for CNN training

---

## Step 1.1: Load and Clean Raw Data
**What**: Load decoded data and perform initial quality control

**Script**: `Phase1/preprocess_step1_clean.py`

**Actions**:
1. Load `Phase0/results/decoded_frames.csv`
2. Remove zero-data signals (from Phase 0 decisions):
   - Drop `speed_ground` column
   - Drop `deviation` column
   - Remove other constant/rejected signals
3. Verify remaining signals match Phase 0 final list (13 signals)
4. Check for data corruption or anomalies
5. Print statistics (before/after)

**Outputs**:
- `Phase1/data/cleaned_signals.csv` (654K rows, 13 signals)
- `Phase1/logs/cleaning_stats.json`

**Validation**:
- Signal count matches Phase 0 decision
- No unexpected NaN patterns
- Coverage percentages verified

---

## Step 1.2: Time Alignment & Resampling
**What**: Create synchronized time-series with 1-second intervals

**Script**: `Phase1/preprocess_step2_align.py`

**Actions**:
1. Load cleaned signals
2. Parse timestamps (HH:MM:SS.mmm format)
3. Convert to seconds since start
4. Round timestamps to nearest second
5. Group by time_rounded
6. Aggregate multiple messages per second (mean)
7. Verify temporal continuity (no time jumps)

**Outputs**:
- `Phase1/data/aligned_timeseries.csv` (~9,859 rows, 13 signals)
- Dense format: each row = 1 second, all signals in same row

**Validation**:
- Row count ≈ total seconds in dataset
- All timestamps sequential (no gaps > 1s)
- Each row has attempt at all 13 signals

---

## Step 1.3: Signal Clustering (Optional Analysis)
**What**: Group correlated signals to understand relationships

**Script**: `Phase1/preprocess_step3_cluster.py`

**Actions**:
1. Load aligned time-series
2. Calculate correlation matrix on non-NaN values
3. Perform hierarchical clustering
4. Generate dendrogram visualization
5. Document signal groups

**Outputs**:
- `Phase1/results/correlation_matrix.png`
- `Phase1/results/signal_dendrogram.png`
- `Phase1/results/signal_clusters.json`

**Use Case**:
- Understand which signals move together
- Validate Phase 0 redundancy analysis
- Inform feature importance later

**Note**: This is analysis only, doesn't change data

---

## Step 1.4: Forward Fill with Gap Limits
**What**: Fill missing values using last known value with temporal constraints

**Script**: `Phase1/preprocess_step4_fill.py`

**Actions**:
1. Load aligned time-series
2. For each signal:
   - Apply forward-fill with max 5-second gap
   - Track: gaps filled, gaps too long (not filled)
3. Handle remaining NaN:
   - Option A: Drop rows still too sparse (< 10 signals)
   - Option B: Backward fill (limited)
   - Choose based on remaining sparsity
4. Final NaN check

**Outputs**:
- `Phase1/data/filled_timeseries.csv` (dense, minimal NaN)
- `Phase1/logs/fill_statistics.json` (gaps filled per signal)

**Validation**:
- No unlimited forward-fill (prevents artificial patterns)
- Document rows dropped (if any)
- Verify NaN percentage < 5%

---

## Step 1.5: Drop Sparse Rows
**What**: Remove rows with too few signals for reliable ML training

**Script**: `Phase1/preprocess_step5_filter.py`

**Actions**:
1. Load filled time-series
2. Count non-NaN signals per row
3. Filter: Keep only rows with ≥ MIN_SIGNALS
   - Recommended: MIN_SIGNALS = 10 (from Phase 0 analysis)
4. Print before/after statistics

**Outputs**:
- `Phase1/data/dense_timeseries.csv` (high-quality rows only)
- `Phase1/logs/filtering_stats.json`

**Expected Result**:
- ~6,000-7,000 rows (67% of 9,859 had ≥10 signals)
- Each row has ≥10 of 13 signals
- Better ML training quality

**Validation**:
- No completely empty rows
- Remaining rows have good signal coverage
- Time range still represents complete journey

---

## Step 1.6: Create Data Queue (Final Time-Series)
**What**: Prepare clean, continuous time-series as model input base

**Script**: `Phase1/preprocess_step6_queue.py`

**Actions**:
1. Load dense time-series
2. Verify structure:
   - Shape: (num_timesteps, num_signals)
   - All values numeric
   - Minimal/no NaN
3. Verify temporal properties:
   - Sequential timestamps
   - No large time gaps
4. Final quality checks:
   - No constant columns (zero variance)
   - Reasonable value ranges per signal
5. Save as numpy array

**Outputs**:
- `Phase1/data/data_queue.npy` (2D array: time × signals)
- `Phase1/data/signal_names.json` (signal order)
- `Phase1/logs/queue_statistics.json`

**Validation**:
- Ready for multi-view generation
- No preprocessing artifacts
- Data integrity verified

---

## Step 1.7: Generate Multi-Scale Views
**What**: Create 3 different temporal views with different sampling periods

**Script**: `Phase1/preprocess_step7_multiview.py`

**Actions**:
1. Load data queue
2. Create View D1 (sampling period T1 = 1 second):
   - Use data queue as-is
3. Create View D2 (sampling period T2 = 5 seconds):
   - Downsample: take every 5th timestep
   - Or aggregate: mean of 5-second windows
4. Create View D3 (sampling period T3 = 10 seconds):
   - Downsample: take every 10th timestep
   - Or aggregate: mean of 10-second windows
5. Verify dimensions

**Outputs**:
- `Phase1/data/view_sp1.npy` (sampling period 1s)
- `Phase1/data/view_sp5.npy` (sampling period 5s)
- `Phase1/data/view_sp10.npy` (sampling period 10s)

**Expected Shapes**:
- View 1: (~6,000 timesteps, 13 signals)
- View 2: (~1,200 timesteps, 13 signals)
- View 3: (~600 timesteps, 13 signals)

**Validation**:
- Views capture different temporal granularities
- No data loss during downsampling
- Temporal alignment verified

---

## Step 1.8: Create Sliding Windows
**What**: Generate overlapping windows of w timesteps for CNN input

**Script**: `Phase1/preprocess_step8_windows.py`

**Actions**:
1. Load all 3 views
2. For each view:
   - Window size: w = 50 timesteps
   - Slide window with step = 1 (or 10 for faster training)
   - Create 3D array: (num_windows, w, num_signals)
3. Calculate number of windows per view

**Outputs**:
- `Phase1/data/windows_sp1.npy` (N1 windows, 50, 13)
- `Phase1/data/windows_sp5.npy` (N2 windows, 50, 13)
- `Phase1/data/windows_sp10.npy` (N3 windows, 50, 13)

**Expected Window Counts** (with step=1):
- View 1: ~5,950 windows (6,000 - 50 + 1)
- View 2: ~1,150 windows
- View 3: ~550 windows

**Note**: For faster training, use step=10:
- View 1: ~595 windows
- View 2: ~115 windows
- View 3: ~55 windows

**Validation**:
- Window shape correct: (w, num_signals)
- No overlapping issues
- Temporal order preserved

---

## Step 1.9: Min-Max Scaling
**What**: Normalize all values to [0, 1] range for neural network training

**Script**: `Phase1/preprocess_step9_scale.py`

**Actions**:
1. Load all windowed arrays
2. Compute min/max per signal:
   - **CRITICAL**: Only from training data!
   - Prevents data leakage from validation/test
3. Apply scaling to all windows:
   - scaled = (value - min) / (max - min)
4. Save scaler parameters for deployment

**Outputs**:
- `Phase1/data/scaled_windows_sp1.npy`
- `Phase1/data/scaled_windows_sp5.npy`
- `Phase1/data/scaled_windows_sp10.npy`
- `Phase1/data/scaler_params.pkl` (min/max per signal)

**Validation**:
- All values in range [0, 1]
- No NaN introduced by scaling
- Scaler saved for future use (test data, deployment)

---

## Step 1.10: Train/Validation Split
**What**: Split data into training and validation sets

**Script**: `Phase1/preprocess_step10_split.py`

**Actions**:
1. Load all scaled windows
2. Split strategy:
   - **Temporal split** (recommended): First 80% train, last 20% validation
   - Maintains temporal order (realistic for time-series)
3. For each view:
   - Split windows into train/validation
   - Save separately

**Outputs**:
- `Phase1/data/train_sp1.npy`, `Phase1/data/val_sp1.npy`
- `Phase1/data/train_sp5.npy`, `Phase1/data/val_sp5.npy`
- `Phase1/data/train_sp10.npy`, `Phase1/data/val_sp10.npy`
- `Phase1/logs/split_statistics.json`

**Split Sizes** (example with step=1):
- View 1 train: ~4,760 windows, val: ~1,190 windows
- View 2 train: ~920 windows, val: ~230 windows
- View 3 train: ~440 windows, val: ~110 windows

**Validation**:
- No data leakage (train/val don't overlap)
- Validation set large enough for reliable metrics
- All views split consistently

---

## Step 1.11: Preprocessing Validation & Report
**What**: Final verification before training

**Script**: `Phase1/preprocess_step11_validate.py`

**Actions**:
1. Load all preprocessed data
2. Verify shapes and dimensions
3. Check value ranges [0, 1]
4. Verify no NaN in training data
5. Generate preprocessing report

**Outputs**:
- `Phase1/PREPROCESSING_REPORT.md`
- `Phase1/data/preprocessing_summary.json`

**Report Contents**:
```
1. Data Pipeline Summary
   - Input: 654K messages, 19 signals
   - After cleaning: 13 signals
   - After alignment: 9,859 seconds
   - After filtering: ~6,000-7,000 dense rows
   
2. Final Datasets
   - 3 views: sp1, sp5, sp10
   - Window size: 50 timesteps
   - Signal count: 13
   - Train/val split: 80/20
   
3. Training Data Shapes
   - View 1 train: (4,760, 50, 13)
   - View 2 train: (920, 50, 13)
   - View 3 train: (440, 50, 13)
   
4. Data Quality
   - Value range: [0, 1]
   - NaN count: 0
   - All temporal order preserved
```

**Validation Checklist**:
- [ ] All preprocessing steps completed successfully
- [ ] Data shapes match expectations
- [ ] No data quality issues (NaN, outliers)
- [ ] Scaler saved for deployment
- [ ] Train/val split verified
- [ ] **You approve before Phase 2**

---

# PHASE 2: MODEL TRAINING
**Goal**: Build and train CNN-Autoencoder ensemble for anomaly detection  
**Duration**: 4-6 days  
**Input**: Preprocessed windowed datasets from Phase 1  
**Output**: 3 trained models + ensemble detector

---

## Step 2.1: Build CNN-Autoencoder Architecture
**What**: Implement CANShield's CNN-AE model in TensorFlow/Keras

**Script**: `Phase2/model.py`

**Architecture** (from CANShield paper):
```python
def build_cnn_autoencoder(time_steps=50, num_signals=13):
    """
    Input: (time_steps, num_signals, 1)
    
    Encoder:
      Conv2D(32, kernel=3, strides=1, activation=LeakyReLU)
      Conv2D(16, kernel=3, strides=1, activation=LeakyReLU)
      Conv2D(16, kernel=3, strides=1, activation=LeakyReLU)
    
    Decoder:
      Conv2D(16, kernel=3, strides=1, activation=LeakyReLU)
      Conv2D(16, kernel=3, strides=1, activation=LeakyReLU)
      Conv2D(32, kernel=3, strides=1, activation=LeakyReLU)
      Conv2D(1, kernel=3, strides=1, activation=Sigmoid)
    
    Output: (time_steps, num_signals, 1)
    Loss: Mean Absolute Error (MAE)
    Optimizer: Adam
    """
    # Implementation here
    return model
```

**Actions**:
1. Implement exact CANShield architecture
2. Test model compilation
3. Print model summary
4. Test forward pass on dummy data

**Outputs**:
- `Phase2/model.py` (model architecture)
- Model summary (layer shapes, parameters)

**Validation**:
- Architecture matches CANShield paper
- Input/output shapes correct
- Model compiles without errors
- Parameter count reasonable (~50K-100K)

---

## Step 2.2: Training Utilities & Configuration
**What**: Create training helpers and configuration

**Script**: `Phase2/train_config.py`

**Components**:
1. **Training configuration**:
   ```python
   EPOCHS = 100
   BATCH_SIZE = 64
   LEARNING_RATE = 0.001
   EARLY_STOPPING_PATIENCE = 10
   ```

2. **Callbacks**:
   - EarlyStopping (patience=10)
   - ModelCheckpoint (save best model)
   - ReduceLROnPlateau (reduce LR if plateau)
   - TensorBoard logging

3. **Metrics**:
   - Training loss (MAE)
   - Validation loss (MAE)

**Outputs**:
- `Phase2/train_config.py` (configuration)
- `Phase2/callbacks.py` (training callbacks)

---

## Step 2.3: Train Model AE₁ (View sp1)
**What**: Train first autoencoder on fine-grained view (1-second sampling)

**Script**: `Phase2/train_ae1.py`

**Actions**:
1. Load training data: `train_sp1.npy`
2. Load validation data: `val_sp1.npy`
3. Reshape for CNN: (N, 50, 13) → (N, 50, 13, 1)
4. Create model: `build_cnn_autoencoder(50, 13)`
5. Train:
   - Epochs: 100 (with early stopping)
   - Batch size: 64
   - Optimizer: Adam
6. Save best model
7. Plot training curves

**Outputs**:
- `Phase2/models/ae_sp1.h5` (trained model)
- `Phase2/logs/training_history_sp1.json`
- `Phase2/results/training_curve_sp1.png`

**Expected Results**:
- Training converges (loss decreases)
- Validation loss stabilizes
- No severe overfitting (train/val gap small)

**Validation**:
- Final validation loss < 0.1 (typical for MAE)
- Training completed without errors
- Model saved successfully

---

## Step 2.4: Train Model AE₂ (View sp5) with Transfer Learning
**What**: Fine-tune second autoencoder on medium-grained view (5-second sampling)

**Script**: `Phase2/train_ae2.py`

**Actions**:
1. Load AE₁ weights: `ae_sp1.h5`
2. Load data: `train_sp5.npy`, `val_sp5.npy`
3. Initialize AE₂ with AE₁ weights (transfer learning)
4. Fine-tune for ~70 epochs (faster than training from scratch)
5. Save best model

**Outputs**:
- `Phase2/models/ae_sp5.h5`
- `Phase2/logs/training_history_sp5.json`
- `Phase2/results/training_curve_sp5.png`

**Expected Benefits**:
- Faster convergence (transfer learning)
- Better performance (pre-trained features)
- Reduced training time

**Validation**:
- Converges faster than AE₁ (fewer epochs to plateau)
- Performance comparable or better
- Transfer learning effect visible

---

## Step 2.5: Train Model AE₃ (View sp10) with Transfer Learning
**What**: Fine-tune third autoencoder on coarse-grained view (10-second sampling)

**Script**: `Phase2/train_ae3.py`

**Actions**:
1. Load AE₂ weights: `ae_sp5.h5`
2. Load data: `train_sp10.npy`, `val_sp10.npy`
3. Initialize AE₃ with AE₂ weights
4. Fine-tune for ~70 epochs
5. Save best model

**Outputs**:
- `Phase2/models/ae_sp10.h5`
- `Phase2/logs/training_history_sp10.json`
- `Phase2/results/training_curve_sp10.png`

**Validation**:
- All 3 models trained successfully
- Transfer learning cascade worked
- Each model specialized for its temporal scale

---

## Step 2.6: Compute Reconstruction Losses
**What**: Calculate reconstruction errors on validation data to set thresholds

**Script**: `Phase2/compute_losses.py`

**Actions**:
1. Load all 3 trained models
2. Load validation data (normal navigation only)
3. For each model:
   - Forward pass (reconstruct)
   - Compute absolute error per signal per timestep
   - Shape: (num_val_samples, 50, 13)
4. Save reconstruction losses

**Outputs**:
- `Phase2/results/reconstruction_losses_sp1.npy`
- `Phase2/results/reconstruction_losses_sp5.npy`
- `Phase2/results/reconstruction_losses_sp10.npy`

**Use Case**:
- Threshold selection in next step
- Understanding normal vs anomalous patterns

**Validation**:
- Loss distributions look reasonable
- No extreme outliers (model errors)

---

## Step 2.7: Threshold Selection (3-Step Algorithm)
**What**: Determine optimal detection thresholds using CANShield's method

**Script**: `Phase2/select_thresholds.py`

**Algorithm** (CANShield Algorithm 1):
```
Step 1: Signal-wise reconstruction threshold (R_Loss)
  - For each signal, compute p-th percentile of reconstruction error
  - Threshold: if error[signal] > R_Loss[signal] → anomaly in that signal
  
Step 2: Signal-wise timestep violations (R_Time)
  - Count timesteps where signal exceeds R_Loss
  - Threshold: if violations > q-th percentile → signal is anomalous
  
Step 3: Overall signal violations (R_Signal)
  - Count how many signals are anomalous
  - Threshold: if anomalous_signals > r-th percentile → ATTACK DETECTED
```

**Actions**:
1. Load reconstruction losses (validation data)
2. Grid search over parameters:
   - p: [90, 95, 97, 99]% (signal reconstruction threshold)
   - q: [95, 97, 99]% (timestep violation threshold)
   - r: [95, 97, 99]% (signal violation threshold)
3. For each combination:
   - Apply 3-step algorithm
   - Compute FPR on validation (should be < 1%)
   - Save configuration
4. Select optimal (p, q, r) with FPR ≈ 1%

**Outputs**:
- `Phase2/results/optimal_thresholds.json`
  ```json
  {
    "sp1": {"p": 95, "q": 97, "r": 95, "fpr": 0.008},
    "sp5": {"p": 97, "q": 97, "r": 97, "fpr": 0.009},
    "sp10": {"p": 95, "q": 95, "r": 95, "fpr": 0.010}
  }
  ```
- `Phase2/results/threshold_analysis.png`

**Validation**:
- FPR on validation < 1% (not too sensitive)
- Thresholds not too loose (will detect attacks)

---

## Step 2.8: Build Ensemble Detector
**What**: Combine 3 models into ensemble voting system

**Script**: `Phase2/ensemble.py`

**Ensemble Strategy**:
```python
def ensemble_detect(data_queue, models, thresholds):
    """
    1. Create 3 views from data queue (sp1, sp5, sp10)
    2. Run each AE model on its view
    3. Apply 3-step threshold to each model → 3 anomaly scores
    4. Combine scores:
       - Simple average
       - Or majority voting
    5. Return: is_attack (True/False), confidence (0-1)
    """
```

**Actions**:
1. Load all 3 models
2. Load optimal thresholds
3. Implement ensemble logic
4. Test on validation data

**Outputs**:
- `Phase2/ensemble.py` (ensemble detector)
- `Phase2/results/ensemble_validation.json` (FPR on validation)

**Expected Result**:
- Ensemble FPR < 1% on normal validation data
- Better than individual models (robustness)

**Validation**:
- Ensemble works correctly
- No coding errors
- Detections make sense

---

## Step 2.9: Training Summary Report
**What**: Document training results and model performance

**Script**: `Phase2/generate_report.py`

**Report Contents**:
```
1. Model Architecture
   - Layers, parameters, activation functions
   
2. Training Results
   - AE₁ (sp1): epochs, final loss, training time
   - AE₂ (sp5): epochs, final loss, training time
   - AE₃ (sp10): epochs, final loss, training time
   
3. Transfer Learning Impact
   - Convergence speedup (epochs saved)
   - Performance comparison
   
4. Threshold Selection
   - Optimal (p, q, r) per model
   - FPR on validation
   
5. Ensemble Performance
   - Combined FPR
   - Robustness metrics
```

**Outputs**:
- `Phase2/TRAINING_REPORT.md`
- `Phase2/results/model_comparison.png`

**Validation Checklist**:
- [ ] All 3 models trained successfully
- [ ] Transfer learning worked (faster convergence)
- [ ] Thresholds selected (FPR < 1%)
- [ ] Ensemble detector implemented
- [ ] Training report complete
- [ ] **You approve before Phase 3**

---

# PHASE 3: EVALUATION & RESULTS
**Goal**: Test performance and create publication-ready visualizations  
**Duration**: 2 days  
**Input**: Trained models + validation data  
**Output**: Metrics, plots, paper-ready results

---

## Step 3.1: Evaluate Individual Models
**What**: Compute comprehensive metrics for each autoencoder

**Script**: `Phase3/evaluate_models.py`

**Metrics** (on validation data):
1. **Reconstruction Error**:
   - MAE per signal
   - Overall MAE
   - Loss distribution

2. **Detection Performance** (if we had attack data):
   - True Positive Rate (TPR)
   - False Positive Rate (FPR)
   - Precision
   - Recall
   - F1 Score
   - AUROC
   - AUPRC

3. **Current Performance** (normal data only):
   - Reconstruction accuracy
   - Signal-wise error analysis
   - FPR at selected thresholds

**Actions**:
1. Load validation data
2. For each model:
   - Reconstruct inputs
   - Compute metrics
   - Save results
3. Compare models

**Outputs**:
- `Phase3/results/metrics_ae_sp1.json`
- `Phase3/results/metrics_ae_sp5.json`
- `Phase3/results/metrics_ae_sp10.json`
- `Phase3/results/model_comparison.csv`

---

## Step 3.2: Evaluate Ensemble
**What**: Test ensemble detector performance

**Script**: `Phase3/evaluate_ensemble.py`

**Metrics**:
1. **Normal Data Performance**:
   - FPR (should be < 1%)
   - Average confidence scores
   - Detection consistency

2. **Robustness**:
   - Agreement between 3 models
   - Ensemble vs individual model comparison

**Actions**:
1. Run ensemble on validation data
2. Compute metrics
3. Analyze detections (if any false alarms)

**Outputs**:
- `Phase3/results/ensemble_metrics.json`
- `Phase3/results/ensemble_detections.csv` (false alarms)

---

## Step 3.3: Create Visualizations
**What**: Generate publication-quality plots

**Script**: `Phase3/create_visualizations.py`

**Plots to Create**:

1. **Training Curves** (3 subplots):
   - AE₁, AE₂, AE₃ training/validation loss over epochs
   - Show transfer learning effect

2. **Reconstruction Examples**:
   - Original vs reconstructed signals
   - Show model learned normal patterns

3. **Loss Distributions**:
   - Histogram of reconstruction errors
   - Normal data distribution
   - Threshold visualization

4. **Signal-Wise Error Analysis**:
   - Bar chart: MAE per signal
   - Identify which signals are harder to reconstruct

5. **Correlation Matrix** (from Phase 0):
   - Clean heatmap of signal correlations
   - Show clustering structure

6. **Model Comparison**:
   - Bar chart: AE₁ vs AE₂ vs AE₃ vs Ensemble
   - Metrics: MAE, FPR

7. **Temporal Analysis**:
   - Reconstruction error over time
   - Show stability across journey

**Outputs**:
- `Phase3/figures/training_curves.png`
- `Phase3/figures/reconstruction_examples.png`
- `Phase3/figures/loss_distributions.png`
- `Phase3/figures/signal_error_analysis.png`
- `Phase3/figures/correlation_matrix.png`
- `Phase3/figures/model_comparison.png`
- `Phase3/figures/temporal_analysis.png`

**Format**: High-resolution PNG + PDF for paper

---

## Step 3.4: Reconstruction Case Studies
**What**: Detailed analysis of specific time windows

**Script**: `Phase3/case_studies.py`

**Cases to Analyze**:
1. **Best Reconstruction**: Lowest error window
2. **Worst Reconstruction**: Highest error window (why?)
3. **Typical Reconstruction**: Median error window

**For Each Case**:
- Plot all 13 signals (original vs reconstructed)
- Compute per-signal error
- Analyze what model learned/missed

**Outputs**:
- `Phase3/figures/case_best.png`
- `Phase3/figures/case_worst.png`
- `Phase3/figures/case_typical.png`

**Purpose**:
- Understand model behavior
- Identify potential issues
- Explain to reviewers

---

## Step 3.5: Create Final Report
**What**: Comprehensive documentation of entire project

**Script**: `Phase3/generate_final_report.py`

**Report Sections**:

```markdown
# N2KShield: NMEA2000 Intrusion Detection System
## Final Project Report

### 1. Executive Summary
- Project goal
- Dataset characteristics
- Final model performance
- Key findings

### 2. Phase 0: Data Exploration
- Dataset description (2.98M frames, 654K messages)
- Signal analysis (19 → 13 signals)
- Data quality findings
- Parameter decisions

### 3. Phase 1: Preprocessing
- Pipeline steps (11 substeps)
- Data transformations
- Final dataset statistics
- Quality assurance

### 4. Phase 2: Model Training
- CNN-AE architecture
- Training results (3 models)
- Transfer learning impact
- Threshold selection
- Ensemble construction

### 5. Phase 3: Evaluation
- Performance metrics
- Model comparison
- Visualizations
- Case studies

### 6. Results Summary
- Key metrics table
- Model comparison
- Strengths & limitations

### 7. Differences from CANShield
- Signal selection (13 vs 20)
- Maritime vs automotive
- Preprocessing adaptations
- Performance comparison

### 8. Conclusions & Future Work
- What we achieved
- Limitations (no attack data yet)
- Next steps (attack synthesis, real-world testing)

### 9. References
- CANShield paper
- NMEA2000 specification
- Related work
```

**Outputs**:
- `FINAL_REPORT.md` (complete project documentation)
- `FINAL_REPORT.pdf` (formatted for presentation)

---

## Step 3.6: Prepare Paper Materials
**What**: Organize results for IEEE paper submission

**Actions**:
1. Select best figures for paper (6-8 figures max)
2. Create tables:
   - Dataset statistics table
   - Signal list with coverage
   - Model architecture table
   - Performance metrics table
3. Write figure captions
4. Prepare supplementary materials

**Outputs**:
- `Paper/figures/` (selected high-res figures)
- `Paper/tables/` (LaTeX formatted tables)
- `Paper/captions.txt` (figure captions)
- `Paper/supplementary/` (additional materials)

---

**Phase 3 Validation Checklist**:
- [ ] All metrics computed correctly
- [ ] Visualizations are publication-quality
- [ ] Case studies provide insights
- [ ] Final report is comprehensive
- [ ] Paper materials ready
- [ ] **Project complete!**

---

# 📊 **SUCCESS CRITERIA**

## Minimum Viable Product (MVP):
- [x] Phase 0: Data exploration complete
- [ ] Phase 1: ML-ready dataset created
- [ ] Phase 2: 3 CNN-AE models trained
- [ ] Phase 2: Ensemble detector working
- [ ] Phase 3: Performance metrics computed
- [ ] Phase 3: Basic visualizations created

## Extended Goals:
- [ ] FPR < 1% on normal data
- [ ] Transfer learning reduces training time by 30%+
- [ ] All visualizations publication-ready
- [ ] IEEE paper draft complete

## Future Work (Not in This Plan):
- [ ] Synthetic attack generation
- [ ] Real attack testing
- [ ] Embedded deployment
- [ ] Real-time detection

---

# 🎯 **NEXT STEPS**

## Immediate (Complete Phase 0):
1. ✅ Finalize signal selection (13 signals decided)
2. ⏳ Decide temporal parameters (window=50, sampling=[1,5,10])
3. ⏳ Write Phase 0 report
4. ⏳ Get your approval

## Then Start Phase 1:
1. Create Phase1 folder structure
2. Run preprocessing pipeline (Steps 1.1-1.11)
3. Validate ML-ready data
4. Get your approval before Phase 2

---

**Document Version**: 2.0 - Simplified  
**Date**: November 25, 2025  
**Status**: Phase 0 90% complete, ready to finalize

**Total Estimated Time**: 10-14 days (excluding future work)
