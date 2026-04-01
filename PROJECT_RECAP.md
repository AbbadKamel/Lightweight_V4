# Complete Project Recap: NMEA 2000 Intrusion Detection System

---

## 🎯 PROJECT GOAL

**Build a lightweight CNN-based intrusion detection system for NMEA 2000 maritime networks**

**Target**: Detect cyberattacks on boat communication systems (GPS spoofing, message injection, replay attacks, etc.)

**Constraint**: Must run on resource-limited embedded devices (Raspberry Pi, ESP32)

---

# PHASE 0: Data Collection & Initial Preprocessing

## What We Had

- **Raw CAN bus data**: NMEA 2000 frames captured from real boat
- **Format**: Timestamp, CAN ID, 8 data bytes per frame
- **Size**: ~1.5 million frames
- **Duration**: ~85 minutes of boat operation

## What We Did

### Step 1: Extract PGN Information
**Tool**: Python script to decode CAN IDs
**Why**: NMEA 2000 uses Parameter Group Numbers (PGNs) to identify message types
**Result**: Identified 15 unique PGNs (GPS, speed, heading, depth, wind, etc.)

**Decision Logic**: 
- ✅ Focus on most frequent PGNs (high data availability)
- ✅ Select PGNs relevant to safety-critical functions
- ❌ Ignore rare PGNs (insufficient training data)

### Step 2: Decode Data Fields
**Tool**: NMEA 2000 specification + custom decoder
**Why**: Raw bytes need to be converted to meaningful values (latitude, longitude, speed, etc.)
**Result**: Extracted 15 signals from PGN data fields

**Example**:
```
PGN 129029 (Position Rapid Update):
- Latitude: 43.123456° N
- Longitude: -5.678901° W

PGN 127250 (Vessel Heading):
- Heading: 315° (Northwest)
```

### Step 3: Create Time-Series Aggregation
**Tool**: Pandas groupby with 1-second intervals
**Why**: CAN frames arrive asynchronously (different PGNs at different rates)
**Result**: Synchronized all signals to 1-second resolution

**Decision Logic**:
- ✅ 1 second = balance between detail and manageability
- ✅ Aligns with NMEA 2000 typical update rates
- ❌ Not milliseconds (too much data, mostly duplicates)
- ❌ Not 5+ seconds (lose important details)

### Step 4: Feature Engineering (4 Aggregations per Signal)
**Tool**: Statistical aggregations
**Why**: Capture different aspects of signal behavior within each second

**For each of 15 signals, we computed**:
1. **Mean**: Average value (e.g., average speed in that second)
2. **Max**: Peak value (e.g., maximum heading change)
3. **Min**: Lowest value (e.g., minimum depth reading)
4. **Std**: Variability (e.g., GPS jitter/stability)

**Result**: 15 signals × 4 aggregations = **60 features per second**

**Decision Logic**:
- ✅ Mean: Represents typical behavior
- ✅ Max/Min: Catches sudden spikes (attack indicators)
- ✅ Std: Detects abnormal fluctuations
- ❌ Not median (computationally expensive, marginal benefit)
- ❌ Not variance (std already captures it)

### Step 5: Handle Missing Data
**Tool**: Forward-fill then backward-fill
**Why**: CAN bus might miss some PGN messages

**Strategy**:
```
If GPS data missing at second 100:
1. Forward-fill: Use value from second 99
2. If still missing, backward-fill: Use value from second 101
3. If still missing (start/end of dataset): Use 0
```

**Decision Logic**:
- ✅ Forward-fill: Assumes sensor value doesn't change instantly
- ✅ Preserves temporal continuity
- ❌ Not delete rows (would create gaps in time-series)
- ❌ Not mean imputation (loses temporal context)

## Phase 0 Output

**File**: `master_table_raw.csv` (or similar)
- **Shape**: 5,095 rows × 60 columns
- **Rows**: Each row = 1 second of boat data
- **Columns**: 60 features (15 signals × 4 aggregations)
- **Values**: Real numbers (latitude, speed, depth, etc.)
- **Missing data**: Handled via forward/backward fill

---

# PHASE 1: Preprocessing & Windowing

## Step 1: Normalization (Min-Max Scaling to [0, 1])

**Tool**: scikit-learn MinMaxScaler
**Formula**: `X_norm = (X - X_min) / (X_max - X_min)`

**Why Normalize?**

### Reason 1: Gradient Descent Stability
- **Problem**: Features have different scales
  - Latitude: 40-45° 
  - Speed: 0-20 knots
  - Depth: 0-200 meters
- **Issue**: Large values dominate gradient calculations
- **Solution**: Scale all to [0, 1] → balanced gradients

### Reason 2: Feature Importance Balance
- Without normalization: Speed (0-20) >> Latitude change (0.001)
- CNN would learn: "Speed matters, latitude doesn't"
- With normalization: All features equally weighted

### Reason 3: Activation Function Range
- **CNN uses**: ReLU, Sigmoid, Tanh
- **Optimal input**: [0, 1] range
- **Out of range**: Causes saturation (neurons stop learning)

### Reason 4: Convergence Speed
- Normalized data → faster training (fewer epochs)
- Non-normalized → slow, unstable convergence

**Decision Logic**:
- ✅ Min-Max [0, 1]: Preserves feature relationships, works with ReLU
- ❌ Not StandardScaler: Can produce negative values (bad for ReLU)
- ❌ Not no normalization: Different scales break CNN training

**File Generated**: `Phase1/results/master_table_final.csv`
- **Shape**: 5,095 × 60
- **Values**: All in [0.0, 1.0]
- **Validation**: No NaN, no out-of-bounds values

---

## Step 2: Validation Visualizations

**Tool**: Custom Python script (`validate_master_table_simple.py`)

**6 Validation Plots Created**:

### Plot 1: Normalization Check
- **Purpose**: Verify all values in [0, 1]
- **What we checked**: Min, max, mean of each feature
- **Result**: ✅ All values properly normalized

### Plot 2: Feature Relationships
- **Purpose**: Check mean/max/min/std relationships make sense
- **Example**: Speed_max ≥ Speed_mean ≥ Speed_min ✅
- **Result**: ✅ Logical relationships preserved

### Plot 3: Signal Patterns Over Time
- **Purpose**: Visualize temporal evolution of key signals
- **What we saw**: GPS coordinates, speed, heading changing over time
- **Result**: ✅ Smooth transitions, no sudden jumps

### Plot 4: Feature Correlation Matrix
- **Purpose**: Identify highly correlated features
- **Insight**: Speed_mean and Speed_max highly correlated (expected)
- **Result**: ✅ No unexpected correlations

### Plot 5: Temporal Continuity
- **Purpose**: Check for gaps or discontinuities
- **What we checked**: Consecutive rows shouldn't have huge differences
- **Result**: ✅ Smooth time-series, no data gaps

### Plot 6: CNN Input Preview
- **Purpose**: Visualize what the CNN will actually see
- **Format**: Heatmap of first 100 rows × 60 features
- **Result**: ✅ Clear patterns visible, ready for CNN

**Decision Logic**:
- ✅ Multiple validation angles catch different issues
- ✅ Visual inspection confirms numerical checks
- ✅ Builds confidence before expensive CNN training

---

## Step 3: Full Dataset Heatmap

**Tool**: `plot_full_dataset_heatmap.py`
**Purpose**: See the ENTIRE dataset at once

**Visualization**: 5,095 rows × 60 features heatmap
- **X-axis**: 60 features (all signals and aggregations)
- **Y-axis**: 5,095 seconds (entire boat journey)
- **Color**: Intensity shows normalized value [0, 1]

**What We Learned**:
- Some features very active (lots of variation)
- Some features stable (minimal change)
- Temporal patterns visible (boat maneuvers, weather changes)

**Decision Logic**:
- ✅ Confirms data quality across entire dataset
- ✅ Identifies interesting temporal patterns
- ✅ Helps judge if 85 minutes is enough data

---

## Step 4: Windowing Configuration (CANShield Methodology)

**Reference**: CANShield research paper (intrusion detection for CAN bus)

### Why Windowing?

**Problem**: CNN needs fixed-size input
- Dataset: 5,095 rows (variable length)
- CNN: Requires consistent shape, e.g., (50, 60)

**Solution**: Sliding window approach
- Extract fixed-size windows from time-series
- Feed each window to CNN independently

### CANShield Strategy: Multi-Scale Temporal Analysis

**Configuration File**: `scripts/config.py`

```python
TIME_STEPS = [50, 75, 100]  # Window sizes (in rows)
SAMPLING_PERIODS = [1, 5, 10]  # Downsampling factors (in seconds)
WINDOW_STEP_TRAIN = 10  # How many rows to move forward (overlap control)
```

**Why These Values?**

#### TIME_STEPS = [50, 75, 100] rows
- **50 rows**: Short-term patterns (fast attacks, sudden changes)
- **75 rows**: Medium-term patterns (gradual drift)
- **100 rows**: Long-term patterns (slow spoofing)
- **Decision**: Cover multiple temporal scales
- **Alternative rejected**: Single window size (misses scale-specific attacks)

#### SAMPLING_PERIODS = [1, 5, 10] seconds
- **1s sampling**: High-resolution (keep all 5,095 rows)
  - Catches: Fast GPS injection, sudden speed changes
  - Window coverage: 50 rows = 50 seconds of data
  
- **5s sampling**: Medium-resolution (keep every 5th row → 1,019 rows)
  - Catches: Gradual drift, medium-term anomalies
  - Window coverage: 50 rows = 250 seconds (4+ minutes)
  
- **10s sampling**: Low-resolution (keep every 10th row → 509 rows)
  - Catches: Long-term trends, slow replay attacks
  - Window coverage: 50 rows = 500 seconds (8+ minutes)

- **Decision**: Downsample to see different time horizons
- **Alternative rejected**: Fixed sampling (misses slow attacks)

#### WINDOW_STEP = 10 rows
- **Overlap**: 80% between consecutive windows
  - Window 1: rows 0-49
  - Window 2: rows 10-59 (40 rows overlap)
- **Why 80%**: Balance between coverage and redundancy
- **Decision**: CANShield proven standard
- **Alternative rejected**: 
  - 1-row step (98% overlap, too redundant, 5,046 windows!)
  - 50-row step (0% overlap, might miss attacks between windows)

### Multi-Scale Analysis Logic

**Why 9 Configurations (3 window sizes × 3 sampling periods)?**

| Config | Window Size | Sampling | Real Time Covered | Best For |
|--------|-------------|----------|-------------------|----------|
| 1      | 50 rows     | 1s       | 50 seconds        | Fast spikes, injections |
| 2      | 50 rows     | 5s       | 250 seconds       | Medium drift |
| 3      | 50 rows     | 10s      | 500 seconds       | Slow spoofing |
| 4      | 75 rows     | 1s       | 75 seconds        | Short bursts |
| 5      | 75 rows     | 5s       | 375 seconds       | Gradual changes |
| 6      | 75 rows     | 10s      | 750 seconds       | Long trends |
| 7      | 100 rows    | 1s       | 100 seconds       | Extended fast attacks |
| 8      | 100 rows    | 5s       | 500 seconds       | Medium-long patterns |
| 9      | 100 rows    | 10s      | 1000 seconds      | Very slow anomalies |

**Ensemble Detection**:
1. Train 9 separate CNNs (one per configuration)
2. Each CNN becomes "expert" at its time scale
3. Final prediction: Majority vote or ANY detection
4. If ANY CNN flags anomaly → investigate

**Decision Logic**:
- ✅ Multi-scale catches diverse attack types
- ✅ Proven effective in CANShield paper
- ✅ Ensemble more robust than single model
- ❌ Not single scale (misses attacks at other resolutions)

---

## Step 5: Window Generation & Visualization

### Part A: Summary Visualizations per Configuration

**Tool**: `generate_window_visualizations.py`

**For EACH of 9 configurations, generated 3 plots**:

#### Plot 1: Example Windows (6 samples)
- **Shows**: 6 evenly-spaced windows from the configuration
- **Purpose**: Preview what CNN will see
- **Format**: 6 heatmaps (each 50×60 or 75×60 or 100×60)

**Why only 6 examples?**
- ✅ Representative sample (beginning, middle, end)
- ✅ Not overwhelming (505 plots would be unreadable)
- ❌ Not all windows (would create 1,925 images, hard to review)

#### Plot 2: Window Statistics
- **Shows**: Distribution of values across all windows
- **Metrics**: Mean, std, min, max of each feature across windows
- **Purpose**: Check consistency (windows should be similar unless attack present)

#### Plot 3: Window Info (Text Summary)
- **Shows**: Configuration details
- **Contents**:
  - Total windows: 505 (for 50s + 1s)
  - Window shape: (50, 60)
  - Real time per window: 50 seconds
  - Overlap: 80%

**Files Generated**: `Phase1/visualizations/[window_size]/sampling_[period]s/`
- example_windows.png
- window_statistics.png
- window_info.txt.png

**Total**: 9 configurations × 3 files = 27 visualization files

**Decision Logic**:
- ✅ Summary plots give quick overview
- ✅ Verify windowing worked correctly before CNN training
- ✅ Compare different configurations visually

---

### Part B: Individual Window Images

**Your Request**: "i want image in each scenario of this intervalls all of them individually separete in png"

**Tool**: `generate_individual_windows.py`

**What It Does**:
For EACH window in EACH configuration, create separate PNG file

**Example for 50s + 1s sampling (505 windows)**:
```
visualizations/50s_window/sampling_1s/individual_windows/
├── window_0000.png   ← First window (rows 0-49)
├── window_0001.png   ← Second window (rows 10-59)
├── window_0002.png   ← Third window (rows 20-69)
...
├── window_0504.png   ← Last window (rows 5,040-5,089)
```

**Total Individual Images Generated**:

| Window Size | 1s Sampling | 5s Sampling | 10s Sampling | Subtotal |
|-------------|-------------|-------------|--------------|----------|
| 50 rows     | 505         | 97          | 47           | 649      |
| 75 rows     | 503         | 95          | 44           | 642      |
| 100 rows    | 500         | 92          | 42           | 634      |
| **TOTAL**   |             |             |              | **1,925**|

**File Size**: ~1,925 PNG files

**Decision Logic**:
- ✅ Allows detailed inspection of ANY specific window
- ✅ Can manually review suspicious patterns
- ✅ Useful for debugging CNN predictions later
- ❌ Not for overview (too many files)

---

## Step 6: Mathematical Verification

**Your Question**: "why for 50s i have only one plot of all windows"

**Confusion**: Mixing example plots (6 windows) vs individual plots (all windows)

**Clarification Process**:
1. Calculated window counts using formula:
   ```
   Windows = ⌊(Total_rows - Window_size) / Step_size⌋ + 1
   ```

2. **For 50s + 1s sampling**:
   ```
   Rows = 5,095
   Window = 50
   Step = 10
   Windows = (5,095 - 50) / 10 + 1 = 505 ✅
   ```

3. Verified file count:
   ```bash
   ls -1 visualizations/50s_window/sampling_1s/individual_windows/ | wc -l
   → 505 files ✅
   ```

**Decision**: Implementation correct, user confusion resolved

---

## Step 7: Understanding Sampling vs Stepping (Your Questions)

### Question 1: "how many rows we have in our full data"
**Answer**: 5,095 rows (each row = 1 second)

### Question 2: "if we have 50S windows with sampling of 1s how many windows we will have"
**Initial confusion**: Thought sampling period affects step size
**Clarification**: 
- Sampling period = downsample data FIRST
- Step size = how to slide window AFTER downsampling
- With 1s sampling + 10-row step → 505 windows

### Question 3: "why you said step size 10s we have 1s and 5s and then 10s"
**Confusion source**: Mixing sampling period (1s/5s/10s) with step size (10 rows)
**Clarification**:
- Sampling period: 1s, 5s, 10s (variable)
- Window step: 10 rows (constant for ALL configs)
- 10 rows ≠ 10 seconds (depends on sampling period)

### Question 4: "then what does that mean sampling 1s 5s et 10s"
**Core confusion**: What changes if window size and step are fixed?

**Breakthrough Explanation**:
```
Original data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

5s sampling: Keep [5, 10, 15, 20]  ← Throw away 1-4, 6-9, 11-14, 16-19

To make 50-row window: [5, 10, 15, 20, 25, 30, ..., 250]
→ 50 values, but covers 0-250 seconds (not 0-50!)
```

**Key Insight**:
- Sampling = downsampling (delete rows) BEFORE windowing
- Same window shape (50, 60), different real-time coverage
- 1s sampling: 50 rows = 50 seconds
- 5s sampling: 50 rows = 250 seconds
- 10s sampling: 50 rows = 500 seconds

---

## Step 8: Your Alternative Approach Discussion

**Your Idea**: 
- Fixed window size: 50 seconds (always)
- Variable step: Move 1s, 5s, or 10s forward
- No downsampling

**Example**:
```
Config 1: 50s window, step 1s → 5,046 windows
Config 2: 50s window, step 5s → 1,010 windows
Config 3: 50s window, step 10s → 505 windows
```

**Comparison**:

| Aspect | Your Approach | CANShield Approach |
|--------|---------------|-------------------|
| **Time coverage** | All see 50 seconds | See 50s, 250s, 500s |
| **Total windows** | 6,561 | 649 (for 50s only) |
| **Overlap** | 98%, 90%, 80% | 80% (all configs) |
| **Training time** | Slower (more data) | Faster (less data) |
| **Attack detection** | Same time scale | Multi-scale (better) |
| **Complexity** | Simpler | More complex |

**Why CANShield is Better**:

1. **Catches Different Attack Types**:
   - Fast injection: 1s sampling (50s windows) ✅
   - Slow spoofing: 10s sampling (500s windows) ✅
   - Your approach: All 50s windows (misses slow attacks) ❌

2. **Ensemble Learning**:
   - 9 specialized models vote together
   - More robust than single-scale detection

3. **Computational Efficiency**:
   - Fewer windows (649 vs 6,561 for 50s configs)
   - Faster training, less storage

4. **Proven Methodology**:
   - Published research paper
   - Validated on automotive CAN bus

**Decision**: Continue with CANShield multi-scale approach

---

# CURRENT STATUS

## What We Have Completed

### Phase 0 Deliverables:
✅ Raw CAN data decoded to meaningful signals
✅ 15 PGNs identified and extracted
✅ Time-series synchronized to 1-second intervals
✅ 60 features engineered (15 signals × 4 aggregations)
✅ Missing data handled (forward/backward fill)
✅ Clean dataset: 5,095 rows × 60 columns

### Phase 1 Deliverables:
✅ Min-Max normalization applied [0, 1]
✅ 6 comprehensive validation plots
✅ Full dataset heatmap visualization
✅ Windowing configuration (CANShield methodology)
✅ 9 window configurations defined
✅ 27 summary visualization plots (3 per config)
✅ 1,925 individual window PNG files
✅ Mathematical verification of window counts
✅ Understanding of sampling vs stepping clarified

### File Organization:
```
Phase1/
├── scripts/
│   ├── config.py (windowing parameters)
│   ├── validate_master_table_simple.py
│   ├── plot_full_dataset_heatmap.py
│   ├── generate_window_visualizations.py
│   └── generate_individual_windows.py
├── results/
│   └── master_table_final.csv (5,095 × 60, normalized)
└── visualizations/
    ├── 50s_window/
    │   ├── sampling_1s/ (505 windows)
    │   ├── sampling_5s/ (97 windows)
    │   └── sampling_10s/ (47 windows)
    ├── 75s_window/
    │   ├── sampling_1s/ (503 windows)
    │   ├── sampling_5s/ (95 windows)
    │   └── sampling_10s/ (44 windows)
    └── 100s_window/
        ├── sampling_1s/ (500 windows)
        ├── sampling_5s/ (92 windows)
        └── sampling_10s/ (42 windows)
```

---

# WHAT'S NEXT (Phase 2 - Not Yet Started)

## CNN Model Development

### Architecture Design
- **Input shape**: (time_steps, 60, 1) - e.g., (50, 60, 1)
- **Layers**: Conv1D → Pooling → Dense → Output
- **Activation**: ReLU (hidden), Sigmoid (output)
- **Output**: Binary classification (normal=0, attack=1)

### Training Strategy
- **Dataset split**: 80% train, 20% validation
- **Augmentation**: Noise injection, time shifting
- **Loss**: Binary cross-entropy
- **Optimizer**: Adam
- **Metrics**: Accuracy, Precision, Recall, F1-score

### Model Training
- Train 9 separate CNNs (one per configuration)
- Each model learns patterns at its temporal scale
- Save best model checkpoints

### Ensemble Prediction
- Load all 9 models
- Feed test window to each model
- Combine predictions (voting or averaging)
- Final decision: Normal vs Attack

### Evaluation
- Test on held-out data
- Measure: Detection rate, False positive rate
- Compare against baselines (SVM, Random Forest)

---

# KEY DECISIONS SUMMARY

## Why These Technologies?

| Technology | Reason |
|------------|--------|
| **Python** | Rich ML ecosystem (TensorFlow, scikit-learn) |
| **Pandas** | Efficient time-series manipulation |
| **NumPy**  | Fast numerical operations |
| **Matplotlib** | Comprehensive visualization |
| **scikit-learn** | Normalization, train/test split |
| **TensorFlow/Keras** | CNN implementation (Phase 2) |

## Why These Methodological Choices?

| Choice | Reason |
|--------|--------|
| **1-second aggregation** | Balance between detail and manageability |
| **4 aggregations (mean/max/min/std)** | Capture different behavioral aspects |
| **Min-Max [0,1] normalization** | CNN-compatible, preserves relationships |
| **Multi-scale windowing** | Detect attacks at different time scales |
| **80% overlap** | Coverage vs redundancy balance |
| **9 configurations** | Comprehensive temporal analysis |
| **Ensemble approach** | Robust, proven methodology |

## Why NOT Alternative Approaches?

| Rejected | Reason |
|----------|--------|
| **Raw CAN frames** | No semantic meaning (just bytes) |
| **Single aggregation** | Loses behavioral information |
| **No normalization** | CNN training fails |
| **Single window size** | Misses scale-specific attacks |
| **No overlap** | Gaps between windows |
| **Your simpler windowing** | Misses slow attacks, less robust |

---

# LESSONS LEARNED

## Technical Insights

1. **Normalization is critical**: Without it, CNN won't converge
2. **Validation before training**: Catch preprocessing errors early
3. **Multi-scale analysis**: Different attacks need different views
4. **Visualization confirms math**: See what formulas produce

## Conceptual Clarity

1. **Sampling ≠ Stepping**: 
   - Sampling = downsample data (throw away rows)
   - Stepping = slide window (overlap control)

2. **Window size (rows) ≠ Real time**:
   - 50 rows with 1s sampling = 50 seconds
   - 50 rows with 10s sampling = 500 seconds

3. **Trade-offs everywhere**:
   - Detail vs efficiency
   - Coverage vs redundancy
   - Simplicity vs robustness

## Process Insights

1. **Question everything**: Your "why not just..." questions revealed important design choices
2. **Explain with examples**: Abstract math → concrete numbers → breakthrough understanding
3. **Compare alternatives**: Understanding "why CANShield" requires understanding "why not simple approach"

---

# PROJECT STATUS: ✅ Phase 1 Complete, Ready for Phase 2

**Next Action**: Build and train CNN models (when you're ready)

**Current Blocker**: None - all preprocessing and windowing complete

**Awaiting**: Your decision to proceed with Phase 2 (CNN model development)

---

**Date**: 1 December 2025
**Phase**: 1 Complete
**Next Phase**: 2 (CNN Training)
