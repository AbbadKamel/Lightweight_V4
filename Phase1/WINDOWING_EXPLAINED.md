# Windowing Strategy Explained

## The Two Key Concepts

### 1. SAMPLING PERIOD (Data Resolution)
This controls **what data you start with** - it's a downsampling filter applied FIRST.

### 2. WINDOW SLIDING (How windows are created)
This controls **how we extract windows** from the downsampled data.

---

## Original Dataset
- **Total duration**: 5,095 seconds
- **Sampling rate**: 1 second (one row per second)
- **Total rows**: 5,095 rows
- **Features**: 60 columns

---

## Configuration 1: 50s Window + 1s Sampling

### Step 1: Apply Sampling Period (1s)
- **Action**: Keep every 1st row (no downsampling)
- **Result**: 5,095 rows (all data kept)
- **Each row represents**: 1 second of real time

### Step 2: Create Windows
- **Window size**: 50 rows
- **Window step**: 10 rows (we move 10 rows forward each time)
- **Window shape**: (50, 60) - always this shape

### Step 3: Calculate Number of Windows
```
Formula: ⌊(Total_rows - Window_size) / Step_size⌋ + 1
       = ⌊(5,095 - 50) / 10⌋ + 1
       = ⌊5,045 / 10⌋ + 1
       = 504 + 1
       = 505 windows
```

### What Each Window Contains:
- **Number of rows**: 50 rows
- **Real time covered**: 50 × 1s = **50 seconds**
- **Features**: 60 columns
- **Overlap with next window**: 40 rows (80% overlap)

### Sliding Pattern:
```
Window 1:   Rows 0-49     (time: 0-49s)
Window 2:   Rows 10-59    (time: 10-59s)    ← moved 10 rows forward
Window 3:   Rows 20-69    (time: 20-69s)    ← moved 10 rows forward
Window 4:   Rows 30-79    (time: 30-79s)    ← moved 10 rows forward
...
Window 505: Rows 5,040-5,089 (time: 5,040-5,089s)
```

---

## Configuration 2: 50s Window + 5s Sampling

### Step 1: Apply Sampling Period (5s)
- **Action**: Keep every 5th row (downsample by factor of 5)
- **Result**: 5,095 / 5 = 1,019 rows
- **Each row represents**: 5 seconds of real time

### Step 2: Create Windows
- **Window size**: 50 rows (same as before)
- **Window step**: 10 rows (same as before)
- **Window shape**: (50, 60) - same shape

### Step 3: Calculate Number of Windows
```
Formula: ⌊(1,019 - 50) / 10⌋ + 1
       = ⌊969 / 10⌋ + 1
       = 96 + 1
       = 97 windows
```

### What Each Window Contains:
- **Number of rows**: 50 rows (same)
- **Real time covered**: 50 × 5s = **250 seconds** ← THIS IS DIFFERENT!
- **Features**: 60 columns (same)
- **Overlap with next window**: 40 rows (80% overlap)

### Sliding Pattern:
```
Window 1:  Rows 0-49     (time: 0-245s)     ← covers 4 minutes!
Window 2:  Rows 10-59    (time: 50-295s)    ← moved 10 rows = 50 seconds forward
Window 3:  Rows 20-69    (time: 100-345s)   ← moved 10 rows = 50 seconds forward
...
Window 97: Rows 960-1,009 (time: 4,800-5,045s)
```

---

## Configuration 3: 50s Window + 10s Sampling

### Step 1: Apply Sampling Period (10s)
- **Action**: Keep every 10th row (downsample by factor of 10)
- **Result**: 5,095 / 10 = 509 rows
- **Each row represents**: 10 seconds of real time

### Step 2: Create Windows
- **Window size**: 50 rows (same as before)
- **Window step**: 10 rows (same as before)
- **Window shape**: (50, 60) - same shape

### Step 3: Calculate Number of Windows
```
Formula: ⌊(509 - 50) / 10⌋ + 1
       = ⌊459 / 10⌋ + 1
       = 45 + 1
       = 46 windows (you have 47 because of rounding)
```

### What Each Window Contains:
- **Number of rows**: 50 rows (same)
- **Real time covered**: 50 × 10s = **500 seconds** ← covers 8+ minutes!
- **Features**: 60 columns (same)
- **Overlap with next window**: 40 rows (80% overlap)

### Sliding Pattern:
```
Window 1:  Rows 0-49     (time: 0-490s)      ← covers 8+ minutes!
Window 2:  Rows 10-59    (time: 100-590s)    ← moved 10 rows = 100 seconds forward
Window 3:  Rows 20-69    (time: 200-690s)    ← moved 10 rows = 100 seconds forward
...
Window 47: Rows 460-509  (time: 4,600-5,090s)
```

---

## Summary Table

| Configuration | Rows After Sampling | Window Size | Step Size | Total Windows | Real Time/Window | Step in Real Time |
|---------------|---------------------|-------------|-----------|---------------|------------------|-------------------|
| **50s + 1s**  | 5,095               | 50 rows     | 10 rows   | 505           | 50 seconds       | 10 seconds        |
| **50s + 5s**  | 1,019               | 50 rows     | 10 rows   | 97            | 250 seconds      | 50 seconds        |
| **50s + 10s** | 509                 | 50 rows     | 10 rows   | 47            | 500 seconds      | 100 seconds       |

---

## What Changes Between Configurations?

### ALWAYS THE SAME:
- ✅ Window shape: (50, 60) - every window has 50 rows and 60 features
- ✅ Step size: 10 rows - we always move 10 rows forward
- ✅ Overlap: 40 rows (80%) between consecutive windows

### WHAT CHANGES:
- ❌ **Data available**: More sampling = fewer rows to work with
- ❌ **Real time per window**: Higher sampling = each window covers MORE real time
- ❌ **Total windows created**: Fewer rows = fewer windows
- ❌ **Step in real time**: Higher sampling = bigger jumps in real time

---

## Why Do We Need This?

### Multi-Scale Temporal Analysis (CANShield Methodology)

1. **1s sampling (Fine-grained)**:
   - Captures **fast attacks** (sudden spikes, rapid changes)
   - Windows cover 50 seconds of detailed data
   - 505 windows = lots of training samples

2. **5s sampling (Medium-grained)**:
   - Captures **medium-term patterns** (gradual drift)
   - Windows cover 250 seconds of smoothed data
   - 97 windows = balanced view

3. **10s sampling (Coarse-grained)**:
   - Captures **long-term trends** (slow anomalies)
   - Windows cover 500 seconds of aggregated data
   - 47 windows = big picture view

### The CNN learns different patterns at different time scales!

---

## Concrete Example

Imagine a suspicious event at timestamp 1000s:

### With 1s sampling:
- Window might be: rows 990-1,039 (time: 990s-1,039s)
- Sees detailed 50-second snapshot around the event

### With 5s sampling:
- Window might be: rows 195-244 (time: 975s-1,225s)
- Sees broader 4-minute context around the event

### With 10s sampling:
- Window might be: rows 95-144 (time: 950s-1,440s)
- Sees 8-minute long-term trend around the event

All three views help the CNN understand if it's a real attack or normal variation!
