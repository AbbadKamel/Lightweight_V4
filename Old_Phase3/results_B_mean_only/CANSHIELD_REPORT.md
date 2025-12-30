# CANShield Attack Evaluation Report - Scenario B (Mean Only)

## 1. Experiment Overview
- **Scenario**: `B_mean_only` (15 features, Mean aggregation only).
- **Objective**: Validate the robustness of the "Mean Only" model against realistic attack vectors defined in the CANShield paper.
- **Attack Types**:
    - **Plateau**: Signal freezes at a specific value for a duration.
    - **Continuous**: Signal drifts linearly away from the true value.
    - **Playback**: A segment of valid past data is replayed.
    - **Suppress**: Signal is forced to 0 (DoS simulation).

## 2. Detection Results (Recall)

| Configuration | Plateau | Continuous | Playback | Suppress | Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **50s Window / 1s Sampling** | 3.95% | 13.16% | 2.63% | 27.63% | 7.92 |
| **50s Window / 5s Sampling** | 6.67% | 0.00% | 13.33% | 60.00% | 7.20 |
| **50s Window / 10s Sampling** | 0.00% | 25.00% | 12.50% | 37.50% | 8.89 |
| **75s Window / 1s Sampling** | 5.26% | 13.16% | 7.89% | 27.63% | 7.41 |
| **75s Window / 5s Sampling** | 13.33% | 13.33% | 6.67% | 33.33% | 8.29 |
| **75s Window / 10s Sampling** | 14.29% | 14.29% | 14.29% | 28.57% | 9.51 |
| **100s Window / 1s Sampling** | 4.00% | 8.00% | 4.00% | 22.67% | 7.70 |
| **100s Window / 5s Sampling** | 7.14% | 7.14% | 7.14% | 42.86% | 8.97 |
| **100s Window / 10s Sampling** | 14.29% | 14.29% | 14.29% | 57.14% | 9.70 |

## 3. Analysis

The detection rates are significantly lower than the generic "Stealthy" attacks (which were ~25%).

### Why is detection so low?
The "Mean Only" feature set was designed to reduce false positives caused by signal noise. However, by removing `Std` (Standard Deviation), `Min`, and `Max`, we removed the primary indicators for these specific attacks:

1.  **Plateau Attack (Freezing)**:
    -   **Characteristic**: The signal value becomes constant. Variance/Std drops to 0.
    -   **Blind Spot**: The "Mean Only" model only sees the average value. If the signal freezes at a value that is "normal" (e.g., 10 knots for speed), the Mean looks perfectly normal. The model cannot see that the *noise* has disappeared.

2.  **Playback Attack**:
    -   **Characteristic**: Valid data is replayed.
    -   **Blind Spot**: Since the data is valid, its Mean is valid. Detecting playback usually requires analyzing the *sequence* or *timing* anomalies, or subtle discontinuities at the splice point. A simple Autoencoder on Mean values will reconstruct this well, resulting in low reconstruction error (no detection).

3.  **Continuous Attack (Drift)**:
    -   **Characteristic**: Slow linear drift.
    -   **Blind Spot**: The Mean changes slowly. Unless the drift pushes the value far outside the normal range (becoming an "Aggressive" attack), the model adapts to it or sees it as a valid trend.

4.  **Suppress Attack**:
    -   **Characteristic**: Value forced to 0.
    -   **Performance**: This had the best detection (up to 60%).
    -   **Reason**: For many signals (like RPM, Depth, etc.), a sudden drop to 0 shifts the Mean significantly enough to cause a high reconstruction error.

## 4. Conclusion & Recommendations

**Scenario B (Mean Only) is not robust against sophisticated attacks like Plateau or Playback.** It trades too much sensitivity for stability.

### Recommendations:
1.  **Re-introduce Variance/Std (Scenario C)**:
    -   We need `Std` to detect Plateau attacks (where Std -> 0).
    -   To avoid the original noise issue, we could use a **smoothed Std** or only flag *extremely* low variance (unnatural silence).
2.  **Hybrid Approach**:
    -   Use "Mean Only" for value validity.
    -   Add a separate, simple check for "Zero Variance" (Plateau detector).
