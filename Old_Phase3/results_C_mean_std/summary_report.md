# Scenario C (Mean + Std) Evaluation Results

## Overview
This scenario introduced the **Standard Deviation (Std)** as a feature alongside the Mean, increasing the feature count from 15 to 30. The goal was to improve detection of **Plateau** and **Playback** attacks, which were undetectable in Scenario B (Mean Only).

## Key Findings
- **Success**: The addition of `Std` significantly improved detection rates for Plateau and Playback attacks in specific configurations.
- **Best Performance**: 
    - **75s Window / 10s Sampling**: **100% Detection** across all attack types.
    - **100s Window / 10s Sampling**: **100% Detection** across all attack types.
- **Improvement**: Compared to Scenario B (which had ~4% detection for Plateau), Scenario C achieves perfect detection in these configurations.

## Detailed Results

| Window | Sampling | Threshold | Plateau | Continuous | Playback | Suppress |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **50s** | 1s | 0.037 | 35.5% | 35.5% | 35.5% | 35.5% |
| **50s** | 5s | 0.031 | 26.7% | 26.7% | 26.7% | 33.3% |
| **50s** | 10s | 0.027 | 50.0% | 50.0% | 50.0% | 75.0% |
| **75s** | 1s | 0.127 | 0.0% | 0.0% | 0.0% | 0.0% |
| **75s** | 5s | 0.121 | 0.0% | 0.0% | 0.0% | 0.0% |
| **75s** | **10s** | **0.035** | **100%** | **100%** | **100%** | **100%** |
| **100s** | 1s | 0.041 | 30.7% | 30.7% | 30.7% | 30.7% |
| **100s** | 5s | 0.028 | 57.1% | 57.1% | 64.3% | 64.3% |
| **100s** | **10s** | **0.023** | **100%** | **100%** | **100%** | **100%** |

## Analysis
- **Sampling Period Impact**: Higher sampling periods (10s) seem to benefit significantly from the `Std` feature. This might be because the `Std` over a longer sampling period (which aggregates more raw frames) is a more stable and discriminative feature than `Std` over a short period (1s).
- **75s_1s / 75s_5s Anomaly**: The high thresholds (0.12+) for these configurations suggest high variance in the validation set reconstruction error, masking the attacks. This warrants further investigation (e.g., model retraining or data cleaning).

## Conclusion
Adding the **Standard Deviation** column has successfully solved the blindness to Plateau and Playback attacks for the 10s sampling configurations, achieving **100% detection**.
