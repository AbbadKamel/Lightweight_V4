# N2KShield: Signal-Level Intrusion Detection for NMEA 2000
*Adapted from "CANShield" (Shahriar et al., 2023) for Maritime Environments*

## 0. Phase 0: Data Acquisition & Decoding (Completed)
*This phase corresponds to the "Preliminaries" and "System Model" inputs described in the paper.*

### 0.1 Raw Data Collection
*   **Status:** Complete.
*   **Output:** `NMEA2000/Frame(0-9999).txt`, etc.
*   **Paper Equivalent:** "recording of the raw CAN messages" (Section III-A).

### 0.2 Decoding to Signal Level
*   **Status:** Complete.
*   **Tool:** `n2k_decoder.py` (Custom Python script replacing CAN-D).
*   **Output:** `Phase0/results/decoded_frames_realtime.csv`.
*   **Paper Equivalent:** "decoding the binary payloads... using the specific car’s database (DBC)" (Section II-A).
*   **Validation:** Data Quality Audit confirmed 99.99% completeness and valid physical ranges.

## 1. System Model

### 1.1 Overview
N2KShield is a Deep-Learning-Based Intrusion Detection System (IDS) designed for NMEA 2000 networks. It operates at the **signal level** (decoded physical values like RPM, Latitude, Wind Speed) rather than the frame level (raw hex).

### 1.2 Architecture
*   **Input:** Real-time stream of decoded NMEA 2000 signals (from `n2k_decoder.py`).
*   **Core:** Ensemble of CNN-Autoencoders treating signal history as images.
*   **Output:** Anomaly Score (0-1) indicating probability of attack.

## 2. Detailed Design Adaptation

### 2.1 Signal Selection & Clustering (Maritime Context)
In automotive, signals are grouped by correlation (e.g., Wheel Speed & Brake). In maritime, we will group by **Physical System**:
*   **Navigation Cluster:** SOG, COG, Heading, Latitude, Longitude (PGNs: 129026, 129029, 127250).
*   **Propulsion Cluster:** Engine RPM, Oil Pressure, Coolant Temp (PGN: 127488).
*   **Environment Cluster:** Wind Speed, Depth, Water Temp (PGNs: 130306, 128267).

**Action:** We will calculate the Pearson Correlation Matrix on your `decoded_frames_realtime.csv` to automatically find these clusters.

### 2.2 Data Preprocessing Module

#### A. The Data Queue ($Q$)
*   **Structure:** A 2D Matrix where:
    *   **Rows ($m$):** Selected Signals (e.g., `Engine_1_RPM`, `GPS_SOG`).
    *   **Columns ($q$):** Time steps (e.g., every 100ms).
*   **Forward Filling:** NMEA 2000 devices broadcast at different rates (Engine @ 10Hz, Temp @ 1Hz). We will forward-fill missing values to maintain a dense matrix.

#### B. Multi-View Generation
We will generate 3 views to capture different dynamics:
1.  **Tactical View ($T_1$):** Sampling every 100ms. Captures fast changes (e.g., throttle spikes).
2.  **Maneuver View ($T_2$):** Sampling every 1s. Captures navigation changes (e.g., turning).
3.  **Strategic View ($T_3$):** Sampling every 5s. Captures slow trends (e.g., engine overheating, gradual course drift).

### 2.3 Data Analyzing Module (The CNN-AE)
We will implement the **Convolutional Autoencoder** described in the paper.

**Network Architecture (per view):**
1.  **Input:** Image $m \times w$ (Signals $\times$ Window Size).
2.  **Encoder:**
    *   Conv2D (32 filters, 3x3) + LeakyReLU
    *   Conv2D (16 filters, 3x3) + LeakyReLU
    *   MaxPooling
3.  **Bottleneck:** Compressed representation of the "State of the Boat".
4.  **Decoder:**
    *   UpSampling
    *   Conv2D (16 filters)
    *   Conv2D (32 filters)
    *   Output (Sigmoid) -> Reconstructed Image.

### 2.4 Attack Detection (Three-Step Thresholding)
We will implement the specific logic from Algorithm 1 & 2 in the paper:
1.  **$R_{Loss}$ (Pixel Threshold):** Is the specific signal value anomalous at this specific time?
2.  **$R_{Time}$ (Duration Threshold):** Has this signal been anomalous for too long?
3.  **$R_{Signal}$ (System Threshold):** Are too many signals acting strangely?

## 3. Implementation Plan

1.  **Data Prep:** Use `decoded_frames_realtime.csv` to generate the training "images".
2.  **Correlation Analysis:** Run Pearson correlation to order the signals.
3.  **Model Training:** Train the CNN-AE on the "Normal" data (Phase 0 data).
4.  **Evaluation:** Simulate attacks (GPS Spoofing, Engine Masquerade) and test detection.
