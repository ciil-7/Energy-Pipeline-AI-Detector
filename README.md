# ⚡ AI-Powered Energy & Gas Pipeline Anomaly Detector

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Domain-Energy%20%26%20Gas-FF8C00?style=for-the-badge&logo=prometheus&logoColor=white" />
  <img src="https://img.shields.io/badge/AI/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Target-Aramco%20Readiness-00A3E0?style=for-the-badge" />
</p>

---

## 📌 Overview
This project demonstrates how unsupervised machine learning can be applied to simulated pipeline sensor data for anomaly detection. 

Using the **Isolation Forest** algorithm, the system evaluates generated telemetry data—including Pressure, Flow Rate, and Temperature—to identify potential operational anomalies such as simulated leaks, pressure drops, and thermal spikes.

---

## 🖥️ Application Demo & Visualizations

<p align="center">
  <img src="anomaly_chart.png" width="850" alt="App Screenshot and Sensor Anomaly Analytics">
</p>

---

## 📊 Data Source & Disclaimer
> **Note on Data Generation:**  
> To ensure continuous demonstration without relying on proprietary industrial telemetry, this project uses **synthetic data generated via Python NumPy standard normal distributions** (`np.random.normal`). Ground-truth anomaly labels are injected into the telemetry stream to rigorously evaluate model metrics.

---

## 📈 Model Performance & Evaluation
To validate detection capabilities beyond visual telemetry charts, synthetic hazard events were injected to benchmark the algorithm:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Detection Accuracy** | **98.0%** | Overall model classification reliability across all telemetry frames. |
| **Precision (Anomalies)** | **100.0%** | Proportion of flagged alerts that were genuine synthetic hazards. |
| **Recall (Anomalies)** | **90.0%** | Percentage of total synthetic hazards successfully identified by AI. |

---

## 🔬 Key Features
- **📡 Synthetic Telemetry Simulation:** Generates simulated physical parameters (Pressure in PSI, Flow Rate in kBPD, Temperature in °C) with injected ground-truth hazard events.
- **🤖 Unsupervised Anomaly Detection:** Applies `Isolation Forest` to flag irregular data patterns without static thresholding.
- **📊 Quantitative Evaluation:** Computes Precision, Recall, Accuracy, and Confusion Matrices to verify hazard detection performance.
- **🚨 Interactive Dashboard:** Features a `Streamlit` web interface to visualize sensor trends and flagged anomalies.

---

## 🛠️ Tech Stack & Dependencies
* **Programming Language:** Python 3.9+
* **Machine Learning & Data Science:** `scikit-learn`, `pandas`, `numpy`
* **Visualization & Web App:** `streamlit`, `matplotlib`

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone [https://github.com/cii1-7/Energy-Pipeline-AI-Detector.git](https://github.com/cii1-7/Energy-Pipeline-AI-Detector.git)
cd Energy-Pipeline-AI-Detector
