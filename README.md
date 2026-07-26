# ⚡ AI-Powered Energy & Gas Pipeline Anomaly Detector

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Domain-Energy%20%26%20Gas-FF8C00?style=for-the-badge&logo=prometheus&logoColor=white" />
  <img src="https://img.shields.io/badge/AI/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Target-Aramco%20Readiness-00A3E0?style=for-the-badge" />
</p>

---

## 📌 Executive Overview
An enterprise-grade **Machine Learning monitoring solution** designed to analyze real-time telemetry data (Pressure, Flow Rate, Temperature) across oil & natural gas pipelines. 

By leveraging the **Isolation Forest** unsupervised algorithm, the system detects critical operational hazards—such as **pipeline leaks, high-pressure blockages, and thermal overheating**—in real-time, preventing catastrophic structural failures.

---

## 🔬 Core Features & System Architecture
- **📡 Multi-Sensor Telemetry Streaming:** Simulates real-time physical parameters including Pressure (PSI), Flow Rate (kBPD), and Temperature (°C).
- **🤖 Unsupervised AI Detection:** Uses `Isolation Forest` to automatically flag structural anomalies without relying on rigid static thresholds.
- **🚨 Automated Incident Logging:** Outputs time-stamped incident logs for field response teams.
- **📊 Interactive Analytics Dashboard:** Built-in web application via `Streamlit` to visualize risk zones dynamically.

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
