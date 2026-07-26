import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import streamlit as st

st.set_page_config(
    page_title="AI Pipeline Monitor | Energy Tech", page_icon="⚡", layout="wide"
)

st.title("⚡ AI-Powered Energy & Gas Pipeline Anomaly Monitor")
st.markdown(
    "**Unsupervised Machine Learning System for Real-time Hazard Detection in"
    " Oil & Gas Infrastructure**"
)

st.sidebar.header("🎛️ Simulation Controls")
sample_size = st.sidebar.slider(
    "Sensor Telemetry Samples:", 50, 300, 100, step=10
)
contamination_rate = st.sidebar.slider(
    "Expected Anomaly Rate:", 0.01, 0.15, 0.05
)


@st.cache_data
def load_and_predict(n_samples, cont):
  np.random.seed(42)
  pressure = np.random.normal(50.0, 2.5, n_samples)
  flow = np.random.normal(120.0, 5.0, n_samples)
  temp = np.random.normal(35.0, 1.5, n_samples)

  anomalies_count = int(n_samples * cont)
  indices = np.random.choice(n_samples, size=anomalies_count, replace=False)
  for idx in indices:
    pressure[idx] -= np.random.uniform(15.0, 25.0)
    flow[idx] -= np.random.uniform(30.0, 50.0)

  df = pd.DataFrame({
      "pressure_psi": pressure,
      "flow_rate_kbpd": flow,
      "temperature_c": temp,
  })

  model = IsolationForest(contamination=cont, random_state=42)
  df["ai_status"] = model.fit_predict(
      df[["pressure_psi", "flow_rate_kbpd", "temperature_c"]]
  )
  return df


df = load_and_predict(sample_size, contamination_rate)
anomalies_found = (df["ai_status"] == -1).sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Pressure", f"{df['pressure_psi'].mean():.1f} PSI")
col2.metric("Avg Flow Rate", f"{df['flow_rate_kbpd'].mean():.1f} kBPD")
col3.metric("Avg Temp", f"{df['temperature_c'].mean():.1f} °C")
col4.metric(
    "⚠️ AI Flagged Anomalies", f"{anomalies_found}", delta_color="inverse"
)

st.divider()

st.subheader("📈 Real-Time Sensor Streams & Hazard Zones")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["pressure_psi"], label="Pressure (PSI)", color="#1f77b4")
ax.plot(df.index, df["flow_rate_kbpd"], label="Flow Rate (kBPD)", color="#2ca02c")

anomaly_df = df[df["ai_status"] == -1]
ax.scatter(
    anomaly_df.index,
    anomaly_df["pressure_psi"],
    color="red",
    s=50,
    label="AI Flagged Anomaly",
    zorder=5,
)

ax.set_xlabel("Time Step (Hours)")
ax.set_ylabel("Telemetry Readings")
ax.legend(loc="upper right")
st.pyplot(fig)

st.subheader("🚨 Field Incident Log")
if anomalies_found > 0:
  st.dataframe(df[df["ai_status"] == -1], use_container_width=True)
else:
  st.success("✅ Operational status nominal. No structural hazards detected.")
