import matplotlib.pyplot as plt
import streamlit as st
# استدعاء الدوال مباشرة من المحرك الرئيسي لتفادي تكرار الكود
from pipeline_ai_detector import generate_sensor_data, train_and_evaluate_model

st.set_page_config(
    page_title="AI Pipeline Monitor | Energy Tech", page_icon="⚡", layout="wide"
)

st.title("⚡ AI-Powered Energy & Gas Pipeline Anomaly Monitor")
st.markdown(
    "**Unsupervised Machine Learning System for Real-Time Hazard Detection in"
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
def run_pipeline_simulation(n_samples, cont):
  # توليد البيانات واستدعاء النموذج من ملف pipeline_ai_detector.py
  df, _ = generate_sensor_data(n_samples=n_samples)
  model, df, metrics = train_and_evaluate_model(df, contamination=cont)
  return df, metrics


df, metrics = run_pipeline_simulation(sample_size, contamination_rate)
anomalies_found = (df["ai_status"] == -1).sum()

# عرض المقاييس الرئيسية
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Pressure", f"{df['pressure_psi'].mean():.1f} PSI")
col2.metric("Avg Flow Rate", f"{df['flow_rate_kbpd'].mean():.1f} kBPD")
col3.metric("Avg Temp", f"{df['temperature_c'].mean():.1f} °C")
col4.metric(
    "⚠️ AI Flagged Hazards", f"{anomalies_found}", delta_color="inverse"
)

st.divider()

# عرض نتائج التقييم (Evaluation Metrics)
st.subheader("🎯 Model Evaluation Metrics")
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("Detection Accuracy", f"{metrics['accuracy']*100:.1f}%")
m_col2.metric("Precision", f"{metrics['precision']*100:.1f}%")
m_col3.metric("Recall", f"{metrics['recall']*100:.1f}%")

st.divider()

# الرسم البياني للبيانات
st.subheader("📈 Real-Time Sensor Streams & Anomaly Detection")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["pressure_psi"], label="Pressure (PSI)", color="#1f77b4")
ax.plot(
    df.index, df["flow_rate_kbpd"], label="Flow Rate (kBPD)", color="#2ca02c"
)

anomaly_df = df[df["ai_status"] == -1]
ax.scatter(
    anomaly_df.index,
    anomaly_df["pressure_psi"],
    color="red",
    s=50,
    label="AI Flagged Hazard",
    zorder=5,
)

ax.set_xlabel("Time Step (Hours)")
ax.set_ylabel("Telemetry Readings")
ax.legend(loc="upper right")
st.pyplot(fig)

# جدول الحوادث المكتشفة
st.subheader("🚨 Field Incident Log")
if anomalies_found > 0:
  st.dataframe(df[df["ai_status"] == -1], use_container_width=True)
else:
  st.success("✅ Operational status nominal. No structural hazards detected.")
