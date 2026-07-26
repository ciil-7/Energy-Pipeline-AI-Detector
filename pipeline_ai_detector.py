import datetime
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def generate_sensor_data(n_samples=200):
  """Simulates real-time telemetry from Oil & Gas Pipelines.

  Features: Pressure (PSI), Flow Rate (kBPD), Temperature (°C)
  """
  np.random.seed(42)

  # Normal operational baseline
  pressure = np.random.normal(loc=50.0, scale=2.5, size=n_samples)
  flow_rate = np.random.normal(loc=120.0, scale=5.0, size=n_samples)
  temperature = np.random.normal(loc=35.0, scale=1.5, size=n_samples)

  # Inject synthetic hazards / anomalies (5% of data)
  num_anomalies = int(n_samples * 0.05)
  anomaly_indices = np.random.choice(
      n_samples, size=num_anomalies, replace=False
  )

  for idx in anomaly_indices:
    hazard_type = np.random.choice(['LEAK', 'BLOCKAGE', 'OVERHEAT'])
    if hazard_type == 'LEAK':
      pressure[idx] -= np.random.uniform(15.0, 25.0)
      flow_rate[idx] -= np.random.uniform(30.0, 50.0)
    elif hazard_type == 'BLOCKAGE':
      pressure[idx] += np.random.uniform(20.0, 35.0)
      flow_rate[idx] -= np.random.uniform(20.0, 40.0)
    elif hazard_type == 'OVERHEAT':
      temperature[idx] += np.random.uniform(15.0, 25.0)

  df = pd.DataFrame({
      'pressure_psi': pressure,
      'flow_rate_kbpd': flow_rate,
      'temperature_c': temperature,
  })
  return df, anomaly_indices


def train_ai_model(df):
  """Trains an Isolation Forest Unsupervised Machine Learning model to detect abnormal patterns in pipeline operational metrics."""
  model = IsolationForest(contamination=0.05, random_state=42)
  model.fit(df[['pressure_psi', 'flow_rate_kbpd', 'temperature_c']])
  return model


def main():
  print('==================================================')
  print('  AI ENERGY & GAS PIPELINE MONITORING SYSTEM     ')
  print('==================================================\n')

  print('[1] Generating real-time sensor streams...')
  df, injected_anomalies = generate_sensor_data(n_samples=100)

  print('[2] Training Isolation Forest AI Model...')
  model = train_ai_model(df)

  # Predict anomalies: -1 indicates anomaly, 1 indicates normal
  predictions = model.predict(
      df[['pressure_psi', 'flow_rate_kbpd', 'temperature_c']]
  )
  df['ai_status'] = predictions

  print('[3] Analyzing telemetry data stream...\n')
  time.sleep(1)

  anomaly_count = 0
  for idx, row in df.iterrows():
    if row['ai_status'] == -1:
      anomaly_count += 1
      timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
      print(f'⚠️ [ANOMALY DETECTED] Time: {timestamp} | Sensor #{idx+1}')
      print(
          f"   -> Pressure: {row['pressure_psi']:.2f} PSI | Flow:"
          f" {row['flow_rate_kbpd']:.2f} kBPD | Temp:"
          f" {row['temperature_c']:.2f} °C"
      )
      print(
          '   -> Action Required: Dispatching Automated Field Inspection'
          ' Routine\n'
      )

  print('--------------------------------------------------')
  print(f'Analysis Complete. Total Sensor Readings: {len(df)}')
  print(f'Anomalies Flagged by AI Model: {anomaly_count}')
  print('==================================================')


if __name__ == '__main__':
  main()
