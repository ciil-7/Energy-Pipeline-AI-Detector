import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score

def generate_sensor_data(n_samples=100, seed=42):
    """
    Generates synthetic pipeline sensor telemetry and injects ground-truth anomalies.
    """
    np.random.seed(seed)
    
    # Baseline nominal operation
    pressure = np.random.normal(loc=100.0, scale=2.0, size=n_samples)
    flow_rate = np.random.normal(loc=50.0, scale=1.5, size=n_samples)
    temperature = np.random.normal(loc=25.0, scale=1.0, size=n_samples)
    
    # Ground truth labels: 1 = Normal, -1 = Anomaly
    ground_truth = np.ones(n_samples, dtype=int)
    
    # Inject synthetic hazards (5% of data)
    num_anomalies = max(1, int(n_samples * 0.05))
    anomaly_indices = np.random.choice(n_samples, size=num_anomalies, replace=False)
    
    for idx in anomaly_indices:
        pressure[idx] += np.random.uniform(15.0, 30.0)  # High pressure spike
        flow_rate[idx] -= np.random.uniform(10.0, 20.0) # Flow drop (leak)
        temperature[idx] += np.random.uniform(8.0, 15.0) # Thermal spike
        ground_truth[idx] = -1
        
    df = pd.DataFrame({
        'pressure_psi': pressure,
        'flow_rate_kbpd': flow_rate,
        'temperature_c': temperature,
        'ground_truth': ground_truth
    })
    
    return df, anomaly_indices

def train_and_evaluate_model(df, contamination=0.05):
    """
    Trains Isolation Forest model and calculates classification performance metrics.
    """
    X = df[['pressure_psi', 'flow_rate_kbpd', 'temperature_c']]
    y_true = df['ground_truth']
    
    model = IsolationForest(contamination=contamination, random_state=42)
    y_pred = model.fit_predict(X)
    
    df['ai_status'] = y_pred
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=-1),
        'recall': recall_score(y_true, y_pred, pos_label=-1),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return model, df, metrics

if __name__ == "__main__":
    data, _ = generate_sensor_data(n_samples=200)
    model, processed_df, metrics = train_and_evaluate_model(data)
    
    print("=== Model Performance Evaluation ===")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
