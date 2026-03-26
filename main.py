import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.preprocessing import StandardScaler
from collections import Counter
from src.utils import create_windows, extract_features, simulate_disturbance
from src.pipeline import OscillationPipeline
from src.models.deep_model import DeepAnomalyDetector
from src.models.transformer_model import TransformerAnomalyDetector
from src.models.forecast_model import LSTMForecast
from src.simulation.psse_parser import PSSEParser
from src.graph.grid_graph import GridGraph
from src.physics.physics_dynamics import PowerSystemDynamics


# CONFIG
DATA_DIR = "data/raw"
RAW_FILE = "data/simulation/Case1_PowerFlow.raw"

WINDOW_SIZE = 30
MAX_SIGNAL_LENGTH = 500
MC_RUNS = 5


# DATA LOADING
def load_signals(folder):
    signals = []

    for file in sorted(os.listdir(folder)):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(folder, file))

                # Basic validation
                if df.shape[1] < 2:
                    continue

                signal = df.iloc[:, 1].dropna().values

                if len(signal) == 0:
                    continue

                signals.append((file, signal))

            except Exception as e:
                print(f"[WARNING] Skipping {file}: {e}")

    if not signals:
        raise ValueError("No valid CSV files found")

    return signals


# PLOT
def save_plot(signal, title, anomalies=None):
    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(signal[:500], label="Signal")

    if anomalies is not None and len(anomalies) > 0:
        anomalies = anomalies[anomalies < 500]
        plt.scatter(anomalies, signal[anomalies], color="red", label="Anomaly")

    plt.legend()
    plt.title(title)

    filepath = os.path.join("outputs/plots", title.replace(" ", "_") + ".png")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    print(f"[SAVED] {filepath}")


# RESULTS
def save_results(name, stats):
    os.makedirs("outputs/results", exist_ok=True)

    with open(os.path.join("outputs/results", f"{name}_results.txt"), "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")


# FORECAST + UNCERTAINTY
def forecast_with_uncertainty(model, data):
    preds = []

    for _ in range(MC_RUNS):
        preds.append(model(data).detach().cpu().numpy())

    preds = np.array(preds)

    return preds.mean(axis=0), preds.std(axis=0)


# ENSEMBLE
def ensemble_decision(*models):
    all_idx = []

    for m in models:
        all_idx.extend(list(m))

    counts = Counter(all_idx)
    threshold = max(2, len(models) // 2)

    return np.array(sorted([i for i, c in counts.items() if c >= threshold]))


# CONSISTENCY
def consistency(a, b):
    return len(set(a) & set(b)) / (len(set(a) | set(b)) + 1e-8)


# RISK SCORING
def risk_score(n, uncertainty):
    score = 0.7 * n + 0.3 * np.mean(uncertainty)

    if score > 50:
        return "HIGH"
    elif score > 20:
        return "MEDIUM"
    else:
        return "LOW"


# MAIN PIPELINE
def main():
    print("\n=== FINAL GRID AI SYSTEM (FULL PHASE-3 — STABLE) ===\n")

    # Load data
    signals = load_signals(DATA_DIR)

    # Load grid topology
    parser = PSSEParser(RAW_FILE)
    buses, lines = parser.parse()

    grid = GridGraph()
    grid.build(buses, lines)
    graph_features = grid.graph_features()

    physics_model = PowerSystemDynamics()


    # Initialize models
    pipeline = OscillationPipeline()
    lstm = DeepAnomalyDetector(WINDOW_SIZE, epochs=1)
    transformer = TransformerAnomalyDetector(WINDOW_SIZE, epochs=1)

    forecast = LSTMForecast()
    optimizer = torch.optim.Adam(forecast.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    scaler = StandardScaler()


    # Process each signal
    for name, signal in signals:

        print(f"\n[INFO] Processing: {name}")

        signal = signal[:MAX_SIGNAL_LENGTH]

  
        # VALIDATION
        if len(signal) <= WINDOW_SIZE:
            print(f"[SKIPPED] {name} (too short: {len(signal)} samples)")
            continue

  
        # AUGMENTATION
        signal = np.concatenate([signal, simulate_disturbance(signal)])


        # WINDOWING
        windows = create_windows(signal, WINDOW_SIZE)


        # FEATURE ENGINEERING
        base_features = extract_features(windows)
        physics_features = physics_model.extract(windows)

        g_feat = np.tile(
            np.mean(graph_features, axis=0),
            (len(windows), 1)
        )

        features = np.hstack([base_features, physics_features, g_feat])
        features = scaler.fit_transform(features)


        # DETECTION MODELS
        a_if = pipeline.detect(features)
        lstm.train(windows)
        a_lstm = lstm.detect(windows)
        transformer.train(windows)
        a_trans = transformer.detect(windows)

        
        # FORECASTING
        data = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1)
        target = data[:, -1, :]

        pred = forecast(data)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred, unc = forecast_with_uncertainty(forecast, data)

        error = np.abs(pred.squeeze() - target.numpy().squeeze())
        error = (error - error.mean()) / (error.std() + 1e-8)

        a_fore = np.where(error > np.percentile(error, 92))[0]

        
        # ENSEMBLE
        ensemble = ensemble_decision(a_if, a_lstm, a_trans, a_fore)

        
        # RISK + OUTPUT
        risk = risk_score(len(ensemble), unc)

        print(f"IF:{len(a_if)} LSTM:{len(a_lstm)} TRANS:{len(a_trans)}")
        print(f"FORE:{len(a_fore)} ENSEMBLE:{len(ensemble)} RISK:{risk}")
        print(f"Consistency IF-LSTM: {consistency(a_if, a_lstm):.2f}")

        if len(ensemble) > 0.10 * len(signal) and np.mean(unc) > 0.1:
            print("[ALERT] Potential grid instability detected")

        save_plot(signal, f"{name}_ensemble", ensemble)

        save_results(name, {
            "IF": len(a_if),
            "LSTM": len(a_lstm),
            "TRANS": len(a_trans),
            "FORECAST": len(a_fore),
            "ENSEMBLE": len(ensemble),
            "RISK": risk
        })

    print("\n=== COMPLETE ===\n")


# ENTRY POINT
if __name__ == "__main__":
    main()
