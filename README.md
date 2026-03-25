# Power Grid Anomaly Detection

Hybrid AI framework for power grid anomaly detection and stability analysis using time-series modeling, machine learning, deep learning, transformer models, and physics-informed features.

---

## Models

Isolation Forest
LSTM Autoencoder
Transformer Model
LSTM Forecasting (with uncertainty)
Graph-based features (grid topology)

---

## Dataset

Power system forced oscillation dataset:

https://ieee-dataport.org/documents/dataset-power-system-forced-oscillation-responses

Place CSV files in:

```
data/raw/
```

---

## Setup

```
pip install numpy pandas matplotlib scikit-learn torch
```

---

## Run

```
python3 main.py
```

---

## Output

* Anomaly detection using multiple models
* Ensemble-based decision
* Risk classification: LOW / MEDIUM / HIGH
* Plots saved to:

```
outputs/plots/
```

* Results saved to:

```
outputs/results/
```

---

## Project Structure

```
src/
  models/
  physics/
  graph/
  simulation/
  pipeline.py
  utils.py

data/
  raw/
  simulation/
  scripts/

outputs/
main.py
```

---

## Notes

* Window size: 30
* Ensemble = agreement of multiple models
* Includes physics-informed features (ROCOF, frequency deviation, energy)
* Supports simulation-based data (.raw, .dyr, .dat)

---
