import torch
import torch.nn as nn
import numpy as np


class TransformerAnomalyDetector:
    """
    Transformer-based anomaly detector using reconstruction error
    """

    def __init__(self, window_size, d_model=64, nhead=4, num_layers=2, epochs=1):
        self.window_size = window_size
        self.epochs = epochs

        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.input_proj = nn.Linear(1, d_model)
        self.output_proj = nn.Linear(d_model, 1)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) +
            list(self.input_proj.parameters()) +
            list(self.output_proj.parameters()),
            lr=0.001
        )

        self.loss_fn = nn.MSELoss()

    def train(self, windows):
        data = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1)

        for _ in range(self.epochs):
            x = self.input_proj(data)

            out = self.model(x)
            out = self.output_proj(out)

            loss = self.loss_fn(out, data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def detect(self, windows):
        data = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            x = self.input_proj(data)
            out = self.model(x)
            out = self.output_proj(out)

        error = torch.mean((out - data) ** 2, dim=(1, 2)).numpy()

        threshold = np.percentile(error, 90)

        return np.where(error > threshold)[0]