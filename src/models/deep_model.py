import torch
import torch.nn as nn
import numpy as np



# LSTM AUTOENCODER

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()

        # Encoder
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Bottleneck dropout
        self.dropout = nn.Dropout(dropout)

        # Decoder
        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        encoded = self.dropout(encoded)

        decoded, _ = self.decoder(encoded)

        return self.output_layer(decoded)



# DETECTOR

class DeepAnomalyDetector:
    """
    LSTM Autoencoder-based anomaly detection
    with early stopping + robust thresholding
    """

    def __init__(self, window_size, epochs=10, lr=0.001, patience=3):
        self.window_size = window_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience

        self.model = LSTMAutoencoder()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    # DATA PREP
    
    def prepare_data(self, windows):
        return torch.tensor(windows, dtype=torch.float32).unsqueeze(-1)

    
    # TRAIN
    
    def train(self, windows):
        data = self.prepare_data(windows)

        best_loss = float('inf')
        patience_counter = 0

        self.model.train()

        for epoch in range(self.epochs):
            output = self.model(data)
            loss = self.criterion(output, data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()

            # Early stopping
            if loss_val < best_loss:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"[LSTM] Early stopping at epoch {epoch}")
                break


    # DETECTION
    
    def detect(self, windows):
        data = self.prepare_data(windows)

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(data)

            # Reconstruction error per window
            loss = torch.mean((data - reconstructed) ** 2, dim=(1, 2))

        loss_np = loss.numpy()

        # Robust threshold (percentile-based)
        threshold = np.percentile(loss_np, 97)

        anomalies = np.where(loss_np > threshold)[0]

        return anomalies